"""Generate selected CNNImageProc top.cpp, testbench, and backend artifacts."""

import json
from pathlib import Path
import re
import time

from Evaluation.CNNImageProc.AI.backend_handoff import build_backend_handoff
from Evaluation.CNNImageProc.AI.generators import (
    generate_pseudocode,
    generate_testbench,
    generate_top,
    required_hls_pragmas,
)
from Evaluation.CNNImageProc.AI.session_store import (
    get_session_partition_spec,
    load_session,
    sha256_file,
    verify_selection_digests,
    verify_source_manifest,
)
from Evaluation.CNNImageProc.AI.top_prompt_builder import build_top_generation_prompt
from Evaluation.CNNImageProc.AI.experiment_metrics import record_llm_usage, update_metrics
from LLM_Interface.LLMClient import LLMClient


class TopGenerationError(RuntimeError):
    """Raised when a selected wrapper cannot be generated safely."""


def _write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path, payload):
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def parse_json_response(response):
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TopGenerationError(f"model response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TopGenerationError("model response must be a JSON object")
    return parsed


def _validate_llm_package(package, partition_id, spec):
    code = package.get("top_cpp")
    top_function = spec["top_function"]
    if not isinstance(code, str) or f"void {top_function}" not in code:
        raise TopGenerationError(f"LLM top package must define {top_function}")
    for include in spec.get("generation", {}).get("top_includes", ["lib.hpp", "weights.hpp"]):
        if f'#include "{include}"' not in code and f"#include <{include}>" not in code:
            raise TopGenerationError(f"LLM top.cpp must include {include}")
    if re.search(r'#\s*include\s*[<\"][^>\"]+\.cpp[>\"]', code):
        raise TopGenerationError("LLM top.cpp must not include implementation .cpp files")
    if re.search(r"#\s*pragma\s+HLS\s+ARRAY_(?:PARTITION|RESHAPE)\s+variable\s*=\s*[ab]\b", code):
        raise TopGenerationError("LLM top.cpp must not partition or reshape top-level ports a or b")
    if re.search(r"#\s*pragma\s+HLS\s+AGGREGATE\s+variable\s*=\s*[ab]\b", code):
        raise TopGenerationError("LLM top.cpp must not aggregate top-level ports a or b")
    definitions = re.findall(r"\bvoid\s+(\w+)\s*\([^;{}]*\)\s*\{", code, re.DOTALL)
    if definitions != [top_function]:
        raise TopGenerationError(f"top.cpp must define only {top_function}; found {definitions}")
    for helper in spec["required_calls"]:
        if not re.search(rf"\b{re.escape(helper)}\s*\(", code):
            raise TopGenerationError(f"LLM top.cpp is missing required call {helper}")
    for pragma in required_hls_pragmas(spec):
        if pragma not in code:
            raise TopGenerationError(f"LLM top.cpp is missing required directive: {pragma}")
    return spec


class CNNImageProcTopGenerationAgent:
    """Generate deterministic partition artifacts for CNNImageProc."""

    def __init__(self, model="gpt-5.6-sol", max_tokens=6000, temperature=0.1, top_p=1.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def run(
        self,
        session_dir,
        output_dir=None,
        force=False,
        use_llm=None,
        mode="deterministic",
        dry_run=False,
    ):
        if use_llm is not None:
            mode = "llm" if use_llm else "deterministic"
        if mode not in {"auto", "llm", "deterministic"}:
            raise TopGenerationError(f"unsupported top generation mode: {mode}")
        state = load_session(session_dir)
        if not state["selection"]:
            raise TopGenerationError("session has no selection.json; select a strategy first")
        verify_selection_digests(state)
        verify_source_manifest(state["source_manifest"])

        selection = state["selection"]
        partition_id = selection["selected_partition"]
        spec = get_session_partition_spec(state, partition_id)
        if not spec["hardware_generable"]:
            raise TopGenerationError(f"{partition_id} cannot generate a single FPGA wrapper")

        run_id = selection.get("run_id") or state["recommendations"].get("run_id")
        source_dir = Path(state["source_manifest"]["source_dir"])
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else source_dir / "generated" / run_id / partition_id
        )
        top_path = destination / "top.cpp"
        testbench_path = destination / "tb.cpp"
        existing_outputs = [path for path in (top_path, testbench_path) if path.exists()]
        if existing_outputs and not force:
            raise TopGenerationError(
                f"generated output already exists: {existing_outputs[0]}; use force to replace it"
            )
        destination.mkdir(parents=True, exist_ok=True)
        metrics_path = state["session_dir"] / "experiment_metrics.json"

        started = time.time()
        prompt_path = destination / "top_prompt.txt"
        response_path = destination / "top_response.txt"
        prompt = None
        package = None
        token_count = 0
        token_usage = None
        llm_time = 0.0
        fallback_reason = None
        mode_used = mode
        if mode in {"auto", "llm"}:
            prompt = build_top_generation_prompt(
                selection, state["source_manifest"], partition_spec=spec
            )
            _write_text(
                prompt_path,
                prompt["system_prompt"] + "\n\n" + prompt["user_prompt"] + "\n",
            )
            if dry_run:
                record_llm_usage(
                    metrics_path,
                    stage="top_generation",
                    token_count=0,
                    model=self.model,
                    generation_time_s=0.0,
                    status="dry_run",
                )
                return {
                    "dry_run": True,
                    "run_id": run_id,
                    "partition_id": partition_id,
                    "output_dir": str(destination),
                    "prompt_path": str(prompt_path),
                }
            try:
                client = LLMClient(self.model)
                llm_started = time.time()
                response, token_count = client.generate_content(
                    prompt=prompt["user_prompt"],
                    system_prompt=prompt["system_prompt"],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                llm_time = time.time() - llm_started
                token_usage = client.last_usage
                record_llm_usage(
                    metrics_path,
                    stage="top_generation",
                    token_count=token_count,
                    model=self.model,
                    generation_time_s=llm_time,
                    status="response_received",
                    usage=token_usage,
                )
                _write_text(response_path, response)
                package = parse_json_response(response)
                _validate_llm_package(package, partition_id, spec)
                top_text = package["top_cpp"]
                pseudocode_text = package.get("pseudocode") or generate_pseudocode(
                    partition_id, spec=spec
                )
                io_mapping_payload = package.get("io_mapping")
                mode_used = "llm"
            except Exception as exc:
                if "llm_started" in locals() and llm_time == 0.0:
                    llm_time = time.time() - llm_started
                if mode == "llm":
                    record_llm_usage(
                        metrics_path,
                        stage="top_generation",
                        token_count=token_count,
                        model=self.model,
                        generation_time_s=llm_time,
                        status="validation_failed",
                        usage=token_usage,
                    )
                    if isinstance(exc, TopGenerationError):
                        raise
                    raise TopGenerationError(f"top-generation LLM call failed: {exc}") from exc
                fallback_reason = f"{type(exc).__name__}: {exc}"
                mode_used = "deterministic_fallback"
                top_text = generate_top(partition_id, spec=spec)
                pseudocode_text = generate_pseudocode(partition_id, spec=spec)
                io_mapping_payload = None
        else:
            top_text = generate_top(partition_id, spec=spec)
            pseudocode_text = generate_pseudocode(partition_id, spec=spec)
            io_mapping_payload = None

        _write_text(top_path, top_text.rstrip() + "\n")
        _write_text(testbench_path, generate_testbench(partition_id, spec=spec))
        pseudocode_path = destination / "pseudocode.md"
        io_mapping_path = destination / "io_mapping.json"
        handoff_path = destination / "backend_handoff.json"
        manifest_path = destination / "generation_manifest.json"
        _write_text(pseudocode_path, pseudocode_text.rstrip() + "\n")
        if not isinstance(io_mapping_payload, dict):
            io_mapping_payload = {
                "partition_id": partition_id,
                "workload": spec["workload"],
                "input": spec["input"],
                "output": spec["output"],
                "axi": {
                    "control_bundle": "CTRL",
                    "input_master_bundle": "a",
                    "output_master_bundle": "b",
                },
            }
        _write_json(io_mapping_path, io_mapping_payload)
        handoff = build_backend_handoff(
            run_id=run_id,
            partition_id=partition_id,
            source_manifest=state["source_manifest"],
            top_path=top_path,
            testbench_path=testbench_path,
            io_mapping_path=io_mapping_path,
            partition_spec=spec,
            experiment_metrics_path=state["session_dir"] / "experiment_metrics.json",
        )
        _write_json(handoff_path, handoff)
        manifest = {
            "schema_version": 1,
            "workload": spec["workload"],
            "run_id": run_id,
            "partition_id": partition_id,
            "model": f"deterministic-{spec['workload']}-generator",
            "top_generation_mode_requested": mode,
            "top_generation_mode": mode_used,
            "top_model": self.model if mode in {"auto", "llm"} else None,
            "fallback_used": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "token_count": token_count,
            "token_usage": token_usage,
            "llm_time_s": llm_time,
            "generation_time_s": time.time() - started,
            "validation": "passed",
            "top_sha256": sha256_file(top_path),
            "testbench_sha256": sha256_file(testbench_path),
            "backend_handoff_sha256": sha256_file(handoff_path),
            "immutable_source_hashes": {
                name: record["sha256"]
                for name, record in state["source_manifest"]["files"].items()
            },
            "partition_specification": spec,
            "assumptions": [
                "LightCNN evidence is used for qualitative strategy suggestions only.",
                f"{spec['workload']} resource, power, and latency must be measured directly.",
            ],
            "artifacts": {
                "top_cpp": str(top_path),
                "testbench_cpp": str(testbench_path),
                "backend_handoff": str(handoff_path),
                "pseudocode": str(pseudocode_path),
                "io_mapping": str(io_mapping_path),
                "top_prompt": str(prompt_path) if mode in {"auto", "llm"} else None,
                "top_response": str(response_path) if mode in {"auto", "llm"} else None,
            },
        }
        _write_json(manifest_path, manifest)
        record_llm_usage(
            metrics_path,
            stage="top_generation",
            token_count=token_count,
            model=self.model if mode in {"auto", "llm"} else None,
            generation_time_s=llm_time,
            status=(
                "passed_with_deterministic_fallback"
                if fallback_reason is not None
                else "passed"
                if mode_used == "llm"
                else "deterministic"
            ),
            usage=token_usage,
        )
        update_metrics(
            metrics_path,
            {
                "generation": {
                    "top": {
                        "mode_requested": mode,
                        "mode_used": mode_used,
                        "model": self.model if mode in {"auto", "llm"} else None,
                        "validation_passed": fallback_reason is None,
                        "fallback_used": fallback_reason is not None,
                        "fallback_reason": fallback_reason,
                    }
                }
            },
        )
        return {
            "run_id": run_id,
            "partition_id": partition_id,
            "output_dir": str(destination),
            "top_path": str(top_path),
            "testbench_path": str(testbench_path),
            "backend_handoff_path": str(handoff_path),
            "pseudocode_path": str(pseudocode_path),
            "io_mapping_path": str(io_mapping_path),
            "manifest_path": str(manifest_path),
            "token_count": token_count,
            "token_usage": token_usage,
            "llm_time_s": llm_time,
            "mode_requested": mode,
            "mode_used": mode_used,
            "fallback_used": fallback_reason is not None,
            "fallback_reason": fallback_reason,
        }
