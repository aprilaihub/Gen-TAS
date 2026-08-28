#!/usr/bin/env python3
"""Run the reusable backend as one staged pipeline.

The stages are Vitis HLS IP generation, Vivado integration, hardware artifact
export, and LLM-assisted PYNQ measurement-script generation.  Commands are
executed without a shell so paths and user-provided values remain literal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation_manifest import ManifestError, prepare_backend_inputs
from artifacts.summarize_vivado_reports import summarize_reports
from Evaluation.CNNImageProc.AI.experiment_metrics import (
    evaluate_requirements,
    new_experiment_metrics,
    partition_metrics,
    update_metrics,
)
HLS_SCRIPT = BACKEND_ROOT / "hls" / "general_vitis_hls.tcl"
VIVADO_SCRIPT = BACKEND_ROOT / "vivado" / "general_vivado_build.tcl"
EXPORT_SCRIPT = BACKEND_ROOT / "artifacts" / "export_hw_artifacts.tcl"
EXPORT_DENSE_WEIGHTS_SCRIPT = BACKEND_ROOT / "artifacts" / "export_cnn_dense_weights.py"
PYNQ_SCRIPT = BACKEND_ROOT / "pynq" / "generate_llm_pynq_measure.py"
STAGES = ("hls", "vivado", "export", "pynq")
FASHION_MNIST_DATA_ZIP = BACKEND_ROOT / "examples" / "cnn_imageproc_fashion" / "FashionMNIST_data.zip"
FASHION_MNIST_WEIGHTS_HEADER = BACKEND_ROOT / "examples" / "cnn_imageproc_fashion" / "weights.hpp"


class PipelineError(RuntimeError):
    """Raised when pipeline configuration or a backend stage fails."""


def load_root_env() -> None:
    """Load repository credentials for LLM subprocesses without overriding the shell."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def existing_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def selected_stages(start_at: str, stop_after: str) -> tuple[str, ...]:
    start = STAGES.index(start_at)
    stop = STAGES.index(stop_after)
    if start > stop:
        raise PipelineError("--start-at must not come after --stop-after")
    return STAGES[start : stop + 1]


def artifact_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    export_dir = args.export_root / args.design_name
    bitstream = export_dir / f"{args.design_name}.bit"
    hwh = export_dir / f"{args.design_name}.hwh"
    output = args.output or export_dir / f"measure_{args.design_name}.py"
    return bitstream, hwh, output


def experiment_metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics_output:
        return args.metrics_output
    prepared = getattr(args, "_prepared_manifest", None)
    configured = prepared.get("handoff", {}).get("experiment_metrics_path") if prepared else None
    if configured:
        return Path(configured).expanduser().resolve()
    return args.export_root / args.design_name / "experiment_metrics.json"


def _implementation_update(stage: str, status: str, args: argparse.Namespace) -> dict:
    if stage.startswith("hls"):
        hls = getattr(args, "_prepared_manifest", {}).get("handoff", {}).get("hls", {})
        values = {"hls_compilation": status}
        for enabled, key in (
            (hls.get("run_csim", True), "hls_c_simulation"),
            (hls.get("run_csynth", True), "hls_synthesis"),
            (hls.get("run_export", True), "ip_packaging"),
        ):
            values[key] = status if enabled else "not_run"
        cosim_status = status if hls.get("run_cosim", False) else "not_run"
        return {
            "implementation": {key: value for key, value in values.items() if key != "hls_c_simulation"},
            "verification": {
                "hls_c_simulation": values["hls_c_simulation"],
                "rtl_cosimulation": cosim_status,
                "output_equivalence": {
                    "software_vs_c_sim": status == "passed" if hls.get("run_csim", True) else None,
                    "c_sim_vs_rtl": status == "passed" if hls.get("run_cosim", False) else None,
                },
            },
        }
    if stage == "vivado":
        return {
            "implementation": {
                "vivado_synthesis": status,
                "place_and_route": status,
                "bitstream_generation": status,
            }
        }
    if stage == "export":
        return {"implementation": {"hardware_artifact_export": status}}
    if stage == "pynq":
        return {"implementation": {"pynq_script_generation": status}}
    return {}


def manifest_weights_header(args: argparse.Namespace) -> Path | None:
    prepared = getattr(args, "_prepared_manifest", None)
    if not prepared:
        return None
    for header in prepared.get("headers", []):
        if Path(header).name == "weights.hpp":
            return Path(header)
    workload = str(prepared.get("handoff", {}).get("workload") or "").lower()
    source_text = " ".join(
        str(value).lower()
        for value in (
            workload,
            prepared.get("handoff", {}).get("source_dir"),
            prepared.get("handoff", {}).get("design_name"),
        )
        if value
    )
    is_fashion = "fashion" in source_text or "fmnist" in source_text
    if workload.startswith("cnn_imageproc") and is_fashion and FASHION_MNIST_WEIGHTS_HEADER.is_file():
        return FASHION_MNIST_WEIGHTS_HEADER
    return None


def manifest_data_zip(args: argparse.Namespace) -> Path | None:
    prepared = getattr(args, "_prepared_manifest", None) or {}
    workload = str(prepared.get("handoff", {}).get("workload") or "").lower()
    if not workload.startswith("cnn_imageproc"):
        return None
    weights_header = manifest_weights_header(args)
    if weights_header is None:
        return None
    weight_context = str(weights_header).lower()
    prefer_fashion = "fashion" in weight_context or "fmnist" in weight_context
    filenames = (
        ("FashionMNIST_data.zip", "data_mnist.zip")
        if prefer_fashion
        else ("data_mnist.zip", "FashionMNIST_data.zip")
    )
    for filename in filenames:
        candidate = weights_header.parent.parent / filename
        if candidate.is_file():
            return candidate
    if prefer_fashion and FASHION_MNIST_DATA_ZIP.is_file():
        return FASHION_MNIST_DATA_ZIP
    return None


def apply_generation_manifest(args: argparse.Namespace) -> None:
    """Populate backend arguments from a selected-generation manifest once."""
    if args.generation_manifest is None or getattr(args, "_manifest_applied", False):
        return
    conflicting = []
    for option, value in (
        ("--hls-config", args.hls_config),
        ("--vivado-config", args.vivado_config),
        ("--source", args.source),
        ("--testbench", args.testbench),
        ("--component-xml", args.component_xml),
        ("--design-yaml", args.design_yaml),
        ("--pynq-handoff", args.pynq_handoff),
    ):
        if value:
            conflicting.append(option)
    if conflicting:
        raise PipelineError(
            "--generation-manifest supplies backend inputs; remove " + ", ".join(conflicting)
        )

    try:
        prepared = prepare_backend_inputs(args.generation_manifest)
    except ManifestError as exc:
        raise PipelineError(f"invalid generation manifest: {exc}") from exc
    manifest_design_name = prepared["handoff"]["design_name"]
    if args.design_name and args.design_name != manifest_design_name:
        raise PipelineError(
            f"--design-name {args.design_name!r} does not match manifest design "
            f"{manifest_design_name!r}"
        )

    args.design_name = manifest_design_name
    args.hls_config = [prepared["hls_config"]]
    args.vivado_config = prepared["vivado_config"]
    args.source = prepared["semantic_sources"]
    args.testbench = [prepared["testbench"]]
    args.component_xml = [prepared["component_xml"]]
    args.pynq_handoff = prepared["handoff_path"]
    args.mode = prepared["pynq_mode"]
    args.request = "\n".join(
        item for item in (prepared["pynq_request"], args.request.strip()) if item
    )
    args._manifest_applied = True
    args._prepared_manifest = prepared
    metrics_path = experiment_metrics_path(args)
    if not metrics_path.is_file():
        handoff = prepared["handoff"]
        base = new_experiment_metrics(
            run_id=str(handoff.get("run_id") or args.design_name),
            workload=str(handoff.get("workload") or "unknown"),
            request=str(handoff.get("pynq", {}).get("request") or args.request),
        )
        pseudo_spec = {
            "fpga_subfunctions": handoff.get("fpga_subfunctions", []),
            "gpp_subfunctions": handoff.get("gpp_subfunctions", []),
            "input": handoff.get("hardware_boundary", {}).get("input"),
            "output": handoff.get("hardware_boundary", {}).get("output"),
            "dtype": handoff.get("hardware_boundary", {}).get("input", {}).get("dtype"),
        }
        base["partition"] = {
            **partition_metrics(pseudo_spec),
            "partition_id": handoff.get("partition_id"),
        }
        update_metrics(metrics_path, base)


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    apply_generation_manifest(args)
    stages = selected_stages(args.start_at, args.stop_after)
    bitstream, hwh, output = artifact_paths(args)
    commands: list[tuple[str, list[str]]] = []

    if "hls" in stages:
        if not args.hls_config:
            raise PipelineError("the HLS stage requires at least one --hls-config")
        for index, config in enumerate(args.hls_config, start=1):
            commands.append(
                (
                    f"hls[{index}/{len(args.hls_config)}]",
                    [args.vitis_hls_bin, "-f", str(HLS_SCRIPT), f"CONFIG_FILE={config}"],
                )
            )

    if "vivado" in stages:
        if args.vivado_config is None:
            raise PipelineError("the Vivado stage requires --vivado-config")
        commands.append(
            (
                "vivado",
                [
                    args.vivado_bin,
                    "-mode",
                    "batch",
                    "-source",
                    str(VIVADO_SCRIPT),
                    "-tclargs",
                    f"CONFIG_FILE={args.vivado_config}",
                ],
            )
        )

    if "export" in stages:
        if args.vivado_config is None:
            raise PipelineError("the export stage requires --vivado-config")
        commands.append(
            (
                "export",
                [
                    args.tclsh_bin,
                    str(EXPORT_SCRIPT),
                    f"CONFIG_FILE={args.vivado_config}",
                    f"DESIGN_NAME={args.design_name}",
                    f"EXPORT_ROOT={args.export_root}",
                ],
            )
        )

    if "pynq" in stages:
        if not args.source and args.design_yaml is None:
            raise PipelineError("the PYNQ stage requires --source or --design-yaml")

        command = [
            args.python_bin,
            str(PYNQ_SCRIPT),
            "--mode",
            args.mode,
            "--generation-mode",
            args.pynq_generation_mode,
            "--hwh",
            str(hwh),
            "--bitstream",
            bitstream.name,
            "--output",
            str(output),
            "--model",
            args.model,
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--iterations",
            str(args.iterations),
            "--warmup",
            str(args.warmup),
            "--kernel-timeout",
            str(args.kernel_timeout),
        ]
        for source in args.source:
            command.extend(("--source", str(source)))
        for testbench in args.testbench:
            command.extend(("--testbench", str(testbench)))
        for component_xml in args.component_xml:
            command.extend(("--component-xml", str(component_xml)))
        for reference in args.reference:
            command.extend(("--reference", str(reference)))
        if args.design_yaml:
            command.extend(("--design-yaml", str(args.design_yaml)))
        if args.pynq_handoff:
            command.extend(("--handoff", str(args.pynq_handoff)))
        if args.request:
            command.extend(("--request", args.request))
        if args.prompt_output:
            command.extend(("--prompt-output", str(args.prompt_output)))
        else:
            command.extend(("--prompt-output", str(output.with_suffix(".prompt.txt"))))
        if args.response_output:
            command.extend(("--response-output", str(args.response_output)))
        else:
            command.extend(("--response-output", str(output.with_suffix(".response.txt"))))
        if args.pynq_response_input:
            command.extend(("--response-input", str(args.pynq_response_input)))
        if args.pynq_dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--force")
        commands.append(("pynq-revalidation" if args.pynq_response_input else "pynq", command))

        weights_header = manifest_weights_header(args)
        if weights_header is not None and not args.pynq_dry_run and not args.dry_run:
            data_zip = manifest_data_zip(args)
            if data_zip is not None:
                commands.append(
                    (
                        "pynq-data",
                        [
                            args.python_bin,
                            "-c",
                            (
                                "from pathlib import Path; import shutil; import sys; "
                                "src=Path(sys.argv[1]); "
                                "dst=Path(sys.argv[2]) / src.name; "
                                "dst.parent.mkdir(parents=True, exist_ok=True); "
                                "shutil.copy2(src, dst); "
                                "print(f'DATA_ZIP: {dst}')"
                            ),
                            str(data_zip),
                            str(output.parent),
                        ],
                    )
                )
            commands.append(
                (
                    "pynq-weights-header",
                    [
                        args.python_bin,
                        "-c",
                        (
                            "from pathlib import Path; import shutil; import sys; "
                            "src=Path(sys.argv[1]); "
                            "dst=Path(sys.argv[2]) / 'weights.hpp'; "
                            "dst.parent.mkdir(parents=True, exist_ok=True); "
                            "shutil.copy2(src, dst); "
                            "print(f'WEIGHTS_HPP: {dst}')"
                        ),
                        str(weights_header),
                        str(output.parent),
                    ],
                )
            )
    return commands


def _check_executable(executable: str) -> None:
    if "/" in executable:
        path = Path(executable).expanduser()
        if not path.is_file():
            raise PipelineError(f"executable does not exist: {path}")
    elif shutil.which(executable) is None:
        raise PipelineError(f"required executable is not on PATH: {executable}")


def _sha256_path(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tool_version(executable: str) -> str | None:
    for flag in ("-version", "--version"):
        try:
            result = subprocess.run(
                [executable, flag],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = "\n".join(item for item in (result.stdout, result.stderr) if item).strip()
        if output:
            return output.splitlines()[0].strip()
    return None


def run_commands(
    commands: list[tuple[str, list[str]]], dry_run: bool, *, args: argparse.Namespace | None = None
) -> None:
    pipeline_started = time.monotonic()
    if not dry_run:
        for _, command in commands:
            _check_executable(command[0])
        if args is not None:
            metrics_path = experiment_metrics_path(args)
            previous_metrics = (
                json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics_path.is_file()
                else {}
            )
            if args.pynq_response_input:
                pynq_llm_time = (
                    previous_metrics.get("llm", {})
                    .get("stages", {})
                    .get("pynq_script_generation", {})
                    .get("generation_time_s")
                )
                recorded_pynq_time = (
                    previous_metrics.get("runtime", {})
                    .get("backend_stage_time_s", {})
                    .get("pynq")
                )
                if (
                    isinstance(pynq_llm_time, (int, float))
                    and (
                        not isinstance(recorded_pynq_time, (int, float))
                        or recorded_pynq_time < pynq_llm_time
                    )
                ):
                    previous_metrics = update_metrics(
                        metrics_path,
                        {"runtime": {"backend_stage_time_s": {"pynq": pynq_llm_time}}},
                    )
            requested_stages = list(
                dict.fromkeys(
                    previous_metrics.get("pipeline", {}).get("requested_stages", [])
                    + list(
                        previous_metrics.get("runtime", {})
                        .get("backend_stage_time_s", {})
                        .keys()
                    )
                    + [name for name, _ in commands]
                )
            )
            handoff = getattr(args, "_prepared_manifest", {}).get("handoff", {})
            hls = handoff.get("hls", {})
            update_metrics(
                metrics_path,
                {
                    "experiment": {
                        "configuration": {
                            "target_part": hls.get("part"),
                            "clock_period_ns": hls.get("clock_period_ns"),
                            "clock_uncertainty_ns": hls.get("clock_uncertainty_ns"),
                            "hls_config_sha256": _sha256_path(
                                args.hls_config[0] if args.hls_config else None
                            ),
                            "vivado_config_sha256": _sha256_path(args.vivado_config),
                            "vitis_hls_executable": args.vitis_hls_bin,
                            "vitis_hls_version": _tool_version(args.vitis_hls_bin),
                            "vivado_executable": args.vivado_bin,
                            "vivado_version": _tool_version(args.vivado_bin),
                        }
                    },
                    "pipeline": {
                        "status": "running",
                        "requested_stages": requested_stages,
                        "overall_success": None,
                    }
                },
            )

    for index, (name, command) in enumerate(commands, start=1):
        rendered = shlex.join(command)
        print(f"\n[{index}/{len(commands)}] {name}: {rendered}", flush=True)
        if dry_run:
            continue
        metrics_path = experiment_metrics_path(args) if args is not None else None
        if metrics_path:
            update_metrics(metrics_path, _implementation_update(name, "running", args))
        stage_started = time.monotonic()
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            if metrics_path:
                elapsed = time.monotonic() - stage_started
                update_metrics(metrics_path, _implementation_update(name, "failed", args))
                update_metrics(
                    metrics_path,
                    {
                        "runtime": {"backend_stage_time_s": {name: elapsed}},
                        "pipeline": {
                            "status": "failed",
                            "failed_stage": name,
                            "overall_success": False,
                        }
                    },
                )
            raise PipelineError(f"stage {name!r} failed with exit code {exc.returncode}") from exc
        if metrics_path:
            elapsed = time.monotonic() - stage_started
            update_metrics(metrics_path, _implementation_update(name, "passed", args))
            update_metrics(
                metrics_path,
                {"runtime": {"backend_stage_time_s": {name: elapsed}}},
            )

        if name == "export" and args is not None:
            export_dir = args.export_root / args.design_name
            reports_dir = export_dir / "reports"
            if all((reports_dir / filename).is_file() for filename in (
                "module_utilization.csv", "power_report.csv", "timing_paths.csv"
            )):
                report_summary = summarize_reports(reports_dir)
                (export_dir / "vivado_report_summary.json").write_text(
                    json.dumps(report_summary, indent=2) + "\n", encoding="utf-8"
                )
                payload = update_metrics(metrics_path, {"hardware": report_summary["key_values"]})
                update_metrics(metrics_path, {"requirements": evaluate_requirements(payload)})
            export_metrics = export_dir / "experiment_metrics.json"
            if export_metrics.resolve() != metrics_path.resolve():
                export_metrics.write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")
    if not dry_run and args is not None:
        metrics_path = experiment_metrics_path(args)
        invocation_time = time.monotonic() - pipeline_started
        current_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        stage_times = current_metrics.get("runtime", {}).get("backend_stage_time_s", {})
        measured_total = sum(
            value for value in stage_times.values() if isinstance(value, (int, float))
        )
        update_metrics(
            metrics_path,
            {
                "runtime": {
                    "backend_invocation_time_s": invocation_time,
                    "backend_total_time_s": measured_total,
                },
                "pipeline": {
                    "status": "build_passed_awaiting_board_execution",
                    "build_success": True,
                    "failed_stage": None,
                    "overall_success": None,
                }
            },
        )
        export_metrics = args.export_root / args.design_name / "experiment_metrics.json"
        if export_metrics.parent.is_dir() and export_metrics.resolve() != metrics_path.resolve():
            export_metrics.write_text(metrics_path.read_text(encoding="utf-8"), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HLS, Vivado, artifact export, and PYNQ generation as one backend pipeline."
    )
    parser.add_argument("--design-name")
    parser.add_argument("--generation-manifest", type=existing_path)
    parser.add_argument("--hls-config", action="append", type=existing_path, default=[])
    parser.add_argument("--vivado-config", type=existing_path)
    parser.add_argument("--source", action="append", type=existing_path, default=[])
    parser.add_argument("--testbench", action="append", type=existing_path, default=[])
    parser.add_argument("--component-xml", action="append", type=existing_path, default=[])
    parser.add_argument("--design-yaml", type=existing_path)
    parser.add_argument("--pynq-handoff", type=existing_path)
    parser.add_argument("--reference", action="append", type=existing_path, default=[])
    parser.add_argument("--mode", choices=("auto", "single", "modular"), default="auto")
    parser.add_argument(
        "--pynq-generation-mode",
        choices=("auto", "llm", "deterministic"),
        default=os.getenv("GENTAS_PYNQ_MODE", os.getenv("LAMDA_PYNQ_MODE", "auto")),
        help="PYNQ software generation mode; auto validates GPT output and falls back",
    )
    parser.add_argument("--request", default="")

    parser.add_argument(
        "--export-root",
        type=lambda value: Path(value).expanduser().resolve(),
        default=(BACKEND_ROOT / "artifacts" / "hardware_exports").resolve(),
    )
    parser.add_argument("--output", type=lambda value: Path(value).expanduser().resolve())
    parser.add_argument("--prompt-output", type=lambda value: Path(value).expanduser().resolve())
    parser.add_argument("--response-output", type=lambda value: Path(value).expanduser().resolve())
    parser.add_argument("--pynq-response-input", type=existing_path)
    parser.add_argument("--metrics-output", type=lambda value: Path(value).expanduser().resolve())

    parser.add_argument(
        "--model",
        default=os.getenv(
            "GENTAS_PYNQ_MODEL",
            os.getenv("PYNQ_LLM_MODEL", os.getenv("LAMDA_PYNQ_MODEL", "gpt-5.6-sol")),
        ),
    )
    parser.add_argument("--max-tokens", type=positive_int, default=40_000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--iterations", type=positive_int, default=10_000)
    parser.add_argument("--warmup", type=non_negative_int, default=100)
    parser.add_argument("--kernel-timeout", type=positive_float, default=5.0)

    parser.add_argument("--start-at", choices=STAGES, default="hls")
    parser.add_argument("--stop-after", choices=STAGES, default="pynq")
    parser.add_argument("--dry-run", action="store_true", help="print every command without executing it")
    parser.add_argument(
        "--pynq-dry-run",
        action="store_true",
        help="run build/export stages but only write the LLM prompt at the PYNQ stage",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing PYNQ output script")

    parser.add_argument("--vitis-hls-bin", default="vitis_hls")
    parser.add_argument("--vivado-bin", default="vivado")
    parser.add_argument("--tclsh-bin", default="tclsh")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args(argv)

    if args.design_name is None and args.generation_manifest is None:
        parser.error("provide --design-name or --generation-manifest")

    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        load_root_env()
        args = parse_args(argv)
        commands = build_commands(args)
        bitstream, hwh, output = artifact_paths(args)
        run_commands(commands, args.dry_run, args=args)
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nBackend pipeline completed." if not args.dry_run else "\nBackend dry run completed.")
    print(f"Bitstream: {bitstream}")
    print(f"HWH:       {hwh}")
    print(f"PYNQ:      {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
