"""Structured, reproducible metrics for Gen-TAS experiments."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = 2
EXPERIMENT_CONDITIONS = (
    "GenTAS_RAG",
    "LLM_NoRAG",
    "Deterministic_Heuristic",
    "Measured_Oracle",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def update_metrics(path: Path | str, updates: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload = deep_merge(current, updates)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload["updated_at"] = utc_now()
    _atomic_write(path, payload)
    return payload


def record_llm_usage(
    path: Path | str,
    *,
    stage: str,
    token_count: int,
    model: str | None,
    generation_time_s: float | None,
    status: str = "passed",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one LLM call and recompute experiment-wide token totals."""
    path = Path(path)
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    llm = current.get("llm", {})
    if not isinstance(llm, dict) or "stages" not in llm:
        legacy = llm if isinstance(llm, dict) else {}
        legacy_stages = {}
        if legacy.get("token_count") is not None:
            legacy_stages["strategy_selection"] = {
                "model": legacy.get("model"),
                "token_count": int(legacy.get("token_count") or 0),
                "generation_time_s": legacy.get("generation_time_s"),
                "status": "migrated_legacy_record",
                "uses_llm_api": True,
            }
        llm = {"stages": legacy_stages}
    stages = llm.setdefault("stages", {})
    normalized_usage = deepcopy(usage or {})
    normalized_usage.setdefault("provider", None)
    normalized_usage.setdefault("model", model)
    normalized_usage.setdefault("input_tokens", None)
    normalized_usage.setdefault("output_tokens", None)
    normalized_usage.setdefault("total_tokens", int(token_count))
    normalized_usage.setdefault("cached_input_tokens", None)
    normalized_usage.setdefault("cache_creation_input_tokens", None)
    normalized_usage.setdefault("reasoning_tokens", None)
    normalized_usage.setdefault(
        "count_source",
        "not_used" if status in {"deterministic", "dry_run", "not_used"} else "legacy_total_only",
    )
    effective_total = normalized_usage.get("total_tokens")
    if not isinstance(effective_total, int):
        effective_total = int(token_count)
        normalized_usage["total_tokens"] = effective_total
    stages[stage] = {
        "model": model,
        "token_count": effective_total,
        "token_usage": normalized_usage,
        "generation_time_s": generation_time_s,
        "status": status,
        "uses_llm_api": status not in {"deterministic", "dry_run", "not_used"},
    }
    llm["total_tokens"] = sum(
        int(item.get("token_count", 0))
        for item in stages.values()
        if isinstance(item, dict)
    )
    for total_name, usage_name in (
        ("total_input_tokens", "input_tokens"),
        ("total_output_tokens", "output_tokens"),
        ("total_cached_input_tokens", "cached_input_tokens"),
        ("total_cache_creation_input_tokens", "cache_creation_input_tokens"),
        ("total_reasoning_tokens", "reasoning_tokens"),
    ):
        values = [
            item.get("token_usage", {}).get(usage_name)
            for item in stages.values()
            if isinstance(item, dict)
        ]
        known = [value for value in values if isinstance(value, int)]
        llm[total_name] = sum(known) if known else None
    llm["api_call_stage_count"] = sum(
        bool(item.get("uses_llm_api"))
        for item in stages.values()
        if isinstance(item, dict)
    )
    llm["stages_recorded"] = sorted(stages)
    return update_metrics(path, {"llm": llm})


def _bytes_per_element(dtype: str | None, cpp_type: str | None = None) -> int | None:
    text = " ".join(value for value in (dtype, cpp_type) if value).lower()
    fixed_match = re.search(r"ap_(?:u)?fixed\s*<\s*(\d+)", text)
    int_match = re.search(r"(?:ap_)?(?:u?int)\s*<\s*(\d+)", text)
    named_match = re.search(r"(?:u?int)(8|16|32|64)(?:_t)?", text)
    bits = None
    if fixed_match or int_match:
        bits = int((fixed_match or int_match).group(1))
    elif named_match:
        bits = int(named_match.group(1))
    elif "fixed16" in text or "data_t" in text:
        bits = 16
    elif "float" in text:
        bits = 32
    elif "double" in text:
        bits = 64
    return math.ceil(bits / 8) if bits else None


def _boundary_bytes(boundary: dict[str, Any] | None, dtype: str | None) -> int | None:
    if not isinstance(boundary, dict) or not isinstance(boundary.get("length"), int):
        return None
    bits = boundary.get("bits_per_element")
    width = math.ceil(bits / 8) if isinstance(bits, int) and bits > 0 else _bytes_per_element(
        dtype, boundary.get("cpp_type") or boundary.get("dtype")
    )
    return boundary["length"] * width if width else None


TASK_INTENSITY_PROFILES = {
    "convolution": ("high", "medium", "low"),
    "activation_pooling": ("medium", "medium", "medium"),
    "classifier": ("high", "high", "low"),
    "coordinate_quantization": ("medium", "low", "medium"),
    "histogram_generation": ("medium", "high", "high"),
    "template_scoring": ("high", "high", "low"),
    "compute": ("medium", "medium", "medium"),
}


def task_characteristics(workload_contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Build qualitative semantic descriptors without inventing operation counts."""
    dtype = (workload_contract.get("definition") or {}).get("dtype") or "fixed16_q4_12"
    records = []
    for stage_id in workload_contract.get("call_order", workload_contract.get("stages", {})):
        stage = workload_contract["stages"][stage_id]
        role = stage.get("role") or "compute"
        compute, memory, control = TASK_INTENSITY_PROFILES.get(
            role, TASK_INTENSITY_PROFILES["compute"]
        )
        input_bytes = _boundary_bytes(stage.get("input"), dtype)
        output_bytes = _boundary_bytes(stage.get("output"), dtype)
        communication_bytes = (
            input_bytes + output_bytes
            if isinstance(input_bytes, int) and isinstance(output_bytes, int) else None
        )
        records.append({
            "task_id": stage_id,
            "function": stage.get("function"),
            "task_type": role,
            "compute_intensity": compute,
            "memory_intensity": memory,
            "communication_volume_bytes": communication_bytes,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "control_intensity": control,
            "descriptor_method": "role_profile_plus_source_derived_io_volume",
        })
    known = [item["communication_volume_bytes"] for item in records if item["communication_volume_bytes"] is not None]
    maximum = max(known, default=0)
    for item in records:
        volume = item["communication_volume_bytes"]
        ratio = volume / maximum if maximum and volume is not None else 0
        item["communication_intensity"] = "high" if ratio >= 0.67 else "medium" if ratio >= 0.34 else "low"
    return records


def partition_metrics(spec: dict[str, Any]) -> dict[str, Any]:
    stages = spec.get("stage_specs", {})
    order = list(stages) or list(spec.get("fpga_subfunctions", [])) + list(
        spec.get("gpp_subfunctions", [])
    )
    if all(re.fullmatch(r"[A-Z]+\d+", stage or "") for stage in order):
        order.sort(key=lambda stage: int(re.search(r"\d+", stage).group()))
    elif all(isinstance(stage, str) for stage in order):
        order.sort()
    fpga = list(spec.get("fpga_subfunctions", []))
    gpp = list(spec.get("gpp_subfunctions", []))
    total = len(order)
    devices = {stage: ("FPGA" if stage in fpga else "GPP") for stage in order}
    boundaries = []
    for left, right in zip(order, order[1:]):
        if devices[left] == devices[right]:
            continue
        boundary = stages.get(left, {}).get("output")
        if boundary is None:
            boundary = spec.get("input") if devices[right] == "FPGA" else spec.get("output")
        boundaries.append(
            {
                "from_stage": left,
                "to_stage": right,
                "direction": f"{devices[left]}_to_{devices[right]}",
                "bytes_per_inference": _boundary_bytes(boundary, spec.get("dtype")),
            }
        )

    input_bytes = _boundary_bytes(spec.get("input"), spec.get("dtype"))
    output_bytes = _boundary_bytes(spec.get("output"), spec.get("dtype"))
    intermediate_bytes = sum(
        item["bytes_per_inference"] for item in boundaries
        if isinstance(item.get("bytes_per_inference"), int)
    )
    known_transfers = [value for value in (input_bytes, output_bytes) if isinstance(value, int)]
    return {
        "fpga_tasks": fpga,
        "gpp_tasks": gpp,
        "num_tasks_fpga": len(fpga),
        "num_tasks_gpp": len(gpp),
        "percent_application_fpga": 100.0 * len(fpga) / total if total else None,
        "percent_application_gpp": 100.0 * len(gpp) / total if total else None,
        "num_hw_sw_boundaries": len(boundaries),
        "boundaries": boundaries,
        "input_transfer_bytes": input_bytes,
        "output_transfer_bytes": output_bytes,
        "intermediate_transfer_bytes": intermediate_bytes,
        "total_communication_bytes_per_inference": sum(known_transfers),
        "host_fpga_buffer_transfers_per_inference": 2 if fpga else 0,
        "dma_transfers": None,
        "dma_note": "The current PYNQ runner uses buffer flush/invalidate; no AXI DMA engine is instantiated.",
        "communication_to_computation_ratio": None,
        "cost": {
            "estimated_communication_overhead_ns": None,
            "estimated_communication_overhead_percent": None,
            "estimated_compute_time_ns": None,
            "estimated_transfer_time_ns": None,
            "estimated_overlap_ns": None,
            "source": "awaiting_pynq_runtime_measurements",
        },
    }


def extract_objectives(request: str, requirements: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    objectives = []
    structured = (requirements or {}).get("objectives", [])
    if isinstance(structured, list):
        objectives.extend(item for item in structured if isinstance(item, dict))

    patterns = (
        ("latency", r"latency.{0,30}?(?:under|below|at most|<=|less than)\s*([0-9.]+)\s*(ns|us|ms|s)"),
        ("power", r"power.{0,30}?(?:under|below|at most|<=|less than)\s*([0-9.]+)\s*(mw|w)"),
        ("lut", r"luts?.{0,30}?(?:under|below|at most|<=|less than)\s*([0-9,.]+)"),
        ("ff", r"(?:ffs?|flip[- ]flops?).{0,30}?(?:under|below|at most|<=|less than)\s*([0-9,.]+)"),
        ("dsp", r"dsps?.{0,30}?(?:under|below|at most|<=|less than)\s*([0-9,.]+)"),
        ("bram", r"brams?.{0,30}?(?:under|below|at most|<=|less than)\s*([0-9,.]+)"),
    )
    lowered = request.lower()
    for name, pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            objectives.append(
                {
                    "metric": name,
                    "operator": "<=",
                    "target": float(match.group(1).replace(",", "")),
                    "unit": match.group(2) if match.lastindex and match.lastindex > 1 else "count",
                    "source": "parsed_user_request",
                }
            )
    return objectives


def new_experiment_metrics(
    *,
    run_id: str,
    workload: str,
    request: str,
    requirements: dict[str, Any] | None = None,
    experiment_condition: str = "GenTAS_RAG",
    repetition_index: int = 1,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if experiment_condition not in EXPERIMENT_CONDITIONS:
        raise ValueError(f"unsupported experiment condition: {experiment_condition}")
    if not isinstance(repetition_index, int) or repetition_index < 1:
        raise ValueError("repetition_index must be a positive integer")
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "workload": workload,
        "created_at": now,
        "updated_at": now,
        "experiment": {
            "condition": experiment_condition,
            "repetition_index": repetition_index,
            "configuration": deepcopy(configuration or {}),
        },
        "requirements": {
            "request": request,
            "objectives": extract_objectives(request, requirements),
            "satisfied": None,
            "satisfaction_percent": None,
            "evaluations": [],
        },
        "rag": {},
        "llm": {
            "stages": {},
            "total_tokens": 0,
            "total_input_tokens": None,
            "total_output_tokens": None,
            "total_cached_input_tokens": None,
            "total_cache_creation_input_tokens": None,
            "total_reasoning_tokens": None,
            "api_call_stage_count": 0,
            "stages_recorded": [],
        },
        "llm_evaluation": {},
        "partition": {},
        "task_characteristics": [],
        "verification": {
            "software_reference": "not_run",
            "hls_c_simulation": "not_run",
            "rtl_cosimulation": "not_run",
            "fpga_execution": "not_run",
            "golden_vector": "not_run",
            "stage_status": {},
            "output_equivalence": {
                "software_vs_c_sim": None,
                "c_sim_vs_rtl": None,
                "rtl_vs_fpga": None,
                "software_vs_fpga": None,
            },
        },
        "implementation": {
            name: "not_run"
            for name in (
                "hls_compilation",
                "hls_synthesis",
                "ip_packaging",
                "vivado_synthesis",
                "place_and_route",
                "bitstream_generation",
                "pynq_script_generation",
                "pynq_execution",
            )
        },
        "runtime": {},
        "energy": {
            "measurement_source": "unavailable",
            "energy_per_inference_j": None,
            "energy_per_sample_j": None,
        },
        "trade_off": {},
        "publication": {},
    }


def evaluate_requirements(metrics: dict[str, Any]) -> dict[str, Any]:
    """Evaluate supported numeric objectives without treating missing data as failure."""
    hardware = metrics.get("hardware", {})
    runtime = metrics.get("runtime", {})
    end_to_end = runtime.get("latency_stats_ns", {}).get("end_to_end_ns", {})
    measured = {
        "latency": (end_to_end.get("mean_ns"), "ns"),
        "power": (hardware.get("total_power_w"), "w"),
        "lut": (hardware.get("lut"), "count"),
        "ff": (hardware.get("ff"), "count"),
        "dsp": (hardware.get("dsp"), "count"),
        "bram": (hardware.get("bram18_equiv"), "count"),
    }
    scale = {
        ("ns", "us"): 1e-3,
        ("ns", "ms"): 1e-6,
        ("ns", "s"): 1e-9,
        ("w", "mw"): 1e3,
    }
    evaluations = []
    for objective in metrics.get("requirements", {}).get("objectives", []):
        value, native_unit = measured.get(objective.get("metric"), (None, None))
        target_unit = str(objective.get("unit", native_unit)).lower()
        converted = value
        if isinstance(value, (int, float)) and native_unit != target_unit:
            converted = value * scale.get((native_unit, target_unit), 1.0)
        satisfied = converted <= objective.get("target") if isinstance(converted, (int, float)) else None
        evaluations.append({
            **objective,
            "measured": converted,
            "measured_unit": target_unit,
            "satisfied": satisfied,
        })
    completed = [item for item in evaluations if item["satisfied"] is not None]
    all_complete = bool(evaluations) and len(completed) == len(evaluations)
    return {
        "evaluations": evaluations,
        "satisfied": all(item["satisfied"] for item in completed) if all_complete else None,
        "satisfaction_percent": (
            100.0 * sum(bool(item["satisfied"]) for item in completed) / len(completed)
            if completed else None
        ),
        "evaluated_objective_count": len(completed),
        "objective_count": len(evaluations),
    }
