"""Persistent recommendation sessions for CNNImageProc experiments."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

from Evaluation.CNNImageProc.AI.schemas import (
    PARTITION_SPECS,
    REQUIRED_SOURCE_FILES,
    SchemaError,
    build_partition_spec,
    get_partition_spec,
)
from Evaluation.CNNImageProc.AI.workload_contract import build_workload_contract
from Evaluation.CNNImageProc.AI.workload_definition import (
    definition_from_manifest,
    load_workload_definition,
)
from Evaluation.CNNImageProc.AI.experiment_metrics import (
    new_experiment_metrics,
    partition_metrics,
    record_llm_usage,
    task_characteristics,
    update_metrics,
)


CNN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS_DIR = CNN_DIR / "Sessions"


class SessionError(RuntimeError):
    """Raised when session state is missing, stale, or inconsistent."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _default_run_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"cnn_imageproc_{timestamp}_{uuid4().hex[:8]}"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest_digest(manifest):
    """Hash source content identities without timestamp or absolute-path noise."""
    digest = hashlib.sha256()
    for name, record in sorted(manifest.get("files", {}).items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(record.get("sha256", "")).encode("ascii"))
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise SessionError(f"session artifact does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionError(f"invalid JSON in {path}: {exc}") from exc


def build_source_manifest(source_dir):
    """Hash the immutable CNNImageProc source bundle."""
    source_dir = Path(source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise SessionError(f"source directory does not exist: {source_dir}")

    workload_definition = load_workload_definition(source_dir)
    required_files = (
        workload_definition["required_source_files"]
        if workload_definition
        else REQUIRED_SOURCE_FILES
    )
    files = {}
    for name in required_files:
        path = source_dir / name
        if not path.is_file():
            raise SessionError(f"required workload source is missing: {path}")
        files[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    workload_contract = build_workload_contract(source_dir)
    source_stage_graph = workload_contract["source_graph"]
    return {
        "workload": workload_contract["workload"],
        "workload_definition": workload_definition,
        "source_dir": str(source_dir),
        "created_at": _utc_now(),
        "source_stage_graph": source_stage_graph,
        "workload_contract": workload_contract,
        "files": files,
    }


def verify_source_manifest(manifest):
    """Fail if any imported source has changed since recommendation."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise SessionError("source_manifest.json has no files object")
    definition = definition_from_manifest(manifest)
    required_files = definition["required_source_files"] if definition else REQUIRED_SOURCE_FILES
    for name in required_files:
        record = manifest["files"].get(name)
        if not record:
            raise SessionError(f"source manifest is missing {name}")
        path = Path(record["path"])
        if not path.is_file():
            raise SessionError(f"manifest source no longer exists: {path}")
        actual = sha256_file(path)
        if actual != record.get("sha256"):
            raise SessionError(f"source changed after recommendation: {name}")
    return True


def _safe_run_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise SessionError("run_id may contain only letters, numbers, dot, dash, and underscore")
    return value


def _recommendations(goal, top_k, workload_definition=None):
    # Curated cold-start ordering. LightCNN teaches the strategy shape, while
    # CNNImageProc's actual numbers must come from CNN measurements.
    priority = workload_definition.get("partition_priority", []) if workload_definition else [
        "FEATURE_FPGA_DENSE_GPP",
        "ALL_FPGA",
        "S1S2_FPGA_REST_GPP",
        "S3S5_FPGA_AFTER_BLOCK1_GPP",
        "S3S4_FPGA_ONLY",
        "DENSE_FPGA_ONLY",
        "S1_FPGA_REST_GPP",
        "S1S3_FPGA_REST_GPP",
        "S2_FPGA_ONLY",
        "ALL_GPP",
    ]
    recommendations = []
    for partition_id in priority[:top_k]:
        spec = get_partition_spec(
            partition_id, workload_definition=workload_definition
        )
        recommendations.append({
            "rank": len(recommendations) + 1,
            "partition_id": partition_id,
            "fpga_subfunctions": spec["fpga_subfunctions"],
            "gpp_subfunctions": spec["gpp_subfunctions"],
            "hardware_generable": spec["hardware_generable"],
            "summary": spec["summary"],
            "goal": goal,
            "justification": (
                "Selected using qualitative partition reasoning. Workload-specific "
                "resource, power, and latency require direct measurement."
            ),
        })
    return recommendations


def _recommendations_from_result(
    result, top_k=3, workload_definition=None, allow_fallback=True
):
    """Normalize selectable recommendations from an LLM allocation result."""
    parsed = result.get("parsed_recommendations")
    recommendations = []
    dynamic_specs = {}
    if isinstance(parsed, dict):
        parsed_specs = parsed.get("dynamic_partition_specs", {})
        for item in parsed.get("recommendations", []):
            if not isinstance(item, dict):
                continue
            partition_id = item.get("partition_id")
            if not isinstance(partition_id, str):
                continue
            try:
                if partition_id in parsed_specs:
                    spec = build_partition_spec(
                        partition_id=partition_id,
                        fpga_subfunctions=parsed_specs[partition_id].get("fpga_subfunctions", []),
                        gpp_subfunctions=parsed_specs[partition_id].get("gpp_subfunctions"),
                        summary=parsed_specs[partition_id].get("summary"),
                        workload_definition=workload_definition,
                    )
                    dynamic_specs[partition_id] = {
                        "partition_id": partition_id,
                        "fpga_subfunctions": spec["fpga_subfunctions"],
                        "gpp_subfunctions": spec["gpp_subfunctions"],
                        "summary": spec["summary"],
                    }
                else:
                    spec = get_partition_spec(
                        partition_id, workload_definition=workload_definition
                    )
            except SchemaError:
                continue
            normalized = dict(item)
            normalized["rank"] = len(recommendations) + 1
            normalized["fpga_subfunctions"] = spec["fpga_subfunctions"]
            normalized["gpp_subfunctions"] = spec["gpp_subfunctions"]
            normalized["hardware_generable"] = spec["hardware_generable"]
            normalized.setdefault("summary", spec["summary"])
            recommendations.append(normalized)
            if len(recommendations) == top_k:
                break
    if not recommendations:
        if not allow_fallback:
            return [], {}
        goal = result.get("requirements", {}).get("primary_goal", "latency")
        recommendations = _recommendations(goal, top_k, workload_definition)
    return recommendations, dynamic_specs


def create_session(
    source_dir,
    sessions_root=None,
    run_id=None,
    request="",
    goal="latency",
    top_k=3,
    experiment_condition="Deterministic_Heuristic",
    repetition_index=1,
):
    """Create an immutable recommendation checkpoint for later selection."""
    sessions_root = Path(sessions_root or DEFAULT_SESSIONS_DIR).expanduser().resolve()
    run_id = _safe_run_id(run_id or _default_run_id())
    session_dir = sessions_root / run_id
    if session_dir.exists():
        raise SessionError(f"session already exists: {session_dir}")

    manifest = build_source_manifest(source_dir)
    workload_definition = definition_from_manifest(manifest)
    recommendations = _recommendations(goal, top_k, workload_definition)
    payload = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "status": "awaiting_selection",
        "workload": manifest["workload"],
        "request": request,
        "primary_goal": goal,
        "experiment_condition": experiment_condition,
        "repetition_index": repetition_index,
        "recommendations": recommendations,
        "primary_recommendation": recommendations[0]["partition_id"],
        "note": (
            "Prior evidence is used only for qualitative strategy shape; numeric "
            "workload predictions are intentionally not invented."
        ),
    }
    recommendations_path = session_dir / "recommendations.json"
    manifest_path = session_dir / "source_manifest.json"
    _write_json(recommendations_path, payload)
    _write_json(manifest_path, manifest)
    metrics_path = session_dir / "experiment_metrics.json"
    metrics = new_experiment_metrics(
        run_id=run_id,
        workload=manifest["workload"],
        request=request,
        requirements={"primary_goal": goal},
        experiment_condition=experiment_condition,
        repetition_index=repetition_index,
        configuration={
            "strategy_mode": "deterministic",
            "strategy_model": None,
            "strategy_temperature": None,
            "strategy_top_p": None,
            "strategy_max_tokens": 0,
            "top_k": top_k,
            "rag_enabled": False,
            "prompt_sha256": None,
            "kb_version": None,
            "source_sha256": source_manifest_digest(manifest),
        },
    )
    metrics["task_characteristics"] = task_characteristics(manifest["workload_contract"])
    metrics["rag"] = {
        "enabled": False,
        "method": "disabled",
        "kb_version": None,
        "number_retrieved": 0,
        "retrieval_success": False,
        "retrieval_confidence": 0.0,
        "retrieved_task_ids": [],
        "similarity_scores": [],
        "condition": (
            "disabled_controlled_oracle_campaign"
            if experiment_condition == "Measured_Oracle"
            else "disabled_deterministic_baseline"
        ),
    }
    _write_json(metrics_path, metrics)
    record_llm_usage(
        metrics_path,
        stage="strategy_selection",
        token_count=0,
        model=None,
        generation_time_s=0.0,
        status="deterministic",
    )
    return {
        "run_id": run_id,
        "session_dir": str(session_dir),
        "recommendations_path": str(recommendations_path),
        "source_manifest_path": str(manifest_path),
        "selectable_partitions": [item["partition_id"] for item in recommendations],
    }


def create_session_from_recommendation(
    recommendation_result,
    source_dir,
    sessions_root=None,
    run_id=None,
    top_k=3,
):
    """Create a session from an LLM allocation result."""
    live_result = not recommendation_result.get("dry_run")
    parsed = recommendation_result.get("parsed_recommendations")
    evaluation = recommendation_result.get("llm_evaluation", {})
    parsed_count = (
        len(parsed.get("recommendations", [])) if isinstance(parsed, dict) else 0
    )
    accepted_count = evaluation.get("accepted_allocation_count", parsed_count)
    if live_result and (
        not isinstance(parsed, dict)
        or not parsed.get("recommendations")
        or evaluation.get("response_parse_valid") is False
        or accepted_count <= 0
    ):
        raise SessionError(
            "live LLM allocation produced no parsed, accepted recommendation; "
            "session creation refused"
        )
    sessions_root = Path(sessions_root or DEFAULT_SESSIONS_DIR).expanduser().resolve()
    run_id = _safe_run_id(run_id or _default_run_id())
    session_dir = sessions_root / run_id
    if session_dir.exists():
        raise SessionError(f"session already exists: {session_dir}")

    manifest = build_source_manifest(source_dir)
    workload_definition = definition_from_manifest(manifest)
    recommendations, dynamic_specs = _recommendations_from_result(
        recommendation_result,
        top_k=top_k,
        workload_definition=workload_definition,
        allow_fallback=not live_result,
    )
    if live_result and not recommendations:
        raise SessionError("live LLM allocation contained no valid selectable partition")
    session_dir.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "status": "awaiting_selection",
        "workload": manifest["workload"],
        "request": recommendation_result.get("user_request", ""),
        "primary_goal": recommendation_result.get("requirements", {}).get("primary_goal"),
        "model": recommendation_result.get("model"),
        "experiment_condition": recommendation_result.get("experiment", {}).get(
            "condition", "GenTAS_RAG"
        ),
        "repetition_index": recommendation_result.get("experiment", {}).get(
            "repetition_index", 1
        ),
        "recommendations": recommendations,
        "dynamic_partition_specs": dynamic_specs,
        "primary_recommendation": recommendations[0]["partition_id"],
        "baseline_comparison": (
            recommendation_result.get("parsed_recommendations", {}).get("baseline_comparison")
            if isinstance(recommendation_result.get("parsed_recommendations"), dict)
            else None
        ),
        "recommendation_output": recommendation_result.get("output_path"),
        "note": (
            "LLM recommendations are generated from source and prior evidence, then "
            "validated against the source-derived stage graph. Numeric workload "
            "metrics are not inferred from LightCNN."
        ),
    }
    recommendations_path = session_dir / "recommendations.json"
    manifest_path = session_dir / "source_manifest.json"
    _write_json(recommendations_path, payload)
    _write_json(manifest_path, manifest)
    experiment = recommendation_result.get("experiment", {})
    configuration = dict(experiment.get("configuration", {}))
    configuration["source_sha256"] = source_manifest_digest(manifest)
    metrics = new_experiment_metrics(
        run_id=run_id,
        workload=manifest["workload"],
        request=recommendation_result.get("user_request", ""),
        requirements=recommendation_result.get("requirements", {}),
        experiment_condition=experiment.get("condition", "GenTAS_RAG"),
        repetition_index=experiment.get("repetition_index", 1),
        configuration=configuration,
    )
    metrics["rag"] = recommendation_result.get("rag_metrics", {})
    metrics["task_characteristics"] = task_characteristics(manifest["workload_contract"])
    metrics["llm_evaluation"] = recommendation_result.get("llm_evaluation", {})
    metrics_path = session_dir / "experiment_metrics.json"
    _write_json(metrics_path, metrics)
    record_llm_usage(
        metrics_path,
        stage="strategy_selection",
        token_count=recommendation_result.get("token_count") or 0,
        model=recommendation_result.get("model"),
        generation_time_s=recommendation_result.get("llm_time_s"),
        status="dry_run" if recommendation_result.get("dry_run") else "passed",
        usage=recommendation_result.get("token_usage"),
    )
    return {
        "run_id": run_id,
        "session_dir": str(session_dir),
        "recommendations_path": str(recommendations_path),
        "source_manifest_path": str(manifest_path),
        "selectable_partitions": [item["partition_id"] for item in recommendations],
    }


def load_session(session_dir):
    session_dir = Path(session_dir).expanduser().resolve()
    return {
        "session_dir": session_dir,
        "recommendations": _read_json(session_dir / "recommendations.json"),
        "source_manifest": _read_json(session_dir / "source_manifest.json"),
        "selection": (
            _read_json(session_dir / "selection.json")
            if (session_dir / "selection.json").is_file()
            else None
        ),
    }


def get_session_partition_spec(state, partition_id):
    """Resolve a selected partition from session-local LLM specs or legacy specs."""
    dynamic_specs = state["recommendations"].get("dynamic_partition_specs", {})
    try:
        return get_partition_spec(
            partition_id,
            dynamic_specs=dynamic_specs,
            workload_definition=definition_from_manifest(state["source_manifest"]),
        )
    except SchemaError as exc:
        raise SessionError(str(exc)) from exc


def verify_selection_digests(state):
    selection = state.get("selection")
    if not selection:
        raise SessionError("session has no selection.json")
    for key, path in (
        ("recommendations_digest", state["session_dir"] / "recommendations.json"),
        ("source_manifest_digest", state["session_dir"] / "source_manifest.json"),
    ):
        expected = selection.get(key)
        actual = sha256_file(path)
        if expected != actual:
            raise SessionError(f"session artifact changed after selection: {path.name}")
    return True


def store_selection(session_dir, partition_id, overwrite=False):
    """Persist a selected CNNImageProc partition."""
    state = load_session(session_dir)
    verify_source_manifest(state["source_manifest"])
    recommendations = state["recommendations"].get("recommendations", [])
    allowed = {item.get("partition_id"): item for item in recommendations}
    if partition_id not in allowed:
        raise SessionError(
            f"partition {partition_id!r} is not selectable; choose one of {sorted(allowed)}"
        )
    spec = get_session_partition_spec(state, partition_id)
    if not spec["hardware_generable"]:
        raise SessionError(f"partition {partition_id!r} has no single FPGA hardware boundary")

    selection_path = state["session_dir"] / "selection.json"
    if selection_path.exists() and not overwrite:
        raise SessionError(f"selection already exists: {selection_path}; use force to replace it")

    selection = {
        "run_id": state["recommendations"].get("run_id"),
        "selected_at": _utc_now(),
        "selected_partition": partition_id,
        "selected_rank": allowed[partition_id].get("rank"),
        "recommendations_digest": sha256_file(state["session_dir"] / "recommendations.json"),
        "source_manifest_digest": sha256_file(state["session_dir"] / "source_manifest.json"),
    }
    _write_json(selection_path, selection)
    metrics_path = state["session_dir"] / "experiment_metrics.json"
    update_metrics(
        metrics_path,
        {
            "partition": {
                **partition_metrics(spec),
                "partition_id": partition_id,
                "candidate_strategy_rank": allowed[partition_id].get("rank"),
                "allocation_confidence_score": allowed[partition_id].get(
                    "allocation_confidence_score"
                ),
                "decision_reason": allowed[partition_id].get("recommendation")
                or allowed[partition_id].get("summary"),
                "decision_basis": allowed[partition_id].get(
                    "decision_basis", "curated_or_source_derived"
                ),
                "evidence_used": allowed[partition_id].get("evidence_used", []),
            }
        },
    )
    return selection
