"""Load optional source-local workload metadata.

Legacy CNNImageProc sources have no descriptor and continue to use the static
schema. New workloads can provide workload.json to describe their source
bundle, stages, partition baselines, and generated wrapper conventions.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re


class WorkloadDefinitionError(RuntimeError):
    """Raised when workload.json does not satisfy the pipeline contract."""


def _require_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise WorkloadDefinitionError(f"{field} must be a non-empty string")
    return value.strip()


def _validate(definition, path):
    if not isinstance(definition, dict):
        raise WorkloadDefinitionError(f"{path} must contain a JSON object")
    if definition.get("schema_version") != 1:
        raise WorkloadDefinitionError(f"{path} schema_version must be 1")
    definition = deepcopy(definition)
    definition["workload"] = _require_string(definition.get("workload"), "workload")
    definition["top_function"] = _require_string(
        definition.get("top_function"), "top_function"
    )
    if not re.fullmatch(r"[A-Za-z_]\w*", definition["top_function"]):
        raise WorkloadDefinitionError("top_function must be a valid C identifier")

    required_files = definition.get("required_source_files")
    if not isinstance(required_files, list) or not required_files:
        raise WorkloadDefinitionError("required_source_files must be a non-empty list")
    definition["required_source_files"] = [
        _require_string(name, "required_source_files[]") for name in required_files
    ]

    stages = definition.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise WorkloadDefinitionError("stages must be a non-empty object")
    expected_ids = [f"S{index}" for index in range(1, len(stages) + 1)]
    if list(stages) != expected_ids:
        raise WorkloadDefinitionError(f"stages must be ordered consecutively as {expected_ids}")
    for stage_id, stage in stages.items():
        if not isinstance(stage, dict):
            raise WorkloadDefinitionError(f"{stage_id} must be an object")
        for field in ("function", "source", "call"):
            stage[field] = _require_string(stage.get(field), f"{stage_id}.{field}")
        if "{input}" not in stage["call"] or "{output}" not in stage["call"]:
            raise WorkloadDefinitionError(
                f"{stage_id}.call must contain {{input}} and {{output}} placeholders"
            )
        for direction in ("input", "output"):
            boundary = stage.get(direction)
            if not isinstance(boundary, dict):
                raise WorkloadDefinitionError(f"{stage_id}.{direction} must be an object")
            for field in ("name", "cpp_type", "shape_expression", "semantic"):
                boundary[field] = _require_string(
                    boundary.get(field), f"{stage_id}.{direction}.{field}"
                )
            if not isinstance(boundary.get("length"), int) or boundary["length"] <= 0:
                raise WorkloadDefinitionError(
                    f"{stage_id}.{direction}.length must be a positive integer"
                )

    headers = definition.get("header_files", ["src/lib.hpp"])
    if not isinstance(headers, list) or not headers:
        raise WorkloadDefinitionError("header_files must be a non-empty list")
    definition["header_files"] = [
        _require_string(name, "header_files[]") for name in headers
    ]
    definition.setdefault("top_includes", ["lib.hpp"])
    definition.setdefault("partitions", {})
    definition.setdefault("partition_priority", list(definition["partitions"]))
    return definition


def load_workload_definition(source_dir, required=False):
    """Return a validated workload.json object, or None for legacy workloads."""
    if source_dir is None:
        return None
    path = Path(source_dir).expanduser().resolve() / "workload.json"
    if not path.is_file():
        if required:
            raise WorkloadDefinitionError(f"workload descriptor does not exist: {path}")
        return None
    try:
        definition = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkloadDefinitionError(f"invalid JSON in {path}: {exc}") from exc
    definition = _validate(definition, path)
    definition["descriptor_path"] = str(path)
    return definition


def definition_from_manifest(source_manifest):
    definition = source_manifest.get("workload_definition")
    return deepcopy(definition) if isinstance(definition, dict) else None
