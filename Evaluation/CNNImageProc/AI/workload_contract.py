"""Generate a source-derived workload contract for CNNImageProc flows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from Evaluation.CNNImageProc.AI.schemas import STAGE_SPECS
from Evaluation.CNNImageProc.AI.source_graph import SourceGraphError, build_source_stage_graph
from Evaluation.CNNImageProc.AI.workload_definition import load_workload_definition


ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
}


def _fallback_source_graph(source_dir=None):
    return {
        "workload": "cnn_imageproc_v2",
        "source_dir": str(Path(source_dir).expanduser().resolve()) if source_dir else None,
        "top_function": "cnn_imageproc_v2",
        "stages": deepcopy(STAGE_SPECS),
        "call_order": list(STAGE_SPECS),
        "fallback": "manual CNNImageProc schema",
    }


def _definition_source_graph(source_dir, definition):
    stages = {}
    previous = None
    for stage_id, stage in definition["stages"].items():
        stages[stage_id] = {
            "function": stage["function"],
            "source": stage["source"],
            "input": {
                key: stage["input"][key]
                for key in ("length", "shape_expression", "semantic")
            },
            "output": {
                key: stage["output"][key]
                for key in ("length", "shape_expression", "semantic")
            },
            "input_array": stage["input"]["name"],
            "output_array": stage["output"]["name"],
            "producer_consumer_dependency": (
                f"input is produced by {previous}" if previous else "pipeline input"
            ),
        }
        previous = stage_id
    return {
        "workload": definition["workload"],
        "source_dir": str(Path(source_dir).expanduser().resolve()),
        "top_function": definition["top_function"],
        "stages": stages,
        "call_order": list(stages),
        "descriptor": definition["descriptor_path"],
    }


def _source_graph(source_dir, definition=None):
    if definition:
        return _definition_source_graph(source_dir, definition)
    if source_dir:
        try:
            return build_source_stage_graph(source_dir)
        except SourceGraphError as exc:
            graph = _fallback_source_graph(source_dir)
            graph["error"] = str(exc)
            return graph
    return _fallback_source_graph()


def _infer_block(stage_id, function, input_semantic, output_semantic):
    text = " ".join(value for value in (function, input_semantic, output_semantic) if value).lower()
    for pattern in (
        r"\bconv([0-9]+)\b",
        r"\bpool([0-9]+)\b",
        r"\brelu[_-]?pool([0-9]+)\b",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)), 0.95
    match = re.fullmatch(r"S([0-9]+)", stage_id)
    if match:
        # Pair adjacent conv/pool style stages into coarse feature blocks.
        return (int(match.group(1)) + 1) // 2, 0.45
    return None, 0.0


def _infer_role(function, input_semantic, output_semantic):
    function_text = (function or "").lower()
    semantic_text = " ".join(value for value in (input_semantic, output_semantic) if value).lower()
    if any(term in function_text for term in ("dense", "classifier")):
        return "classifier", 0.95
    if any(term in function_text for term in ("relu", "pool", "activation")):
        return "activation_pooling", 0.95
    if any(term in function_text for term in ("conv", "convolution")):
        return "convolution", 0.95
    if any(term in semantic_text for term in ("logits", "class")):
        return "classifier", 0.75
    if any(term in semantic_text for term in ("pool", "activation")):
        return "activation_pooling", 0.65
    if any(term in semantic_text for term in ("conv", "convolution")):
        return "convolution", 0.65
    return "compute", 0.35


def _aliases(stage_id, function, role, block):
    aliases = [stage_id.lower(), function]
    ordinal = ORDINALS.get(block)
    if role == "convolution":
        aliases.extend(["convolution", "conv"])
        if ordinal:
            aliases.extend([f"{ordinal} convolution", f"conv{block}"])
    elif role == "activation_pooling":
        aliases.extend(["activation", "pooling", "relu", "activation/pooling", "relu/pooling"])
        if ordinal:
            aliases.extend([
                f"{ordinal} activation",
                f"{ordinal} pooling",
                f"{ordinal} relu",
                f"{ordinal} activation/pooling",
                f"{ordinal} relu/pooling",
                f"pool{block}",
            ])
    elif role == "classifier":
        aliases.extend(["classifier", "dense", "dense classifier"])
    if block and role in {"convolution", "activation_pooling"}:
        aliases.append(f"{ORDINALS.get(block, block)} feature block")
    deduped = []
    for alias in aliases:
        alias = str(alias).strip().lower()
        if alias and alias not in deduped:
            deduped.append(alias)
    return deduped


def _block_aliases(stages):
    by_block = {}
    for stage_id, stage in stages.items():
        block = stage.get("block")
        if block is None:
            continue
        by_block.setdefault(block, []).append(stage_id)
    blocks = {}
    for block, stage_ids in sorted(by_block.items()):
        roles = {stages[stage_id]["role"] for stage_id in stage_ids}
        ordinal = ORDINALS.get(block, str(block))
        aliases = [f"{ordinal} feature block"]
        if {"convolution", "activation_pooling"} & roles:
            aliases.append(f"{ordinal} convolution/pooling block")
            aliases.append(f"{ordinal} convolution/pooling feature block")
            aliases.append(f"{ordinal} convolution and pooling block")
            aliases.append(f"{ordinal} convolution and pooling feature block")
        blocks[str(block)] = {
            "block": block,
            "stage_ids": stage_ids,
            "roles": sorted(roles),
            "aliases": aliases,
        }
    return blocks


def _add_terminal_aliases(stages):
    by_role = {}
    for stage_id, stage in stages.items():
        by_role.setdefault(stage.get("role"), []).append(stage_id)
    for role, stage_ids in by_role.items():
        if not stage_ids:
            continue
        terminal_stage = stages[stage_ids[-1]]
        aliases = terminal_stage["aliases"]
        if role == "activation_pooling":
            aliases.extend(["final pooling", "last pooling", "final activation/pooling", "last activation/pooling"])
        elif role == "classifier":
            aliases.extend(["final dense", "final classifier", "final dense classifier"])
        terminal_stage["aliases"] = list(dict.fromkeys(aliases))


def build_workload_contract(source_dir=None):
    """Build the contract consumed by prompts, sessions, and the GUI."""
    definition = load_workload_definition(source_dir)
    graph = _source_graph(source_dir, definition)
    stages = {}
    warnings = []
    for stage_id in graph.get("call_order", graph.get("stages", {}).keys()):
        raw = graph["stages"][stage_id]
        function = raw.get("function", "")
        input_semantic = raw.get("input", {}).get("semantic")
        output_semantic = raw.get("output", {}).get("semantic")
        described = definition["stages"].get(stage_id, {}) if definition else {}
        enriched = dict(raw)
        for direction in ("input", "output"):
            if isinstance(described.get(direction), dict):
                enriched[direction] = {
                    **raw.get(direction, {}),
                    **described[direction],
                }
        role = described.get("role")
        role_confidence = 1.0 if role else 0.0
        if not role:
            role, role_confidence = _infer_role(function, input_semantic, output_semantic)
        block = described.get("block")
        block_confidence = 1.0 if block is not None else 0.0
        if block is None:
            block, block_confidence = _infer_block(stage_id, function, input_semantic, output_semantic)
        if role == "classifier":
            block = None
        confidence = min(role_confidence, block_confidence if role != "classifier" else role_confidence)
        if confidence < 0.7:
            warnings.append(
                f"{stage_id} ({function}) has low-confidence role/block inference; review mapping."
            )
        stages[stage_id] = {
            **enriched,
            "role": role,
            "block": block,
            "aliases": _aliases(stage_id, function, role, block),
            "confidence": confidence,
        }
    _add_terminal_aliases(stages)
    return {
        "schema_version": 1,
        "workload": graph.get("workload", "cnn_imageproc_v2"),
        "source_dir": graph.get("source_dir"),
        "top_function": graph.get("top_function"),
        "stages": stages,
        "blocks": _block_aliases(stages),
        "call_order": graph.get("call_order", list(stages)),
        "source_graph": graph,
        "warnings": warnings,
        "definition": definition,
    }


def stage_glossary(contract):
    glossary = {}
    for stage_id, stage in contract.get("stages", {}).items():
        role = stage.get("role", "compute").replace("_", "/")
        block = stage.get("block")
        block_text = f", block {block}" if block is not None and role != "classifier" else ""
        glossary[stage_id] = f"{role}{block_text}: {stage.get('function')}"
    return glossary
