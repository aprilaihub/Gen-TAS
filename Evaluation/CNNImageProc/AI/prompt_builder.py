"""Prompt builder for CNNImageProc FPGA/GPP strategy selection."""

import json
import hashlib
import math
from pathlib import Path
import re
import time
from collections import Counter

from Evaluation.CNNImageProc.AI.schemas import REQUIRED_SOURCE_FILES
from Evaluation.CNNImageProc.AI.experiment_metrics import task_characteristics
from Evaluation.CNNImageProc.AI.workload_contract import (
    build_workload_contract,
    stage_glossary,
)


SYSTEM_PROMPT = """You are an FPGA/GPP allocation advisor for the supplied workload.
You must read the supplied source summaries and propose FPGA/GPP partition strategies.
Use previous dataset records only as qualitative evidence, not as a fixed candidate list.
When prior evidence is used, cite its exact supplied kb_task_id in related_prior_dataset_examples.
Identify tightly coupled subfunctions, shared intermediate buffers, producer-consumer dependencies,
and data-movement costs before proposing allocations.
Do not invent workload latency, resource, timing, or power numbers.
If numeric workload evidence is missing, explicitly say it must be measured."""


OUTPUT_CONTRACT = {
    "recommendations": [
        {
            "rank": 1,
            "partition_id": "string",
            "fpga_subfunctions": ["S1"],
            "gpp_subfunctions": ["S2"],
            "task_groups": [
                {
                    "group_id": "string",
                    "subfunctions": ["S1"],
                    "reason": "string",
                }
            ],
            "recommendation": "string",
            "expected_latency_impact": "qualitative string; no invented numbers",
            "resource_impact": "qualitative string; no invented numbers",
            "power_impact": "qualitative string; no invented numbers",
            "timing_assessment": "qualitative string; no invented numbers",
            "interface_and_io_mapping": "string",
            "source_code_evidence": ["string"],
            "related_prior_dataset_examples": ["exact supplied kb_task_id"],
            "evidence_used": ["string"],
            "risks_and_assumptions": ["string"],
        }
    ],
    "baseline_comparison": "string",
    "final_recommendation": "string",
}


def _read_text(path, limit=80_000):
    path = Path(path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        text = text[:limit] + f"\n... truncated after {limit} characters ...\n"
    return text


def _source_bundle(source_dir, workload_contract):
    source_dir = Path(source_dir).expanduser().resolve()
    bundle = {}
    definition = workload_contract.get("definition")
    required_files = (
        definition["required_source_files"]
        if definition
        else REQUIRED_SOURCE_FILES
    )
    for name in required_files:
        limit = 12_000 if name.endswith("weights.hpp") else 35_000
        content = _read_text(source_dir / name, limit=limit)
        if content is not None:
            bundle[name] = content
    return bundle


def _tokens(value):
    return Counter(re.findall(r"[a-z0-9]+", value.lower()))


def _cosine_similarity(left, right):
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    dot = sum(count * right_tokens.get(token, 0) for token, count in left_tokens.items())
    left_norm = math.sqrt(sum(count * count for count in left_tokens.values()))
    right_norm = math.sqrt(sum(count * count for count in right_tokens.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _lightcnn_evidence(limit=12, query="", with_metrics=False):
    started = time.perf_counter_ns()
    root = Path(__file__).resolve().parents[3]
    candidates = [
        root / "Evaluation" / "LightCNN" / "KnowledgeBase" / "active" / "lightcnn_evidence.json",
    ]
    records = []
    kb_digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            continue
        kb_digest.update(path.name.encode("utf-8"))
        if path.suffix == ".jsonl":
            kb_digest.update(path.read_bytes())
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("include_in_allocation_profiles") is False:
                    continue
                records.append(_summarize_lightcnn_record(record))
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if payload.get("schema_version") == "lightcnn-active-kb-v1":
                kb_digest.update(
                    payload.get("knowledge_base_version", "").encode("utf-8")
                )
                for profile in payload.get("profiles", []):
                    records.append(_summarize_active_lightcnn_profile(profile))
                for task in payload.get("tasks", []):
                    records.append(_summarize_active_lightcnn_task(task))
                continue
            kb_digest.update(path.read_bytes())
            if path.name == "kb_characterization.json":
                for task in payload.get("subfunctions", []):
                    records.append(_summarize_lightcnn_task(task))
                continue
            iterable = payload if isinstance(payload, list) else payload.get("records", [])
            for record in iterable:
                if isinstance(record, dict) and record.get("include_in_allocation_profiles") is not False:
                    records.append(_summarize_lightcnn_record(record))

    ranked = []
    unique_records = {}
    for index, record in enumerate(records):
        task_id = "lightcnn:{stage}:{name}".format(
            stage=record.get("implementation_stage") or "unknown",
            name=record.get("design_name") or f"record-{index + 1}",
        )
        unique_records.setdefault(task_id, record)
    for task_id, record in unique_records.items():
        searchable = json.dumps(record, sort_keys=True)
        enriched = dict(record)
        enriched["kb_task_id"] = task_id
        enriched["retrieval_similarity_score"] = round(
            _cosine_similarity(query, searchable), 6
        )
        ranked.append(enriched)
    ranked.sort(
        key=lambda item: (-item["retrieval_similarity_score"], item["kb_task_id"])
    )
    selected = ranked[:limit]
    task_records = [
        item for item in ranked if item.get("implementation_stage") == "task_characterization"
    ]
    partition_keys = {
        (tuple(item.get("fpga_subfunctions", [])), tuple(item.get("gpp_subfunctions", [])))
        for item in ranked
        if item.get("implementation_stage") != "task_characterization"
        if item.get("fpga_subfunctions") is not None and item.get("gpp_subfunctions") is not None
    }
    hardware_designs = {
        (tuple(item.get("fpga_subfunctions", [])), tuple(item.get("gpp_subfunctions", [])))
        for item in ranked
        if item.get("fpga_subfunctions")
        and item.get("implementation_stage") != "task_characterization"
    }
    metrics = {
        "method": "deterministic_lexical_cosine",
        "kb_version": "sha256:" + kb_digest.hexdigest(),
        "retrieval_latency_ns": time.perf_counter_ns() - started,
        "number_retrieved": len(selected),
        "retrieval_success": bool(selected),
        "retrieval_confidence": max(
            (item["retrieval_similarity_score"] for item in selected), default=0.0
        ),
        "retrieved_task_ids": [item["kb_task_id"] for item in selected],
        "similarity_scores": [
            {
                "kb_task_id": item["kb_task_id"],
                "score": item["retrieval_similarity_score"],
            }
            for item in selected
        ],
        "correct_task_category_retrieval_rate": None,
        "correct_task_category_note": "Requires labelled expected retrievals for each benchmark query.",
        "kb_growth": {
            "number_of_tasks": len(task_records),
            "number_of_applications": len({item.get("workload") for item in ranked if item.get("workload")}),
            "number_of_historical_partitions": len(partition_keys),
            "number_of_hardware_implementations": len(hardware_designs),
            "number_of_kb_records": len(ranked),
            "counting_method": "unique versioned KB records after exclusion and deduplication",
        },
    }
    return (selected, metrics) if with_metrics else selected


def _summarize_lightcnn_record(record):
    summary = {
        "design_name": record.get("design_name") or record.get("partition_id"),
        "workload": record.get("workload", "LightCNN"),
        "implementation_stage": record.get("implementation_stage"),
        "fpga_subfunctions": record.get("fpga_subfunctions", []),
        "gpp_subfunctions": record.get("gpp_subfunctions", []),
        "notes": record.get("notes"),
    }
    if isinstance(record.get("latency_ms"), dict):
        summary["latency_ms_mean"] = record["latency_ms"].get("mean")
    if isinstance(record.get("resources"), dict):
        summary["resources"] = {
            key: record["resources"].get(key)
            for key in ("clb_luts", "clb_registers_ff", "bram_tiles")
            if key in record["resources"]
        }
    if isinstance(record.get("power"), dict):
        summary["power_total_on_chip_w"] = record["power"].get("total_on_chip_w")
    return summary


def _summarize_active_lightcnn_profile(profile):
    metrics = profile.get("summary_metrics", {})
    return {
        "design_name": profile.get("record_id") or profile.get("partition_id"),
        "workload": "LightCNN",
        "implementation_stage": "measured_partition",
        "partition_id": profile.get("partition_id"),
        "placement_id": profile.get("placement_id"),
        "topology": profile.get("topology"),
        "selection_eligible": profile.get("selection_eligible", True),
        "fpga_subfunctions": profile.get("fpga_subfunctions", []),
        "gpp_subfunctions": profile.get("gpp_subfunctions", []),
        "latency_ms_mean": metrics.get("pynq_mean_latency_ms"),
        "latency_ms_median": metrics.get("pynq_median_latency_ms"),
        "golden_vector_verification_pass": metrics.get(
            "golden_vector_verification_pass"
        ),
        "resources": {
            "clb_luts": metrics.get("vivado_clb_luts"),
            "clb_registers_ff": metrics.get("vivado_clb_registers_ff"),
            "bram18_equiv": metrics.get("vivado_bram18_equiv"),
            "dsp": metrics.get("vivado_dsp"),
        },
        "timing_met": metrics.get("vivado_timing_constraints_met"),
        "power_total_on_chip_w": metrics.get("vivado_total_on_chip_power_w"),
        "communication": profile.get("communication", {}),
    }


def _summarize_active_lightcnn_task(task):
    input_bytes = task.get("input", {}).get("bytes")
    output_bytes = task.get("output", {}).get("bytes")
    communication = (
        input_bytes + output_bytes
        if isinstance(input_bytes, int) and isinstance(output_bytes, int)
        else None
    )
    return {
        "design_name": task.get("record_id") or f"lightcnn_task_{task.get('task_id')}",
        "workload": "LightCNN",
        "implementation_stage": "task_characterization",
        "fpga_subfunctions": [task["task_id"]] if task.get("task_id") else [],
        "gpp_subfunctions": [],
        "task_type": task.get("task_type"),
        "compute_intensity": task.get("compute_intensity"),
        "memory_intensity": task.get("memory_intensity"),
        "communication_volume_bytes": communication,
        "communication_intensity": task.get("communication_intensity"),
        "control_intensity": task.get("control_intensity"),
        "operation_count": task.get("mac_count", task.get("operation_count")),
    }


def _summarize_lightcnn_task(task):
    task_id = task.get("id")
    operations = task.get("operations", {})
    operation_count = sum(value for value in operations.values() if isinstance(value, (int, float)))
    input_bytes = task.get("input", {}).get("bytes")
    output_bytes = task.get("output", {}).get("bytes")
    communication = input_bytes + output_bytes if isinstance(input_bytes, int) and isinstance(output_bytes, int) else None
    profiles = {
        "convolutional_feature_extraction": ("high", "medium", "low"),
        "activation_and_max_pooling": ("medium", "medium", "medium"),
        "dense_classification": ("high", "high", "low"),
    }
    compute, memory, control = profiles.get(task.get("task_type"), ("medium", "medium", "medium"))
    return {
        "design_name": f"lightcnn_task_{task_id}",
        "workload": "LightCNN",
        "implementation_stage": "task_characterization",
        "fpga_subfunctions": [task_id] if task_id else [],
        "gpp_subfunctions": [],
        "task_type": task.get("task_type"),
        "compute_intensity": compute,
        "memory_intensity": memory,
        "communication_volume_bytes": communication,
        "control_intensity": control,
        "operation_count": operation_count,
        "memory_pattern": task.get("memory_pattern"),
        "control": task.get("control"),
    }


def _task_family(task_type):
    mapping = {
        "convolution": "convolution",
        "convolutional_feature_extraction": "convolution",
        "activation_pooling": "activation_pooling",
        "activation_and_max_pooling": "activation_pooling",
        "relu_max_pool": "activation_pooling",
        "classifier": "classifier",
        "dense_classification": "classifier",
    }
    return mapping.get(task_type, task_type)


def _retrieval_coverage(query_tasks, retrieved_profiles):
    kb_tasks = [
        item for item in retrieved_profiles
        if item.get("implementation_stage") == "task_characterization"
    ]
    coverage = []
    for task in query_tasks:
        exact = [
            item for item in kb_tasks
            if _task_family(item.get("task_type")) == _task_family(task.get("task_type"))
        ]
        partial = []
        if not exact:
            signature = (
                task.get("compute_intensity"), task.get("memory_intensity"),
                task.get("control_intensity"),
            )
            partial = [
                item for item in kb_tasks
                if sum(a == b for a, b in zip(signature, (
                    item.get("compute_intensity"), item.get("memory_intensity"),
                    item.get("control_intensity"),
                ))) >= 2
            ]
        matches = exact or partial
        coverage.append({
            "task_id": task.get("task_id"),
            "task_type": task.get("task_type"),
            "coverage": "exact_match" if exact else "partial_match" if partial else "no_match",
            "matched_kb_task_ids": [item.get("kb_task_id") for item in matches],
        })
    counts = {
        status: sum(item["coverage"] == status for item in coverage)
        for status in ("exact_match", "partial_match", "no_match")
    }
    total = len(coverage)
    return {
        "tasks": coverage,
        "counts": counts,
        "exact_match_percent": 100.0 * counts["exact_match"] / total if total else None,
        "partial_match_percent": 100.0 * counts["partial_match"] / total if total else None,
        "no_match_percent": 100.0 * counts["no_match"] / total if total else None,
        "inferred_task_percent": 100.0 * (counts["partial_match"] + counts["no_match"]) / total if total else None,
    }


def _dedupe_blocks(blocks):
    deduped = []
    for block in blocks:
        if block and block not in deduped:
            deduped.append(block)
    return deduped


def _stages_by(contract, role=None, block=None):
    stages = []
    for stage_id, stage in contract.get("stages", {}).items():
        if role is not None and stage.get("role") != role:
            continue
        if block is not None and stage.get("block") != block:
            continue
        stages.append(stage_id)
    return sorted(stages, key=lambda value: int(value[1:]) if value[1:].isdigit() else value)


def _has_any(text, phrases):
    return any(phrase and phrase in text for phrase in phrases)


def _negative_context(text, phrases):
    for phrase in phrases:
        if not phrase:
            continue
        negative_forms = (
            f"keep {phrase}",
            f"keep the {phrase}",
            f"keep both {phrase} stages",
            f"leave {phrase}",
            f"leave the {phrase}",
            f"leave both {phrase} stages",
            f"leaving {phrase}",
            f"leaving the {phrase}",
            f"leaving both {phrase} stages",
            f"{phrase} on the gpp",
            f"{phrase} on gpp",
            f"{phrase} stages on the gpp",
            f"{phrase} stages on gpp",
            f"{phrase} work on the gpp",
        )
        if any(form in text for form in negative_forms):
            return True
    return False


def _listed_gpp_context(text, phrase):
    if not phrase:
        return False
    for starter in ("leaving", "leave", "keeping", "keep"):
        start = text.find(starter)
        while start != -1:
            end_candidates = [
                index for index in (text.find(" on gpp", start), text.find(" on the gpp", start))
                if index != -1
            ]
            if end_candidates:
                end = min(end_candidates)
                if phrase in text[start:end]:
                    return True
            start = text.find(starter, start + 1)
    return False


def _positive_context(text, phrases):
    for phrase in phrases:
        if not phrase:
            continue
        positive_forms = (
            f"accelerate {phrase}",
            f"accelerate the {phrase}",
            f"offload {phrase}",
            f"offload the {phrase}",
            f"map {phrase} to fpga",
            f"map the {phrase} to fpga",
            f"{phrase} to fpga",
            f"{phrase} on fpga",
        )
        if any(form in text for form in positive_forms):
            return True
    return False


def _stages_after(contract, stage_id):
    call_order = contract.get("call_order", list(contract.get("stages", {})))
    if stage_id not in call_order:
        return []
    return call_order[call_order.index(stage_id) + 1:]


def _stage_specific_aliases(stage):
    generic_positive_aliases = {
        "convolution",
        "conv",
        "activation",
        "pooling",
        "relu",
        "activation/pooling",
        "relu/pooling",
        "classifier",
        "dense",
        "dense classifier",
    }
    return [
        alias
        for alias in stage.get("aliases", [])
        if "feature block" not in alias
        and "convolution/pooling block" not in alias
        and alias not in generic_positive_aliases
    ]


def _stage_range_from_text(contract, text):
    call_order = contract.get("call_order", list(contract.get("stages", {})))
    for start_index, start_id in enumerate(call_order):
        start_aliases = _stage_specific_aliases(contract["stages"][start_id])
        for end_index, end_id in enumerate(call_order[start_index:], start=start_index):
            end_aliases = _stage_specific_aliases(contract["stages"][end_id])
            for start_alias in start_aliases:
                for end_alias in end_aliases:
                    range_forms = (
                        f"from {start_alias} through {end_alias}",
                        f"from the {start_alias} through the {end_alias}",
                        f"from {start_alias} to {end_alias}",
                        f"from the {start_alias} to the {end_alias}",
                    )
                    if any(form in text for form in range_forms):
                        return call_order[start_index : end_index + 1]
    return []


def _merge_adjacent_singletons(contract, blocks):
    call_order = contract.get("call_order", list(contract.get("stages", {})))
    order_index = {stage_id: index for index, stage_id in enumerate(call_order)}
    singleton_indexes = sorted(
        order_index[block[0]]
        for block in blocks
        if len(block) == 1 and block[0] in order_index
    )
    merged = []
    run = []
    for index in singleton_indexes:
        if not run or index == run[-1] + 1:
            run.append(index)
        else:
            if len(run) > 1:
                merged.append([call_order[item] for item in run])
            run = [index]
    if len(run) > 1:
        merged.append([call_order[item] for item in run])
    return merged + blocks


def _request_intent(user_request, goal, contract):
    text = f"{user_request} {goal}".lower()
    hints = []
    preferred_blocks = []
    avoid_blocks = []
    specific_positive_roles = set()
    generic_positive_aliases = {
        "convolution",
        "conv",
        "activation",
        "pooling",
        "relu",
        "activation/pooling",
        "relu/pooling",
        "classifier",
        "dense",
        "dense classifier",
    }

    valid_stage_ids = set(contract.get("stages", {}))
    stage_mentions = sorted(
        {
            stage.upper()
            for stage in re.findall(r"\bs\d+\b", text)
            if stage.upper() in valid_stage_ids
        },
        key=lambda stage: int(stage[1:]),
    )
    if stage_mentions:
        indexes = [int(stage[1:]) for stage in stage_mentions]
        if indexes == list(range(indexes[0], indexes[-1] + 1)):
            preferred_blocks.append(stage_mentions)

    stage_range = _stage_range_from_text(contract, text)
    if stage_range:
        preferred_blocks.append(stage_range)

    for stage_id, stage in contract.get("stages", {}).items():
        if stage.get("role") != "activation_pooling":
            continue
        aliases = [
            alias
            for alias in stage.get("aliases", [])
            if alias not in generic_positive_aliases and "feature block" not in alias
        ]
        after_aliases = [f"after {alias} stage" for alias in aliases] + [
            f"after the {alias} stage" for alias in aliases
        ]
        if _has_any(text, after_aliases):
            preferred_blocks.append(_stages_after(contract, stage_id))

    for block_record in contract.get("blocks", {}).values():
        stage_ids = block_record.get("stage_ids", [])
        aliases = block_record.get("aliases", [])
        if not _has_any(text, aliases):
            continue
        if _negative_context(text, aliases):
            avoid_blocks.append(stage_ids)
        else:
            preferred_blocks.append(stage_ids)

    # Combine adjacent source-derived concepts when the user requests a block
    # plus the next named stage, e.g. first feature block + second convolution.
    for block_record in contract.get("blocks", {}).values():
        block_aliases = block_record.get("aliases", [])
        if not _has_any(text, block_aliases) or _negative_context(text, block_aliases):
            continue
        block_number = block_record.get("block")
        next_convolution = _stages_by(contract, role="convolution", block=block_number + 1)
        next_aliases = []
        for stage_id in next_convolution:
            next_aliases.extend(
                alias
                for alias in contract["stages"][stage_id].get("aliases", [])
                if alias not in generic_positive_aliases and "feature block" not in alias
            )
        if next_convolution and _has_any(text, next_aliases) and not _negative_context(text, next_aliases):
            preferred_blocks.insert(0, _dedupe_blocks([block_record["stage_ids"] + next_convolution])[0])

    for stage_id, stage in contract.get("stages", {}).items():
        aliases = _stage_specific_aliases(stage)
        if not _has_any(text, aliases):
            continue
        after_aliases = [f"after {alias} stage" for alias in aliases] + [
            f"after the {alias} stage" for alias in aliases
        ]
        if _has_any(text, after_aliases):
            continue
        if _negative_context(text, aliases) or any(_listed_gpp_context(text, alias) for alias in aliases):
            avoid_blocks.append([stage_id])
        else:
            preferred_blocks.append([stage_id])
            specific_positive_roles.add(stage.get("role"))

    for role, readable, phrases in (
        ("convolution", "convolution", ["convolution", "conv"]),
        ("activation_pooling", "activation/pooling", ["activation/pooling", "activation and pooling", "relu/pooling", "pooling"]),
        ("classifier", "classifier", ["classifier", "dense classifier", "dense"]),
    ):
        role_stages = _stages_by(contract, role=role)
        if not role_stages:
            continue
        if _has_any(text, phrases) and _negative_context(text, phrases):
            hints.append(f"If the request keeps {readable} work on the GPP, do not assign those stages to FPGA.")
            for stage_id in role_stages:
                avoid_blocks.append([stage_id])
        elif role not in specific_positive_roles and _positive_context(text, phrases):
            preferred_blocks.append(role_stages)

    feature_extractor_stages = [
        stage_id
        for stage_id in contract.get("call_order", contract.get("stages", {}))
        if contract["stages"][stage_id].get("role") in {"convolution", "activation_pooling"}
    ]
    feature_phrases = [
        "convolution and pooling feature extractor",
        "convolution/pooling feature extractor",
        "feature extractor",
        "all convolution/pooling stages",
        "convolution/pooling stages",
    ]
    if _has_any(text, feature_phrases):
        if _negative_context(text, feature_phrases):
            for stage_id in feature_extractor_stages:
                avoid_blocks.append([stage_id])
        elif _positive_context(text, feature_phrases):
            preferred_blocks.append(feature_extractor_stages)

    if "diagnostic" in text or "only" in text or "smallest" in text or "minimal" in text:
        hints.append("Prefer the smallest contiguous FPGA block that matches the request.")
    if goal == "resource" or "resource" in text:
        hints.append("Resource-focused requests should rank smaller matching FPGA blocks ahead of broad all-FPGA designs.")
        avoid_blocks.append(list(contract.get("stages", {})))
    if goal == "power" or "power" in text:
        hints.append("Power-focused requests should avoid full-FPGA unless the request explicitly asks for maximum acceleration.")
        avoid_blocks.append(list(contract.get("stages", {})))
    if goal == "latency" or "latency" in text:
        hints.append("Latency-focused requests may prefer broad FPGA blocks, but should still follow explicit stage/block wording.")
    if goal == "balanced" or "balanced" in text or "reasonable" in text:
        hints.append("Balanced requests should trade latency against FPGA resource/power and follow the explicit block wording.")
    if "full" in text or "maximum" in text or "as much as possible" in text:
        all_stages = list(contract.get("stages", {}))
        preferred_blocks.append(all_stages)
        avoid_blocks = [block for block in avoid_blocks if block != all_stages]

    preferred_blocks = _merge_adjacent_singletons(contract, preferred_blocks)
    preferred_sets = [set(block) for block in preferred_blocks]
    if preferred_sets:
        preferred_blocks = [
            block
            for block in preferred_blocks
            if not any(set(block) < other for other in preferred_sets)
        ]

    deduped_preferred = _dedupe_blocks(preferred_blocks)
    deduped_avoid = []
    for block in avoid_blocks:
        if any(set(block).issubset(set(preferred)) for preferred in deduped_preferred):
            continue
        if block not in deduped_avoid and block not in deduped_preferred:
            deduped_avoid.append(block)

    return {
        "goal": goal,
        "preferred_fpga_blocks": deduped_preferred,
        "avoid_default_blocks_when_not_requested": deduped_avoid,
        "request_specific_guidance": hints,
    }


def build_prompt(
    user_request="",
    requirements=None,
    top_k=3,
    include_lightcnn_context=True,
    source_dir=None,
):
    """Build the complete system/user prompt pair."""
    requirements = requirements or {}
    primary_goal = requirements.get("primary_goal", "")
    workload_contract = build_workload_contract(source_dir)
    semantic_tasks = task_characteristics(workload_contract)
    retrieval_query = " ".join((
        user_request,
        json.dumps(requirements, sort_keys=True),
        json.dumps(semantic_tasks, sort_keys=True),
    ))
    retrieved_profiles, rag_metrics = _lightcnn_evidence(
        query=retrieval_query, with_metrics=True
    ) if include_lightcnn_context else ([], {})
    expected_ids = requirements.get("expected_kb_task_ids", [])
    if expected_ids:
        retrieved_ids = set(rag_metrics.get("retrieved_task_ids", []))
        rag_metrics["retrieval_success_rate"] = (
            len(retrieved_ids.intersection(expected_ids)) / len(set(expected_ids))
        )
    else:
        rag_metrics["retrieval_success_rate"] = None
        rag_metrics["retrieval_success_rate_note"] = (
            "Provide requirements.expected_kb_task_ids for labelled RAG evaluation."
        )
    expected_categories = requirements.get("expected_task_categories", [])
    if expected_categories:
        categories = {item.get("implementation_stage") for item in retrieved_profiles}
        rag_metrics["correct_task_category_retrieval_rate"] = (
            len(categories.intersection(expected_categories)) / len(set(expected_categories))
        )
    rag_metrics["retrieval_coverage"] = _retrieval_coverage(
        semantic_tasks, retrieved_profiles
    )
    payload = {
        "task": f"Generate {workload_contract['workload']} FPGA/GPP partition strategies from source evidence.",
        "user_request": user_request,
        "requirements": requirements,
        "workload_contract": workload_contract,
        "task_characteristics": semantic_tasks,
        "stage_glossary": stage_glossary(workload_contract),
        "stage_graph": workload_contract["source_graph"]["stages"],
        "request_intent": _request_intent(user_request, primary_goal, workload_contract),
        "source_files": _source_bundle(source_dir, workload_contract) if source_dir else {},
        "top_k": top_k,
        "lightcnn_transfer_context": (
            {
                "allowed_use": "qualitative strategy reasoning only",
                "prior_dataset_records": retrieved_profiles,
                "not_allowed": "numeric resource, power, timing, or latency prediction for the current workload",
                "reason": (
                    "The current workload's data types, lookup tables, DSP/BRAM use, "
                    "and measured latency are substantially different from LightCNN."
                ),
            }
            if include_lightcnn_context
            else None
        ),
        "instructions": [
            "Return exactly the requested JSON object.",
            "Read the source files and identify computational subfunctions.",
            "Group subfunctions that are tightly coupled, share intermediate buffers, or should not be separated due to data movement.",
            "Use previous dataset records only as evidence, not as a fixed candidate list.",
            "Generate up to top_k valid FPGA/GPP allocation strategies.",
            "Make recommendations request-specific; do not return the same default trio for every goal.",
            "If request_intent.preferred_fpga_blocks is non-empty, include the first valid preferred block as rank 1 unless source evidence clearly contradicts it.",
            "If request_intent.avoid_default_blocks_when_not_requested names broad defaults, do not include those broad defaults unless explicitly justified by the user request.",
            "Use only stage IDs from workload_contract.stages.",
            "Use workload_contract roles, blocks, aliases, and stage_glossary as the source of truth for stage meaning.",
            "If the user asks to keep a role or block on the GPP, do not assign those stages to the FPGA.",
            "Assign every stage exactly once across fpga_subfunctions and gpp_subfunctions.",
            "For buildable single-IP tests, fpga_subfunctions must be one contiguous stage block.",
            "Prefer buildable strategies with at least one FPGA stage.",
            "Do not invent numeric workload measurements.",
            "Mention that actual resource, power, and latency require HLS/Vivado/PYNQ measurement.",
        ],
        "output_contract": OUTPUT_CONTRACT,
    }
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": json.dumps(payload, indent=2),
        "retrieved_profiles": payload["lightcnn_transfer_context"]["prior_dataset_records"]
        if include_lightcnn_context
        else [],
        "workload_definition": workload_contract.get("definition"),
        "rag_metrics": rag_metrics,
    }
