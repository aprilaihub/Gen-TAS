"""LLM-backed CNNImageProc allocation strategy agent."""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from uuid import uuid4

from Evaluation.CNNImageProc.AI.prompt_builder import build_prompt
from Evaluation.CNNImageProc.AI.schemas import (
    SchemaError,
    build_partition_spec,
    make_partition_id,
)
from LLM_Interface.LLMClient import LLMClient


CNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(CNN_DIR, "Outputs")


def try_parse_json(text):
    """Best-effort parse for model responses that follow the JSON contract."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _aggregate_token_usage(usages, model):
    totals = {}
    for name in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
    ):
        values = [usage.get(name) for usage in usages if isinstance(usage.get(name), int)]
        totals[name] = sum(values) if values else None
    return {
        "provider": usages[-1].get("provider") if usages else None,
        "model": model,
        **totals,
        "count_source": "provider_reported_multiple_calls" if len(usages) > 1 else (
            usages[0].get("count_source") if usages else "unavailable"
        ),
        "request_parameters": [usage.get("request_parameters") for usage in usages],
        "attempt_count": len(usages),
        "finish_reasons": [usage.get("finish_reason") for usage in usages],
        "truncated_attempts": [
            index + 1 for index, usage in enumerate(usages) if usage.get("truncated")
        ],
    }


def _build_repair_prompt(response, prompt_package, top_k):
    contract = prompt_package.get("workload_contract", {})
    if not contract:
        try:
            contract = json.loads(prompt_package.get("user_prompt", "{}" )).get(
                "workload_contract", {}
            )
        except (json.JSONDecodeError, TypeError):
            contract = {}
    stages = contract.get("stages", {}) if isinstance(contract, dict) else {}
    return json.dumps(
        {
            "task": "Repair the incomplete allocation response into compact valid JSON.",
            "rules": [
                "Return JSON only, with no Markdown or prose outside the object.",
                f"Return at most {top_k} recommendations.",
                "Assign every supplied stage exactly once between FPGA and GPP.",
                "Use one contiguous non-empty FPGA stage block.",
                "Do not invent numeric measurements.",
                "Keep every string concise so the response cannot be truncated.",
            ],
            "valid_stage_ids": list(stages),
            "required_shape": {
                "recommendations": [
                    {
                        "rank": 1,
                        "partition_id": "short label",
                        "fpga_subfunctions": ["stage ID"],
                        "gpp_subfunctions": ["stage ID"],
                        "recommendation": "concise rationale",
                        "source_code_evidence": ["concise evidence"],
                        "related_prior_dataset_examples": ["exact supplied kb_task_id"],
                        "evidence_used": ["concise evidence"],
                        "risks_and_assumptions": ["concise risk"],
                    }
                ],
                "baseline_comparison": "concise qualitative comparison",
                "final_recommendation": "partition_id",
            },
            "incomplete_response": response,
        },
        separators=(",", ":"),
    )
def make_output_path(output_dir, model):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_model = model.replace("/", "_").replace(":", "_")
    unique_suffix = uuid4().hex[:8]
    return os.path.join(
        output_dir,
        f"allocation_options_{safe_model}_{timestamp}_{unique_suffix}.json",
    )


def _filter_recommendations(
    parsed, top_k, require_hardware=True, workload_definition=None, telemetry=None
):
    telemetry = telemetry if telemetry is not None else {}
    telemetry.update({
        "submitted_allocation_count": 0,
        "invalid_allocation_count": 0,
        "duplicate_allocation_count": 0,
        "non_buildable_allocation_count": 0,
        "accepted_allocation_count": 0,
    })
    if not isinstance(parsed, dict):
        telemetry["response_parse_valid"] = False
        return None
    telemetry["response_parse_valid"] = True
    valid = []
    seen = set()
    dynamic_specs = {}
    for item in parsed.get("recommendations", []):
        telemetry["submitted_allocation_count"] += 1
        if not isinstance(item, dict):
            telemetry["invalid_allocation_count"] += 1
            continue
        fpga_subfunctions = item.get("fpga_subfunctions", [])
        gpp_subfunctions = item.get("gpp_subfunctions")
        try:
            partition_id = item.get("partition_id") or make_partition_id(
                fpga_subfunctions, workload_definition
            )
            if not isinstance(partition_id, str):
                telemetry["invalid_allocation_count"] += 1
                continue
            if not partition_id.startswith("LLM_"):
                partition_id = make_partition_id(fpga_subfunctions, workload_definition)
            if partition_id in seen:
                telemetry["duplicate_allocation_count"] += 1
                continue
            spec = build_partition_spec(
                partition_id=partition_id,
                fpga_subfunctions=fpga_subfunctions,
                gpp_subfunctions=gpp_subfunctions,
                summary=item.get("recommendation") or item.get("summary"),
                workload_definition=workload_definition,
            )
        except SchemaError:
            telemetry["invalid_allocation_count"] += 1
            continue
        if require_hardware and not spec["hardware_generable"]:
            telemetry["non_buildable_allocation_count"] += 1
            continue
        normalized = dict(item)
        normalized["rank"] = len(valid) + 1
        normalized["partition_id"] = partition_id
        normalized["fpga_subfunctions"] = spec["fpga_subfunctions"]
        normalized["gpp_subfunctions"] = spec["gpp_subfunctions"]
        normalized["hardware_generable"] = spec["hardware_generable"]
        normalized.setdefault("summary", spec["summary"])
        valid.append(normalized)
        telemetry["accepted_allocation_count"] += 1
        seen.add(partition_id)
        dynamic_specs[partition_id] = {
            "partition_id": partition_id,
            "fpga_subfunctions": spec["fpga_subfunctions"],
            "gpp_subfunctions": spec["gpp_subfunctions"],
            "summary": spec["summary"],
        }
        if len(valid) == top_k:
            break
    if not valid:
        telemetry["valid_allocation_rate"] = 0.0
        return None
    submitted = telemetry["submitted_allocation_count"]
    telemetry["valid_allocation_rate"] = len(valid) / submitted if submitted else 0.0
    telemetry["constraint_violation_count"] = (
        telemetry["invalid_allocation_count"]
        + telemetry["duplicate_allocation_count"]
        + telemetry["non_buildable_allocation_count"]
    )
    result = dict(parsed)
    result["recommendations"] = valid
    result["dynamic_partition_specs"] = dynamic_specs
    return result


def _intent_recommendation(fpga_subfunctions, workload_definition=None):
    partition_id = make_partition_id(fpga_subfunctions, workload_definition)
    spec = build_partition_spec(
        partition_id=partition_id,
        fpga_subfunctions=fpga_subfunctions,
        summary="Contract-derived preferred FPGA/GPP allocation.",
        workload_definition=workload_definition,
    )
    return {
        "rank": 1,
        "partition_id": partition_id,
        "fpga_subfunctions": spec["fpga_subfunctions"],
        "gpp_subfunctions": spec["gpp_subfunctions"],
        "task_groups": [
            {
                "group_id": "contract_preferred_block",
                "subfunctions": spec["fpga_subfunctions"],
                "reason": "Matches the first preferred FPGA block inferred from the workload contract and user request.",
            }
        ],
        "recommendation": (
            "Use the first request-specific preferred FPGA block inferred from the workload contract."
        ),
        "expected_latency_impact": "Qualitative only; workload latency must be measured.",
        "resource_impact": "Qualitative only; workload resources must be measured.",
        "power_impact": "Qualitative only; workload power must be measured.",
        "timing_assessment": "Timing closure must be checked in HLS/Vivado.",
        "interface_and_io_mapping": "Use the selected contiguous stage boundary from the workload contract.",
        "source_code_evidence": ["Generated workload contract request_intent preferred_fpga_blocks."],
        "related_prior_dataset_examples": [],
        "evidence_used": ["workload_contract", "request_intent"],
        "risks_and_assumptions": ["LLM-generated rationale may need review against measured hardware results."],
        "hardware_generable": spec["hardware_generable"],
        "summary": spec["summary"],
    }


def _enforce_request_intent(
    parsed, prompt_package, top_k, workload_definition=None
):
    try:
        prompt_payload = json.loads(prompt_package["user_prompt"])
    except (KeyError, json.JSONDecodeError, TypeError):
        return parsed

    preferred_blocks = (
        prompt_payload.get("request_intent", {}).get("preferred_fpga_blocks", [])
    )
    if not preferred_blocks:
        return parsed

    preferred = preferred_blocks[0]
    try:
        preferred_item = _intent_recommendation(preferred, workload_definition)
    except SchemaError:
        return parsed

    parsed = dict(parsed) if isinstance(parsed, dict) else {}
    recommendations = [
        item for item in parsed.get("recommendations", []) if isinstance(item, dict)
    ]
    preferred_id = preferred_item["partition_id"]

    for item in recommendations:
        if item.get("partition_id") == preferred_id:
            preferred_item = item
            break

    reordered = [preferred_item] + [
        item for item in recommendations if item.get("partition_id") != preferred_id
    ]
    reordered = reordered[:top_k]
    dynamic_specs = dict(parsed.get("dynamic_partition_specs", {}))
    normalized = []
    for item in reordered:
        try:
            spec = build_partition_spec(
                partition_id=item["partition_id"],
                fpga_subfunctions=item.get("fpga_subfunctions", []),
                gpp_subfunctions=item.get("gpp_subfunctions"),
                summary=item.get("recommendation") or item.get("summary"),
                workload_definition=workload_definition,
            )
        except SchemaError:
            continue
        updated = dict(item)
        updated["rank"] = len(normalized) + 1
        updated["fpga_subfunctions"] = spec["fpga_subfunctions"]
        updated["gpp_subfunctions"] = spec["gpp_subfunctions"]
        updated["hardware_generable"] = spec["hardware_generable"]
        updated.setdefault("summary", spec["summary"])
        normalized.append(updated)
        dynamic_specs[spec["partition_id"]] = {
            "partition_id": spec["partition_id"],
            "fpga_subfunctions": spec["fpga_subfunctions"],
            "gpp_subfunctions": spec["gpp_subfunctions"],
            "summary": spec["summary"],
        }

    if not normalized:
        return parsed or None
    parsed["recommendations"] = normalized
    parsed["dynamic_partition_specs"] = dynamic_specs
    parsed.setdefault("baseline_comparison", "Workload measurements must be collected.")
    parsed["final_recommendation"] = normalized[0]["partition_id"]
    return parsed


def build_recommendation_text(parsed):
    if not isinstance(parsed, dict):
        return None
    lines = []
    for item in parsed.get("recommendations", []):
        lines.append(f"{item.get('rank')}. {item.get('partition_id')}: {item.get('recommendation')}")
    if parsed.get("final_recommendation"):
        lines.append("")
        lines.append(f"Final recommendation: {parsed['final_recommendation']}")
    return "\n".join(lines) if lines else None


class CNNImageProcAllocationAgent:
    """Build a grounded prompt, call the LLM, and save recommendation output."""

    def __init__(self, model="gpt-5.6-sol", max_tokens=8000, temperature=0.2, top_p=0.9):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def run(
        self,
        user_request,
        requirements=None,
        top_k=3,
        output_path=None,
        dry_run=False,
        source_dir=None,
        include_lightcnn_context=True,
        experiment_condition="GenTAS_RAG",
        repetition_index=1,
    ):
        prompt_package = build_prompt(
            user_request=user_request,
            requirements=requirements or {},
            top_k=top_k,
            source_dir=source_dir,
            include_lightcnn_context=include_lightcnn_context,
        )
        prompt_package["rag_metrics"] = dict(prompt_package.get("rag_metrics", {}))
        prompt_package["rag_metrics"].update({
            "enabled": bool(include_lightcnn_context),
            "condition": (
                "active_lightcnn_kb"
                if include_lightcnn_context
                else "disabled_no_rag_ablation"
            ),
        })
        if not include_lightcnn_context:
            prompt_package["rag_metrics"].update({
                "method": "disabled",
                "kb_version": None,
                "number_retrieved": 0,
                "retrieval_success": False,
                "retrieval_confidence": 0.0,
                "retrieved_task_ids": [],
                "similarity_scores": [],
            })
        if dry_run:
            content = None
            parsed = None
            token_count = 0
            elapsed = 0.0
            effective_request_parameters = None
        else:
            client = LLMClient(self.model)
            workload_definition = prompt_package.get("workload_definition")
            attempts = []
            usages = []
            elapsed = 0.0
            parsed = None
            content = None
            validation_metrics = {}
            request_prompt = prompt_package["user_prompt"]
            request_system = prompt_package["system_prompt"]
            for attempt_index in range(2):
                started = time.time()
                content, attempt_tokens = client.generate_content(
                    prompt=request_prompt,
                    system_prompt=request_system,
                    max_tokens=self.max_tokens if attempt_index == 0 else min(4000, self.max_tokens),
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                attempt_elapsed = time.time() - started
                elapsed += attempt_elapsed
                usage = dict(client.last_usage or {})
                usages.append(usage)
                attempt_validation = {}
                candidate = _filter_recommendations(
                    try_parse_json(content),
                    top_k,
                    workload_definition=workload_definition,
                    telemetry=attempt_validation,
                )
                attempts.append({
                    "attempt": attempt_index + 1,
                    "kind": "initial" if attempt_index == 0 else "compact_json_repair",
                    "response": content,
                    "token_count": attempt_tokens,
                    "token_usage": usage,
                    "generation_time_s": attempt_elapsed,
                    "validation": attempt_validation,
                })
                validation_metrics = attempt_validation
                if isinstance(candidate, dict) and candidate.get("recommendations"):
                    parsed = candidate
                    break
                request_prompt = _build_repair_prompt(content, prompt_package, top_k)
                request_system = (
                    "You repair FPGA/GPP allocation output. Return one compact valid JSON "
                    "object only. Never add Markdown fences or explanatory prose."
                )

            if isinstance(parsed, dict):
                parsed = _enforce_request_intent(
                    parsed,
                    prompt_package,
                    top_k,
                    workload_definition=workload_definition,
                )
            token_usage = _aggregate_token_usage(usages, self.model)
            token_count = int(token_usage.get("total_tokens") or 0)
            effective_request_parameters = usages[-1].get("request_parameters") if usages else None

            retrieved = prompt_package.get("retrieved_profiles", [])
            used_ids = []
            if isinstance(parsed, dict):
                for item in parsed.get("recommendations", []):
                    related = " ".join(item.get("related_prior_dataset_examples", [])).lower()
                    used = []
                    for profile in retrieved:
                        kb_task_id = profile.get("kb_task_id")
                        aliases = (
                            kb_task_id,
                            profile.get("design_name"),
                            profile.get("partition_id"),
                        )
                        if kb_task_id and any(
                            str(alias).lower() in related for alias in aliases if alias
                        ):
                            used.append(kb_task_id)
                    item["retrieved_kb_task_ids_used"] = used
                    item["decision_basis"] = (
                        "retrieved_knowledge_and_source_inference"
                        if used else "source_evidence_and_llm_inference"
                    )
                    evidence_count = len(item.get("source_code_evidence", []))
                    rationale_count = len(item.get("evidence_used", []))
                    item["allocation_confidence_score"] = round(
                        min(1.0, 0.45 + min(evidence_count, 3) * 0.1
                            + min(rationale_count, 2) * 0.075
                            + (0.1 if used else 0.0)),
                        3,
                    )
                    used_ids.extend(used)
            rag_metrics = dict(prompt_package.get("rag_metrics", {}))
            rag_metrics["retrieved_task_ids_actually_used"] = sorted(set(used_ids))
            rag_metrics["number_retrieved_examples_actually_used"] = len(set(used_ids))
            validation_metrics["regeneration_retry_count"] = max(0, len(attempts) - 1)
            validation_metrics["attempt_count"] = len(attempts)
            validation_metrics["finish_reasons"] = token_usage.get("finish_reasons", [])
            validation_metrics["truncated_attempts"] = token_usage.get("truncated_attempts", [])
            validation_metrics["hardware_objective_satisfaction_score"] = None
            validation_metrics["hardware_objective_note"] = (
                "Calculated after measured hardware objectives are available."
            )

        result = {
            "task": "FPGA/GPP allocation suggestion",
            "model": self.model,
            "dry_run": dry_run,
            "requirements": requirements or {},
            "user_request": user_request,
            "retrieved_profiles": prompt_package["retrieved_profiles"],
            "source_dir": str(source_dir) if source_dir else None,
            "system_prompt": prompt_package["system_prompt"],
            "user_prompt": prompt_package["user_prompt"],
            "llm_response": content,
            "llm_attempts": [] if dry_run else attempts,
            "parsed_recommendations": parsed,
            "recommendation_text": build_recommendation_text(parsed),
            "token_count": token_count,
            "token_usage": None if dry_run else token_usage,
            "llm_time_s": elapsed,
            "rag_metrics": prompt_package.get("rag_metrics", {}) if dry_run else rag_metrics,
            "llm_evaluation": {} if dry_run else validation_metrics,
            "experiment": {
                "condition": experiment_condition,
                "repetition_index": repetition_index,
                "configuration": {
                    "strategy_model": self.model,
                    "strategy_temperature": self.temperature,
                    "strategy_top_p": self.top_p,
                    "strategy_max_tokens": self.max_tokens,
                    "strategy_effective_request_parameters": effective_request_parameters,
                    "strategy_attempt_request_parameters": (
                        [] if dry_run else [usage.get("request_parameters") for usage in usages]
                    ),
                    "top_k": top_k,
                    "rag_enabled": include_lightcnn_context,
                    "prompt_sha256": hashlib.sha256(
                        prompt_package["user_prompt"].encode("utf-8")
                    ).hexdigest(),
                    "kb_version": prompt_package.get("rag_metrics", {}).get("kb_version"),
                },
            },
        }
        if output_path is None:
            output_path = make_output_path(DEFAULT_OUTPUT_DIR, self.model)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        result["output_path"] = str(output_path)
        return result
