#!/usr/bin/env python3
"""Source-described allocation/session/generation entry point.

CNNImageProc remains the legacy default workload. A source directory containing
workload.json can define another ordered stage graph and wrapper contract.
"""

import argparse
import json
import os
from pathlib import Path
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Evaluation.CNNImageProc.AI.schemas import PARTITION_SPECS, get_partition_spec
from Evaluation.CNNImageProc.AI.workload_definition import load_workload_definition
from Evaluation.CNNImageProc.AI.allocation_agent import CNNImageProcAllocationAgent
from Evaluation.CNNImageProc.AI.session_store import (
    DEFAULT_SESSIONS_DIR,
    SessionError,
    create_session,
    create_session_from_recommendation,
    load_session,
    store_selection,
)
from Evaluation.CNNImageProc.AI.top_generation_agent import (
    CNNImageProcTopGenerationAgent,
    TopGenerationError,
)
from Evaluation.CNNImageProc.AI.experiment_metrics import EXPERIMENT_CONDITIONS


CNN_DIR = Path(__file__).resolve().parent
ROOT_DIR = CNN_DIR.parents[1]
DEFAULT_SOURCE_DIR = ROOT_DIR / "Backend" / "examples" / "cnn_imageproc_fashion"


def load_root_env():
    """Load simple KEY=VALUE pairs from the repository root .env file."""
    env_path = ROOT_DIR / ".env"
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def print_partition_table(source_dir=DEFAULT_SOURCE_DIR):
    definition = load_workload_definition(source_dir)
    partitions = definition.get("partitions", {}) if definition else PARTITION_SPECS
    workload = definition["workload"] if definition else "CNNImageProc"
    print(f"{workload} partitions:")
    for partition_id in partitions:
        spec = get_partition_spec(partition_id, workload_definition=definition)
        print(
            f"- {partition_id}: FPGA={spec['fpga_subfunctions'] or ['none']}, "
            f"GPP={spec['gpp_subfunctions'] or ['none']}, "
            f"hardware_generable={spec['hardware_generable']}"
        )


def run_new_session(args):
    if args.deterministic:
        result = None
        session = create_session(
            source_dir=args.source_dir,
            sessions_root=args.sessions_root,
            run_id=args.run_id,
            request=args.request,
            goal=args.goal,
            top_k=args.top_k,
            experiment_condition=args.experiment_condition,
            repetition_index=args.repetition_index,
        )
    else:
        requirements = {"primary_goal": args.goal}
        agent = CNNImageProcAllocationAgent(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        result = agent.run(
            user_request=args.request,
            requirements=requirements,
            top_k=args.top_k,
            output_path=args.output,
            dry_run=args.dry_run,
            source_dir=args.source_dir,
            include_lightcnn_context=not args.no_rag,
            experiment_condition=args.experiment_condition,
            repetition_index=args.repetition_index,
        )
        session = create_session_from_recommendation(
            recommendation_result=result,
            source_dir=args.source_dir,
            sessions_root=args.sessions_root,
            run_id=args.run_id,
            top_k=args.top_k,
        )
    state = load_session(session["session_dir"])
    output = args.output if args.deterministic else None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(
            json.dumps(state["recommendations"], indent=2) + "\n",
            encoding="utf-8",
        )
    if result is not None:
        print(f"Output written to {result['output_path']}")
        if args.dry_run:
            print("Dry run completed; no LLM call was made.")
        else:
            print(f"LLM time: {result['llm_time_s']:.2f}s")
            print(f"Provider-reported total tokens: {result['token_count']}")
            if result.get("recommendation_text"):
                print("")
                print(result["recommendation_text"])
        print("")
    print(f"Selection session: {session['session_dir']}")
    print("Selectable partitions: " + ", ".join(session["selectable_partitions"]))
    print("")
    for item in state["recommendations"]["recommendations"]:
        print(
            f"{item['rank']}. {item['partition_id']} "
            f"FPGA={item['fpga_subfunctions'] or ['none']} "
            f"GPP={item['gpp_subfunctions'] or ['none']}"
        )
        print(f"   {item['summary']}")
    return 0


def run_session_action(args):
    if args.select:
        selection = store_selection(args.session, args.select, overwrite=args.force)
        print(
            f"Selected {selection['selected_partition']} "
            f"(rank {selection['selected_rank']})"
        )

    state = load_session(args.session)
    if not args.generate_top:
        print(f"Session: {state['session_dir']}")
        print("Selectable partitions:")
        for recommendation in state["recommendations"].get("recommendations", []):
            print(f"- {recommendation.get('rank')}. {recommendation.get('partition_id')}")
        if state["selection"]:
            print(f"Current selection: {state['selection']['selected_partition']}")
        else:
            print("Current selection: none")
        return 0

    agent = CNNImageProcTopGenerationAgent(
        model=args.top_model,
        max_tokens=args.top_max_tokens,
        temperature=args.top_temperature,
        top_p=args.top_p,
    )
    result = agent.run(
        session_dir=args.session,
        output_dir=args.generation_output,
        force=args.force,
        mode=args.top_mode,
        dry_run=args.generation_dry_run,
    )
    if result.get("dry_run"):
        print(f"Top-generation dry run: {result['prompt_path']}")
        return 0
    print(f"Generated top.cpp: {result['top_path']}")
    print(f"Generated tb.cpp:  {result['testbench_path']}")
    print(f"Backend handoff:   {result['backend_handoff_path']}")
    print(f"Pseudocode:        {result['pseudocode_path']}")
    print(f"I/O mapping:       {result['io_mapping_path']}")
    print(f"Manifest:          {result['manifest_path']}")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the source-described allocation pipeline.")
    parser.add_argument("--request", default="", help="Inline user request text")
    parser.add_argument("--goal", default="latency", help="Primary goal label")
    parser.add_argument("--top-k", type=int, default=3, help="Number of strategies to save")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--sessions-root", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", default=None, help="Optional recommendations JSON copy")
    parser.add_argument("--list-partitions", action="store_true")
    parser.add_argument(
        "--model",
        default=os.getenv(
            "GENTAS_STRATEGY_MODEL",
            os.getenv("LAMDA_STRATEGY_MODEL", "gpt-5.6-sol"),
        ),
        help="LLM model for strategy selection",
    )
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Exclude the active LightCNN KB from strategy selection",
    )
    parser.add_argument(
        "--experiment-condition",
        choices=EXPERIMENT_CONDITIONS,
        default=None,
        help="Publication experiment condition; inferred from --no-rag/--deterministic when omitted",
    )
    parser.add_argument("--repetition-index", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Build allocation prompt only; do not call LLM")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use curated predefined ranking instead of LLM strategy selection",
    )

    parser.add_argument("--session", type=Path, default=None)
    parser.add_argument("--select", default=None)
    parser.add_argument("--generate-top", action="store_true")
    parser.add_argument("--generation-output", default=None)
    parser.add_argument("--llm-top", action="store_true", help="Use LLM top.cpp generation")
    parser.add_argument(
        "--top-mode",
        choices=("auto", "llm", "deterministic"),
        default=os.getenv("GENTAS_TOP_MODE", os.getenv("LAMDA_TOP_MODE", "auto")),
        help="top.cpp generation mode; auto validates the LLM result and falls back safely",
    )
    parser.add_argument("--generation-dry-run", action="store_true", help="Write top prompt only")
    parser.add_argument(
        "--top-model",
        default=os.getenv("GENTAS_TOP_MODEL", os.getenv("LAMDA_TOP_MODEL", "gpt-5.6-sol")),
    )
    parser.add_argument("--top-max-tokens", type=int, default=6000)
    parser.add_argument("--top-temperature", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.repetition_index <= 0:
        parser.error("--repetition-index must be positive")
    inferred_condition = (
        "Deterministic_Heuristic"
        if args.deterministic
        else "LLM_NoRAG"
        if args.no_rag
        else "GenTAS_RAG"
    )
    if args.experiment_condition is None:
        args.experiment_condition = inferred_condition
    if args.experiment_condition == "LLM_NoRAG":
        if args.deterministic:
            parser.error("LLM_NoRAG cannot be combined with --deterministic")
        args.no_rag = True
    if args.experiment_condition in {"Deterministic_Heuristic", "Measured_Oracle"}:
        args.deterministic = True
        args.no_rag = True
    if args.experiment_condition == "GenTAS_RAG" and (args.no_rag or args.deterministic):
        parser.error("GenTAS_RAG cannot be combined with --no-rag or --deterministic")
    if args.list_partitions:
        return args
    if args.generation_dry_run:
        args.generate_top = True
    if args.llm_top:
        args.top_mode = "llm"
    if (args.select or args.generate_top) and not args.session:
        parser.error("--select/--generate-top require --session")
    if args.session and args.source_dir != DEFAULT_SOURCE_DIR:
        parser.error("--source-dir creates a new session and cannot be combined with --session")
    return args


def main(argv=None):
    try:
        load_root_env()
        args = parse_args(argv)
        if args.list_partitions:
            print_partition_table(args.source_dir)
            return 0
        if args.session:
            return run_session_action(args)
        return run_new_session(args)
    except (SessionError, TopGenerationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
