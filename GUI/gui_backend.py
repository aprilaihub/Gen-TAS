"""Thin Streamlit adapter for the Gen-TAS flow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.CNNImageProc.AI.workload_contract import build_workload_contract

CNN_RUN = ROOT / "Evaluation" / "CNNImageProc" / "Run.py"
BACKEND_RUN = ROOT / "Backend" / "run_backend.py"
SESSIONS_ROOT = ROOT / "Evaluation" / "CNNImageProc" / "Sessions"
DEFAULT_SOURCE_DIR = ROOT / "Backend" / "examples" / "cnn_imageproc_fashion"
EXPORT_ROOT = ROOT / "Backend" / "artifacts" / "hardware_exports"
PRE_SESSION_ROOT = Path(__file__).resolve().parent / "runs"
UI_ONLY_FILES = {"ui_status.json", "ui.log"}


class GuiBackendError(RuntimeError):
    """Raised when a GUI-triggered backend stage fails."""


@dataclass(frozen=True)
class CommandResult:
    stage: str
    command: list[str]
    returncode: int
    output: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return f"cnn-gui-{timestamp}-{uuid4().hex[:8]}"


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def session_dir(run_id: str) -> Path:
    return SESSIONS_ROOT / run_id


def pre_session_dir(run_id: str) -> Path:
    return PRE_SESSION_ROOT / run_id


def is_real_session(run_id: str) -> bool:
    path = session_dir(run_id)
    return (path / "recommendations.json").is_file() and (path / "source_manifest.json").is_file()


def ui_dir(run_id: str) -> Path:
    return session_dir(run_id) if is_real_session(run_id) else pre_session_dir(run_id)


def ui_status_path(run_id: str) -> Path:
    return ui_dir(run_id) / "ui_status.json"


def ui_log_path(run_id: str) -> Path:
    return ui_dir(run_id) / "ui.log"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def append_log(run_id: str, message: str) -> None:
    path = ui_log_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{now_utc()}] {message}\n")


def append_terminal_output(run_id: str, output: str) -> None:
    path = ui_log_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(output)
        if output and not output.endswith("\n"):
            fh.write("\n")


def read_log(run_id: str, max_chars: int = 12000) -> str:
    paths = [pre_session_dir(run_id) / "ui.log", session_dir(run_id) / "ui.log"]
    chunks = [path.read_text(encoding="utf-8", errors="replace") for path in paths if path.is_file()]
    if not chunks:
        return "No UI log yet."
    text = "\n".join(chunks)
    return text[-max_chars:]


def load_status(run_id: str) -> dict[str, Any]:
    path = ui_status_path(run_id)
    if path.is_file():
        return read_json(path)
    return {
        "run_id": run_id,
        "status": "not_started",
        "source_dir": rel(DEFAULT_SOURCE_DIR),
        "session_dir": rel(session_dir(run_id)),
        "selected_partition": None,
        "manifest_path": None,
        "export_dir": None,
        "bitstream": None,
        "hwh": None,
        "pynq_script": None,
        "experiment_metrics": None,
        "current_stage": None,
        "stage_message": "Waiting for strategy generation",
        "status_message": "No run started yet. Enter a request and click Generate Strategies.",
        "last_error": None,
        "updated_at": now_utc(),
    }


def save_status(run_id: str, **updates: Any) -> dict[str, Any]:
    status = load_status(run_id)
    status.update(updates)
    status["updated_at"] = now_utc()
    write_json(ui_status_path(run_id), status)
    return status


def list_run_ids() -> list[str]:
    if not SESSIONS_ROOT.is_dir():
        return []
    return sorted(
        [path.name for path in SESSIONS_ROOT.iterdir() if path.is_dir() and is_real_session(path.name)],
        reverse=True,
    )


def migrate_ui_only_session_dir(run_id: str) -> None:
    """Move UI-only files out of Sessions so Run.py can create the real session."""
    path = session_dir(run_id)
    if not path.is_dir() or is_real_session(run_id):
        return
    files = [item for item in path.iterdir() if item.is_file()]
    dirs = [item for item in path.iterdir() if item.is_dir()]
    if dirs or any(item.name not in UI_ONLY_FILES for item in files):
        raise GuiBackendError(f"session already exists: {path}")

    target = pre_session_dir(run_id)
    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        destination = target / item.name
        if destination.exists():
            destination.write_text(
                destination.read_text(encoding="utf-8", errors="replace")
                + item.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )
            item.unlink()
        else:
            item.replace(destination)
    path.rmdir()


def _run(
    run_id: str,
    stage: str,
    command: list[str],
    on_output: Callable[[str], None] | None = None,
) -> CommandResult:
    append_log(run_id, f"START {stage}: {' '.join(command)}")
    running_messages = {
        "strategy": (
            "Generating strategy recommendations",
            "Running LLM strategy generation. Waiting for strategy recommendations.",
        ),
        "top_generation": (
            "Generating design files",
            "Strategy selected. Generating top/testbench and design setup files.",
        ),
        "hardware_export": (
            "Building and exporting hardware",
            "Generated design files are ready. Running hardware build and export.",
        ),
        "pynq": (
            "Generating PYNQ measurement script",
            "Hardware export is ready. Generating the PYNQ measurement script.",
        ),
    }
    stage_message, status_message = running_messages.get(
        stage,
        (stage, f"Running {stage}."),
    )
    save_status(
        run_id,
        status="running",
        current_stage=stage,
        stage_message=stage_message,
        status_message=status_message,
        last_error=None,
    )
    if on_output:
        on_output(read_log(run_id))

    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        bufsize=1,
    )
    output_parts: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        output_parts.append(line)
        append_terminal_output(run_id, line)
        if on_output:
            on_output(read_log(run_id))

    returncode = proc.wait()
    output = "".join(output_parts)
    result = CommandResult(stage, command, returncode, output)
    if returncode != 0:
        message = f"{stage} failed with exit code {returncode}"
        append_log(run_id, message)
        save_status(
            run_id,
            status="failed",
            current_stage=stage,
            stage_message=f"{stage_message} failed",
            status_message=f"{stage_message} failed. Check the live log for details.",
            last_error=message,
        )
        if on_output:
            on_output(read_log(run_id))
        raise GuiBackendError(message)
    append_log(run_id, f"DONE {stage}")
    if on_output:
        on_output(read_log(run_id))
    return result


def recommendations_path(run_id: str) -> Path:
    return session_dir(run_id) / "recommendations.json"


def selection_path(run_id: str) -> Path:
    return session_dir(run_id) / "selection.json"


def source_manifest_path(run_id: str) -> Path:
    return session_dir(run_id) / "source_manifest.json"


def load_recommendations(run_id: str) -> dict[str, Any] | None:
    path = recommendations_path(run_id)
    if not path.is_file():
        return None
    return read_json(path)


def load_selection(run_id: str) -> dict[str, Any] | None:
    path = selection_path(run_id)
    if not path.is_file():
        return None
    return read_json(path)


def source_dir_for_run(run_id: str) -> Path:
    path = source_manifest_path(run_id)
    if path.is_file():
        manifest = read_json(path)
        return Path(manifest["source_dir"])
    return DEFAULT_SOURCE_DIR


def workload_contract_for_source(source_dir: Path | str) -> dict[str, Any]:
    return build_workload_contract(Path(source_dir).expanduser().resolve())


def workload_contract_for_run(run_id: str, fallback_source_dir: Path | str = DEFAULT_SOURCE_DIR) -> dict[str, Any]:
    path = source_manifest_path(run_id)
    if path.is_file():
        manifest = read_json(path)
        contract = manifest.get("workload_contract")
        if isinstance(contract, dict):
            return contract
    return workload_contract_for_source(fallback_source_dir)


def workload_mapping_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stage_id in contract.get("call_order", contract.get("stages", {}).keys()):
        stage = contract["stages"][stage_id]
        rows.append(
            {
                "Stage": stage_id,
                "Function": stage.get("function"),
                "Role": stage.get("role"),
                "Block": stage.get("block"),
                "Input": stage.get("input", {}).get("semantic"),
                "Output": stage.get("output", {}).get("semantic"),
                "Confidence": f"{stage.get('confidence', 0):.2f}",
            }
        )
    return rows


def manifest_path_for(run_id: str, partition_id: str) -> Path:
    return source_dir_for_run(run_id) / "generated" / run_id / partition_id / "generation_manifest.json"


def load_manifest(path: Path | str | None) -> dict[str, Any] | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return read_json(path)


def load_backend_handoff(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if not manifest:
        return None
    handoff = manifest.get("artifacts", {}).get("backend_handoff")
    if not handoff:
        return None
    path = Path(handoff)
    if not path.is_file():
        return None
    return read_json(path)


def _parse_float(value: str) -> float | None:
    value = value.strip().replace(",", "")
    if value.startswith("<"):
        value = value[1:]
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None else None


def _split_report_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def parse_utilization_report(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "| design_1_wrapper" not in line:
            continue
        parts = _split_report_row(line)
        if len(parts) < 10:
            continue
        lut = _parse_int(parts[2])
        ff = _parse_int(parts[6])
        ramb36 = _parse_int(parts[7]) or 0
        ramb18 = _parse_int(parts[8]) or 0
        dsp = _parse_int(parts[10]) if len(parts) > 10 else None
        bram = ramb36 + (ramb18 / 2.0)
        return {
            "LUT": f"{lut:,}" if lut is not None else "n/a",
            "FF": f"{ff:,}" if ff is not None else "n/a",
            "DSP": f"{dsp:,}" if dsp is not None else "n/a",
            "BRAM": f"{bram:g}",
        }
    return {}


def parse_power_report(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    total = None
    gpp = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Total On-Chip Power (W)" in line:
            parts = _split_report_row(line)
            if len(parts) >= 2:
                total = _parse_float(parts[1])
        elif re.search(r"\|\s*PS8\s*\|", line):
            parts = _split_report_row(line)
            if len(parts) >= 2:
                gpp = _parse_float(parts[1])
    metrics: dict[str, str] = {}
    if total is not None:
        metrics["Power_Total"] = f"{total:.3f} W"
    if gpp is not None:
        metrics["Power_GPP"] = f"{gpp:.3f} W"
    if total is not None and gpp is not None:
        metrics["Power_FPGA"] = f"{max(total - gpp, 0):.3f} W"
    return metrics


def parse_timing_report(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    delays = []
    slacks = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not re.search(r"\|\s*Path #\d+", line):
            continue
        parts = _split_report_row(line)
        if len(parts) < 7:
            continue
        delay = _parse_float(parts[2])
        slack = _parse_float(parts[6])
        if delay is not None:
            delays.append(delay)
        if slack is not None:
            slacks.append(slack)
    metrics: dict[str, str] = {}
    if delays:
        worst_delay = max(delays)
        metrics["Fmax"] = f"{1000.0 / worst_delay:.1f} MHz"
    if slacks:
        metrics["Slack"] = f"{min(slacks):.3f} ns"
    return metrics


def report_metrics(export_dir: Path) -> dict[str, str]:
    reports_dir = export_dir / "reports"
    metrics: dict[str, str] = {}
    metrics.update(parse_timing_report(reports_dir / "timing_paths.csv"))
    metrics.update(parse_utilization_report(reports_dir / "module_utilization.csv"))
    metrics.update(parse_power_report(reports_dir / "power_report.csv"))
    return metrics


def downloadable_artifacts(export_dir: Path, design_name: str) -> list[dict[str, str]]:
    candidates = [
        ("Bitstream", export_dir / f"{design_name}.bit"),
        ("Hardware handoff", export_dir / f"{design_name}.hwh"),
        ("PYNQ measurement script", export_dir / f"measure_{design_name}.py"),
        ("Experiment metrics", export_dir / "experiment_metrics.json"),
        ("Vivado metrics summary", export_dir / "vivado_report_summary.json"),
        ("Weights header", export_dir / "weights.hpp"),
        ("FashionMNIST data", export_dir / "FashionMNIST_data.zip"),
    ]
    return [
        {"label": label, "path": str(path), "filename": path.name}
        for label, path in candidates
        if path.is_file()
    ]


def export_summary_from_manifest(
    manifest_path: Path | str | None,
    export_root: Path | str = EXPORT_ROOT,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    handoff = load_backend_handoff(manifest)
    if not handoff:
        return {}

    design_name = handoff["design_name"]
    export_dir = Path(export_root).expanduser().resolve() / design_name
    bitstream = export_dir / f"{design_name}.bit"
    hwh = export_dir / f"{design_name}.hwh"
    pynq_script = export_dir / f"measure_{design_name}.py"
    reports_dir = export_dir / "reports"
    experiment_metrics = export_dir / "experiment_metrics.json"
    report_files = sorted(path.name for path in reports_dir.glob("*")) if reports_dir.is_dir() else []
    return {
        "design_name": design_name,
        "export_dir": str(export_dir),
        "bitstream": str(bitstream) if bitstream.is_file() else None,
        "hwh": str(hwh) if hwh.is_file() else None,
        "pynq_script": str(pynq_script) if pynq_script.is_file() else None,
        "experiment_metrics": (
            str(experiment_metrics) if experiment_metrics.is_file() else None
        ),
        "report_files": report_files,
        "metrics": report_metrics(export_dir),
        "artifacts": downloadable_artifacts(export_dir, design_name),
    }


def refresh_status_from_files(
    run_id: str,
    export_root: Path | str = EXPORT_ROOT,
) -> dict[str, Any]:
    selection = load_selection(run_id)
    selected_partition = selection.get("selected_partition") if selection else None
    manifest_path = None
    if selected_partition:
        candidate = manifest_path_for(run_id, selected_partition)
        if candidate.is_file():
            manifest_path = str(candidate)
    export_summary = export_summary_from_manifest(manifest_path, export_root=export_root)
    return save_status(
        run_id,
        selected_partition=selected_partition,
        manifest_path=manifest_path,
        export_dir=export_summary.get("export_dir"),
        bitstream=export_summary.get("bitstream"),
        hwh=export_summary.get("hwh"),
        pynq_script=export_summary.get("pynq_script"),
        experiment_metrics=export_summary.get("experiment_metrics"),
    )


def generate_strategies(
    *,
    run_id: str,
    request: str,
    goal: str,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    top_k: int = 3,
    model: str = os.getenv(
        "GENTAS_STRATEGY_MODEL",
        os.getenv("LAMDA_STRATEGY_MODEL", "gpt-5.6-sol"),
    ),
    deterministic: bool = False,
    experiment_condition: str = "GenTAS_RAG",
    repetition_index: int = 1,
    on_output: Callable[[str], None] | None = None,
) -> CommandResult:
    command = [
        sys.executable,
        str(CNN_RUN),
        "--request",
        request,
        "--goal",
        goal,
        "--source-dir",
        str(source_dir),
        "--run-id",
        run_id,
        "--top-k",
        str(top_k),
        "--model",
        model,
        "--experiment-condition",
        experiment_condition,
        "--repetition-index",
        str(repetition_index),
    ]
    if experiment_condition == "LLM_NoRAG":
        command.append("--no-rag")
    if deterministic or experiment_condition in {
        "Deterministic_Heuristic", "Measured_Oracle"
    }:
        command.append("--deterministic")
    migrate_ui_only_session_dir(run_id)
    result = _run(run_id, "strategy", command, on_output=on_output)
    save_status(
        run_id,
        status="awaiting_selection",
        current_stage="strategy",
        stage_message="Strategy generation complete",
        status_message="Generated strategy recommendations. Waiting for a strategy selection.",
        source_dir=rel(source_dir),
        session_dir=rel(session_dir(run_id)),
        experiment_condition=experiment_condition,
        repetition_index=repetition_index,
    )
    return result


def select_strategy_and_generate_top(
    *,
    run_id: str,
    partition_id: str,
    llm_top: bool | None = None,
    top_mode: str = "deterministic",
    top_model: str | None = None,
    force: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> CommandResult:
    command = [
        sys.executable,
        str(CNN_RUN),
        "--session",
        str(session_dir(run_id)),
        "--select",
        partition_id,
        "--generate-top",
    ]
    if llm_top is True:
        command.append("--llm-top")
    else:
        command.extend(("--top-mode", top_mode))
    if top_model:
        command.extend(("--top-model", top_model))
    if force:
        command.append("--force")
    result = _run(run_id, "top_generation", command, on_output=on_output)
    manifest = manifest_path_for(run_id, partition_id)
    save_status(
        run_id,
        status="top_generated",
        current_stage="top_generation",
        stage_message="Design file generation complete",
        status_message="Generated top/testbench and design setup files. Waiting for Full Hardware Build and Export.",
        selected_partition=partition_id,
        manifest_path=str(manifest) if manifest.is_file() else None,
    )
    refresh_status_from_files(run_id)
    return result


def run_hardware_export(
    *,
    run_id: str,
    manifest_path: Path | str,
    export_root: Path | str = EXPORT_ROOT,
    on_output: Callable[[str], None] | None = None,
) -> CommandResult:
    command = [
        sys.executable,
        str(BACKEND_RUN),
        "--generation-manifest",
        str(manifest_path),
        "--stop-after",
        "export",
        "--export-root",
        str(export_root),
    ]
    result = _run(run_id, "hardware_export", command, on_output=on_output)
    save_status(
        run_id,
        status="export_complete",
        current_stage="hardware_export",
        stage_message="Hardware build and export complete",
        status_message="Hardware build/export completed. Waiting for Generate PYNQ Measurement Script.",
        export_root=str(export_root),
    )
    refresh_status_from_files(run_id, export_root=export_root)
    return result


def generate_pynq(
    *,
    run_id: str,
    manifest_path: Path | str,
    export_root: Path | str = EXPORT_ROOT,
    generation_mode: str = "deterministic",
    model: str | None = None,
    force: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> CommandResult:
    command = [
        sys.executable,
        str(BACKEND_RUN),
        "--generation-manifest",
        str(manifest_path),
        "--start-at",
        "pynq",
        "--export-root",
        str(export_root),
        "--pynq-generation-mode",
        generation_mode,
    ]
    if model:
        command.extend(("--model", model))
    if force:
        command.append("--force")
    result = _run(run_id, "pynq", command, on_output=on_output)
    save_status(
        run_id,
        status="pynq_complete",
        current_stage="pynq",
        stage_message="PYNQ measurement script complete",
        status_message="PYNQ measurement script generated. Hardware test files are ready to download.",
        export_root=str(export_root),
    )
    refresh_status_from_files(run_id, export_root=export_root)
    return result
