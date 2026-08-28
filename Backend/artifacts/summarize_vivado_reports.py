#!/usr/bin/env python3
"""Summarize Vivado report_utilization/report_power/report_design_analysis output.

Vivado writes these files with a `.csv` extension in this flow, but the content
is fixed-width report text with pipe-delimited tables.  This script extracts a
compact, UI-friendly JSON document from:

* module_utilization.csv
* power_report.csv
* timing_paths.csv

The JSON keeps both headline metrics and the most useful table
rows so a UI can render cards, charts, or a detailed expandable view without
needing to parse Vivado's report text.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_NAMES = {
    "utilization": "module_utilization.csv",
    "power": "power_report.csv",
    "timing": "timing_paths.csv",
}

KEY_VALUE_DEFINITIONS = {
    "design": "Vivado top-level design name used for implementation.",
    "device": "Target FPGA/SoC part used by Vivado.",
    "design_state": "Implementation state of the reports, normally Routed for final bitstream reports.",
    "timing_met": "True when the worst reported setup slack is non-negative.",
    "fmax_mhz": "Estimated maximum clock frequency from the worst setup slack: 1000 / (target_period_ns - worst_slack_ns).",
    "target_period_ns": "Requested clock period constraint in nanoseconds.",
    "worst_slack_ns": "Worst setup slack in nanoseconds; positive means timing margin remains.",
    "lut": "Total LUT count reported at the implemented top level, including logic and LUTRAM/SRL use.",
    "logic_lut": "LUTs used as combinational/sequential logic, excluding LUTRAM and SRL usage.",
    "ff": "Flip-flop count reported at the implemented top level.",
    "dsp": "DSP block count used by the implemented design.",
    "bram18_equiv": "Block RAM use converted to RAMB18 equivalents: 2 * RAMB36 + RAMB18.",
    "ramb36": "Number of 36 Kb block RAM primitives used.",
    "ramb18": "Number of 18 Kb block RAM primitives used.",
    "uram": "UltraRAM block count used.",
    "total_power_w": "Total estimated on-chip power in watts from Vivado power analysis.",
    "dynamic_power_w": "Estimated switching/activity-dependent power in watts.",
    "static_power_w": "Estimated device static/leakage power in watts.",
    "ps_power_w": "Estimated Processing System power in watts, from the PS8 row.",
    "pl_power_w": "Estimated Programmable Logic plus non-PS on-chip power in watts, computed as total_power_w - ps_power_w.",
    "ps_static_power_w": "Estimated static/leakage power attributed to the Processing System.",
    "pl_static_power_w": "Estimated static/leakage power attributed to the Programmable Logic.",
    "power_confidence": "Vivado's confidence level for the power estimate; Medium usually means vectorless or incomplete activity data.",
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def clean_cell(value: str) -> str:
    return " ".join(value.strip().split())


def snake_case(value: str) -> str:
    value = value.strip().replace("#", "number")
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "value"


def parse_number(value: str) -> Any:
    text = clean_cell(value)

    if text in {"", "---", "NA", "Unspecified", "Unspecified*"}:
        return text

    less_than = text.startswith("<")
    numeric = text[1:] if less_than else text
    numeric = numeric.replace(",", "")

    try:
        parsed: Any
        if re.fullmatch(r"[-+]?\d+", numeric):
            parsed = int(numeric)
        else:
            parsed = float(numeric)
    except ValueError:
        return text

    if less_than:
        return {"lt": parsed}

    return parsed


def parse_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped.startswith("|") or ":" not in stripped:
            continue

        content = stripped.strip("|").strip()
        key, value = content.split(":", 1)
        key = snake_case(key)
        metadata[key] = clean_cell(value)

    return metadata


def parse_pipe_tables(text: str) -> list[list[dict[str, Any]]]:
    tables: list[list[dict[str, Any]]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current

        rows = parse_pipe_table_block(current)
        if rows:
            tables.append(rows)

        current = []

    for line in text.splitlines():
        stripped = line.rstrip()

        if stripped.startswith("|") or stripped.startswith("+"):
            current.append(stripped)
        else:
            if current:
                flush()

    if current:
        flush()

    return tables


def parse_pipe_table_block(lines: list[str]) -> list[dict[str, Any]]:
    data_lines = [line for line in lines if line.startswith("|")]

    if len(data_lines) < 2:
        return []

    header = [snake_case(cell) for cell in data_lines[0].strip("|").split("|")]
    rows: list[dict[str, Any]] = []

    for line in data_lines[1:]:
        cells = [clean_cell(cell) for cell in line.strip("|").split("|")]

        if len(cells) != len(header):
            continue

        row = {
            header[index]: parse_number(cells[index])
            for index in range(len(header))
        }
        rows.append(row)

    return rows


def first_table_with_columns(
    tables: list[list[dict[str, Any]]],
    required_columns: set[str],
) -> list[dict[str, Any]]:
    for table in tables:
        if not table:
            continue

        if required_columns.issubset(table[0].keys()):
            return table

    return []


def first_row(table: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in table:
        if str(row.get(key, "")).strip() == value:
            return row

    return {}


def value_from_row(
    rows: list[dict[str, Any]],
    match_key: str,
    match_value: str,
    value_key: str,
) -> Any:
    row = first_row(rows, match_key, match_value)
    return row.get(value_key) if row else None


def rounded(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        return round(value, digits)

    return value


def parse_key_value_section(
    text: str,
    start_heading: str,
    stop_heading: str | None = None,
) -> dict[str, Any]:
    """Parse a simple two-column Vivado table as key/value pairs."""

    in_section = False
    values: dict[str, Any] = {}

    for line in text.splitlines():
        stripped = line.strip()

        if stripped == start_heading:
            in_section = True
            continue

        if in_section and stop_heading and stripped == stop_heading:
            break

        if not in_section or not stripped.startswith("|"):
            continue

        cells = [clean_cell(cell) for cell in stripped.strip("|").split("|")]

        if len(cells) != 2:
            continue

        key, value = cells

        if not key or key.startswith("-"):
            continue

        values[snake_case(key)] = parse_number(value.rstrip("*"))

    return values


def parse_known_key_values(text: str, allowed_keys: set[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped.startswith("|"):
            continue

        cells = [clean_cell(cell) for cell in stripped.strip("|").split("|")]

        if len(cells) != 2:
            continue

        key, value = cells
        normalized = snake_case(key)

        if normalized in allowed_keys:
            values[normalized] = parse_number(value.rstrip("*"))

    return values


def parse_utilization(path: Path) -> dict[str, Any]:
    text = read_text(path)
    tables = parse_pipe_tables(text)
    hierarchy = first_table_with_columns(
        tables,
        {"instance", "module", "total_luts", "ffs", "ramb36", "ramb18", "dsp_blocks"},
    )
    top = hierarchy[0] if hierarchy else {}

    return {
        "source": str(path),
        "metadata": parse_metadata(text),
        "top": top,
        "hierarchy": hierarchy,
    }


def parse_power(path: Path) -> dict[str, Any]:
    text = read_text(path)
    tables = parse_pipe_tables(text)

    summary = parse_known_key_values(
        text,
        {
            "total_on_chip_power",
            "design_power_budget",
            "power_budget_margin",
            "dynamic",
            "device_static",
            "effective_tja",
            "max_ambient",
            "junction_temperature",
            "confidence_level",
            "setting_file",
            "simulation_activity_file",
            "design_nets_matched",
        },
    )

    components = first_table_with_columns(
        tables,
        {"on_chip", "power", "used", "available", "utilization"},
    )
    hierarchy = first_table_with_columns(
        tables,
        {"name", "power", "clock_number", "signal_number", "logic_number"},
    )

    return {
        "source": str(path),
        "metadata": parse_metadata(text),
        "summary": summary,
        "on_chip_components": components,
        "hierarchy": hierarchy,
    }


def parse_timing(path: Path) -> dict[str, Any]:
    text = read_text(path)
    tables = parse_pipe_tables(text)
    paths = first_table_with_columns(
        tables,
        {"paths", "requirement", "path_delay", "logic_delay", "net_delay", "slack"},
    )

    worst_path = paths[0] if paths else {}

    return {
        "source": str(path),
        "metadata": parse_metadata(text),
        "worst_path": worst_path,
        "paths": paths,
        "timing_met": bool(worst_path and parse_number(str(worst_path.get("slack", ""))) >= 0),
    }


def build_headline(summary: dict[str, Any]) -> dict[str, Any]:
    util_top = summary["reports"]["utilization"].get("top", {})
    power_summary = summary["reports"]["power"].get("summary", {})
    timing_worst = summary["reports"]["timing"].get("worst_path", {})

    return {
        "design": (
            summary["reports"]["utilization"].get("metadata", {}).get("design")
            or summary["reports"]["power"].get("metadata", {}).get("design")
            or summary["reports"]["timing"].get("metadata", {}).get("design")
        ),
        "device": (
            summary["reports"]["utilization"].get("metadata", {}).get("device")
            or summary["reports"]["power"].get("metadata", {}).get("device")
            or summary["reports"]["timing"].get("metadata", {}).get("device")
        ),
        "design_state": (
            summary["reports"]["utilization"].get("metadata", {}).get("design_state")
            or summary["reports"]["power"].get("metadata", {}).get("design_state")
            or summary["reports"]["timing"].get("metadata", {}).get("design_state")
        ),
        "timing_met": summary["reports"]["timing"].get("timing_met"),
        "worst_slack_ns": timing_worst.get("slack"),
        "worst_path_delay_ns": timing_worst.get("path_delay"),
        "target_period_ns": timing_worst.get("requirement"),
        "total_on_chip_power_w": power_summary.get("total_on_chip_power"),
        "dynamic_power_w": power_summary.get("dynamic"),
        "static_power_w": power_summary.get("device_static"),
        "confidence_level": power_summary.get("confidence_level"),
        "total_luts": util_top.get("total_luts"),
        "logic_luts": util_top.get("logic_luts"),
        "lutram": util_top.get("lutrams"),
        "srls": util_top.get("srls"),
        "ffs": util_top.get("ffs"),
        "ramb36": util_top.get("ramb36"),
        "ramb18": util_top.get("ramb18"),
        "uram": util_top.get("uram"),
        "dsp_blocks": util_top.get("dsp_blocks"),
    }


def build_key_values(summary: dict[str, Any]) -> dict[str, Any]:
    """Build the deliberately tiny payload intended for UI cards."""

    headline = summary["headline"]
    power_components = summary["reports"]["power"].get("on_chip_components", [])

    target_period_ns = headline.get("target_period_ns")
    worst_slack_ns = headline.get("worst_slack_ns")
    fmax_mhz = None

    if isinstance(target_period_ns, (int, float)) and isinstance(worst_slack_ns, (int, float)):
        effective_period_ns = target_period_ns - worst_slack_ns

        if effective_period_ns > 0:
            fmax_mhz = 1000.0 / effective_period_ns

    total_power_w = headline.get("total_on_chip_power_w")
    ps_power_w = value_from_row(power_components, "on_chip", "PS8", "power")
    pl_power_w = None

    if isinstance(total_power_w, (int, float)) and isinstance(ps_power_w, (int, float)):
        pl_power_w = total_power_w - ps_power_w

    ramb36 = headline.get("ramb36")
    ramb18 = headline.get("ramb18")
    bram18_equiv = None

    if isinstance(ramb36, int) and isinstance(ramb18, int):
        bram18_equiv = (2 * ramb36) + ramb18

    return {
        "design": headline.get("design"),
        "device": headline.get("device"),
        "design_state": headline.get("design_state"),
        "timing_met": headline.get("timing_met"),
        "fmax_mhz": rounded(fmax_mhz),
        "target_period_ns": target_period_ns,
        "worst_slack_ns": worst_slack_ns,
        "lut": headline.get("total_luts"),
        "logic_lut": headline.get("logic_luts"),
        "ff": headline.get("ffs"),
        "dsp": headline.get("dsp_blocks"),
        "bram18_equiv": bram18_equiv,
        "ramb36": ramb36,
        "ramb18": ramb18,
        "uram": headline.get("uram"),
        "total_power_w": total_power_w,
        "dynamic_power_w": headline.get("dynamic_power_w"),
        "static_power_w": headline.get("static_power_w"),
        "ps_power_w": rounded(ps_power_w),
        "pl_power_w": rounded(pl_power_w),
        "ps_static_power_w": value_from_row(power_components, "on_chip", "PS Static", "power"),
        "pl_static_power_w": value_from_row(power_components, "on_chip", "PL Static", "power"),
        "power_confidence": headline.get("confidence_level"),
    }


def summarize_reports(report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    paths = {name: report_dir / filename for name, filename in REPORT_NAMES.items()}

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report file(s): " + ", ".join(missing))

    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_dir": str(report_dir),
        "reports": {
            "utilization": parse_utilization(paths["utilization"]),
            "power": parse_power(paths["power"]),
            "timing": parse_timing(paths["timing"]),
        },
    }
    summary["headline"] = build_headline(summary)
    summary["key_values"] = build_key_values(summary)

    return summary


def condense_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": summary["schema_version"],
        "generated_at": summary["generated_at"],
        "report_dir": summary["report_dir"],
        "key_values": summary["key_values"],
        "definitions": KEY_VALUE_DEFINITIONS,
        "sources": {
            name: report["source"]
            for name, report in summary["reports"].items()
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Vivado utilization, power, and timing reports into one JSON file."
    )
    parser.add_argument(
        "report_dir",
        type=Path,
        help="Directory containing module_utilization.csv, power_report.csv, and timing_paths.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON file. Defaults to <report_dir>/vivado_report_summary.json.",
    )
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Include parsed Vivado tables in addition to the condensed key_values payload.",
    )

    args = parser.parse_args()
    output = args.output or (args.report_dir / "vivado_report_summary.json")
    summary = summarize_reports(args.report_dir)
    write_json(output, summary if args.include_details else condense_summary(summary))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
