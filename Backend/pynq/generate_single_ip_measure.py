#!/usr/bin/env python3
"""
Generate a PYNQ measurement script from a simple Vitis HLS/Vivado config pair.

This targets the LightCNN-style HLS top where pointer/array arguments are exposed
as m_axi buffers and controlled through AXI-Lite address registers.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def tcl_dict_value(text: str, key: str) -> str | None:
    text = re.sub(r'\\\s*\n\s*', ' ', text)
    pattern = re.compile(r'dict\s+set\s+CONFIG\s+' + re.escape(key) + r'\s+(.+?)(?:\n(?=\S)|\Z)', re.S)
    match = pattern.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    value = value.replace('\\\n', ' ')
    value = re.sub(r'\s+', ' ', value).strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def parse_hls_specs(spec: str) -> dict[str, str]:
    first = spec.split(';')[0].strip().strip('"')
    fields = first.split('|')
    if len(fields) < 7:
        raise ValueError(f"Bad HLS_SPECS entry: {spec}")
    return {
        "ip_name": fields[0],
        "vlnv": fields[1],
        "ctrl_if": fields[2],
        "data_ports": fields[3],
        "clk": fields[4],
        "rst": fields[5],
        "irq": fields[6],
    }


def parse_defines(source: str) -> dict[str, int]:
    defines: dict[str, int] = {}
    for name, value in re.findall(r'^#define\s+(\w+)\s+([0-9]+)\s*$', source, re.M):
        defines[name] = int(value)
    return defines


def resolve_expr(expr: str, defines: dict[str, int]) -> int:
    safe = expr.strip()
    for name, value in defines.items():
        safe = re.sub(rf'\b{re.escape(name)}\b', str(value), safe)
    if not re.fullmatch(r'[0-9\s+\-*/()]+', safe):
        raise ValueError(f"Unsupported array size expression: {expr}")
    return int(eval(safe, {"__builtins__": {}}, {}))


def parse_data_type(source: str) -> str:
    match = re.search(r'typedef\s+ap_int<\s*(\d+)\s*>\s+data_t\s*;', source)
    if not match:
        return "np.int32"
    bits = int(match.group(1))
    if bits <= 8:
        return "np.int8"
    if bits <= 16:
        return "np.int16"
    return "np.int32"


def parse_top_args(source: str, top_function: str, defines: dict[str, int]) -> list[dict[str, object]]:
    pattern = re.compile(r'void\s+' + re.escape(top_function) + r'\s*\((.*?)\)\s*;', re.S)
    match = pattern.search(source)
    if not match:
        pattern = re.compile(r'void\s+' + re.escape(top_function) + r'\s*\((.*?)\)\s*\{', re.S)
        match = pattern.search(source)
    if not match:
        raise ValueError(f"Could not find top function signature for {top_function}")

    args = []
    for raw_arg in match.group(1).split(','):
        arg = raw_arg.strip()
        arg_match = re.search(r'(?:const\s+)?(?P<type>\w+)\s+(?P<name>\w+)\s*\[(?P<size>[^\]]+)\]', arg)
        if not arg_match:
            continue
        name = arg_match.group('name')
        size_expr = arg_match.group('size')
        args.append({"name": name, "size_expr": size_expr, "size": resolve_expr(size_expr, defines)})
    if len(args) < 2:
        raise ValueError("Expected at least one input buffer and one output buffer")
    return args


def default_input_vector(size: int) -> str:
    values = [(i % 8) + 1 for i in range(size)]
    rows = []
    for i in range(0, len(values), 8):
        rows.append("    " + ",".join(str(v) for v in values[i:i + 8]))
    return ",\n".join(rows)


def render_pynq_script(design_name: str, bitstream: str, ip_name: str, top_function: str,
                       np_dtype: str, buffers: list[dict[str, object]], iterations: int,
                       warmup: int) -> str:
    input_buf = buffers[0]
    output_buf = buffers[-1]
    input_values = default_input_vector(int(input_buf["size"]))

    buffer_allocs = []
    register_writes = []
    zero_outputs = []
    flushes = []
    invalidates = []
    output_reads = []

    reg = 0x10
    for buf in buffers:
        name = str(buf["name"])
        size = int(buf["size"])
        buffer_allocs.append(f'{name} = allocate(shape=({size},), dtype={np_dtype})')
        register_writes.append(f'ip.write(0x{reg:02X}, {name}.physical_address & 0xFFFFFFFF)')
        register_writes.append(f'ip.write(0x{reg + 4:02X}, ({name}.physical_address >> 32) & 0xFFFFFFFF)')
        flushes.append(f'{name}.flush()')
        if name != str(input_buf["name"]):
            zero_outputs.append(f'{name}[:] = 0')
            invalidates.append(f'{name}.invalidate()')
            output_reads.append(f'{name}_tmp = np.array({name})')
        reg += 0x0C

    zero_outputs_code = "\n    ".join(zero_outputs) if zero_outputs else "pass"
    flushes_code = "\n    ".join(flushes)
    invalidates_code = "\n    ".join(invalidates) if invalidates else "pass"
    output_reads_code = "\n    ".join(output_reads) if output_reads else "pass"

    return f'''# Auto-generated PYNQ measurement script for {design_name}.
# Generated from Vitis HLS/Vivado config and C++ top function metadata.

from pynq import Overlay, allocate
import numpy as np
import time
import platform

BITSTREAM = "{bitstream}"
N = {iterations}
WARMUP = {warmup}
DESIGN_NAME = "{design_name}"
TOP_FUNCTION = "{top_function}"
IP_NAME = "{ip_name}"

ol = Overlay(BITSTREAM)
ip = getattr(ol, IP_NAME)

print("Loaded bitstream:", BITSTREAM)
print("Available IPs:", list(ol.ip_dict.keys()))
print("Using IP:", IP_NAME)
print(ip.register_map)

{input_buf['name']}_np = np.array([
{input_values}
], dtype={np_dtype})

{chr(10).join(buffer_allocs)}

# AXI-Lite register setup. Address registers follow the HLS order:
# control at 0x00, then 64-bit buffer addresses at 0x10/0x14, 0x1C/0x20, ...
{chr(10).join(register_writes)}


def run_kernel_once():
    {input_buf['name']}[:] = {input_buf['name']}_np
    {zero_outputs_code}
    {flushes_code}

    ip.write(0x00, 1)
    while (ip.read(0x00) & 0x2) == 0:
        pass

    {invalidates_code}
    {output_reads_code}
    return {output_buf['name']}_tmp.copy()


for _ in range(WARMUP):
    _ = run_kernel_once()

transfer_to_fpga_times = []
fpga_times = []
transfer_from_fpga_times = []
end_to_end_times = []
last_output = None

for _ in range(N):
    t_total_start = time.perf_counter_ns()

    t0 = time.perf_counter_ns()
    {input_buf['name']}[:] = {input_buf['name']}_np
    {zero_outputs_code}
    {flushes_code}
    t1 = time.perf_counter_ns()

    t2 = time.perf_counter_ns()
    ip.write(0x00, 1)
    while (ip.read(0x00) & 0x2) == 0:
        pass
    t3 = time.perf_counter_ns()

    t4 = time.perf_counter_ns()
    {invalidates_code}
    {output_buf['name']}_tmp = np.array({output_buf['name']})
    t5 = time.perf_counter_ns()

    t_total_end = time.perf_counter_ns()

    transfer_to_fpga_times.append(t1 - t0)
    fpga_times.append(t3 - t2)
    transfer_from_fpga_times.append(t5 - t4)
    end_to_end_times.append(t_total_end - t_total_start)
    last_output = {output_buf['name']}_tmp.copy()


def stats(x):
    x = np.array(x, dtype=np.float64)
    return {{
        "mean_ns": np.mean(x),
        "min_ns": np.min(x),
        "max_ns": np.max(x),
        "std_ns": np.std(x),
        "median_ns": np.median(x),
        "p95_ns": np.percentile(x, 95),
        "p99_ns": np.percentile(x, 99),
    }}


results = {{
    "Transfer_to_FPGA_input": stats(transfer_to_fpga_times),
    "FPGA_kernel": stats(fpga_times),
    "Transfer_from_FPGA_output": stats(transfer_from_fpga_times),
    "End_to_end": stats(end_to_end_times),
}}

print("\\n============================================================")
print("Benchmark:", DESIGN_NAME)
print("Iterations:", N)
print("Warm-up:", WARMUP)
print("Platform:", platform.platform())
print("============================================================")
print("Final output:", last_output)

for name, r in results.items():
    print(f"\\n{{name}}")
    print(f"Mean latency:   {{r['mean_ns']:.1f}} ns")
    print(f"Min latency:    {{r['min_ns']:.1f}} ns")
    print(f"Max latency:    {{r['max_ns']:.1f}} ns")
    print(f"Std latency:    {{r['std_ns']:.1f}} ns")
    print(f"Median latency: {{r['median_ns']:.1f}} ns")
    print(f"P95 latency:    {{r['p95_ns']:.1f}} ns")
    print(f"P99 latency:    {{r['p99_ns']:.1f}} ns")

print("\\n" + "=" * 120)
print("FINAL BENCHMARK SUMMARY")
print("=" * 120)
print(f"{{'Component':35s}}{{'Mean(ns)':>15s}}{{'Min(ns)':>15s}}{{'Max(ns)':>15s}}{{'Std(ns)':>15s}}{{'Median(ns)':>15s}}{{'P95(ns)':>15s}}{{'P99(ns)':>15s}}")
print("-" * 120)

for component, r in results.items():
    print(f"{{component:35s}}{{r['mean_ns']:15.1f}}{{r['min_ns']:15.1f}}{{r['max_ns']:15.1f}}{{r['std_ns']:15.1f}}{{r['median_ns']:15.1f}}{{r['p95_ns']:15.1f}}{{r['p99_ns']:15.1f}}")

print("=" * 120)
print("\\nRESEARCH RESULT")
print("-" * 120)
r = results["End_to_end"]
print(f"Design: {{DESIGN_NAME}}")
print(f"End-to-end Mean Latency : {{r['mean_ns']:.1f}} ns")
print(f"End-to-end Min Latency  : {{r['min_ns']:.1f}} ns")
print(f"End-to-end Max Latency  : {{r['max_ns']:.1f}} ns")
print(f"End-to-end Std Latency  : {{r['std_ns']:.1f}} ns")
print("=" * 120)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a PYNQ measurement script for a LightCNN-style HLS IP.")
    parser.add_argument("--hls-config", type=Path, required=True)
    parser.add_argument("--vivado-config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--design-name", default="lightcnn_all")
    parser.add_argument("--benchmark-name", default="ABC_FPGA")
    parser.add_argument("--bitstream", default="lightcnn_all.bit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    hls_config = read_text(args.hls_config)
    vivado_config = read_text(args.vivado_config)
    source = read_text(args.source)

    top_function = tcl_dict_value(hls_config, "TOP_FUNCTION") or "lightcnn_top"
    hls_specs = tcl_dict_value(vivado_config, "HLS_SPECS")
    if not hls_specs:
        raise ValueError("Could not find HLS_SPECS in Vivado config")
    ip_info = parse_hls_specs(hls_specs)

    defines = parse_defines(source)
    np_dtype = parse_data_type(source)
    buffers = parse_top_args(source, top_function, defines)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_pynq_script(
            design_name=args.benchmark_name,
            bitstream=args.bitstream,
            ip_name=ip_info["ip_name"],
            top_function=top_function,
            np_dtype=np_dtype,
            buffers=buffers,
            iterations=args.iterations,
            warmup=args.warmup,
        ),
        encoding="utf-8",
    )

    print("Generated PYNQ measurement script")
    print(f"Design:       {args.design_name}")
    print(f"Benchmark:    {args.benchmark_name}")
    print(f"Top function: {top_function}")
    print(f"IP instance:  {ip_info['ip_name']}")
    print(f"Buffers:      {', '.join(str(b['name']) + '[' + str(b['size']) + ']' for b in buffers)}")
    print(f"Output:       {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

