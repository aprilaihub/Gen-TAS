#!/usr/bin/env python3
"""Deterministic correctness and latency runner for one fused CAMC partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pynq import Overlay, allocate


DESIGN_NAME = "__DESIGN_NAME__"
BITSTREAM = "__BITSTREAM__"
IP_NAME = "__IP_NAME__"
PARTITION_ID = "__PARTITION_ID__"
FPGA_STAGES = __FPGA_STAGES__
GPP_STAGES = __GPP_STAGES__
HW_INPUT_LENGTH = __HW_INPUT_LENGTH__
HW_OUTPUT_LENGTH = __HW_OUTPUT_LENGTH__
STATIC_EXPERIMENT_METRICS = __STATIC_EXPERIMENT_METRICS__

ITERATIONS = 1000
WARMUP = 100
KERNEL_TIMEOUT_SECONDS = 5.0
STAGES = ("S1", "S2", "S3")
SAMPLE_COUNT = 800
INPUT_SIZE = 1600
COORDINATE_SIZE = 1600
HISTOGRAM_SIZE = 10_000
SCORE_COUNT = 4
GRID_SIZE = 100
MASK20 = (1 << 20) - 1
MASK64 = (1 << 64) - 1
LAST_OUTPUT_DIR = None

EXPECTED_SCORES = {
    "original_16qam": [0, 0, 0, 91121],
    "distributed_minus4_to_plus4": [1993, 1866, 1590, 2035],
    "bpsk_like": [1058800, 756400, 314000, 0],
    "four_corners_plus_minus2_5": [0, 0, 0, 0],
    "distributed_minus3_5_to_plus3_5": [2121, 2572, 2098, 2341],
}
PREFIXES = (
    "2PSK", "2PSK_45m", "2PSK_45p", "2PSK_90p", "4PSK",
    "4PSK_45m", "8PSK", "8PSK_45m", "16QAM", "16QAM_45m",
)


def base_dir() -> Path:
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed20(value: float) -> int:
    return math.floor(float(value) * 4096.0) & MASK20


def make_case(name: str) -> np.ndarray:
    original_x = (-2.91027081, -1.043975, 0.843665, -1.005242, 3.0777832,
                  -2.987052, -2.97210, 3.10400, 0.9596310)
    original_y = (2.9327373, 2.879154, 3.0109, 2.920826, -3.1413207,
                  -1.02065, -0.9094878, 1.150434, 2.984448)
    values = np.empty(INPUT_SIZE, dtype=np.uint32)
    for i in range(SAMPLE_COUNT):
        if name == "original_16qam":
            x, y = original_x[i % 9], original_y[i % 9]
        elif name == "distributed_minus4_to_plus4":
            x = ((i * 37) % 801) / 100.0 - 4.0
            y = ((i * 53 + 17) % 801) / 100.0 - 4.0
        elif name == "bpsk_like":
            x, y = (1.0 if i & 1 else -1.0), 0.0
        elif name == "four_corners_plus_minus2_5":
            x = 2.5 if i & 1 else -2.5
            y = 2.5 if i & 2 else -2.5
        elif name == "distributed_minus3_5_to_plus3_5":
            x = ((i * 29 + 11) % 701) / 100.0 - 3.5
            y = ((i * 71 + 23) % 701) / 100.0 - 3.5
        else:
            raise ValueError(f"Unknown CAMC case: {name}")
        values[i] = fixed20(x)
        values[SAMPLE_COUNT + i] = fixed20(y)
    return values


def stage_s1(values: np.ndarray) -> np.ndarray:
    coordinates = np.zeros(COORDINATE_SIZE, dtype=np.uint8)
    for i, raw_value in enumerate(values):
        raw = int(raw_value) & MASK20
        if raw & (1 << 19):
            axis_raw = 5 * 4096 - ((1 << 20) - raw)
        else:
            axis_raw = raw + 5 * 4096
        if axis_raw < 0 or axis_raw >= 16 * 4096:
            continue
        scaled_raw = axis_raw * 10
        rounded = scaled_raw >> 12
        if (scaled_raw & 0xFFF) >= 2048 and scaled_raw < GRID_SIZE * 4096:
            rounded += 1
        elif rounded >= GRID_SIZE:
            rounded = 0
        coordinates[i] = rounded
    return coordinates


def stage_s2(coordinates: np.ndarray) -> np.ndarray:
    histogram = np.zeros(HISTOGRAM_SIZE, dtype=np.uint16)
    for i in range(SAMPLE_COUNT):
        x = int(coordinates[i])
        y = int(coordinates[SAMPLE_COUNT + i])
        if 0 < x < GRID_SIZE and 0 < y < GRID_SIZE:
            histogram[(x - 1) * GRID_SIZE + y - 1] += 1
    return histogram


def load_weights(path: str | Path = "weights.hpp") -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    candidates = (Path(path), base_dir() / path, base_dir() / "weights.hpp", base_dir() / "src" / "weights.hpp")
    weights_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if weights_path is None:
        raise FileNotFoundError("CAMC weights.hpp must be beside the runner or in src/")
    text = weights_path.read_text(encoding="utf-8")

    def array(name: str) -> np.ndarray:
        match = re.search(rf"\b{name}\s*\[[^]]+\]\s*=\s*\{{([^}}]+)\}}", text)
        if not match:
            raise RuntimeError(f"Missing {name} in {weights_path}")
        return np.asarray([int(value) for value in re.findall(r"\d+", match.group(1))], dtype=np.uint64)

    result = {}
    for prefix in PREFIXES:
        values = array(f"Lite_{prefix}_weight_10")
        xs = array(f"Lite_{prefix}_weight_X_10")
        ys = array(f"Lite_{prefix}_weight_Y_10")
        if not (len(values) == len(xs) == len(ys)):
            raise RuntimeError(f"CAMC weight length mismatch for {prefix}")
        result[prefix] = values, xs, ys
    return result


def stage_s3(histogram: np.ndarray, weights) -> np.ndarray:
    scores = []
    for prefix in PREFIXES:
        values, xs, ys = weights[prefix]
        indexes = (xs.astype(np.int64) - 1) * GRID_SIZE + ys.astype(np.int64) - 1
        score = int(np.sum(values * histogram[indexes].astype(np.uint64), dtype=np.uint64))
        scores.append(score & MASK64)
    return np.asarray([
        max(scores[0:4]), max(scores[4:6]), max(scores[6:8]), max(scores[8:10])
    ], dtype=np.uint64)


def run_gpp_stage(stage: str, data: np.ndarray, weights) -> np.ndarray:
    if stage == "S1":
        return stage_s1(data)
    if stage == "S2":
        return stage_s2(data)
    if stage == "S3":
        return stage_s3(data, weights)
    raise ValueError(stage)


def boundary_dtype(stage: str, is_input: bool):
    if is_input:
        return {"S1": np.uint32, "S2": np.uint8, "S3": np.uint16}[stage]
    return {"S1": np.uint8, "S2": np.uint16, "S3": np.uint64}[stage]


def setup_overlay():
    bit = base_dir() / BITSTREAM
    hwh = bit.with_suffix(".hwh")
    if not bit.is_file() or not hwh.is_file():
        raise FileNotFoundError(f"Missing {bit} or {hwh}")
    overlay = Overlay(str(bit))
    if IP_NAME not in overlay.ip_dict:
        raise RuntimeError(f"Missing {IP_NAME}; available={sorted(overlay.ip_dict)}")
    ip = getattr(overlay, IP_NAME)
    a = allocate(shape=(HW_INPUT_LENGTH,), dtype=boundary_dtype(FPGA_STAGES[0], True))
    b = allocate(shape=(HW_OUTPUT_LENGTH,), dtype=boundary_dtype(FPGA_STAGES[-1], False))
    ip.write(0x10, a.physical_address & 0xFFFFFFFF)
    ip.write(0x14, a.physical_address >> 32)
    ip.write(0x1C, b.physical_address & 0xFFFFFFFF)
    ip.write(0x20, b.physical_address >> 32)
    return overlay, ip, a, b


def run_fpga(ip, a, b, values: np.ndarray):
    started = time.perf_counter_ns()
    a[:] = values
    a.flush()
    transfer_in = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    ip.write(0x00, 1)
    deadline = time.monotonic() + KERNEL_TIMEOUT_SECONDS
    while (ip.read(0x00) & 0x2) == 0:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{IP_NAME} timed out")
    kernel = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    b.invalidate()
    output = np.asarray(b).copy()
    transfer_out = time.perf_counter_ns() - started
    return output, transfer_in, kernel, transfer_out


def run_partition(ip, a, b, input_values, weights, samples=None):
    samples = samples if samples is not None else defaultdict(list)
    data = input_values
    fpga_started = False
    for stage in STAGES:
        if stage == FPGA_STAGES[0]:
            data, transfer_in, kernel, transfer_out = run_fpga(ip, a, b, data)
            samples["transfer_to_fpga_ns"].append(transfer_in)
            samples["fpga_kernel_ns"].append(kernel)
            samples["transfer_from_fpga_ns"].append(transfer_out)
            fpga_started = True
            continue
        if fpga_started and stage in FPGA_STAGES:
            continue
        started = time.perf_counter_ns()
        data = run_gpp_stage(stage, data, weights)
        samples[f"gpp_{stage.lower()}_ns"].append(time.perf_counter_ns() - started)
    return data


def golden_trace(input_values, weights):
    trace = {}
    data = input_values
    for stage in STAGES:
        data = run_gpp_stage(stage, data, weights)
        trace[stage] = data
    return trace


def measure_gpp_baseline(cases, weights, warmup: int, iterations: int):
    for index in range(warmup):
        golden_trace(cases[index % len(cases)][1], weights)
    samples = defaultdict(list)
    for index in range(iterations):
        data = cases[index % len(cases)][1]
        total_started = time.perf_counter_ns()
        for stage in STAGES:
            started = time.perf_counter_ns()
            data = run_gpp_stage(stage, data, weights)
            samples[f"gpp_{stage.lower()}_ns"].append(time.perf_counter_ns() - started)
        samples["end_to_end_ns"].append(time.perf_counter_ns() - total_started)
    return {name: stats(values) for name, values in samples.items()}


def verify_all(ip, a, b, weights):
    records = []
    for name, expected_scores in EXPECTED_SCORES.items():
        values = make_case(name)
        trace = golden_trace(values, weights)
        if trace["S3"].astype(object).tolist() != expected_scores:
            raise AssertionError(f"CAMC software oracle mismatch for {name}: {trace['S3'].tolist()}")
        actual = run_partition(ip, a, b, values, weights)
        expected_boundary = trace[FPGA_STAGES[-1]]
        boundary_actual, _, _, _ = run_fpga(
            ip, a, b,
            values if FPGA_STAGES[0] == "S1" else trace[STAGES[STAGES.index(FPGA_STAGES[0]) - 1]],
        )
        if not np.array_equal(boundary_actual, expected_boundary):
            raise AssertionError(
                f"CAMC FPGA boundary mismatch for {name}: "
                f"actual {boundary_actual.astype(object).tolist()} != "
                f"expected {expected_boundary.astype(object).tolist()}"
            )
        if actual.astype(object).tolist() != expected_scores:
            raise AssertionError(f"CAMC final mismatch for {name}: {actual.tolist()}")
        records.append({"name": name, "scores": expected_scores, "pass": True})
        print("PASS", name, expected_scores)
    return records


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ns": float(np.mean(values)), "median_ns": float(np.median(values)),
        "std_ns": float(np.std(values)), "min_ns": float(np.min(values)),
        "max_ns": float(np.max(values)), "p95_ns": float(np.percentile(values, 95)),
        "p99_ns": float(np.percentile(values, 99)),
    }


def benchmark(args):
    global LAST_OUTPUT_DIR
    LAST_OUTPUT_DIR = Path(args.output_dir)
    weights_path = next(
        (candidate for candidate in (
            Path(args.weights), base_dir() / args.weights,
            base_dir() / "weights.hpp", base_dir() / "src" / "weights.hpp",
        ) if candidate.is_file()),
        None,
    )
    if weights_path is None:
        raise FileNotFoundError("CAMC weights.hpp must be beside the runner or in src/")
    weights = load_weights(weights_path)
    _, ip, a, b = setup_overlay()
    verified = verify_all(ip, a, b, weights)
    cases = [(name, make_case(name)) for name in EXPECTED_SCORES]
    gpp_baseline = measure_gpp_baseline(cases, weights, args.warmup, args.iterations)
    for index in range(args.warmup):
        run_partition(ip, a, b, cases[index % len(cases)][1], weights)
    samples = defaultdict(list)
    for index in range(args.iterations):
        started = time.perf_counter_ns()
        run_partition(ip, a, b, cases[index % len(cases)][1], weights, samples)
        samples["end_to_end_ns"].append(time.perf_counter_ns() - started)
    latency = {name: stats(values) for name, values in samples.items()}
    gpp_median = gpp_baseline["end_to_end_ns"]["median_ns"]
    partition_median = latency["end_to_end_ns"]["median_ns"]
    payload = {
        "schema_version": "camc-partition-results-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design_name": DESIGN_NAME,
        "partition_id": PARTITION_ID,
        "fpga_stages": FPGA_STAGES,
        "gpp_stages": GPP_STAGES,
        "topology": "partition_specific_fused",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "golden_vectors": verified,
        "golden_vector_pass": True,
        "latency_ns": latency,
        "gpp_baseline_latency_ns": gpp_baseline,
        "gpp_baseline_implementation": "Python/NumPy integer software reference on the PYNQ ARM host",
        "speedup_vs_gpp_median": gpp_median / partition_median,
        "latency_improvement_vs_gpp_percent": 100.0 * (gpp_median - partition_median) / gpp_median,
        "measurement_note": "Transfer fields measure shared-DDR copy/cache-coherency operations, not DMA.",
        "artifact_sha256": {
            "bitstream": sha256_file(base_dir() / BITSTREAM),
            "hardware_handoff": sha256_file((base_dir() / BITSTREAM).with_suffix(".hwh")),
            "weights": sha256_file(weights_path),
        },
        "static_experiment_metrics": STATIC_EXPERIMENT_METRICS,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "camc_results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["latency_ns"], indent=2))
    print("Saved", output)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights.hpp")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--output-dir", default=f"results_{DESIGN_NAME}")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        BENCHMARK_RESULTS = benchmark(parse_args())
    except Exception as exc:
        destination = LAST_OUTPUT_DIR or Path(f"results_{DESIGN_NAME}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "camc_failure.json").write_text(json.dumps({
            "design_name": DESIGN_NAME, "partition_id": PARTITION_ID,
            "status": "failed", "type": type(exc).__name__, "message": str(exc),
        }, indent=2) + "\n", encoding="utf-8")
        raise
