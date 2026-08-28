#!/usr/bin/env python3
"""PYNQ accuracy + latency runner for CNNImageProc partitions."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import struct
import time
import zipfile
from datetime import datetime
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

NUM_SAMPLES = 10_000
WARMUP = 10
PROGRESS_EVERY = 1000
KERNEL_TIMEOUT_SECONDS = 5.0

IMG_H = 28
IMG_W = 28
IMG_SIZE = IMG_H * IMG_W
CONV1_OUT_CH = 16
CONV1_H = 28
CONV1_W = 28
POOL1_H = 14
POOL1_W = 14
POOL1_SIZE = CONV1_OUT_CH * POOL1_H * POOL1_W
CONV2_OUT_CH = 32
CONV2_H = 14
CONV2_W = 14
POOL2_H = 7
POOL2_W = 7
POOL2_SIZE = CONV2_OUT_CH * POOL2_H * POOL2_W
NUM_CLASSES = 10
DENSE_IN_SIZE = POOL2_SIZE
K = 3
PAD = 1

DATA_FRAC_BITS = 12
WEIGHT_FRAC_BITS = 14
DATA_SCALE = 1 << DATA_FRAC_BITS
WEIGHT_SCALE = 1 << WEIGHT_FRAC_BITS
DTYPE = np.int16
LAST_OUTPUT_DIR = None


def _base_dir() -> Path:
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def _artifact_path(filename: str) -> Path:
    path = Path(filename)
    return path if path.is_absolute() else _base_dir() / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_overlay_artifacts(bitstream: str) -> Path:
    bit_path = _artifact_path(bitstream)
    hwh_path = bit_path.with_suffix(".hwh")
    missing = [str(p) for p in (bit_path, hwh_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required overlay artifact(s) not found: " + ", ".join(missing))
    return bit_path


def _trunc_fixed(values: np.ndarray, frac_bits: int, min_raw: int, max_raw: int) -> np.ndarray:
    # ap_fixed AP_TRN drops fractional bits toward negative infinity.
    raw = np.floor(values.astype(np.float64) * (1 << frac_bits))
    return np.clip(raw, min_raw, max_raw).astype(np.int64)


def quantize_data_t(values: np.ndarray) -> np.ndarray:
    return _trunc_fixed(values, DATA_FRAC_BITS, -32768, 32767).astype(np.int16)


def dequantize_data_t(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) / DATA_SCALE


def quantize_weight_t_raw(values: np.ndarray) -> np.ndarray:
    raw = _trunc_fixed(values, WEIGHT_FRAC_BITS, -32768, 32767)
    return raw.astype(np.int16)


def load_weights(weights_hpp: str | Path = "weights.hpp") -> dict[str, np.ndarray]:
    base = _base_dir()
    candidates = [
        Path(weights_hpp),
        base / weights_hpp,
        base / "weights.hpp",
        base / "weights(1).hpp",
        base / "src" / "weights.hpp",
    ]
    weights_path = next((p for p in candidates if p.is_file()), None)
    if weights_path is None:
        raise FileNotFoundError("Could not find weights.hpp. Put it beside this script or in src/weights.hpp.")
    text = weights_path.read_text(encoding="utf-8")

    def extract(name: str, expected: int) -> np.ndarray:
        match = re.search(rf"const\s+weight_t\s+{name}\s*\[[^\]]+\]\s*=\s*\{{(.*?)\}}\s*;", text, re.S)
        if not match:
            raise RuntimeError(f"Could not find {name} array in {weights_path}")
        vals = re.findall(r"weight_t\(([-+0-9.eE]+)\)", match.group(1))
        if len(vals) != expected:
            raise RuntimeError(f"{name} size mismatch: got {len(vals)}, expected {expected}")
        return quantize_weight_t_raw(np.array([float(v) for v in vals], dtype=np.float64))

    return {
        "conv1_weights": extract("conv1_weights", CONV1_OUT_CH * K * K).reshape(CONV1_OUT_CH, K, K),
        "conv1_bias": extract("conv1_bias", CONV1_OUT_CH),
        "conv2_weights": extract("conv2_weights", CONV2_OUT_CH * CONV1_OUT_CH * K * K).reshape(
            CONV2_OUT_CH, CONV1_OUT_CH, K, K
        ),
        "conv2_bias": extract("conv2_bias", CONV2_OUT_CH),
        "dense_weights": extract("dense_weights", NUM_CLASSES * DENSE_IN_SIZE).reshape(NUM_CLASSES, DENSE_IN_SIZE),
        "dense_bias": extract("dense_bias", NUM_CLASSES),
    }


def _read_idx_images_from_bytes(blob: bytes) -> np.ndarray:
    magic, n, rows, cols = struct.unpack(">IIII", blob[:16])
    if magic != 2051:
        raise ValueError(f"Bad image IDX magic: {magic}")
    return np.frombuffer(blob, dtype=np.uint8, offset=16).reshape(n, rows, cols)


def _read_idx_labels_from_bytes(blob: bytes) -> np.ndarray:
    magic, n = struct.unpack(">II", blob[:8])
    if magic != 2049:
        raise ValueError(f"Bad label IDX magic: {magic}")
    return np.frombuffer(blob, dtype=np.uint8, offset=8).reshape(n)


def load_mnist_test(data_zip: str | Path = "FashionMNIST_data.zip") -> tuple[np.ndarray, np.ndarray, str]:
    base = _base_dir()
    zip_path = next(
        (
            p
            for p in (
                Path(data_zip),
                base / data_zip,
                base / "FashionMNIST_data.zip",
                base / "data_mnist.zip",
            )
            if p.is_file()
        ),
        None,
    )
    if zip_path is not None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            for dataset in ("FashionMNIST", "MNIST"):
                img = f"data/{dataset}/raw/t10k-images-idx3-ubyte"
                lbl = f"data/{dataset}/raw/t10k-labels-idx1-ubyte"
                if img in names and lbl in names:
                    return _read_idx_images_from_bytes(zf.read(img)), _read_idx_labels_from_bytes(zf.read(lbl)), dataset
                if img + ".gz" in names and lbl + ".gz" in names:
                    return _read_idx_images_from_bytes(gzip.decompress(zf.read(img + ".gz"))), _read_idx_labels_from_bytes(
                        gzip.decompress(zf.read(lbl + ".gz"))
                    ), dataset
    for dataset in ("FashionMNIST", "MNIST"):
        raw_dir = base / "data" / dataset / "raw"
        img_path = raw_dir / "t10k-images-idx3-ubyte"
        lbl_path = raw_dir / "t10k-labels-idx1-ubyte"
        if img_path.is_file() and lbl_path.is_file():
            return _read_idx_images_from_bytes(img_path.read_bytes()), _read_idx_labels_from_bytes(lbl_path.read_bytes()), dataset
    raise FileNotFoundError("Could not find FashionMNIST_data.zip/data_mnist.zip or extracted IDX test files.")


def quantize_mnist_image(image_u8: np.ndarray) -> np.ndarray:
    return quantize_data_t(image_u8.reshape(-1).astype(np.float32) / 255.0)


def conv1_feature_extract(a_raw: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    a = np.pad(a_raw.astype(np.int64).reshape(IMG_H, IMG_W), PAD)
    bias = weights["conv1_bias"].astype(np.int64)
    kernel = weights["conv1_weights"].astype(np.int64)
    acc = np.broadcast_to(bias[:, None, None] << 10, (CONV1_OUT_CH, CONV1_H, CONV1_W)).copy()
    for ky in range(K):
        for kx in range(K):
            pixels = a[ky:ky + CONV1_H, kx:kx + CONV1_W]
            acc += (kernel[:, ky, kx, None, None] * pixels[None, :, :]) >> 2
    return np.clip(acc >> 12, -32768, 32767).astype(np.int16).reshape(-1)


def relu_pool1(conv1_raw: np.ndarray) -> np.ndarray:
    x = np.maximum(conv1_raw.reshape(CONV1_OUT_CH, CONV1_H, CONV1_W), 0)
    pooled = x.reshape(CONV1_OUT_CH, POOL1_H, 2, POOL1_W, 2).max(axis=(2, 4))
    return pooled.astype(np.int16).reshape(-1)


def conv2_feature_extract(pool1_raw: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    pool1 = np.pad(pool1_raw.astype(np.int64).reshape(CONV1_OUT_CH, POOL1_H, POOL1_W),
                   ((0, 0), (PAD, PAD), (PAD, PAD)))
    bias = weights["conv2_bias"].astype(np.int64)
    kernel = weights["conv2_weights"].astype(np.int64)
    acc = np.broadcast_to(bias[:, None, None] << 10, (CONV2_OUT_CH, CONV2_H, CONV2_W)).copy()
    for ic in range(CONV1_OUT_CH):
        for ky in range(K):
            for kx in range(K):
                pixels = pool1[ic, ky:ky + CONV2_H, kx:kx + CONV2_W]
                acc += (kernel[:, ic, ky, kx, None, None] * pixels[None, :, :]) >> 2
    return np.clip(acc >> 12, -32768, 32767).astype(np.int16).reshape(-1)


def relu_pool2(conv2_raw: np.ndarray) -> np.ndarray:
    x = np.maximum(conv2_raw.reshape(CONV2_OUT_CH, CONV2_H, CONV2_W), 0)
    pooled = x.reshape(CONV2_OUT_CH, POOL2_H, 2, POOL2_W, 2).max(axis=(2, 4))
    return pooled.astype(np.int16).reshape(-1)


def dense_classifier(pool2_raw: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pool2 = pool2_raw.astype(np.int64)
    dense_weights = weights["dense_weights"].astype(np.int64)
    acc = (weights["dense_bias"].astype(np.int64) << 10) + np.sum(
        (dense_weights * pool2[None, :]) >> 2, axis=1, dtype=np.int64
    )
    logits_raw = np.clip(acc >> 12, -32768, 32767).astype(np.int16)
    return dequantize_data_t(logits_raw), logits_raw


def run_cpu_stage(stage: str, data: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    if stage == "S1":
        return conv1_feature_extract(data, weights)
    if stage == "S2":
        return relu_pool1(data)
    if stage == "S3":
        return conv2_feature_extract(data, weights)
    if stage == "S4":
        return relu_pool2(data)
    if stage == "S5":
        return dense_classifier(data, weights)[1]
    raise ValueError(f"Unknown stage: {stage}")


def _write_buffer_addresses(ip, input_buffer, output_buffer) -> None:
    ip.write(0x10, input_buffer.physical_address & 0xFFFFFFFF)
    ip.write(0x14, (input_buffer.physical_address >> 32) & 0xFFFFFFFF)
    ip.write(0x1C, output_buffer.physical_address & 0xFFFFFFFF)
    ip.write(0x20, (output_buffer.physical_address >> 32) & 0xFFFFFFFF)


def setup_overlay_and_buffers(bitstream: str, ip_name: str):
    bit_path = _check_overlay_artifacts(bitstream)
    ol = Overlay(str(bit_path))
    if ip_name not in ol.ip_dict:
        available = ", ".join(sorted(ol.ip_dict.keys()))
        raise RuntimeError(f"Missing IP instance {ip_name}; available IPs: {available}")
    ip = getattr(ol, ip_name)
    a = allocate(shape=(HW_INPUT_LENGTH,), dtype=DTYPE)
    b = allocate(shape=(HW_OUTPUT_LENGTH,), dtype=DTYPE)
    _write_buffer_addresses(ip, a, b)
    return ol, ip, a, b


def run_fpga_once(ip, a, b, input_q: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    timings = {}
    t0 = time.perf_counter_ns()
    a[:] = input_q
    a.flush()
    t1 = time.perf_counter_ns()

    ip.write(0x00, 1)
    deadline = time.monotonic() + KERNEL_TIMEOUT_SECONDS
    t2 = time.perf_counter_ns()
    while (ip.read(0x00) & 0x2) == 0:
        if time.monotonic() > deadline:
            raise TimeoutError(f"FPGA kernel did not finish within {KERNEL_TIMEOUT_SECONDS} seconds")
    t3 = time.perf_counter_ns()

    b.invalidate()
    output_raw = np.array(b, dtype=np.int16)
    t4 = time.perf_counter_ns()
    timings["transfer_to_fpga_ns"] = t1 - t0
    timings["fpga_kernel_ns"] = t3 - t2
    timings["transfer_from_fpga_ns"] = t4 - t3
    return output_raw, timings


def run_partition_once(ip, a, b, image_q: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    data = image_q
    timings: dict[str, int] = {}
    t_gpp = 0

    for stage in ("S1", "S2", "S3", "S4", "S5"):
        if stage in FPGA_STAGES:
            fpga_out, fpga_timings = run_fpga_once(ip, a, b, data)
            timings.update(fpga_timings)
            data = fpga_out
            for skipped in FPGA_STAGES[1:]:
                if skipped == stage:
                    continue
            break
        t0 = time.perf_counter_ns()
        data = run_cpu_stage(stage, data, weights)
        t_gpp += time.perf_counter_ns() - t0

    last_fpga_stage = FPGA_STAGES[-1] if FPGA_STAGES else None
    remaining = ("S1", "S2", "S3", "S4", "S5")
    start_index = remaining.index(last_fpga_stage) + 1 if last_fpga_stage else 0
    for stage in remaining[start_index:]:
        if stage in FPGA_STAGES:
            continue
        t0 = time.perf_counter_ns()
        data = run_cpu_stage(stage, data, weights)
        t_gpp += time.perf_counter_ns() - t0

    timings["gpp_stages_ns"] = t_gpp
    logits_raw = data
    logits_q = dequantize_data_t(logits_raw)
    return logits_q, logits_raw, timings


def run_software_once(image_q: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    data = image_q
    for stage in ("S1", "S2", "S3", "S4", "S5"):
        data = run_cpu_stage(stage, data, weights)
    return np.asarray(data, dtype=np.int16)


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def select_verification_indices(labels: np.ndarray, cases: int) -> list[int]:
    """Choose a stable class-stratified prefix without selecting only easy cases."""
    limit = min(max(1, cases), len(labels))
    selected = []
    for class_id in range(NUM_CLASSES):
        matches = np.flatnonzero(labels == class_id)
        if matches.size and len(selected) < limit:
            selected.append(int(matches[0]))
    selected_set = set(selected)
    for index in range(len(labels)):
        if len(selected) >= limit:
            break
        if index not in selected_set:
            selected.append(index)
    return selected


def software_stage_trace(image_q: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    trace = {}
    data = image_q
    for stage in ("S1", "S2", "S3", "S4", "S5"):
        data = run_cpu_stage(stage, data, weights)
        trace[stage] = np.asarray(data, dtype=np.int16)
    return trace


def verify_partition(ip, a, b, images, labels, weights, cases: int) -> dict:
    indices = select_verification_indices(labels, cases)
    checked = len(indices)
    mismatches = []
    boundary_mismatches = []
    golden_cases = []
    stages = ("S1", "S2", "S3", "S4", "S5")
    for index in indices:
        input_q = quantize_mnist_image(images[index])
        trace = software_stage_trace(input_q, weights)
        expected_raw = trace["S5"]
        _, actual_raw, _ = run_partition_once(ip, a, b, input_q, weights)
        if not np.array_equal(expected_raw, actual_raw):
            mismatches.append({
                "sample_index": index,
                "software_raw": expected_raw.astype(int).tolist(),
                "fpga_partition_raw": actual_raw.astype(int).tolist(),
            })
        if FPGA_STAGES:
            boundary_input = input_q
            for stage in stages[:stages.index(FPGA_STAGES[0])]:
                boundary_input = run_cpu_stage(stage, boundary_input, weights)
            expected_boundary = boundary_input
            for stage in FPGA_STAGES:
                expected_boundary = trace[stage]
            actual_boundary, _ = run_fpga_once(ip, a, b, boundary_input)
            if not np.array_equal(expected_boundary, actual_boundary):
                boundary_mismatches.append(index)
        golden_cases.append({
            "sample_index": index,
            "label": int(labels[index]),
            "input_sha256": _sha256_array(input_q),
            "stage_sha256": {stage: _sha256_array(trace[stage]) for stage in stages},
            "prediction": int(np.argmax(dequantize_data_t(expected_raw))),
            "final_logits_raw": expected_raw.astype(int).tolist(),
        })
    passed = checked > 0 and not mismatches and not boundary_mismatches
    boundary_passed = checked > 0 and not boundary_mismatches
    return {
        "cases": checked,
        "selection": "first occurrence of each class, then lowest unused indices",
        "sample_indices": indices,
        "class_coverage": sorted({int(labels[index]) for index in indices}),
        "golden_cases": golden_cases,
        "status": "passed" if passed else "failed",
        "software_vs_fpga": passed,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
        "fpga_block_boundary_equivalence": boundary_passed,
        "fpga_block_boundary_mismatch_samples": boundary_mismatches[:10],
        "stage_status": {
            stage: (
                ("passed_as_fused_fpga_block" if boundary_passed else "failed_as_fused_fpga_block")
                if stage in FPGA_STAGES
                else "software_reference"
            )
            for stage in stages
        },
    }


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean_ns": float(np.mean(x)),
        "min_ns": float(np.min(x)),
        "max_ns": float(np.max(x)),
        "std_ns": float(np.std(x)),
        "median_ns": float(np.median(x)),
        "p95_ns": float(np.percentile(x, 95)),
        "p99_ns": float(np.percentile(x, 99)),
    }


def measure_cpu_baseline(images, weights, samples: int, warmup: int) -> dict:
    count = min(samples, len(images))
    for index in range(min(warmup, count)):
        run_software_once(quantize_mnist_image(images[index]), weights)
    values = []
    for index in range(count):
        input_q = quantize_mnist_image(images[index])
        started = time.perf_counter_ns()
        run_software_once(input_q, weights)
        values.append(time.perf_counter_ns() - started)
    return {
        "samples": count,
        "implementation": "Python/NumPy fixed-point software reference on the PYNQ ARM host",
        "latency_ns": stats(values),
    }


def fmt_us(ns: float) -> str:
    return f"{ns / 1000.0:,.3f} us"


def evaluate_requirements(metrics: dict) -> None:
    hardware = metrics.get("hardware", {})
    end_to_end = metrics.get("runtime", {}).get("latency_stats_ns", {}).get("end_to_end_ns", {})
    measured = {
        "latency": (end_to_end.get("mean_ns"), "ns"),
        "power": (hardware.get("total_power_w"), "w"),
        "lut": (hardware.get("lut"), "count"),
        "ff": (hardware.get("ff"), "count"),
        "dsp": (hardware.get("dsp"), "count"),
        "bram": (hardware.get("bram18_equiv"), "count"),
    }
    scale = {("ns", "us"): 1e-3, ("ns", "ms"): 1e-6, ("ns", "s"): 1e-9, ("w", "mw"): 1e3}
    evaluations = []
    for objective in metrics.get("requirements", {}).get("objectives", []):
        value, native_unit = measured.get(objective.get("metric"), (None, None))
        target_unit = str(objective.get("unit", native_unit)).lower()
        converted = value
        if isinstance(value, (int, float)) and native_unit != target_unit:
            converted = value * scale.get((native_unit, target_unit), 1.0)
        satisfied = converted <= objective.get("target") if isinstance(converted, (int, float)) else None
        evaluations.append({**objective, "measured": converted, "measured_unit": target_unit, "satisfied": satisfied})
    completed = [item for item in evaluations if item["satisfied"] is not None]
    requirements = metrics.setdefault("requirements", {})
    requirements["evaluations"] = evaluations
    requirements["satisfied"] = (
        all(item["satisfied"] for item in completed)
        if evaluations and len(completed) == len(evaluations) else None
    )
    requirements["satisfaction_percent"] = (
        100.0 * sum(bool(item["satisfied"]) for item in completed) / len(completed)
        if completed else None
    )


def measure_accuracy(ip, a, b, images, labels, weights, num_samples: int, warmup: int, progress_every: int):
    num_samples = min(num_samples, len(labels))
    warm_input = quantize_mnist_image(images[0])
    for _ in range(warmup):
        run_partition_once(ip, a, b, warm_input, weights)

    correct = 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    first16 = []
    rows = []
    timings = {
        "transfer_to_fpga_ns": [],
        "fpga_kernel_ns": [],
        "transfer_from_fpga_ns": [],
        "gpp_stages_ns": [],
        "end_to_end_ns": [],
    }

    for i in range(num_samples):
        input_q = quantize_mnist_image(images[i])
        t_start = time.perf_counter_ns()
        logits_q, logits_raw, t_parts = run_partition_once(ip, a, b, input_q, weights)
        t_end = time.perf_counter_ns()

        pred = int(np.argmax(logits_q))
        label = int(labels[i])
        is_correct = int(pred == label)
        correct += is_correct
        confusion[label, pred] += 1

        row = {
            "sample_index": i,
            "label": label,
            "prediction": pred,
            "correct": is_correct,
            "transfer_to_fpga_ns": t_parts.get("transfer_to_fpga_ns", 0),
            "fpga_kernel_ns": t_parts.get("fpga_kernel_ns", 0),
            "transfer_from_fpga_ns": t_parts.get("transfer_from_fpga_ns", 0),
            "gpp_stages_ns": t_parts.get("gpp_stages_ns", 0),
            "end_to_end_ns": t_end - t_start,
        }
        for c in range(NUM_CLASSES):
            row[f"logit_q_c{c}"] = float(logits_q[c])
            row[f"logit_raw_c{c}"] = int(logits_raw[c])
        rows.append(row)

        for key in ("transfer_to_fpga_ns", "fpga_kernel_ns", "transfer_from_fpga_ns", "gpp_stages_ns"):
            timings[key].append(t_parts.get(key, 0))
        timings["end_to_end_ns"].append(t_end - t_start)

        if i < 16:
            first16.append((i, label, pred, logits_q.copy()))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"Sample {i + 1}/{num_samples}: accuracy = {100.0 * correct / (i + 1):.2f}%")

    return correct, confusion, first16, timings, rows


def save_results(output_dir: Path, correct: int, confusion: np.ndarray, timings: dict, rows: list[dict], num_samples: int, args, verification: dict, cpu_baseline: dict | None, dataset_name: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / "predictions_and_timings.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    confusion_csv = output_dir / "confusion_matrix.csv"
    np.savetxt(confusion_csv, confusion, delimiter=",", fmt="%d")
    latency_stats = {name: stats(values) for name, values in timings.items()}
    mean_e2e_ns = latency_stats["end_to_end_ns"]["mean_ns"]
    summary = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "design_name": args.design_name,
        "partition_id": PARTITION_ID,
        "fpga_stages": FPGA_STAGES,
        "gpp_stages": GPP_STAGES,
        "bitstream": args.bitstream,
        "ip_name": args.ip_name,
        "dataset": f"{dataset_name} test",
        "num_samples": num_samples,
        "correct": correct,
        "incorrect": num_samples - correct,
        "accuracy_percent": 100.0 * correct / num_samples,
        "throughput_fps_from_mean_end_to_end": 1e9 / mean_e2e_ns if mean_e2e_ns > 0 else None,
        "warmup": args.warmup,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "artifact_sha256": {
            "bitstream": _sha256_file(_artifact_path(args.bitstream)),
            "hardware_handoff": _sha256_file(_artifact_path(args.bitstream).with_suffix(".hwh")),
        },
        "latency_stats_ns": latency_stats,
        "verification": verification,
        "cpu_only_baseline": cpu_baseline,
        "measurement_note": "Transfer fields measure shared-DDR copies and cache-coherency operations, not DMA.",
        "outputs": {
            "predictions_and_timings_csv": str(predictions_csv),
            "confusion_matrix_csv": str(confusion_csv),
        },
    }
    summary_json = output_dir / "summary.json"
    summary["outputs"]["summary_json"] = str(summary_json)
    experiment_metrics = json.loads(json.dumps(STATIC_EXPERIMENT_METRICS))
    experiment_metrics["runtime"] = {
        "latency_stats_ns": latency_stats,
        "throughput_fps": summary["throughput_fps_from_mean_end_to_end"],
        "num_samples": num_samples,
        "warmup": args.warmup,
    }
    mixed_mean_ns = latency_stats["end_to_end_ns"]["mean_ns"]
    cpu_mean_ns = cpu_baseline["latency_ns"]["mean_ns"] if cpu_baseline else None
    experiment_metrics["trade_off"] = {
        "cpu_only_latency_ns": cpu_mean_ns,
        "fpga_only_latency_ns": mixed_mean_ns if len(FPGA_STAGES) == 5 else None,
        "mixed_hw_sw_latency_ns": mixed_mean_ns if GPP_STAGES else None,
        "latency_improvement_vs_cpu_percent": (
            100.0 * (cpu_mean_ns - mixed_mean_ns) / cpu_mean_ns
            if isinstance(cpu_mean_ns, (int, float)) and cpu_mean_ns else None
        ),
        "power_improvement_percent": None,
        "power_improvement_note": "CPU-only board power is not measured by this runner.",
        "best_partition_identified": None,
        "pareto_optimal": None,
    }
    experiment_metrics.setdefault("verification", {}).update({
        "fpga_execution": "passed",
        "golden_vector": verification["status"],
        "stage_status": verification["stage_status"],
    })
    experiment_metrics["verification"].setdefault("output_equivalence", {})[
        "software_vs_fpga"
    ] = verification["software_vs_fpga"]
    experiment_metrics.setdefault("implementation", {})["pynq_execution"] = "passed"
    experiment_metrics.setdefault("pipeline", {}).update({
        "status": "passed",
        "overall_success": True,
        "success_rate_percent_for_this_attempt": 100.0,
    })
    transfer_mean = (
        latency_stats["transfer_to_fpga_ns"]["mean_ns"]
        + latency_stats["transfer_from_fpga_ns"]["mean_ns"]
    )
    kernel_mean = latency_stats["fpga_kernel_ns"]["mean_ns"]
    gpp_mean = latency_stats["gpp_stages_ns"]["mean_ns"]
    compute_mean = kernel_mean + gpp_mean
    e2e_mean = latency_stats["end_to_end_ns"]["mean_ns"]
    partition_record = experiment_metrics.setdefault("partition", {})
    partition_record[
        "communication_to_computation_latency_ratio"
    ] = transfer_mean / compute_mean if compute_mean > 0 else None
    partition_record["cost"] = {
        "estimated_communication_overhead_ns": transfer_mean,
        "estimated_communication_overhead_percent": (
            100.0 * transfer_mean / e2e_mean if e2e_mean > 0 else None
        ),
        "estimated_compute_time_ns": compute_mean,
        "estimated_transfer_time_ns": transfer_mean,
        "estimated_overlap_ns": max(0.0, transfer_mean + compute_mean - e2e_mean),
        "source": "measured_pynq_component_means",
        "transfer_semantics": "shared_ddr_copy_and_cache_coherency_not_dma",
        "overlap_note": "Derived from component-sum minus measured end-to-end time; the current runner executes components sequentially.",
    }
    total_power_w = experiment_metrics.get("hardware", {}).get("total_power_w")
    mean_latency_ns = latency_stats["end_to_end_ns"]["mean_ns"]
    experiment_metrics["energy"] = {
        "measurement_source": (
            "vivado_estimated_total_power_times_measured_end_to_end_latency"
            if isinstance(total_power_w, (int, float)) else "unavailable"
        ),
        "power_w": total_power_w,
        "energy_per_inference_j": (
            total_power_w * mean_latency_ns * 1e-9
            if isinstance(total_power_w, (int, float)) else None
        ),
        "energy_per_sample_j": (
            total_power_w * mean_latency_ns * 1e-9
            if isinstance(total_power_w, (int, float)) else None
        ),
        "is_board_power_measurement": False,
    }
    evaluate_requirements(experiment_metrics)
    experiment_metrics.setdefault("llm_evaluation", {})[
        "hardware_objective_satisfaction_score"
    ] = experiment_metrics.get("requirements", {}).get("satisfaction_percent")
    metrics_json = output_dir / "experiment_metrics.json"
    experiment_metrics["updated_at_on_board"] = datetime.now().isoformat(timespec="seconds")
    metrics_json.write_text(json.dumps(experiment_metrics, indent=2), encoding="utf-8")
    summary["outputs"]["experiment_metrics_json"] = str(metrics_json)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_results(correct, confusion, first16, timings, num_samples, summary: dict | None, print_logits: bool, dataset_name: str):
    accuracy = 100.0 * correct / num_samples
    latency_stats = {name: stats(values) for name, values in timings.items()}
    mean_e2e_ns = latency_stats["end_to_end_ns"]["mean_ns"]
    fps = 1e9 / mean_e2e_ns if mean_e2e_ns > 0 else float("nan")
    print("\n" + "=" * 100)
    print("Hardware partition accuracy and latency")
    print("=" * 100)
    print(f"Design       : {DESIGN_NAME}")
    print(f"Partition    : {PARTITION_ID} | FPGA {FPGA_STAGES or ['none']} | GPP {GPP_STAGES or ['none']}")
    print(f"Dataset      : {dataset_name} test")
    print(f"Samples      : {num_samples}")
    print(f"Correct      : {correct}/{num_samples}")
    print(f"Incorrect    : {num_samples - correct}/{num_samples}")
    print(f"Accuracy     : {accuracy:.2f}%")
    print(f"Platform     : {platform.platform()}")
    print(f"Throughput   : {fps:,.2f} FPS, based on mean end-to-end latency")
    print("\nFirst 16 predictions for HLS/testbench validation:")
    for i, label, pred, logits in first16:
        ok = "OK" if label == pred else "MISS"
        if print_logits:
            print(f"  Sample {i:2d}: expected={label}, predicted={pred}, {ok}, logits={[float(x) for x in logits]}")
        else:
            print(f"  Sample {i:2d}: expected={label}, predicted={pred}, {ok}")
    print("\nLatency summary:")
    print(f"{'Component':28s}{'Mean':>14s}{'Min':>14s}{'Max':>14s}{'Median':>14s}{'P95':>14s}{'P99':>14s}")
    print("-" * 100)
    labels = {
        "transfer_to_fpga_ns": "Transfer to FPGA",
        "fpga_kernel_ns": "FPGA kernel",
        "transfer_from_fpga_ns": "Transfer from FPGA",
        "gpp_stages_ns": "CPU/GPP stages",
        "end_to_end_ns": "End-to-end",
    }
    for key, name in labels.items():
        r = latency_stats[key]
        print(
            f"{name:28s}{fmt_us(r['mean_ns']):>14s}{fmt_us(r['min_ns']):>14s}{fmt_us(r['max_ns']):>14s}"
            f"{fmt_us(r['median_ns']):>14s}{fmt_us(r['p95_ns']):>14s}{fmt_us(r['p99_ns']):>14s}"
        )
    print("\nConfusion matrix rows=true labels, columns=predicted labels:")
    print(confusion)
    if summary is not None:
        print("\nSaved results:")
        for _, path in summary["outputs"].items():
            print(f"  {path}")
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description="PYNQ accuracy + latency runner for CNNImageProc FPGA/GPP partitions")
    parser.add_argument("--design-name", default=DESIGN_NAME)
    parser.add_argument("--bitstream", default=BITSTREAM)
    parser.add_argument("--ip-name", default=IP_NAME)
    parser.add_argument("--weights", default="weights.hpp")
    parser.add_argument("--data-zip", default="FashionMNIST_data.zip")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    parser.add_argument("--output-dir", default=f"results_{DESIGN_NAME}")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--print-logits", action="store_true")
    parser.add_argument("--verification-samples", type=int, default=16)
    parser.add_argument("--cpu-baseline-samples", type=int, default=100)
    parser.add_argument("--skip-cpu-baseline", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    global LAST_OUTPUT_DIR
    LAST_OUTPUT_DIR = Path(args.output_dir)
    global DESIGN_NAME
    DESIGN_NAME = args.design_name
    print("Loading weights...")
    weights = load_weights(args.weights)
    print("Loading image-classification test set...")
    images, labels, dataset_name = load_mnist_test(args.data_zip)
    print("Loading overlay and allocating buffers...")
    _, ip, a, b = setup_overlay_and_buffers(args.bitstream, args.ip_name)
    num_samples = min(args.num_samples, len(labels))
    print(f"Verifying software/FPGA equivalence on {args.verification_samples} samples...")
    verification = verify_partition(
        ip, a, b, images, labels, weights, max(1, args.verification_samples)
    )
    if verification["status"] != "passed":
        for mismatch in verification["mismatches"][:3]:
            print(
                "Verification mismatch:", mismatch["sample_index"],
                "software=", mismatch["software_raw"],
                "fpga=", mismatch["fpga_partition_raw"],
            )
        raise AssertionError(
            f"Software/FPGA golden verification failed for {verification['mismatch_count']} samples"
        )
    cpu_baseline = None
    if not args.skip_cpu_baseline:
        print(f"Measuring CPU-only baseline on {args.cpu_baseline_samples} samples...")
        cpu_baseline = measure_cpu_baseline(
            images, weights, max(1, args.cpu_baseline_samples), min(args.warmup, 10)
        )
    print(f"Running {num_samples} {dataset_name} samples for {PARTITION_ID}...")
    correct, confusion, first16, timings, rows = measure_accuracy(
        ip, a, b, images, labels, weights, num_samples, args.warmup, args.progress_every
    )
    summary = None
    if not args.no_save:
        summary = save_results(
            Path(args.output_dir), correct, confusion, timings, rows, num_samples, args,
            verification, cpu_baseline, dataset_name
        )
    print_results(correct, confusion, first16, timings, num_samples, summary, args.print_logits, dataset_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = json.loads(json.dumps(STATIC_EXPERIMENT_METRICS))
        failure.setdefault("verification", {})["fpga_execution"] = "failed"
        failure.setdefault("implementation", {})["pynq_execution"] = "failed"
        failure.setdefault("pipeline", {}).update({"status": "failed", "overall_success": False})
        failure["board_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        destination = LAST_OUTPUT_DIR or Path(f"results_{DESIGN_NAME}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "experiment_metrics_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise
