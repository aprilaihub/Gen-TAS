#!/usr/bin/env python3
"""Export CNNImageProc dense weights from weights.hpp for PYNQ scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


def extract_weight_array(text: str, name: str, expected_count: int) -> list[float]:
    pattern = (
        rf"const\s+weight_t\s+{re.escape(name)}\s*\[[^\]]+\]\s*=\s*"
        rf"\{{(.*?)\}};"
    )
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"could not find {name} in weights header")
    values = [
        float(value)
        for value in re.findall(
            r"weight_t\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\)",
            match.group(1),
            re.IGNORECASE,
        )
    ]
    if len(values) != expected_count:
        raise ValueError(f"{name} expected {expected_count} values, found {len(values)}")
    return values


def to_q4_12(values: list[float]) -> np.ndarray:
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (1 << 12))
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def export_dense_weights(weights_hpp: Path, output_dir: Path) -> tuple[Path, Path]:
    text = weights_hpp.read_text(encoding="utf-8")
    dense_weights = to_q4_12(extract_weight_array(text, "dense_weights", 10 * 1568))
    dense_bias = to_q4_12(extract_weight_array(text, "dense_bias", 10))

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "dense_weights.npy"
    bias_path = output_dir / "dense_bias.npy"
    np.save(weights_path, dense_weights.reshape(10, 1568))
    np.save(bias_path, dense_bias)
    return weights_path, bias_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export dense_weights.npy and dense_bias.npy from CNNImageProc weights.hpp."
    )
    parser.add_argument("--weights-hpp", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        weights_path, bias_path = export_dense_weights(args.weights_hpp, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(f"DENSE_WEIGHTS: {weights_path}")
    print(f"DENSE_BIAS:    {bias_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
