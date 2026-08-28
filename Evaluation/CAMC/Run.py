#!/usr/bin/env python3
"""Run the shared Gen-TAS evaluation pipeline with CAMC as the workload."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CAMC_SOURCE_DIR = ROOT / "Backend" / "examples" / "camc"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.CNNImageProc.Run import main as run_shared_pipeline


def camc_args(argv=None):
    """Supply CAMC's source directory for new sessions and partition listings."""
    args = list(sys.argv[1:] if argv is None else argv)
    has_session = any(arg == "--session" or arg.startswith("--session=") for arg in args)
    has_source = any(
        arg == "--source-dir" or arg.startswith("--source-dir=") for arg in args
    )
    if not has_session and not has_source:
        args.extend(("--source-dir", str(CAMC_SOURCE_DIR)))
    return args


def main(argv=None):
    return run_shared_pipeline(camc_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
