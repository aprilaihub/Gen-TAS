"""Build the backend handoff for generated CNNImageProc partitions."""

import re

from Evaluation.CNNImageProc.AI.schemas import STAGE_SPECS, get_partition_spec
from Evaluation.CNNImageProc.AI.session_store import sha256_file


def _slug(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "run"


def _source_record(source_manifest, name, subfunction=None):
    record = source_manifest["files"][name]
    result = {
        "name": name,
        "path": record["path"],
        "sha256": record["sha256"],
    }
    if subfunction:
        result["subfunction"] = subfunction
    return result


def build_backend_handoff(
    *,
    run_id,
    partition_id,
    source_manifest,
    top_path,
    testbench_path,
    io_mapping_path,
    partition_spec=None,
    experiment_metrics_path=None,
):
    """Create a strategy-specific, hash-grounded backend input contract."""
    spec = partition_spec or get_partition_spec(partition_id)
    stage_specs = spec.get("stage_specs", STAGE_SPECS)
    fpga_sources = [
        _source_record(source_manifest, stage_specs[stage]["source"], stage)
        for stage in spec["fpga_subfunctions"]
    ]
    gpp_sources = [
        _source_record(source_manifest, stage_specs[stage]["source"], stage)
        for stage in spec["gpp_subfunctions"]
    ]
    workload = spec["workload"]
    design_name = f"{_slug(workload)}_{_slug(run_id)}_{_slug(partition_id)}"
    design_name = re.sub(r"_+", "_", design_name)
    return {
        "schema_version": 1,
        "workload": workload,
        "run_id": run_id,
        "partition_id": partition_id,
        "design_name": design_name,
        "top_function": spec["top_function"],
        "fpga_subfunctions": spec["fpga_subfunctions"],
        "gpp_subfunctions": spec["gpp_subfunctions"],
        "required_helper_calls": spec["required_calls"],
        "hardware_boundary": {
            "input": spec["input"],
            "output": spec["output"],
        },
        "experiment_metrics_path": (
            str(experiment_metrics_path) if experiment_metrics_path else None
        ),
        "generated_top": {
            "path": str(top_path),
            "sha256": sha256_file(top_path),
        },
        "generated_testbench": {
            "path": str(testbench_path),
            "sha256": sha256_file(testbench_path),
        },
        "io_mapping": {
            "path": str(io_mapping_path),
            "sha256": sha256_file(io_mapping_path),
        },
        "header_files": [
            _source_record(source_manifest, name)
            for name in spec.get("generation", {}).get(
                "header_files", ["src/lib.hpp", "src/weights.hpp"]
            )
        ],
        "fpga_sources": fpga_sources,
        "gpp_sources": gpp_sources,
        "hls": {
            "part": "xczu7ev-ffvc1156-2-e",
            "clock_period_ns": 10.0,
            "clock_uncertainty_ns": 1.25,
            "run_csim": True,
            "run_csynth": True,
            "run_cosim": True,
            "run_export": True,
        },
        "vivado": {
            "part": "xczu7ev-ffvc1156-2-e",
            "board_part": "xilinx.com:zcu104:part0:1.1",
            "bd_name": "design_1",
            "ip_instance": f"{spec['top_function']}_0",
        },
        "pynq": {
            "mode": "modular",
            "request": (
                f"Implement the selected {workload} {partition_id} allocation. "
                f"The single fused FPGA IP executes stages "
                f"{', '.join(spec['fpga_subfunctions'])}; execute stages "
                f"{', '.join(spec['gpp_subfunctions']) or 'none'} on the GPP. "
                "Measure transfer, FPGA, GPP, and end-to-end components. "
                "Do not infer numeric workload metrics from prior evidence."
            ),
        },
    }
