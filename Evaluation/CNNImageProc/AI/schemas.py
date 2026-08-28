"""Workload contract for CNNImageProc partition experiments.

This module deliberately mirrors the LightCNN schema shape while keeping the
CNN-specific dimensions and five-stage graph explicit.  It is the second
workload instance that can later be lifted into a generalized pipeline.
"""

from copy import deepcopy
import re


class SchemaError(ValueError):
    """Raised when a pipeline artifact does not satisfy its contract."""


STAGE_SPECS = {
    "S1": {
        "function": "conv1_feature_extract",
        "source": "src/sub_1_conv1_feature_extract.cpp",
        "input": {"length": 784, "shape_expression": "IMG_SIZE", "semantic": "image"},
        "output": {"length": 12544, "shape_expression": "CONV1_SIZE", "semantic": "conv1"},
    },
    "S2": {
        "function": "relu_pool1",
        "source": "src/sub_2_relu_pool1.cpp",
        "input": {"length": 12544, "shape_expression": "CONV1_SIZE", "semantic": "conv1"},
        "output": {"length": 3136, "shape_expression": "POOL1_SIZE", "semantic": "pool1"},
    },
    "S3": {
        "function": "conv2_feature_extract",
        "source": "src/sub_3_conv2_feature_extract.cpp",
        "input": {"length": 3136, "shape_expression": "POOL1_SIZE", "semantic": "pool1"},
        "output": {"length": 6272, "shape_expression": "CONV2_SIZE", "semantic": "conv2"},
    },
    "S4": {
        "function": "relu_pool2",
        "source": "src/sub_4_relu_pool2.cpp",
        "input": {"length": 6272, "shape_expression": "CONV2_SIZE", "semantic": "conv2"},
        "output": {"length": 1568, "shape_expression": "POOL2_SIZE", "semantic": "pool2"},
    },
    "S5": {
        "function": "dense_classifier",
        "source": "src/sub_5_dense_classifier.cpp",
        "input": {"length": 1568, "shape_expression": "POOL2_SIZE", "semantic": "pool2"},
        "output": {"length": 10, "shape_expression": "NUM_CLASSES", "semantic": "logits"},
    },
}

STAGE_ORDER = tuple(STAGE_SPECS)


REQUIRED_SOURCE_FILES = (
    "src/lib.hpp",
    "src/weights.hpp",
    "src/top.cpp",
    "src/tb.cpp",
    "src/sub_1_conv1_feature_extract.cpp",
    "src/sub_2_relu_pool1.cpp",
    "src/sub_3_conv2_feature_extract.cpp",
    "src/sub_4_relu_pool2.cpp",
    "src/sub_5_dense_classifier.cpp",
)


PARTITION_SPECS = {
    "ALL_GPP": {
        "fpga_subfunctions": [],
        "gpp_subfunctions": ["S1", "S2", "S3", "S4", "S5"],
        "summary": "CPU/GPP baseline; no FPGA hardware is generated.",
    },
    "ALL_FPGA": {
        "fpga_subfunctions": ["S1", "S2", "S3", "S4", "S5"],
        "gpp_subfunctions": [],
        "summary": "Current full CNNImageProc accelerator baseline.",
    },
    "S1_FPGA_REST_GPP": {
        "fpga_subfunctions": ["S1"],
        "gpp_subfunctions": ["S2", "S3", "S4", "S5"],
        "summary": "First convolution only on FPGA.",
    },
    "S1S2_FPGA_REST_GPP": {
        "fpga_subfunctions": ["S1", "S2"],
        "gpp_subfunctions": ["S3", "S4", "S5"],
        "summary": "First feature block on FPGA.",
    },
    "FEATURE_FPGA_DENSE_GPP": {
        "fpga_subfunctions": ["S1", "S2", "S3", "S4"],
        "gpp_subfunctions": ["S5"],
        "summary": "Both convolution/pooling feature blocks on FPGA, dense on GPP.",
    },
    "S3S5_FPGA_AFTER_BLOCK1_GPP": {
        "fpga_subfunctions": ["S3", "S4", "S5"],
        "gpp_subfunctions": ["S1", "S2"],
        "summary": "Second feature block and dense classifier on FPGA.",
    },
    "S3S4_FPGA_ONLY": {
        "fpga_subfunctions": ["S3", "S4"],
        "gpp_subfunctions": ["S1", "S2", "S5"],
        "summary": "Second convolution/pooling block only on FPGA.",
    },
    "DENSE_FPGA_ONLY": {
        "fpga_subfunctions": ["S5"],
        "gpp_subfunctions": ["S1", "S2", "S3", "S4"],
        "summary": "Dense classifier only on FPGA.",
    },
    "S1S3_FPGA_REST_GPP": {
        "fpga_subfunctions": ["S1", "S2", "S3"],
        "gpp_subfunctions": ["S4", "S5"],
        "summary": "Boundary after conv2, before pool2.",
    },
    "S2_FPGA_ONLY": {
        "fpga_subfunctions": ["S2"],
        "gpp_subfunctions": ["S1", "S3", "S4", "S5"],
        "summary": "First pooling stage only on FPGA; mainly a diagnostic split.",
    },
}


def _stage_index(stage):
    return int(stage[1:])


def _is_contiguous(stages):
    if not stages:
        return False
    indexes = [_stage_index(stage) for stage in stages]
    return indexes == list(range(indexes[0], indexes[-1] + 1))


def _stage_specs(workload_definition=None):
    return workload_definition["stages"] if workload_definition else STAGE_SPECS


def _stage_order(workload_definition=None):
    return tuple(_stage_specs(workload_definition))


def _normalize_stage_list(stages, field_name, workload_definition=None):
    if not isinstance(stages, list):
        raise SchemaError(f"{field_name} must be a list")
    normalized = []
    for stage in stages:
        if not isinstance(stage, str):
            raise SchemaError(f"{field_name} contains a non-string stage")
        stage = stage.strip().upper()
        if stage not in _stage_specs(workload_definition):
            raise SchemaError(f"{field_name} contains unknown stage: {stage}")
        if stage in normalized:
            raise SchemaError(f"{field_name} contains duplicate stage: {stage}")
        normalized.append(stage)
    return sorted(normalized, key=_stage_index)


def make_partition_id(fpga_subfunctions, workload_definition=None):
    """Create a stable readable ID for an LLM-proposed FPGA block."""
    fpga_subfunctions = _normalize_stage_list(
        fpga_subfunctions, "fpga_subfunctions", workload_definition
    )
    if not fpga_subfunctions:
        return "LLM_ALL_GPP"
    if fpga_subfunctions == list(_stage_order(workload_definition)):
        return "LLM_ALL_FPGA"
    return "LLM_FPGA_" + "_".join(fpga_subfunctions)


def build_partition_spec(
    partition_id,
    fpga_subfunctions,
    gpp_subfunctions=None,
    summary=None,
    workload_definition=None,
):
    """Validate and materialize one CNNImageProc allocation spec.

    LLM strategy generation may propose new stage groupings.  This function is
    the deterministic gate before any proposal can reach selection or hardware
    generation.
    """
    if not isinstance(partition_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", partition_id):
        raise SchemaError(f"invalid partition_id: {partition_id!r}")
    stage_order = _stage_order(workload_definition)
    fpga_subfunctions = _normalize_stage_list(
        fpga_subfunctions, "fpga_subfunctions", workload_definition
    )
    if gpp_subfunctions is None:
        gpp_subfunctions = [stage for stage in stage_order if stage not in fpga_subfunctions]
    else:
        gpp_subfunctions = _normalize_stage_list(
            gpp_subfunctions, "gpp_subfunctions", workload_definition
        )
    overlap = set(fpga_subfunctions) & set(gpp_subfunctions)
    if overlap:
        raise SchemaError(f"stages assigned to both FPGA and GPP: {sorted(overlap)}")
    assigned = set(fpga_subfunctions) | set(gpp_subfunctions)
    missing = set(stage_order) - assigned
    extra = assigned - set(stage_order)
    if missing or extra:
        raise SchemaError(
            f"stage assignment must cover exactly {list(stage_order)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    spec = {
        "partition_id": partition_id,
        "fpga_subfunctions": fpga_subfunctions,
        "gpp_subfunctions": gpp_subfunctions,
        "summary": summary or "LLM-proposed FPGA/GPP allocation.",
    }
    return complete_partition_spec(spec, workload_definition)


def complete_partition_spec(spec, workload_definition=None):
    """Add common derived hardware fields to an already validated spec."""
    spec = deepcopy(spec)
    partition_id = spec["partition_id"]
    spec["partition_id"] = partition_id
    stage_specs = _stage_specs(workload_definition)
    spec["workload"] = (
        workload_definition["workload"] if workload_definition else "cnn_imageproc_v2"
    )
    spec["top_function"] = (
        workload_definition["top_function"] if workload_definition else "cnn_imageproc_top"
    )
    spec["dtype"] = workload_definition.get("dtype", "mixed") if workload_definition else "fixed16_q4_12"
    spec["input"] = None
    spec["output"] = None
    spec["required_calls"] = [stage_specs[stage]["function"] for stage in spec["fpga_subfunctions"]]
    spec["hardware_generable"] = bool(spec["fpga_subfunctions"]) and _is_contiguous(
        spec["fpga_subfunctions"]
    )
    if spec["hardware_generable"]:
        first = spec["fpga_subfunctions"][0]
        last = spec["fpga_subfunctions"][-1]
        spec["input"] = {
            **stage_specs[first]["input"],
            "name": "a",
            "dtype": spec["dtype"],
        }
        spec["output"] = {
            **stage_specs[last]["output"],
            "name": "b",
            "dtype": spec["dtype"],
        }
    if workload_definition:
        spec["stage_specs"] = deepcopy(stage_specs)
        spec["generation"] = {
            "header_files": deepcopy(workload_definition["header_files"]),
            "top_includes": deepcopy(workload_definition["top_includes"]),
            "test_comparison": workload_definition.get("test_comparison", "exact"),
        }
    return spec


def get_partition_spec(partition_id, dynamic_specs=None, workload_definition=None):
    """Return an isolated partition specification.

    dynamic_specs is a session-stored mapping of LLM-proposed partition specs.
    Legacy PARTITION_SPECS remains available for deterministic fallback and old
    sessions.
    """
    if dynamic_specs and partition_id in dynamic_specs:
        return complete_partition_spec(dynamic_specs[partition_id], workload_definition)
    if workload_definition:
        partitions = workload_definition.get("partitions", {})
        if partition_id not in partitions:
            raise SchemaError(
                f"unsupported {workload_definition['workload']} partition: {partition_id}"
            )
        base = deepcopy(partitions[partition_id])
        base["partition_id"] = partition_id
        return complete_partition_spec(base, workload_definition)
    if partition_id not in PARTITION_SPECS:
        raise SchemaError(f"unsupported CNNImageProc partition: {partition_id}")
    base = deepcopy(PARTITION_SPECS[partition_id])
    base["partition_id"] = partition_id
    return complete_partition_spec(base)


def hardware_partitions(workload_definition=None):
    """Return partition IDs with a single contiguous FPGA boundary."""
    return [
        partition_id
        for partition_id in (
            workload_definition.get("partitions", {}) if workload_definition else PARTITION_SPECS
        )
        if get_partition_spec(
            partition_id, workload_definition=workload_definition
        )["hardware_generable"]
    ]
