"""Deterministic CNNImageProc top/testbench generation."""

from Evaluation.CNNImageProc.AI.schemas import STAGE_SPECS, get_partition_spec


DECLARATIONS = """void conv1_feature_extract(
    data_t a[IMG_SIZE],
    const weight_t weights[CONV1_OUT_CH * IN_CH * K * K],
    const weight_t bias[CONV1_OUT_CH],
    data_t conv1_out[CONV1_SIZE]
);

void relu_pool1(
    data_t conv1_out[CONV1_SIZE],
    data_t pool1_out[POOL1_SIZE]
);

void conv2_feature_extract(
    data_t pool1_out[POOL1_SIZE],
    const weight_t weights[CONV2_OUT_CH * CONV1_OUT_CH * K * K],
    const weight_t bias[CONV2_OUT_CH],
    data_t conv2_out[CONV2_SIZE]
);

void relu_pool2(
    data_t conv2_out[CONV2_SIZE],
    data_t pool2_out[POOL2_SIZE]
);

void dense_classifier(
    data_t pool2_out[POOL2_SIZE],
    const weight_t weights[NUM_CLASSES * DENSE_IN_SIZE],
    const weight_t bias[NUM_CLASSES],
    data_t b[NUM_CLASSES]
);"""


def _descriptor_calls(spec, input_name="a", output_name="b"):
    lines = []
    current = input_name
    stages = spec["fpga_subfunctions"]
    for index, stage_id in enumerate(stages):
        stage = spec["stage_specs"][stage_id]
        is_last = index == len(stages) - 1
        target = output_name if is_last else stage["output"]["name"]
        if not is_last:
            boundary = stage["output"]
            lines.append(
                f"    {boundary['cpp_type']} {target}[{boundary['shape_expression']}];"
            )
        lines.append(
            "    " + stage["call"].format(input=current, output=target).rstrip(";") + ";"
        )
        current = target
    return lines


def _generate_descriptor_top(spec):
    includes = "\n".join(
        f'#include "{name}"' for name in spec["generation"]["top_includes"]
    )
    input_info = spec["input"]
    output_info = spec["output"]
    calls = "\n".join(_descriptor_calls(spec))
    pragmas = "\n".join(required_hls_pragmas(spec))
    return f'''{includes}

void {spec["top_function"]}(
    {input_info["cpp_type"]} a[{input_info["shape_expression"]}],
    {output_info["cpp_type"]} b[{output_info["shape_expression"]}]
) {{
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=a
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=b
#pragma HLS INTERFACE s_axilite port=a bundle=CTRL
#pragma HLS INTERFACE s_axilite port=b bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

{pragmas}

{calls}
}}
'''


def _generate_descriptor_testbench(partition_id, spec):
    includes = "\n".join(
        f'#include "{name}"' for name in spec["generation"]["top_includes"]
    )
    input_info = spec["input"]
    output_info = spec["output"]
    calls = "\n".join(_descriptor_calls(spec, "input", "expected"))
    test_expression = input_info.get("test_expression", "i")
    return f'''#include <iostream>
{includes}

void {spec["top_function"]}(
    {input_info["cpp_type"]} a[{input_info["shape_expression"]}],
    {output_info["cpp_type"]} b[{output_info["shape_expression"]}]
);

int main() {{
    {input_info["cpp_type"]} input[{input_info["shape_expression"]}];
    {output_info["cpp_type"]} actual[{output_info["shape_expression"]}];
    {output_info["cpp_type"]} expected[{output_info["shape_expression"]}];

    for (int i = 0; i < {input_info["shape_expression"]}; ++i) {{
        input[i] = {test_expression};
    }}
    for (int i = 0; i < {output_info["shape_expression"]}; ++i) {{
        actual[i] = 0;
        expected[i] = 0;
    }}

    {spec["top_function"]}(input, actual);
{calls}

    int mismatches = 0;
    for (int i = 0; i < {output_info["shape_expression"]}; ++i) {{
        if (actual[i] != expected[i]) {{
            std::cerr << "Mismatch at " << i
                      << ": actual=" << actual[i]
                      << ", expected=" << expected[i] << std::endl;
            ++mismatches;
        }}
    }}
    if (mismatches != 0) {{
        std::cerr << "FAIL: {partition_id} produced " << mismatches
                  << " mismatched outputs." << std::endl;
        return 1;
    }}
    std::cout << "PASS: {partition_id} matched all {output_info['length']} outputs."
              << std::endl;
    return 0;
}}
'''


def _stage_call(stage, source, target):
    if stage == "S1":
        return f"    conv1_feature_extract({source}, conv1_weights, conv1_bias, {target});"
    if stage == "S2":
        return f"    relu_pool1({source}, {target});"
    if stage == "S3":
        return f"    conv2_feature_extract({source}, conv2_weights, conv2_bias, {target});"
    if stage == "S4":
        return f"    relu_pool2({source}, {target});"
    if stage == "S5":
        return f"    dense_classifier({source}, dense_weights, dense_bias, {target});"
    raise ValueError(f"unknown stage: {stage}")


def _intermediate_name(stage):
    return {
        "S1": "conv1_out",
        "S2": "pool1_out",
        "S3": "conv2_out",
        "S4": "pool2_out",
    }[stage]


def _intermediate_decl(stage):
    info = STAGE_SPECS[stage]["output"]
    return f"    data_t {_intermediate_name(stage)}[{info['shape_expression']}];"


def required_hls_pragmas(spec):
    """Return the bounded CNN directives required for predictable HLS scheduling."""
    if "stage_specs" in spec:
        stages = spec["fpga_subfunctions"]
        functions = {
            stage: spec["stage_specs"][stage]["function"] for stage in stages
        }
        pragmas = []
        if "conv1_feature_extract" in functions.values():
            pragmas.extend([
                "#pragma HLS ARRAY_PARTITION variable=conv1_weights complete",
                "#pragma HLS ARRAY_PARTITION variable=conv1_bias complete",
            ])
        if "conv2_feature_extract" in functions.values():
            pragmas.append("#pragma HLS ARRAY_PARTITION variable=conv2_bias complete")
        if "dense_classifier" in functions.values():
            pragmas.append("#pragma HLS ARRAY_PARTITION variable=dense_bias complete")
        for stage in stages[:-1]:
            boundary = spec["stage_specs"][stage]["output"]
            if boundary["cpp_type"] == "data_t":
                pragmas.append(
                    "#pragma HLS ARRAY_PARTITION "
                    f"variable={boundary['name']} cyclic factor=16"
                )
        return pragmas
    stages = spec["fpga_subfunctions"]
    pragmas = []
    if "S1" in stages:
        pragmas.extend([
            "#pragma HLS ARRAY_PARTITION variable=conv1_weights complete",
            "#pragma HLS ARRAY_PARTITION variable=conv1_bias complete",
        ])
    if "S3" in stages:
        pragmas.append("#pragma HLS ARRAY_PARTITION variable=conv2_bias complete")
    if "S5" in stages:
        pragmas.append("#pragma HLS ARRAY_PARTITION variable=dense_bias complete")
    for stage in stages[:-1]:
        pragmas.append(
            f"#pragma HLS ARRAY_PARTITION variable={_intermediate_name(stage)} cyclic factor=16"
        )
    return pragmas


def _calls_for_stages(stages, input_name="a", output_name="b", with_hls_pragmas=False):
    lines = []
    current = input_name
    for index, stage in enumerate(stages):
        is_last = index == len(stages) - 1
        target = output_name if is_last else _intermediate_name(stage)
        if not is_last:
            lines.append(_intermediate_decl(stage))
            if with_hls_pragmas:
                lines.append(
                    f"#pragma HLS ARRAY_PARTITION variable={target} cyclic factor=16"
                )
        lines.append(_stage_call(stage, current, target))
        current = target
    return lines


def generate_top(partition_id, spec=None):
    """Generate a deterministic HLS top wrapper for one contiguous FPGA block."""
    spec = spec or get_partition_spec(partition_id)
    if not spec["hardware_generable"]:
        raise ValueError(f"{partition_id} has no single contiguous FPGA block")
    if "stage_specs" in spec:
        return _generate_descriptor_top(spec)
    input_expr = spec["input"]["shape_expression"]
    output_expr = spec["output"]["shape_expression"]
    stages = spec["fpga_subfunctions"]
    constant_pragmas = [
        pragma for pragma in required_hls_pragmas(spec)
        if not any(f"variable={_intermediate_name(stage)} " in pragma for stage in stages[:-1])
    ]
    calls = "\n".join(_calls_for_stages(stages, with_hls_pragmas=True))
    return f'''#include "lib.hpp"
#include "weights.hpp"

{DECLARATIONS}

void cnn_imageproc_top(
    data_t a[{input_expr}],
    data_t b[{output_expr}]
) {{
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=a
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=b
#pragma HLS INTERFACE s_axilite port=a bundle=CTRL
#pragma HLS INTERFACE s_axilite port=b bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

{chr(10).join(constant_pragmas)}

{calls}
}}
'''


def generate_testbench(partition_id, spec=None):
    """Generate a self-checking C simulation testbench for the hardware boundary."""
    spec = spec or get_partition_spec(partition_id)
    if not spec["hardware_generable"]:
        raise ValueError(f"{partition_id} has no single contiguous FPGA block")
    if "stage_specs" in spec:
        return _generate_descriptor_testbench(partition_id, spec)
    input_expr = spec["input"]["shape_expression"]
    output_expr = spec["output"]["shape_expression"]
    calls = "\n".join(_calls_for_stages(spec["fpga_subfunctions"], "input", "expected"))
    return f'''#include <cmath>
#include <iostream>
#include "lib.hpp"
#include "weights.hpp"

{DECLARATIONS}

void cnn_imageproc_top(
    data_t a[{input_expr}],
    data_t b[{output_expr}]
);

int main() {{
    data_t input[{input_expr}];
    data_t actual[{output_expr}];
    data_t expected[{output_expr}];

    for (int i = 0; i < {input_expr}; ++i) {{
        int pattern = (i % 17) - 8;
        input[i] = data_t(pattern) / data_t(16);
    }}
    for (int i = 0; i < {output_expr}; ++i) {{
        actual[i] = 0;
        expected[i] = 0;
    }}

    cnn_imageproc_top(input, actual);
{calls}

    int mismatches = 0;
    for (int i = 0; i < {output_expr}; ++i) {{
        double diff = std::fabs(actual[i].to_double() - expected[i].to_double());
        if (diff > 0.00025) {{
            std::cerr << "Mismatch at " << i
                      << ": actual=" << actual[i].to_double()
                      << ", expected=" << expected[i].to_double()
                      << std::endl;
            ++mismatches;
        }}
    }}
    if (mismatches != 0) {{
        std::cerr << "FAIL: {partition_id} produced " << mismatches
                  << " mismatched outputs." << std::endl;
        return 1;
    }}
    std::cout << "PASS: {partition_id} matched all {spec['output']['length']} outputs."
              << std::endl;
    return 0;
}}
'''


def generate_pseudocode(partition_id, spec=None):
    spec = spec or get_partition_spec(partition_id)
    lines = [f"# {partition_id}", ""]
    lines.append(f"FPGA stages: {', '.join(spec['fpga_subfunctions'])}")
    lines.append(f"GPP stages: {', '.join(spec['gpp_subfunctions']) or 'none'}")
    lines.append("")
    stage_specs = spec.get("stage_specs", STAGE_SPECS)
    for stage in spec["fpga_subfunctions"]:
        lines.append(f"- Call `{stage_specs[stage]['function']}`")
    lines.append("")
    lines.append(
        "Prior evidence is used for qualitative strategy structure only; "
        "metrics must come from this generated design."
    )
    return "\n".join(lines) + "\n"
