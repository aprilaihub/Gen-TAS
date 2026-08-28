from pathlib import Path

from Evaluation.CNNImageProc.AI.generators import generate_testbench, generate_top
from Evaluation.CNNImageProc.AI.schemas import get_partition_spec
from Evaluation.CNNImageProc.AI.workload_contract import build_workload_contract
from Evaluation.CNNImageProc.AI.workload_definition import load_workload_definition
from Evaluation.CAMC.Run import CAMC_SOURCE_DIR, camc_args


ROOT = Path(__file__).resolve().parents[1]


def test_imageproc_contract_and_deterministic_top():
    source = ROOT / "Backend" / "examples" / "cnn_imageproc_fashion"
    definition = load_workload_definition(source, required=True)
    contract = build_workload_contract(source)
    assert contract["call_order"] == ["S1", "S2", "S3", "S4", "S5"]

    spec = get_partition_spec(
        "FEATURE_FPGA_DENSE_GPP", workload_definition=definition
    )
    top = generate_top("FEATURE_FPGA_DENSE_GPP", spec)
    testbench = generate_testbench("FEATURE_FPGA_DENSE_GPP", spec)
    assert "conv1_feature_extract" in top
    assert "relu_pool2" in top
    assert "dense_classifier" not in top.split("void cnn_imageproc_top", 1)[1]
    assert "PASS: FEATURE_FPGA_DENSE_GPP" in testbench


def test_camc_contract_and_deterministic_top():
    source = ROOT / "Backend" / "examples" / "camc"
    definition = load_workload_definition(source, required=True)
    contract = build_workload_contract(source)
    assert contract["call_order"] == ["S1", "S2", "S3"]

    spec = get_partition_spec("A_GPP_BC_FPGA", workload_definition=definition)
    top = generate_top("A_GPP_BC_FPGA", spec)
    testbench = generate_testbench("A_GPP_BC_FPGA", spec)
    assert "matrix_generation(a, histogram);" in top
    assert "array_product(histogram, b);" in top
    assert "PASS: A_GPP_BC_FPGA" in testbench


def test_camc_evaluation_wrapper_defaults_to_camc_source():
    args = camc_args(["--list-partitions"])
    assert args[-2:] == ["--source-dir", str(CAMC_SOURCE_DIR)]
    assert camc_args(["--session", "/tmp/example"]) == [
        "--session",
        "/tmp/example",
    ]


def test_active_lightcnn_kb_is_present():
    kb = ROOT / "Evaluation" / "LightCNN" / "KnowledgeBase" / "active" / "lightcnn_evidence.json"
    assert kb.is_file()
    assert '"profile_count": 8' in kb.read_text(encoding="utf-8")
