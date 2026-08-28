"""Validate a selected generation manifest and prepare backend inputs."""

import hashlib
import json
from pathlib import Path
import re


class ManifestError(RuntimeError):
    """Raised when a generation manifest is incomplete, stale, or unsafe."""


LIGHTCNN_PARTITION_ASSIGNMENTS = {
    "A_FPGA_BC_GPP": (["A"], ["B", "C"]),
    "B_FPGA_AC_GPP": (["B"], ["A", "C"]),
    "C_FPGA_AB_GPP": (["C"], ["A", "B"]),
    "AB_FPGA_C_GPP": (["A", "B"], ["C"]),
    "A_GPP_BC_FPGA": (["B", "C"], ["A"]),
    "ABC_FPGA": (["A", "B", "C"], []),
}

LIGHTCNN_SUBFUNCTION_FILES = {"A": "sub_a.cpp", "B": "sub_b.cpp", "C": "sub_c.cpp"}


def _read_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"JSON artifact must contain an object: {path}")
    return payload


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_path(record, label):
    if not isinstance(record, dict):
        raise ManifestError(f"{label} must be an object")
    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ManifestError(f"{label} requires path and sha256 strings")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ManifestError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ManifestError(f"{label} hash does not match its handoff record: {path}")
    return path


def _tcl_quote(value):
    text = str(value)
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )
    return f'"{escaped}"'


def _tcl_list(paths):
    return "[list " + " ".join(_tcl_quote(path) for path in paths) + "]"


def _write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def load_handoff(generation_manifest_path):
    """Load and fully verify a generation manifest and its backend handoff."""
    manifest_path = Path(generation_manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ManifestError("generation manifest schema_version must be 1")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts.get("backend_handoff"):
        raise ManifestError("generation manifest has no backend_handoff artifact")
    handoff_path = Path(artifacts["backend_handoff"]).expanduser().resolve()
    if not handoff_path.is_file():
        raise ManifestError(f"backend handoff does not exist: {handoff_path}")
    expected_handoff_hash = manifest.get("backend_handoff_sha256")
    if _sha256(handoff_path) != expected_handoff_hash:
        raise ManifestError("backend handoff hash does not match generation manifest")

    handoff = _read_json(handoff_path)
    if handoff.get("schema_version") != 1:
        raise ManifestError("backend handoff schema_version must be 1")
    if handoff.get("partition_id") != manifest.get("partition_id"):
        raise ManifestError("partition differs between generation manifest and handoff")
    partition_id = handoff["partition_id"]
    workload = handoff.get("workload", "lightcnn")
    if str(workload).lower() == "lightcnn" and partition_id in LIGHTCNN_PARTITION_ASSIGNMENTS:
        expected_fpga, expected_gpp = LIGHTCNN_PARTITION_ASSIGNMENTS[partition_id]
    else:
        expected_fpga = handoff.get("fpga_subfunctions")
        expected_gpp = handoff.get("gpp_subfunctions")
        if not isinstance(expected_fpga, list) or not expected_fpga:
            raise ManifestError(f"unsupported or empty FPGA hardware partition: {partition_id}")
        if not isinstance(expected_gpp, list):
            raise ManifestError("handoff gpp_subfunctions must be a list")
    if handoff.get("fpga_subfunctions") != expected_fpga:
        raise ManifestError("handoff FPGA subfunctions do not match the selected partition")
    if handoff.get("gpp_subfunctions") != expected_gpp:
        raise ManifestError("handoff GPP subfunctions do not match the selected partition")
    design_name = handoff.get("design_name")
    if not isinstance(design_name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", design_name):
        raise ManifestError("handoff design_name must use lowercase letters, numbers, and underscores")

    top = _verified_path(handoff.get("generated_top"), "generated_top")
    testbench = _verified_path(handoff.get("generated_testbench"), "generated_testbench")
    io_mapping = _verified_path(handoff.get("io_mapping"), "io_mapping")
    if _read_json(io_mapping).get("partition_id") != partition_id:
        raise ManifestError("I/O mapping does not match the selected partition")
    if _sha256(top) != manifest.get("top_sha256"):
        raise ManifestError("top.cpp hash differs between manifest and handoff")
    if _sha256(testbench) != manifest.get("testbench_sha256"):
        raise ManifestError("tb.cpp hash differs between manifest and handoff")

    headers = [
        _verified_path(record, f"header_files[{index}]")
        for index, record in enumerate(handoff.get("header_files", []))
    ]
    fpga_records = handoff.get("fpga_sources", [])
    gpp_records = handoff.get("gpp_sources", [])
    if [record.get("subfunction") for record in fpga_records] != expected_fpga:
        raise ManifestError("FPGA source assignments do not match the selected partition")
    if [record.get("subfunction") for record in gpp_records] != expected_gpp:
        raise ManifestError("GPP source assignments do not match the selected partition")
    for record in [*fpga_records, *gpp_records]:
        subfunction = record.get("subfunction")
        expected_name = LIGHTCNN_SUBFUNCTION_FILES.get(subfunction)
        if expected_name is not None and record.get("name") != expected_name:
            raise ManifestError(f"source filename does not match subfunction {subfunction}")
    fpga_sources = [
        _verified_path(record, f"fpga_sources[{index}]")
        for index, record in enumerate(fpga_records)
    ]
    gpp_sources = [
        _verified_path(record, f"gpp_sources[{index}]")
        for index, record in enumerate(gpp_records)
    ]
    if not headers or not fpga_sources:
        raise ManifestError("handoff requires a header and at least one FPGA source")

    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "handoff_path": handoff_path,
        "handoff": handoff,
        "top": top,
        "testbench": testbench,
        "io_mapping": io_mapping,
        "headers": headers,
        "fpga_sources": fpga_sources,
        "gpp_sources": gpp_sources,
        "workload": workload,
    }


def _render_hls_config(context, backend_dir):
    handoff = context["handoff"]
    design_name = handoff["design_name"]
    hls = handoff["hls"]
    project_name = f"{design_name}_hls"
    project_dir = backend_dir / "build" / "hls" / project_name
    include_dirs = sorted({str(path.parent) for path in context["headers"]})
    cflags = " ".join(f"-I{path}" for path in include_dirs)
    source_files = [context["top"], *context["fpga_sources"]]
    return f'''# Auto-generated from {context["manifest_path"]}
dict set CONFIG PROJECT_NAME {_tcl_quote(project_name)}
dict set CONFIG PROJECT_DIR {_tcl_quote(project_dir)}
dict set CONFIG SOLUTION_NAME "solution1"
dict set CONFIG PART {_tcl_quote(hls["part"])}
dict set CONFIG TOP_FUNCTION {_tcl_quote(handoff["top_function"])}
dict set CONFIG SOURCE_DIR {_tcl_quote(context["headers"][0].parent)}
dict set CONFIG SOURCE_FILES {_tcl_list(source_files)}
dict set CONFIG TB_FILES {_tcl_list([context["testbench"]])}
dict set CONFIG SOURCE_CFLAGS {_tcl_quote(cflags)}
dict set CONFIG TB_CFLAGS {_tcl_quote(cflags)}
dict set CONFIG CLOCK_PERIOD_NS {hls["clock_period_ns"]}
dict set CONFIG CLOCK_UNCERTAINTY_NS {hls["clock_uncertainty_ns"]}
dict set CONFIG RUN_CSIM {int(bool(hls["run_csim"]))}
dict set CONFIG RUN_CSYNTH {int(bool(hls["run_csynth"]))}
dict set CONFIG RUN_COSIM {int(bool(hls["run_cosim"]))}
dict set CONFIG RUN_EXPORT {int(bool(hls["run_export"]))}
dict set CONFIG EXPORT_FORMAT "ip_catalog"
dict set CONFIG RTL_LANGUAGE "verilog"
dict set CONFIG IP_VERSION "1.0"
dict set CONFIG IP_DISPLAY_NAME {_tcl_quote(design_name + "_IP")}
dict set CONFIG IP_DESCRIPTION {_tcl_quote("Generated " + handoff.get("workload", "LightCNN") + " partition " + handoff["partition_id"])}
''', project_name, project_dir


def _render_vivado_config(context, backend_dir, hls_project_dir):
    handoff = context["handoff"]
    design_name = handoff["design_name"]
    vivado = handoff["vivado"]
    project_name = f"{design_name}_vivado"
    project_dir = backend_dir / "build" / "vivado" / project_name
    ip_repo = hls_project_dir / "solution1" / "impl" / "ip"
    instance = vivado["ip_instance"]
    top_function = handoff["top_function"]
    return f'''# Auto-generated from {context["manifest_path"]}
dict set CONFIG PROJECT_NAME {_tcl_quote(project_name)}
dict set CONFIG PROJECT_DIR {_tcl_quote(project_dir)}
dict set CONFIG PART {_tcl_quote(vivado["part"])}
dict set CONFIG BOARD_PART {_tcl_quote(vivado["board_part"])}
dict set CONFIG BD_NAME {_tcl_quote(vivado["bd_name"])}
dict set CONFIG IP_REPO {_tcl_quote(ip_repo)}
dict set CONFIG PS_NAME "zynq_ultra_ps_e_0"
dict set CONFIG RESET_NAME "proc_sys_reset_0"
dict set CONFIG CTRL_IC_NAME "axi_interconnect_0"
dict set CONFIG PS_CTRL_PORT "M_AXI_HPM0_FPD"
dict set CONFIG PS_CTRL_ACLK "maxihpm0_fpd_aclk"
dict set CONFIG DEFAULT_PL_CLK "pl_clk0"
dict set CONFIG DEFAULT_PL_RESET "pl_resetn0"
dict set CONFIG HLS_SPECS {_tcl_quote(f"{instance}|xilinx.com:hls:{top_function}:1.0|s_axi_CTRL|m_axi_a,m_axi_b|ap_clk|ap_rst_n|")}
dict set CONFIG DEFAULT_PS_DDR_PORT "S_AXI_HP0_FPD"
dict set CONFIG AXI_MM_MAP {_tcl_quote(f"{instance}.m_axi_a=S_AXI_HP0_FPD,{instance}.m_axi_b=S_AXI_HP0_FPD")}
dict set CONFIG PS_DDR_ACLK_MAP "S_AXI_HP0_FPD=saxihp0_fpd_aclk,S_AXI_HP1_FPD=saxihp1_fpd_aclk,S_AXI_HP2_FPD=saxihp2_fpd_aclk,S_AXI_HP3_FPD=saxihp3_fpd_aclk,S_AXI_HPC0_FPD=saxihpc0_fpd_aclk,S_AXI_HPC1_FPD=saxihpc1_fpd_aclk"
dict set CONFIG PS_ENABLE_PROPS "CONFIG.PSU__USE__M_AXI_GP0=1,CONFIG.PSU__USE__M_AXI_GP1=0,CONFIG.PSU__USE__S_AXI_GP2=1"
dict set CONFIG AXIS_LINKS ""
dict set CONFIG CLOCK_MAP ""
dict set CONFIG RESET_MAP ""
dict set CONFIG IRQ_CONCAT_NAME "xlconcat_irq"
dict set CONFIG PS_IRQ_PORT "pl_ps_irq0"
dict set CONFIG JOBS 16
dict set CONFIG RUN_SYNTH 1
dict set CONFIG RUN_IMPL 1
''', project_name, project_dir, ip_repo


def prepare_backend_inputs(generation_manifest_path):
    """Validate the handoff and materialize deterministic HLS/Vivado configs."""
    context = load_handoff(generation_manifest_path)
    backend_dir = context["manifest_path"].parent / "backend"
    hls_text, _, hls_project_dir = _render_hls_config(context, backend_dir)
    vivado_text, _, _, ip_repo = _render_vivado_config(context, backend_dir, hls_project_dir)
    hls_config = backend_dir / "configs" / "hls.tcl"
    vivado_config = backend_dir / "configs" / "vivado.tcl"
    _write_text(hls_config, hls_text)
    _write_text(vivado_config, vivado_text)

    semantic_sources = [
        context["top"],
        *context["fpga_sources"],
        *context["gpp_sources"],
        *context["headers"],
    ]
    return {
        **context,
        "backend_dir": backend_dir,
        "hls_config": hls_config,
        "vivado_config": vivado_config,
        "component_xml": ip_repo / "component.xml",
        "semantic_sources": list(dict.fromkeys(semantic_sources)),
        "pynq_mode": context["handoff"]["pynq"]["mode"],
        "pynq_request": context["handoff"]["pynq"]["request"],
    }
