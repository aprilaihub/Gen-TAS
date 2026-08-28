from __future__ import annotations

import base64
from html import escape
import os
from pathlib import Path
import sys

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GUI import gui_backend as backend


ASSET_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSET_DIR / "gen_tas_logo_bg_removed.png"
DEFAULT_TOP_K = 3
DEFAULT_STRATEGY_MODEL = os.getenv(
    "GENTAS_STRATEGY_MODEL",
    os.getenv("LAMDA_STRATEGY_MODEL", "gpt-5.6-sol"),
)
DEFAULT_EXPERIMENT_CONDITION = "GenTAS_RAG"


def logo_html(css_class: str = "sidebar-logo") -> str:
    data = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" class="{css_class}">'


def init_state() -> None:
    if "run_id" not in st.session_state:
        st.session_state.run_id = backend.default_run_id()
    if "new_run_id" not in st.session_state:
        st.session_state.new_run_id = st.session_state.run_id
    if "last_output" not in st.session_state:
        st.session_state.last_output = ""


def show_command_output() -> None:
    if st.session_state.last_output:
        with st.expander("Last command output", expanded=False):
            st.code(st.session_state.last_output, language="text")


def render_log_box(text: str) -> str:
    safe_text = escape(text or "No UI log yet.")
    return f'<div class="live-log"><pre>{safe_text}</pre></div>'


def resolve_repo_path(value: str) -> Path:
    return (backend.ROOT / value).resolve() if not value.startswith("/") else Path(value).expanduser().resolve()


def run_with_feedback(action, success_message: str) -> None:
    def update_live_log(text: str) -> None:
        st.session_state.last_output = text
        if "status_text_slot" in st.session_state:
            current = backend.load_status(st.session_state.run_id)
            st.session_state.status_text_slot.write(
                f"Status: {current.get('status_message') or current.get('status')}"
            )
            st.session_state.stage_text_slot.write(
                f"Stage: {current.get('stage_message') or current.get('current_stage') or 'Waiting to start'}"
            )
        if "log_code_slot" in st.session_state:
            st.session_state.log_code_slot.markdown(render_log_box(text), unsafe_allow_html=True)

    try:
        result = action(update_live_log)
    except backend.GuiBackendError as exc:
        st.error(str(exc))
        st.session_state.last_output = backend.read_log(st.session_state.run_id)
        update_live_log(st.session_state.last_output)
    else:
        st.session_state.last_output = result.output
        st.success(success_message)
        update_live_log(backend.read_log(st.session_state.run_id))


st.set_page_config(page_title="Gen-TAS", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
.block-container {padding-top: 1.4rem; padding-bottom: 1.4rem;}
div[data-testid="stColumn"] {min-width: 0;}
.sidebar-logo {display: block; width: 100%; max-width: 160px; margin: 0 auto 1.2rem auto;}
.brand-header {display: flex; align-items: center; gap: 1.6rem; margin-bottom: 1rem;}
.brand-logo {width: 208px; max-height: 146px; object-fit: contain;}
.brand-title {font-size: 3.35rem; font-weight: 750; line-height: 1.08; margin: 0;}
.brand-caption {color: #64748b; margin: .3rem 0 1.35rem 0; font-size: 1.4rem; line-height: 1.42; max-width: 86rem;}
div[data-testid="stWidgetLabel"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label {
  font-size: 1.1rem !important;
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stCaptionContainer"] {
  font-size: 1.1rem;
}
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)),
div[data-testid="stVerticalBlock"]:has(.marker-strategies):not(:has(div[data-testid="stVerticalBlock"] .marker-strategies)),
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)),
div[data-testid="stVerticalBlock"]:has(.marker-status):not(:has(div[data-testid="stVerticalBlock"] .marker-status)),
div[data-testid="stVerticalBlock"]:has(.marker-results):not(:has(div[data-testid="stVerticalBlock"] .marker-results)) {
  --panel-accent: #64748b;
  --panel-bg: #f8fafc;
  --panel-border: #cbd5e1;
  --panel-heading: #334155;
  border: 1px solid var(--panel-border);
  border-left-width: 5px;
  border-left-color: var(--panel-accent);
  border-radius: 8px;
  padding: 1.2rem;
  min-height: 110px;
  background: var(--panel-bg);
  box-shadow: 0 1px 5px rgba(15, 23, 42, 0.05);
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
}
.panel-title {
  font-size: 1.42rem;
  margin: 0 0 .8rem 0;
  color: var(--panel-heading);
  font-weight: 700;
}
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) {
  --panel-accent: #6366f1; --panel-border: #a5b4fc; --panel-bg: #f5f6fe; --panel-heading: #3730a3;
}
div[data-testid="stVerticalBlock"]:has(.marker-strategies):not(:has(div[data-testid="stVerticalBlock"] .marker-strategies)) {
  --panel-accent: #10b981; --panel-border: #86efac; --panel-bg: #f0fdf9; --panel-heading: #047857;
}
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)) {
  --panel-accent: #8b5cf6; --panel-border: #c4b5fd; --panel-bg: #f8f5ff; --panel-heading: #6d28d9;
}
div[data-testid="stVerticalBlock"]:has(.marker-status):not(:has(div[data-testid="stVerticalBlock"] .marker-status)) {
  --panel-accent: #64748b; --panel-border: #cbd5e1; --panel-bg: #f8fafc; --panel-heading: #334155;
}
div[data-testid="stVerticalBlock"]:has(.marker-results):not(:has(div[data-testid="stVerticalBlock"] .marker-results)) {
  --panel-accent: #0ea5e9; --panel-border: #7dd3fc; --panel-bg: #f0f9ff; --panel-heading: #0369a1;
}
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) button,
div[data-testid="stVerticalBlock"]:has(.marker-strategies):not(:has(div[data-testid="stVerticalBlock"] .marker-strategies)) button,
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)) button,
div[data-testid="stVerticalBlock"]:has(.marker-results):not(:has(div[data-testid="stVerticalBlock"] .marker-results)) button {
  border-color: var(--panel-accent) !important;
}
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) div[data-testid="stButton"] button,
div[data-testid="stVerticalBlock"]:has(.marker-strategies):not(:has(div[data-testid="stVerticalBlock"] .marker-strategies)) div[data-testid="stButton"] button,
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)) div[data-testid="stButton"] button,
div[data-testid="stVerticalBlock"]:has(.marker-results):not(:has(div[data-testid="stVerticalBlock"] .marker-results)) div[data-testid="stDownloadButton"] button {
  background: var(--panel-accent) !important;
  color: #ffffff !important;
  font-size: 1.1rem !important;
}
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) textarea,
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) input,
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) div[data-baseweb="select"] > div,
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)) input,
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)) div[data-baseweb="select"] > div {
  background: #ffffff !important;
  border-color: var(--panel-border) !important;
}
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) div[data-testid="stTextInput"] div,
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) div[data-testid="stTextArea"] div,
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) div[data-testid="stSelectbox"] div,
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)) div[data-testid="stTextInput"] div,
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)) div[data-testid="stSelectbox"] div {
  background: #ffffff !important;
}
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) div[data-testid="stElementContainer"],
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)) div[data-testid="stElementContainer"],
div[data-testid="stVerticalBlock"]:has(.marker-status):not(:has(div[data-testid="stVerticalBlock"] .marker-status)) div[data-testid="stElementContainer"] {
  box-sizing: border-box;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
}
div[data-testid="stVerticalBlock"]:has(.marker-input):not(:has(div[data-testid="stVerticalBlock"] .marker-input)) div[data-testid="stCaptionContainer"],
div[data-testid="stVerticalBlock"]:has(.marker-automation):not(:has(div[data-testid="stVerticalBlock"] .marker-automation)) div[data-testid="stCaptionContainer"],
div[data-testid="stVerticalBlock"]:has(.marker-status):not(:has(div[data-testid="stVerticalBlock"] .marker-status)) div[data-testid="stMarkdownContainer"] {
  box-sizing: border-box;
  width: calc(100% - 2.4rem) !important;
  max-width: calc(100% - 2.4rem) !important;
  overflow-wrap: anywhere;
}
div[data-testid="stTextArea"],
div[data-testid="stTextInput"],
div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"],
div[data-testid="stButton"],
div[data-testid="stDownloadButton"],
div[data-baseweb="select"] {
  box-sizing: border-box;
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
}
textarea {
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: 100% !important;
  resize: vertical !important;
}
div[data-testid="stVerticalBlock"]:has(.marker-results):not(:has(div[data-testid="stVerticalBlock"] .marker-results)) div[data-testid="stMetric"],
div[data-testid="stVerticalBlock"]:has(.marker-results):not(:has(div[data-testid="stVerticalBlock"] .marker-results)) div[data-testid="stMetricValue"],
div[data-testid="stVerticalBlock"]:has(.marker-results):not(:has(div[data-testid="stVerticalBlock"] .marker-results)) div[data-testid="stHorizontalBlock"] > div {
  background: #ffffff !important;
}
.strategy-card {
  border: 1px solid #d1d5db;
  border-radius: 8px;
  padding: .75rem .85rem;
  margin: .55rem 0;
  background: #ffffff;
}
.strategy-card strong {font-size: 1.16rem;}
.strategy-card.selected {border-color: #059669; background: #ecfdf5;}
.muted {color: #64748b; font-size: 1.1rem;}
.path-box {
  font-family: monospace;
  font-size: 1.08rem;
  padding: .45rem .6rem;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  overflow-wrap: anywhere;
}
.live-log {
  height: 180px;
  overflow-y: auto;
  display: flex;
  flex-direction: column-reverse;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  padding: .55rem .65rem;
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
}
.live-log pre {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 1.06rem;
  line-height: 1.35;
  color: #0f172a;
}
.download-row {
  display: flex;
  align-items: center;
  min-height: 2.4rem;
  font-size: 1.1rem;
}
textarea,
input,
div[data-baseweb="select"] {
  font-size: 1.12rem !important;
}
button {
  font-size: 1.1rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

init_state()

with st.sidebar:
    st.markdown(logo_html(), unsafe_allow_html=True)
    st.subheader("Gen-TAS")
    st.caption(
        "Generative AI for Task Allocation in FPGA-GPP Heterogeneous Systems. "
        "Transform a design request into FPGA/GPP partition strategies, hardware implementation artifacts, and PYNQ measurement files."
    )

existing_runs = backend.list_run_ids()
run_choices = ["new run"] + existing_runs

top_left, top_right = st.columns([2.1, 1], gap="large")
with top_left:
    st.markdown(
        f"""
<div class="brand-header">
  {logo_html("brand-logo")}
  <div>
    <div class="brand-title">Gen-TAS</div>
    <p class="brand-caption">Generative AI for Task Allocation in FPGA-GPP Heterogeneous Systems: from natural-language design request to FPGA/GPP strategy selection, Vivado hardware exports, and PYNQ hardware measurement files.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with top_right:
    choice = st.selectbox(
        "Run",
        run_choices,
        index=run_choices.index(st.session_state.run_id) if st.session_state.run_id in existing_runs else 0,
    )
    if choice == "new run":
        st.session_state.run_id = st.text_input("New run id", key="new_run_id")
    else:
        st.session_state.run_id = choice

run_id = st.session_state.run_id.strip()
initial_status = backend.load_status(run_id)
default_export_root = initial_status.get("export_root") or str(backend.EXPORT_ROOT.relative_to(backend.ROOT))
status = backend.refresh_status_from_files(run_id, export_root=resolve_repo_path(default_export_root))
recommendations = backend.load_recommendations(run_id)
selection = backend.load_selection(run_id)

input_col, strategy_col, auto_col = st.columns([1.05, 1.65, 1.2], gap="medium")
log_col, result_col = st.columns([1.05, 2.90], gap="medium")

with log_col:
    st.markdown('<div class="marker-status"></div><div class="panel-title">Status / Log</div>', unsafe_allow_html=True)
    st.session_state.status_text_slot = st.empty()
    st.session_state.stage_text_slot = st.empty()
    st.session_state.error_slot = st.empty()
    st.session_state.log_code_slot = st.empty()
    st.session_state.status_text_slot.write(
        f"Status: {status.get('status_message') or status.get('status')}"
    )
    st.session_state.stage_text_slot.write(
        f"Stage: {status.get('stage_message') or status.get('current_stage') or 'Waiting to start'}"
    )
    if status.get("last_error"):
        st.session_state.error_slot.error(status["last_error"])
    st.session_state.log_code_slot.markdown(render_log_box(backend.read_log(run_id)), unsafe_allow_html=True)

with input_col:
    st.markdown('<div class="marker-input"></div><div class="panel-title">1. Input</div>', unsafe_allow_html=True)
    request = st.text_area(
        "Request",
        value="Minimise end-to-end latency while keeping power reasonable",
        height=96,
    )
    goal = st.selectbox("Goal", ["latency", "power", "resource", "balanced"], index=0)
    llm_model = st.text_input("LLM model", value=DEFAULT_STRATEGY_MODEL)
    top_mode = st.selectbox("Top generation", ["deterministic", "llm"], index=0)
    pynq_mode = st.selectbox("PYNQ generation", ["deterministic", "llm"], index=0)
    source_dir = st.text_input("Source directory", value=str(backend.DEFAULT_SOURCE_DIR.relative_to(backend.ROOT)))
    st.caption("Generate Strategies reads the source files and requests valid LLM partition strategies.")
    try:
        preview_contract = backend.workload_contract_for_run(run_id, resolve_repo_path(source_dir))
    except Exception as exc:
        preview_contract = None
        st.warning(f"Workload mapping could not be inferred: {exc}")
    if preview_contract:
        with st.expander("Review workload mapping", expanded=False):
            st.dataframe(backend.workload_mapping_rows(preview_contract), use_container_width=True, hide_index=True)
            for warning in preview_contract.get("warnings", []):
                st.warning(warning)
    if st.button("Generate Strategies", type="primary", use_container_width=True):
        source_path = resolve_repo_path(source_dir)
        run_with_feedback(
            lambda on_output: backend.generate_strategies(
                run_id=run_id,
                request=request,
                goal=goal,
                source_dir=source_path,
                top_k=DEFAULT_TOP_K,
                model=llm_model.strip() or DEFAULT_STRATEGY_MODEL,
                experiment_condition=DEFAULT_EXPERIMENT_CONDITION,
                on_output=on_output,
            ),
            "Strategies generated.",
        )
        st.rerun()

with strategy_col:
    st.markdown('<div class="marker-strategies"></div><div class="panel-title">2. Strategies</div>', unsafe_allow_html=True)
    if not recommendations:
        st.caption("Generate strategies or select an existing run with recommendations.")
    else:
        selected_partition = status.get("selected_partition") or (selection or {}).get("selected_partition")
        visible_recommendations = recommendations.get("recommendations", [])[:DEFAULT_TOP_K]
        if len(recommendations.get("recommendations", [])) > DEFAULT_TOP_K:
            st.caption("Showing the highest-ranked strategies from this run.")
        for item in visible_recommendations:
            partition_id = item["partition_id"]
            selected = partition_id == selected_partition
            cls = "strategy-card selected" if selected else "strategy-card"
            st.markdown(
                f"""
<div class="{cls}">
  <strong>{item.get('rank')}. {partition_id}</strong><br>
  <span class="muted">{item.get('summary', '')}</span><br>
  <span class="muted">FPGA: {', '.join(item.get('fpga_subfunctions') or ['none'])}</span><br>
  <span class="muted">GPP: {', '.join(item.get('gpp_subfunctions') or ['none'])}</span>
</div>
""",
                unsafe_allow_html=True,
            )
            col_a, col_b = st.columns([1, 2])
            with col_a:
                if st.button("Select", key=f"select_{partition_id}", use_container_width=True):
                    run_with_feedback(
                        lambda on_output, partition_id=partition_id: backend.select_strategy_and_generate_top(
                            run_id=run_id,
                            partition_id=partition_id,
                            top_mode=top_mode,
                            top_model=llm_model.strip() or DEFAULT_STRATEGY_MODEL,
                            force=True,
                            on_output=on_output,
                        ),
                        "Strategy selected and top/testbench generated.",
                    )
                    st.rerun()
            with col_b:
                st.caption(item.get("expected_latency_impact", ""))
        # Strategy selection remains LLM-assisted; deterministic generation is
        # the controlled default, with LLM generation available as an ablation.

with auto_col:
    st.markdown('<div class="marker-automation"></div><div class="panel-title">3. Automation</div>', unsafe_allow_html=True)
    manifest_path = status.get("manifest_path")
    if manifest_path:
        st.caption("Selected strategy is ready for hardware generation.")
    else:
        st.caption("Select a strategy to generate the design files.")
    st.caption("Click 1. Full Hardware Build and Export first, then click 2. Generate PYNQ Measurement Script afterwards.")
    export_root_value = st.text_input("Export directory", value=default_export_root)
    export_root_path = resolve_repo_path(export_root_value)

    export_disabled = not manifest_path
    if st.button("1. Full Hardware Build and Export", type="primary", use_container_width=True, disabled=export_disabled):
        run_with_feedback(
            lambda on_output: backend.run_hardware_export(
                run_id=run_id,
                manifest_path=manifest_path,
                export_root=export_root_path,
                generation_mode=pynq_mode,
                model=llm_model.strip() or DEFAULT_STRATEGY_MODEL,
                on_output=on_output,
            ),
            "Hardware export completed.",
        )
        st.rerun()

    if st.button("2. Generate PYNQ Measurement Script", use_container_width=True, disabled=export_disabled):
        run_with_feedback(
            lambda on_output: backend.generate_pynq(
                run_id=run_id,
                manifest_path=manifest_path,
                export_root=export_root_path,
                force=True,
                on_output=on_output,
            ),
            "PYNQ script generated.",
        )
        st.rerun()

with result_col:
    st.markdown('<div class="marker-results"></div><div class="panel-title">Results</div>', unsafe_allow_html=True)
    summary = backend.export_summary_from_manifest(status.get("manifest_path"), export_root=export_root_path)
    if summary:
        metrics = summary.get("metrics", {})
        st.markdown("**Estimated from design builds and simulation**")
        if metrics:
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            m1.metric("Fmax", metrics.get("Fmax", "n/a"))
            m2.metric("LUT", metrics.get("LUT", "n/a"))
            m3.metric("FF", metrics.get("FF", "n/a"))
            m4.metric("DSP", metrics.get("DSP", "n/a"))
            m5.metric("BRAM", metrics.get("BRAM", "n/a"))
            m6.metric("Processor system power", metrics.get("Power_GPP", "n/a"))
            m7.metric("FPGA fabric power", metrics.get("Power_FPGA", "n/a"))
            st.caption(
                "Processor system includes the ARM/PS block baseline. "
                "FPGA fabric is the programmable logic portion estimated from Vivado."
            )
        else:
            st.caption("Run hardware build/export to populate Fmax, utilization, and power estimates.")

        st.markdown("**Hardware Test Downloads**")
        artifacts = summary.get("artifacts", [])
        if artifacts:
            for index, artifact in enumerate(artifacts):
                path = Path(artifact["path"])
                col_label, col_button = st.columns([2, 1])
                with col_label:
                    st.markdown(
                        f'<div class="download-row">{artifact["label"]}: '
                        f'<code>{artifact["filename"]}</code></div>',
                        unsafe_allow_html=True,
                    )
                with col_button:
                    st.download_button(
                        "Download",
                        data=path.read_bytes(),
                        file_name=artifact["filename"],
                        key=f"download_{index}_{artifact['filename']}",
                        use_container_width=True,
                    )
        else:
            st.caption("Exported `.bit`, `.hwh`, PYNQ script, and weight files will appear here after generation.")
    else:
        st.caption("Build/export results will appear here after generated design files exist.")
