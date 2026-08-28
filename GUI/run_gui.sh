#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

source activate.sh

if [ -f /opt/Xilinx/Vivado/2024.1/settings64.sh ]; then
    source /opt/Xilinx/Vivado/2024.1/settings64.sh
else
    echo "Warning: Vivado settings not found at /opt/Xilinx/Vivado/2024.1/settings64.sh"
fi

if [ -f /opt/Xilinx/Vitis_HLS/2024.1/settings64.sh ]; then
    source /opt/Xilinx/Vitis_HLS/2024.1/settings64.sh
else
    echo "Warning: Vitis HLS settings not found at /opt/Xilinx/Vitis_HLS/2024.1/settings64.sh"
fi

command -v vitis_hls >/dev/null 2>&1 || echo "Warning: vitis_hls is not on PATH"
command -v vivado >/dev/null 2>&1 || echo "Warning: vivado is not on PATH"

if ! command -v streamlit >/dev/null 2>&1; then
    echo "Streamlit is not installed in this environment."
    echo "Install it with: pip install streamlit"
    exit 1
fi

exec streamlit run GUI/app.py "$@"
