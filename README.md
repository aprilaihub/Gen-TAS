# Gen-TAS

Gen-TAS performs LLM-assisted hardware/software task allocation for FPGA-GPP
systems. It integrates reusable HLS, Vivado, artifact-export, and PYNQ backend
stages with validated task graphs, retrieval-augmented partition
recommendations, and reproducible implementation.

## Included workloads

- **ImageProc Fashion-MNIST:** five stages comprising two convolution/pooling
  blocks and a dense classifier.
- **CAMC:** three stages for axis quantization, constellation-histogram
  generation, and modulation-template scoring.
- **LightCNN knowledge base:** the active, task and partition evidence
  used by deterministic retrieval.

## Controlled workflow

The recommended evaluation path is:

1. Deterministically construct and validate the workload task graph.
2. Retrieve relevant LightCNN evidence and ask the selected LLM to rank valid
   contiguous FPGA/GPP allocations.
3. Let the user select a strategy, or use rank 1 under a declared policy.
4. Generate `top.cpp`, its self-checking testbench, I/O metadata, and backend
   handoff deterministically.
5. Run Vitis HLS, RTL co-simulation, Vivado implementation, and artifact export.
6. Generate a deterministic PYNQ runner.

The GUI also exposes an `llm` generation mode for experiments that evaluate
LLM-written top-level or PYNQ code. One model selection is shared by all LLM
stages. Generated code is validated before it reaches the backend. Keeping
deterministic generation as the default isolates the quality of LLM partition
decisions from code-generation variability. The backend retains `auto` mode for
command-line compatibility, but it is hidden from the GUI.

## Requirements

- Python 3.9 or newer
- Vitis HLS and Vivado 2024.1 for hardware generation
- A supported LLM provider key for `GenTAS_RAG` and `LLM_NoRAG`
- No API key for the deterministic heuristic condition

Create a local environment:

```bash
cp .env.example .env
# Add only the provider keys needed for your run.
source activate.sh
```

The setup script creates `venv/` and installs `requirements.txt` on first use.
Credentials are read from environment variables or the ignored `.env` file.

## Run the GUI

```bash
./GUI/run_gui.sh --server.port 8510 --server.address 0.0.0.0 --server.headless true
```

Open `http://localhost:8510`. The default application is ImageProc
Fashion-MNIST. To evaluate CAMC, set the source directory to
`Backend/examples/camc`.

The GUI uses GenTAS_RAG strategy selection followed by deterministic top and
PYNQ generation. Alternative evaluation conditions remain available through the
command-line runners. Full HLS and Vivado stages require the Xilinx tools and
ZCU104 target support installed on the host.

Use `--repetition-index N` with a command-line evaluation runner to label
independent publication trials. GUI runs use repetition 1.

## Command-line smoke checks

List CAMC partitions without an API call:

```bash
python Evaluation/CAMC/Run.py --list-partitions
```

Create a deterministic ImageProc session and top wrapper:

```bash
python Evaluation/CNNImageProc/Run.py \
  --request "Minimise latency while limiting communication" \
  --goal latency \
  --source-dir Backend/examples/cnn_imageproc_fashion \
  --run-id smoke-imageproc \
  --deterministic

python Evaluation/CNNImageProc/Run.py \
  --session Evaluation/CNNImageProc/Sessions/smoke-imageproc \
  --select FEATURE_FPGA_DENSE_GPP \
  --generate-top \
  --top-mode deterministic
```

## Repository layout

- `GUI/`: Streamlit interface and subprocess adapter.
- `Evaluation/CNNImageProc/AI/`: task-graph contracts, retrieval/prompt logic,
  allocation validation, deterministic generators, and optional LLM generation.
- `Evaluation/CAMC/`: CAMC-facing wrapper around the shared evaluation pipeline.
- `Evaluation/LightCNN/KnowledgeBase/active/`: allocation evidence consumed by
  RAG.
- `Backend/examples/`: source applications and stage implementations.
- `Backend/`: Vitis HLS, Vivado, export, and PYNQ generation backend.
- `LLM_Interface/`: provider-neutral OpenAI, Google, and Anthropic client.

## Authorship and citation

This repository contains the Gen-TAS v1 code prepared by Mary Kong,
August 2026.

It accompanies the publication **"Gen-TAS: A Generative AI-Aided
Hardware-Software Task Allocation Framework for FPGA-GPP Heterogeneous
Systems"** by Mary Kong, Yuqin Zhao, Semih Vazgecen, Cristian Sestito, and
Themis Prodromakis. When using this code, please cite the Gen-TAS publication.

## License

Gen-TAS is licensed under the Apache License, Version 2.0. See
[LICENSE](LICENSE) for the project license notice.
