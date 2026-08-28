# Gen-TAS GUI

The Streamlit GUI drives the complete task-allocation flow: workload mapping,
LightCNN-KB retrieval, LLM strategy ranking, user selection, validated top-level
generation, HLS/Vivado execution, export, and PYNQ runner generation.

The controlled defaults are:

- strategy selection: GenTAS_RAG with active LightCNN evidence;
- task graph: deterministic and schema validated;
- top generation: deterministic;
- PYNQ generation: deterministic.

The GUI exposes deterministic and LLM generation modes. A single model field is
shared by strategy selection, top generation, and PYNQ generation so controlled
runs use one consistent model. Run the GUI from the repository root with
`./GUI/run_gui.sh`.

Alternative evaluation conditions such as `LLM_NoRAG` and
`Deterministic_Heuristic` remain available through the command-line runners.
Independent publication trials can be labelled with `--repetition-index`; GUI
runs use repetition 1.
