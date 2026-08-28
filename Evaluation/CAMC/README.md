# CAMC Evaluation

This directory is the CAMC evaluation entry point. The synthesizable application and workload descriptor remain in
`Backend/examples/camc/`; CAMC uses the same validated Gen-TAS allocation,
RAG, generation, and backend pipeline as ImageProc Fashion-MNIST.

List CAMC placements without an API call:

```bash
python Evaluation/CAMC/Run.py --list-partitions
```

Create a deterministic smoke-test session:

```bash
python Evaluation/CAMC/Run.py \
  --request "Minimise latency while limiting communication" \
  --goal latency \
  --run-id smoke-camc \
  --deterministic
```

LLM/RAG runs use the same model and experiment flags documented by the shared
`Evaluation/CNNImageProc/Run.py` entry point.
