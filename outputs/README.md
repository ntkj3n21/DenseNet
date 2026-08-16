# Experiment Outputs

This directory is reserved for transient or future runtime outputs that should
not be treated as committed research evidence by default.

The completed portfolio benchmark uses a clearer separation of artifacts.

`results/` contains the lightweight, verified artifacts committed to Git,
including:

```text
metrics.csv
summary.json
summary.csv
benchmark_summary.csv
figures/
test_metrics.json
```

Large runtime artifacts such as model checkpoints are intentionally kept
outside the Git repository. During the completed experiments, checkpoints were
stored separately in the runtime/Google Drive environment.

The tracked artifacts under `results/` are derived only from real experiment
runs. Fabricated, placeholder, mock, or hardcoded metrics must not be added as
evidence for the current study.

If future experiments use `outputs/`, each run should keep its configuration,
environment metadata, metrics, and checkpoints clearly separated from the
verified portfolio artifacts already stored under `results/`.