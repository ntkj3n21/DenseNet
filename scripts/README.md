# Utility Scripts

This directory contains lightweight command-line utilities for working with
the tracked experiment artifacts.

The model architecture, training loop, evaluation logic, dataset construction,
and reproducibility utilities remain implemented in
`src/densenet_experiments/`. Scripts in this directory are thin utilities
rather than alternative implementations of the experimental pipeline.

## `aggregate_results.py`

Regenerates and verifies the derived benchmark CSV tables from the raw
per-run artifacts:

```text
results/pilot_20/<variant>/metrics.csv
results/pilot_20/<variant>/summary.json
results/extended_50/<variant>/metrics.csv
results/extended_50/<variant>/summary.json
```

The generated tables are:

```text
results/pilot_20/summary.csv
results/extended_50/summary.csv
results/benchmark_summary.csv
```

Verify that the committed tables are up to date:

```bash
python scripts/aggregate_results.py
```

Regenerate them:

```bash
python scripts/aggregate_results.py --write
```

The script validates the epoch sequence and checks that the best validation
epoch and accuracy recorded in each `summary.json` agree with the corresponding
raw `metrics.csv`.

## `plot_results.py`

Regenerates the benchmark figures from the tracked CSV artifacts.

Install the optional analysis dependency:

```bash
pip install -e ".[analysis]"
```

Generate the figures used by the project documentation:

```bash
python scripts/plot_results.py
```

By default, the figures are written to:

```text
results/figures/
```

To generate them elsewhere without modifying the tracked figures:

```bash
python scripts/plot_results.py --output-dir /tmp/densenet-figures
```

The script produces:

```text
validation_accuracy_50.png
validation_loss_50.png
best_validation_accuracy.png
pilot_vs_extended.png
best_model_train_vs_validation.png
```

## Scope

These utilities operate only on completed experiment artifacts. They do not
retrain models, change the locked benchmark configuration, or perform model
selection using the official CIFAR-10 test set.

The final held-out test result for the validation-selected Mish checkpoint is
stored separately at:

```text
results/extended_50/mish/test_metrics.json
```

It is not included in the four-variant validation aggregation or plotting
workflow.