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