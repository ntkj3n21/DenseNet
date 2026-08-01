# Reproducible DenseNet-40 Ablation Study

This research project will study the effects of Mish activation and Squeeze-and-Excitation (SE) on a DenseNet-40 image-classification baseline. The first target dataset is CIFAR-10.

> **Status:** No official experimental results are available yet. Historical accuracy values from the legacy course project are not results of this research project and must not be reported as such.

## Planned variants

| ID | Activation | Squeeze-and-Excitation |
| --- | --- | --- |
| B0 | ReLU | No |
| B1 | Mish | No |
| B2 | ReLU | Yes |
| B3 | Mish | Yes |

The study is designed around reproducibility, fair ablation, multiple random seeds, real metric logs, and validation-based model selection.

## Repository structure

- `configs/`: future machine-readable dataset and experiment configurations.
- `docs/`: experimental protocol and research documentation.
- `notebooks/`: Google Colab wrappers and analysis notebooks only.
- `outputs/`: generated experiment artifacts, excluded from Git except for its README.
- `scripts/`: future command-line entry points.
- `src/densenet_experiments/`: Python package for the new research pipeline.
- `tests/`: verification and reproducibility tests.
- `legacy/`: archived course-project code and artifacts; it is not source code for the new pipeline.

## Google Colab

Google Colab will be supported through thin notebook wrappers that call the same Python scripts used elsewhere. Model implementation, training behavior, evaluation behavior, and other business logic will not live in notebooks.

The training protocol and implementation remain to be developed. No performance claims should be made until a complete configuration, real metrics, checkpoints, and evaluation outputs are available.
