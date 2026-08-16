# Configuration Directory

This directory contains the machine-readable inputs used by the DenseNet-40
ablation study.

## Structure

- `datasets/`
  - dataset sizes
  - preprocessing
  - normalization
  - augmentation protocol

- `splits/`
  - verified train/validation indices
  - split metadata and checksums

- `experiments/`
  - four controlled model variants:
    - baseline
    - Mish
    - SE
    - Mish + SE

- `training/`
  - execution-level training configurations:
    - smoke
    - 20-epoch pilot
    - 50-epoch extended benchmark
    - unexecuted 200-epoch multi-seed research-scale configuration

Within the same benchmark, all four model variants use the same dataset split,
training configuration, and seed set. Only the activation function and
Squeeze-and-Excitation usage differ between variants.

The completed benchmark results are stored under `results/`. Historical
accuracy values under `legacy/` are course-project artifacts and are not used as
evidence for the current study.