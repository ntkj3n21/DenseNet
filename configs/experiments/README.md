# Experiment Configurations

The machine-readable ablation configurations are:

- `baseline.json`: ReLU without Squeeze-and-Excitation.
- `mish.json`: Mish without Squeeze-and-Excitation.
- `se.json`: ReLU with Squeeze-and-Excitation.
- `mish_se.json`: Mish with Squeeze-and-Excitation.

Only activation and SE usage differ. Architecture, dataset split, training
protocol, and seed set must remain identical across all four variants. The
legacy project and its reported accuracies are not evidence for this study.
