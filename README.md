# DenseNet-40 Ablation Study: Mish and Squeeze-and-Excitation on CIFAR-10

A reproducible PyTorch ablation study evaluating how **Mish activation** and
**Squeeze-and-Excitation (SE)** affect a basic DenseNet-40 on CIFAR-10.

The project compares four controlled variants under the same data split,
architecture, optimizer, scheduler, and random seed:

| Variant | Activation | SE |
| --- | --- | --- |
| Baseline | ReLU | No |
| Mish | Mish | No |
| SE | ReLU | Yes |
| Mish + SE | Mish | Yes |

The main completed benchmark trains each variant for **50 epochs using seed 42**.

> **Validation-selected model:** Mish reached **92.48% validation accuracy at
> epoch 47**, improving over the ReLU baseline by **+0.62 percentage points**
> without increasing the model parameter count.
>
> The preselected Mish checkpoint was then evaluated once on the official
> CIFAR-10 test set, achieving **91.28% test accuracy**.

![50-epoch validation accuracy](results/figures/validation_accuracy_50.png)

## Main Results

| Variant | Parameters | Best Epoch | Best Val Accuracy | Δ vs Baseline | Final Val Accuracy | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1,019,722 | 44 | 91.86% | — | 91.66% | 86.77 min |
| **Mish** | 1,019,722 | 47 | **92.48%** | **+0.62 pp** | **92.34%** | 92.40 min |
| SE | 1,060,531 | 46 | 91.82% | -0.04 pp | 91.72% | 97.41 min |
| Mish + SE | 1,060,531 | 49 | 92.16% | +0.30 pp | 92.12% | 104.41 min |

Mish produced the strongest result in the 50-epoch benchmark. The SE-only
variant approximately matched the baseline but did not improve it, while
Mish + SE remained above the baseline but below Mish alone.

This means the two modifications were not simply additive under the tested
architecture and training protocol.

![Best validation accuracy](results/figures/best_validation_accuracy.png)

## Final Held-Out Test

Model selection was performed using validation accuracy only. Mish was selected
after reaching the highest validation accuracy in the 50-epoch benchmark.

The selected epoch-47 checkpoint was then evaluated once on the official
CIFAR-10 test set.

| Metric | Result |
| --- | ---: |
| Selected variant | Mish |
| Checkpoint epoch | 47 |
| Validation accuracy | 92.48% |
| **Official test accuracy** | **91.28%** |
| Test loss | 0.348849 |
| Correct predictions | 9,128 / 10,000 |

The official test set was not used to rank the four variants or to revise model
selection after evaluation. The other three variants were not evaluated on the
test set for comparison.

The machine-readable result is stored at:

```text
results/extended_50/mish/test_metrics.json

## Why This Study?

DenseNet improves feature reuse by concatenating the output of previous layers
instead of repeatedly replacing earlier representations.

This project investigates two modifications to that baseline.

**Mish** replaces ReLU in the DenseNet main path with a smooth non-monotonic
activation function. The experiment tests whether this change can improve
optimization and validation performance without increasing parameter count.

**Squeeze-and-Excitation** introduces channel-wise feature recalibration. In
the SE variants used here, an SE block with reduction ratio 16 is applied after
each DenseBlock.

The combined Mish + SE variant tests whether nonlinear feature transformation
and adaptive channel recalibration provide complementary benefits.

## Architecture

The controlled baseline is a basic DenseNet-40 configured as follows:

| Parameter | Value |
| --- | --- |
| Architecture | DenseNet-40 basic |
| Growth rate | 12 |
| Dense blocks | `[12, 12, 12]` |
| Initial channels | 16 |
| Compression | 1.0 |
| Bottleneck | Disabled |
| Dropout | 0.0 |
| Baseline activation | ReLU |
| SE reduction ratio | 16 |

Only the activation function and SE usage change between the four variants.

## CIFAR-10 Protocol

The official CIFAR-10 training set is divided into:

| Split | Samples |
| --- | ---: |
| Training | 45,000 |
| Validation | 5,000 |
| Official test | 10,000 |

The train/validation split is fixed with seed 42 and stored in:

```text
configs/splits/cifar10_seed42.json
```

The validation subset is class-balanced with 500 samples from each CIFAR-10
class.

Training augmentation consists of random 32×32 cropping with padding 4 and
random horizontal flipping. Validation and test data use no stochastic
augmentation.

Normalization:

```text
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

The official CIFAR-10 test set was not used during training, checkpoint
selection, or comparison between variants. After Mish was selected using
validation accuracy, its epoch-47 checkpoint was evaluated once on the held-out
test set.

## Training Configuration

The completed 50-epoch benchmark uses:

| Setting | Value |
| --- | --- |
| Epochs | 50 |
| Seed | 42 |
| Loss | CrossEntropyLoss |
| Optimizer | SGD |
| Learning rate | 0.1 |
| Momentum | 0.9 |
| Nesterov | Yes |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| T_max | 50 |
| AMP | Disabled |

Validation accuracy is the checkpoint-selection metric.

The corresponding machine-readable configuration is:

```text
configs/training/extended_50.json
```

## Training Horizon Matters

A 20-epoch pilot was completed before the 50-epoch benchmark.

| Variant | 20-Epoch Best | 50-Epoch Best | Gain |
| --- | ---: | ---: | ---: |
| Baseline | 89.80% | 91.86% | +2.06 pp |
| Mish | 89.64% | **92.48%** | **+2.84 pp** |
| SE | 88.98% | 91.82% | +2.84 pp |
| Mish + SE | **90.16%** | 92.16% | +2.00 pp |

The ranking changed with the longer training horizon. Mish + SE was strongest
after 20 epochs, while Mish alone became the best variant after 50 epochs.

This is why the 20-epoch experiment is treated as a pilot rather than the main
result.

![Training horizon comparison](results/figures/pilot_vs_extended.png)

## Best Model Behavior

Mish reached its best validation accuracy of 92.48% at epoch 47.

At epoch 50:

```text
Training accuracy   = 99.23%
Validation accuracy = 92.34%
```

The late training/validation gap shows that training accuracy continues to
increase after validation performance begins to saturate.

![Mish training vs validation accuracy](results/figures/best_model_train_vs_validation.png)

## Validation Loss

![50-epoch validation loss](results/figures/validation_loss_50.png)

All variants reduce validation loss substantially during training. Mish and
Mish + SE finish with lower validation loss than the baseline in the completed
50-epoch runs.

## Repository Structure

```text
DenseNet/
├── configs/
│   ├── datasets/
│   ├── experiments/
│   ├── splits/
│   └── training/
├── docs/
│   ├── EXPERIMENT_PROTOCOL.md
│   └── RESULTS.md
├── results/
│   ├── pilot_20/
│   ├── extended_50/
│   ├── figures/
│   └── benchmark_summary.csv
├── src/
│   └── densenet_experiments/
│       ├── data/
│       ├── engine/
│       ├── models/
│       └── utils/
├── tests/
├── notebooks/
├── scripts/
├── outputs/
├── legacy/
├── pyproject.toml
└── README.md
```

`src/densenet_experiments/` contains the reusable implementation for model
construction, data handling, configuration loading, training/evaluation loops,
and reproducibility utilities.

`results/` contains lightweight artifacts from real completed runs. Raw
`metrics.csv` and `summary.json` files are preserved alongside derived summary
tables and figures.

Large model checkpoints are intentionally excluded from Git.

## Result Artifacts

The main benchmark artifacts are available under:

```text
results/extended_50/
```

Each variant contains:

```markdown
Each completed variant contains its training artifacts:

```text
metrics.csv
summary.json

Derived comparison files include:

```text
results/extended_50/summary.csv
results/pilot_20/summary.csv
results/benchmark_summary.csv
results/figures/
```

For the full interpretation of the experiments, see
[`docs/RESULTS.md`](docs/RESULTS.md).

For the exact architecture, data split, training, evaluation, and artifact
rules, see
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/ntkj3n21/DenseNet.git
cd DenseNet

python -m pip install -e ".[test]"
```

Run the test suite with:

```bash
pytest -q
```

The reusable experiment framework is implemented under
`src/densenet_experiments/`.

The repository does not yet expose the framework through completed
command-line training/evaluation scripts; `scripts/` currently documents the
planned thin CLI wrappers.

## Experiment Levels

| Configuration | Epochs | Seeds | Status | Purpose |
| --- | ---: | --- | --- | --- |
| `smoke.json` | 1 | 42 | Available | Pipeline integration |
| `pilot.json` | 20 | 42 | Completed | Preliminary benchmark |
| `extended_50.json` | 50 | 42 | **Completed** | **Main portfolio benchmark** |
| `final.json` | 200 | 5 seeds | Not executed | Research-scale future study |

The 200-epoch multi-seed configuration is retained for provenance and future
work. Results from that configuration must not be described as completed.

## Limitations

The main benchmark currently uses a single random seed. Differences such as
0.30 or 0.62 percentage points may change across repeated runs and therefore
should not be interpreted as statistical proof that one method is universally
superior.

The SE comparison also evaluates one specific design: block-level SE with
reduction ratio 16. Different insertion points or reduction ratios may behave
differently.

The four-variant comparison remains validation-based. After Mish was selected,
its epoch-47 checkpoint was evaluated once on the official CIFAR-10 test set,
reaching 91.28% accuracy. This single held-out result is an estimate of the
selected model's test performance, not a test-set comparison between variants.

## Legacy Course Project

The `legacy/` directory preserves the earlier course-project implementation,
reports, notebooks, and historical artifacts that motivated this reproducible
rewrite.

Legacy accuracies and mock/hardcoded evaluation artifacts are not used as
evidence for the current benchmark.

---

Detailed results: [`docs/RESULTS.md`](docs/RESULTS.md)  
Experimental protocol: [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)