# Experiment Protocol

This document defines the experimental protocol used for the reproducible
DenseNet-40 ablation study on CIFAR-10.

The project compares the effects of Mish activation and
Squeeze-and-Excitation (SE) under controlled training conditions.

## 1. Baseline Architecture

The project baseline is a basic DenseNet-40.

| Parameter | Value |
| --- | --- |
| Architecture | DenseNet-40 basic |
| Growth rate | 12 |
| Block configuration | `[12, 12, 12]` |
| Initial channels | 16 |
| Compression | 1.0 |
| Activation | ReLU |
| Squeeze-and-Excitation | Disabled |
| Bottleneck | Disabled |
| Dropout | 0.0 |

The historical course-project implementation under `legacy/` used a different
configuration and is preserved only for reference.

## 2. Ablation Variants

Four controlled variants are evaluated.

| Variant | Activation | SE |
| --- | --- | --- |
| `baseline` | ReLU | No |
| `mish` | Mish | No |
| `se` | ReLU | Yes |
| `mish_se` | Mish | Yes |

Only activation and SE usage differ between variants.

All other architectural, dataset, training, and evaluation settings are shared
within the same benchmark.

For SE-enabled variants, the reduction ratio is 16 and the SE module is applied
after each DenseBlock.

## 3. CIFAR-10 Data Protocol

The official CIFAR-10 training set contains 50,000 images.

It is divided into:

- 45,000 training samples
- 5,000 validation samples
- 10,000 untouched official test samples

The validation split is class-balanced with 500 validation examples from each
of the 10 CIFAR-10 classes.

Split seed:

```text
42