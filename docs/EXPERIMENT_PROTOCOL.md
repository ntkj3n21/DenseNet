# Experiment Protocol

## Official project baseline

The official baseline for the new project is a basic DenseNet-40 with the following architectural definition:

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

The legacy implementation used compression 0.5. It is retained for historical reference and is not the official baseline of this project.

## Ablation variants

- `baseline`: ReLU without SE.
- `mish`: Mish without SE.
- `se`: ReLU with SE.
- `mish_se`: Mish with SE.

Only the activation and SE settings may differ across these variants. All other training and evaluation conditions must be shared.

## CIFAR-10 data protocol

The official CIFAR-10 training set will be divided into 45,000 training samples
and 5,000 validation samples. The split will be equally stratified across all
10 classes, assigning 500 validation samples to each class, with split seed 42.

The generated split manifest will store the exact train and validation indices
plus a checksum of the ordered labels. All four ablation variants will use the
same manifest.

The official test set will remain unchanged and will be used only for final
evaluation. It must not be used to select an epoch, checkpoint, or model
variant.

### Transform protocol

Training uses a random 32x32 crop with padding 4, random horizontal flip,
tensor conversion, and CIFAR-10 normalization. Validation and test use tensor
conversion and the same normalization without stochastic augmentation.

Training and validation are constructed as separate dataset wrappers so they
can use different transforms. Both wrappers reference the same underlying
official training split and are restricted by the same manifest indices. The
official test set is not used during training or model selection.

## Planned protocol

1. CIFAR-10 will be the first dataset.
2. A fixed train/validation split will be defined and reused across all variants and seeds.
3. The test set will not be evaluated after every epoch.
4. Validation metrics will select the best checkpoint.
5. The test set will be used only for final evaluation after model selection.
6. All four variants will use the same training protocol.
7. Experiments will run with multiple random seeds.
8. Every run will store a real `metrics.csv` file.
9. Every run will store a configuration snapshot and environment metadata.
10. Every run will store both the best checkpoint and the last checkpoint.
11. Training will support resume with complete state restoration.
12. Evaluation errors must stop the run; evaluation must never fall back to mock data.
13. Results will not be published unless the corresponding configuration, metrics, checkpoint, and evaluation output are all available.

## Decisions pending protocol lock

The optimizer, learning rate, and number of epochs are intentionally
unspecified at this stage. These values will be selected, documented, and
locked in the next protocol phase before official experiments begin.
