# Verified Split Manifests

This directory contains verified dataset split manifests. A manifest stores
sample indices only; it does not contain dataset images.

`cifar10_seed42.json` is the official split manifest shared by all four
ablation variants. It uses split seed 42 and partitions the 50,000 samples in
the official CIFAR-10 training set into 45,000 training indices and 5,000
validation indices. The validation split is class-balanced with 500 samples
per class.

The official CIFAR-10 test set is not included in this manifest. It was kept
outside the train/validation split and used only after validation-based model
selection for the final held-out evaluation of the selected Mish checkpoint.
Do not edit the manifest by hand.

- File SHA-256: `454a6d5f3a72d6881c32343afb7c9c147ac018650dfe8ba1946205f386ef5557`
- Labels SHA-256: `a3a0d804911c71de4b73015af980e237de5f82da7b1482a8efaf7adcc1722f45`
