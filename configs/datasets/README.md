# Dataset Configurations

CIFAR-10 will be the first dataset supported by the new experimental pipeline.

"cifar10.json" locks the train/validation sizes, split seed, normalization, and
transform protocol. Training uses a 32x32 random crop with padding 4 followed
by random horizontal flip. Validation and test use no augmentation. All three
splits use the same CIFAR-10 normalization.

The official test set is reserved for final evaluation. After CIFAR-10 is
downloaded in a later phase, the real split indices and ordered-label checksum
will be saved in a split manifest shared by all model variants. Dataset images
must not be committed to Git.

Configuration files must remain machine-readable, portable, and free of paths
tied to a contributor's machine. The official test set is used only for final
evaluation.
