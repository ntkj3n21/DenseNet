# Dataset Configurations

CIFAR-10 is the dataset used by the current experimental pipeline and completed benchmark.

"cifar10.json" locks the train/validation sizes, split seed, normalization, and
transform protocol. Training uses a 32x32 random crop with padding 4 followed
by random horizontal flip. Validation and test use no augmentation. All three
splits use the same CIFAR-10 normalization.

The official test set was reserved for final held-out evaluation and was not
used during training, checkpoint selection, or comparison between variants.
After Mish was selected by validation accuracy, its epoch-47 checkpoint was
evaluated once on the official test set.

The verified manifest is stored at `configs/splits/cifar10_seed42.json` and is
the authoritative source of split indices for every model variant. All official
experiments must use this exact manifest and must not generate an alternative
split.

A new manifest may be created only when starting a new experimental protocol.
That protocol must record and verify the new manifest checksum. Dataset images
must not be committed to Git.

Configuration files must remain machine-readable, portable, and free of paths
tied to a contributor's machine.
