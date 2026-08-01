"""Dataset and split utilities exposed by the package."""

from .cifar10 import CIFAR10Datasets, build_cifar10_datasets
from .splits import (
    SplitManifest,
    compute_labels_sha256,
    create_stratified_split,
    load_split_manifest,
    save_split_manifest,
)
from .transforms import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    build_cifar10_eval_transform,
    build_cifar10_train_transform,
)

__all__ = [
    "CIFAR10Datasets",
    "build_cifar10_datasets",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "build_cifar10_train_transform",
    "build_cifar10_eval_transform",
    "SplitManifest",
    "compute_labels_sha256",
    "create_stratified_split",
    "save_split_manifest",
    "load_split_manifest",
]
