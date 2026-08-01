"""CIFAR-10 dataset construction with a shared split manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision import datasets

from .splits import (
    SplitManifest,
    create_stratified_split,
    load_split_manifest,
    save_split_manifest,
)
from .transforms import (
    build_cifar10_eval_transform,
    build_cifar10_train_transform,
)


@dataclass(frozen=True)
class CIFAR10Datasets:
    """Train, validation, and test datasets plus their split manifest."""

    train: Dataset
    validation: Dataset
    test: Dataset
    manifest: SplitManifest


def _normalized_path(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a string or Path, got {value!r}.")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must not be an empty string.")
    return Path(value).expanduser()


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")


def build_cifar10_datasets(
    data_root: str | Path,
    manifest_path: str | Path,
    validation_size: int = 5000,
    split_seed: int = 42,
    download: bool = False,
    create_manifest_if_missing: bool = False,
) -> CIFAR10Datasets:
    """Build CIFAR-10 dataset views using one validated split manifest."""
    normalized_root = _normalized_path(data_root, "data_root")
    normalized_manifest_path = _normalized_path(manifest_path, "manifest_path")
    _validate_positive_integer(validation_size, "validation_size")
    _validate_non_negative_integer(split_seed, "split_seed")
    if not isinstance(download, bool):
        raise ValueError(f"download must be a bool, got {download!r}.")
    if not isinstance(create_manifest_if_missing, bool):
        raise ValueError(
            "create_manifest_if_missing must be a bool, "
            f"got {create_manifest_if_missing!r}."
        )

    train_transform = build_cifar10_train_transform()
    eval_transform = build_cifar10_eval_transform()

    train_base = datasets.CIFAR10(
        root=normalized_root,
        train=True,
        transform=train_transform,
        download=download,
    )
    validation_base = datasets.CIFAR10(
        root=normalized_root,
        train=True,
        transform=eval_transform,
        download=download,
    )
    test = datasets.CIFAR10(
        root=normalized_root,
        train=False,
        transform=eval_transform,
        download=download,
    )

    labels = train_base.targets
    if len(train_base) != 50_000:
        raise ValueError(
            f"Expected 50,000 CIFAR-10 training samples, found {len(train_base)}."
        )
    if len(labels) != 50_000:
        raise ValueError(
            f"Expected 50,000 CIFAR-10 training labels, found {len(labels)}."
        )
    if len(validation_base) != 50_000:
        raise ValueError(
            "Expected the CIFAR-10 validation base to expose 50,000 samples, "
            f"found {len(validation_base)}."
        )
    if len(test) != 10_000:
        raise ValueError(f"Expected 10,000 CIFAR-10 test samples, found {len(test)}.")

    if normalized_manifest_path.exists():
        manifest = load_split_manifest(
            path=normalized_manifest_path,
            labels=labels,
            expected_dataset="cifar10",
        )
        if manifest.validation_size != validation_size:
            raise ValueError(
                "Manifest validation_size does not match the requested value: "
                f"{manifest.validation_size} != {validation_size}."
            )
        if manifest.seed != split_seed:
            raise ValueError(
                "Manifest seed does not match the requested split_seed: "
                f"{manifest.seed} != {split_seed}."
            )
    elif create_manifest_if_missing:
        manifest = create_stratified_split(
            labels=labels,
            validation_size=validation_size,
            seed=split_seed,
            dataset="cifar10",
        )
        save_split_manifest(manifest, normalized_manifest_path)
    else:
        raise FileNotFoundError(
            "CIFAR-10 split manifest does not exist and automatic creation is "
            f"disabled: {normalized_manifest_path}"
        )

    return CIFAR10Datasets(
        train=Subset(train_base, manifest.train_indices),
        validation=Subset(validation_base, manifest.validation_indices),
        test=test,
        manifest=manifest,
    )
