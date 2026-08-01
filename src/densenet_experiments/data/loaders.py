"""Reproducible DataLoader construction for CIFAR-10 dataset views."""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from densenet_experiments.utils import make_generator, seed_worker

from .cifar10 import CIFAR10Datasets


@dataclass(frozen=True)
class CIFAR10DataLoaders:
    """CIFAR-10 loaders and their independent random generators."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_generator: torch.Generator
    validation_generator: torch.Generator
    test_generator: torch.Generator


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")


def _validate_bool(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}.")


def build_cifar10_dataloaders(
    datasets: CIFAR10Datasets,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 2,
    seed: int = 42,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last_train: bool = False,
) -> CIFAR10DataLoaders:
    """Build independently seeded train, validation, and test loaders.

    Recreating the loaders with the same seed, datasets, environment, and
    configuration reproduces the first-epoch training order. Generator state
    advances when an iterator is used; exact resume will therefore require
    checkpointing and restoring that state in a later phase.
    """
    if not isinstance(datasets, CIFAR10Datasets):
        raise ValueError("datasets must be a CIFAR10Datasets instance.")
    _validate_positive_integer(train_batch_size, "train_batch_size")
    _validate_positive_integer(eval_batch_size, "eval_batch_size")
    _validate_non_negative_integer(num_workers, "num_workers")
    _validate_non_negative_integer(seed, "seed")
    _validate_bool(pin_memory, "pin_memory")
    _validate_bool(persistent_workers, "persistent_workers")
    _validate_bool(drop_last_train, "drop_last_train")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")

    train_generator = make_generator(seed)
    validation_generator = make_generator(seed)
    test_generator = make_generator(seed)

    train = DataLoader(
        datasets.train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        worker_init_fn=seed_worker,
        generator=train_generator,
        persistent_workers=persistent_workers,
    )
    validation = DataLoader(
        datasets.validation,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=validation_generator,
        persistent_workers=persistent_workers,
    )
    test = DataLoader(
        datasets.test,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=test_generator,
        persistent_workers=persistent_workers,
    )
    return CIFAR10DataLoaders(
        train=train,
        validation=validation,
        test=test,
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )
