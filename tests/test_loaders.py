"""Tests for reproducible CIFAR-10 DataLoader construction."""

from collections.abc import Iterator

import pytest
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

from densenet_experiments.data import (
    CIFAR10DataLoaders,
    CIFAR10Datasets,
    SplitManifest,
    build_cifar10_dataloaders,
)
from densenet_experiments.utils import seed_worker


class IndexDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(index), index


@pytest.fixture
def synthetic_datasets() -> CIFAR10Datasets:
    manifest = SplitManifest(
        schema_version=1,
        dataset="synthetic",
        seed=42,
        num_samples=13,
        validation_size=5,
        labels_sha256="synthetic",
        train_indices=tuple(range(8)),
        validation_indices=tuple(range(8, 13)),
    )
    return CIFAR10Datasets(
        train=IndexDataset(13),
        validation=IndexDataset(7),
        test=IndexDataset(9),
        manifest=manifest,
    )


def _sample_order(loader: torch.utils.data.DataLoader) -> list[int]:
    return [int(value) for batch, _ in loader for value in batch]


def test_default_configuration(synthetic_datasets: CIFAR10Datasets) -> None:
    result = build_cifar10_dataloaders(synthetic_datasets)
    assert isinstance(result, CIFAR10DataLoaders)
    assert result.train.batch_size == 128
    assert result.validation.batch_size == result.test.batch_size == 256
    assert result.train.num_workers == 2
    assert result.train.pin_memory is False
    assert result.train.drop_last is False


def test_samplers_and_generators(synthetic_datasets: CIFAR10Datasets) -> None:
    result = build_cifar10_dataloaders(synthetic_datasets)
    assert isinstance(result.train.sampler, RandomSampler)
    assert isinstance(result.validation.sampler, SequentialSampler)
    assert isinstance(result.test.sampler, SequentialSampler)
    generators = (
        result.train_generator,
        result.validation_generator,
        result.test_generator,
    )
    assert len({id(generator) for generator in generators}) == 3
    assert result.train.generator is result.train_generator
    assert result.validation.generator is result.validation_generator
    assert result.test.generator is result.test_generator


def test_same_seed_reproduces_first_epoch(synthetic_datasets: CIFAR10Datasets) -> None:
    first = build_cifar10_dataloaders(synthetic_datasets, num_workers=0, seed=7)
    second = build_cifar10_dataloaders(synthetic_datasets, num_workers=0, seed=7)
    assert _sample_order(first.train) == _sample_order(second.train)


def test_different_seeds_change_first_epoch(synthetic_datasets: CIFAR10Datasets) -> None:
    first = build_cifar10_dataloaders(synthetic_datasets, num_workers=0, seed=7)
    second = build_cifar10_dataloaders(synthetic_datasets, num_workers=0, seed=8)
    assert _sample_order(first.train) != _sample_order(second.train)


def test_evaluation_order_is_sequential_for_any_seed(
    synthetic_datasets: CIFAR10Datasets,
) -> None:
    first = build_cifar10_dataloaders(synthetic_datasets, num_workers=0, seed=1)
    second = build_cifar10_dataloaders(synthetic_datasets, num_workers=0, seed=2)
    assert _sample_order(first.validation) == _sample_order(second.validation) == list(range(7))
    assert _sample_order(first.test) == _sample_order(second.test) == list(range(9))


def test_custom_batch_sizes_and_no_drop(synthetic_datasets: CIFAR10Datasets) -> None:
    result = build_cifar10_dataloaders(
        synthetic_datasets, train_batch_size=5, eval_batch_size=4, num_workers=0
    )
    assert result.train.batch_size == 5
    assert result.validation.batch_size == result.test.batch_size == 4
    assert len(_sample_order(result.train)) == 13


def test_drop_last_applies_only_to_train(synthetic_datasets: CIFAR10Datasets) -> None:
    result = build_cifar10_dataloaders(
        synthetic_datasets,
        train_batch_size=5,
        eval_batch_size=4,
        num_workers=0,
        drop_last_train=True,
    )
    assert len(_sample_order(result.train)) == 10
    assert _sample_order(result.validation) == list(range(7))
    assert _sample_order(result.test) == list(range(9))
    assert result.validation.drop_last is result.test.drop_last is False


def test_worker_initialization_and_runtime_flags(
    synthetic_datasets: CIFAR10Datasets,
) -> None:
    result = build_cifar10_dataloaders(
        synthetic_datasets, num_workers=1, persistent_workers=True, pin_memory=True
    )
    for loader in (result.train, result.validation, result.test):
        assert loader.worker_init_fn is seed_worker
        assert loader.persistent_workers is True
        assert loader.pin_memory is True


def test_persistent_workers_requires_worker(
    synthetic_datasets: CIFAR10Datasets,
) -> None:
    with pytest.raises(ValueError, match="num_workers > 0"):
        build_cifar10_dataloaders(
            synthetic_datasets, num_workers=0, persistent_workers=True
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_batch_size", 0),
        ("train_batch_size", -1),
        ("train_batch_size", True),
        ("eval_batch_size", 0),
        ("eval_batch_size", -1),
        ("eval_batch_size", False),
        ("num_workers", -1),
        ("num_workers", True),
        ("seed", -1),
        ("seed", False),
        ("pin_memory", 1),
        ("persistent_workers", 0),
        ("drop_last_train", 1),
    ],
)
def test_invalid_arguments_fail(
    synthetic_datasets: CIFAR10Datasets, field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        build_cifar10_dataloaders(synthetic_datasets, **{field: value})  # type: ignore[arg-type]


def test_invalid_datasets_fail() -> None:
    with pytest.raises(ValueError, match="CIFAR10Datasets"):
        build_cifar10_dataloaders(object())  # type: ignore[arg-type]


def test_train_generator_state_advances(
    synthetic_datasets: CIFAR10Datasets,
) -> None:
    result = build_cifar10_dataloaders(synthetic_datasets, num_workers=0)
    before = result.train_generator.get_state().clone()
    iterator: Iterator[object] = iter(result.train)
    next(iterator)
    after = result.train_generator.get_state()
    assert not torch.equal(before, after)
