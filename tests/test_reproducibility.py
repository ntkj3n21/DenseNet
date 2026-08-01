"""Tests for random-seed reproducibility utilities."""

import random

import numpy as np
import pytest
import torch

from densenet_experiments.utils import (
    make_generator,
    seed_everything,
    seed_worker,
)


def _random_outputs() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(4)


def test_seed_everything_reproduces_all_random_sources() -> None:
    seed_everything(42)
    first = _random_outputs()
    seed_everything(42)
    second = _random_outputs()

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_different_seeds_produce_different_outputs() -> None:
    seed_everything(1)
    first = _random_outputs()
    seed_everything(2)
    second = _random_outputs()

    assert first[0] != second[0]
    assert first[1] != second[1]
    assert not torch.equal(first[2], second[2])


@pytest.mark.parametrize("seed", [-1, True, 1.5, "42"])
def test_seed_everything_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError):
        seed_everything(seed)  # type: ignore[arg-type]


def test_seed_everything_rejects_non_boolean_deterministic() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        seed_everything(42, deterministic=1)  # type: ignore[arg-type]


def test_deterministic_state_is_restored_after_test() -> None:
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        seed_everything(42, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark


def test_make_generator_is_repeatable_and_returns_new_objects() -> None:
    first = make_generator(42)
    second = make_generator(42)

    assert first is not second
    assert torch.equal(torch.rand(5, generator=first), torch.rand(5, generator=second))


def test_make_generator_changes_sequence_for_different_seed() -> None:
    first = torch.rand(5, generator=make_generator(1))
    second = torch.rand(5, generator=make_generator(2))
    assert not torch.equal(first, second)


@pytest.mark.parametrize("worker_id", [-1, True, 1.5, "0"])
def test_seed_worker_rejects_invalid_worker_id(worker_id: object) -> None:
    with pytest.raises(ValueError):
        seed_worker(worker_id)  # type: ignore[arg-type]


def test_seed_worker_uses_torch_initial_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    initial_seed = 2**32 + 123
    worker_seed = initial_seed % 2**32
    monkeypatch.setattr(torch, "initial_seed", lambda: initial_seed)

    seed_worker(0)
    actual_python = random.random()
    actual_numpy = float(np.random.random())

    expected_python = random.Random(worker_seed).random()
    expected_numpy = float(np.random.RandomState(worker_seed).random_sample())
    assert actual_python == expected_python
    assert actual_numpy == expected_numpy
