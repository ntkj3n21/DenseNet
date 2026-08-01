"""Utilities for repeatable random number generation.

Exact reproducibility is not guaranteed across different PyTorch versions,
hardware, or operating-system platforms.
"""

import random

import numpy as np
import torch


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    _validate_seed(seed)
    if not isinstance(deterministic, bool):
        raise ValueError(
            f"deterministic must be a bool, got {deterministic!r}."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy for a PyTorch data-loading worker."""
    if isinstance(worker_id, bool) or not isinstance(worker_id, int) or worker_id < 0:
        raise ValueError(
            f"worker_id must be a non-negative integer, got {worker_id!r}."
        )

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a newly seeded PyTorch generator."""
    _validate_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
