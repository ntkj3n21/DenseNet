"""In-memory orchestration for a fixed number of training epochs."""

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch.utils.data import DataLoader

from .loops import evaluate, train_one_epoch
from .metrics import EpochMetrics


@dataclass(frozen=True)
class EpochRecord:
    """Metrics and learning rate recorded for one completed epoch."""

    epoch: int
    train_metrics: EpochMetrics
    validation_metrics: EpochMetrics
    learning_rate: float


@dataclass(frozen=True)
class TrainingHistory:
    """Immutable epoch records and the best validation result."""

    records: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_accuracy: float


def _learning_rate(optimizer: torch.optim.Optimizer) -> float:
    value = optimizer.param_groups[0].get("lr")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Optimizer learning rate must be a real number.")
    learning_rate = float(value)
    if not math.isfinite(learning_rate) or learning_rate < 0.0:
        raise ValueError("Optimizer learning rate must be finite and non-negative.")
    return learning_rate


def run_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epochs: int,
    scheduler: object | None = None,
    grad_clip_norm: float | None = None,
) -> TrainingHistory:
    """Run train/validation epochs and return an in-memory immutable history."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer, not bool.")

    records: list[EpochRecord] = []
    best_epoch = 0
    best_accuracy = -1.0
    for epoch in range(1, epochs + 1):
        learning_rate = _learning_rate(optimizer)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
        )
        validation_metrics = evaluate(
            model, validation_loader, criterion, device
        )
        records.append(
            EpochRecord(
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                learning_rate=learning_rate,
            )
        )
        if validation_metrics.accuracy > best_accuracy:
            best_epoch = epoch
            best_accuracy = validation_metrics.accuracy
        if scheduler is not None:
            scheduler.step()  # type: ignore[attr-defined]

    return TrainingHistory(tuple(records), best_epoch, best_accuracy)
