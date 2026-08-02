"""Tests for in-memory multi-epoch training orchestration."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from densenet_experiments.engine import EpochMetrics, run_training
from densenet_experiments.engine import runner as runner_module


def _loader() -> DataLoader:
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
    )
    targets = torch.tensor([0, 1, 1, 0])
    return DataLoader(TensorDataset(inputs, targets), batch_size=2)


def _training_objects() -> tuple[
    torch.nn.Module, torch.nn.Module, torch.optim.Optimizer
]:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, criterion, optimizer


def test_runs_requested_epochs_and_updates_model() -> None:
    model, criterion, optimizer = _training_objects()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    history = run_training(
        model, _loader(), _loader(), criterion, optimizer, "cpu", epochs=3
    )
    assert len(history.records) == 3
    assert [record.epoch for record in history.records] == [1, 2, 3]
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters())
    )
    with pytest.raises(FrozenInstanceError):
        history.best_epoch = 2  # type: ignore[misc]


class CountingScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, factor: float = 0.5) -> None:
        self.optimizer = optimizer
        self.factor = factor
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        for group in self.optimizer.param_groups:
            group["lr"] *= self.factor


def test_scheduler_steps_once_per_epoch_and_learning_rates_are_recorded() -> None:
    model, criterion, optimizer = _training_objects()
    scheduler = CountingScheduler(optimizer)
    history = run_training(
        model,
        _loader(),
        _loader(),
        criterion,
        optimizer,
        "cpu",
        epochs=3,
        scheduler=scheduler,
    )
    assert scheduler.steps == 3
    assert [record.learning_rate for record in history.records] == pytest.approx(
        [0.1, 0.05, 0.025]
    )


def test_scheduler_none_works() -> None:
    model, criterion, optimizer = _training_objects()
    history = run_training(
        model, _loader(), _loader(), criterion, optimizer, "cpu", 1
    )
    assert len(history.records) == 1
    assert history.records[0].learning_rate == pytest.approx(0.1)


def test_best_epoch_and_tie_keep_earliest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, criterion, optimizer = _training_objects()
    train_metrics = EpochMetrics(1.0, 0.5, 2, 1)
    validation = iter(
        [
            EpochMetrics(1.0, 0.5, 2, 1),
            EpochMetrics(0.8, 1.0, 2, 2),
            EpochMetrics(0.7, 1.0, 2, 2),
        ]
    )
    monkeypatch.setattr(
        runner_module, "train_one_epoch", lambda *args, **kwargs: train_metrics
    )
    monkeypatch.setattr(
        runner_module, "evaluate", lambda *args, **kwargs: next(validation)
    )
    history = run_training(
        model, _loader(), _loader(), criterion, optimizer, "cpu", 3
    )
    assert history.best_epoch == 2
    assert history.best_validation_accuracy == 1.0


@pytest.mark.parametrize("epochs", [0, -1, True, 1.5, "2"])
def test_invalid_epochs_fail(epochs: object) -> None:
    model, criterion, optimizer = _training_objects()
    with pytest.raises(ValueError, match="epochs"):
        run_training(
            model,
            _loader(),
            _loader(),
            criterion,
            optimizer,
            "cpu",
            epochs,  # type: ignore[arg-type]
        )


def test_train_exception_is_not_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    model, criterion, optimizer = _training_objects()

    def fail(*args: object, **kwargs: object) -> EpochMetrics:
        raise RuntimeError("training failed")

    monkeypatch.setattr(runner_module, "train_one_epoch", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        run_training(
            model, _loader(), _loader(), criterion, optimizer, "cpu", 1
        )


def test_evaluation_exception_is_not_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, criterion, optimizer = _training_objects()

    def fail(*args: object, **kwargs: object) -> EpochMetrics:
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(runner_module, "evaluate", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        run_training(
            model, _loader(), _loader(), criterion, optimizer, "cpu", 1
        )
