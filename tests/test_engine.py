"""Tests for epoch metrics and framework-independent engine loops."""

import math

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from densenet_experiments.engine import EpochMetrics, evaluate, train_one_epoch


def _classification_loader(batch_size: int = 2) -> DataLoader:
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    targets = torch.tensor([0, 1, 1, 0, 1])
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


def _model() -> torch.nn.Module:
    torch.manual_seed(7)
    return torch.nn.Linear(2, 2)


def test_epoch_metrics_and_percentage() -> None:
    metrics = EpochMetrics(loss=0.5, accuracy=0.75, num_samples=4, num_correct=3)
    assert metrics.accuracy_percent == 75.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loss": -1.0},
        {"loss": math.nan},
        {"loss": math.inf},
        {"accuracy": -0.1},
        {"accuracy": 1.1},
        {"accuracy": math.nan},
        {"accuracy": math.inf},
        {"num_samples": 0},
        {"num_samples": -1},
        {"num_samples": True},
        {"num_correct": -1},
        {"num_correct": False},
        {"num_correct": 5},
        {"accuracy": 0.5},
    ],
)
def test_epoch_metrics_reject_invalid_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "loss": 0.5,
        "accuracy": 0.75,
        "num_samples": 4,
        "num_correct": 3,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        EpochMetrics(**values)  # type: ignore[arg-type]


def test_train_returns_metrics_changes_parameters_and_creates_gradients() -> None:
    model = _model()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = train_one_epoch(
        model, _classification_loader(), torch.nn.CrossEntropyLoss(), optimizer, "cpu"
    )
    assert math.isfinite(metrics.loss) and metrics.loss >= 0.0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.num_samples == 5
    assert 0 <= metrics.num_correct <= 5
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("initial_training", [True, False])
def test_evaluate_preserves_parameters_gradients_and_mode(
    initial_training: bool,
) -> None:
    model = _model()
    model.train(initial_training)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = evaluate(model, _classification_loader(), torch.nn.CrossEntropyLoss(), "cpu")
    assert metrics.num_samples == 5
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training is initial_training


class PerSampleLoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return targets.float().mean()


class FixedTwoClassModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack((inputs[:, 0], inputs[:, 1]), dim=1)


def test_weighted_loss_uses_sample_count() -> None:
    inputs = torch.tensor([[1.0, 0.0]] * 3)
    targets = torch.tensor([0, 0, 3])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    metrics = evaluate(FixedTwoClassModel(), loader, PerSampleLoss(), "cpu")
    assert metrics.loss == pytest.approx(1.0)


def test_accuracy_is_top_one_count() -> None:
    inputs = torch.tensor([[3.0, 1.0], [1.0, 4.0], [5.0, 2.0], [1.0, 2.0]])
    targets = torch.tensor([0, 1, 1, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=3)
    metrics = evaluate(
        FixedTwoClassModel(), loader, torch.nn.CrossEntropyLoss(), "cpu"
    )
    assert metrics.num_correct == 3
    assert metrics.accuracy == 0.75


@pytest.mark.parametrize("operation", ["train", "evaluate"])
def test_empty_loader_fails(operation: str) -> None:
    loader = DataLoader(
        TensorDataset(torch.empty((0, 2)), torch.empty(0, dtype=torch.long)),
        batch_size=2,
        shuffle=False,
    )
    model = _model()
    with pytest.raises(ValueError, match="did not return any samples"):
        if operation == "train":
            train_one_epoch(
                model,
                loader,
                torch.nn.CrossEntropyLoss(),
                torch.optim.SGD(model.parameters(), lr=0.1),
                "cpu",
            )
        else:
            evaluate(model, loader, torch.nn.CrossEntropyLoss(), "cpu")


class SingleItemDataset(Dataset):
    def __init__(self, item: object) -> None:
        self.item = item

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> object:
        return self.item


@pytest.mark.parametrize(
    "item, message",
    [
        ((torch.ones(2),), "pair"),
        (("input", torch.tensor(0)), "inputs"),
        ((torch.ones(2), 0), "targets"),
    ],
)
def test_invalid_batch_fails(item: object, message: str) -> None:
    loader = DataLoader(SingleItemDataset(item), batch_size=1)
    with pytest.raises(ValueError, match=message):
        evaluate(_model(), loader, torch.nn.CrossEntropyLoss(), "cpu")


class InvalidOutputModel(torch.nn.Module):
    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output

    def forward(self, inputs: torch.Tensor) -> object:
        return self.output


@pytest.mark.parametrize(
    "output",
    ["not a tensor", torch.ones(2), torch.ones(3, 2)],
)
def test_invalid_model_output_fails(output: object) -> None:
    with pytest.raises(ValueError, match="Model output"):
        evaluate(
            InvalidOutputModel(output),
            _classification_loader(batch_size=2),
            torch.nn.CrossEntropyLoss(),
            "cpu",
        )


class VectorLoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.ones(targets.shape[0])


class NonFiniteLoss(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return logits.sum() * 0.0 + self.value


@pytest.mark.parametrize(
    "criterion",
    [VectorLoss(), NonFiniteLoss(math.nan), NonFiniteLoss(math.inf)],
)
def test_invalid_loss_fails(criterion: torch.nn.Module) -> None:
    with pytest.raises(ValueError, match="loss"):
        evaluate(_model(), _classification_loader(), criterion, "cpu")


def test_invalid_common_arguments_fail() -> None:
    model = _model()
    loader = _classification_loader()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError):
        evaluate(object(), loader, criterion, "cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluate(model, object(), criterion, "cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluate(model, loader, object(), "cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        evaluate(model, loader, criterion, "invalid device")
    with pytest.raises(ValueError):
        train_one_epoch(model, loader, criterion, object(), "cpu")  # type: ignore[arg-type]
    assert optimizer is not None


@pytest.mark.parametrize("value", [0.0, -1.0, True, math.nan, math.inf])
def test_invalid_gradient_clip_norm_fails(value: object) -> None:
    model = _model()
    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            _classification_loader(),
            torch.nn.CrossEntropyLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
            "cpu",
            grad_clip_norm=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("clip_norm, expected_calls", [(None, 0), (1.0, 3)])
def test_gradient_clipping_calls_utility(
    monkeypatch: pytest.MonkeyPatch,
    clip_norm: float | None,
    expected_calls: int,
) -> None:
    calls: list[float] = []

    def record_clip(parameters: object, max_norm: float) -> torch.Tensor:
        calls.append(max_norm)
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    model = _model()
    train_one_epoch(
        model,
        _classification_loader(),
        torch.nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        "cpu",
        grad_clip_norm=clip_norm,
    )
    assert calls == ([] if expected_calls == 0 else [1.0] * expected_calls)


class RaisingModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("evaluation failure")


def test_evaluation_exception_restores_mode() -> None:
    model = RaisingModel()
    model.train()
    with pytest.raises(RuntimeError, match="evaluation failure"):
        evaluate(model, _classification_loader(), torch.nn.CrossEntropyLoss(), "cpu")
    assert model.training is True
