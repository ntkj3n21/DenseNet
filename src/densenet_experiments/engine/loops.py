"""Framework-independent single-epoch training and evaluation loops."""

import math
from numbers import Real

import torch
from torch.utils.data import DataLoader

from .metrics import EpochMetrics


def _validate_common_inputs(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device | str,
) -> torch.device:
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a torch.nn.Module.")
    if not isinstance(data_loader, DataLoader):
        raise ValueError("data_loader must be a torch.utils.data.DataLoader.")
    if not isinstance(criterion, torch.nn.Module):
        raise ValueError("criterion must be a torch.nn.Module.")
    if not isinstance(device, (torch.device, str)):
        raise ValueError("device must be a torch.device or valid device string.")
    try:
        return torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid device: {device!r}.") from error


def _validate_grad_clip_norm(grad_clip_norm: float | None) -> None:
    if grad_clip_norm is None:
        return
    if isinstance(grad_clip_norm, bool) or not isinstance(grad_clip_norm, Real):
        raise ValueError("grad_clip_norm must be None or a positive real number.")
    if not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be finite and positive.")


def _prepare_batch(
    batch: object, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Each batch must be a pair of inputs and targets.")
    inputs, targets = batch
    if not isinstance(inputs, torch.Tensor):
        raise ValueError("Batch inputs must be a torch.Tensor.")
    if not isinstance(targets, torch.Tensor):
        raise ValueError("Batch targets must be a torch.Tensor.")
    if inputs.ndim == 0 or inputs.shape[0] <= 0:
        raise ValueError("Batch size must be positive.")
    if targets.ndim != 1:
        raise ValueError("Classification targets must be a one-dimensional tensor.")
    batch_size = inputs.shape[0]
    if targets.shape[0] != batch_size:
        raise ValueError("Input and target batch dimensions must match.")
    return inputs.to(device), targets.to(device), batch_size


def _validate_logits(logits: object, batch_size: int) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise ValueError("Model output must be a torch.Tensor.")
    if logits.ndim != 2:
        raise ValueError("Model output must have shape [batch, classes].")
    if logits.shape[0] != batch_size:
        raise ValueError("Model output batch dimension must match targets.")
    if logits.shape[1] <= 0:
        raise ValueError("Model output must contain at least one class.")
    return logits


def _validate_loss(loss: object) -> torch.Tensor:
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("Criterion loss must be a scalar tensor.")
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("Criterion loss must be finite.")
    if loss.item() < 0.0:
        raise ValueError("Criterion loss must be non-negative.")
    return loss


def _metrics(total_loss: float, num_samples: int, num_correct: int) -> EpochMetrics:
    if num_samples == 0:
        raise ValueError("data_loader did not return any samples.")
    return EpochMetrics(
        loss=total_loss / num_samples,
        accuracy=num_correct / num_samples,
        num_samples=num_samples,
        num_correct=num_correct,
    )


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    grad_clip_norm: float | None = None,
) -> EpochMetrics:
    """Train a model for one epoch and return sample-weighted metrics."""
    normalized_device = _validate_common_inputs(
        model, data_loader, criterion, device
    )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer must be a torch.optim.Optimizer.")
    _validate_grad_clip_norm(grad_clip_norm)

    model.train()
    total_loss = 0.0
    num_samples = 0
    num_correct = 0
    for batch in data_loader:
        inputs, targets, batch_size = _prepare_batch(batch, normalized_device)
        optimizer.zero_grad(set_to_none=True)
        logits = _validate_logits(model(inputs), batch_size)
        loss = _validate_loss(criterion(logits, targets))
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += float(loss.item()) * batch_size
        num_samples += batch_size
        num_correct += int((logits.argmax(dim=1) == targets).sum().item())

    return _metrics(total_loss, num_samples, num_correct)


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device | str,
) -> EpochMetrics:
    """Evaluate one epoch without gradients and restore the prior model mode."""
    normalized_device = _validate_common_inputs(
        model, data_loader, criterion, device
    )
    was_training = model.training
    model.eval()
    try:
        total_loss = 0.0
        num_samples = 0
        num_correct = 0
        with torch.inference_mode():
            for batch in data_loader:
                inputs, targets, batch_size = _prepare_batch(batch, normalized_device)
                logits = _validate_logits(model(inputs), batch_size)
                loss = _validate_loss(criterion(logits, targets))
                total_loss += float(loss.item()) * batch_size
                num_samples += batch_size
                num_correct += int((logits.argmax(dim=1) == targets).sum().item())
        return _metrics(total_loss, num_samples, num_correct)
    finally:
        model.train(was_training)
