"""Activation factory for DenseNet ablation variants."""

from torch import nn


def get_activation(name: str) -> nn.Module:
    """Return a new activation module for a supported activation name."""
    if not isinstance(name, str):
        raise ValueError("Activation name must be a string: 'relu' or 'mish'.")

    normalized_name = name.strip().lower()
    if normalized_name == "relu":
        return nn.ReLU(inplace=True)
    if normalized_name == "mish":
        return nn.Mish(inplace=True)

    raise ValueError(
        f"Unsupported activation {name!r}. Expected 'relu' or 'mish'."
    )
