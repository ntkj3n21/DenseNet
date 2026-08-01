"""Squeeze-and-Excitation building block."""

from torch import Tensor, nn


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


class SEBlock(nn.Module):
    """Apply channel attention while preserving the input tensor shape."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        _validate_positive_integer(channels, "channels")
        _validate_positive_integer(reduction, "reduction")

        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        weights = self.excitation(self.pool(inputs))
        return inputs * weights
