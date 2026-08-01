"""Unified basic DenseNet-40 architecture for controlled ablations."""

from numbers import Real

import torch
from torch import Tensor, nn

from .activations import get_activation
from .se import SEBlock


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_drop_rate(drop_rate: float) -> None:
    if (
        isinstance(drop_rate, bool)
        or not isinstance(drop_rate, Real)
        or not 0.0 <= drop_rate < 1.0
    ):
        raise ValueError(
            f"drop_rate must satisfy 0.0 <= drop_rate < 1.0, got {drop_rate!r}."
        )


class DenseLayer(nn.Module):
    """A basic BN-activation-3x3 convolution DenseNet layer."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        activation: str,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_integer(in_channels, "in_channels")
        _validate_positive_integer(growth_rate, "growth_rate")
        _validate_drop_rate(drop_rate)

        self.normalization = nn.BatchNorm2d(in_channels)
        self.activation = get_activation(activation)
        self.convolution = nn.Conv2d(
            in_channels,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.dropout = nn.Dropout(p=float(drop_rate)) if drop_rate > 0.0 else None

    def forward(self, inputs: Tensor) -> Tensor:
        new_features = self.convolution(
            self.activation(self.normalization(inputs))
        )
        if self.dropout is not None:
            new_features = self.dropout(new_features)
        return torch.cat((inputs, new_features), dim=1)


class DenseBlock(nn.Module):
    """A sequence of densely connected basic layers."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        activation: str,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_integer(num_layers, "num_layers")
        _validate_positive_integer(in_channels, "in_channels")
        _validate_positive_integer(growth_rate, "growth_rate")
        _validate_drop_rate(drop_rate)

        self.layers = nn.ModuleList(
            DenseLayer(
                in_channels + layer_index * growth_rate,
                growth_rate,
                activation,
                drop_rate,
            )
            for layer_index in range(num_layers)
        )
        self.output_channels = in_channels + num_layers * growth_rate

    def forward(self, inputs: Tensor) -> Tensor:
        features = inputs
        for layer in self.layers:
            features = layer(features)
        return features


class Transition(nn.Module):
    """Reduce channels and halve spatial resolution between dense blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
    ) -> None:
        super().__init__()
        _validate_positive_integer(in_channels, "in_channels")
        _validate_positive_integer(out_channels, "out_channels")
        if out_channels > in_channels:
            raise ValueError(
                "out_channels must be less than or equal to in_channels, "
                f"got {out_channels} > {in_channels}."
            )

        self.normalization = nn.BatchNorm2d(in_channels)
        self.activation = get_activation(activation)
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.normalization(inputs)
        features = self.activation(features)
        features = self.convolution(features)
        return self.pool(features)


class DenseNet40(nn.Module):
    """Basic three-block DenseNet-40 for CIFAR-sized image classification."""

    def __init__(
        self,
        num_classes: int = 10,
        growth_rate: int = 12,
        block_config: tuple[int, int, int] = (12, 12, 12),
        initial_channels: int = 16,
        compression: float = 1.0,
        activation: str = "relu",
        use_se: bool = False,
        se_reduction: int = 16,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_integer(num_classes, "num_classes")
        _validate_positive_integer(growth_rate, "growth_rate")
        _validate_positive_integer(initial_channels, "initial_channels")
        if not isinstance(block_config, tuple) or len(block_config) != 3:
            raise ValueError(
                "block_config must be a tuple containing exactly three "
                "positive integers."
            )
        for index, num_layers in enumerate(block_config):
            _validate_positive_integer(num_layers, f"block_config[{index}]")
        if (
            isinstance(compression, bool)
            or not isinstance(compression, Real)
            or not 0.0 < compression <= 1.0
        ):
            raise ValueError(
                "compression must satisfy 0.0 < compression <= 1.0, "
                f"got {compression!r}."
            )
        _validate_positive_integer(se_reduction, "se_reduction")
        _validate_drop_rate(drop_rate)
        if not isinstance(use_se, bool):
            raise ValueError(f"use_se must be a bool, got {use_se!r}.")

        normalized_activation = (
            activation.strip().lower() if isinstance(activation, str) else activation
        )
        get_activation(activation)

        self.activation_name = normalized_activation
        self.use_se = use_se
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.initial_channels = initial_channels
        self.compression = float(compression)

        self.initial_convolution = nn.Conv2d(
            3,
            initial_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        current_channels = initial_channels

        for block_index, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers,
                current_channels,
                growth_rate,
                normalized_activation,
                drop_rate,
            )
            self.blocks.append(block)
            current_channels = block.output_channels

            if use_se:
                self.se_blocks.append(SEBlock(current_channels, se_reduction))

            if block_index < len(block_config) - 1:
                transition_channels = max(
                    int(current_channels * self.compression), 1
                )
                self.transitions.append(
                    Transition(
                        current_channels,
                        transition_channels,
                        normalized_activation,
                    )
                )
                current_channels = transition_channels

        self.num_features = current_channels
        self.final_normalization = nn.BatchNorm2d(self.num_features)
        self.final_activation = get_activation(normalized_activation)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.num_features, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # One Kaiming protocol is used for both ReLU and Mish ablations.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.initial_convolution(inputs)
        for block_index, block in enumerate(self.blocks):
            features = block(features)
            if self.use_se:
                features = self.se_blocks[block_index](features)
            if block_index < len(self.transitions):
                features = self.transitions[block_index](features)

        features = self.final_normalization(features)
        features = self.final_activation(features)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        return self.classifier(features)
