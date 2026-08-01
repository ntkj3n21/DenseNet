"""Unit tests for the unified DenseNet-40 ablation model."""

import pytest
import torch
from torch import nn

from densenet_experiments.models import DenseNet40, SEBlock, get_activation


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [("relu", nn.ReLU), ("mish", nn.Mish), (" ReLU ", nn.ReLU), ("MISH", nn.Mish)],
)
def test_activation_factory(name: str, expected_type: type[nn.Module]) -> None:
    assert isinstance(get_activation(name), expected_type)


def test_activation_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        get_activation("gelu")


def test_activation_factory_returns_new_instances() -> None:
    assert get_activation("relu") is not get_activation("relu")


def test_se_block_preserves_shape() -> None:
    inputs = torch.randn(2, 8, 16, 16)
    assert SEBlock(8, reduction=4)(inputs).shape == inputs.shape


def test_se_block_hidden_channels_have_minimum_of_one() -> None:
    block = SEBlock(3, reduction=16)
    first_convolution = block.excitation[0]
    assert isinstance(first_convolution, nn.Conv2d)
    assert first_convolution.out_channels == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channels": 0},
        {"channels": -1},
        {"channels": 8, "reduction": 0},
        {"channels": 8, "reduction": -1},
    ],
)
def test_se_block_rejects_invalid_configuration(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SEBlock(**kwargs)


@pytest.mark.parametrize(
    ("activation", "use_se"),
    [("relu", False), ("mish", False), ("relu", True), ("mish", True)],
)
def test_all_variants_produce_class_logits(
    activation: str, use_se: bool
) -> None:
    model = DenseNet40(activation=activation, use_se=use_se)
    outputs = model(torch.randn(2, 3, 32, 32))
    assert outputs.shape == (2, 10)


def test_baseline_feature_channels() -> None:
    assert DenseNet40().num_features == 448


@pytest.mark.parametrize(("use_se", "expected_count"), [(False, 0), (True, 3)])
def test_se_block_count(use_se: bool, expected_count: int) -> None:
    model = DenseNet40(use_se=use_se)
    count = sum(isinstance(module, SEBlock) for module in model.modules())
    assert count == expected_count


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_parameter_counts_are_controlled_by_se_only() -> None:
    relu = _parameter_count(DenseNet40(activation="relu", use_se=False))
    mish = _parameter_count(DenseNet40(activation="mish", use_se=False))
    relu_se = _parameter_count(DenseNet40(activation="relu", use_se=True))
    mish_se = _parameter_count(DenseNet40(activation="mish", use_se=True))

    assert relu == mish
    assert relu_se == mish_se
    assert relu_se > relu


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_classes": 0},
        {"growth_rate": 0},
        {"initial_channels": 0},
        {"block_config": (12, 12)},
        {"block_config": (12, 0, 12)},
        {"compression": 0.0},
        {"compression": 1.1},
        {"drop_rate": -0.1},
        {"drop_rate": 1.0},
        {"se_reduction": 0},
        {"use_se": 1},
        {"activation": "gelu"},
    ],
)
def test_densenet_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DenseNet40(**kwargs)


def test_baseline_transition_spatial_dimensions() -> None:
    model = DenseNet40()
    spatial_dimensions: list[tuple[int, int]] = []

    def record_shape(
        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        spatial_dimensions.append(tuple(output.shape[-2:]))

    hooks = [
        transition.register_forward_hook(record_shape)
        for transition in model.transitions
    ]
    try:
        model(torch.randn(2, 3, 32, 32))
    finally:
        for hook in hooks:
            hook.remove()

    assert spatial_dimensions == [(16, 16), (8, 8)]
