"""Tests for strict experiment configuration loading."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from densenet_experiments import (
    ExperimentConfiguration,
    load_experiment_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    shutil.copytree(PROJECT_ROOT / "configs", root / "configs")
    return root


def _paths(variant: str = "baseline", training: str = "smoke") -> tuple[str, str]:
    return (
        f"configs/experiments/{variant}.json",
        f"configs/training/{training}.json",
    )


def _load(repository: Path, variant: str = "baseline", training: str = "smoke") -> ExperimentConfiguration:
    return load_experiment_configuration(*_paths(variant, training), repository)


def _json(repository: Path, relative_path: str) -> dict[str, Any]:
    return json.loads((repository / relative_path).read_text(encoding="utf-8"))


def _write(repository: Path, relative_path: str, value: dict[str, Any]) -> None:
    (repository / relative_path).write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("variant", ["baseline", "mish", "se", "mish_se"])
@pytest.mark.parametrize("training", ["smoke", "pilot", "final"])
def test_all_variant_and_training_combinations_load(
    repository: Path, variant: str, training: str
) -> None:
    result = _load(repository, variant, training)
    assert isinstance(result, ExperimentConfiguration)
    assert result.variant.name == variant
    assert result.training.run_name == training
    assert result.split_manifest_path.is_absolute()
    assert result.split_manifest_path.is_file()
    with pytest.raises(FrozenInstanceError):
        result.variant = result.variant  # type: ignore[misc]


def test_missing_file_fails(repository: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment_configuration(
            "configs/experiments/missing.json", _paths()[1], repository
        )


def test_invalid_json_fails(repository: Path) -> None:
    path = repository / _paths()[0]
    path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid UTF-8 JSON"):
        _load(repository)


def test_missing_field_fails(repository: Path) -> None:
    relative = _paths()[0]
    value = _json(repository, relative)
    del value["activation"]
    _write(repository, relative, value)
    with pytest.raises(ValueError, match="missing fields"):
        _load(repository)


@pytest.mark.parametrize(
    ("relative", "path", "value"),
    [
        (_paths()[0], ("model", "num_classes"), True),
        (_paths()[0], ("model", "growth_rate"), 0),
        (_paths()[0], ("model", "block_config"), [12, 12]),
        (_paths()[0], ("model", "compression"), 0.0),
        (_paths()[0], ("se_reduction",), False),
        (_paths()[1], ("epochs",), True),
        (_paths()[1], ("epochs",), 0),
        (_paths()[1], ("seeds",), []),
        (_paths()[1], ("seeds",), [True]),
        (_paths()[1], ("optimizer", "learning_rate"), 0.0),
        (_paths()[1], ("optimizer", "momentum"), 1.0),
        (_paths()[1], ("optimizer", "nesterov"), 1),
    ],
)
def test_invalid_types_and_ranges_fail(
    repository: Path, relative: str, path: tuple[str, ...], value: object
) -> None:
    document = _json(repository, relative)
    target: dict[str, Any] = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _write(repository, relative, document)
    with pytest.raises(ValueError):
        _load(repository)


@pytest.mark.parametrize(
    ("name", "activation", "use_se"),
    [
        ("unknown", "relu", False),
        ("baseline", "mish", False),
        ("baseline", "relu", True),
        ("mish_se", "relu", True),
    ],
)
def test_inconsistent_variant_fails(
    repository: Path, name: str, activation: str, use_se: bool
) -> None:
    relative = _paths()[0]
    value = _json(repository, relative)
    value.update(name=name, activation=activation, use_se=use_se)
    _write(repository, relative, value)
    with pytest.raises(ValueError):
        _load(repository)


def test_t_max_must_equal_epochs(repository: Path) -> None:
    relative = _paths()[1]
    value = _json(repository, relative)
    value["scheduler"]["T_max"] = 2
    _write(repository, relative, value)
    with pytest.raises(ValueError, match="T_max"):
        _load(repository)


def test_duplicate_seed_fails(repository: Path) -> None:
    relative = _paths()[1]
    value = _json(repository, relative)
    value["seeds"] = [42, 42]
    _write(repository, relative, value)
    with pytest.raises(ValueError, match="duplicates"):
        _load(repository)


def test_missing_manifest_fails(repository: Path) -> None:
    (repository / "configs/splits/cifar10_seed42.json").unlink()
    with pytest.raises(FileNotFoundError, match="Split manifest"):
        _load(repository)


def test_manifest_checksum_mismatch_fails(repository: Path) -> None:
    relative = "configs/datasets/cifar10.json"
    value = _json(repository, relative)
    value["split_manifest_sha256"] = "0" * 64
    _write(repository, relative, value)
    with pytest.raises(ValueError, match="SHA-256"):
        _load(repository)
