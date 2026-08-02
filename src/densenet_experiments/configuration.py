"""Strict loading and validation for experiment configuration files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VariantConfiguration:
    name: str
    activation: str
    use_se: bool
    se_reduction: int


@dataclass(frozen=True)
class ModelConfiguration:
    num_classes: int
    growth_rate: int
    block_config: tuple[int, int, int]
    initial_channels: int
    compression: float
    drop_rate: float


@dataclass(frozen=True)
class OptimizerConfiguration:
    name: str
    learning_rate: float
    momentum: float
    nesterov: bool
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfiguration:
    name: str
    eta_min: float
    t_max: int


@dataclass(frozen=True)
class TrainingConfiguration:
    run_name: str
    epochs: int
    seeds: tuple[int, ...]
    dataset_config: str
    split_manifest: str
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    drop_last_train: bool
    optimizer: OptimizerConfiguration
    scheduler: SchedulerConfiguration
    criterion: str
    selection_metric: str
    selection_mode: str
    test_usage: str
    deterministic: bool
    amp: bool
    grad_clip_norm: float | None


@dataclass(frozen=True)
class DatasetConfiguration:
    dataset: str
    official_train_size: int
    train_size: int
    validation_size: int
    official_test_size: int
    split_strategy: str
    split_seed: int
    split_manifest: str
    split_manifest_sha256: str
    labels_sha256: str
    test_usage: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    train_augmentation: tuple[str, ...]
    validation_augmentation: tuple[str, ...]
    test_augmentation: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfiguration:
    variant: VariantConfiguration
    model: ModelConfiguration
    training: TrainingConfiguration
    dataset: DatasetConfiguration
    split_manifest_path: Path


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSON in {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object.")
    return value


def _fields(value: dict[str, Any], expected: set[str], description: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing:
        raise ValueError(f"{description} is missing fields: {sorted(missing)}.")
    if extra:
        raise ValueError(f"{description} has unexpected fields: {sorted(extra)}.")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without outer whitespace.")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool.")
    return value


def _integer(value: Any, name: str, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, not bool.")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _real(
    value: Any, name: str, *, minimum: float, minimum_inclusive: bool = True,
    maximum: float | None = None, maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, not bool.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if result < minimum or (not minimum_inclusive and result == minimum):
        raise ValueError(f"{name} is below its allowed range.")
    if maximum is not None and (
        result > maximum or (not maximum_inclusive and result == maximum)
    ):
        raise ValueError(f"{name} is above its allowed range.")
    return result


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return tuple(_string(item, f"{name}[{index}]") for index, item in enumerate(value))


def _variant(raw: dict[str, Any]) -> tuple[VariantConfiguration, ModelConfiguration]:
    _fields(raw, {"name", "activation", "use_se", "se_reduction", "model"}, "variant")
    name = _string(raw["name"], "variant.name")
    expected = {
        "baseline": ("relu", False),
        "mish": ("mish", False),
        "se": ("relu", True),
        "mish_se": ("mish", True),
    }
    if name not in expected:
        raise ValueError(f"Unsupported variant: {name!r}.")
    activation = _string(raw["activation"], "variant.activation")
    if activation not in {"relu", "mish"}:
        raise ValueError(f"Unsupported activation: {activation!r}.")
    use_se = _bool(raw["use_se"], "variant.use_se")
    if (activation, use_se) != expected[name]:
        raise ValueError("Variant name, activation, and use_se are inconsistent.")
    se_reduction = _integer(raw["se_reduction"], "variant.se_reduction", positive=True)

    model_raw = raw["model"]
    if not isinstance(model_raw, dict):
        raise ValueError("variant.model must be an object.")
    _fields(model_raw, {"num_classes", "growth_rate", "block_config", "initial_channels", "compression", "drop_rate"}, "model")
    block = model_raw["block_config"]
    if not isinstance(block, list) or len(block) != 3:
        raise ValueError("model.block_config must contain exactly three integers.")
    block_config = tuple(
        _integer(item, f"model.block_config[{index}]", positive=True)
        for index, item in enumerate(block)
    )
    model = ModelConfiguration(
        num_classes=_integer(model_raw["num_classes"], "model.num_classes", positive=True),
        growth_rate=_integer(model_raw["growth_rate"], "model.growth_rate", positive=True),
        block_config=block_config,  # type: ignore[arg-type]
        initial_channels=_integer(model_raw["initial_channels"], "model.initial_channels", positive=True),
        compression=_real(model_raw["compression"], "model.compression", minimum=0.0, minimum_inclusive=False, maximum=1.0),
        drop_rate=_real(model_raw["drop_rate"], "model.drop_rate", minimum=0.0, maximum=1.0, maximum_inclusive=False),
    )
    return VariantConfiguration(name, activation, use_se, se_reduction), model


def _training(raw: dict[str, Any]) -> TrainingConfiguration:
    expected_fields = {"run_name", "epochs", "seeds", "dataset_config", "split_manifest", "train_batch_size", "eval_batch_size", "num_workers", "pin_memory", "persistent_workers", "drop_last_train", "optimizer", "scheduler", "criterion", "selection_metric", "selection_mode", "test_usage", "deterministic", "amp", "grad_clip_norm"}
    _fields(raw, expected_fields, "training")
    epochs = _integer(raw["epochs"], "training.epochs", positive=True)
    seeds_raw = raw["seeds"]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError("training.seeds must be a non-empty list.")
    seeds = tuple(
        _integer(seed, f"training.seeds[{index}]", positive=False)
        for index, seed in enumerate(seeds_raw)
    )
    if len(seeds) != len(set(seeds)):
        raise ValueError("training.seeds must not contain duplicates.")

    optimizer_raw = raw["optimizer"]
    if not isinstance(optimizer_raw, dict):
        raise ValueError("training.optimizer must be an object.")
    _fields(optimizer_raw, {"name", "learning_rate", "momentum", "nesterov", "weight_decay"}, "optimizer")
    if optimizer_raw["name"] != "sgd":
        raise ValueError("optimizer.name must be 'sgd'.")
    optimizer = OptimizerConfiguration(
        name="sgd",
        learning_rate=_real(optimizer_raw["learning_rate"], "optimizer.learning_rate", minimum=0.0, minimum_inclusive=False),
        momentum=_real(optimizer_raw["momentum"], "optimizer.momentum", minimum=0.0, maximum=1.0, maximum_inclusive=False),
        nesterov=_bool(optimizer_raw["nesterov"], "optimizer.nesterov"),
        weight_decay=_real(optimizer_raw["weight_decay"], "optimizer.weight_decay", minimum=0.0),
    )

    scheduler_raw = raw["scheduler"]
    if not isinstance(scheduler_raw, dict):
        raise ValueError("training.scheduler must be an object.")
    _fields(scheduler_raw, {"name", "eta_min", "T_max"}, "scheduler")
    if scheduler_raw["name"] != "cosine_annealing":
        raise ValueError("scheduler.name must be 'cosine_annealing'.")
    t_max = _integer(scheduler_raw["T_max"], "scheduler.T_max", positive=True)
    if t_max != epochs:
        raise ValueError("scheduler.T_max must equal training.epochs.")
    scheduler = SchedulerConfiguration(
        name="cosine_annealing",
        eta_min=_real(scheduler_raw["eta_min"], "scheduler.eta_min", minimum=0.0),
        t_max=t_max,
    )

    criterion = _string(raw["criterion"], "training.criterion")
    selection_metric = _string(raw["selection_metric"], "training.selection_metric")
    selection_mode = _string(raw["selection_mode"], "training.selection_mode")
    test_usage = _string(raw["test_usage"], "training.test_usage")
    if criterion != "cross_entropy":
        raise ValueError("training.criterion must be 'cross_entropy'.")
    if selection_metric != "validation_accuracy":
        raise ValueError("selection_metric must be 'validation_accuracy'.")
    if selection_mode != "max":
        raise ValueError("selection_mode must be 'max'.")
    if test_usage != "final_evaluation_only":
        raise ValueError("test_usage must be 'final_evaluation_only'.")
    grad_clip = raw["grad_clip_norm"]
    if grad_clip is not None:
        grad_clip = _real(grad_clip, "training.grad_clip_norm", minimum=0.0, minimum_inclusive=False)

    num_workers = _integer(raw["num_workers"], "training.num_workers", positive=False)
    persistent_workers = _bool(raw["persistent_workers"], "training.persistent_workers")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0.")
    return TrainingConfiguration(
        run_name=_string(raw["run_name"], "training.run_name"), epochs=epochs,
        seeds=seeds, dataset_config=_string(raw["dataset_config"], "training.dataset_config"),
        split_manifest=_string(raw["split_manifest"], "training.split_manifest"),
        train_batch_size=_integer(raw["train_batch_size"], "training.train_batch_size", positive=True),
        eval_batch_size=_integer(raw["eval_batch_size"], "training.eval_batch_size", positive=True),
        num_workers=num_workers, pin_memory=_bool(raw["pin_memory"], "training.pin_memory"),
        persistent_workers=persistent_workers,
        drop_last_train=_bool(raw["drop_last_train"], "training.drop_last_train"),
        optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        selection_metric=selection_metric, selection_mode=selection_mode,
        test_usage=test_usage, deterministic=_bool(raw["deterministic"], "training.deterministic"),
        amp=_bool(raw["amp"], "training.amp"), grad_clip_norm=grad_clip,
    )


def _dataset(raw: dict[str, Any]) -> DatasetConfiguration:
    expected = {"dataset", "official_train_size", "train_size", "validation_size", "official_test_size", "split_strategy", "split_seed", "split_manifest", "split_manifest_sha256", "labels_sha256", "test_usage", "normalization", "train_augmentation", "validation_augmentation", "test_augmentation"}
    _fields(raw, expected, "dataset")
    normalization = raw["normalization"]
    if not isinstance(normalization, dict):
        raise ValueError("dataset.normalization must be an object.")
    _fields(normalization, {"mean", "std"}, "dataset.normalization")

    def triplet(value: Any, name: str) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"{name} must contain three numbers.")
        result = tuple(_real(item, f"{name}[{index}]", minimum=0.0) for index, item in enumerate(value))
        return result  # type: ignore[return-value]

    dataset_name = _string(raw["dataset"], "dataset.dataset")
    if dataset_name != "cifar10":
        raise ValueError("dataset.dataset must be 'cifar10'.")
    test_usage = _string(raw["test_usage"], "dataset.test_usage")
    if test_usage != "final_evaluation_only":
        raise ValueError("dataset.test_usage must be 'final_evaluation_only'.")
    checksum = _string(raw["split_manifest_sha256"], "dataset.split_manifest_sha256")
    labels_checksum = _string(raw["labels_sha256"], "dataset.labels_sha256")
    if len(checksum) != 64 or len(labels_checksum) != 64:
        raise ValueError("Dataset SHA-256 values must contain 64 hexadecimal characters.")
    try:
        int(checksum, 16)
        int(labels_checksum, 16)
    except ValueError as error:
        raise ValueError("Dataset SHA-256 values must be hexadecimal.") from error
    return DatasetConfiguration(
        dataset=dataset_name,
        official_train_size=_integer(raw["official_train_size"], "dataset.official_train_size", positive=True),
        train_size=_integer(raw["train_size"], "dataset.train_size", positive=True),
        validation_size=_integer(raw["validation_size"], "dataset.validation_size", positive=True),
        official_test_size=_integer(raw["official_test_size"], "dataset.official_test_size", positive=True),
        split_strategy=_string(raw["split_strategy"], "dataset.split_strategy"),
        split_seed=_integer(raw["split_seed"], "dataset.split_seed", positive=False),
        split_manifest=_string(raw["split_manifest"], "dataset.split_manifest"),
        split_manifest_sha256=checksum, labels_sha256=labels_checksum,
        test_usage=test_usage, normalization_mean=triplet(normalization["mean"], "dataset.normalization.mean"),
        normalization_std=triplet(normalization["std"], "dataset.normalization.std"),
        train_augmentation=_string_list(raw["train_augmentation"], "dataset.train_augmentation"),
        validation_augmentation=_string_list(raw["validation_augmentation"], "dataset.validation_augmentation"),
        test_augmentation=_string_list(raw["test_augmentation"], "dataset.test_augmentation"),
    )


def load_experiment_configuration(
    variant_path: str | Path,
    training_path: str | Path,
    repository_root: str | Path,
) -> ExperimentConfiguration:
    """Load and strictly validate one variant/training configuration pair."""
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository root does not exist: {root}")
    variant_file = (root / Path(variant_path)).resolve()
    training_file = (root / Path(training_path)).resolve()
    variant, model = _variant(_read_json(variant_file, "variant configuration"))
    training = _training(_read_json(training_file, "training configuration"))
    dataset_path = (root / training.dataset_config).resolve()
    dataset = _dataset(_read_json(dataset_path, "dataset configuration"))
    if training.split_manifest != dataset.split_manifest:
        raise ValueError("Training and dataset split_manifest paths do not match.")
    manifest_path = (root / training.split_manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
    actual_checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_checksum != dataset.split_manifest_sha256:
        raise ValueError("Split manifest SHA-256 does not match dataset configuration.")
    return ExperimentConfiguration(variant, model, training, dataset, manifest_path)
