"""Stratified split creation and validated JSON manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any


@dataclass(frozen=True)
class SplitManifest:
    """Immutable description of a train/validation split."""

    schema_version: int
    dataset: str
    seed: int
    num_samples: int
    validation_size: int
    labels_sha256: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _validate_dataset(dataset: str) -> str:
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string.")
    return dataset.strip()


def _validated_labels(labels: Sequence[int]) -> tuple[int, ...]:
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise ValueError("labels must be a non-empty sequence of integers.")
    if len(labels) == 0:
        raise ValueError("labels must not be empty.")

    validated: list[int] = []
    for index, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"labels[{index}] must be an integer, got {label!r}.")
        if label < 0:
            raise ValueError(f"labels[{index}] must be non-negative, got {label}.")
        validated.append(label)

    class_counts = Counter(validated)
    if len(class_counts) < 2:
        raise ValueError("labels must contain at least two classes.")
    undersized = [label for label, count in class_counts.items() if count < 2]
    if undersized:
        raise ValueError(
            "Every class must contain at least two samples; classes with fewer "
            f"than two samples: {sorted(undersized)}."
        )
    return tuple(validated)


def compute_labels_sha256(labels: Sequence[int]) -> str:
    """Return a stable SHA-256 checksum for an ordered label sequence."""
    validated = _validated_labels(labels)
    payload = json.dumps(validated, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_stratified_split(
    labels: Sequence[int],
    validation_size: int,
    seed: int,
    dataset: str = "cifar10",
) -> SplitManifest:
    """Create a deterministic class-balanced train/validation split."""
    validated_labels = _validated_labels(labels)
    _validate_positive_integer(validation_size, "validation_size")
    _validate_non_negative_integer(seed, "seed")
    normalized_dataset = _validate_dataset(dataset)

    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(validated_labels):
        indices_by_class[label].append(index)

    num_classes = len(indices_by_class)
    if validation_size % num_classes != 0:
        raise ValueError(
            "validation_size must be divisible by the number of classes."
        )
    validation_per_class = validation_size // num_classes
    if validation_per_class < 1:
        raise ValueError("validation_size must allocate at least one sample per class.")

    for label, class_indices in indices_by_class.items():
        if validation_per_class >= len(class_indices):
            raise ValueError(
                f"Class {label} must retain at least one training sample after "
                "the validation split."
            )

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in sorted(indices_by_class):
        class_indices = indices_by_class[label].copy()
        rng.shuffle(class_indices)
        validation_indices.extend(class_indices[:validation_per_class])
        train_indices.extend(class_indices[validation_per_class:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return SplitManifest(
        schema_version=1,
        dataset=normalized_dataset,
        seed=seed,
        num_samples=len(validated_labels),
        validation_size=validation_size,
        labels_sha256=compute_labels_sha256(validated_labels),
        train_indices=tuple(train_indices),
        validation_indices=tuple(validation_indices),
    )


def save_split_manifest(manifest: SplitManifest, path: str | Path) -> None:
    """Write a split manifest as human-readable UTF-8 JSON."""
    if not isinstance(manifest, SplitManifest):
        raise ValueError("manifest must be a SplitManifest instance.")

    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"Manifest path is a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as manifest_file:
        json.dump(asdict(manifest), manifest_file, indent=2, ensure_ascii=False)
        manifest_file.write("\n")


def _required_manifest_fields() -> set[str]:
    return {
        "schema_version",
        "dataset",
        "seed",
        "num_samples",
        "validation_size",
        "labels_sha256",
        "train_indices",
        "validation_indices",
    }


def _validate_index_list(value: Any, name: str, num_samples: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array of integers.")

    indices: list[int] = []
    for position, index in enumerate(value):
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{name}[{position}] must be an integer.")
        if not 0 <= index < num_samples:
            raise ValueError(
                f"{name}[{position}]={index} is outside [0, {num_samples})."
            )
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError(f"{name} contains duplicate indices.")
    return tuple(indices)


def load_split_manifest(
    path: str | Path,
    labels: Sequence[int],
    expected_dataset: str | None = None,
) -> SplitManifest:
    """Load a manifest and validate it against the current ordered labels."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {source}")
    if not source.is_file():
        raise ValueError(f"Split manifest path is not a file: {source}")

    try:
        with source.open("r", encoding="utf-8") as manifest_file:
            raw = json.load(manifest_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid split manifest JSON: {error}") from error

    if not isinstance(raw, dict):
        raise ValueError("Split manifest must be a JSON object.")
    required_fields = _required_manifest_fields()
    missing_fields = required_fields - raw.keys()
    if missing_fields:
        raise ValueError(
            f"Split manifest is missing required fields: {sorted(missing_fields)}."
        )
    unexpected_fields = raw.keys() - required_fields
    if unexpected_fields:
        raise ValueError(
            f"Split manifest contains unexpected fields: {sorted(unexpected_fields)}."
        )

    schema_version = raw["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError(f"Unsupported schema_version: {schema_version!r}.")
    dataset = _validate_dataset(raw["dataset"])
    if dataset != raw["dataset"]:
        raise ValueError("Manifest dataset must not contain surrounding whitespace.")
    _validate_non_negative_integer(raw["seed"], "seed")
    _validate_positive_integer(raw["num_samples"], "num_samples")
    _validate_positive_integer(raw["validation_size"], "validation_size")

    validated_labels = _validated_labels(labels)
    num_samples = raw["num_samples"]
    if num_samples != len(validated_labels):
        raise ValueError(
            f"Manifest num_samples={num_samples} does not match labels length "
            f"{len(validated_labels)}."
        )

    labels_sha256 = raw["labels_sha256"]
    if not isinstance(labels_sha256, str):
        raise ValueError("labels_sha256 must be a string.")
    current_checksum = compute_labels_sha256(validated_labels)
    if labels_sha256 != current_checksum:
        raise ValueError("Manifest labels_sha256 does not match the current labels.")

    if expected_dataset is not None:
        normalized_expected = _validate_dataset(expected_dataset)
        if dataset != normalized_expected:
            raise ValueError(
                f"Manifest dataset {dataset!r} does not match expected dataset "
                f"{normalized_expected!r}."
            )

    train_indices = _validate_index_list(
        raw["train_indices"], "train_indices", num_samples
    )
    validation_indices = _validate_index_list(
        raw["validation_indices"], "validation_indices", num_samples
    )
    if len(validation_indices) != raw["validation_size"]:
        raise ValueError("validation_size does not match validation_indices length.")

    train_set = set(train_indices)
    validation_set = set(validation_indices)
    if train_set & validation_set:
        raise ValueError("Train and validation indices overlap.")
    if train_set | validation_set != set(range(num_samples)):
        raise ValueError("Train and validation indices do not cover the dataset.")

    validation_counts = Counter(validated_labels[index] for index in validation_indices)
    expected_classes = set(validated_labels)
    if (
        set(validation_counts) != expected_classes
        or len(set(validation_counts.values())) != 1
    ):
        raise ValueError("Validation indices are not equally stratified by class.")

    return SplitManifest(
        schema_version=1,
        dataset=dataset,
        seed=raw["seed"],
        num_samples=num_samples,
        validation_size=raw["validation_size"],
        labels_sha256=labels_sha256,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
