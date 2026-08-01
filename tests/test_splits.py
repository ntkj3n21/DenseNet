"""Tests for deterministic stratified split manifests."""

from collections import Counter
import json
from pathlib import Path

import pytest

from densenet_experiments.data import (
    compute_labels_sha256,
    create_stratified_split,
    load_split_manifest,
    save_split_manifest,
)


@pytest.fixture
def labels() -> list[int]:
    return [label for label in range(10) for _ in range(100)]


def test_same_seed_creates_same_manifest(labels: list[int]) -> None:
    first = create_stratified_split(labels, 200, 42)
    second = create_stratified_split(labels, 200, 42)
    assert first == second


def test_different_seed_can_create_different_split(labels: list[int]) -> None:
    first = create_stratified_split(labels, 200, 1)
    second = create_stratified_split(labels, 200, 2)
    assert first.validation_indices != second.validation_indices


def test_split_is_disjoint_complete_and_correctly_sized(labels: list[int]) -> None:
    manifest = create_stratified_split(labels, 200, 42)
    train = set(manifest.train_indices)
    validation = set(manifest.validation_indices)

    assert not train & validation
    assert train | validation == set(range(1000))
    assert len(manifest.train_indices) == 800
    assert len(manifest.validation_indices) == 200


def test_validation_split_is_equally_stratified(labels: list[int]) -> None:
    manifest = create_stratified_split(labels, 200, 42)
    counts = Counter(labels[index] for index in manifest.validation_indices)
    assert counts == {label: 20 for label in range(10)}


def test_checksum_is_stable_and_order_sensitive(labels: list[int]) -> None:
    assert compute_labels_sha256(labels) == compute_labels_sha256(labels.copy())
    assert compute_labels_sha256(labels) != compute_labels_sha256(list(reversed(labels)))


def test_manifest_save_load_round_trip(
    labels: list[int], tmp_path: Path
) -> None:
    manifest = create_stratified_split(labels, 200, 42)
    path = tmp_path / "nested" / "split.json"
    save_split_manifest(manifest, path)
    assert load_split_manifest(path, labels) == manifest


def test_expected_dataset_match_loads(labels: list[int], tmp_path: Path) -> None:
    manifest = create_stratified_split(labels, 200, 42, dataset="cifar10")
    path = tmp_path / "split.json"
    save_split_manifest(manifest, path)
    assert load_split_manifest(path, labels, expected_dataset="cifar10") == manifest


def test_expected_dataset_mismatch_fails(labels: list[int], tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    save_split_manifest(create_stratified_split(labels, 200, 42), path)
    with pytest.raises(ValueError, match="expected dataset"):
        load_split_manifest(path, labels, expected_dataset="svhn")


def test_changed_labels_fail_checksum_validation(
    labels: list[int], tmp_path: Path
) -> None:
    path = tmp_path / "split.json"
    save_split_manifest(create_stratified_split(labels, 200, 42), path)
    changed = labels.copy()
    changed[0], changed[-1] = changed[-1], changed[0]
    with pytest.raises(ValueError, match="labels_sha256"):
        load_split_manifest(path, changed)


def _manifest_json(labels: list[int], tmp_path: Path) -> tuple[Path, dict[str, object]]:
    path = tmp_path / "split.json"
    save_split_manifest(create_stratified_split(labels, 200, 42), path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_duplicate_index_fails(labels: list[int], tmp_path: Path) -> None:
    path, payload = _manifest_json(labels, tmp_path)
    train = payload["train_indices"]
    assert isinstance(train, list)
    train[1] = train[0]
    _write_manifest(path, payload)
    with pytest.raises(ValueError, match="duplicate"):
        load_split_manifest(path, labels)


def test_overlapping_indices_fail(labels: list[int], tmp_path: Path) -> None:
    path, payload = _manifest_json(labels, tmp_path)
    train = payload["train_indices"]
    validation = payload["validation_indices"]
    assert isinstance(train, list) and isinstance(validation, list)
    validation[0] = train[0]
    _write_manifest(path, payload)
    with pytest.raises(ValueError, match="overlap"):
        load_split_manifest(path, labels)


def test_missing_index_fails(labels: list[int], tmp_path: Path) -> None:
    path, payload = _manifest_json(labels, tmp_path)
    train = payload["train_indices"]
    assert isinstance(train, list)
    train.pop()
    _write_manifest(path, payload)
    with pytest.raises(ValueError, match="cover"):
        load_split_manifest(path, labels)


def test_out_of_range_index_fails(labels: list[int], tmp_path: Path) -> None:
    path, payload = _manifest_json(labels, tmp_path)
    train = payload["train_indices"]
    assert isinstance(train, list)
    train[0] = len(labels)
    _write_manifest(path, payload)
    with pytest.raises(ValueError, match="outside"):
        load_split_manifest(path, labels)


def test_invalid_json_fails(labels: list[int], tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid split manifest JSON"):
        load_split_manifest(path, labels)


def test_unsupported_schema_fails(labels: list[int], tmp_path: Path) -> None:
    path, payload = _manifest_json(labels, tmp_path)
    payload["schema_version"] = 2
    _write_manifest(path, payload)
    with pytest.raises(ValueError, match="schema_version"):
        load_split_manifest(path, labels)


def test_missing_manifest_file_fails(labels: list[int], tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split_manifest(tmp_path / "missing.json", labels)


@pytest.mark.parametrize(
    ("invalid_labels", "validation_size"),
    [
        ([], 2),
        ("0011", 2),
        ([False, False, 1, 1], 2),
        ([-1, -1, 0, 0], 2),
        ([0, 0, 0], 1),
        ([0, 0, 1], 2),
    ],
)
def test_invalid_labels_fail(
    invalid_labels: object, validation_size: int
) -> None:
    with pytest.raises(ValueError):
        create_stratified_split(
            invalid_labels, validation_size, 42  # type: ignore[arg-type]
        )


def test_validation_size_must_be_divisible_by_classes(labels: list[int]) -> None:
    with pytest.raises(ValueError, match="divisible"):
        create_stratified_split(labels, 201, 42)


def test_validation_size_must_leave_training_samples(labels: list[int]) -> None:
    with pytest.raises(ValueError, match="retain"):
        create_stratified_split(labels, 1000, 42)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "42"])
def test_invalid_seed_fails(labels: list[int], seed: object) -> None:
    with pytest.raises(ValueError):
        create_stratified_split(labels, 200, seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("dataset", ["", "   "])
def test_empty_dataset_fails(labels: list[int], dataset: str) -> None:
    with pytest.raises(ValueError, match="dataset"):
        create_stratified_split(labels, 200, 42, dataset=dataset)
