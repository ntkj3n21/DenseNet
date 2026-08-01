"""Tests for CIFAR-10 transforms and manifest-backed dataset views."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

from PIL import Image
import pytest
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from densenet_experiments.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    build_cifar10_datasets,
    build_cifar10_eval_transform,
    build_cifar10_train_transform,
)
from densenet_experiments.data import cifar10 as cifar10_module


@pytest.fixture
def fake_cifar10(monkeypatch: pytest.MonkeyPatch) -> tuple[type[Dataset], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    class FakeCIFAR10(Dataset):
        train_size = 50_000
        test_size = 10_000

        def __init__(
            self,
            root: str | Path,
            train: bool,
            transform: Any,
            download: bool,
        ) -> None:
            self.root = root
            self.train = train
            self.transform = transform
            self.download = download
            size = self.train_size if train else self.test_size
            self.targets = [index % 10 for index in range(size)]
            calls.append(
                {
                    "instance": self,
                    "root": root,
                    "train": train,
                    "transform": transform,
                    "download": download,
                }
            )

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            image: Any = Image.new("RGB", (32, 32))
            if self.transform is not None:
                image = self.transform(image)
            return image, self.targets[index]

    monkeypatch.setattr(cifar10_module.datasets, "CIFAR10", FakeCIFAR10)
    return FakeCIFAR10, calls


def test_train_transform_order_and_constants() -> None:
    transform = build_cifar10_train_transform()
    assert [type(item) for item in transform.transforms] == [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    normalization = transform.transforms[-1]
    assert tuple(normalization.mean) == CIFAR10_MEAN
    assert tuple(normalization.std) == CIFAR10_STD


def test_eval_transform_order_and_constants() -> None:
    transform = build_cifar10_eval_transform()
    assert [type(item) for item in transform.transforms] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]
    normalization = transform.transforms[-1]
    assert tuple(normalization.mean) == CIFAR10_MEAN
    assert tuple(normalization.std) == CIFAR10_STD


def test_transform_builders_return_distinct_objects() -> None:
    first_train = build_cifar10_train_transform()
    second_train = build_cifar10_train_transform()
    first_eval = build_cifar10_eval_transform()
    second_eval = build_cifar10_eval_transform()
    assert first_train is not second_train
    assert first_eval is not second_eval
    assert first_train is not first_eval


def test_create_manifest_builds_expected_dataset_lengths(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifests" / "cifar10.json"
    result = build_cifar10_datasets(
        tmp_path / "data",
        manifest_path,
        create_manifest_if_missing=True,
    )
    assert len(result.train) == 45_000
    assert len(result.validation) == 5_000
    assert len(result.test) == 10_000
    assert manifest_path.is_file()


def test_validation_has_equal_class_counts(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    result = build_cifar10_datasets(
        tmp_path / "data",
        tmp_path / "split.json",
        create_manifest_if_missing=True,
    )
    validation_base = result.validation.dataset
    counts = Counter(
        validation_base.targets[index] for index in result.manifest.validation_indices
    )
    assert counts == {label: 500 for label in range(10)}


def test_train_validation_and_test_use_correct_dataset_views(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    result = build_cifar10_datasets(
        tmp_path / "data",
        tmp_path / "split.json",
        create_manifest_if_missing=True,
    )
    assert isinstance(result.train, Subset)
    assert isinstance(result.validation, Subset)
    assert result.train.dataset is not result.validation.dataset

    train_items = result.train.dataset.transform.transforms
    validation_items = result.validation.dataset.transform.transforms
    test_items = result.test.transform.transforms
    assert isinstance(train_items[0], transforms.RandomCrop)
    assert isinstance(train_items[1], transforms.RandomHorizontalFlip)
    assert [type(item) for item in validation_items] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert [type(item) for item in test_items] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]


def test_existing_manifest_reuses_identical_indices(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "split.json"
    first = build_cifar10_datasets(
        tmp_path / "data",
        manifest_path,
        create_manifest_if_missing=True,
    )
    second = build_cifar10_datasets(tmp_path / "data", manifest_path)
    assert first.manifest.train_indices == second.manifest.train_indices
    assert first.manifest.validation_indices == second.manifest.validation_indices


def test_missing_manifest_without_creation_flag_fails(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="automatic creation is disabled"):
        build_cifar10_datasets(tmp_path / "data", tmp_path / "missing.json")


def test_manifest_seed_mismatch_fails(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.json"
    build_cifar10_datasets(
        tmp_path / "data", path, create_manifest_if_missing=True
    )
    with pytest.raises(ValueError, match="seed"):
        build_cifar10_datasets(tmp_path / "data", path, split_seed=43)


def test_manifest_validation_size_mismatch_fails(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.json"
    build_cifar10_datasets(
        tmp_path / "data", path, create_manifest_if_missing=True
    )
    with pytest.raises(ValueError, match="validation_size"):
        build_cifar10_datasets(tmp_path / "data", path, validation_size=4_000)


def test_manifest_checksum_mismatch_fails(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.json"
    build_cifar10_datasets(
        tmp_path / "data", path, create_manifest_if_missing=True
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["labels_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="labels_sha256"):
        build_cifar10_datasets(tmp_path / "data", path)


def test_wrong_train_length_fails(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    fake_class, _ = fake_cifar10
    fake_class.train_size = 49_990
    with pytest.raises(ValueError, match="50,000 CIFAR-10 training"):
        build_cifar10_datasets(
            tmp_path / "data",
            tmp_path / "split.json",
            create_manifest_if_missing=True,
        )


def test_wrong_test_length_fails(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    fake_class, _ = fake_cifar10
    fake_class.test_size = 9_990
    with pytest.raises(ValueError, match="10,000 CIFAR-10 test samples"):
        build_cifar10_datasets(
            tmp_path / "data",
            tmp_path / "split.json",
            create_manifest_if_missing=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_size": 0},
        {"validation_size": -1},
        {"validation_size": True},
        {"split_seed": -1},
        {"split_seed": True},
        {"download": 1},
        {"create_manifest_if_missing": 1},
        {"data_root": ""},
        {"manifest_path": "   "},
    ],
)
def test_invalid_arguments_fail(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "data_root": tmp_path / "data",
        "manifest_path": tmp_path / "split.json",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        build_cifar10_datasets(**arguments)  # type: ignore[arg-type]


def test_cifar10_constructor_arguments_are_forwarded(
    fake_cifar10: tuple[type[Dataset], list[dict[str, Any]]],
    tmp_path: Path,
) -> None:
    _, calls = fake_cifar10
    data_root = tmp_path / "data"
    build_cifar10_datasets(
        data_root,
        tmp_path / "split.json",
        download=True,
        create_manifest_if_missing=True,
    )
    assert [call["train"] for call in calls] == [True, True, False]
    assert all(call["download"] is True for call in calls)
    assert all(call["root"] == data_root for call in calls)
