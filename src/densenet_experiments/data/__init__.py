"""Dataset-independent split utilities exposed by the package."""

from .splits import (
    SplitManifest,
    compute_labels_sha256,
    create_stratified_split,
    load_split_manifest,
    save_split_manifest,
)

__all__ = [
    "SplitManifest",
    "compute_labels_sha256",
    "create_stratified_split",
    "save_split_manifest",
    "load_split_manifest",
]
