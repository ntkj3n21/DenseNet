"""Validated epoch-level training and evaluation metrics."""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class EpochMetrics:
    """Immutable sample-level metrics aggregated over one epoch."""

    loss: float
    accuracy: float
    num_samples: int
    num_correct: int

    def __post_init__(self) -> None:
        if isinstance(self.loss, bool) or not isinstance(self.loss, Real):
            raise ValueError(f"loss must be a real number, got {self.loss!r}.")
        if not math.isfinite(self.loss) or self.loss < 0.0:
            raise ValueError("loss must be finite and non-negative.")
        if isinstance(self.accuracy, bool) or not isinstance(self.accuracy, Real):
            raise ValueError(
                f"accuracy must be a real number, got {self.accuracy!r}."
            )
        if not math.isfinite(self.accuracy) or not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("accuracy must be finite and within [0.0, 1.0].")
        if (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, int)
            or self.num_samples <= 0
        ):
            raise ValueError("num_samples must be a positive integer.")
        if (
            isinstance(self.num_correct, bool)
            or not isinstance(self.num_correct, int)
            or self.num_correct < 0
        ):
            raise ValueError("num_correct must be a non-negative integer.")
        if self.num_correct > self.num_samples:
            raise ValueError("num_correct must not exceed num_samples.")

        expected_accuracy = self.num_correct / self.num_samples
        if not math.isclose(
            float(self.accuracy), expected_accuracy, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("accuracy is inconsistent with the sample counts.")

    @property
    def accuracy_percent(self) -> float:
        """Return top-1 accuracy as a percentage."""
        return float(self.accuracy) * 100.0
