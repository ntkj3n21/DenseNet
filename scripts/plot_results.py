"""Regenerate benchmark figures from tracked result artifacts.

The script reads the committed CSV result files and produces the five figures
used by the project documentation:

- validation_accuracy_50.png
- validation_loss_50.png
- best_validation_accuracy.png
- pilot_vs_extended.png
- best_model_train_vs_validation.png

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --output-dir /tmp/densenet-figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


VARIANTS = ("baseline", "mish", "se", "mish_se")

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "mish": "Mish",
    "se": "SE",
    "mish_se": "Mish + SE",
}

EXPECTED_METRIC_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return rows


def load_metrics(path: Path) -> list[dict[str, float]]:
    rows = load_csv(path)

    fields = tuple(rows[0].keys())
    if fields != EXPECTED_METRIC_FIELDS:
        raise ValueError(
            f"Unexpected metrics columns in {path}: {fields}"
        )

    metrics: list[dict[str, float]] = []

    for row in rows:
        metrics.append(
            {
                "epoch": float(row["epoch"]),
                "learning_rate": float(row["learning_rate"]),
                "train_loss": float(row["train_loss"]),
                "train_accuracy": float(row["train_accuracy"]),
                "validation_loss": float(row["validation_loss"]),
                "validation_accuracy": float(
                    row["validation_accuracy"]
                ),
            }
        )

    expected_epochs = list(range(1, len(metrics) + 1))
    actual_epochs = [int(row["epoch"]) for row in metrics]

    if actual_epochs != expected_epochs:
        raise ValueError(
            f"Missing or reordered epochs in {path}"
        )

    return metrics


def column(
    rows: Iterable[dict[str, float]],
    key: str,
    *,
    percentage: bool = False,
) -> list[float]:
    values = [float(row[key]) for row in rows]

    if percentage:
        return [value * 100.0 for value in values]

    return values


def finish_figure(
    fig: plt.Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"WROTE    {path}")


def plot_validation_accuracy(
    results_root: Path,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant in VARIANTS:
        metrics = load_metrics(
            results_root
            / "extended_50"
            / variant
            / "metrics.csv"
        )

        ax.plot(
            column(metrics, "epoch"),
            column(
                metrics,
                "validation_accuracy",
                percentage=True,
            ),
            label=DISPLAY_NAMES[variant],
            linewidth=2,
        )

    ax.set_title("50-Epoch Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    finish_figure(
        fig,
        output_dir / "validation_accuracy_50.png",
    )


def plot_validation_loss(
    results_root: Path,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant in VARIANTS:
        metrics = load_metrics(
            results_root
            / "extended_50"
            / variant
            / "metrics.csv"
        )

        ax.plot(
            column(metrics, "epoch"),
            column(metrics, "validation_loss"),
            label=DISPLAY_NAMES[variant],
            linewidth=2,
        )

    ax.set_title("50-Epoch Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    finish_figure(
        fig,
        output_dir / "validation_loss_50.png",
    )


def plot_best_validation_accuracy(
    results_root: Path,
    output_dir: Path,
) -> None:
    rows = load_csv(
        results_root / "extended_50" / "summary.csv"
    )

    by_variant = {
        row["variant"]: row
        for row in rows
    }

    names = [
        DISPLAY_NAMES[variant]
        for variant in VARIANTS
    ]
    values = [
        float(
            by_variant[variant][
                "best_validation_accuracy"
            ]
        )
        * 100.0
        for variant in VARIANTS
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, values)

    ax.set_title("Best Validation Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(
        min(values) - 1.0,
        max(values) + 1.0,
    )
    ax.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    finish_figure(
        fig,
        output_dir / "best_validation_accuracy.png",
    )


def plot_pilot_vs_extended(
    results_root: Path,
    output_dir: Path,
) -> None:
    rows = load_csv(
        results_root / "benchmark_summary.csv"
    )

    by_variant = {
        row["variant"]: row
        for row in rows
    }

    names = [
        DISPLAY_NAMES[variant]
        for variant in VARIANTS
    ]

    pilot = [
        float(
            by_variant[variant][
                "pilot_best_validation_accuracy"
            ]
        )
        * 100.0
        for variant in VARIANTS
    ]

    extended = [
        float(
            by_variant[variant][
                "extended_best_validation_accuracy"
            ]
        )
        * 100.0
        for variant in VARIANTS
    ]

    positions = list(range(len(VARIANTS)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    pilot_bars = ax.bar(
        [position - width / 2 for position in positions],
        pilot,
        width,
        label="20-Epoch Pilot",
    )

    extended_bars = ax.bar(
        [position + width / 2 for position in positions],
        extended,
        width,
        label="50-Epoch Extended",
    )

    ax.set_title("Effect of Training Horizon")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels(names)

    all_values = pilot + extended
    ax.set_ylim(
        min(all_values) - 1.0,
        max(all_values) + 1.0,
    )

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for bars in (pilot_bars, extended_bars):
        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.07,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    finish_figure(
        fig,
        output_dir / "pilot_vs_extended.png",
    )


def plot_best_model_training_behavior(
    results_root: Path,
    output_dir: Path,
) -> None:
    metrics = load_metrics(
        results_root
        / "extended_50"
        / "mish"
        / "metrics.csv"
    )

    epochs = column(metrics, "epoch")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        epochs,
        column(
            metrics,
            "train_accuracy",
            percentage=True,
        ),
        label="Training Accuracy",
        linewidth=2,
    )

    ax.plot(
        epochs,
        column(
            metrics,
            "validation_accuracy",
            percentage=True,
        ),
        label="Validation Accuracy",
        linewidth=2,
    )

    best_validation = max(
        column(
            metrics,
            "validation_accuracy",
            percentage=True,
        )
    )

    best_epoch = (
        column(
            metrics,
            "validation_accuracy",
            percentage=True,
        ).index(best_validation)
        + 1
    )

    ax.scatter(
        [best_epoch],
        [best_validation],
        zorder=3,
    )

    ax.annotate(
        f"Best: {best_validation:.2f}% @ epoch {best_epoch}",
        xy=(best_epoch, best_validation),
        xytext=(best_epoch - 17, best_validation - 6),
        arrowprops={"arrowstyle": "->"},
    )

    ax.set_title("Mish Training vs Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    finish_figure(
        fig,
        output_dir
        / "best_model_train_vs_validation.png",
    )


def generate_all(
    results_root: Path,
    output_dir: Path,
) -> None:
    plot_validation_accuracy(
        results_root,
        output_dir,
    )
    plot_validation_loss(
        results_root,
        output_dir,
    )
    plot_best_validation_accuracy(
        results_root,
        output_dir,
    )
    plot_pilot_vs_extended(
        results_root,
        output_dir,
    )
    plot_best_model_training_behavior(
        results_root,
        output_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate DenseNet benchmark figures "
            "from tracked result CSV files."
        )
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Result root directory. Default: results",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help=(
            "Directory for generated PNG files. "
            "Default: results/figures"
        ),
    )

    args = parser.parse_args()

    generate_all(
        results_root=args.results_root,
        output_dir=args.output_dir,
    )

    print("All benchmark figures generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())