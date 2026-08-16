"""Regenerate and verify derived benchmark CSV tables.

Raw run artifacts under results/<experiment>/<variant>/ are treated as the
source of truth. This script derives:

- results/pilot_20/summary.csv
- results/extended_50/summary.csv
- results/benchmark_summary.csv

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --write
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VARIANTS = ("baseline", "mish", "se", "mish_se")
EXPERIMENTS = ("pilot_20", "extended_50")

SUMMARY_FIELDS = (
    "variant",
    "seed",
    "epochs",
    "parameters",
    "best_epoch",
    "best_validation_accuracy",
    "delta_vs_baseline_pp",
    "final_train_accuracy",
    "final_validation_accuracy",
    "final_validation_loss",
    "elapsed_seconds",
    "elapsed_minutes",
)

BENCHMARK_FIELDS = (
    "variant",
    "parameters",
    "pilot_best_validation_accuracy",
    "pilot_best_epoch",
    "extended_best_validation_accuracy",
    "extended_best_epoch",
    "gain_20_to_50_pp",
    "extended_delta_vs_baseline_pp",
    "extended_final_validation_accuracy",
    "extended_elapsed_minutes",
)

METRIC_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
)


@dataclass(frozen=True)
class RunResult:
    variant: str
    seed: int
    epochs: int
    parameters: int
    best_epoch: int
    best_validation_accuracy: float
    final_train_accuracy: float
    final_validation_accuracy: float
    final_validation_loss: float
    elapsed_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary artifact: {path}")

    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return value


def _load_metrics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics artifact: {path}")

    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)

        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise ValueError(
                f"Unexpected metrics columns in {path}: {reader.fieldnames}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(f"No metric rows found in {path}")

    return rows


def _epochs_from_summary(summary: dict[str, Any]) -> int:
    if "epochs" in summary:
        return int(summary["epochs"])

    if "completed_epochs" in summary:
        return int(summary["completed_epochs"])

    if "target_epochs" in summary:
        return int(summary["target_epochs"])

    raise ValueError("Summary does not contain an epoch count.")


def _load_run(run_directory: Path, expected_variant: str) -> RunResult:
    summary = _load_json(run_directory / "summary.json")
    metrics = _load_metrics(run_directory / "metrics.csv")

    variant = str(summary["variant"])
    if variant != expected_variant:
        raise ValueError(
            f"Variant mismatch in {run_directory}: "
            f"{variant!r} != {expected_variant!r}"
        )

    epochs = _epochs_from_summary(summary)

    if len(metrics) != epochs:
        raise ValueError(
            f"Metric row count mismatch in {run_directory}: "
            f"{len(metrics)} != {epochs}"
        )

    epoch_numbers = [int(row["epoch"]) for row in metrics]
    if epoch_numbers != list(range(1, epochs + 1)):
        raise ValueError(
            f"Metrics contain missing or reordered epochs in {run_directory}"
        )

    validation_accuracies = [
        float(row["validation_accuracy"])
        for row in metrics
    ]

    derived_best_accuracy = max(validation_accuracies)
    derived_best_epoch = (
        validation_accuracies.index(derived_best_accuracy) + 1
    )

    summary_best_accuracy = float(
        summary["best_validation_accuracy"]
    )
    summary_best_epoch = int(summary["best_epoch"])

    if abs(derived_best_accuracy - summary_best_accuracy) > 1e-12:
        raise ValueError(
            f"Best validation accuracy mismatch in {run_directory}: "
            f"metrics={derived_best_accuracy}, "
            f"summary={summary_best_accuracy}"
        )

    if derived_best_epoch != summary_best_epoch:
        raise ValueError(
            f"Best epoch mismatch in {run_directory}: "
            f"metrics={derived_best_epoch}, "
            f"summary={summary_best_epoch}"
        )

    final = metrics[-1]

    return RunResult(
        variant=variant,
        seed=int(summary["seed"]),
        epochs=epochs,
        parameters=int(summary["parameters"]),
        best_epoch=summary_best_epoch,
        best_validation_accuracy=summary_best_accuracy,
        final_train_accuracy=float(final["train_accuracy"]),
        final_validation_accuracy=float(
            final["validation_accuracy"]
        ),
        final_validation_loss=float(final["validation_loss"]),
        elapsed_seconds=float(summary["elapsed_seconds"]),
    )


def _summary_rows(
    results_root: Path,
    experiment: str,
) -> list[dict[str, object]]:
    runs = [
        _load_run(
            results_root / experiment / variant,
            variant,
        )
        for variant in VARIANTS
    ]

    baseline = runs[0].best_validation_accuracy

    rows: list[dict[str, object]] = []

    for run in runs:
        rows.append(
            {
                "variant": run.variant,
                "seed": run.seed,
                "epochs": run.epochs,
                "parameters": run.parameters,
                "best_epoch": run.best_epoch,
                "best_validation_accuracy": (
                    run.best_validation_accuracy
                ),
                "delta_vs_baseline_pp": round(
                    (
                        run.best_validation_accuracy
                        - baseline
                    )
                    * 100,
                    2,
                ),
                "final_train_accuracy": round(
                    run.final_train_accuracy,
                    4,
                ),
                "final_validation_accuracy": round(
                    run.final_validation_accuracy,
                    6,
                ),
                "final_validation_loss": round(
                    run.final_validation_loss,
                    6,
                ),
                "elapsed_seconds": round(
                    run.elapsed_seconds,
                    3,
                ),
                "elapsed_minutes": round(
                    run.elapsed_seconds / 60,
                    2,
                ),
            }
        )

    return rows


def _benchmark_rows(
    pilot_rows: list[dict[str, object]],
    extended_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    pilot = {
        str(row["variant"]): row
        for row in pilot_rows
    }
    extended = {
        str(row["variant"]): row
        for row in extended_rows
    }

    rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        pilot_row = pilot[variant]
        extended_row = extended[variant]

        pilot_best = float(
            pilot_row["best_validation_accuracy"]
        )
        extended_best = float(
            extended_row["best_validation_accuracy"]
        )

        rows.append(
            {
                "variant": variant,
                "parameters": int(
                    extended_row["parameters"]
                ),
                "pilot_best_validation_accuracy": pilot_best,
                "pilot_best_epoch": int(
                    pilot_row["best_epoch"]
                ),
                "extended_best_validation_accuracy": (
                    extended_best
                ),
                "extended_best_epoch": int(
                    extended_row["best_epoch"]
                ),
                "gain_20_to_50_pp": round(
                    (extended_best - pilot_best) * 100,
                    2,
                ),
                "extended_delta_vs_baseline_pp": float(
                    extended_row["delta_vs_baseline_pp"]
                ),
                "extended_final_validation_accuracy": float(
                    extended_row["final_validation_accuracy"]
                ),
                "extended_elapsed_minutes": float(
                    extended_row["elapsed_minutes"]
                ),
            }
        )

    return rows


def _render_csv(
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> str:
    buffer = io.StringIO(newline="")

    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(rows)

    return buffer.getvalue()


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _check(path: Path, expected: str) -> bool:
    if not path.is_file():
        print(f"MISSING  {path}")
        return False

    actual = path.read_text(encoding="utf-8")

    if actual == expected:
        print(f"OK       {path}")
        return True

    print(f"STALE    {path}")
    return False


def build_outputs(
    results_root: Path,
) -> dict[Path, str]:
    pilot_rows = _summary_rows(
        results_root,
        "pilot_20",
    )
    extended_rows = _summary_rows(
        results_root,
        "extended_50",
    )
    benchmark_rows = _benchmark_rows(
        pilot_rows,
        extended_rows,
    )

    return {
        results_root / "pilot_20" / "summary.csv": (
            _render_csv(
                SUMMARY_FIELDS,
                pilot_rows,
            )
        ),
        results_root / "extended_50" / "summary.csv": (
            _render_csv(
                SUMMARY_FIELDS,
                extended_rows,
            )
        ),
        results_root / "benchmark_summary.csv": (
            _render_csv(
                BENCHMARK_FIELDS,
                benchmark_rows,
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate or verify derived DenseNet benchmark tables."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Result directory. Default: results",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write regenerated CSV files instead of only checking them.",
    )

    args = parser.parse_args()

    outputs = build_outputs(args.results_root)

    if args.write:
        for path, content in outputs.items():
            _write(path, content)
            print(f"WROTE    {path}")
        return 0

    success = True

    for path, expected in outputs.items():
        success = _check(path, expected) and success

    if success:
        print("All derived result tables are up to date.")
        return 0

    print(
        "Derived tables are stale. "
        "Run with --write to regenerate them."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())