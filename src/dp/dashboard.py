"""Data preparation helpers for the Streamlit benchmark explorer.

The dashboard is deliberately read-only: it visualises versioned benchmark
outputs and never retrains models or accepts sensitive user data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_SUMMARY_COLUMNS = {
    "task",
    "model",
    "epsilon",
    "metric",
    "mean",
    "ci_lower",
    "ci_upper",
    "n_runs",
}

NONPRIVATE_MODELS = ("dummy", "logistic", "svm", "gradient_boosting", "neural")
UTILITY_METRICS = ("roc_auc", "pr_auc", "macro_f1", "brier", "ece")
FAIRNESS_METRICS = (
    "demographic_parity_difference",
    "equalized_odds_difference",
)
AUDIT_METRICS = ("mia_loss_threshold_auc", "mia_tpr_at_fpr_0.01")

TASK_LABELS = {
    "high_cost": "High-cost classification",
    "smoker_without_charges": "Smoker prediction — charges removed",
    "smoker_with_charges_legacy": "Smoker prediction — legacy proxy task",
}
MODEL_LABELS = {
    "dummy": "Dummy baseline",
    "logistic": "Logistic regression",
    "svm": "RBF SVM",
    "gradient_boosting": "Histogram gradient boosting",
    "neural": "Neural network",
    "dpsgd": "DP-SGD neural network",
}
METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "macro_f1": "Macro F1",
    "brier": "Brier score",
    "ece": "Expected calibration error",
    "demographic_parity_difference": "Demographic parity difference",
    "equalized_odds_difference": "Equalized odds difference",
    "mia_loss_threshold_auc": "Membership-inference AUC",
    "mia_tpr_at_fpr_0.01": "MIA TPR at 1% FPR",
}


def load_summary(path: str | Path) -> pd.DataFrame:
    """Load and validate the generated benchmark summary."""
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Benchmark summary not found at {summary_path}. "
            "Run `python -m dp.benchmark` and `python -m dp.reporting` first."
        )

    frame = pd.read_csv(summary_path)
    missing = REQUIRED_SUMMARY_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Summary is missing required columns: {sorted(missing)}")

    numeric = ["epsilon", "mean", "ci_lower", "ci_upper", "n_runs"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[["mean", "ci_lower", "ci_upper"]].isna().all(axis=None):
        raise ValueError("Summary contains no numeric metric estimates.")

    return frame.sort_values(["task", "metric", "model", "epsilon"]).reset_index(drop=True)


def metric_rows(
    summary: pd.DataFrame,
    *,
    task: str,
    metric: str,
    models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return one task/metric slice with presentation labels."""
    subset = summary[(summary["task"] == task) & (summary["metric"] == metric)].copy()
    if models is not None:
        subset = subset[subset["model"].isin(tuple(models))]

    subset["task_label"] = subset["task"].map(TASK_LABELS).fillna(subset["task"])
    subset["model_label"] = subset["model"].map(MODEL_LABELS).fillna(subset["model"])
    subset["metric_label"] = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    subset["epsilon_label"] = subset["epsilon"].map(
        lambda value: f"{value:.3f}" if np.isfinite(value) else "Non-private"
    )
    subset["configuration"] = subset.apply(configuration_label, axis=1)
    return subset.reset_index(drop=True)


def configuration_label(row: pd.Series) -> str:
    """Human-readable model label including achieved epsilon for private runs."""
    model = MODEL_LABELS.get(str(row["model"]), str(row["model"]))
    epsilon = float(row["epsilon"])
    if np.isfinite(epsilon):
        return f"{model} (ε={epsilon:.2f})"
    return model


def best_nonprivate(summary: pd.DataFrame, task: str, metric: str = "roc_auc") -> pd.Series:
    """Return the strongest non-private, non-dummy configuration for a metric."""
    subset = metric_rows(
        summary,
        task=task,
        metric=metric,
        models=("logistic", "svm", "gradient_boosting", "neural"),
    )
    if subset.empty:
        raise ValueError(f"No non-private rows found for task={task!r}, metric={metric!r}")

    lower_is_better = metric in {"brier", "ece"}
    index = subset["mean"].idxmin() if lower_is_better else subset["mean"].idxmax()
    return subset.loc[index]


def dp_budget_row(
    summary: pd.DataFrame,
    task: str,
    metric: str = "roc_auc",
    *,
    strictest: bool = True,
) -> pd.Series:
    """Return the strictest or loosest available DP-SGD budget row."""
    subset = metric_rows(summary, task=task, metric=metric, models=("dpsgd",))
    subset = subset[np.isfinite(subset["epsilon"])]
    if subset.empty:
        raise ValueError(f"No DP-SGD rows found for task={task!r}, metric={metric!r}")
    index = subset["epsilon"].idxmin() if strictest else subset["epsilon"].idxmax()
    return subset.loc[index]


def headline_statistics(summary: pd.DataFrame) -> dict[str, float | str]:
    """Compute the dashboard's headline values from generated results."""
    legacy = best_nonprivate(summary, "smoker_with_charges_legacy")
    corrected = best_nonprivate(summary, "smoker_without_charges")
    high_cost_best = best_nonprivate(summary, "high_cost")
    high_cost_dp = dp_budget_row(summary, "high_cost", strictest=True)

    fairness = metric_rows(
        summary,
        task="smoker_without_charges",
        metric="demographic_parity_difference",
        models=("dpsgd",),
    ).sort_values("epsilon")
    fairness_change = np.nan
    if len(fairness) >= 2:
        fairness_change = float(fairness.iloc[-1]["mean"] - fairness.iloc[0]["mean"])

    return {
        "legacy_roc_auc": float(legacy["mean"]),
        "corrected_roc_auc": float(corrected["mean"]),
        "proxy_drop": float(corrected["mean"] - legacy["mean"]),
        "high_cost_best_roc_auc": float(high_cost_best["mean"]),
        "high_cost_dp_roc_auc": float(high_cost_dp["mean"]),
        "high_cost_dp_epsilon": float(high_cost_dp["epsilon"]),
        "high_cost_retention": float(high_cost_dp["mean"] / high_cost_best["mean"]),
        "fairness_change": fairness_change,
        "best_high_cost_model": str(high_cost_best["model_label"]),
    }


def available_metrics(summary: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    """Return candidate metrics that are present in the current results."""
    present = set(summary["metric"].dropna().astype(str))
    return [metric for metric in candidates if metric in present]


def group_rate_rows(summary: pd.DataFrame, task: str, rate: str) -> pd.DataFrame:
    """Return female/male DP-SGD TPR or FPR rows with a group column."""
    if rate not in {"tpr", "fpr"}:
        raise ValueError("rate must be 'tpr' or 'fpr'")

    metrics = [f"{rate}_group_female", f"{rate}_group_male"]
    subset = summary[
        (summary["task"] == task)
        & (summary["model"] == "dpsgd")
        & summary["metric"].isin(metrics)
    ].copy()
    subset["group"] = subset["metric"].str.rsplit("_", n=1).str[-1].str.title()
    return subset.sort_values(["group", "epsilon"]).reset_index(drop=True)
