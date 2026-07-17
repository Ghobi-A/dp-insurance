"""Uncertainty utilities: repeated-seed and bootstrap confidence intervals.

Two flavours of interval are provided and must not be conflated:

* :func:`seed_confidence_interval` / :func:`summarize_runs` — intervals over
  *repeated model seeds*.  Seeds are not independent samples from a
  population, so these intervals describe run-to-run variability of the
  pipeline, **not** an independent-sample statistical guarantee about the
  metric.  They are labelled ``method="repeated_seed_t"`` in outputs.
* :func:`bootstrap_confidence_interval` — a percentile bootstrap over test
  examples for a single fitted model, labelled ``method="bootstrap_percentile"``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

#: Columns of the tidy per-run results schema.
TIDY_COLUMNS = (
    "task",
    "model",
    "privacy_mechanism",
    "epsilon",
    "delta",
    "seed",
    "metric",
    "value",
)

#: Columns of the summary schema.
SUMMARY_COLUMNS = (
    "task",
    "model",
    "privacy_mechanism",
    "epsilon",
    "delta",
    "metric",
    "mean",
    "std",
    "ci_lower",
    "ci_upper",
    "n_runs",
    "method",
)


def seed_confidence_interval(
    values: Sequence[float], confidence: float = 0.95
) -> tuple[float, float, float, float]:
    """Mean, std and t-based interval across repeated-seed runs.

    Returns ``(mean, std, ci_lower, ci_upper)`` using the Student-t
    quantile on ``n - 1`` degrees of freedom.  With a single run the
    interval collapses to the point estimate.  NaN values are dropped.

    This is a *repeated-seed* interval: it quantifies variability across
    seeds of the same pipeline on the same data, not sampling uncertainty
    over independent datasets.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0, mean, mean
    std = float(arr.std(ddof=1))
    t = float(stats.t.ppf(0.5 + confidence / 2, df=arr.size - 1))
    half = t * std / np.sqrt(arr.size)
    return mean, std, mean - half, mean + half


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap over test examples for one fitted model.

    Returns ``(point_estimate, ci_lower, ci_upper)``.  Resamples that end
    up single-class (where the metric raises) are skipped.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(random_state)
    point = float(metric_fn(y_true, y_score))
    stats_: list[float] = []
    n = y_true.size
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            stats_.append(float(metric_fn(y_true[idx], y_score[idx])))
        except ValueError:
            continue
    if not stats_:
        return point, float("nan"), float("nan")
    alpha = 1 - confidence
    lower, upper = np.quantile(stats_, [alpha / 2, 1 - alpha / 2])
    return point, float(lower), float(upper)


def summarize_runs(tidy: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Aggregate tidy per-seed results into the summary schema.

    Groups by everything except ``seed``/``value`` and computes the
    repeated-seed mean, std and t-interval.  The ``method`` column records
    ``"repeated_seed_t"`` so downstream reports label the intervals
    accurately.

    Raises:
        ValueError: If required tidy columns are missing.
    """
    missing = [c for c in TIDY_COLUMNS if c not in tidy.columns]
    if missing:
        raise ValueError(f"Tidy results missing columns: {missing}")
    group_cols = ["task", "model", "privacy_mechanism", "epsilon", "delta", "metric"]
    rows: list[dict[str, object]] = []
    for keys, grp in tidy.groupby(group_cols, dropna=False):
        mean, std, lo, hi = seed_confidence_interval(grp["value"], confidence)
        row = dict(zip(group_cols, keys))
        row.update(
            mean=mean,
            std=std,
            ci_lower=lo,
            ci_upper=hi,
            n_runs=int(grp["seed"].nunique()),
            method="repeated_seed_t",
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
