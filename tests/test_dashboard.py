from __future__ import annotations

import pandas as pd
import pytest

from dp.dashboard import (
    best_nonprivate,
    configuration_label,
    dp_budget_row,
    group_rate_rows,
    headline_statistics,
    load_summary,
)


def _summary() -> pd.DataFrame:
    rows = []

    def add(task: str, model: str, epsilon: float, metric: str, mean: float) -> None:
        rows.append(
            {
                "task": task,
                "model": model,
                "epsilon": epsilon,
                "metric": metric,
                "mean": mean,
                "ci_lower": mean - 0.01,
                "ci_upper": mean + 0.01,
                "n_runs": 5,
            }
        )

    add("smoker_with_charges_legacy", "gradient_boosting", float("inf"), "roc_auc", 0.99)
    add("smoker_without_charges", "logistic", float("inf"), "roc_auc", 0.54)
    add("high_cost", "gradient_boosting", float("inf"), "roc_auc", 0.95)
    add("high_cost", "dpsgd", 0.5, "roc_auc", 0.93)
    add("high_cost", "dpsgd", 6.0, "roc_auc", 0.94)
    add("smoker_without_charges", "dpsgd", 0.5, "demographic_parity_difference", 0.06)
    add("smoker_without_charges", "dpsgd", 6.0, "demographic_parity_difference", 0.21)
    add("high_cost", "dpsgd", 0.5, "tpr_group_female", 0.80)
    add("high_cost", "dpsgd", 0.5, "tpr_group_male", 0.84)
    return pd.DataFrame(rows)


def test_load_summary_validates_required_columns(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame({"task": ["high_cost"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_summary(path)


def test_best_nonprivate_and_dp_budget_selection():
    summary = _summary()
    assert best_nonprivate(summary, "high_cost")["mean"] == pytest.approx(0.95)
    assert dp_budget_row(summary, "high_cost", strictest=True)["epsilon"] == pytest.approx(0.5)
    assert dp_budget_row(summary, "high_cost", strictest=False)["epsilon"] == pytest.approx(6.0)


def test_headline_statistics_are_computed_from_rows():
    stats = headline_statistics(_summary())
    assert stats["proxy_drop"] == pytest.approx(-0.45)
    assert stats["high_cost_retention"] == pytest.approx(0.93 / 0.95)
    assert stats["fairness_change"] == pytest.approx(0.15)


def test_group_rate_rows_extracts_group_name():
    rows = group_rate_rows(_summary(), "high_cost", "tpr")
    assert set(rows["group"]) == {"Female", "Male"}


def test_configuration_label_includes_achieved_epsilon():
    label = configuration_label(pd.Series({"model": "dpsgd", "epsilon": 1.9979}))
    assert label == "DP-SGD neural network (ε=2.00)"
