"""Tests for the multi-task benchmark: tasks, metrics, uncertainty, CLI."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from dp.benchmark import (
    build_sklearn_model,
    export_results,
    main,
    run_benchmark,
    run_single_seed,
)
from dp.metrics import (
    classification_metrics,
    expected_calibration_error,
    group_fairness_report,
    select_threshold,
)
from dp.pipeline import build_preprocessor
from dp.tasks import TASKS, get_task, prepare_task_data
from dp.uncertainty import (
    SUMMARY_COLUMNS,
    TIDY_COLUMNS,
    bootstrap_confidence_interval,
    seed_confidence_interval,
    summarize_runs,
)


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """Small synthetic dataset with the insurance schema."""
    rng = np.random.default_rng(0)
    n = 160
    age = rng.integers(18, 65, size=n)
    bmi = rng.normal(30, 5, size=n)
    smoker = rng.random(n) < 0.25
    charges = 200 * age + 5000 * bmi / 30 + 20000 * smoker + rng.normal(0, 1000, n)
    return pd.DataFrame(
        {
            "age": age,
            "sex": rng.choice(["male", "female"], size=n),
            "bmi": bmi,
            "children": rng.integers(0, 4, size=n),
            "smoker": np.where(smoker, "yes", "no"),
            "region": rng.choice(["ne", "nw", "se", "sw"], size=n),
            "charges": charges,
        }
    )


class TestTasks:
    def test_known_tasks_registered(self):
        assert {"smoker_without_charges", "high_cost", "smoker_with_charges_legacy"} <= set(TASKS)

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nope")

    @pytest.mark.parametrize("name", ["smoker_without_charges", "high_cost"])
    def test_charges_excluded(self, synthetic_df, name):
        data = prepare_task_data(synthetic_df, get_task(name), random_state=0)
        for X in (data.X_train, data.X_val, data.X_test):
            assert "charges" not in X.columns

    def test_target_not_in_features(self, synthetic_df):
        for name in TASKS:
            data = prepare_task_data(synthetic_df, get_task(name), random_state=0)
            assert get_task(name).target_column not in data.X_train.columns or name == "high_cost"
            if name != "high_cost":
                assert get_task(name).target_column not in data.X_train.columns

    def test_legacy_task_keeps_charges(self, synthetic_df):
        data = prepare_task_data(
            synthetic_df, get_task("smoker_with_charges_legacy"), random_state=0
        )
        assert "charges" in data.X_train.columns
        assert TASKS["smoker_with_charges_legacy"].legacy

    def test_high_cost_threshold_training_only(self, synthetic_df):
        data = prepare_task_data(synthetic_df, get_task("high_cost"), random_state=0)
        threshold = data.metadata["threshold"]
        # The threshold must be the median of the training charges, which we
        # can reconstruct from the training index, not the full-data median.
        train_median = synthetic_df.loc[data.X_train.index, "charges"].median()
        assert threshold == pytest.approx(train_median)
        assert threshold != pytest.approx(synthetic_df["charges"].median())
        # And targets must be consistent with that threshold.
        expected = (synthetic_df.loc[data.X_test.index, "charges"] > threshold).astype(int)
        np.testing.assert_array_equal(data.y_test, expected.to_numpy())

    def test_partitions_disjoint(self, synthetic_df):
        data = prepare_task_data(synthetic_df, get_task("high_cost"), random_state=0)
        idx = [set(x.index) for x in (data.X_train, data.X_val, data.X_test)]
        assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
        assert sum(map(len, idx)) == len(synthetic_df)

    def test_stratification_preserved_for_smoker(self, synthetic_df):
        data = prepare_task_data(
            synthetic_df, get_task("smoker_without_charges"), random_state=0
        )
        overall = (synthetic_df["smoker"] == "yes").mean()
        for y in (data.y_train, data.y_val, data.y_test):
            assert abs(y.mean() - overall) < 0.05

    def test_missing_column_raises(self, synthetic_df):
        with pytest.raises(ValueError, match="missing required columns"):
            prepare_task_data(
                synthetic_df.drop(columns=["charges"]), get_task("high_cost")
            )


class TestPreprocessing:
    def test_stable_column_order(self, synthetic_df):
        X = synthetic_df.drop(columns=["smoker", "charges"])
        a = build_preprocessor(X).fit_transform(X)
        b = build_preprocessor(X).fit_transform(X)
        np.testing.assert_array_equal(a, b)

    def test_pipeline_fits_on_train_only(self, synthetic_df):
        data = prepare_task_data(synthetic_df, get_task("high_cost"), random_state=0)
        model = build_sklearn_model("logistic", data.X_train, seed=0)
        model.fit(data.X_train, data.y_train)
        scaler = model.named_steps["preprocess"].named_transformers_["num"]
        assert scaler.mean_[0] == pytest.approx(data.X_train["age"].mean())


class TestMetrics:
    def test_ece_perfect_calibration_zero(self):
        y_score = np.array([0.0, 0.0, 1.0, 1.0])
        y_true = np.array([0, 0, 1, 1])
        assert expected_calibration_error(y_true, y_score) == pytest.approx(0.0)

    def test_ece_known_value(self):
        # All predictions 0.75, actual rate 0.5 -> ECE = 0.25.
        y_score = np.full(4, 0.75)
        y_true = np.array([1, 0, 1, 0])
        assert expected_calibration_error(y_true, y_score) == pytest.approx(0.25)

    def test_ece_validates(self):
        with pytest.raises(ValueError):
            expected_calibration_error(np.array([]), np.array([]))

    def test_classification_metrics_perfect(self):
        y = np.array([0, 1, 0, 1])
        s = np.array([0.1, 0.9, 0.2, 0.8])
        m = classification_metrics(y, s, threshold=0.5)
        assert m["roc_auc"] == 1.0 and m["f1"] == 1.0 and m["accuracy"] == 1.0
        assert m["brier"] == pytest.approx(np.mean((s - y) ** 2))

    def test_select_threshold_uses_given_partition(self):
        y = np.array([0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        t = select_threshold(y, s)
        assert 0.3 < t <= 0.8

    def test_group_fairness_small_group_warning(self, caplog):
        y_true = np.array([1, 0] * 20)
        y_pred = np.array([1, 0] * 20)
        sensitive = np.array(["a"] * 38 + ["b"] * 2)
        with caplog.at_level(logging.WARNING, logger="dp.metrics"):
            report = group_fairness_report(y_true, y_pred, sensitive)
        assert report.small_groups == ("b",)
        assert any("unstable" in r.message for r in caplog.records)
        assert report.group_sizes == {"a": 38, "b": 2}

    def test_group_fairness_nan_for_empty_class(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        sensitive = np.array(["a", "a", "b", "b"])  # group b has no positives
        report = group_fairness_report(y_true, y_pred, sensitive)
        assert np.isnan(report.group_tpr["b"])
        assert np.isnan(report.group_fpr["a"])


class TestUncertainty:
    def test_seed_ci_known_values(self):
        mean, std, lo, hi = seed_confidence_interval([1.0, 2.0, 3.0])
        assert mean == pytest.approx(2.0)
        assert std == pytest.approx(1.0)
        # t(0.975, df=2) = 4.3027; half-width = 4.3027/sqrt(3)
        assert hi - mean == pytest.approx(4.302652 / np.sqrt(3), rel=1e-4)
        assert lo == pytest.approx(mean - (hi - mean))

    def test_seed_ci_single_value(self):
        assert seed_confidence_interval([0.7]) == (0.7, 0.0, 0.7, 0.7)

    def test_seed_ci_drops_nan(self):
        mean, *_ = seed_confidence_interval([1.0, np.nan, 3.0])
        assert mean == pytest.approx(2.0)

    def test_bootstrap_ci_brackets_point(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        s = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
        from sklearn.metrics import roc_auc_score

        point, lo, hi = bootstrap_confidence_interval(
            y, s, roc_auc_score, n_bootstrap=200, random_state=0
        )
        assert lo <= point <= hi

    def test_summarize_runs_schema_and_method_label(self):
        tidy = pd.DataFrame(
            {
                "task": ["t"] * 4,
                "model": ["m"] * 4,
                "privacy_mechanism": ["none"] * 4,
                "epsilon": [np.inf] * 4,
                "delta": [np.nan] * 4,
                "seed": [0, 1, 2, 3],
                "metric": ["roc_auc"] * 4,
                "value": [0.8, 0.82, 0.79, 0.81],
            }
        )
        summary = summarize_runs(tidy)
        assert list(summary.columns) == list(SUMMARY_COLUMNS)
        assert summary["method"].iloc[0] == "repeated_seed_t"
        assert summary["n_runs"].iloc[0] == 4

    def test_summarize_runs_missing_columns_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            summarize_runs(pd.DataFrame({"task": []}))


class TestBenchmarkRunner:
    def test_tidy_schema_and_determinism(self, synthetic_df):
        rec1 = run_single_seed(synthetic_df, "high_cost", ["dummy", "logistic"], seed=0)
        rec2 = run_single_seed(synthetic_df, "high_cost", ["dummy", "logistic"], seed=0)
        t1 = pd.DataFrame(rec1.tidy_rows)
        t2 = pd.DataFrame(rec2.tidy_rows)
        assert list(t1.columns) == list(TIDY_COLUMNS)
        pd.testing.assert_frame_equal(t1, t2)

    def test_fairness_metrics_present(self, synthetic_df):
        rec = run_single_seed(synthetic_df, "high_cost", ["logistic"], seed=0)
        metrics = {r["metric"] for r in rec.tidy_rows}
        assert {
            "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1",
            "brier", "ece", "demographic_parity_difference",
            "equalized_odds_difference",
        } <= metrics
        assert any(m.startswith("group_size_") for m in metrics)

    def test_unknown_model_raises(self, synthetic_df):
        with pytest.raises(ValueError, match="Unknown model"):
            run_single_seed(synthetic_df, "high_cost", ["nope"], seed=0)

    def test_export_results(self, synthetic_df, tmp_path):
        rec = run_benchmark(synthetic_df, ["high_cost"], ["dummy"], seeds=[0, 1])
        paths = export_results(rec, tmp_path / "out")
        assert paths["tidy"].exists() and paths["summary"].exists()
        summary = pd.read_csv(paths["summary"])
        assert list(summary.columns) == list(SUMMARY_COLUMNS)
        assert (summary["n_runs"] == 2).all()
        assert json.loads(paths["dpsgd_metadata"].read_text()) == []


class TestCLI:
    def test_invalid_task_fails(self, capsys):
        assert main(["--task", "bogus"]) == 2
        assert "unknown task" in capsys.readouterr().err

    def test_invalid_model_fails(self, capsys):
        assert main(["--task", "high_cost", "--models", "bogus"]) == 2
        assert "unknown model" in capsys.readouterr().err

    def test_missing_data_fails(self, capsys):
        assert main(["--task", "high_cost", "--data", "no/such.csv"]) == 2
        assert "not found" in capsys.readouterr().err

    def test_cli_end_to_end(self, synthetic_df, tmp_path, capsys):
        data_path = tmp_path / "data.csv"
        synthetic_df.to_csv(data_path, index=False)
        code = main(
            [
                "--task", "high_cost", "--models", "dummy", "--seeds", "0",
                "--data", str(data_path), "--output-dir", str(tmp_path / "out"),
            ]
        )
        assert code == 0
        assert (tmp_path / "out" / "tidy_results.csv").exists()


class TestReporting:
    def test_report_generation_from_synthetic(self, synthetic_df, tmp_path):
        from dp.reporting import generate_reports

        rec = run_benchmark(
            synthetic_df, ["high_cost"], ["dummy", "logistic"], seeds=[0, 1]
        )
        export_results(rec, tmp_path / "results")
        report = generate_reports(tmp_path / "results", tmp_path / "reports")
        assert report.exists()
        text = report.read_text()
        assert "repeated-seed" in text
        assert (tmp_path / "reports" / "tables" / "model_comparison.csv").exists()
        assert list((tmp_path / "reports" / "figures").glob("*.png"))

    def test_reporting_cli_missing_results(self, tmp_path, capsys):
        from dp.reporting import main as reporting_main

        assert reporting_main(["--results", str(tmp_path / "none")]) == 2


@pytest.mark.slow
class TestDPSGDIntegration:
    def test_dpsgd_metadata_recorded(self, synthetic_df):
        pytest.importorskip("opacus")
        rec = run_single_seed(
            synthetic_df, "high_cost", ["dpsgd"], seed=0, epsilons=[4.0]
        )
        assert len(rec.dpsgd_metadata) == 1
        meta = rec.dpsgd_metadata[0]
        for key in (
            "target_epsilon", "achieved_epsilon", "delta", "noise_multiplier",
            "max_grad_norm", "batch_size", "epochs", "seed", "task",
        ):
            assert key in meta
        # Achieved epsilon comes from the accountant and should be close to
        # (and never wildly above) the calibrated target.
        assert 0 < meta["achieved_epsilon"] <= meta["target_epsilon"] * 1.05
        metrics = {r["metric"] for r in rec.tidy_rows}
        assert "achieved_epsilon" in metrics
        assert "mia_loss_threshold_auc" in metrics
