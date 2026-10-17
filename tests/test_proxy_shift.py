"""Research invariants: selection isolation, accounting and paired controls."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dp.proxy_shift import (
    StudyConfig,
    choose_candidate,
    insurance_cohorts,
    proxy_environment,
    run_study,
    synthetic_cohort,
)


def test_environment_changes_only_proxy_and_stable_control_is_identical():
    config = StudyConfig()
    cohort = synthetic_cohort(4000, 12, config, 0.9)
    source = proxy_environment(cohort, 1, 0.9, "synthetic")
    shifted = proxy_environment(cohort, 0, 0.9, "synthetic")
    assert np.array_equal(source, cohort.X)
    assert np.array_equal(source[:, [0, *range(2, 8)]], shifted[:, [0, *range(2, 8)]])
    assert abs((source[:, 1] == 2 * cohort.y - 1).mean() - 0.9) < 0.03
    assert abs((shifted[:, 1] == 2 * cohort.y - 1).mean() - 0.5) < 0.03
    assert np.array_equal(proxy_environment(cohort, 0, 0.9, "synthetic",
                                           stable_control=True), cohort.X)


def validation_rows():
    return pd.DataFrame([
        {"partition": "validation", "environment": environment, "condition": "dpsgd",
         "requested_epsilon": epsilon, "achieved_epsilon": epsilon,
         "gap": 0.01 if environment == "source" or epsilon == 4 else 0.08, "auc": 0.85}
        for epsilon in (2, 4)
        for environment in ("source", "generic_0.5", "generic_0.75", "proxy_0.5", "proxy_0.75")
    ])


def test_selection_uses_validation_and_achieved_cap_and_abstains():
    config = StudyConfig()
    rows = validation_rows()
    assert choose_candidate(rows, "source_only", config) == 2
    assert choose_candidate(rows, "proxy_stress", config) == 4
    assert choose_candidate(rows, "generic_shift", config) == 4
    assert choose_candidate(rows, "proxy_stress", replace(config, privacy_cap=3)) is None
    rows.loc[rows.requested_epsilon == 2, "achieved_epsilon"] = 9
    assert choose_candidate(rows, "source_only", config) == 4
    with pytest.raises(ValueError, match="validation"):
        choose_candidate(rows.assign(partition="test"), "source_only", config)
    with pytest.raises(ValueError, match="missing"):
        choose_candidate(rows[rows.environment != "proxy_0.5"], "proxy_stress", config)


@pytest.mark.parametrize("kwargs", [
    {"seeds": (0, 0)}, {"epsilons": (0,)}, {"stress_levels": (0.25,)},
    {"delta": 0.1}, {"generic_noise": -1}, {"privacy_cap": float("nan")},
])
def test_invalid_protocol_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(StudyConfig(), **kwargs).validate()


def test_insurance_fixed_preparation_and_imputation():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "insurance.csv"
    train, validation, test, metadata = insurance_cohorts(path, 0)
    assert sum(metadata["partition_sizes"]) == 1338
    assert train.X.shape[1] == test.X.shape[1] == 10
    assert np.all((validation.X >= 0) & (validation.X <= 1))
    missing = proxy_environment(test, 0, 1, "insurance")
    assert np.all(missing[:, 1] == 0)
    assert np.array_equal(missing[:, 0], test.X[:, 0])


def test_opacus_controls_equal_when_clipping_inactive_and_accounting_counts_empty():
    pytest.importorskip("torch")
    pytest.importorskip("opacus")
    from dp.proxy_shift import score, train_model

    config = replace(StudyConfig(), n_train=64, n_eval=64, batch_size=1, epochs=1,
                     clip_norm=100000)
    cohort = synthetic_cohort(64, 12, config, 0.9)
    ordinary, meta0 = train_model(cohort, 0, config, "ordinary")
    clipped, meta1 = train_model(cohort, 0, config, "clipped")
    assert meta0["steps"] == meta1["steps"] == 64
    assert meta1["empty_steps"] > 0
    assert meta1["clip_rate"] == 0
    np.testing.assert_allclose(score(ordinary, cohort.X), score(clipped, cohort.X), atol=1e-6)
    _, private = train_model(cohort, 0, config, "dpsgd", 2)
    assert private["steps"] == meta1["steps"]
    assert private["empty_steps"] == meta1["empty_steps"]
    assert 0 < private["achieved_epsilon"] <= 2.01
    assert private["noise_multiplier"] > 0


def test_end_to_end_manifest_outputs_and_repeatability(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("opacus")
    config = replace(StudyConfig(), seeds=(0,), strengths=(0.9,), epsilons=(2,),
                     n_train=64, n_eval=64, epochs=1, batch_size=16)
    first = run_study(config, tmp_path / "first")
    second = run_study(config, tmp_path / "second")
    assert first["accounting"] == second["accounting"]
    for name in ("metrics.csv", "decisions.csv", "seed_summary.csv", "report.md"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    assert (tmp_path / "first" / "gap_interaction.png").exists()
    from dp.proxy_shift_components import decompose

    metrics = pd.read_csv(tmp_path / "first" / "metrics.csv")
    components = decompose(metrics)
    np.testing.assert_allclose(components.gap_interaction,
                               components.clipping_interaction + components.noise_interaction,
                               atol=1e-12)
    with pytest.raises(ValueError, match="missing clipping"):
        decompose(metrics[metrics.condition != "clipped"])
    stable_proxy_environments = metrics[(metrics.variant == "stable_signal_control")
                                       & ~metrics.environment.str.startswith("generic")]
    assert np.allclose(stable_proxy_environments.gap_interaction, 0)
    with pytest.raises(ValueError, match="new or empty"):
        run_study(config, tmp_path / "first")
