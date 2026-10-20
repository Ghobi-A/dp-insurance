"""Tests for the independent operating-characteristic simulation of the ladder rule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from dp import ladder_rule_power as lrp

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    sys.path.insert(0, str(ROOT / "research"))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ladder = _load("ladder_for_power_tests", "research/detectability_noise_ladder.py")

#: Executed ladder (Actions run 32441370755, artifact 9433303328): per-seed
#: offline AUCs and Holm-adjusted permutation p-values for seeds 42, 43, 44.
RECORDED = {
    "non-private": ([0.5859, 0.5794, 0.5582], [0.0060, 0.0120, 0.0779], True),
    "eps32": ([0.5917, 0.5258, 0.4994], [0.0060, 0.3556, 0.7073], False),
    "eps16": ([0.5876, 0.5340, 0.5101], [0.0060, 0.3556, 0.7073], False),
    "eps8": ([0.5792, 0.5479, 0.5238], [0.0060, 0.1449, 0.5744], False),
    "eps4": ([0.5715, 0.5289, 0.5288], [0.0140, 0.3556, 0.5744], False),
    "eps2": ([0.5472, 0.5198, 0.5296], [0.0330, 0.3556, 0.5744], False),
}


def test_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(1)
    members = np.round(rng.normal(0.3, 1, 60), 1)
    nonmembers = np.round(rng.normal(0, 1, 70), 1)
    labels = np.r_[np.ones(60), np.zeros(70)]
    expected = roc_auc_score(labels, np.r_[members, nonmembers])
    assert lrp.auc_from_scores(members, nonmembers) == pytest.approx(expected)


def test_null_sd_matches_permutation_distribution():
    rng = np.random.default_rng(2)
    scores = rng.standard_normal(80)
    labels = np.r_[np.ones(40), np.zeros(40)]
    draws = [roc_auc_score(rng.permutation(labels), scores) for _ in range(4000)]
    assert np.std(draws) == pytest.approx(lrp.null_auc_sd(40, 40), rel=0.05)


def test_approximate_pvalue_tracks_permutation_pvalue():
    rng = np.random.default_rng(3)
    n = 150
    members, nonmembers = rng.normal(0.25, 1, n), rng.standard_normal(n)
    scores, labels = np.r_[members, nonmembers], np.r_[np.ones(n), np.zeros(n)]
    observed = roc_auc_score(labels, scores)
    null = np.array([roc_auc_score(rng.permutation(labels), scores) for _ in range(3000)])
    exact = (1 + np.sum(null >= observed)) / (len(null) + 1)
    approx = float(lrp.approx_permutation_pvalue(observed, n, n, reps=3000))
    assert approx == pytest.approx(exact, abs=0.01)
    assert float(lrp.approx_permutation_pvalue(0.99, n, n)) == pytest.approx(1 / 1001)


def test_holm_matches_research_implementation():
    p = [0.001, 0.2, 0.03, 0.04, 0.5, 0.012]
    assert lrp.holm_adjust(p) == pytest.approx(ladder.holm_adjust(p))


def _seed_results(aucs, holm):
    return [
        {
            "attack_status": "completed",
            "offline": {
                "aggregate": {"auc": a},
                "permutation": {"aggregate_auc": {"p_value_holm": p}},
            },
        }
        for a, p in zip(aucs, holm)
    ]


@pytest.mark.parametrize("label", sorted(RECORDED))
def test_reimplemented_rule_reproduces_recorded_ladder_decisions(label):
    aucs, holm, detected = RECORDED[label]
    assert lrp.frozen_rule_detects(aucs, holm) is detected
    assert ladder.aggregate_detected(_seed_results(aucs, holm))[0] is detected


def test_rule_agrees_with_research_rule_on_random_inputs():
    rng = np.random.default_rng(4)
    for _ in range(300):
        aucs = rng.uniform(0.45, 0.65, 3)
        holm = rng.uniform(0, 0.2, 3)
        ours = lrp.frozen_rule_detects(aucs, holm)
        theirs = ladder.aggregate_detected(_seed_results(aucs, holm))[0]
        assert ours is theirs


def test_rule_is_conservative_under_null_and_powerful_for_large_effects():
    null = lrp.simulate_ladder_point_power(0.5, sims=400, seed=5)
    strong = lrp.simulate_ladder_point_power(0.62, sims=400, seed=6)
    assert null.frozen_rule_rate <= 0.01
    assert null.pooled_stouffer_rate <= 0.05
    assert strong.frozen_rule_rate >= 0.95


def test_power_increases_with_effect_size():
    rates = [
        lrp.simulate_ladder_point_power(a, sims=600, seed=7).frozen_rule_rate
        for a in (0.53, 0.56, 0.59)
    ]
    assert rates[0] < rates[1] < rates[2]


@pytest.mark.parametrize("bad", [{"finite_auc": 1.2}, {"finite_auc": 0.55, "tau": -1}])
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        lrp.simulate_ladder_point_power(sims=10, **bad)


def test_pooled_exploratory_reproduces_archived_ladder_values():
    import json

    archived = ROOT / "reports" / "detectability_noise_ladder" / "noise_ladder.json"
    rows = {r["label"]: r for r in lrp.pooled_seed_exploratory(
        json.loads(archived.read_text())["points"]
    )}
    for label, (aucs, _, detected) in RECORDED.items():
        assert rows[label]["seed_aucs"] == pytest.approx(aucs, abs=5e-5)
        assert (rows[label]["frozen_decision"].startswith("DETECTABLE")) is detected
    assert rows["eps8"]["stouffer_z"] == pytest.approx(3.42, abs=0.01)
    assert "stouffer_p_holm_finite" not in rows["non-private"]
    finite = [r for r in rows.values() if r["requested_epsilon"] is not None]
    assert len(finite) == 5 and all(r["stouffer_p_holm_finite"] < 0.05 for r in finite)
