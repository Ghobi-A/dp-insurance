"""Unit tests for the attack-power control experiment.

These exercise the pure logic only -- schedules, cohorts, gates, bootstrap and
report rendering. Training shadow models is far too expensive for the unit
suite, so the one end-to-end check is marked ``slow``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "attack_power_control", ROOT / "research" / "attack_power_control.py"
)
apc = importlib.util.module_from_spec(_SPEC)
sys.modules["attack_power_control"] = apc
assert _SPEC.loader is not None
_SPEC.loader.exec_module(apc)


# --------------------------------------------------------------------------- #
# Shadow schedules
# --------------------------------------------------------------------------- #


def test_balanced_schedule_uses_exact_train_size():
    y_pool = np.tile([0, 1], 150)
    train_sets, _ = apc.balanced_shadow_train_sets(y_pool, train_size=200, num_shadows=16, seed=0)
    assert len(train_sets) == 16
    assert all(len(idx) == 200 for idx in train_sets)
    assert all(len(np.unique(idx)) == 200 for idx in train_sets)


def test_balanced_schedule_gives_every_example_in_and_out_coverage():
    y_pool = np.tile([0, 1], 150)
    n = len(y_pool)
    num_shadows = 32
    train_sets, excluded_counts = apc.balanced_shadow_train_sets(
        y_pool, train_size=200, num_shadows=num_shadows, seed=7
    )
    in_counts = np.zeros(n, dtype=int)
    for idx in train_sets:
        in_counts[idx] += 1

    assert np.array_equal(in_counts + excluded_counts, np.full(n, num_shadows))
    assert excluded_counts.min() >= apc.MIN_GAUSSIAN_OBSERVATIONS
    assert in_counts.min() >= apc.MIN_GAUSSIAN_OBSERVATIONS
    # Coverage is balanced, not merely non-zero.
    assert excluded_counts.max() - excluded_counts.min() <= 1


def test_balanced_schedule_rejects_oversized_train_set():
    y_pool = np.tile([0, 1], 10)
    with pytest.raises(ValueError):
        apc.balanced_shadow_train_sets(y_pool, train_size=20, num_shadows=4, seed=0)


# --------------------------------------------------------------------------- #
# Equalised cohorts
# --------------------------------------------------------------------------- #


def test_equalised_cohort_cells_are_equal_sized():
    rng = np.random.default_rng(0)
    groups = np.array(["female"] * 120 + ["male"] * 80)
    membership = np.concatenate(
        [
            rng.permutation(np.array([1] * 90 + [0] * 30)),
            rng.permutation(np.array([1] * 50 + [0] * 30)),
        ]
    )
    idx, counts = apc.make_equalised_attack_cohort(groups, membership, seed=1)

    assert len(set(counts.values())) == 1
    per_cell = next(iter(counts.values()))
    assert len(idx) == 4 * per_cell
    for group in ("female", "male"):
        for member in (0, 1):
            mask = (groups[idx] == group) & (membership[idx] == member)
            assert int(mask.sum()) == per_cell


def test_equalised_cohort_rejects_empty_cell():
    groups = np.array(["female"] * 4 + ["male"] * 4)
    membership = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    with pytest.raises(RuntimeError):
        apc.make_equalised_attack_cohort(groups, membership, seed=0)


# --------------------------------------------------------------------------- #
# Memorisation gate
# --------------------------------------------------------------------------- #


def test_memorisation_metrics_report_a_generalisation_gap():
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([0, 1, 0, 1])
    train_prob = np.array([0.01, 0.99, 0.02, 0.98])  # memorised
    test_prob = np.array([0.45, 0.55, 0.60, 0.40])  # barely better than chance
    metrics = apc.memorisation_metrics(y_train, train_prob, y_test, test_prob)

    assert metrics["train_auc"] == pytest.approx(1.0)
    assert metrics["loss_gap"] > 0.5
    assert metrics["auc_gap"] > 0.0
    assert metrics["accuracy_gap"] > 0.0
    assert apc.memorisation_gate_passed(metrics)


@pytest.mark.parametrize(
    ("auc_gap", "loss_gap", "acc_gap", "expected"),
    [
        (0.05, 0.0, 0.0, True),
        (0.0, 0.05, 0.0, True),
        (0.0, 0.0, 0.05, True),
        (0.03, 0.0, 0.0, True),  # boundary is inclusive
        (0.029, 0.029, 0.029, False),
        (-0.2, -0.2, -0.2, False),
    ],
)
def test_memorisation_gate_logic(auc_gap, loss_gap, acc_gap, expected):
    metrics = {"auc_gap": auc_gap, "loss_gap": loss_gap, "accuracy_gap": acc_gap}
    assert apc.memorisation_gate_passed(metrics) is expected


# --------------------------------------------------------------------------- #
# Subgroup gaps
# --------------------------------------------------------------------------- #


def test_signed_difference_is_male_minus_female():
    assert apc.signed_subgroup_difference({"female": 0.10, "male": 0.25}) == pytest.approx(0.15)
    assert apc.signed_subgroup_difference({"female": 0.30, "male": 0.05}) == pytest.approx(-0.25)


def test_absolute_gap_is_max_minus_min():
    assert apc.absolute_subgroup_gap({"female": 0.10, "male": 0.25}) == pytest.approx(0.15)
    assert apc.absolute_subgroup_gap({"female": 0.30, "male": 0.05}) == pytest.approx(0.25)
    assert apc.absolute_subgroup_gap({"female": 0.2, "male": 0.2}) == pytest.approx(0.0)


def test_operating_point_counts_are_consistent():
    membership = np.array([1] * 50 + [0] * 50)
    scores = np.concatenate([np.linspace(1.0, 2.0, 50), np.linspace(-2.0, -1.0, 50)])
    counts = apc.operating_point_counts(membership, scores, 0.01)
    assert counts["n_members"] == 50
    assert counts["n_nonmembers"] == 50
    assert counts["detected_members"] == 50  # perfectly separable
    assert counts["false_positives"] == 0


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def _toy_cohort(seed: int = 0, signal: float = 1.0):
    rng = np.random.default_rng(seed)
    groups = np.array(["female"] * 100 + ["male"] * 100)
    membership = np.tile(np.array([1] * 50 + [0] * 50), 2)
    scores = rng.normal(size=200) + signal * membership
    return groups, membership, scores


def test_bootstrap_returns_finite_ordered_intervals():
    groups, membership, scores = _toy_cohort()
    summary = apc.stratified_bootstrap(groups, membership, scores, reps=200, seed=3)

    for key in ("aggregate", "female", "male", "signed_difference", "absolute_gap"):
        entry = summary[key]
        assert np.isfinite(entry["ci95_low"])
        assert np.isfinite(entry["ci95_high"])
        assert entry["ci95_low"] <= entry["mean"] <= entry["ci95_high"]
        assert entry["reps"] == 200
    assert summary["absolute_gap"]["ci95_low"] >= 0.0


def test_bootstrap_rejects_missing_cells():
    groups = np.array(["female"] * 10 + ["male"] * 10)
    membership = np.ones(20, dtype=int)
    with pytest.raises(RuntimeError):
        apc.stratified_bootstrap(groups, membership, np.zeros(20), reps=10, seed=0)


def test_attack_metrics_shape_and_keys():
    groups, membership, scores = _toy_cohort(signal=2.0)
    metrics = apc.attack_metrics(membership, groups, scores, bootstrap_reps=100, seed=0)

    assert metrics["aggregate"]["auc"] > 0.5
    for fpr in apc.ATTACK_FPRS:
        assert f"tpr_at_fpr_{fpr}" in metrics["aggregate"]
    assert set(metrics["subgroups"]) == {"female", "male"}
    assert np.isfinite(metrics["signed_subgroup_difference"])
    assert metrics["absolute_subgroup_gap"] >= 0.0


# --------------------------------------------------------------------------- #
# Decision table and reporting
# --------------------------------------------------------------------------- #


def _seed_result(control, seed, *, gate=True, status="completed", auc=0.5, ci_low=0.0):
    groups, membership, scores = _toy_cohort(seed=seed)
    metrics = (
        apc.attack_metrics(membership, groups, scores, bootstrap_reps=50, seed=seed)
        if status == "completed"
        else None
    )
    if metrics is not None:
        metrics["aggregate"]["auc"] = auc
        metrics["bootstrap"]["aggregate"]["ci95_low"] = ci_low
    return {
        "control": control.name,
        "kind": control.kind,
        "architecture": control.architecture(),
        "positive_control": control.positive_control,
        "seed": seed,
        "train_size": 856,
        "val_size": 214,
        "test_size": 268,
        "cohort_counts": {"female|member=0": 60, "male|member=0": 60},
        "num_attack_examples": 240,
        "memorisation": {
            "train_auc": 1.0,
            "test_auc": 0.85,
            "train_loss": 0.05,
            "test_loss": 0.60,
            "train_accuracy": 1.0,
            "test_accuracy": 0.80,
            "auc_gap": 0.15 if gate else 0.0,
            "loss_gap": 0.55 if gate else 0.0,
            "accuracy_gap": 0.20 if gate else 0.0,
        },
        "memorisation_gate": apc.MEMORISATION_GATE,
        "memorisation_gate_passed": gate,
        "attack_status": status,
        "num_shadows": 32,
        "min_out_observations": 11,
        "min_in_observations": 21,
        "offline": metrics,
        "online": None,
        "online_available": False,
        "online_unavailable_reason": "test fixture",
    }


def test_classify_control_detects_leakage():
    control = apc.CONTROLS[1]
    results = [_seed_result(control, seed, auc=0.72, ci_low=0.08) for seed in (42, 43, 44)]
    verdict, basis = apc.classify_control(control, results)
    assert verdict == apc.DETECTS
    assert basis


def test_classify_control_fails_positive_control():
    control = apc.CONTROLS[1]
    results = [_seed_result(control, seed, auc=0.505, ci_low=0.0) for seed in (42, 43, 44)]
    verdict, _ = apc.classify_control(control, results)
    assert verdict == apc.FAILS_POSITIVE_CONTROL


def test_classify_control_inconclusive_without_memorisation():
    control = apc.CONTROLS[0]
    results = [
        _seed_result(control, seed, gate=False, status="skipped_memorisation_gate")
        for seed in (42, 43, 44)
    ]
    verdict, basis = apc.classify_control(control, results)
    assert verdict == apc.INCONCLUSIVE
    assert "memorisation gate" in basis


def test_classify_control_does_not_fail_non_positive_control():
    control = apc.CONTROLS[0]
    results = [_seed_result(control, seed, auc=0.505) for seed in (42, 43, 44)]
    verdict, _ = apc.classify_control(control, results)
    assert verdict == apc.INCONCLUSIVE


def _payload(seed_results, decisions):
    return {
        "experiment": "attack_power_control",
        "task": apc.TASK_NAME,
        "sensitive_attribute": apc.SENSITIVE_ATTRIBUTE,
        "seeds": [42, 43, 44],
        "num_shadows": 32,
        "bootstrap_reps": 1000,
        "controls": [
            {**dataclasses.asdict(apc.CONTROLS[0]), "architecture": apc.CONTROLS[0].architecture()}
        ],
        "seed_results": seed_results,
        "decisions": decisions,
    }


def test_report_generator_handles_skipped_attacks():
    control = apc.CONTROLS[0]
    seed_results = [
        _seed_result(control, seed, gate=False, status="skipped_memorisation_gate")
        for seed in (42, 43, 44)
    ]
    decisions = [
        {
            "control": control.name,
            "positive_control": control.positive_control,
            "seeds_attacked": 0,
            "mean_auc_gap": 0.0,
            "mean_lira_auc": None,
            "verdict": apc.INCONCLUSIVE,
            "basis": "gate never passed",
        }
    ]
    markdown = apc.render_markdown(_payload(seed_results, decisions))

    assert "SKIPPED" in markdown
    assert apc.INCONCLUSIVE in markdown
    assert "Go/no-go" in markdown
    rows = apc.build_rows(_payload(seed_results, decisions))
    assert len(rows) == 3
    assert all(row["attack_status"] == "skipped_memorisation_gate" for row in rows)


def test_report_states_no_go_when_positive_control_fails():
    control = apc.CONTROLS[1]
    seed_results = [_seed_result(control, seed, auc=0.501) for seed in (42, 43, 44)]
    decisions = [
        {
            "control": control.name,
            "positive_control": True,
            "seeds_attacked": 3,
            "mean_auc_gap": 0.15,
            "mean_lira_auc": 0.501,
            "verdict": apc.FAILS_POSITIVE_CONTROL,
            "basis": "clear gap, chance attack",
        }
    ]
    markdown = apc.render_markdown(_payload(seed_results, decisions))

    assert "Natural membership leakage is not reliably measurable" in markdown
    assert "NO-GO" in markdown
    rows = apc.build_rows(_payload(seed_results, decisions))
    assert {row["attack"] for row in rows} == {"lira-offline"}


@pytest.mark.slow
def test_end_to_end_smoke_on_matched_capacity_control(tmp_path):
    pytest.importorskip("torch")
    apc.main(
        [
            "--shadows",
            "32",
            "--seeds",
            "42",
            "43",
            "44",
            "--bootstrap-reps",
            "200",
            "--controls",
            "matched_capacity_mlp",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "attack_power_control.json").exists()
    assert (tmp_path / "attack_power_control.md").exists()
