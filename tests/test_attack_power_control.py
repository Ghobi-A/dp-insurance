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
    metrics = apc.attack_metrics(
        membership, groups, scores, bootstrap_reps=100, seed=0, permutation_reps=50
    )

    assert metrics["aggregate"]["auc"] > 0.5
    for fpr in apc.ATTACK_FPRS:
        assert f"tpr_at_fpr_{fpr}" in metrics["aggregate"]
    assert set(metrics["subgroups"]) == {"female", "male"}
    assert np.isfinite(metrics["signed_subgroup_difference"])
    assert metrics["absolute_subgroup_gap"] >= 0.0



# --------------------------------------------------------------------------- #
# Permutation null
# --------------------------------------------------------------------------- #


def _separated_cohort(members_per_group=50, separation=5.0, female_boost=0.0):
    """Deterministic cohort: members score higher, optionally more so for one sex."""
    groups = np.array(["female"] * (2 * members_per_group) + ["male"] * (2 * members_per_group))
    membership = np.tile(np.array([1] * members_per_group + [0] * members_per_group), 2)
    base = np.tile(np.linspace(0.0, 1.0, members_per_group), 4)
    scores = base + separation * membership
    scores[: 2 * members_per_group] += female_boost * membership[: 2 * members_per_group]
    return groups, membership, scores


def test_permutation_preserves_group_member_counts():
    groups, membership, scores = _separated_cohort()
    observed_counts = {
        group: (
            int((membership[groups == group] == 1).sum()),
            int((membership[groups == group] == 0).sum()),
        )
        for group in ("female", "male")
    }

    # Reproduce the internal shuffle and assert the invariant it relies on.
    rng = np.random.default_rng(0)
    group_idx = {group: np.flatnonzero(groups == group) for group in ("female", "male")}
    for _ in range(20):
        permuted = membership.copy()
        for idx in group_idx.values():
            permuted[idx] = rng.permutation(membership[idx])
        for group, idx in group_idx.items():
            assert (
                int((permuted[idx] == 1).sum()),
                int((permuted[idx] == 0).sum()),
            ) == observed_counts[group]
        assert permuted.sum() == membership.sum()


def test_permutation_null_is_not_significant_for_identical_distributions():
    # Scores carry no membership information at all.
    groups = np.array(["female"] * 100 + ["male"] * 100)
    membership = np.tile(np.array([1] * 50 + [0] * 50), 2)
    scores = np.tile(np.linspace(0.0, 1.0, 50), 4)

    summary = apc.stratified_membership_permutation_test(
        groups, membership, scores, reps=200, seed=0
    )
    assert summary["aggregate_auc"]["p_value"] > apc.ALPHA
    assert summary["aggregate_tpr"]["p_value"] > apc.ALPHA
    assert summary["signed_difference"]["p_value"] > apc.ALPHA
    assert summary["aggregate_auc"]["observed"] == pytest.approx(0.5, abs=0.05)


def test_permutation_null_is_significant_for_separated_scores():
    groups, membership, scores = _separated_cohort(separation=10.0)
    summary = apc.stratified_membership_permutation_test(
        groups, membership, scores, reps=200, seed=0
    )
    assert summary["aggregate_auc"]["observed"] == pytest.approx(1.0)
    assert summary["aggregate_auc"]["p_value"] < apc.ALPHA
    assert summary["aggregate_tpr"]["p_value"] < apc.ALPHA
    assert summary["aggregate_auc"]["null_mean"] == pytest.approx(0.5, abs=0.05)


def test_reversing_group_advantage_reverses_signed_difference():
    groups, membership, scores = _separated_cohort(separation=1.0, female_boost=10.0)
    female_favoured = apc.stratified_membership_permutation_test(
        groups, membership, scores, reps=100, seed=0
    )
    # Swapping the labels swaps which group carries the advantage.
    swapped_groups = np.where(groups == "female", "male", "female")
    male_favoured = apc.stratified_membership_permutation_test(
        swapped_groups, membership, scores, reps=100, seed=0
    )

    assert female_favoured["signed_difference"]["observed"] < 0
    assert male_favoured["signed_difference"]["observed"] > 0
    assert female_favoured["signed_difference"]["observed"] == pytest.approx(
        -male_favoured["signed_difference"]["observed"]
    )
    assert female_favoured["absolute_gap"]["observed"] == pytest.approx(
        male_favoured["absolute_gap"]["observed"]
    )


def test_permutation_results_are_reproducible_for_a_fixed_seed():
    groups, membership, scores = _separated_cohort(separation=2.0)
    first = apc.stratified_membership_permutation_test(
        groups, membership, scores, reps=100, seed=11
    )
    second = apc.stratified_membership_permutation_test(
        groups, membership, scores, reps=100, seed=11
    )
    different = apc.stratified_membership_permutation_test(
        groups, membership, scores, reps=100, seed=12
    )

    assert first == second
    assert different["aggregate_auc"]["observed"] == first["aggregate_auc"]["observed"]


@pytest.mark.parametrize("separation", [0.0, 0.5, 10.0])
def test_p_values_stay_in_the_attainable_range(separation):
    groups, membership, scores = _separated_cohort(separation=separation)
    reps = 100
    summary = apc.stratified_membership_permutation_test(
        groups, membership, scores, reps=reps, seed=5
    )
    floor = 1.0 / (reps + 1)
    for entry in summary.values():
        assert floor <= entry["p_value"] <= 1.0
        assert entry["reps"] == reps


def test_permutation_pvalue_correction_and_alternatives():
    null = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    # One-sided: two draws are >= 1.0, so (1 + 2) / (5 + 1).
    assert apc._permutation_pvalue(1.0, null, "greater") == pytest.approx(3 / 6)
    # Two-sided compares absolute values: |-3|, |3| and |1| and |-1| are >= 1.
    assert apc._permutation_pvalue(1.0, null, "two-sided") == pytest.approx(5 / 6)
    # Nothing is more extreme than the maximum, so p hits the attainable floor.
    assert apc._permutation_pvalue(4.0, null, "greater") == pytest.approx(1 / 6)


def test_permutation_requires_both_classes_in_every_group():
    groups = np.array(["female"] * 10 + ["male"] * 10)
    membership = np.concatenate([np.ones(10, dtype=int), np.tile([1, 0], 5)])
    with pytest.raises(RuntimeError):
        apc.stratified_membership_permutation_test(
            groups, membership, np.linspace(0, 1, 20), reps=10, seed=0
        )


def test_attack_metrics_include_permutation_block():
    groups, membership, scores = _toy_cohort(signal=2.0)
    metrics = apc.attack_metrics(
        membership, groups, scores, bootstrap_reps=50, seed=0, permutation_reps=100
    )
    perm = metrics["permutation"]
    assert set(perm) >= {
        "aggregate_auc",
        "aggregate_tpr",
        "signed_difference",
        "absolute_gap",
        "female",
        "male",
    }
    assert perm["signed_difference"]["alternative"] == "two-sided"
    assert perm["aggregate_auc"]["alternative"] == "greater"


def test_subgroup_disparity_not_claimed_from_aggregate_signal():
    """A strong aggregate attack with no stable subgroup direction stays unsupported."""
    control = apc.CONTROLS[1]
    results = [
        _seed_result(control, seed, auc=0.80, ci_low=0.20, signed=sign * 0.02, signed_p=0.4)
        for seed, sign in zip((42, 43, 44), (1, -1, 1))
    ]
    verdict, _ = apc.classify_control(control, results)
    subgroup_verdict, basis = apc.assess_subgroup_disparity(results)

    assert verdict == apc.DETECTS
    assert subgroup_verdict == apc.SUBGROUP_UNSUPPORTED
    assert basis


def test_subgroup_disparity_supported_when_stable_and_significant():
    control = apc.CONTROLS[1]
    results = [
        _seed_result(control, seed, auc=0.80, ci_low=0.20, signed=0.25, signed_p=0.001)
        for seed in (42, 43, 44)
    ]
    verdict, _ = apc.assess_subgroup_disparity(results)
    assert verdict == apc.SUBGROUP_SUPPORTED


def test_subgroup_disparity_inconclusive_without_completed_attacks():
    control = apc.CONTROLS[0]
    results = [
        _seed_result(control, seed, gate=False, status="skipped_memorisation_gate")
        for seed in (42, 43, 44)
    ]
    verdict, _ = apc.assess_subgroup_disparity(results)
    assert verdict == apc.SUBGROUP_INCONCLUSIVE


# --------------------------------------------------------------------------- #
# Decision table and reporting
# --------------------------------------------------------------------------- #


def _seed_result(
    control,
    seed,
    *,
    gate=True,
    status="completed",
    auc=0.5,
    ci_low=0.0,
    signed=0.0,
    signed_p=1.0,
):
    groups, membership, scores = _toy_cohort(seed=seed)
    metrics = (
        apc.attack_metrics(
            membership, groups, scores, bootstrap_reps=50, seed=seed, permutation_reps=50
        )
        if status == "completed"
        else None
    )
    if metrics is not None:
        # Overwrite the headline statistics so the decision logic is exercised
        # against known values rather than fixture noise.
        metrics["aggregate"]["auc"] = auc
        metrics["bootstrap"]["aggregate"]["ci95_low"] = ci_low
        metrics["signed_subgroup_difference"] = signed
        metrics["absolute_subgroup_gap"] = abs(signed)
        metrics["permutation"]["aggregate_auc"]["p_value"] = 0.001 if auc > 0.6 else 0.5
        metrics["permutation"]["aggregate_tpr"]["p_value"] = 0.001 if auc > 0.6 else 0.5
        metrics["permutation"]["signed_difference"]["p_value"] = signed_p
        metrics["permutation"]["absolute_gap"]["p_value"] = signed_p
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


def test_report_includes_permutation_results_and_separate_conclusions():
    control = apc.CONTROLS[1]
    seed_results = [
        _seed_result(control, seed, auc=0.80, ci_low=0.20, signed=0.02, signed_p=0.4)
        for seed in (42, 43, 44)
    ]
    decisions = [
        {
            "control": control.name,
            "positive_control": True,
            "seeds_attacked": 3,
            "mean_auc_gap": 0.15,
            "mean_lira_auc": 0.80,
            "mean_permutation_p_auc": 0.001,
            "verdict": apc.DETECTS,
            "basis": "permutation null rejected on 3/3 seeds",
            "subgroup_verdict": apc.SUBGROUP_UNSUPPORTED,
            "subgroup_basis": "signed direction not stable",
        }
    ]
    payload = _payload(seed_results, decisions)
    payload["permutation_reps"] = 50
    markdown = apc.render_markdown(payload)

    assert "permutation" in markdown.lower()
    assert "Stratified permutation tests" in markdown
    # The three questions are answered separately.
    assert "Is aggregate membership leakage detectable?" in markdown
    assert "Does subgroup leakage differ by sex?" in markdown
    assert "still suitable for the iso-epsilon" in markdown
    # Aggregate detection must not be reported as subgroup disparity.
    assert apc.SUBGROUP_UNSUPPORTED in markdown
    assert "CONDITIONAL GO" in markdown

    rows = apc.build_rows(payload)
    assert all("permutation_p_aggregate_auc" in row for row in rows)
    assert all("permutation_p_signed_difference" in row for row in rows)
    assert all("female_permutation_p" in row for row in rows)


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
