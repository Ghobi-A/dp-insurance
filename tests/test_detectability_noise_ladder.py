"""Unit tests for the DP-SGD detectability noise ladder.

These cover the pre-registered decision logic, the Holm adjustment, the
pairing invariants and the report/aggregation plumbing. Training even one
ladder point is far too expensive for the unit suite, so anything that trains
is marked ``slow``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


apc = _load("attack_power_control", "research/attack_power_control.py")
ladder = _load("detectability_noise_ladder", "research/detectability_noise_ladder.py")


# --------------------------------------------------------------------------- #
# Holm adjustment
# --------------------------------------------------------------------------- #


def test_holm_adjustment_matches_worked_example():
    # m = 4: sorted 0.01, 0.02, 0.03, 0.04 scaled by 4, 3, 2, 1.
    adjusted = ladder.holm_adjust([0.01, 0.02, 0.03, 0.04])
    assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_adjustment_is_monotone_and_clipped():
    adjusted = ladder.holm_adjust([0.2, 0.5, 0.9, 1.0])
    assert all(value <= 1.0 for value in adjusted)
    # Step-down enforces non-decreasing adjusted values in sorted order.
    assert adjusted == sorted(adjusted)


def test_holm_adjustment_never_lowers_a_p_value():
    raw = [0.001, 0.02, 0.3, 0.75, 0.9, 0.99]
    adjusted = ladder.holm_adjust(raw)
    assert all(a >= r - 1e-12 for a, r in zip(adjusted, raw))


def test_holm_adjustment_preserves_input_order():
    adjusted = ladder.holm_adjust([0.04, 0.01, 0.03, 0.02])
    # Same multiset as the sorted-input case, reordered to match the input.
    assert adjusted[1] == pytest.approx(0.04)
    assert adjusted == pytest.approx([0.06, 0.04, 0.06, 0.06])


def test_holm_adjustment_passes_nan_through():
    adjusted = ladder.holm_adjust([0.01, float("nan"), 0.02])
    assert np.isnan(adjusted[1])
    assert np.isfinite(adjusted[0]) and np.isfinite(adjusted[2])


def test_holm_adjustment_handles_single_value_and_empty():
    assert ladder.holm_adjust([0.03]) == pytest.approx([0.03])
    assert ladder.holm_adjust([]) == []


# --------------------------------------------------------------------------- #
# Pairing invariants
# --------------------------------------------------------------------------- #


def test_cohort_indices_are_identical_across_ladder_points():
    """Cohorts derive from the target seed only, never from the privacy level."""
    rng = np.random.default_rng(0)
    groups = np.array(["female"] * 200 + ["male"] * 200)
    membership = np.concatenate(
        [
            rng.permutation(np.array([1] * 150 + [0] * 50)),
            rng.permutation(np.array([1] * 140 + [0] * 60)),
        ]
    )
    per_point = [
        apc.make_equalised_attack_cohort(groups, membership, seed=42)
        for _ in ladder.LADDER_EPSILONS
    ]
    first_idx, first_counts = per_point[0]
    for idx, counts in per_point[1:]:
        assert np.array_equal(idx, first_idx)
        assert counts == first_counts


def test_shadow_schedules_are_identical_across_ladder_points():
    y_pool = np.tile([0, 1], 300)
    schedules = [
        apc.balanced_shadow_train_sets(y_pool, train_size=400, num_shadows=32, seed=42 + 10_000)
        for _ in ladder.LADDER_EPSILONS
    ]
    first_sets, first_counts = schedules[0]
    for train_sets, counts in schedules[1:]:
        assert len(train_sets) == len(first_sets)
        for a, b in zip(train_sets, first_sets):
            assert np.array_equal(a, b)
        assert np.array_equal(counts, first_counts)


def test_shadow_and_target_training_sizes_match():
    y_pool = np.tile([0, 1], 300)
    train_size = 400
    train_sets, excluded = apc.balanced_shadow_train_sets(
        y_pool, train_size=train_size, num_shadows=32, seed=1
    )
    assert all(len(idx) == train_size for idx in train_sets)
    assert excluded.min() >= apc.MIN_GAUSSIAN_OBSERVATIONS


def test_ladder_point_labels_are_stable_and_distinct():
    labels = [ladder.point_label(e) for e in ladder.LADDER_EPSILONS]
    assert labels[0] == ladder.NON_PRIVATE_LABEL
    assert len(set(labels)) == len(labels)
    assert labels[1:] == ["eps32", "eps16", "eps8", "eps4", "eps2"]


# --------------------------------------------------------------------------- #
# Privacy accounting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("epsilon", [32.0, 16.0, 8.0, 4.0, 2.0])
def test_noise_multiplier_solves_to_requested_epsilon(epsilon):
    pytest.importorskip("opacus")
    sample_rate = 64 / 856
    noise = ladder.solve_noise_multiplier(epsilon, sample_rate, epochs=20)
    achieved = ladder.epsilon_for_noise(noise, sample_rate, steps=20 * 14)
    assert noise > 0
    # The solver targets from below, so achieved must not exceed the request.
    assert achieved <= epsilon * 1.05


def test_noise_multiplier_is_monotonic_in_epsilon():
    """Tighter privacy requires more noise."""
    pytest.importorskip("opacus")
    sample_rate = 64 / 856
    noises = [
        ladder.solve_noise_multiplier(eps, sample_rate, epochs=20)
        for eps in (32.0, 16.0, 8.0, 4.0, 2.0)
    ]
    assert noises == sorted(noises), "noise must increase as epsilon decreases"
    assert all(b > a for a, b in zip(noises, noises[1:]))


def test_achieved_epsilon_decreases_as_noise_increases():
    pytest.importorskip("opacus")
    sample_rate = 64 / 856
    epsilons = [
        ladder.epsilon_for_noise(noise, sample_rate, steps=280)
        for noise in (0.8, 1.5, 3.0, 6.0, 12.0)
    ]
    assert epsilons == sorted(epsilons, reverse=True)
    assert all(b < a for a, b in zip(epsilons, epsilons[1:]))


# --------------------------------------------------------------------------- #
# Fixtures for decision logic
# --------------------------------------------------------------------------- #


def _seed_result(
    seed,
    *,
    label="eps8",
    requested_epsilon=8.0,
    gate=True,
    auc=0.60,
    holm_p=0.01,
    signed=0.0,
    signed_holm=1.0,
    members=128,
    status="completed",
):
    """Minimal seed result carrying only what the decision rules read."""
    offline = None
    if status == "completed":
        offline = {
            "aggregate": {
                "auc": auc,
                "tpr_at_fpr_0.001": 0.01,
                "tpr_at_fpr_0.01": 0.05,
                "tpr_at_fpr_0.1": 0.2,
                "operating_point": {
                    "detected_members": 6,
                    "false_positives": 1,
                    "n_members": members,
                    "n_nonmembers": members,
                },
            },
            "subgroups": {
                "female": {
                    "auc": auc,
                    "tpr_at_fpr_0.001": 0.0,
                    "tpr_at_fpr_0.01": 0.05,
                    "tpr_at_fpr_0.1": 0.2,
                    "members": members,
                    "operating_point": {
                        "detected_members": 3,
                        "false_positives": 1,
                        "n_members": members,
                        "n_nonmembers": members,
                    },
                },
                "male": {
                    "auc": auc,
                    "tpr_at_fpr_0.001": 0.0,
                    "tpr_at_fpr_0.01": 0.05 + signed,
                    "tpr_at_fpr_0.1": 0.2,
                    "members": members,
                    "operating_point": {
                        "detected_members": 3,
                        "false_positives": 1,
                        "n_members": members,
                        "n_nonmembers": members,
                    },
                },
            },
            "signed_subgroup_difference": signed,
            "absolute_subgroup_gap": abs(signed),
            "bootstrap": {
                "aggregate": {"reps": 10, "mean": 0.05, "ci95_low": 0.02, "ci95_high": 0.09},
                "female": {"reps": 10, "mean": 0.05, "ci95_low": 0.0, "ci95_high": 0.1},
                "male": {"reps": 10, "mean": 0.05, "ci95_low": 0.0, "ci95_high": 0.1},
                "signed_difference": {
                    "reps": 10, "mean": signed, "ci95_low": -0.1, "ci95_high": 0.1
                },
                "absolute_gap": {"reps": 10, "mean": 0.02, "ci95_low": 0.0, "ci95_high": 0.1},
            },
            "permutation": {
                "aggregate_auc": {
                    "reps": 10, "observed": auc, "null_mean": 0.5,
                    "null_ci95_low": 0.45, "null_ci95_high": 0.55,
                    "alternative": "greater", "p_value": holm_p, "p_value_holm": holm_p,
                },
                "aggregate_tpr": {
                    "reps": 10, "observed": 0.05, "null_mean": 0.01,
                    "null_ci95_low": 0.0, "null_ci95_high": 0.03,
                    "alternative": "greater", "p_value": holm_p, "p_value_holm": holm_p,
                },
                "signed_difference": {
                    "reps": 10, "observed": signed, "null_mean": 0.0,
                    "null_ci95_low": -0.04, "null_ci95_high": 0.04,
                    "alternative": "two-sided", "p_value": signed_holm,
                    "p_value_holm": signed_holm,
                },
                "absolute_gap": {
                    "reps": 10, "observed": abs(signed), "null_mean": 0.014,
                    "null_ci95_low": 0.0, "null_ci95_high": 0.05,
                    "alternative": "greater", "p_value": signed_holm,
                    "p_value_holm": signed_holm,
                },
                "female": {
                    "reps": 10, "observed": 0.05, "null_mean": 0.01,
                    "null_ci95_low": 0.0, "null_ci95_high": 0.03,
                    "alternative": "greater", "p_value": 0.2, "p_value_holm": 0.4,
                },
                "male": {
                    "reps": 10, "observed": 0.05, "null_mean": 0.01,
                    "null_ci95_low": 0.0, "null_ci95_high": 0.03,
                    "alternative": "greater", "p_value": 0.2, "p_value_holm": 0.4,
                },
            },
        }
    return {
        "label": label,
        "requested_epsilon": requested_epsilon,
        "seed": seed,
        "delta": ladder.DELTA,
        "noise_multiplier": None if requested_epsilon is None else 3.5,
        "achieved_epsilon": None if requested_epsilon is None else requested_epsilon * 0.99,
        "epochs": 400,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "architecture": "test",
        "train_size": 856,
        "val_size": 214,
        "test_size": 268,
        "cohort_counts": {"female|member=1": members, "male|member=1": members},
        "num_attack_examples": 4 * members,
        "memorisation": {
            "train_auc": 0.99, "test_auc": 0.90,
            "train_loss": 0.05, "test_loss": 0.5,
            "train_accuracy": 0.98, "test_accuracy": 0.88,
            "auc_gap": 0.09 if gate else 0.001,
            "loss_gap": 0.45 if gate else 0.002,
            "accuracy_gap": 0.10 if gate else 0.001,
        },
        "memorisation_gate": ladder.MEMORISATION_GATE,
        "memorisation_gate_passed": gate,
        "attack_status": status,
        "num_shadows": 32,
        "min_out_observations": 11,
        "min_in_observations": 20,
        "subgroup_member_counts": {"female": members, "male": members},
        "subgroup_resolution": 1.0 / members,
        "offline": offline,
        "online": None,
        "online_available": False,
        "online_unavailable_reason": "test fixture",
    }


def _point(label, requested_epsilon, **kwargs):
    return {
        "experiment": "detectability_noise_ladder_point",
        "label": label,
        "requested_epsilon": requested_epsilon,
        "task": ladder.TASK_NAME,
        "sensitive_attribute": ladder.SENSITIVE_ATTRIBUTE,
        "delta": ladder.DELTA,
        "seeds": [42, 43, 44],
        "num_shadows": 32,
        "bootstrap_reps": 10,
        "permutation_reps": 10,
        "epochs": 400,
        "seed_results": [
            _seed_result(seed, label=label, requested_epsilon=requested_epsilon, **kwargs)
            for seed in (42, 43, 44)
        ],
    }


# --------------------------------------------------------------------------- #
# Pre-registered detection rule
# --------------------------------------------------------------------------- #


def test_detectable_when_all_criteria_met():
    results = _point("eps8", 8.0, auc=0.60, holm_p=0.01)["seed_results"]
    decision, basis = ladder.classify_ladder_point(results)
    assert decision == ladder.DETECTABLE
    assert "3/3" in basis


def test_undetectable_when_mean_auc_below_threshold():
    results = _point("eps4", 4.0, auc=0.52, holm_p=0.01)["seed_results"]
    decision, basis = ladder.classify_ladder_point(results)
    assert decision == ladder.UNDETECTABLE
    assert "mean AUC" in basis
    assert "not evidence of privacy" in basis


def test_undetectable_when_holm_adjusted_significance_insufficient():
    results = _point("eps4", 4.0, auc=0.60, holm_p=0.20)["seed_results"]
    decision, basis = ladder.classify_ladder_point(results)
    assert decision == ladder.UNDETECTABLE
    assert "0/3" in basis


def test_detection_requires_two_of_three_seeds():
    results = [
        _seed_result(42, auc=0.60, holm_p=0.001),
        _seed_result(43, auc=0.60, holm_p=0.001),
        _seed_result(44, auc=0.60, holm_p=0.90),
    ]
    assert ladder.classify_ladder_point(results)[0] == ladder.DETECTABLE

    results[1]["offline"]["permutation"]["aggregate_auc"]["p_value_holm"] = 0.90
    results[1]["offline"]["permutation"]["aggregate_tpr"]["p_value_holm"] = 0.90
    assert ladder.classify_ladder_point(results)[0] == ladder.UNDETECTABLE


def test_undetectable_when_any_seed_at_or_below_chance():
    results = [
        _seed_result(42, auc=0.62, holm_p=0.001),
        _seed_result(43, auc=0.61, holm_p=0.001),
        _seed_result(44, auc=0.48, holm_p=0.001),
    ]
    decision, basis = ladder.classify_ladder_point(results)
    assert decision == ladder.UNDETECTABLE
    assert "chance" in basis


def test_failed_memorisation_gate_is_classified_not_dropped():
    results = _point("eps2", 2.0, gate=False, status="skipped_memorisation_gate")[
        "seed_results"
    ]
    decision, basis = ladder.classify_ladder_point(results)
    assert decision == ladder.DID_NOT_MEMORISE
    assert "memorisation gate failed" in basis


def test_partial_gate_failure_blocks_detection_claim():
    results = [
        _seed_result(42, auc=0.70, holm_p=0.001),
        _seed_result(43, auc=0.70, holm_p=0.001),
        _seed_result(44, gate=False, status="skipped_memorisation_gate"),
    ]
    assert ladder.classify_ladder_point(results)[0] == ladder.DID_NOT_MEMORISE


# --------------------------------------------------------------------------- #
# Subgroup rule
# --------------------------------------------------------------------------- #


def test_signed_contrast_and_resolution_arithmetic():
    assert apc.signed_subgroup_difference({"female": 0.10, "male": 0.25}) == pytest.approx(0.15)
    assert apc.signed_subgroup_difference({"female": 0.25, "male": 0.10}) == pytest.approx(-0.15)
    assert apc.absolute_subgroup_gap({"female": 0.25, "male": 0.10}) == pytest.approx(0.15)
    result = _seed_result(42, members=128)
    assert result["subgroup_resolution"] == pytest.approx(1 / 128)


def test_subgroup_unsupported_when_direction_unstable():
    results = [
        _seed_result(42, signed=0.20, signed_holm=0.001),
        _seed_result(43, signed=-0.20, signed_holm=0.001),
        _seed_result(44, signed=0.20, signed_holm=0.001),
    ]
    verdict, basis = ladder.classify_subgroup(results)
    assert verdict == ladder.SUBGROUP_UNSUPPORTED
    assert "direction is not stable" in basis


def test_subgroup_unsupported_when_within_resolution():
    tiny = 1 / 256  # below the 1/128 one-person resolution
    results = [_seed_result(seed, signed=tiny, signed_holm=0.001) for seed in (42, 43, 44)]
    verdict, basis = ladder.classify_subgroup(results)
    assert verdict == ladder.SUBGROUP_UNSUPPORTED
    assert "resolution" in basis


def test_subgroup_unsupported_when_driven_by_one_seed():
    results = [
        _seed_result(42, signed=0.30, signed_holm=0.001),
        _seed_result(43, signed=0.0, signed_holm=0.001),
        _seed_result(44, signed=0.0, signed_holm=0.001),
    ]
    verdict, basis = ladder.classify_subgroup(results)
    assert verdict == ladder.SUBGROUP_UNSUPPORTED
    assert "single seed" in basis or "resolution" in basis


def test_subgroup_supported_when_all_criteria_met():
    results = [_seed_result(seed, signed=0.25, signed_holm=0.001) for seed in (42, 43, 44)]
    verdict, basis = ladder.classify_subgroup(results)
    assert verdict == ladder.SUBGROUP_SUPPORTED
    assert "male" in basis


def test_subgroup_inconclusive_without_completed_attacks():
    results = _point("eps2", 2.0, gate=False, status="skipped_memorisation_gate")[
        "seed_results"
    ]
    assert ladder.classify_subgroup(results)[0] == ladder.SUBGROUP_INCONCLUSIVE


# --------------------------------------------------------------------------- #
# Frontier bracketing
# --------------------------------------------------------------------------- #


def _summaries(decisions):
    points = []
    for (label, eps), decision in zip(
        [
            (ladder.NON_PRIVATE_LABEL, None),
            ("eps32", 32.0),
            ("eps16", 16.0),
            ("eps8", 8.0),
            ("eps4", 4.0),
            ("eps2", 2.0),
        ],
        decisions,
    ):
        points.append(
            {
                "label": label,
                "requested_epsilon": eps,
                "achieved_epsilon": eps,
                "noise_multiplier": None if eps is None else 1.0,
                "decision": decision,
            }
        )
    return points


def test_frontier_brackets_the_transition():
    frontier = ladder.detectability_frontier(
        _summaries(
            [
                ladder.DETECTABLE,
                ladder.DETECTABLE,
                ladder.DETECTABLE,
                ladder.UNDETECTABLE,
                ladder.UNDETECTABLE,
                ladder.UNDETECTABLE,
            ]
        )
    )
    assert not frontier["non_monotonic"]
    assert frontier["bracket"]["detectable_at"] == "eps16"
    assert frontier["bracket"]["not_detectable_at"] == "eps8"
    assert "transition lies between" in frontier["summary"]


def test_frontier_flags_non_monotonic_ladders():
    frontier = ladder.detectability_frontier(
        _summaries(
            [
                ladder.DETECTABLE,
                ladder.UNDETECTABLE,
                ladder.DETECTABLE,  # detection returns at lower epsilon
                ladder.UNDETECTABLE,
                ladder.UNDETECTABLE,
                ladder.UNDETECTABLE,
            ]
        )
    )
    assert frontier["non_monotonic"] is True
    assert frontier["stable"] is False
    assert frontier["bracket"] is None
    assert "NON-MONOTONIC" in frontier["summary"]


def test_frontier_when_every_point_stays_detectable():
    frontier = ladder.detectability_frontier(_summaries([ladder.DETECTABLE] * 6))
    assert frontier["bracket"] is None
    assert "below the lowest epsilon tested" in frontier["summary"]


def test_frontier_when_nothing_is_detectable():
    frontier = ladder.detectability_frontier(_summaries([ladder.UNDETECTABLE] * 6))
    assert frontier["bracket"] is None
    assert "attack-power problem" in frontier["summary"]


def test_frontier_ignores_points_that_did_not_memorise():
    frontier = ladder.detectability_frontier(
        _summaries(
            [
                ladder.DETECTABLE,
                ladder.DETECTABLE,
                ladder.DID_NOT_MEMORISE,
                ladder.UNDETECTABLE,
                ladder.UNDETECTABLE,
                ladder.DID_NOT_MEMORISE,
            ]
        )
    )
    assert not frontier["non_monotonic"]
    assert frontier["bracket"]["detectable_at"] == "eps32"
    assert frontier["bracket"]["not_detectable_at"] == "eps8"


# --------------------------------------------------------------------------- #
# Aggregation and reporting
# --------------------------------------------------------------------------- #


def test_holm_is_applied_across_points_within_each_seed():
    points = [
        _point(ladder.NON_PRIVATE_LABEL, None, holm_p=0.001),
        _point("eps32", 32.0, holm_p=0.01),
        _point("eps16", 16.0, holm_p=0.02),
    ]
    for point in points:
        for result in point["seed_results"]:
            result["offline"]["permutation"]["aggregate_auc"].pop("p_value_holm")

    ladder.apply_holm_across_points(points)
    for point in points:
        for result in point["seed_results"]:
            entry = result["offline"]["permutation"]["aggregate_auc"]
            assert "p_value_holm" in entry
            assert entry["p_value_holm"] >= entry["p_value"]
    # Three points per seed: smallest raw p is scaled by 3.
    assert points[0]["seed_results"][0]["offline"]["permutation"]["aggregate_auc"][
        "p_value_holm"
    ] == pytest.approx(0.003)


def test_aggregation_fails_clearly_when_a_point_is_missing(tmp_path):
    points_dir = tmp_path / "points"
    points_dir.mkdir()
    (points_dir / "non-private.json").write_text(
        json.dumps(_point(ladder.NON_PRIVATE_LABEL, None), default=float)
    )

    args = ladder.argparse.Namespace(
        input_dir=str(points_dir),
        output_dir=str(tmp_path / "out"),
        expect_all=True,
    )
    with pytest.raises(SystemExit) as excinfo:
        ladder.run_aggregate(args)
    message = str(excinfo.value)
    assert "missing ladder-point artifacts" in message
    for label in ("eps32", "eps16", "eps8", "eps4", "eps2"):
        assert label in message


def test_aggregation_fails_clearly_when_no_artifacts_exist(tmp_path):
    empty = tmp_path / "points"
    empty.mkdir()
    args = ladder.argparse.Namespace(
        input_dir=str(empty), output_dir=str(tmp_path / "out"), expect_all=False
    )
    with pytest.raises(SystemExit) as excinfo:
        ladder.run_aggregate(args)
    assert "no ladder-point artifacts" in str(excinfo.value)


def _full_payload(decisions_holm):
    points = [
        _point(ladder.NON_PRIVATE_LABEL, None, holm_p=decisions_holm[0], auc=0.62),
        _point("eps32", 32.0, holm_p=decisions_holm[1], auc=0.58),
        _point("eps16", 16.0, holm_p=decisions_holm[2], auc=0.52),
        _point("eps8", 8.0, holm_p=decisions_holm[3], auc=0.51),
        _point("eps4", 4.0, holm_p=decisions_holm[4], auc=0.50),
        _point("eps2", 2.0, holm_p=decisions_holm[5], auc=0.50),
    ]
    summaries = ladder.summarise_points(points)
    frontier = ladder.detectability_frontier(summaries)
    return {
        "experiment": "detectability_noise_ladder",
        "task": ladder.TASK_NAME,
        "sensitive_attribute": ladder.SENSITIVE_ATTRIBUTE,
        "delta": ladder.DELTA,
        "seeds": [42, 43, 44],
        "num_shadows": 32,
        "bootstrap_reps": 10,
        "permutation_reps": 10,
        "epochs": 400,
        "alpha": ladder.ALPHA,
        "points": summaries,
        "frontier": frontier,
    }


def test_report_contains_every_required_section():
    payload = _full_payload([0.001, 0.01, 0.5, 0.6, 0.9, 0.9])
    markdown = ladder.render_markdown(payload)

    for heading in (
        "## 1. Pre-registered question and rules",
        "## 2. Threat model summary",
        "## 3. Fixed experimental design",
        "## 4. Privacy accounting",
        "## 5. Memorisation metrics",
        "## 6. Aggregate LiRA results",
        "## 7. Permutation and bootstrap results",
        "## 8. Detectability decision per ladder point",
        "## 9. Detectability-frontier bracket",
        "## 10. Secondary subgroup analysis",
        "## 11. Conclusion",
    ):
        assert heading in markdown

    for question in (
        "Does the attack detect aggregate leakage without DP?",
        "At which DP noise levels is aggregate leakage detectable?",
        "Where does detection disappear?",
        "Are subgroup differences supported?",
        "Is the insurance dataset adequate for aggregate auditing?",
        "Is it adequate for small subgroup-leakage comparisons?",
    ):
        assert question in markdown

    assert "detectable at epsilon" in markdown
    assert "never as privacy" in markdown


def test_report_flags_non_monotonic_frontier():
    payload = _full_payload([0.001, 0.9, 0.001, 0.9, 0.9, 0.9])
    # Force the middle point above the AUC floor so it classifies as detectable.
    for summary in payload["points"]:
        if summary["label"] == "eps16":
            for result in summary["seed_results"]:
                result["offline"]["aggregate"]["auc"] = 0.62
    payload["points"] = ladder.summarise_points(
        [
            {
                **{k: v for k, v in point.items() if k != "seed_results"},
                "seed_results": point["seed_results"],
                "label": point["label"],
                "requested_epsilon": point["requested_epsilon"],
            }
            for point in payload["points"]
        ]
    )
    payload["frontier"] = ladder.detectability_frontier(payload["points"])
    markdown = ladder.render_markdown(payload)
    if payload["frontier"]["non_monotonic"]:
        assert "NON-MONOTONIC" in markdown
        assert "FRONTIER UNSTABLE" in markdown


def test_report_handles_online_lira_unavailable():
    payload = _full_payload([0.001, 0.01, 0.5, 0.6, 0.9, 0.9])
    for summary in payload["points"]:
        for result in summary["seed_results"]:
            assert result["online"] is None  # fixture has online unavailable
    markdown = ladder.render_markdown(payload)
    assert "unavailable" in markdown
    # The offline rows must still be present and complete.
    assert "lira-offline" in markdown


def test_report_keeps_points_that_did_not_memorise():
    points = [
        _point(ladder.NON_PRIVATE_LABEL, None, holm_p=0.001, auc=0.62),
        _point("eps2", 2.0, gate=False, status="skipped_memorisation_gate"),
    ]
    summaries = ladder.summarise_points(points)
    payload = {
        "task": ladder.TASK_NAME,
        "sensitive_attribute": ladder.SENSITIVE_ATTRIBUTE,
        "delta": ladder.DELTA,
        "seeds": [42, 43, 44],
        "num_shadows": 32,
        "bootstrap_reps": 10,
        "permutation_reps": 10,
        "epochs": 400,
        "points": summaries,
        "frontier": ladder.detectability_frontier(summaries),
    }
    markdown = ladder.render_markdown(payload)
    assert "eps2" in markdown
    assert ladder.DID_NOT_MEMORISE in markdown
    assert "skipped_memorisation_gate" in markdown


def test_csv_rows_cover_every_point_and_seed():
    payload = _full_payload([0.001, 0.01, 0.5, 0.6, 0.9, 0.9])
    rows = ladder.build_rows(payload["points"])
    assert len(rows) == 6 * 3  # six points, three seeds, offline only
    assert {row["label"] for row in rows} == {
        ladder.NON_PRIVATE_LABEL, "eps32", "eps16", "eps8", "eps4", "eps2"
    }
    assert all("permutation_p_auc_holm" in row for row in rows)
    assert all("signed_gap" in row for row in rows)
    assert all("subgroup_resolution" in row for row in rows)


@pytest.mark.slow
def test_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    payload = _full_payload([0.001, 0.01, 0.5, 0.6, 0.9, 0.9])
    written = ladder.make_figures(payload["points"], tmp_path)
    assert set(written) == {
        "detectability_auc.png",
        "detectability_tpr_1pct.png",
        "memorisation_gap.png",
        "subgroup_signed_contrast.png",
    }
    for name in written:
        assert (tmp_path / name).stat().st_size > 0


@pytest.mark.slow
def test_smoke_ladder_point_runs(tmp_path):
    """Two seeds, few epochs, tiny replicate counts. Still trains models."""
    pytest.importorskip("torch")
    pytest.importorskip("opacus")
    ladder.main(
        [
            "point",
            "--epsilon", "8",
            "--seeds", "42",
            "--shadows", "32",
            "--bootstrap-reps", "50",
            "--permutation-reps", "50",
            "--epochs", "2",
            "--output-dir", str(tmp_path),
        ]
    )
    assert (tmp_path / "eps8.json").exists()
