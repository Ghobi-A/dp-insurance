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
                # Vary with auc so rank correlations are defined in fixtures.
                "tpr_at_fpr_0.01": round(max(0.0, auc - 0.5), 4),
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
        "cohort_index_digest": 111_000 + seed,
        "shadow_schedule_digest": 222_000 + seed,
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


def test_detectable_with_memorisation():
    results = _point("eps8", 8.0, auc=0.60, holm_p=0.01)["seed_results"]
    decision, basis = ladder.classify_ladder_point(results)
    assert decision == ladder.DETECTABLE_WITH_MEMORISATION
    assert "3/3" in basis


def test_detectable_despite_low_measured_memorisation():
    """Amendment 1: a low-memorisation point is still attacked and can detect."""
    results = _point("eps8", 8.0, gate=False, auc=0.60, holm_p=0.01)["seed_results"]
    decision, basis = ladder.classify_ladder_point(results)
    assert decision == ladder.DETECTABLE_LOW_MEMORISATION
    assert "below" in basis


def test_undetectable_despite_memorisation():
    results = _point("eps4", 4.0, auc=0.52, holm_p=0.01)["seed_results"]
    decision, basis = ladder.classify_ladder_point(results)
    assert decision == ladder.UNDETECTABLE_WITH_MEMORISATION
    assert "not a privacy claim" in basis


def test_undetectable_with_low_memorisation():
    results = _point("eps2", 2.0, gate=False, auc=0.50, holm_p=0.9)["seed_results"]
    decision, _ = ladder.classify_ladder_point(results)
    assert decision == ladder.UNDETECTABLE_LOW_MEMORISATION


def test_all_four_statuses_are_reachable_and_distinct():
    combinations = {
        (True, True): ladder.DETECTABLE_WITH_MEMORISATION,
        (True, False): ladder.DETECTABLE_LOW_MEMORISATION,
        (False, True): ladder.UNDETECTABLE_WITH_MEMORISATION,
        (False, False): ladder.UNDETECTABLE_LOW_MEMORISATION,
    }
    seen = set()
    for (detect, memorise), expected in combinations.items():
        results = _point(
            "eps8", 8.0, gate=memorise,
            auc=0.60 if detect else 0.50,
            holm_p=0.01 if detect else 0.9,
        )["seed_results"]
        assert ladder.classify_ladder_point(results)[0] == expected
        seen.add(expected)
    assert len(seen) == 4


def test_no_result_is_ever_labelled_private_or_safe():
    for gate in (True, False):
        for auc, holm in ((0.60, 0.01), (0.50, 0.9)):
            results = _point("eps8", 8.0, gate=gate, auc=auc, holm_p=holm)["seed_results"]
            status, basis = ladder.classify_ladder_point(results)
            text = f"{status} {basis}".lower()
            for banned in ("private", "safe", "verified"):
                assert banned not in text.replace("privacy claim", "")


def test_detection_rule_thresholds_are_unchanged_by_the_amendment():
    """The numeric rule is identical to the original registration."""
    assert ladder.DETECT_MEAN_AUC == 0.55
    assert ladder.MIN_SIGNIFICANT_SEEDS == 2
    assert ladder.ALPHA == 0.05
    assert ladder.MEMORISATION_GATE == 0.03


def test_detection_requires_two_of_three_seeds():
    results = [
        _seed_result(42, auc=0.60, holm_p=0.001),
        _seed_result(43, auc=0.60, holm_p=0.001),
        _seed_result(44, auc=0.60, holm_p=0.90),
    ]
    assert ladder.aggregate_detected(results)[0] is True

    for stat in ("aggregate_auc", "aggregate_tpr"):
        results[1]["offline"]["permutation"][stat]["p_value_holm"] = 0.90
    assert ladder.aggregate_detected(results)[0] is False


def test_detection_requires_every_seed_above_chance():
    results = [
        _seed_result(42, auc=0.62, holm_p=0.001),
        _seed_result(43, auc=0.61, holm_p=0.001),
        _seed_result(44, auc=0.48, holm_p=0.001),
    ]
    detected, basis = ladder.aggregate_detected(results)
    assert detected is False
    assert "chance" in basis


def test_memorisation_present_requires_every_seed():
    assert ladder.memorisation_present(_point("eps8", 8.0)["seed_results"]) is True
    mixed = [
        _seed_result(42, gate=True),
        _seed_result(43, gate=True),
        _seed_result(44, gate=False),
    ]
    assert ladder.memorisation_present(mixed) is False


def test_low_memorisation_points_still_carry_full_offline_metrics():
    """The whole point of Amendment 1: no observation is discarded."""
    results = _point("eps2", 2.0, gate=False)["seed_results"]
    for result in results:
        assert result["attack_status"] == "completed"
        assert result["offline"] is not None
        assert "aggregate" in result["offline"]
        assert "permutation" in result["offline"]
        assert "bootstrap" in result["offline"]
        assert "subgroups" in result["offline"]
        assert result["subgroup_resolution"] is not None


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


def test_subgroup_inconclusive_only_when_an_attack_genuinely_failed():
    """Under Amendment 1 this can no longer arise from the memorisation gate."""
    results = _point("eps2", 2.0, gate=False, status="failed")["seed_results"]
    assert ladder.classify_subgroup(results)[0] == ladder.SUBGROUP_INCONCLUSIVE
    # A low-memorisation point that *did* complete is assessed normally.
    completed = _point("eps2", 2.0, gate=False)["seed_results"]
    assert ladder.classify_subgroup(completed)[0] != ladder.SUBGROUP_INCONCLUSIVE


# --------------------------------------------------------------------------- #
# Frontier bracketing
# --------------------------------------------------------------------------- #


DETECTED = ladder.DETECTABLE_WITH_MEMORISATION
NOT_DETECTED = ladder.UNDETECTABLE_WITH_MEMORISATION


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
                DETECTED,
                DETECTED,
                DETECTED,
                NOT_DETECTED,
                NOT_DETECTED,
                NOT_DETECTED,
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
                DETECTED,
                NOT_DETECTED,
                DETECTED,  # detection returns at lower epsilon
                NOT_DETECTED,
                NOT_DETECTED,
                NOT_DETECTED,
            ]
        )
    )
    assert frontier["non_monotonic"] is True
    assert frontier["stable"] is False
    assert frontier["bracket"] is None
    assert "NON-MONOTONIC" in frontier["summary"]


def test_frontier_when_every_point_stays_detectable():
    frontier = ladder.detectability_frontier(_summaries([DETECTED] * 6))
    assert frontier["bracket"] is None
    assert "below the lowest epsilon tested" in frontier["summary"]


def test_frontier_when_nothing_is_detectable():
    frontier = ladder.detectability_frontier(_summaries([NOT_DETECTED] * 6))
    assert frontier["bracket"] is None
    assert "attack-power problem" in frontier["summary"]


def test_frontier_includes_low_memorisation_points():
    """Every point is classifiable now, so none is skipped when bracketing."""
    frontier = ladder.detectability_frontier(
        _summaries(
            [
                DETECTED,
                DETECTED,
                ladder.DETECTABLE_LOW_MEMORISATION,
                ladder.UNDETECTABLE_LOW_MEMORISATION,
                NOT_DETECTED,
                ladder.UNDETECTABLE_LOW_MEMORISATION,
            ]
        )
    )
    assert not frontier["non_monotonic"]
    assert frontier["bracket"]["detectable_at"] == "eps16"
    assert frontier["bracket"]["not_detectable_at"] == "eps8"
    assert len(frontier["ordered_points"]) == 6


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


def test_pairing_verification_accepts_a_properly_paired_ladder():
    points = [_point(ladder.point_label(e), e) for e in ladder.LADDER_EPSILONS]
    pairing = ladder.verify_pairing(points)
    assert sorted(pairing) == ["42", "43", "44"]
    assert pairing["42"]["cohort_index_digest"] == 111_042
    assert pairing["43"]["shadow_schedule_digest"] == 222_043


@pytest.mark.parametrize(
    "field", ["cohort_index_digest", "shadow_schedule_digest", "num_attack_examples", "train_size"]
)
def test_pairing_verification_rejects_a_point_that_drifted(field):
    points = [_point(ladder.point_label(e), e) for e in ladder.LADDER_EPSILONS]
    points[3]["seed_results"][1][field] = 999_999
    with pytest.raises(ValueError) as excinfo:
        ladder.verify_pairing(points)
    message = str(excinfo.value)
    assert field in message
    assert "seed 43" in message
    assert "eps8" in message


def test_pairing_verification_rejects_artifacts_without_digests():
    points = [_point(ladder.point_label(e), e) for e in ladder.LADDER_EPSILONS[:2]]
    del points[1]["seed_results"][0]["cohort_index_digest"]
    with pytest.raises(ValueError, match="cohort_index_digest"):
        ladder.verify_pairing(points)


def test_aggregation_fails_loudly_when_points_are_not_paired(tmp_path):
    points_dir = tmp_path / "points"
    points_dir.mkdir()
    for epsilon in ladder.LADDER_EPSILONS:
        point = _point(ladder.point_label(epsilon), epsilon)
        if epsilon == 4.0:
            point["seed_results"][0]["cohort_index_digest"] = 7
        (points_dir / f"{point['label']}.json").write_text(json.dumps(point, default=float))

    args = ladder.argparse.Namespace(
        input_dir=str(points_dir), output_dir=str(tmp_path / "out"), expect_all=True
    )
    with pytest.raises(SystemExit) as excinfo:
        ladder.run_aggregate(args)
    assert "not paired" in str(excinfo.value)


def test_aggregation_writes_per_example_records(tmp_path):
    points_dir = tmp_path / "points"
    points_dir.mkdir()
    for epsilon in ladder.LADDER_EPSILONS:
        point = _point(ladder.point_label(epsilon), epsilon)
        for result in point["seed_results"]:
            result["per_example"] = [
                {
                    "label": result["label"],
                    "requested_epsilon": epsilon,
                    "seed": result["seed"],
                    "cohort_position": position,
                    "example_index": 1000 + position,
                    "group": "male" if position % 2 else "female",
                    "membership": position % 2,
                    "target_loss": 0.5,
                    "offline_score": 0.1 * position,
                    "out_observations": 11,
                    "in_observations": 20,
                }
                for position in range(8)
            ]
        (points_dir / f"{point['label']}.json").write_text(json.dumps(point, default=float))

    out = tmp_path / "out"
    ladder.run_aggregate(
        ladder.argparse.Namespace(
            input_dir=str(points_dir), output_dir=str(out), expect_all=True
        )
    )
    frame = ladder.pd.read_csv(out / "per_example.csv")
    assert len(frame) == 6 * 3 * 8
    assert set(frame["label"]) == {ladder.point_label(e) for e in ladder.LADDER_EPSILONS}
    # The pairing that makes a later between-recipe test possible: identical
    # cohort positions and labels at every ladder point, for every seed.
    for (label, seed), block in frame.groupby(["label", "seed"]):
        assert sorted(block["cohort_position"]) == list(range(8))
    payload = json.loads((out / "noise_ladder.json").read_text())
    assert payload["per_example_rows"] == 6 * 3 * 8
    # Per-example rows live in the CSV, not inlined into the summary JSON.
    assert "per_example" not in payload["points"][0]["seed_results"][0]


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


def test_report_keeps_low_memorisation_points_with_full_metrics():
    points = [
        _point(ladder.NON_PRIVATE_LABEL, None, holm_p=0.001, auc=0.62),
        _point("eps2", 2.0, gate=False, auc=0.50, holm_p=0.9),
    ]
    summaries = ladder.summarise_points(points)
    observations = ladder.collect_observations(summaries)
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
        "observations": observations,
        "gap_associations": ladder.gap_associations(observations, reps=20, seed=0),
        "association_only_statement": ladder.ASSOCIATION_ONLY_STATEMENT,
    }
    markdown = ladder.render_markdown(payload)

    assert "eps2" in markdown
    assert ladder.UNDETECTABLE_LOW_MEMORISATION in markdown
    # Amendment 1: the discarded-point vocabulary must be gone entirely.
    assert "skipped_memorisation_gate" not in markdown
    assert "TARGET DID NOT MEMORISE" not in markdown
    # The low-memorisation point still contributes real numbers.
    assert len(observations) == 6


def test_no_output_ever_contains_the_skipped_status():
    points = [
        _point(ladder.NON_PRIVATE_LABEL, None, gate=False, auc=0.50, holm_p=0.9),
        _point("eps2", 2.0, gate=False, auc=0.50, holm_p=0.9),
    ]
    summaries = ladder.summarise_points(points)
    rows = ladder.build_rows(summaries)
    blob = json.dumps(summaries, default=float) + json.dumps(rows, default=float)
    assert "skipped_memorisation_gate" not in blob
    assert all(row["attack_status"] == "completed" for row in rows)


def test_every_point_enters_holm_adjustment():
    """Including low-memorisation points, which the old gate would have removed."""
    points = [
        _point(ladder.NON_PRIVATE_LABEL, None, holm_p=0.001),
        _point("eps32", 32.0, gate=False, holm_p=0.01),
        _point("eps16", 16.0, gate=False, holm_p=0.02),
    ]
    for point in points:
        for result in point["seed_results"]:
            result["offline"]["permutation"]["aggregate_auc"].pop("p_value_holm")
    ladder.apply_holm_across_points(points)
    for point in points:
        for result in point["seed_results"]:
            assert "p_value_holm" in result["offline"]["permutation"]["aggregate_auc"]


def test_every_point_enters_the_gap_analysis():
    points = [
        _point(ladder.NON_PRIVATE_LABEL, None, auc=0.62),
        _point("eps32", 32.0, gate=False, auc=0.55),
        _point("eps2", 2.0, gate=False, auc=0.50),
    ]
    observations = ladder.collect_observations(ladder.summarise_points(points))
    assert len(observations) == 9
    # Low-memorisation observations are present, not filtered out.
    assert sum(1 for o in observations if not o["memorisation_threshold_passed"]) == 6
    assert all(np.isfinite(o["lira_auc"]) for o in observations)
    for field in ("bce_gap", "auc_gap", "accuracy_gap", "achieved_epsilon",
                  "noise_multiplier", "tpr_at_1pct"):
        assert field in observations[0]


def test_gap_associations_return_finite_ordered_intervals():
    points = [
        _point(ladder.NON_PRIVATE_LABEL, None, auc=0.62),
        _point("eps32", 32.0, auc=0.58),
        _point("eps8", 8.0, gate=False, auc=0.52),
        _point("eps2", 2.0, gate=False, auc=0.50),
    ]
    observations = ladder.collect_observations(ladder.summarise_points(points))
    associations = ladder.gap_associations(observations, reps=50, seed=1)

    assert set(associations) == {
        f"{gap}__{leak}"
        for gap in ladder.GAP_METRICS
        for leak in ladder.LEAKAGE_METRICS
    }
    for entry in associations.values():
        assert entry["ci95_low"] <= entry["ci95_high"]
        assert -1.0 <= entry["spearman"] <= 1.0 or np.isnan(entry["spearman"])
        assert entry["n_observations"] == len(observations)


def test_gap_associations_handle_degenerate_constant_inputs():
    """A constant metric makes Spearman undefined; that must not crash."""
    points = [_point(ladder.NON_PRIVATE_LABEL, None, auc=0.60), _point("eps8", 8.0, auc=0.60)]
    observations = ladder.collect_observations(ladder.summarise_points(points))
    associations = ladder.gap_associations(observations, reps=20, seed=0)
    for entry in associations.values():
        low, high = entry["ci95_low"], entry["ci95_high"]
        assert np.isnan(low) or low <= high


def test_report_contains_the_association_not_causation_statement_verbatim():
    points = [_point(ladder.NON_PRIVATE_LABEL, None, auc=0.62), _point("eps8", 8.0)]
    summaries = ladder.summarise_points(points)
    observations = ladder.collect_observations(summaries)
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
        "observations": observations,
        "gap_associations": ladder.gap_associations(observations, reps=20, seed=0),
        "association_only_statement": ladder.ASSOCIATION_ONLY_STATEMENT,
    }
    markdown = ladder.render_markdown(payload)

    expected = (
        "These analyses measure association only. DP noise changes both "
        "generalisation and leakage, so this experiment does not identify an effect "
        "of DP on membership leakage independently of its effect on memorisation."
    )
    assert expected == ladder.ASSOCIATION_ONLY_STATEMENT
    assert expected in markdown
    for banned in ("mediation", "beyond generalisation", "causes", "causal effect"):
        assert banned not in markdown.lower().replace(
            'no causal, mediation or "beyond generalisation" claim', ""
        )


def test_preregistration_amendment_exists():
    text = (ROOT / "docs" / "NOISE_LADDER_PREREGISTRATION.md").read_text()
    assert "## Preregistration Amendment 1 — attack all ladder points" in text
    for required in (
        "execution gate",
        "explanatory metadata",
        ladder.DETECTABLE_WITH_MEMORISATION,
        ladder.DETECTABLE_LOW_MEMORISATION,
        ladder.UNDETECTABLE_WITH_MEMORISATION,
        ladder.UNDETECTABLE_LOW_MEMORISATION,
        "30218898999",
        "exploratory",
    ):
        assert required in text
    assert "0.55" in text and "0.03" in text


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


# --------------------------------------------------------------------------- #
# Conditional canary pilot
# --------------------------------------------------------------------------- #

canary = _load("canary_pilot", "research/canary_pilot.py")


def _canary_payload(accuracy, eps_lb):
    entry = {
        "construction": "label-flip",
        "probe": "record-level membership",
        "num_canaries": 200,
        "duplication": 1,
        "num_guesses": 200,
        "num_correct": int(accuracy * 200),
        "attacker_accuracy": accuracy,
        "epsilon_lower_bound": eps_lb,
        "confidence": 0.95,
        "passes_pilot": accuracy >= canary.MIN_ATTACKER_ACCURACY
        and eps_lb > canary.MIN_EPSILON_LOWER_BOUND,
    }
    passed = entry["passes_pilot"]
    return {
        "experiment": "canary_pilot_non_private",
        "task": ladder.TASK_NAME,
        "private": False,
        "epochs": 400,
        "architecture": "Linear(d,128->128->64->1) + ReLU",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "seed": 42,
        "min_attacker_accuracy": canary.MIN_ATTACKER_ACCURACY,
        "min_epsilon_lower_bound": canary.MIN_EPSILON_LOWER_BOUND,
        "results": [entry],
        "verdict": canary.PILOT_PASSED if passed else canary.UNDERPOWERED,
        "extend_to_dp_sentinels": passed,
    }


def test_canary_pilot_matches_the_ladder_architecture():
    assert canary.HIDDEN_UNITS == ladder.HIDDEN_UNITS == (128, 128, 64)
    assert canary.BATCH_SIZE == ladder.BATCH_SIZE == 64
    assert canary.LEARNING_RATE == ladder.LEARNING_RATE == 1e-3
    assert canary.EPOCHS == ladder.EPOCHS == 400
    assert canary.TASK_NAME == ladder.TASK_NAME == "high_cost"


def test_failed_canary_pilot_blocks_further_canary_runs():
    payload = _canary_payload(accuracy=0.55, eps_lb=0.0)
    assert payload["verdict"] == canary.UNDERPOWERED
    assert payload["extend_to_dp_sentinels"] is False

    markdown = canary.render_markdown(payload)
    assert canary.UNDERPOWERED in markdown
    assert "stops here" in markdown
    assert "never *proven private*" in markdown


def test_canary_pilot_requires_both_thresholds():
    # High accuracy but a zero bound is not enough.
    assert _canary_payload(0.95, 0.0)["extend_to_dp_sentinels"] is False
    # A positive bound with weak accuracy is not enough either.
    assert _canary_payload(0.60, 1.2)["extend_to_dp_sentinels"] is False
    # Both together justify extension.
    assert _canary_payload(0.80, 1.2)["extend_to_dp_sentinels"] is True


def test_passing_canary_pilot_does_not_auto_run_dp_points():
    markdown = canary.render_markdown(_canary_payload(0.80, 1.2))
    assert "does not start automatically" in markdown
    assert "separate, explicitly requested run" in markdown


def test_duplicated_canaries_are_labelled_as_a_group_privacy_probe():
    markdown = canary.render_markdown(_canary_payload(0.80, 1.2))
    assert "group-privacy" in markdown.lower()
    assert "k*epsilon" in markdown
