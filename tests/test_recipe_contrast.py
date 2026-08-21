"""Unit tests for the paired between-recipe subgroup-leakage contrast.

The contrast is the inferential heart of the corrective iso-epsilon design: it
asks whether two DP-SGD recipes at matched achieved epsilon redistribute
detectable membership leakage between sexes. Two properties matter more than
anything else here, and both are tested directly on synthetic data where the
truth is known: a null design must not produce a redistribution verdict, and an
injected redistribution must be detected.
"""

from __future__ import annotations

import importlib.util
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
rc = _load("recipe_contrast", "research/recipe_contrast.py")

SEEDS = (42, 43, 44)
PER_CELL = 100


def _design() -> list[dict[str, object]]:
    """The paired cohort skeleton: same rows, same order, for every seed."""
    records = []
    for seed in SEEDS:
        position = 0
        for group in ("female", "male"):
            for membership in (1, 0):
                for _ in range(PER_CELL):
                    records.append(
                        {
                            "seed": seed,
                            "cohort_position": position,
                            "example_index": position,
                            "group": group,
                            "membership": membership,
                        }
                    )
                    position += 1
    return records


def _scored(
    rng: np.random.Generator,
    *,
    member_shift: float = 0.0,
    male_member_shift: float = 0.0,
    female_member_shift: float = 0.0,
) -> list[dict[str, object]]:
    """Attach LiRA-like scores; shifts inject leakage where asked."""
    rows = []
    for record in _design():
        score = rng.normal()
        if record["membership"] == 1:
            score += member_shift
            if record["group"] == "male":
                score += male_member_shift
            else:
                score += female_member_shift
        rows.append({**record, "offline_score": float(score)})
    return rows


# --------------------------------------------------------------------------- #
# Alignment and design invariants
# --------------------------------------------------------------------------- #


def test_alignment_requires_the_same_paired_keys():
    rng = np.random.default_rng(0)
    a = _scored(rng)
    b = _scored(rng)[:-1]
    with pytest.raises(ValueError, match="not paired"):
        rc.align_recipes(a, b)


def test_alignment_rejects_disagreeing_group_labels():
    rng = np.random.default_rng(0)
    a = _scored(rng)
    b = [dict(row) for row in _scored(rng)]
    b[0]["group"] = "male" if b[0]["group"] == "female" else "female"
    with pytest.raises(ValueError, match="group differs"):
        rc.align_recipes(a, b)


def test_alignment_rejects_disagreeing_membership():
    rng = np.random.default_rng(0)
    a = _scored(rng)
    b = [dict(row) for row in _scored(rng)]
    b[0]["membership"] = 1 - int(b[0]["membership"])
    with pytest.raises(ValueError, match="membership differs"):
        rc.align_recipes(a, b)


def test_alignment_reports_missing_required_fields():
    rng = np.random.default_rng(0)
    a = [dict(row) for row in _scored(rng)]
    a[0].pop("offline_score")
    with pytest.raises(ValueError, match="offline_score"):
        rc.align_recipes(a, _scored(rng))


def test_alignment_preserves_the_pairing_order():
    rng = np.random.default_rng(1)
    a = _scored(rng)
    b = list(reversed(_scored(rng)))
    shared, scores_a, scores_b = rc.align_recipes(a, b)
    keys = [(int(r["seed"]), int(r["cohort_position"])) for r in shared]
    assert keys == sorted(keys)
    assert len(scores_a) == len(scores_b) == len(shared)


def test_one_person_resolution_matches_the_smallest_member_count():
    groups = np.array(["female"] * 60 + ["male"] * 40)
    membership = np.array([1] * 30 + [0] * 30 + [1] * 20 + [0] * 20)
    assert rc.one_person_resolution(groups, membership) == pytest.approx(1 / 20)


def test_signed_contrast_orientation_is_male_minus_female():
    groups = np.array(["male"] * 200 + ["female"] * 200)
    membership = np.array(([1] * 100 + [0] * 100) * 2)
    # Male members score far above everyone; female members do not.
    scores = np.concatenate(
        [np.full(100, 10.0), np.zeros(100), np.zeros(100), np.zeros(100)]
    )
    assert rc.signed_contrast(groups, membership, scores) > 0
    flipped = np.concatenate(
        [np.zeros(100), np.zeros(100), np.full(100, 10.0), np.zeros(100)]
    )
    assert rc.signed_contrast(groups, membership, flipped) < 0


# --------------------------------------------------------------------------- #
# The two decisive cases
# --------------------------------------------------------------------------- #


def test_null_design_does_not_produce_a_redistribution_verdict():
    """Two recipes leaking identically must not look like redistribution."""
    rng = np.random.default_rng(7)
    a = _scored(rng, member_shift=0.8)
    b = _scored(rng, member_shift=0.8)
    contrast = rc.paired_recipe_permutation_test(a, b, reps=400, seed=3)
    assert contrast["p_value"] > 0.05
    verdict, basis = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.01,
        achieved_epsilon_b=8.02,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "permutation null is not rejected" in basis


def test_injected_redistribution_is_detected():
    """Recipe A exposes male members, recipe B exposes female members."""
    rng = np.random.default_rng(11)
    a = _scored(rng, member_shift=0.3, male_member_shift=3.0)
    b = _scored(rng, member_shift=0.3, female_member_shift=3.0)
    contrast = rc.paired_recipe_permutation_test(a, b, reps=400, seed=5)
    assert contrast["observed"] > contrast["one_person_resolution"]
    assert contrast["p_value"] < 0.05
    verdict, basis = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.01,
        achieved_epsilon_b=8.02,
    )
    assert verdict == rc.REDISTRIBUTION_SUPPORTED
    assert "recipe A" in basis
    # The finding is stated as empirical and bounded, never as a property of
    # epsilon itself.
    assert "epsilon itself remains the formal global worst-case guarantee" in basis


def test_reversed_injection_flips_the_sign_but_not_the_verdict():
    rng = np.random.default_rng(11)
    a = _scored(rng, member_shift=0.3, female_member_shift=3.0)
    b = _scored(rng, member_shift=0.3, male_member_shift=3.0)
    contrast = rc.paired_recipe_permutation_test(a, b, reps=400, seed=5)
    assert contrast["observed"] < 0
    verdict, basis = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_SUPPORTED
    assert "recipe B" in basis


# --------------------------------------------------------------------------- #
# Pre-registered criteria
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def strong_contrast():
    rng = np.random.default_rng(11)
    a = _scored(rng, member_shift=0.3, male_member_shift=3.0)
    b = _scored(rng, member_shift=0.3, female_member_shift=3.0)
    return rc.paired_recipe_permutation_test(a, b, reps=400, seed=5)


def test_criterion_aggregate_power_must_be_established(strong_contrast):
    verdict, basis = rc.classify_redistribution(
        strong_contrast,
        aggregate_power_established=False,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "aggregate attack power is not established" in basis


def test_criterion_achieved_epsilons_must_match_within_tolerance(strong_contrast):
    verdict, basis = rc.classify_redistribution(
        strong_contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=9.5,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "not an iso-epsilon comparison" in basis


def test_criterion_requested_epsilon_cannot_stand_in_for_achieved(strong_contrast):
    verdict, basis = rc.classify_redistribution(
        strong_contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=None,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "achieved epsilon is missing" in basis


def test_criterion_multiplicity_adjusted_p_value_is_the_one_that_counts(strong_contrast):
    verdict, basis = rc.classify_redistribution(
        strong_contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
        adjusted_p_value=0.4,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "adjusted p = 0.4000" in basis


def test_criterion_single_seed_effects_are_rejected():
    contrast = {
        "observed": 0.2,
        "observed_by_seed": {"42": 0.6, "43": 0.0, "44": 0.0},
        "one_person_resolution": 0.01,
        "p_value": 0.001,
    }
    verdict, basis = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "single-seed" in basis


def test_criterion_direction_must_be_reproducible_across_seeds():
    contrast = {
        "observed": 0.02,
        "observed_by_seed": {"42": 0.5, "43": -0.46, "44": 0.02},
        "one_person_resolution": 0.01,
        "p_value": 0.001,
    }
    verdict, basis = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "not reproducible across seeds" in basis


def test_criterion_effect_must_exceed_one_person_resolution():
    contrast = {
        "observed": 0.004,
        "observed_by_seed": {"42": 0.004, "43": 0.004, "44": 0.004},
        "one_person_resolution": 0.01,
        "p_value": 0.001,
    }
    verdict, basis = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "one-person TPR resolution" in basis


def test_inconclusive_when_no_finite_contrast_exists():
    contrast = {
        "observed": float("nan"),
        "observed_by_seed": {"42": float("nan")},
        "one_person_resolution": 0.01,
        "p_value": float("nan"),
    }
    verdict, basis = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_INCONCLUSIVE
    assert "no comparison is possible" in basis


# --------------------------------------------------------------------------- #
# Permutation-null invariants
# --------------------------------------------------------------------------- #


def test_permutation_null_is_paired_and_group_stratified():
    """A swap moves recipe labels only: cells and cohort geometry are fixed."""
    rng = np.random.default_rng(2)
    a = _scored(rng, member_shift=0.5)
    b = _scored(rng, member_shift=0.5)
    contrast = rc.paired_recipe_permutation_test(a, b, reps=200, seed=1)
    assert contrast["n_examples"] == len(SEEDS) * 4 * PER_CELL
    assert contrast["seeds"] == list(SEEDS)
    # The paired null is symmetric about zero by construction.
    assert contrast["null_mean"] == pytest.approx(0.0, abs=0.02)
    assert contrast["null_ci95_low"] < 0 < contrast["null_ci95_high"]


def test_permutation_p_value_uses_the_finite_sample_correction():
    rng = np.random.default_rng(4)
    a = _scored(rng, member_shift=0.5)
    contrast = rc.paired_recipe_permutation_test(a, a, reps=50, seed=1)
    # Identical recipes: delta_D is exactly zero and every draw ties it.
    assert contrast["observed"] == pytest.approx(0.0)
    assert contrast["p_value"] == pytest.approx(1.0)


def test_permutation_test_is_reproducible():
    rng = np.random.default_rng(6)
    a = _scored(rng, member_shift=0.4, male_member_shift=1.0)
    b = _scored(rng, member_shift=0.4)
    first = rc.paired_recipe_permutation_test(a, b, reps=100, seed=9)
    second = rc.paired_recipe_permutation_test(a, b, reps=100, seed=9)
    assert first["p_value"] == second["p_value"]
    assert first["observed"] == second["observed"]


def test_holm_adjustment_is_shared_with_the_ladder():
    values = [0.01, 0.04, 0.03]
    assert rc.holm_adjust(values) == ladder.holm_adjust(values)


# --------------------------------------------------------------------------- #
# Terminology guard
# --------------------------------------------------------------------------- #


FORBIDDEN = (
    "verified private",
    "proves privacy",
    "proven private",
    "guaranteed safe",
    "privacy verified",
)


def test_no_verdict_text_claims_privacy_from_non_detection(strong_contrast):
    texts = []
    for established in (True, False):
        _, basis = rc.classify_redistribution(
            strong_contrast,
            aggregate_power_established=established,
            achieved_epsilon_a=8.0,
            achieved_epsilon_b=8.0,
        )
        texts.append(basis.lower())
    for text in texts:
        for phrase in FORBIDDEN:
            assert phrase not in text


def test_module_documentation_states_the_null_hypothesis():
    source = (ROOT / "research" / "recipe_contrast.py").read_text().lower()
    assert "no recipe-dependent redistribution" in source
    for phrase in FORBIDDEN:
        assert phrase not in source


# --------------------------------------------------------------------------- #
# The authoritative corrective-study report
# --------------------------------------------------------------------------- #

STUDY = ROOT / "reports" / "corrective_iso_epsilon_study.md"

REQUIRED_SECTIONS = (
    "## 1. Motivation",
    "## 2. Original null",
    "## 3. Attack-power control",
    "## 4. Corrected detectability ladder",
    "## 5. Selected operating privacy level",
    "## 6. Corrected iso-epsilon experimental design",
    "## 7. Aggregate leakage results",
    "## 8. Subgroup-conditioned leakage results",
    "## 9. Between-recipe subgroup redistribution test",
    "## 10. Mechanism diagnostics",
    "## 11. Limitations",
    "## 12. Decision / next step",
)


def test_study_report_has_every_required_section():
    text = STUDY.read_text()
    for section in REQUIRED_SECTIONS:
        assert section in text


def test_study_report_labels_the_result_as_pending():
    text = STUDY.read_text()
    assert "SCIENTIFIC RESULT PENDING" in text
    assert "NOT YET TESTED" in text


def test_study_report_gives_the_exact_full_run_command():
    text = STUDY.read_text()
    assert "detectability_noise_ladder.py point" in text
    assert "--shadows 32" in text
    assert "--bootstrap-reps 1000" in text
    assert "--permutation-reps 1000" in text
    assert "--expect-all" in text


def test_study_report_never_claims_privacy_from_non_detection():
    text = STUDY.read_text().lower()
    for phrase in FORBIDDEN:
        assert phrase not in text
    assert "non-detection is not privacy" in text


def test_study_report_keeps_epsilon_a_global_guarantee():
    text = STUDY.read_text().lower()
    assert "epsilon itself" not in text or "not subgroup-specific" in text
    assert "formal global worst-case guarantee" in text
