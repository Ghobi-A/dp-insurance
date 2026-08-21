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

SEEDS = (42, 43, 44, 45, 46, 47, 48, 49)
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
    contrast = rc.exact_paired_signflip_test(a, b)
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
    contrast = rc.exact_paired_signflip_test(a, b)
    assert contrast["observed"] > max(contrast["resolution_by_seed"].values())
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
    contrast = rc.exact_paired_signflip_test(a, b)
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
    return rc.exact_paired_signflip_test(a, b)


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
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": 0.2,
        "observed_by_seed": {"42": 0.6, "43": 0.0, "44": 0.0},
        "resolution_by_seed": {"42": 0.01, "43": 0.01, "44": 0.01},
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
    assert "seed 42: 0.0100" in basis


def test_criterion_direction_must_be_reproducible_across_seeds():
    contrast = {
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": 0.02,
        "observed_by_seed": {"42": 0.5, "43": -0.46, "44": 0.02},
        "resolution_by_seed": {"42": 0.01, "43": 0.01, "44": 0.01},
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
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": 0.004,
        "observed_by_seed": {"42": 0.004, "43": 0.004, "44": 0.004},
        "resolution_by_seed": {"42": 0.01, "43": 0.01, "44": 0.01},
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
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": float("nan"),
        "observed_by_seed": {"42": float("nan")},
        "resolution_by_seed": {"42": 0.01},
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


def test_confirmatory_null_is_paired_at_the_seed_level():
    """An exchange moves whole recipe vectors within a seed, nothing else."""
    rng = np.random.default_rng(2)
    a = _scored(rng, member_shift=0.5)
    b = _scored(rng, member_shift=0.5)
    contrast = rc.exact_paired_signflip_test(a, b)
    assert contrast["n_examples"] == len(SEEDS) * 4 * PER_CELL
    # Resolution is per seed, on that seed's own cohort -- not the pooled count.
    assert contrast["resolution_by_seed"] == {
        str(seed): 1 / PER_CELL for seed in SEEDS
    }
    assert contrast["pooled_one_person_resolution"] == pytest.approx(
        1 / (len(SEEDS) * PER_CELL)
    )
    assert contrast["seeds"] == list(SEEDS)
    # The paired null is symmetric about zero by construction.
    # The exact sign-flip null is symmetric about zero by construction.
    assert contrast["null_mean"] == pytest.approx(0.0, abs=1e-12)
    assert contrast["null_min"] == pytest.approx(-contrast["null_max"])
    assert contrast["n_sign_assignments"] == 2 ** len(SEEDS)


def test_identical_recipes_give_an_exact_p_of_one():
    rng = np.random.default_rng(4)
    a = _scored(rng, member_shift=0.5)
    contrast = rc.exact_paired_signflip_test(a, a)
    # Identical recipes: every delta_D_s is exactly zero, so every sign
    # assignment ties the observation.
    assert contrast["observed"] == pytest.approx(0.0)
    assert contrast["p_value"] == pytest.approx(1.0)


def test_exact_test_is_deterministic():
    rng = np.random.default_rng(6)
    a = _scored(rng, member_shift=0.4, male_member_shift=1.0)
    b = _scored(rng, member_shift=0.4)
    first = rc.exact_paired_signflip_test(a, b)
    second = rc.exact_paired_signflip_test(a, b)
    # Exhaustive enumeration: no RNG is involved at all.
    assert first["p_value"] == second["p_value"]
    assert first["observed"] == second["observed"]
    assert first["null_distribution"] == second["null_distribution"]


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


# --------------------------------------------------------------------------- #
# One-person resolution is per seed, never pooled
# --------------------------------------------------------------------------- #


def test_resolution_is_computed_within_each_seed():
    groups = np.array(["female"] * 4 + ["male"] * 4)
    membership = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    seeds = np.array([42] * 8)
    # Seed 42 alone: two female members, one male member.
    assert rc.resolution_by_seed(groups, membership, seeds) == {"42": pytest.approx(1.0)}


def test_resolution_by_seed_tracks_uneven_cohorts():
    groups = np.array(["female", "male"] * 6)
    membership = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
    seeds = np.array([42] * 6 + [43] * 6)
    resolutions = rc.resolution_by_seed(groups, membership, seeds)
    # Seed 42: 2 female members, 2 male members. Seed 43: 2 female, 1 male.
    assert resolutions["42"] == pytest.approx(0.5)
    assert resolutions["43"] == pytest.approx(1.0)


def test_pooled_resolution_is_finer_than_every_per_seed_resolution():
    rng = np.random.default_rng(3)
    contrast = rc.exact_paired_signflip_test(_scored(rng), _scored(rng))
    pooled = contrast["pooled_one_person_resolution"]
    assert all(pooled < value for value in contrast["resolution_by_seed"].values())


def test_effect_clearing_pooled_but_no_per_seed_resolution_is_rejected():
    """The exact false positive the pooled count would have let through.

    Each seed's cohort resolves 1/100; pooling three seeds resolves 1/300. A
    contrast of 0.005 sits above the pooled grid and below every seed's own, so
    it is not a measurable effect on any seed and must be rejected.
    """
    contrast = {
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": 0.005,
        "observed_by_seed": {"42": 0.005, "43": 0.005, "44": 0.005},
        "resolution_by_seed": {"42": 0.01, "43": 0.01, "44": 0.01},
        "pooled_one_person_resolution": 1 / 300,
        "p_value": 0.001,
    }
    assert abs(contrast["observed"]) > contrast["pooled_one_person_resolution"]
    verdict, basis = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "single-seed" in basis
    assert "worst-case per-seed one-person TPR resolution" in basis


def test_mean_threshold_is_the_worst_case_per_seed_resolution():
    """A mean may not clear the bar by borrowing the finest seed's grid."""
    contrast = {
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": 0.03,
        # Two seeds clear their own resolution, so criterion 3 passes.
        "observed_by_seed": {"42": 0.06, "43": 0.02, "44": 0.01},
        "resolution_by_seed": {"42": 0.01, "43": 0.02, "44": 0.05},
        "p_value": 0.001,
    }
    verdict, basis = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED
    assert "0.0500 worst-case per-seed one-person TPR resolution" in basis


def test_classification_refuses_a_contrast_without_per_seed_resolution():
    contrast = {
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": 0.2,
        "observed_by_seed": {"42": 0.2, "43": 0.2, "44": 0.2},
        "one_person_resolution": 0.01,
        "p_value": 0.001,
    }
    with pytest.raises(ValueError, match="resolution_by_seed"):
        rc.classify_redistribution(
            contrast,
            aggregate_power_established=True,
            achieved_epsilon_a=8.0,
            achieved_epsilon_b=8.0,
        )


def test_classification_refuses_a_seed_with_no_resolution():
    contrast = {
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": 0.2,
        "observed_by_seed": {"42": 0.2, "43": 0.2, "44": 0.2},
        "resolution_by_seed": {"42": 0.01, "43": 0.01},
        "p_value": 0.001,
    }
    with pytest.raises(ValueError, match="44"):
        rc.classify_redistribution(
            contrast,
            aggregate_power_established=True,
            achieved_epsilon_a=8.0,
            achieved_epsilon_b=8.0,
        )


# --------------------------------------------------------------------------- #
# Frozen recipes and the epsilon-match tolerance
# --------------------------------------------------------------------------- #

PREREG = ROOT / "docs" / "NOISE_LADDER_PREREGISTRATION.md"

FROZEN_FIELDS = (
    "architecture",
    "optimiser",
    "learning_rate",
    "batch_size",
    "epochs",
    "optimisation_steps",
    "max_grad_norm",
    "regularisation",
    "sampling",
    "delta",
    "accountant",
    "train_size",
    "num_shadows",
    "target_seeds",
)


def test_both_recipes_are_fully_frozen():
    assert set(rc.FROZEN_RECIPES) == {"recipe_a", "recipe_b"}
    for recipe in rc.FROZEN_RECIPES.values():
        for field in FROZEN_FIELDS:
            assert field in recipe, field
            assert recipe[field] not in (None, "")


def test_recipes_share_everything_except_training_dynamics():
    a, b = rc.FROZEN_RECIPES["recipe_a"], rc.FROZEN_RECIPES["recipe_b"]
    for field in rc.RECIPE_SHARED_FIELDS:
        assert a[field] == b[field], field
    for field in rc.RECIPE_DIVERGENT_FIELDS:
        assert a[field] != b[field], field


def test_recipes_differ_materially_in_training_dynamics():
    a, b = rc.FROZEN_RECIPES["recipe_a"], rc.FROZEN_RECIPES["recipe_b"]
    assert max(a["optimisation_steps"], b["optimisation_steps"]) >= 10 * min(
        a["optimisation_steps"], b["optimisation_steps"]
    )
    assert max(a["batch_size"], b["batch_size"]) >= 4 * min(a["batch_size"], b["batch_size"])
    assert max(a["max_grad_norm"], b["max_grad_norm"]) >= 4 * min(
        a["max_grad_norm"], b["max_grad_norm"]
    )


def test_neither_recipe_fixes_the_operating_epsilon():
    """Epsilon comes from the ladder, so it cannot be frozen with the recipes."""
    for recipe in rc.FROZEN_RECIPES.values():
        assert "epsilon" not in recipe
        assert "solved independently" in str(recipe["noise_multiplier"])


def test_recipe_definitions_are_in_the_preregistration():
    text = PREREG.read_text()
    assert "Amendment 3" in text
    assert "Recipe A" in text and "Recipe B" in text
    a, b = rc.FROZEN_RECIPES["recipe_a"], rc.FROZEN_RECIPES["recipe_b"]
    for recipe in (a, b):
        assert str(recipe["batch_size"]) in text
        assert str(recipe["epochs"]) in text
        assert str(recipe["max_grad_norm"]) in text
    assert f"{a['optimisation_steps']:,}" in text or str(a["optimisation_steps"]) in text
    assert f"{b['optimisation_steps']:,}" in text


def test_preregistration_fixes_the_order_of_operations():
    text = PREREG.read_text()
    assert "recipes fixed before ladder result" in text
    assert "AGGREGATE leakage rule only" in text
    assert "one pre-registered ΔD test" in text
    assert "not** inspected to choose ε" in text or "not** inspected" in text


def test_preregistration_states_the_per_seed_resolution_rule():
    text = PREREG.read_text()
    assert "resolution_by_seed[seed] = 1 / min(male_member_count[seed]" in text
    assert "max(resolution_by_seed.values())" in text
    assert "pooled" in text.lower()


def test_epsilon_match_tolerance_is_five_hundredths():
    assert rc.EPSILON_MATCH_TOLERANCE == 0.05
    assert "0.05" in PREREG.read_text()


def test_epsilon_tolerance_boundary_behaviour(strong_contrast):
    inside, _ = rc.classify_redistribution(
        strong_contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.00,
        achieved_epsilon_b=8.05,
    )
    outside, basis = rc.classify_redistribution(
        strong_contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.00,
        achieved_epsilon_b=8.06,
    )
    assert inside == rc.REDISTRIBUTION_SUPPORTED
    assert outside == rc.REDISTRIBUTION_UNSUPPORTED
    assert "not an iso-epsilon comparison" in basis


# --------------------------------------------------------------------------- #
# Randomisation unit: the target seed, not the example
# --------------------------------------------------------------------------- #


def test_per_example_swapping_is_not_the_confirmatory_test():
    """A: the exploratory analysis cannot supply a confirmatory verdict."""
    rng = np.random.default_rng(11)
    a = _scored(rng, member_shift=0.3, male_member_shift=3.0)
    b = _scored(rng, member_shift=0.3, female_member_shift=3.0)
    exploratory = rc.per_example_exchange_sensitivity(a, b, reps=100, seed=1)

    assert exploratory["confirmatory"] is False
    assert exploratory["randomisation_unit"] == "example"
    # It does not even expose a key named p_value.
    assert "p_value" not in exploratory
    assert "exploratory_p_value" in exploratory
    assert "EXPLORATORY SENSITIVITY ANALYSIS ONLY" in exploratory["warning"]
    assert "recipe is assigned at" in exploratory["warning"]

    with pytest.raises(ValueError, match="confirmatory seed-level sign-flip"):
        rc.classify_redistribution(
            exploratory,
            aggregate_power_established=True,
            achieved_epsilon_a=8.0,
            achieved_epsilon_b=8.0,
        )


def test_confirmatory_test_is_stamped_as_seed_level():
    rng = np.random.default_rng(11)
    contrast = rc.exact_paired_signflip_test(
        _scored(rng, member_shift=0.3), _scored(rng, member_shift=0.3)
    )
    assert contrast["confirmatory"] is True
    assert contrast["randomisation_unit"] == "target_seed"
    assert contrast["test"] == "exact_paired_signflip"
    assert contrast["exact"] is True


def test_whole_recipe_vector_exchanges_together_within_a_seed():
    """B: the null is exactly the sign-flip family over per-seed contrasts.

    Exchanging both of a seed's complete score vectors negates that seed's
    contrast and touches no other seed, so the enumerated null must equal the
    set of means of ``z_s * delta_D_s``. Anything finer than the seed would
    produce values outside that family.
    """
    rng = np.random.default_rng(11)
    a = _scored(rng, member_shift=0.3, male_member_shift=2.0)
    b = _scored(rng, member_shift=0.3, female_member_shift=2.0)
    contrast = rc.exact_paired_signflip_test(a, b)

    deltas = np.array([contrast["observed_by_seed"][str(s)] for s in SEEDS])
    expected = sorted(
        float(np.mean(deltas * signs))
        for signs in _all_sign_vectors(len(SEEDS))
    )
    assert sorted(contrast["null_distribution"]) == pytest.approx(expected)

    # Swapping the two recipes for one seed reproduces exactly one null member.
    swapped_a = [dict(r) for r in a]
    swapped_b = [dict(r) for r in b]
    for left, right in zip(swapped_a, swapped_b):
        if int(left["seed"]) == SEEDS[0]:
            left["offline_score"], right["offline_score"] = (
                right["offline_score"],
                left["offline_score"],
            )
    swapped = rc.exact_paired_signflip_test(swapped_a, swapped_b)
    flipped = deltas.copy()
    flipped[0] *= -1
    assert swapped["observed"] == pytest.approx(float(np.mean(flipped)))
    # The null family is closed under a seed-level swap: same set of values,
    # a different observation drawn from it. (The p-value moves with the
    # observation, which is the point -- one seed is one randomisation.)
    assert sorted(swapped["null_distribution"]) == pytest.approx(
        sorted(contrast["null_distribution"])
    )
    assert any(
        value == pytest.approx(float(np.mean(flipped)))
        for value in contrast["null_distribution"]
    )


def _all_sign_vectors(n: int):
    for mask in range(2**n):
        yield np.array([1.0 if not (mask >> i) & 1 else -1.0 for i in range(n)])


def test_eight_seeds_enumerate_exactly_256_assignments():
    """C."""
    rng = np.random.default_rng(2)
    contrast = rc.exact_paired_signflip_test(_scored(rng), _scored(rng))
    assert contrast["n_seeds"] == 8
    assert contrast["n_sign_assignments"] == 256
    assert len(contrast["null_distribution"]) == 256
    assert contrast["exact"] is True
    # Smallest attainable two-sided p at eight seeds.
    assert min(contrast["p_value"], 2 / 256) >= 2 / 256 - 1e-12


def test_reversing_the_recipes_negates_delta_but_not_the_two_sided_p():
    """D."""
    rng = np.random.default_rng(11)
    a = _scored(rng, member_shift=0.3, male_member_shift=3.0)
    b = _scored(rng, member_shift=0.3, female_member_shift=3.0)
    forward = rc.exact_paired_signflip_test(a, b)
    reverse = rc.exact_paired_signflip_test(b, a)

    assert reverse["observed"] == pytest.approx(-forward["observed"])
    for seed in SEEDS:
        assert reverse["observed_by_seed"][str(seed)] == pytest.approx(
            -forward["observed_by_seed"][str(seed)]
        )
    assert reverse["p_value"] == pytest.approx(forward["p_value"])


def test_all_zero_contrasts_give_p_of_one():
    """E."""
    contrast = {
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": 0.0,
        "observed_by_seed": {str(s): 0.0 for s in SEEDS},
        "resolution_by_seed": {str(s): 0.01 for s in SEEDS},
        "p_value": 1.0,
    }
    verdict, _ = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.0,
        achieved_epsilon_b=8.0,
    )
    assert verdict == rc.REDISTRIBUTION_UNSUPPORTED

    rng = np.random.default_rng(4)
    a = _scored(rng, member_shift=0.5)
    assert rc.exact_paired_signflip_test(a, a)["p_value"] == pytest.approx(1.0)


def test_strong_consistent_contrast_rejects_and_is_supported():
    """F."""
    rng = np.random.default_rng(11)
    a = _scored(rng, member_shift=0.3, male_member_shift=3.0)
    b = _scored(rng, member_shift=0.3, female_member_shift=3.0)
    contrast = rc.exact_paired_signflip_test(a, b)

    assert contrast["p_value"] < 0.05
    signs = {np.sign(v) for v in contrast["observed_by_seed"].values() if v != 0}
    assert len(signs) == 1
    verdict, _ = rc.classify_redistribution(
        contrast,
        aggregate_power_established=True,
        achieved_epsilon_a=8.00,
        achieved_epsilon_b=8.02,
    )
    assert verdict == rc.REDISTRIBUTION_SUPPORTED


def test_large_effect_confined_to_one_seed_is_not_supported():
    """G: eight seeds, one carrying everything -> no redistribution claim."""
    by_seed = {str(s): 0.0 for s in SEEDS}
    by_seed[str(SEEDS[0])] = 0.8
    contrast = {
        "confirmatory": True,
        "randomisation_unit": "target_seed",
        "observed": 0.8 / len(SEEDS),
        "observed_by_seed": by_seed,
        "resolution_by_seed": {str(s): 0.01 for s in SEEDS},
        # Even a generous p cannot rescue a single-seed effect.
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


def test_frozen_recipes_carry_exactly_the_eight_confirmatory_seeds():
    """H."""
    expected = [42, 43, 44, 45, 46, 47, 48, 49]
    for recipe in rc.FROZEN_RECIPES.values():
        assert recipe["target_seeds"] == expected
    assert 2 ** len(expected) == 256
    text = PREREG.read_text()
    assert "42, 43, 44, 45, 46, 47, 48, 49" in text
    assert "2^8 = 256" in text or "256" in text


def test_ladder_seeds_remain_three_and_are_not_the_confirmatory_seeds():
    """I: the ladder's job is epsilon selection, and it does not change."""
    assert ladder.DEFAULT_SEEDS == (42, 43, 44)
    assert list(ladder.DEFAULT_SEEDS) != rc.FROZEN_RECIPES["recipe_a"]["target_seeds"]
    text = PREREG.read_text()
    assert "detectability ladder is unchanged at seeds 42, 43, 44" in text


def test_preregistration_amendment_four_fixes_the_randomisation_unit():
    text = PREREG.read_text()
    assert "Amendment 4" in text
    assert "randomisation unit is the target seed" in text
    assert "exchange **together within a seed**" in text
    assert "exploratory" in text and "sensitivity analysis" in text
    assert "count(|T*| ≥ |T_obs|) / 2^S" in text
