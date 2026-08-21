"""Paired between-recipe subgroup-leakage contrast.

The corrective iso-epsilon experiment asks a stronger question than the original
spike did. The original asked, for one recipe,

    D = male TPR@1% FPR - female TPR@1% FPR   !=   0 ?

which conflates a recipe-specific redistribution with whatever subgroup
asymmetry the cohort carries anyway. The quantity that actually bears on the
candidate claim is the *difference of differences* under matched achieved
epsilon:

    delta_D = D_recipe_A - D_recipe_B

with the null hypothesis "no recipe-dependent redistribution of subgroup
leakage". Because the two recipes are run on deliberately paired designs --
same target seeds, same cohort rows in the same order, same shadow inclusion
schedule -- the correct null is an *exchange* of the two recipes' scores on the
same example, not a reshuffle of membership labels. This module implements that
paired, group-stratified permutation test and the pre-registered criteria a
redistribution finding has to clear.

Nothing here trains a model. It consumes the per-example records both the
detectability ladder and the iso-epsilon runner persist, so the design
information (pairing, group, membership) survives into the inference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attack_power_control import (  # noqa: E402
    HEADLINE_FPR,
    signed_subgroup_difference,
    tpr_at_fpr,
)

ALPHA = 0.05

#: Predeclared tolerance on the two recipes' achieved epsilons. Recipes whose
#: achieved epsilons differ by more than this are not an iso-epsilon comparison
#: and the contrast between them is not interpretable as one.
EPSILON_MATCH_TOLERANCE = 0.05

REDISTRIBUTION_SUPPORTED = "RECIPE REDISTRIBUTION SUPPORTED"
REDISTRIBUTION_UNSUPPORTED = "RECIPE REDISTRIBUTION UNSUPPORTED"
REDISTRIBUTION_INCONCLUSIVE = "RECIPE REDISTRIBUTION INCONCLUSIVE"

#: The two DP-SGD recipes for the conditional iso-epsilon follow-up, frozen
#: **before** any authoritative ladder result exists so that no recipe search
#: can follow from seeing the results. Everything except the training dynamics
#: is held identical, and the operating epsilon is deliberately absent: it is
#: supplied by the ladder's aggregate detectability rule, never chosen here and
#: never chosen from subgroup outcomes. Mirrors
#: ``docs/NOISE_LADDER_PREREGISTRATION.md`` Amendment 3.
FROZEN_RECIPES: dict[str, dict[str, object]] = {
    "recipe_a": {
        "name": "large-batch, few steps, tight clipping",
        "architecture": "Linear(d,128)->ReLU->Linear(128,128)->ReLU->Linear(128,64)->ReLU->Linear(64,1)",
        "optimiser": "Adam",
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 100,
        "optimisation_steps": 400,
        "max_grad_norm": 0.5,
        "regularisation": "none",
        "sampling": "poisson",
        "sample_rate": 256 / 856,
        "delta": 1e-5,
        "accountant": "rdp",
        "train_size": 856,
        "num_shadows": 32,
        "target_seeds": [42, 43, 44],
        "noise_multiplier": "solved independently at the ladder-selected epsilon",
    },
    "recipe_b": {
        "name": "small-batch, many steps, loose clipping",
        "architecture": "Linear(d,128)->ReLU->Linear(128,128)->ReLU->Linear(128,64)->ReLU->Linear(64,1)",
        "optimiser": "Adam",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 400,
        "optimisation_steps": 10_800,
        "max_grad_norm": 4.0,
        "regularisation": "none",
        "sampling": "poisson",
        "sample_rate": 32 / 856,
        "delta": 1e-5,
        "accountant": "rdp",
        "train_size": 856,
        "num_shadows": 32,
        "target_seeds": [42, 43, 44],
        "noise_multiplier": "solved independently at the ladder-selected epsilon",
    },
}

#: Fields the two recipes must hold in common, and those they must differ on.
RECIPE_SHARED_FIELDS = (
    "architecture",
    "optimiser",
    "learning_rate",
    "regularisation",
    "sampling",
    "delta",
    "accountant",
    "train_size",
    "num_shadows",
    "target_seeds",
)
RECIPE_DIVERGENT_FIELDS = ("batch_size", "epochs", "optimisation_steps", "max_grad_norm")

#: Field names expected on every per-example record.
REQUIRED_FIELDS = ("seed", "cohort_position", "group", "membership", "offline_score")


def _key(record: dict[str, object]) -> tuple[int, int]:
    return int(record["seed"]), int(record["cohort_position"])


def align_recipes(
    rows_a: Iterable[dict[str, object]],
    rows_b: Iterable[dict[str, object]],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """Align two recipes' per-example records onto the shared paired design.

    Returns the shared records (taken from recipe A, which carries the group and
    membership labels both recipes must agree on) plus the two aligned score
    vectors.

    Raises:
        ValueError: if a record is missing a required field, if the two designs
            do not cover exactly the same ``(seed, cohort_position)`` keys, or
            if they disagree about an example's group or membership. Any of
            those means the pairing the test relies on does not exist.
    """
    listed_a = list(rows_a)
    listed_b = list(rows_b)
    for name, rows in (("A", listed_a), ("B", listed_b)):
        for record in rows:
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(
                    f"recipe {name} per-example record missing {', '.join(missing)}"
                )

    index_a = {_key(r): r for r in listed_a}
    index_b = {_key(r): r for r in listed_b}
    if len(index_a) != len(listed_a) or len(index_b) != len(listed_b):
        raise ValueError("duplicate (seed, cohort_position) keys in a recipe's records")
    if set(index_a) != set(index_b):
        only_a = len(set(index_a) - set(index_b))
        only_b = len(set(index_b) - set(index_a))
        raise ValueError(
            "recipes are not paired: "
            f"{only_a} examples only in A, {only_b} only in B"
        )

    shared: list[dict[str, object]] = []
    scores_a: list[float] = []
    scores_b: list[float] = []
    for key in sorted(index_a):
        left, right = index_a[key], index_b[key]
        if str(left["group"]) != str(right["group"]):
            raise ValueError(f"group differs between recipes at {key}")
        if int(left["membership"]) != int(right["membership"]):
            raise ValueError(f"membership differs between recipes at {key}")
        shared.append(left)
        scores_a.append(float(left["offline_score"]))
        scores_b.append(float(right["offline_score"]))
    return shared, np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)


def signed_contrast(
    groups: np.ndarray,
    membership: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = HEADLINE_FPR,
) -> float:
    """``male TPR@fpr - female TPR@fpr`` for one score vector."""
    groups = np.asarray(groups).astype(str)
    tprs = {
        group: tpr_at_fpr(membership[groups == group], scores[groups == group], target_fpr)
        for group in sorted(np.unique(groups))
    }
    return signed_subgroup_difference(tprs)


def one_person_resolution(groups: np.ndarray, membership: np.ndarray) -> float:
    """Smallest subgroup TPR movement a single member can produce.

    Applied to a single seed's cohort this is the resolution that governs that
    seed's contrast. Applied to several seeds pooled it is **descriptive only**:
    pooling multiplies the member count and so understates the grid the
    per-seed statistics actually move on.
    """
    groups = np.asarray(groups).astype(str)
    membership = np.asarray(membership)
    counts = [
        int(((groups == group) & (membership == 1)).sum())
        for group in np.unique(groups)
    ]
    return 1.0 / max(min(counts) if counts else 1, 1)


def resolution_by_seed(
    groups: np.ndarray,
    membership: np.ndarray,
    seeds: np.ndarray,
) -> dict[str, float]:
    """One-person TPR resolution computed within each target seed's own cohort.

    ``1 / min(male_member_count, female_member_count)`` for that seed alone.
    Each seed's contrast is measured on its own cohort, so this is the grid its
    contrast can move on; the pooled figure is coarser by roughly the number of
    seeds and would let a sub-resolution effect pass.
    """
    groups = np.asarray(groups).astype(str)
    membership = np.asarray(membership)
    seeds = np.asarray(seeds)
    return {
        str(seed): one_person_resolution(groups[seeds == seed], membership[seeds == seed])
        for seed in sorted(set(seeds.tolist()))
    }


def paired_recipe_permutation_test(
    rows_a: Iterable[dict[str, object]],
    rows_b: Iterable[dict[str, object]],
    reps: int = 1000,
    seed: int = 0,
    target_fpr: float = HEADLINE_FPR,
) -> dict[str, object]:
    """Test ``delta_D = D_A - D_B`` against a paired, group-stratified null.

    Under "no recipe-dependent redistribution", which recipe produced which of
    an example's two scores carries no information, so the null distribution is
    generated by swapping the pair on a random subset of examples. The swap is
    drawn independently per example, which leaves every ``group x membership``
    cell, the cohort geometry and the operating-point discreteness exactly as
    observed; only the recipe labelling moves.

    The statistic is computed per target seed and averaged, so a single seed
    cannot carry the result through sheer cohort size, and the same swap draw is
    applied across seeds within a replicate.
    """
    shared, scores_a, scores_b = align_recipes(rows_a, rows_b)
    groups = np.asarray([str(r["group"]) for r in shared])
    membership = np.asarray([int(r["membership"]) for r in shared])
    seeds = np.asarray([int(r["seed"]) for r in shared])
    unique_seeds = sorted(set(seeds.tolist()))

    def per_seed_deltas(left: np.ndarray, right: np.ndarray) -> dict[int, float]:
        deltas: dict[int, float] = {}
        for value in unique_seeds:
            mask = seeds == value
            deltas[value] = signed_contrast(
                groups[mask], membership[mask], left[mask], target_fpr
            ) - signed_contrast(groups[mask], membership[mask], right[mask], target_fpr)
        return deltas

    observed_by_seed = per_seed_deltas(scores_a, scores_b)
    observed = float(np.mean([observed_by_seed[s] for s in unique_seeds]))

    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)
    for rep in range(reps):
        swap = rng.random(len(shared)) < 0.5
        left = np.where(swap, scores_b, scores_a)
        right = np.where(swap, scores_a, scores_b)
        draws = per_seed_deltas(left, right)
        null[rep] = float(np.mean([draws[s] for s in unique_seeds]))

    finite = null[np.isfinite(null)]
    extreme = int(np.sum(np.abs(finite) >= abs(observed))) if np.isfinite(observed) else 0
    p_value = float((1 + extreme) / (reps + 1)) if np.isfinite(observed) else float("nan")
    low, high = (
        np.percentile(finite, [2.5, 97.5]) if finite.size else (float("nan"),) * 2
    )

    return {
        "statistic": "delta_D = (male-female TPR@%g) recipe A - recipe B" % target_fpr,
        "alternative": "two-sided",
        "reps": int(reps),
        "n_examples": int(len(shared)),
        "seeds": [int(s) for s in unique_seeds],
        "observed": observed,
        "observed_by_seed": {str(k): float(v) for k, v in observed_by_seed.items()},
        "null_mean": float(finite.mean()) if finite.size else float("nan"),
        "null_ci95_low": float(low),
        "null_ci95_high": float(high),
        "p_value": p_value,
        "resolution_by_seed": resolution_by_seed(groups, membership, seeds),
        # Pooled across seeds; descriptive only, never used as a criterion.
        "pooled_one_person_resolution": one_person_resolution(groups, membership),
        "recipe_a_signed_by_seed": {
            str(s): float(
                signed_contrast(
                    groups[seeds == s], membership[seeds == s], scores_a[seeds == s], target_fpr
                )
            )
            for s in unique_seeds
        },
        "recipe_b_signed_by_seed": {
            str(s): float(
                signed_contrast(
                    groups[seeds == s], membership[seeds == s], scores_b[seeds == s], target_fpr
                )
            )
            for s in unique_seeds
        },
    }


def classify_redistribution(
    contrast: dict[str, object],
    *,
    aggregate_power_established: bool,
    achieved_epsilon_a: float | None,
    achieved_epsilon_b: float | None,
    adjusted_p_value: float | None = None,
    epsilon_tolerance: float = EPSILON_MATCH_TOLERANCE,
    alpha: float = ALPHA,
) -> tuple[str, str]:
    """Apply the pre-registered criteria for a redistribution finding.

    All seven must hold: aggregate attack power established at this epsilon;
    achieved epsilons matched within the predeclared tolerance; the effect not
    driven by one seed, judged per seed against *that seed's own* one-person
    resolution; direction reproducible across seeds; the mean effect above the
    worst-case per-seed resolution; the permutation null rejected after
    multiplicity adjustment; and the *between-recipe* contrast itself supported,
    not merely one recipe individually showing ``D != 0``.

    The pooled-across-seeds resolution is never a criterion: pooling three
    seeds' cohorts roughly triples the member count and would admit effects
    finer than any single seed can actually resolve.

    Returns the verdict and the reason, which is written out whether or not the
    verdict is supportive -- a failing criterion is a result, not an omission.
    """
    observed_by_seed = {str(k): float(v) for k, v in dict(contrast["observed_by_seed"]).items()}
    by_seed = np.asarray(list(observed_by_seed.values()), dtype=float)
    if "resolution_by_seed" not in contrast:
        raise ValueError(
            "contrast is missing resolution_by_seed; the pooled resolution is "
            "descriptive only and must not be used as a criterion"
        )
    resolutions = {str(k): float(v) for k, v in dict(contrast["resolution_by_seed"]).items()}
    missing = sorted(set(observed_by_seed) - set(resolutions))
    if missing:
        raise ValueError(f"no per-seed resolution for seed(s) {', '.join(missing)}")
    # Pre-registered threshold for the across-seed mean: the most conservative
    # of the per-seed grids, so a mean cannot clear the bar by borrowing the
    # finest cohort's resolution.
    mean_threshold = max(resolutions[seed] for seed in observed_by_seed)
    p_value = (
        float(contrast["p_value"]) if adjusted_p_value is None else float(adjusted_p_value)
    )

    if not by_seed.size or not np.isfinite(by_seed).any():
        return (
            REDISTRIBUTION_INCONCLUSIVE,
            "No finite per-seed contrast was computed, so no comparison is possible.",
        )

    failures: list[str] = []
    if not aggregate_power_established:
        failures.append(
            "aggregate attack power is not established at this operating epsilon, so a "
            "subgroup contrast measured here is uninterpretable"
        )

    if achieved_epsilon_a is None or achieved_epsilon_b is None:
        failures.append("an achieved epsilon is missing, so the recipes cannot be matched")
        epsilon_gap = float("nan")
    else:
        epsilon_gap = abs(float(achieved_epsilon_a) - float(achieved_epsilon_b))
        # 1e-9 of slack so a gap exactly at the declared tolerance is not
        # rejected by floating-point representation alone.
        if epsilon_gap > epsilon_tolerance + 1e-9:
            failures.append(
                f"achieved epsilons differ by {epsilon_gap:.4f} > {epsilon_tolerance}, so "
                "this is not an iso-epsilon comparison"
            )

    nonzero = by_seed[np.isfinite(by_seed) & (by_seed != 0.0)]
    if not nonzero.size or not np.all(np.sign(nonzero) == np.sign(nonzero[0])):
        failures.append("the sign of the between-recipe contrast is not reproducible across seeds")

    # Each seed is judged against its own cohort's one-person grid.
    carrying_seeds = [
        seed
        for seed, value in observed_by_seed.items()
        if np.isfinite(value) and abs(value) >= resolutions[seed]
    ]
    if len(carrying_seeds) < 2:
        failures.append(
            f"only {len(carrying_seeds)} seed(s) carry a contrast at or above that "
            "seed's own one-person resolution ("
            + ", ".join(f"seed {seed}: {value:.4f}" for seed, value in resolutions.items())
            + "), so the effect is single-seed"
        )

    if abs(float(contrast["observed"])) < mean_threshold:
        failures.append(
            f"the mean contrast {float(contrast['observed']):.4f} is within the "
            f"{mean_threshold:.4f} worst-case per-seed one-person TPR resolution"
        )

    if not np.isfinite(p_value) or p_value >= alpha:
        failures.append(
            f"the paired between-recipe permutation null is not rejected "
            f"(adjusted p = {p_value:.4f} >= {alpha})"
        )

    if failures:
        return (
            REDISTRIBUTION_UNSUPPORTED,
            "Not supported: " + "; ".join(failures) + ".",
        )

    direction = "recipe A" if float(contrast["observed"]) > 0 else "recipe B"
    return (
        REDISTRIBUTION_SUPPORTED,
        (
            f"The paired between-recipe contrast delta_D = {float(contrast['observed']):.4f} "
            f"keeps one sign across seeds ({direction} exposes male records relatively more), "
            f"exceeds each seed's own one-person resolution on seeds "
            f"{', '.join(carrying_seeds)} and clears the {mean_threshold:.4f} worst-case "
            "per-seed resolution in the mean, and "
            f"the paired permutation null is rejected (adjusted p = {p_value:.4f}) with "
            f"achieved epsilons matched to within {epsilon_gap:.4f}. This is an empirical "
            "finding for this dataset, task, architecture, adversary and pair of recipes; "
            "epsilon itself remains the formal global worst-case guarantee and is not "
            "subgroup-specific."
        ),
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjustment, re-exported so callers need one import."""
    from detectability_noise_ladder import holm_adjust as _holm

    return _holm(p_values)
