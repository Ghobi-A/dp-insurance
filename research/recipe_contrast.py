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
    """Smallest subgroup TPR movement a single member can produce."""
    groups = np.asarray(groups).astype(str)
    membership = np.asarray(membership)
    counts = [
        int(((groups == group) & (membership == 1)).sum())
        for group in np.unique(groups)
    ]
    return 1.0 / max(min(counts) if counts else 1, 1)


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
        "one_person_resolution": one_person_resolution(groups, membership),
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
    driven by one seed; direction reproducible across seeds; effect above
    one-person resolution; the permutation null rejected after multiplicity
    adjustment; and the *between-recipe* contrast itself supported, not merely
    one recipe individually showing ``D != 0``.

    Returns the verdict and the reason, which is written out whether or not the
    verdict is supportive -- a failing criterion is a result, not an omission.
    """
    by_seed = np.asarray(
        [float(v) for v in dict(contrast["observed_by_seed"]).values()], dtype=float
    )
    resolution = float(contrast["one_person_resolution"])
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
        if epsilon_gap > epsilon_tolerance:
            failures.append(
                f"achieved epsilons differ by {epsilon_gap:.4f} > {epsilon_tolerance}, so "
                "this is not an iso-epsilon comparison"
            )

    nonzero = by_seed[np.isfinite(by_seed) & (by_seed != 0.0)]
    if not nonzero.size or not np.all(np.sign(nonzero) == np.sign(nonzero[0])):
        failures.append("the sign of the between-recipe contrast is not reproducible across seeds")

    carrying = int(np.sum(np.abs(by_seed) >= resolution))
    if carrying < 2:
        failures.append(
            f"only {carrying} seed(s) carry a contrast at or above the "
            f"{resolution:.4f} one-person resolution, so the effect is single-seed"
        )

    if abs(float(contrast["observed"])) < resolution:
        failures.append(
            f"the mean contrast {float(contrast['observed']):.4f} is within the "
            f"{resolution:.4f} one-person TPR resolution"
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
            f"exceeds the {resolution:.4f} one-person resolution on at least two seeds, and "
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
