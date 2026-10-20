"""Independent operating characteristics of the frozen ladder detection rule.

This module is part of the independent statistical review, not the frozen
experiment. It never trains a model and it never changes the frozen rule.
It answers one question: under a simple, explicitly stated score model, how
often does the frozen aggregate rule in
``docs/NOISE_LADDER_PREREGISTRATION.md`` declare detection?

The frozen rule requires mean offline-LiRA AUC >= 0.55, AUC > 0.5 on every
seed, and a Holm-adjusted (across the six ladder points, within seed)
permutation p < 0.05 on at least two of three seeds.

Score model (an assumption, not a fitted model of the insurance targets):

* each seed's membership scores are equal-variance binormal, with the
  per-seed true AUC drawn as ``N(auc, tau^2)`` clipped to (0, 1), so ``tau``
  is between-seed heterogeneity;
* seeds and ladder points are simulated independently. The real ladder pairs
  points within seed, so within-seed point estimates are positively
  correlated; Holm is valid under any dependence, but the simulated power of
  the joint rule is an approximation;
* the 1000-replicate permutation p-value is replaced by its normal
  approximation, discretised like ``(1 + k) / (reps + 1)``. Under random
  membership the permutation distribution of the AUC has mean 1/2 and
  variance ``(n1 + n2 + 1) / (12 n1 n2)`` exactly (no ties).

Run::

    python -m dp.ladder_rule_power --output-dir reports/statistical_review \
        --ladder-json reports/detectability_noise_ladder/noise_ladder.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

ALPHA = 0.05
DETECT_MEAN_AUC = 0.55
MIN_SIGNIFICANT_SEEDS = 2
PERMUTATION_REPS = 1000
#: Ladder points per seed in the Holm family (non-private + five finite).
LADDER_POINTS = 6
#: Members (= non-members) per seed in the executed ladder cohorts: 256-262.
MEMBERS_PER_SEED = 256


def auc_from_scores(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """Mann-Whitney AUC with midranks (equals ``sklearn.metrics.roc_auc_score``)."""
    member_scores = np.asarray(member_scores, dtype=float)
    nonmember_scores = np.asarray(nonmember_scores, dtype=float)
    n1, n0 = member_scores.size, nonmember_scores.size
    if n1 == 0 or n0 == 0:
        raise ValueError("both score arrays must be non-empty")
    ranks = stats.rankdata(np.concatenate([member_scores, nonmember_scores]))
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def null_auc_sd(n_members: int, n_nonmembers: int) -> float:
    """Exact SD of the AUC under random membership labels (continuous scores)."""
    if n_members <= 0 or n_nonmembers <= 0:
        raise ValueError("group sizes must be positive")
    return float(np.sqrt((n_members + n_nonmembers + 1) / (12.0 * n_members * n_nonmembers)))


def approx_permutation_pvalue(
    auc: np.ndarray | float,
    n_members: int,
    n_nonmembers: int,
    reps: int = PERMUTATION_REPS,
) -> np.ndarray:
    """One-sided permutation p-value via the exact null moments.

    Returns ``(1 + reps * p_normal) / (reps + 1)``, the expected value of the
    ``(1 + k) / (reps + 1)`` estimator, so its floor matches the executed test.
    """
    z = (np.asarray(auc, dtype=float) - 0.5) / null_auc_sd(n_members, n_nonmembers)
    return (1.0 + reps * stats.norm.sf(z)) / (reps + 1.0)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (p.size - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def frozen_rule_detects(aucs: Sequence[float], holm_p: Sequence[float]) -> bool:
    """The preregistered aggregate rule for one ladder point (all seeds completed)."""
    aucs = np.asarray(aucs, dtype=float)
    holm_p = np.asarray(holm_p, dtype=float)
    required = min(MIN_SIGNIFICANT_SEEDS, aucs.size)
    return bool(
        aucs.mean() >= DETECT_MEAN_AUC
        and np.all(aucs > 0.5)
        and int(np.sum(holm_p < ALPHA)) >= required
    )


def _binormal_shift(auc: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0) * stats.norm.ppf(np.clip(auc, 1e-6, 1 - 1e-6))


def simulate_seed_aucs(
    true_aucs: np.ndarray,
    n_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one empirical AUC per entry of ``true_aucs`` from binormal scores."""
    true_aucs = np.asarray(true_aucs, dtype=float)
    flat = true_aucs.ravel()
    shift = _binormal_shift(flat)
    members = rng.standard_normal((flat.size, n_per_class)) + shift[:, None]
    nonmembers = rng.standard_normal((flat.size, n_per_class))
    ranks = stats.rankdata(np.concatenate([members, nonmembers], axis=1), axis=1)
    u = ranks[:, :n_per_class].sum(axis=1) - n_per_class * (n_per_class + 1) / 2
    return (u / n_per_class**2).reshape(true_aucs.shape)


@dataclass(frozen=True)
class PowerResult:
    finite_auc: float
    tau: float
    sims: int
    frozen_rule_rate: float
    frozen_rule_mc_se: float
    seed_holm_rejection_rate: float
    mean_criterion_rate: float
    pooled_stouffer_rate: float


def simulate_ladder_point_power(
    finite_auc: float,
    tau: float = 0.0,
    nonprivate_auc: float = 0.575,
    n_seeds: int = 3,
    n_per_class: int = MEMBERS_PER_SEED,
    sims: int = 2000,
    seed: int = 0,
) -> PowerResult:
    """Detection rate of the frozen rule for the ``eps`` points of a ladder.

    All five finite points share true mean AUC ``finite_auc``; the non-private
    point has ``nonprivate_auc`` (it only matters through the Holm family).
    The returned rates are for one representative finite point.

    ``pooled_stouffer_rate`` is an **alternative, non-preregistered** rule
    (Stouffer-combined seed z-scores, one-sided, Holm over the five finite
    points) included only to show how much of the frozen rule's
    conservativeness comes from its effect-size and per-seed requirements.
    """
    if not 0.0 < finite_auc < 1.0:
        raise ValueError("finite_auc must lie in (0, 1)")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    rng = np.random.default_rng(seed)
    point_means = np.array([nonprivate_auc] + [finite_auc] * (LADDER_POINTS - 1))
    true = point_means[None, None, :] + tau * rng.standard_normal(
        (sims, n_seeds, LADDER_POINTS)
    )
    true = np.clip(true, 1e-4, 1 - 1e-4)
    aucs = simulate_seed_aucs(true, n_per_class, rng)
    p = approx_permutation_pvalue(aucs, n_per_class, n_per_class)
    holm = np.apply_along_axis(holm_adjust, 2, p)

    focal = 1  # first finite point; finite points are exchangeable here
    detected = np.array(
        [frozen_rule_detects(aucs[s, :, focal], holm[s, :, focal]) for s in range(sims)]
    )
    sd = null_auc_sd(n_per_class, n_per_class)
    z = (aucs[:, :, 1:] - 0.5) / sd
    stouffer_p = stats.norm.sf(z.sum(axis=1) / np.sqrt(n_seeds))
    stouffer_holm = np.apply_along_axis(holm_adjust, 1, stouffer_p)
    rate = float(detected.mean())
    return PowerResult(
        finite_auc=float(finite_auc),
        tau=float(tau),
        sims=int(sims),
        frozen_rule_rate=rate,
        frozen_rule_mc_se=float(np.sqrt(rate * (1 - rate) / sims)),
        seed_holm_rejection_rate=float((holm[:, :, focal] < ALPHA).mean()),
        mean_criterion_rate=float((aucs[:, :, focal].mean(axis=1) >= DETECT_MEAN_AUC).mean()),
        pooled_stouffer_rate=float((stouffer_holm[:, 0] < ALPHA).mean()),
    )


def pooled_seed_exploratory(points: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Post hoc pooled-seed association test for each executed ladder point.

    **Exploratory and not preregistered.** It cannot change a frozen decision.
    Seeds use different splits, models and shadow schedules, so under the
    no-association null their offline AUCs are independent; ladder points within
    a seed are not, which is why pooling is across seeds only. Stouffer combines
    seed z-scores formed from the exact permutation-null moments, Fisher
    combines the recorded permutation p-values, and both are Holm-adjusted over
    the finite-epsilon points. A rejection says some seed's scores are
    associated with membership; it is not a pooled effect size, a privacy
    estimate, or evidence about subgroups.
    """
    rows = []
    for point in points:
        zs, ps, aucs = [], [], []
        for result in point["seed_results"]:
            offline = result["offline"]
            groups = offline["subgroups"].values()
            members = sum(int(g["members"]) for g in groups)
            nonmembers = sum(int(g["nonmembers"]) for g in groups)
            auc = float(offline["aggregate"]["auc"])
            aucs.append(auc)
            zs.append((auc - 0.5) / null_auc_sd(members, nonmembers))
            ps.append(float(offline["permutation"]["aggregate_auc"]["p_value"]))
        stouffer = float(np.sum(zs) / np.sqrt(len(zs)))
        rows.append(
            {
                "label": point["label"],
                "requested_epsilon": point.get("requested_epsilon"),
                "frozen_decision": point.get("decision"),
                "seed_aucs": aucs,
                "seed_z": [float(z) for z in zs],
                "stouffer_z": stouffer,
                "stouffer_p": float(stats.norm.sf(stouffer)),
                "fisher_p": float(stats.combine_pvalues(ps, method="fisher").pvalue),
            }
        )
    finite = [i for i, r in enumerate(rows) if r["requested_epsilon"] is not None]
    for key in ("stouffer_p", "fisher_p"):
        adjusted = holm_adjust([rows[i][key] for i in finite])
        for i, value in zip(finite, adjusted):
            rows[i][f"{key}_holm_finite"] = float(value)
    return rows


DEFAULT_GRID = (0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.62)
DEFAULT_TAUS = (0.0, 0.02)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="reports/statistical_review")
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--ladder-json",
        default=None,
        help="Archived noise_ladder.json; also writes the exploratory pooled-seed table.",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.ladder_json:
        ladder = json.loads(Path(args.ladder_json).read_text())
        pooled = {
            "status": "exploratory, post hoc, not preregistered; frozen decisions unchanged",
            "source": str(args.ladder_json),
            "points": pooled_seed_exploratory(ladder["points"]),
        }
        (output_dir / "ladder_pooled_exploratory.json").write_text(json.dumps(pooled, indent=2))

    results = []
    for tau in DEFAULT_TAUS:
        for i, auc in enumerate(DEFAULT_GRID):
            results.append(
                simulate_ladder_point_power(
                    auc, tau=tau, sims=args.sims, seed=args.seed + 1000 * i + int(tau * 1e4)
                ).__dict__
            )
    payload = {
        "description": "Operating characteristics of the frozen ladder aggregate rule "
        "under an equal-variance binormal score model (independent review simulation).",
        "assumptions": [
            "binormal equal-variance scores; 256 members and 256 non-members per seed",
            "three independent seeds; points simulated independently within seed",
            "normal approximation to the 1000-replicate permutation p-value",
            "Holm across six ladder points within seed; non-private true AUC 0.575",
            "tau = SD of true per-seed AUC around the point mean",
        ],
        "simulation_seed": args.seed,
        "sims_per_cell": args.sims,
        "results": results,
    }
    (output_dir / "ladder_rule_power.json").write_text(json.dumps(payload, indent=2))
    for row in results:
        print(json.dumps(row))


if __name__ == "__main__":  # pragma: no cover
    main()
