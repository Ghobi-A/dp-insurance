"""When does pooled variance help offline LiRA? A labelled synthetic experiment.

This is a simulation, not a unit test and not evidence about the insurance
models. It compares per-example and pooled ("global") OUT variance in
:func:`dp.attacks.lira_offline_attack` across shadow counts, for

* ``homoscedastic``: every example has the same shadow-to-shadow spread, the
  case in which pooling is the efficient estimator; and
* ``heteroscedastic``: per-example spreads vary log-normally, the case in
  which pooling misstates each example's scale and can lose to per-example
  estimates once enough shadows are available.

Observations are generated as Gaussians on the logit-confidence scale and
mapped to valid binary cross-entropy losses (strictly positive), so the
Gaussian assumption is exact for ``statistic="logit_confidence"``.

Run::

    python -m dp.lira_variance_simulation --output-dir reports/statistical_review
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .attacks import lira_offline_attack

SCENARIOS = ("homoscedastic", "heteroscedastic")
DEFAULT_SHADOW_COUNTS = (4, 11, 32, 64, 128, 256)


def bce_from_logit_confidence(logit: np.ndarray) -> np.ndarray:
    """Inverse of ``logit_confidence_from_bce``: ``-log(sigmoid(logit))`` > 0."""
    return np.logaddexp(0.0, -np.asarray(logit, dtype=float))


def simulate_losses(
    scenario: str,
    n: int,
    k: int,
    rng: np.random.Generator,
    member_shift: float = 0.6,
    sigma_log_sd: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target losses, membership and ``(n, k)`` OUT shadow losses.

    Members' logit confidence is raised by ``member_shift`` example-level
    standard deviations, so the per-example signal-to-noise ratio is the same
    in both scenarios and only the variance structure differs.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    mu = rng.normal(1.5, 1.0, n)
    sigma = np.full(n, 0.5)
    if scenario == "heteroscedastic":
        sigma = 0.5 * np.exp(rng.normal(0.0, sigma_log_sd, n))
    membership = (np.arange(n) % 2 == 0).astype(int)
    shadow = mu[:, None] + sigma[:, None] * rng.standard_normal((n, k))
    target = mu + sigma * rng.standard_normal(n) + member_shift * sigma * membership
    return bce_from_logit_confidence(target), membership, bce_from_logit_confidence(shadow)


def compare_variance_rules(
    scenario: str,
    k: int,
    n: int = 520,
    replicates: int = 200,
    seed: int = 0,
    fpr: float = 0.01,
) -> dict[str, float | int | str]:
    """Mean AUC and TPR at ``fpr`` for both variance rules, with paired SEs."""
    rng = np.random.default_rng(seed)
    auc = {"per_example": [], "global": []}
    tpr = {"per_example": [], "global": []}
    for _ in range(replicates):
        target, membership, shadow = simulate_losses(scenario, n, k, rng)
        for rule in auc:
            result = lira_offline_attack(
                target,
                membership,
                shadow,
                fprs=(fpr,),
                statistic="logit_confidence",
                variance=rule,
            )
            auc[rule].append(result.auc)
            tpr[rule].append(result.tpr_at_fpr[fpr])
    auc_diff = np.subtract(auc["global"], auc["per_example"])
    tpr_diff = np.subtract(tpr["global"], tpr["per_example"])
    root = np.sqrt(replicates)
    return {
        "scenario": scenario,
        "shadows_out": int(k),
        "n_examples": int(n),
        "replicates": int(replicates),
        "auc_per_example": float(np.mean(auc["per_example"])),
        "auc_global": float(np.mean(auc["global"])),
        "auc_diff_global_minus_per_example": float(auc_diff.mean()),
        "auc_diff_se": float(auc_diff.std(ddof=1) / root),
        "tpr_per_example": float(np.mean(tpr["per_example"])),
        "tpr_global": float(np.mean(tpr["global"])),
        "tpr_diff_global_minus_per_example": float(tpr_diff.mean()),
        "tpr_diff_se": float(tpr_diff.std(ddof=1) / root),
        "fpr": float(fpr),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="reports/statistical_review")
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args(argv)

    results = [
        compare_variance_rules(
            scenario, k, replicates=args.replicates, seed=args.seed + 1000 * i + k
        )
        for i, scenario in enumerate(SCENARIOS)
        for k in DEFAULT_SHADOW_COUNTS
    ]
    payload = {
        "description": "Synthetic comparison of per-example and pooled OUT variance in "
        "offline LiRA. Not evidence about the insurance models.",
        "assumptions": [
            "Gaussian observations on the logit-confidence scale, mapped to valid BCE",
            "member shift of 0.6 example-level standard deviations in both scenarios",
            "heteroscedastic: per-example sigma log-normal with log-sd 0.6",
            "520 examples, half members; TPR read at 1% FPR",
        ],
        "simulation_seed": args.seed,
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lira_variance_simulation.json").write_text(json.dumps(payload, indent=2))
    for row in results:
        print(json.dumps(row))


if __name__ == "__main__":  # pragma: no cover
    main()
