"""Attack-power control experiment for the insurance ``high_cost`` task.

The powered two-configuration DP spike (``research/two_config_spike.py``)
returned a null result: subgroup LiRA was near chance under every DP-SGD
recipe. That is only informative if the attack pipeline can detect membership
leakage *at all* on this dataset. This script is the control experiment:

* **Control A** -- a non-private MLP matching the DP target architecture.
* **Control B** -- a deliberately over-parameterised, long-trained MLP with no
  regularisation, meant to memorise.
* **Control C** -- an unconstrained decision tree, the classic positive control.

Nothing here is differentially private. There is no Opacus, no clipping and no
noise: the point is to give the attack the easiest possible target. If offline
and online LiRA stay near chance against a model with a clear train--test gap,
the attack -- not DP -- is the limiting factor, and the iso-epsilon subgroup
research needs a larger dataset.

Attack statistics are tested against a stratified membership permutation null
(labels reshuffled within each sensitive group, scores fixed) as well as being
bracketed by stratified bootstrap intervals. The bootstrap says how precise an
estimate is; the permutation test says whether an estimate that large happens by
chance.

Run from the repository root::

    python research/attack_power_control.py --shadows 32 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier

TASK_NAME = "high_cost"
SENSITIVE_ATTRIBUTE = "sex"
ATTACK_FPRS = (0.001, 0.01, 0.1)
HEADLINE_FPR = 0.01
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_SHADOWS = 32
DEFAULT_BOOTSTRAP_REPS = 1000
DEFAULT_PERMUTATION_REPS = 1000

#: A control must show at least this much train--test separation on one of the
#: three memorisation probes before the attack is worth running.
MEMORISATION_GATE = 0.03
#: LiRA needs a usable Gaussian fit; never fit one from fewer observations.
MIN_GAUSSIAN_OBSERVATIONS = 10

#: Decision-table thresholds.
DETECT_MEAN_AUC = 0.55
CHANCE_AUC_BAND = 0.53
#: Significance level for the permutation tests.
ALPHA = 0.05
#: A verdict needs this many seeds to reject the permutation null.
MIN_SIGNIFICANT_SEEDS = 2

SUBGROUP_SUPPORTED = "SUBGROUP DISPARITY SUPPORTED"
SUBGROUP_UNSUPPORTED = "SUBGROUP DISPARITY UNSUPPORTED"
SUBGROUP_INCONCLUSIVE = "SUBGROUP DISPARITY INCONCLUSIVE"

DETECTS = "ATTACK DETECTS LEAKAGE"
INCONCLUSIVE = "ATTACK INCONCLUSIVE"
FAILS_POSITIVE_CONTROL = "ATTACK FAILS POSITIVE CONTROL"


@dataclass(frozen=True)
class Control:
    """One non-private control model definition.

    ``positive_control`` marks the models that are *supposed* to memorise; only
    those can produce an ``ATTACK FAILS POSITIVE CONTROL`` verdict.
    """

    name: str
    kind: str  # "mlp" or "tree"
    description: str
    positive_control: bool = False
    hidden_units: tuple[int, ...] = ()
    epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 1e-2
    tree_kwargs: dict[str, object] = field(default_factory=dict)

    def architecture(self) -> str:
        if self.kind == "tree":
            params = ", ".join(f"{k}={v}" for k, v in sorted(self.tree_kwargs.items()))
            return f"DecisionTreeClassifier({params or 'unconstrained'})"
        layers = " -> ".join(str(units) for units in self.hidden_units)
        return f"Linear(input_dim, {layers} -> 1) with ReLU between layers"


CONTROLS = (
    Control(
        name="matched_capacity_mlp",
        kind="mlp",
        description=(
            "Non-private MLP with the same architecture, optimiser family, "
            "learning rate and epoch count as the DP-SGD target recipe."
        ),
        hidden_units=(32,),
        epochs=30,
        batch_size=64,
        learning_rate=1e-2,
    ),
    Control(
        name="memorising_mlp",
        kind="mlp",
        description=(
            "Deliberately over-parameterised MLP trained far longer than needed, "
            "with no dropout, weight decay or early stopping."
        ),
        positive_control=True,
        hidden_units=(128, 128, 64),
        epochs=400,
        batch_size=64,
        learning_rate=1e-3,
    ),
    Control(
        name="unbounded_decision_tree",
        kind="tree",
        description=(
            "Unconstrained decision tree grown to purity; the strongest "
            "memorising positive control available in this repository."
        ),
        positive_control=True,
        tree_kwargs={"max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2},
    ),
)


# --------------------------------------------------------------------------- #
# Pure helpers (no torch import needed, so the tests stay cheap)
# --------------------------------------------------------------------------- #


def per_example_bce(y_true: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Numerically stable per-example binary cross-entropy."""
    prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    y_true = np.asarray(y_true, dtype=float)
    return -(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))


def make_equalised_attack_cohort(
    groups: np.ndarray,
    membership: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Equalise every ``group x membership`` cell to the smallest cell size.

    Same construction as the powered DP spike, so cohort sizes stay comparable
    between the two experiments.
    """
    rng = np.random.default_rng(seed)
    string_groups = groups.astype(str)
    unique_groups = sorted(str(value) for value in np.unique(string_groups))
    cells: dict[tuple[str, int], np.ndarray] = {}
    for group in unique_groups:
        for member in (0, 1):
            idx = np.flatnonzero((string_groups == group) & (membership == member))
            if idx.size == 0:
                raise RuntimeError(f"empty attack cell group={group!r}, membership={member}")
            cells[(group, member)] = idx

    per_cell = min(len(idx) for idx in cells.values())
    selected: list[int] = []
    counts: dict[str, int] = {}
    for (group, member), idx in cells.items():
        chosen = rng.choice(idx, size=per_cell, replace=False)
        selected.extend(chosen.tolist())
        counts[f"{group}|member={member}"] = int(per_cell)

    return np.asarray(sorted(selected), dtype=int), counts


def balanced_shadow_train_sets(
    y_pool: np.ndarray,
    train_size: int,
    num_shadows: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Exact-size shadow training sets with near-uniform IN/OUT coverage.

    Ported from ``research/powered_spike_runner.py``: a cyclic sliding window
    over a random permutation excludes each pool row a near-equal number of
    times, so every audited example collects a similar number of IN and OUT
    shadow observations while the training size stays pinned to the target's.

    Returns:
        The per-shadow training indices and, per pool row, how often it was
        excluded (its OUT count).
    """
    n = len(y_pool)
    if train_size >= n:
        raise ValueError("shadow train_size must be smaller than the shadow pool")
    if num_shadows < 2:
        raise ValueError("num_shadows must be at least two")

    exclude_size = n - train_size
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    excluded_counts = np.zeros(n, dtype=int)
    train_sets: list[np.ndarray] = []

    for shadow_id in range(num_shadows):
        start = (shadow_id * n) // num_shadows
        positions = (start + np.arange(exclude_size)) % n
        excluded_idx = permutation[positions]
        excluded_counts[excluded_idx] += 1

        in_mask = np.ones(n, dtype=bool)
        in_mask[excluded_idx] = False
        train_idx = np.flatnonzero(in_mask)
        if len(train_idx) != train_size:
            raise AssertionError("balanced schedule changed shadow train size")
        if np.unique(y_pool[train_idx]).size != 2:
            raise RuntimeError("balanced schedule produced a one-class shadow set")
        train_sets.append(train_idx)

    return train_sets, excluded_counts


def memorisation_metrics(
    y_train: np.ndarray,
    train_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
) -> dict[str, float]:
    """Utility and memorisation probes for one target model."""
    train_loss = float(per_example_bce(y_train, train_prob).mean())
    test_loss = float(per_example_bce(y_test, test_prob).mean())
    train_auc = float(roc_auc_score(y_train, train_prob))
    test_auc = float(roc_auc_score(y_test, test_prob))
    train_acc = float(accuracy_score(y_train, (np.asarray(train_prob) >= 0.5).astype(int)))
    test_acc = float(accuracy_score(y_test, (np.asarray(test_prob) >= 0.5).astype(int)))
    return {
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "auc_gap": train_auc - test_auc,
        "loss_gap": test_loss - train_loss,
        "accuracy_gap": train_acc - test_acc,
    }


def memorisation_gate_passed(metrics: dict[str, float], gate: float = MEMORISATION_GATE) -> bool:
    """True when any of the three memorisation probes clears ``gate``.

    Deliberately *not* the old ROC-AUC >= 0.70 utility gate: a model can
    generalise well and still be a useless membership-inference target.
    """
    return (
        metrics["auc_gap"] >= gate
        or metrics["loss_gap"] >= gate
        or metrics["accuracy_gap"] >= gate
    )


def tpr_at_fpr(membership: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Highest TPR attainable without exceeding ``target_fpr``."""
    membership = np.asarray(membership)
    if np.unique(membership).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(membership, scores)
    allowed = fpr <= target_fpr
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def operating_point_counts(
    membership: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> dict[str, int | float]:
    """Detected members and false positives at the selected FPR threshold."""
    membership = np.asarray(membership)
    fpr, tpr, thresholds = roc_curve(membership, scores)
    allowed = np.flatnonzero(fpr <= target_fpr)
    n_members = int((membership == 1).sum())
    n_nonmembers = int((membership == 0).sum())
    if allowed.size == 0:
        return {
            "threshold": float("nan"),
            "detected_members": 0,
            "false_positives": 0,
            "n_members": n_members,
            "n_nonmembers": n_nonmembers,
        }
    best = allowed[int(np.argmax(tpr[allowed]))]
    return {
        "threshold": float(thresholds[best]),
        "detected_members": int(round(tpr[best] * n_members)),
        "false_positives": int(round(fpr[best] * n_nonmembers)),
        "n_members": n_members,
        "n_nonmembers": n_nonmembers,
    }


def signed_subgroup_difference(group_tprs: dict[str, float]) -> float:
    """``male_tpr - female_tpr`` at the headline operating point."""
    keys = {key.lower(): key for key in group_tprs}
    if "male" in keys and "female" in keys:
        return float(group_tprs[keys["male"]] - group_tprs[keys["female"]])
    ordered = sorted(group_tprs)
    if len(ordered) != 2:
        return float("nan")
    return float(group_tprs[ordered[1]] - group_tprs[ordered[0]])


def absolute_subgroup_gap(group_tprs: dict[str, float]) -> float:
    """``max_group_tpr - min_group_tpr``."""
    values = list(group_tprs.values())
    if not values:
        return float("nan")
    return float(max(values) - min(values))


def stratified_bootstrap(
    groups: np.ndarray,
    membership: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = HEADLINE_FPR,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Bootstrap the headline attack statistics within each cell.

    Resampling happens separately inside every ``group x membership`` cell, so
    each replicate keeps the equalised cohort design of the point estimate.
    """
    groups = np.asarray(groups).astype(str)
    membership = np.asarray(membership)
    scores = np.asarray(scores, dtype=float)
    unique_groups = sorted(np.unique(groups))
    cells = {
        (group, member): np.flatnonzero((groups == group) & (membership == member))
        for group in unique_groups
        for member in (0, 1)
    }
    if any(len(idx) == 0 for idx in cells.values()):
        raise RuntimeError("bootstrap requires non-empty group x membership cells")

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"aggregate": []}
    for group in unique_groups:
        draws[group] = []
    draws["signed_difference"] = []
    draws["absolute_gap"] = []

    for _ in range(reps):
        group_tprs: dict[str, float] = {}
        all_idx: list[np.ndarray] = []
        for group in unique_groups:
            member_idx = rng.choice(cells[(group, 1)], size=len(cells[(group, 1)]), replace=True)
            nonmember_idx = rng.choice(
                cells[(group, 0)], size=len(cells[(group, 0)]), replace=True
            )
            idx = np.concatenate([member_idx, nonmember_idx])
            all_idx.append(idx)
            group_tprs[group] = tpr_at_fpr(membership[idx], scores[idx], target_fpr)
        pooled = np.concatenate(all_idx)
        draws["aggregate"].append(tpr_at_fpr(membership[pooled], scores[pooled], target_fpr))
        for group, value in group_tprs.items():
            draws[group].append(value)
        draws["signed_difference"].append(signed_subgroup_difference(group_tprs))
        draws["absolute_gap"].append(absolute_subgroup_gap(group_tprs))

    summary: dict[str, dict[str, float]] = {}
    for key, values in draws.items():
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            summary[key] = {
                "reps": int(reps),
                "mean": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
            }
            continue
        low, high = np.percentile(finite, [2.5, 97.5])
        summary[key] = {
            "reps": int(reps),
            "mean": float(finite.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return summary


def _permutation_pvalue(observed: float, null_draws: np.ndarray, alternative: str) -> float:
    """Empirical p-value with the finite-sample ``(1 + k) / (reps + 1)`` correction.

    ``alternative`` is ``"greater"`` for one-sided tests (AUC, TPR, absolute
    gap) and ``"two-sided"`` for the signed subgroup difference, where the null
    draws and the observation are compared on absolute value.
    """
    finite = null_draws[np.isfinite(null_draws)]
    reps = int(len(null_draws))
    if not np.isfinite(observed) or finite.size == 0:
        return float("nan")
    if alternative == "two-sided":
        extreme = int(np.sum(np.abs(finite) >= abs(observed)))
    else:
        extreme = int(np.sum(finite >= observed))
    return float((1 + extreme) / (reps + 1))


def stratified_membership_permutation_test(
    groups: np.ndarray,
    membership: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = HEADLINE_FPR,
    reps: int = DEFAULT_PERMUTATION_REPS,
    seed: int = 0,
) -> dict[str, dict[str, float | str]]:
    """Permutation null for the attack statistics, stratified by sensitive group.

    Each replicate reshuffles the membership labels *within* each sensitive
    group, so every group keeps its exact member/non-member counts while the
    scores stay fixed. That destroys any association between score and
    membership but preserves the cohort geometry, the class balance and the
    discreteness of the operating point -- which is what a bootstrap interval
    alone cannot tell you. The bootstrap says how precise the estimate is; this
    says whether an estimate that size happens by chance.

    Returns:
        One entry per statistic (``aggregate_auc``, ``aggregate_tpr``, each
        group's TPR, ``signed_difference`` and ``absolute_gap``) holding the
        observed value, the permutation-null mean and 95% interval, and the
        empirical p-value.
    """
    groups = np.asarray(groups).astype(str)
    membership = np.asarray(membership)
    scores = np.asarray(scores, dtype=float)
    unique_groups = sorted(np.unique(groups))
    group_idx = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for group, idx in group_idx.items():
        if np.unique(membership[idx]).size < 2:
            raise RuntimeError(f"permutation test needs both classes in group {group!r}")

    def statistics(labels: np.ndarray) -> dict[str, float]:
        group_tprs = {
            group: tpr_at_fpr(labels[idx], scores[idx], target_fpr)
            for group, idx in group_idx.items()
        }
        values = {
            "aggregate_auc": float(roc_auc_score(labels, scores)),
            "aggregate_tpr": tpr_at_fpr(labels, scores, target_fpr),
            "signed_difference": signed_subgroup_difference(group_tprs),
            "absolute_gap": absolute_subgroup_gap(group_tprs),
        }
        values.update(group_tprs)
        return values

    observed = statistics(membership)
    rng = np.random.default_rng(seed)
    null_draws: dict[str, list[float]] = {key: [] for key in observed}

    for _ in range(reps):
        permuted = membership.copy()
        for idx in group_idx.values():
            # Shuffling within the group's own labels preserves its exact
            # member and non-member counts.
            permuted[idx] = rng.permutation(membership[idx])
        for key, value in statistics(permuted).items():
            null_draws[key].append(value)

    alternatives = {"signed_difference": "two-sided"}
    summary: dict[str, dict[str, float | str]] = {}
    for key, value in observed.items():
        draws = np.asarray(null_draws[key], dtype=float)
        finite = draws[np.isfinite(draws)]
        low, high = (
            np.percentile(finite, [2.5, 97.5]) if finite.size else (float("nan"),) * 2
        )
        summary[key] = {
            "reps": int(reps),
            "observed": float(value),
            "null_mean": float(finite.mean()) if finite.size else float("nan"),
            "null_ci95_low": float(low),
            "null_ci95_high": float(high),
            "alternative": alternatives.get(key, "greater"),
            "p_value": _permutation_pvalue(
                value, draws, alternatives.get(key, "greater")
            ),
        }
    return summary


def attack_metrics(
    membership: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    bootstrap_reps: int,
    seed: int,
    permutation_reps: int = DEFAULT_PERMUTATION_REPS,
) -> dict[str, object]:
    """Aggregate and subgroup metrics for one score vector."""
    membership = np.asarray(membership)
    groups = np.asarray(groups).astype(str)
    scores = np.asarray(scores, dtype=float)

    aggregate = {
        "auc": float(roc_auc_score(membership, scores)),
        **{
            f"tpr_at_fpr_{fpr}": tpr_at_fpr(membership, scores, fpr) for fpr in ATTACK_FPRS
        },
        "operating_point": operating_point_counts(membership, scores, HEADLINE_FPR),
    }

    subgroups: dict[str, dict[str, object]] = {}
    group_tprs: dict[str, float] = {}
    for group in sorted(np.unique(groups)):
        mask = groups == group
        subgroups[group] = {
            "n": int(mask.sum()),
            "members": int((membership[mask] == 1).sum()),
            "nonmembers": int((membership[mask] == 0).sum()),
            "auc": float(roc_auc_score(membership[mask], scores[mask])),
            **{
                f"tpr_at_fpr_{fpr}": tpr_at_fpr(membership[mask], scores[mask], fpr)
                for fpr in ATTACK_FPRS
            },
            "operating_point": operating_point_counts(
                membership[mask], scores[mask], HEADLINE_FPR
            ),
        }
        group_tprs[group] = float(subgroups[group][f"tpr_at_fpr_{HEADLINE_FPR}"])

    return {
        "aggregate": aggregate,
        "subgroups": subgroups,
        "signed_subgroup_difference": signed_subgroup_difference(group_tprs),
        "absolute_subgroup_gap": absolute_subgroup_gap(group_tprs),
        "bootstrap": stratified_bootstrap(
            groups, membership, scores, HEADLINE_FPR, bootstrap_reps, seed
        ),
        "permutation": stratified_membership_permutation_test(
            groups, membership, scores, HEADLINE_FPR, permutation_reps, seed + 1
        ),
    }


def classify_control(
    control: Control,
    seed_results: Sequence[dict[str, object]],
) -> tuple[str, str]:
    """Return the decision-table verdict and its one-line justification."""
    completed = [r for r in seed_results if r["attack_status"] == "completed"]
    gate_passed = [bool(r["memorisation_gate_passed"]) for r in seed_results]

    if not completed:
        if not any(gate_passed):
            return (
                INCONCLUSIVE,
                "No seed cleared the memorisation gate, so the attack never ran.",
            )
        return INCONCLUSIVE, "No attack completed for this control."

    aucs = np.asarray(
        [float(r["offline"]["aggregate"]["auc"]) for r in completed], dtype=float
    )
    ci_lows = np.asarray(
        [float(r["offline"]["bootstrap"]["aggregate"]["ci95_low"]) for r in completed],
        dtype=float,
    )
    auc_p = np.asarray(
        [float(r["offline"]["permutation"]["aggregate_auc"]["p_value"]) for r in completed],
        dtype=float,
    )
    tpr_p = np.asarray(
        [float(r["offline"]["permutation"]["aggregate_tpr"]["p_value"]) for r in completed],
        dtype=float,
    )
    mean_auc = float(np.nanmean(aucs))
    # Random-guessing baseline at a 1% FPR operating point is a 1% TPR.
    above_baseline = int(np.sum(ci_lows > HEADLINE_FPR))
    significant = int(np.sum((auc_p < ALPHA) | (tpr_p < ALPHA)))
    consistent_direction = bool(np.all(aucs > 0.5))

    if (
        all(gate_passed)
        and mean_auc >= DETECT_MEAN_AUC
        and significant >= MIN_SIGNIFICANT_SEEDS
        and consistent_direction
    ):
        return (
            DETECTS,
            f"Mean aggregate offline-LiRA AUC {mean_auc:.4f}, above chance on every seed; "
            f"the stratified membership permutation null is rejected at p<{ALPHA} on "
            f"{significant}/{len(completed)} seeds "
            f"({above_baseline}/{len(completed)} bootstrap TPR@1% intervals also clear the "
            "1% random baseline).",
        )

    if (
        control.positive_control
        and all(gate_passed)
        and mean_auc < CHANCE_AUC_BAND
        and significant < MIN_SIGNIFICANT_SEEDS
    ):
        return (
            FAILS_POSITIVE_CONTROL,
            f"Every seed shows a clear train-test gap, yet mean aggregate LiRA AUC is "
            f"{mean_auc:.4f} and the permutation null survives on "
            f"{len(completed) - significant}/{len(completed)} seeds.",
        )

    return (
        INCONCLUSIVE,
        f"Mean aggregate LiRA AUC {mean_auc:.4f} with the permutation null rejected on "
        f"only {significant}/{len(completed)} seeds.",
    )


def assess_subgroup_disparity(seed_results: Sequence[dict[str, object]]) -> tuple[str, str]:
    """Judge subgroup TPR disparity separately from aggregate attack power.

    Aggregate leakage says nothing about whether one sex leaks more than the
    other, so this never inherits the aggregate verdict. Disparity is called
    supported only when the signed direction is stable across seeds, the
    permutation null is rejected on enough seeds, and the bootstrap interval is
    not simply the discrete operating-point resolution.
    """
    completed = [r for r in seed_results if r["attack_status"] == "completed"]
    if not completed:
        return SUBGROUP_INCONCLUSIVE, "No attack completed, so no subgroup comparison exists."

    signed = np.asarray(
        [float(r["offline"]["signed_subgroup_difference"]) for r in completed], dtype=float
    )
    signed_p = np.asarray(
        [
            float(r["offline"]["permutation"]["signed_difference"]["p_value"])
            for r in completed
        ],
        dtype=float,
    )
    gap_p = np.asarray(
        [float(r["offline"]["permutation"]["absolute_gap"]["p_value"]) for r in completed],
        dtype=float,
    )
    significant = int(np.sum((signed_p < ALPHA) | (gap_p < ALPHA)))
    nonzero = signed[signed != 0.0]
    stable_direction = bool(nonzero.size and np.all(np.sign(nonzero) == np.sign(nonzero[0])))

    # One extra member at the 1% FPR operating point moves a subgroup TPR by
    # 1/n_members. A "gap" smaller than that is resolution, not disparity.
    resolutions = [
        1.0 / max(int(r["offline"]["subgroups"][group]["members"]), 1)
        for r in completed
        for group in r["offline"]["subgroups"]
    ]
    resolution = float(max(resolutions)) if resolutions else float("inf")
    resolvable = bool(np.all(np.abs(signed) >= resolution) and signed.size)

    if significant >= MIN_SIGNIFICANT_SEEDS and stable_direction and resolvable:
        direction = "male" if np.mean(signed) > 0 else "female"
        return (
            SUBGROUP_SUPPORTED,
            f"Signed male-female TPR@1% difference keeps the same sign on every seed "
            f"({direction} more exposed), permutation p<{ALPHA} on "
            f"{significant}/{len(completed)} seeds, and the effect exceeds the "
            f"{resolution:.4f} operating-point resolution.",
        )

    reasons = []
    if not stable_direction:
        reasons.append("the signed direction is not stable across seeds")
    if significant < MIN_SIGNIFICANT_SEEDS:
        reasons.append(
            f"the permutation null is rejected on only {significant}/{len(completed)} seeds"
        )
    if not resolvable:
        reasons.append(
            f"at least one gap is within the {resolution:.4f} discrete operating-point "
            "resolution"
        )
    return SUBGROUP_UNSUPPORTED, "; ".join(reasons).capitalize() + "."


# --------------------------------------------------------------------------- #
# Model training (torch is imported lazily so the tests do not need it)
# --------------------------------------------------------------------------- #


def build_mlp(n_features: int, hidden_units: Sequence[int], seed: int):
    """Build a plain feed-forward classifier with ReLU activations."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    in_dim = n_features
    for units in hidden_units:
        layers.append(nn.Linear(in_dim, units))
        layers.append(nn.ReLU())
        in_dim = units
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def train_control_model(X: np.ndarray, y: np.ndarray, control: Control, seed: int):
    """Train one non-private control model. No Opacus, clipping or noise."""
    if control.kind == "tree":
        model = DecisionTreeClassifier(random_state=seed, **control.tree_kwargs)
        model.fit(np.asarray(X), np.asarray(y))
        return model

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(control.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    model = build_mlp(X.shape[1], control.hidden_units, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=control.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(control.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
    return model


def predict_probability(model, X: np.ndarray) -> np.ndarray:
    """Positive-class probabilities for either control family."""
    if isinstance(model, DecisionTreeClassifier):
        proba = model.predict_proba(np.asarray(X))
        if proba.shape[1] == 1:
            only = float(model.classes_[0])
            return np.full(len(X), only, dtype=float)
        return np.asarray(proba[:, 1], dtype=float)

    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


def collect_shadow_losses(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    attack_idx: np.ndarray,
    target_train_size: int,
    control: Control,
    num_shadows: int,
    seed: int,
    log,
) -> dict[str, object]:
    """Train balanced, size-matched shadows and keep both IN and OUT losses."""
    train_sets, excluded_counts = balanced_shadow_train_sets(
        y_pool, train_size=target_train_size, num_shadows=num_shadows, seed=seed
    )
    out_losses: list[list[float]] = [[] for _ in attack_idx]
    in_losses: list[list[float]] = [[] for _ in attack_idx]

    for shadow_id, train_idx in enumerate(train_sets):
        if len(train_idx) != target_train_size:
            raise AssertionError("shadow and target train sizes differ")
        in_mask = np.zeros(len(y_pool), dtype=bool)
        in_mask[train_idx] = True

        model = train_control_model(
            X_pool[train_idx], y_pool[train_idx], control, seed=seed + shadow_id + 1
        )
        losses = per_example_bce(
            y_pool[attack_idx], predict_probability(model, X_pool[attack_idx])
        )
        attack_in = in_mask[attack_idx]
        for local_idx, is_in in enumerate(attack_in):
            if is_in:
                in_losses[local_idx].append(float(losses[local_idx]))
            else:
                out_losses[local_idx].append(float(losses[local_idx]))

        log(
            f"[{control.name}] shadow {shadow_id + 1}/{num_shadows} "
            f"n={len(train_idx)} in_cover={int(attack_in.sum())}/{len(attack_idx)}"
        )

    min_out = min(len(values) for values in out_losses)
    min_in = min(len(values) for values in in_losses)
    expected_min_out = int(excluded_counts[attack_idx].min())
    if min_out != expected_min_out:
        raise AssertionError(
            f"recorded OUT losses ({min_out}) do not match schedule ({expected_min_out})"
        )
    if min_out < MIN_GAUSSIAN_OBSERVATIONS:
        raise RuntimeError(
            f"offline LiRA needs at least {MIN_GAUSSIAN_OBSERVATIONS} OUT observations "
            f"per example; minimum was {min_out}. Increase --shadows."
        )

    out_matrix = np.vstack([np.asarray(v[:min_out], dtype=float) for v in out_losses])
    in_matrix = (
        np.vstack([np.asarray(v[:min_in], dtype=float) for v in in_losses])
        if min_in >= MIN_GAUSSIAN_OBSERVATIONS
        else None
    )
    return {
        "out_matrix": out_matrix,
        "in_matrix": in_matrix,
        "min_out": int(min_out),
        "min_in": int(min_in),
    }


def run_control_seed(
    control: Control,
    data_arrays: dict[str, np.ndarray],
    num_shadows: int,
    bootstrap_reps: int,
    permutation_reps: int,
    seed: int,
    log,
) -> dict[str, object]:
    """Train one control target, gate it, then attack it if it memorised."""
    from dp.attacks import lira_offline_attack, lira_online_attack

    X_train = data_arrays["X_train"]
    y_train = data_arrays["y_train"]
    X_val = data_arrays["X_val"]
    y_val = data_arrays["y_val"]
    X_test = data_arrays["X_test"]
    y_test = data_arrays["y_test"]
    groups_train = data_arrays["groups_train"]
    groups_test = data_arrays["groups_test"]

    X_attack = np.vstack([X_train, X_test])
    y_attack = np.concatenate([y_train, y_test])
    groups_attack = np.concatenate([groups_train, groups_test]).astype(str)
    membership = np.concatenate(
        [np.ones(len(y_train), dtype=int), np.zeros(len(y_test), dtype=int)]
    )
    attack_idx, cohort_counts = make_equalised_attack_cohort(groups_attack, membership, seed)

    target = train_control_model(X_train, y_train, control, seed=seed)
    metrics = memorisation_metrics(
        y_train,
        predict_probability(target, X_train),
        y_test,
        predict_probability(target, X_test),
    )
    gate_passed = memorisation_gate_passed(metrics)

    result: dict[str, object] = {
        "control": control.name,
        "kind": control.kind,
        "architecture": control.architecture(),
        "positive_control": control.positive_control,
        "seed": int(seed),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "cohort_counts": cohort_counts,
        "num_attack_examples": int(len(attack_idx)),
        "memorisation": metrics,
        "memorisation_gate": MEMORISATION_GATE,
        "memorisation_gate_passed": bool(gate_passed),
        "attack_status": "pending",
        "offline": None,
        "online": None,
        "online_available": False,
    }

    log(
        f"[{control.name} seed={seed}] train_auc={metrics['train_auc']:.4f} "
        f"test_auc={metrics['test_auc']:.4f} auc_gap={metrics['auc_gap']:.4f} "
        f"loss_gap={metrics['loss_gap']:.4f} acc_gap={metrics['accuracy_gap']:.4f} "
        f"gate={'PASS' if gate_passed else 'FAIL'}"
    )
    if not gate_passed:
        result["attack_status"] = "skipped_memorisation_gate"
        return result

    # Shadow pool spans all partitions; remap attack indices into pool space.
    X_pool = np.vstack([X_train, X_val, X_test])
    y_pool = np.concatenate([y_train, y_val, y_test])
    attack_pool_idx = attack_idx.copy()
    attack_pool_idx[attack_pool_idx >= len(y_train)] += len(y_val)

    target_losses = per_example_bce(
        y_attack[attack_idx], predict_probability(target, X_attack)[attack_idx]
    )
    shadows = collect_shadow_losses(
        X_pool,
        y_pool,
        attack_pool_idx,
        target_train_size=len(y_train),
        control=control,
        num_shadows=num_shadows,
        seed=seed + 10_000,
        log=log,
    )

    attack_groups = groups_attack[attack_idx]
    attack_membership = membership[attack_idx]

    offline = lira_offline_attack(
        target_losses, attack_membership, shadows["out_matrix"], fprs=ATTACK_FPRS
    )
    result.update(
        {
            "attack_status": "completed",
            "num_shadows": int(num_shadows),
            "shadow_train_size": int(len(y_train)),
            "min_out_observations": shadows["min_out"],
            "min_in_observations": shadows["min_in"],
            "offline": attack_metrics(
                attack_membership,
                attack_groups,
                offline.scores,
                bootstrap_reps,
                seed + 20_000,
                permutation_reps,
            ),
        }
    )

    if shadows["in_matrix"] is not None:
        online = lira_online_attack(
            target_losses,
            attack_membership,
            shadows["in_matrix"],
            shadows["out_matrix"],
            fprs=ATTACK_FPRS,
        )
        result["online_available"] = True
        result["online"] = attack_metrics(
            attack_membership,
            attack_groups,
            online.scores,
            bootstrap_reps,
            seed + 30_000,
            permutation_reps,
        )
    else:
        result["online_unavailable_reason"] = (
            f"only {shadows['min_in']} IN observations per example; "
            f"{MIN_GAUSSIAN_OBSERVATIONS} required"
        )
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _ci(entry: dict[str, float] | None) -> str:
    if not entry:
        return "-"
    return f"[{_fmt(entry['ci95_low'])}, {_fmt(entry['ci95_high'])}]"


def build_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    """Flatten the payload into one CSV row per control/seed/attack variant."""
    rows: list[dict[str, object]] = []
    for result in payload["seed_results"]:
        for variant in ("offline", "online"):
            metrics = result.get(variant)
            if not metrics:
                continue
            aggregate = metrics["aggregate"]
            bootstrap = metrics["bootstrap"]
            permutation = metrics["permutation"]
            row: dict[str, object] = {
                "control": result["control"],
                "seed": result["seed"],
                "attack": f"lira-{variant}",
                "attack_status": result["attack_status"],
                "memorisation_gate_passed": result["memorisation_gate_passed"],
                "train_size": result["train_size"],
                "num_attack_examples": result["num_attack_examples"],
                "num_shadows": result.get("num_shadows"),
                "min_out_observations": result.get("min_out_observations"),
                "min_in_observations": result.get("min_in_observations"),
                "train_auc": result["memorisation"]["train_auc"],
                "test_auc": result["memorisation"]["test_auc"],
                "auc_gap": result["memorisation"]["auc_gap"],
                "loss_gap": result["memorisation"]["loss_gap"],
                "accuracy_gap": result["memorisation"]["accuracy_gap"],
                "lira_auc": aggregate["auc"],
                "tpr_at_fpr_0.001": aggregate["tpr_at_fpr_0.001"],
                "tpr_at_fpr_0.01": aggregate["tpr_at_fpr_0.01"],
                "tpr_at_fpr_0.1": aggregate["tpr_at_fpr_0.1"],
                "detected_members_at_1pct_fpr": aggregate["operating_point"][
                    "detected_members"
                ],
                "false_positives_at_1pct_fpr": aggregate["operating_point"][
                    "false_positives"
                ],
                "aggregate_tpr_ci_low": bootstrap["aggregate"]["ci95_low"],
                "aggregate_tpr_ci_high": bootstrap["aggregate"]["ci95_high"],
                "permutation_p_aggregate_auc": permutation["aggregate_auc"]["p_value"],
                "permutation_p_aggregate_tpr": permutation["aggregate_tpr"]["p_value"],
                "permutation_null_auc_mean": permutation["aggregate_auc"]["null_mean"],
                "permutation_null_auc_ci_low": permutation["aggregate_auc"]["null_ci95_low"],
                "permutation_null_auc_ci_high": permutation["aggregate_auc"]["null_ci95_high"],
                "permutation_null_tpr_mean": permutation["aggregate_tpr"]["null_mean"],
                "permutation_null_tpr_ci_low": permutation["aggregate_tpr"]["null_ci95_low"],
                "permutation_null_tpr_ci_high": permutation["aggregate_tpr"]["null_ci95_high"],
                "permutation_p_signed_difference": permutation["signed_difference"]["p_value"],
                "permutation_p_absolute_gap": permutation["absolute_gap"]["p_value"],
                "permutation_null_signed_ci_low": permutation["signed_difference"][
                    "null_ci95_low"
                ],
                "permutation_null_signed_ci_high": permutation["signed_difference"][
                    "null_ci95_high"
                ],
                "permutation_null_gap_ci_low": permutation["absolute_gap"]["null_ci95_low"],
                "permutation_null_gap_ci_high": permutation["absolute_gap"]["null_ci95_high"],
                "signed_subgroup_difference": metrics["signed_subgroup_difference"],
                "absolute_subgroup_gap": metrics["absolute_subgroup_gap"],
                "signed_difference_ci_low": bootstrap["signed_difference"]["ci95_low"],
                "signed_difference_ci_high": bootstrap["signed_difference"]["ci95_high"],
                "absolute_gap_ci_low": bootstrap["absolute_gap"]["ci95_low"],
                "absolute_gap_ci_high": bootstrap["absolute_gap"]["ci95_high"],
            }
            for group, group_metrics in metrics["subgroups"].items():
                row[f"{group}_lira_auc"] = group_metrics["auc"]
                row[f"{group}_tpr_at_fpr_0.01"] = group_metrics["tpr_at_fpr_0.01"]
                row[f"{group}_tpr_ci_low"] = bootstrap[group]["ci95_low"]
                row[f"{group}_tpr_ci_high"] = bootstrap[group]["ci95_high"]
                row[f"{group}_permutation_p"] = permutation[group]["p_value"]
            rows.append(row)

        if not result.get("offline"):
            rows.append(
                {
                    "control": result["control"],
                    "seed": result["seed"],
                    "attack": "lira-offline",
                    "attack_status": result["attack_status"],
                    "memorisation_gate_passed": result["memorisation_gate_passed"],
                    "train_size": result["train_size"],
                    "num_attack_examples": result["num_attack_examples"],
                    "train_auc": result["memorisation"]["train_auc"],
                    "test_auc": result["memorisation"]["test_auc"],
                    "auc_gap": result["memorisation"]["auc_gap"],
                    "loss_gap": result["memorisation"]["loss_gap"],
                    "accuracy_gap": result["memorisation"]["accuracy_gap"],
                }
            )
    return rows


def render_markdown(payload: dict[str, object]) -> str:
    """Render the examiner-facing report, including the go/no-go decision."""
    seed_results: list[dict[str, object]] = list(payload["seed_results"])
    lines = [
        "# Attack-power control experiment",
        "",
        "## 1. Experimental design",
        "",
        f"Task: `{payload['task']}`. Sensitive attribute: `{payload['sensitive_attribute']}`.",
        (
            "This is a control for the powered two-configuration DP spike, which "
            "returned a null subgroup result. Every model here is **non-private**: "
            "no Opacus, no gradient clipping, no noise. If the pipeline cannot "
            "detect membership in an unprotected, deliberately memorising model, "
            "the null DP result says nothing about DP."
        ),
        f"Seeds: {', '.join(str(seed) for seed in payload['seeds'])}.",
        f"Shadow models per control and seed: {payload['num_shadows']} "
        "(balanced fixed-size schedule, training size pinned to the target's).",
        f"Bootstrap replicates: {payload['bootstrap_reps']}, resampled within each "
        "`sex x membership` cell.",
        f"Permutation replicates: "
        f"{payload.get('permutation_reps', DEFAULT_PERMUTATION_REPS)}, with membership "
        "labels reshuffled within each sensitive group. Bootstrap intervals are kept, "
        "but they are not used as the hypothesis test against chance; the permutation "
        "null is.",
        (
            f"Memorisation gate: the attack runs only when train-test ROC-AUC, "
            f"test-train loss or train-test accuracy differs by at least "
            f"{MEMORISATION_GATE}. The old ROC-AUC >= 0.70 utility gate is not used "
            "on its own -- a model can generalise well and still be an unattackable "
            "membership target."
        ),
        "",
        "## 2. Training sizes and cohort counts",
        "",
        "| Control | Seed | Train | Val | Test | Attack cohort | Cells (sex\\|membership) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in seed_results:
        cells = ", ".join(f"{k}={v}" for k, v in sorted(result["cohort_counts"].items()))
        lines.append(
            f"| {result['control']} | {result['seed']} | {result['train_size']} | "
            f"{result['val_size']} | {result['test_size']} | "
            f"{result['num_attack_examples']} | {cells} |"
        )

    lines += [
        "",
        "Target and shadow training sizes are exactly matched; every cohort cell "
        "holds the same number of rows by construction.",
        "",
        "## 3. Target and shadow architectures",
        "",
        "| Control | Kind | Architecture | Training | Positive control |",
        "|---|---|---|---|---|",
    ]
    for control in payload["controls"]:
        if control["kind"] == "tree":
            training = "fitted to purity"
        else:
            training = (
                f"Adam, lr={control['learning_rate']}, "
                f"batch={control['batch_size']}, epochs={control['epochs']}"
            )
        lines.append(
            f"| {control['name']} | {control['kind']} | `{control['architecture']}` | "
            f"{training} | {'yes' if control['positive_control'] else 'no'} |"
        )
    lines.append("")
    lines.append("Shadow models use the identical architecture and recipe as their target.")

    lines += [
        "",
        "## 4. Memorisation metrics",
        "",
        "| Control | Seed | Train AUC | Test AUC | AUC gap | Train BCE | Test BCE | "
        "Loss gap | Train acc | Test acc | Acc gap | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in seed_results:
        m = result["memorisation"]
        lines.append(
            f"| {result['control']} | {result['seed']} | {_fmt(m['train_auc'])} | "
            f"{_fmt(m['test_auc'])} | {_fmt(m['auc_gap'])} | {_fmt(m['train_loss'])} | "
            f"{_fmt(m['test_loss'])} | {_fmt(m['loss_gap'])} | {_fmt(m['train_accuracy'])} | "
            f"{_fmt(m['test_accuracy'])} | {_fmt(m['accuracy_gap'])} | "
            f"{'PASS' if result['memorisation_gate_passed'] else 'FAIL (attack skipped)'} |"
        )

    lines += [
        "",
        "## 5. Aggregate LiRA results",
        "",
        "| Control | Seed | Attack | LiRA AUC | TPR@0.1% | TPR@1% | TPR@10% | "
        "Detected members @1% | False positives @1% | OUT obs | IN obs |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in seed_results:
        if result["attack_status"] != "completed":
            lines.append(
                f"| {result['control']} | {result['seed']} | - | "
                f"**SKIPPED: {result['attack_status']}** | - | - | - | - | - | - | - |"
            )
            continue
        for variant in ("offline", "online"):
            metrics = result.get(variant)
            if not metrics:
                lines.append(
                    f"| {result['control']} | {result['seed']} | lira-{variant} | "
                    f"**unavailable: {result.get('online_unavailable_reason', 'n/a')}** "
                    "| - | - | - | - | - | - | - |"
                )
                continue
            agg = metrics["aggregate"]
            op = agg["operating_point"]
            lines.append(
                f"| {result['control']} | {result['seed']} | lira-{variant} | "
                f"{_fmt(agg['auc'])} | {_fmt(agg['tpr_at_fpr_0.001'])} | "
                f"{_fmt(agg['tpr_at_fpr_0.01'])} | {_fmt(agg['tpr_at_fpr_0.1'])} | "
                f"{op['detected_members']}/{op['n_members']} | "
                f"{op['false_positives']}/{op['n_nonmembers']} | "
                f"{result['min_out_observations']} | {result['min_in_observations']} |"
            )

    lines += [
        "",
        "## 6. Subgroup LiRA results",
        "",
        "| Control | Seed | Attack | Female AUC | Male AUC | Female TPR@1% | "
        "Male TPR@1% | Signed diff (male-female) | Absolute gap |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in seed_results:
        for variant in ("offline", "online"):
            metrics = result.get(variant)
            if not metrics:
                continue
            groups = metrics["subgroups"]

            def group_value(name: str, key: str) -> object:
                for candidate, values in groups.items():
                    if candidate.lower() == name:
                        return values[key]
                return None

            lines.append(
                f"| {result['control']} | {result['seed']} | lira-{variant} | "
                f"{_fmt(group_value('female', 'auc'))} | {_fmt(group_value('male', 'auc'))} | "
                f"{_fmt(group_value('female', 'tpr_at_fpr_0.01'))} | "
                f"{_fmt(group_value('male', 'tpr_at_fpr_0.01'))} | "
                f"{_fmt(metrics['signed_subgroup_difference'])} | "
                f"{_fmt(metrics['absolute_subgroup_gap'])} |"
            )
    if not any(result.get("offline") for result in seed_results):
        lines.append("| - | - | - | - | - | - | - | - | - |")

    lines += [
        "",
        "## 7. Bootstrap 95% confidence intervals",
        "",
        "Stratified within each `sex x membership` cell.",
        "",
        "| Control | Seed | Attack | Aggregate TPR@1% | Female TPR@1% | Male TPR@1% | "
        "Signed diff | Absolute gap |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for result in seed_results:
        for variant in ("offline", "online"):
            metrics = result.get(variant)
            if not metrics:
                continue
            bootstrap = metrics["bootstrap"]

            def ci_for(name: str) -> str:
                for candidate, values in bootstrap.items():
                    if candidate.lower() == name:
                        return _ci(values)
                return "-"

            lines.append(
                f"| {result['control']} | {result['seed']} | lira-{variant} | "
                f"{_ci(bootstrap['aggregate'])} | {ci_for('female')} | {ci_for('male')} | "
                f"{_ci(bootstrap['signed_difference'])} | {_ci(bootstrap['absolute_gap'])} |"
            )
    if not any(result.get("offline") for result in seed_results):
        lines.append("| - | - | - | - | - | - | - | - |")

    lines += [
        "",
        "## 8. Stratified permutation tests against the membership null",
        "",
        (
            f"Membership labels are reshuffled within each sensitive group "
            f"({payload.get('permutation_reps', DEFAULT_PERMUTATION_REPS)} replicates), "
            "preserving each group's exact member and non-member counts while the "
            "attack scores stay fixed. Bootstrap intervals above describe the "
            "precision of an estimate; these p-values ask whether an estimate that "
            "large arises from finite-sample ROC variation alone. AUC, TPR and the "
            "absolute gap use one-sided tests; the signed difference is two-sided on "
            f"absolute value. All p-values use the (1+k)/(reps+1) correction, so the "
            "smallest attainable value is "
            f"{1.0 / (int(payload.get('permutation_reps', DEFAULT_PERMUTATION_REPS)) + 1):.4f}."
        ),
        "",
        "| Control | Seed | Attack | AUC (p) | Null AUC 95% | TPR@1% (p) | Null TPR 95% | "
        "Signed diff (p, two-sided) | Null signed 95% | Absolute gap (p) | Null gap 95% |",
        "|---|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for result in seed_results:
        for variant in ("offline", "online"):
            metrics = result.get(variant)
            if not metrics:
                continue
            perm = metrics["permutation"]

            def cell(key: str) -> str:
                entry = perm[key]
                return f"{_fmt(entry['observed'])} (p={_fmt(entry['p_value'], 4)})"

            def null_ci(key: str) -> str:
                entry = perm[key]
                return f"[{_fmt(entry['null_ci95_low'])}, {_fmt(entry['null_ci95_high'])}]"

            lines.append(
                f"| {result['control']} | {result['seed']} | lira-{variant} | "
                f"{cell('aggregate_auc')} | {null_ci('aggregate_auc')} | "
                f"{cell('aggregate_tpr')} | {null_ci('aggregate_tpr')} | "
                f"{cell('signed_difference')} | {null_ci('signed_difference')} | "
                f"{cell('absolute_gap')} | {null_ci('absolute_gap')} |"
            )
    if not any(result.get("offline") for result in seed_results):
        lines.append("| - | - | - | - | - | - | - | - | - | - | - |")

    lines += [
        "",
        "Per-subgroup TPR@1% permutation p-values:",
        "",
        "| Control | Seed | Attack | Female TPR@1% (p) | Male TPR@1% (p) |",
        "|---|---:|---|---|---|",
    ]
    for result in seed_results:
        for variant in ("offline", "online"):
            metrics = result.get(variant)
            if not metrics:
                continue
            perm = metrics["permutation"]

            def group_cell(name: str) -> str:
                for candidate, entry in perm.items():
                    if candidate.lower() == name:
                        return f"{_fmt(entry['observed'])} (p={_fmt(entry['p_value'], 4)})"
                return "-"

            lines.append(
                f"| {result['control']} | {result['seed']} | lira-{variant} | "
                f"{group_cell('female')} | {group_cell('male')} |"
            )
    if not any(result.get("offline") for result in seed_results):
        lines.append("| - | - | - | - | - |")

    lines += [
        "",
        "## 9. Decision table",
        "",
        "| Control | Positive control | Seeds attacked | Mean memorisation gap (AUC) | "
        "Mean offline LiRA AUC | Mean permutation p (AUC) | Verdict | Basis |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for decision in payload["decisions"]:
        lines.append(
            f"| {decision['control']} | {'yes' if decision['positive_control'] else 'no'} | "
            f"{decision['seeds_attacked']} | {_fmt(decision['mean_auc_gap'])} | "
            f"{_fmt(decision['mean_lira_auc'])} | "
            f"{_fmt(decision.get('mean_permutation_p_auc'))} | "
            f"**{decision['verdict']}** | {decision['basis']} |"
        )

    lines += [
        "",
        "Subgroup disparity is judged separately; an aggregate detection is never "
        "carried over into a disparity claim.",
        "",
        "| Control | Subgroup verdict | Basis |",
        "|---|---|---|",
    ]
    for decision in payload["decisions"]:
        lines.append(
            f"| {decision['control']} | **{decision.get('subgroup_verdict', SUBGROUP_INCONCLUSIVE)}** "
            f"| {decision.get('subgroup_basis', '-')} |"
        )

    failed = [d for d in payload["decisions"] if d["verdict"] == FAILS_POSITIVE_CONTROL]
    detected = [d for d in payload["decisions"] if d["verdict"] == DETECTS]
    supported = [
        d for d in payload["decisions"] if d.get("subgroup_verdict") == SUBGROUP_SUPPORTED
    ]

    lines += [
        "",
        "## Conclusions",
        "",
        "These three questions are answered separately. Aggregate attack success is "
        "not evidence of subgroup disparity, and neither is evidence about the DP "
        "results on its own.",
        "",
        "### 1. Is aggregate membership leakage detectable?",
        "",
    ]
    if detected:
        names = ", ".join(d["control"] for d in detected)
        lines.append(
            f"Yes, on: {names}. Mean aggregate LiRA AUC clears {DETECT_MEAN_AUC} with the "
            f"stratified membership permutation null rejected at p<{ALPHA} on at least "
            f"{MIN_SIGNIFICANT_SEEDS} of the target seeds, in a consistent direction."
        )
    elif failed:
        lines.append(
            "No. Every positive control memorised -- a clear train-test gap on all "
            "seeds -- yet aggregate LiRA stayed near chance and the permutation null "
            "was not rejected consistently."
        )
    else:
        lines.append(
            "Not established. No control produced both an above-chance mean AUC and "
            "consistent rejection of the permutation null."
        )

    lines += ["", "### 2. Does subgroup leakage differ by sex?", ""]
    if supported:
        names = ", ".join(d["control"] for d in supported)
        lines.append(
            f"Supported on: {names} -- stable signed direction across seeds, "
            f"permutation p<{ALPHA} on at least {MIN_SIGNIFICANT_SEEDS} seeds, and an "
            "effect larger than the discrete operating-point resolution."
        )
    else:
        lines.append(
            "Not supported. No control satisfies all three requirements (stable signed "
            "direction across seeds, permutation rejection on at least "
            f"{MIN_SIGNIFICANT_SEEDS} seeds, and a gap exceeding the operating-point "
            "resolution). Where an aggregate attack does succeed, that success is "
            "reported as aggregate leakage only."
        )

    lines += ["", "### 3. Is the insurance dataset still suitable for the iso-epsilon "
              "subgroup research question?", ""]
    if failed and not detected:
        lines += [
            "**Natural membership leakage is not reliably measurable with the current "
            "insurance dataset and attack implementation. The iso-epsilon "
            "subgroup-privacy research should move to a larger dataset rather than "
            "adding further DP configurations.**",
            "",
            "Go/no-go: **NO-GO** for further DP-SGD configurations on the insurance "
            "dataset. The null subgroup result from the powered spike cannot be "
            "attributed to differential privacy, because the same pipeline fails "
            "against an unprotected, demonstrably memorising target.",
        ]
    elif detected and not supported:
        lines += [
            "Partially. The pipeline has measurable aggregate attack power at "
            "non-private capacity, so the powered spike's null is informative about "
            "the DP-SGD configurations rather than about a dead attack. But the "
            "subgroup contrast -- the actual research question -- is not resolvable "
            "here: subgroup TPR@1% FPR moves in units of one member, so the cohort is "
            "too small to separate a real disparity from operating-point resolution.",
            "",
            "Go/no-go: **CONDITIONAL GO** -- aggregate auditing on the insurance "
            "dataset is sound, but the iso-epsilon *subgroup* comparison needs a "
            "larger dataset to reach usable resolution.",
        ]
    elif detected and supported:
        lines += [
            "Yes. The pipeline detects aggregate leakage and a stable subgroup "
            "disparity at non-private capacity, so the dataset can in principle "
            "support the iso-epsilon subgroup comparison.",
            "",
            "Go/no-go: **GO** -- subgroup DP work on this dataset remains "
            "interpretable.",
        ]
    else:
        lines += [
            "Undetermined. No positive control both memorised and stayed at chance, "
            "and no control produced a stable above-chance attack.",
            "",
            "Go/no-go: **HOLD** -- re-run with more shadows or a stronger memorising "
            "target before drawing conclusions about the insurance dataset.",
        ]

    lines += [
        "",
        "Attack AUC near 0.5 is reported here as a failure of attack power, not as "
        "evidence of privacy. No claim of subgroup privacy variance is made unless "
        "section 2 above explicitly supports it.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadows", type=int, default=DEFAULT_SHADOWS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--permutation-reps", type=int, default=DEFAULT_PERMUTATION_REPS)
    parser.add_argument("--output-dir", default="reports/attack_power_control")
    parser.add_argument(
        "--controls",
        nargs="+",
        default=[control.name for control in CONTROLS],
        help="Subset of control names to run.",
    )
    args = parser.parse_args(argv)

    if args.shadows < DEFAULT_SHADOWS:
        raise ValueError(f"Use at least {DEFAULT_SHADOWS} shadows.")
    if len(args.seeds) < 3:
        raise ValueError("Use at least three target seeds.")
    if args.bootstrap_reps < 200:
        raise ValueError("Use at least 200 bootstrap replicates.")
    if args.permutation_reps < 200:
        raise ValueError("Use at least 200 permutation replicates.")

    from dp.pipeline import build_preprocessor
    from dp.tasks import get_task, prepare_task_data

    selected = [control for control in CONTROLS if control.name in set(args.controls)]
    if not selected:
        raise ValueError(f"No known controls selected from {args.controls}")
    if any(control.kind == "mlp" for control in selected):
        import torch

        torch.set_num_threads(2)

    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "attack_power_control.log"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    df = pd.read_csv(root / "data" / "insurance.csv")
    payload: dict[str, object] = {
        "experiment": "attack_power_control",
        "task": TASK_NAME,
        "sensitive_attribute": SENSITIVE_ATTRIBUTE,
        "private": False,
        "seeds": [int(seed) for seed in args.seeds],
        "num_shadows": int(args.shadows),
        "bootstrap_reps": int(args.bootstrap_reps),
        "permutation_reps": int(args.permutation_reps),
        "alpha": ALPHA,
        "memorisation_gate": MEMORISATION_GATE,
        "controls": [
            {**asdict(control), "architecture": control.architecture()}
            for control in selected
        ],
        "seed_results": [],
        "decisions": [],
    }

    for control in selected:
        control_results: list[dict[str, object]] = []
        for seed in args.seeds:
            log(f"\n=== {control.name} | target seed {seed} ===")
            data = prepare_task_data(df, get_task(TASK_NAME), random_state=seed)
            preprocessor = build_preprocessor(data.X_train)
            arrays = {
                "X_train": np.asarray(
                    preprocessor.fit_transform(data.X_train), dtype=np.float32
                ),
                "X_val": np.asarray(preprocessor.transform(data.X_val), dtype=np.float32),
                "X_test": np.asarray(preprocessor.transform(data.X_test), dtype=np.float32),
                "y_train": data.y_train,
                "y_val": data.y_val,
                "y_test": data.y_test,
                "groups_train": data.sensitive_train,
                "groups_test": data.sensitive_test,
            }
            result = run_control_seed(
                control,
                arrays,
                num_shadows=args.shadows,
                bootstrap_reps=args.bootstrap_reps,
                permutation_reps=args.permutation_reps,
                seed=seed,
                log=log,
            )
            control_results.append(result)
            payload["seed_results"].append(result)

        verdict, basis = classify_control(control, control_results)
        subgroup_verdict, subgroup_basis = assess_subgroup_disparity(control_results)
        completed = [r for r in control_results if r["attack_status"] == "completed"]
        payload["decisions"].append(
            {
                "control": control.name,
                "positive_control": control.positive_control,
                "seeds_attacked": len(completed),
                "mean_auc_gap": float(
                    np.mean([r["memorisation"]["auc_gap"] for r in control_results])
                ),
                "mean_lira_auc": (
                    float(np.mean([r["offline"]["aggregate"]["auc"] for r in completed]))
                    if completed
                    else None
                ),
                "mean_permutation_p_auc": (
                    float(
                        np.mean(
                            [
                                r["offline"]["permutation"]["aggregate_auc"]["p_value"]
                                for r in completed
                            ]
                        )
                    )
                    if completed
                    else None
                ),
                "verdict": verdict,
                "basis": basis,
                "subgroup_verdict": subgroup_verdict,
                "subgroup_basis": subgroup_basis,
            }
        )
        log(f"[{control.name}] verdict: {verdict} -- {basis}")
        log(f"[{control.name}] subgroup: {subgroup_verdict} -- {subgroup_basis}")

    json_path = output_dir / "attack_power_control.json"
    csv_path = output_dir / "attack_power_control.csv"
    md_path = output_dir / "attack_power_control.md"
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    pd.DataFrame(build_rows(payload)).to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(payload))

    log("\n" + md_path.read_text())
    log(f"JSON: {json_path}")
    log(f"CSV: {csv_path}")
    log(f"Markdown: {md_path}")
    log_handle.close()


if __name__ == "__main__":
    sys.exit(main())
