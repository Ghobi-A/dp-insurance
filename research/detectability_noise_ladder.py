"""Worst-case neural auditability ladder for the insurance ``high_cost`` task.

This is an **auditability ladder for a deliberately memorising neural target**,
not a universal DP-SGD detectability frontier. Its subject is one intentionally
attackable architecture; nothing here characterises DP-SGD in general, or any
other model family, or any other dataset.

The attack-power control (``research/attack_power_control.py``) established that
aggregate membership leakage *is* measurable on this dataset when a model
memorises: the deliberately over-parameterised MLP reached offline-LiRA AUC
0.558-0.586 with the permutation null rejected on all three seeds. What it did
not establish is where that detectability ends once DP-SGD noise is applied.

This script traces that frontier. It takes the same memorising architecture and
runs it at six privacy levels -- non-private, then epsilon = 32, 16, 8, 4, 2 --
changing **only the noise multiplier** between finite points. Cohort indices,
shadow inclusion schedules, shadow base seeds and model-initialisation seeds are
all derived from the target seed alone, never from the ladder point, so every
point is paired against every other.

Decision rules are pre-registered in ``docs/NOISE_LADDER_PREREGISTRATION.md``
and the adversary is fixed in ``docs/THREAT_MODEL.md``. Both were committed
before this experiment was run. The classification functions below implement
those rules mechanically.

Run one ladder point (matrix-friendly)::

    python research/detectability_noise_ladder.py point --epsilon 8 --seeds 42 43 44

Aggregate completed points into the final report::

    python research/detectability_noise_ladder.py aggregate --input-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The attack-power control is the tested implementation of every statistic used
# here. Reuse it rather than growing a second experiment framework.
from attack_power_control import (  # noqa: E402
    ATTACK_FPRS,
    HEADLINE_FPR,
    MEMORISATION_GATE,
    MIN_GAUSSIAN_OBSERVATIONS,
    attack_metrics,
    balanced_shadow_train_sets,
    build_mlp,
    make_equalised_attack_cohort,
    memorisation_gate_passed,
    memorisation_metrics,
    per_example_bce,
)

TASK_NAME = "high_cost"
SENSITIVE_ATTRIBUTE = "sex"
DELTA = 1e-5
ACCOUNTANT = "rdp"
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_SHADOWS = 32
DEFAULT_BOOTSTRAP_REPS = 1000
DEFAULT_PERMUTATION_REPS = 1000
ALPHA = 0.05

#: Pre-registered aggregate detection thresholds.
DETECT_MEAN_AUC = 0.55
MIN_SIGNIFICANT_SEEDS = 2

#: Fixed training recipe -- identical at every ladder point.
HIDDEN_UNITS = (128, 128, 64)
EPOCHS = 400
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0
TRAIN_SIZE = 856

#: Requested privacy points. ``None`` is the non-private control.
LADDER_EPSILONS: tuple[float | None, ...] = (None, 32.0, 16.0, 8.0, 4.0, 2.0)
NON_PRIVATE_LABEL = "non-private"

#: Amendment 1 classifications: detection and memorisation are two independent
#: binary decisions, and every point is labelled by their combination.
DETECTABLE_WITH_MEMORISATION = "DETECTABLE WITH MEMORISATION"
DETECTABLE_LOW_MEMORISATION = "DETECTABLE DESPITE LOW MEASURED MEMORISATION"
UNDETECTABLE_WITH_MEMORISATION = "UNDETECTABLE DESPITE MEMORISATION"
UNDETECTABLE_LOW_MEMORISATION = "UNDETECTABLE WITH LOW MEMORISATION"
DETECTABLE_STATUSES = (DETECTABLE_WITH_MEMORISATION, DETECTABLE_LOW_MEMORISATION)

SUBGROUP_SUPPORTED = "SUBGROUP DISPARITY SUPPORTED"
SUBGROUP_UNSUPPORTED = "SUBGROUP DISPARITY UNSUPPORTED"
SUBGROUP_INCONCLUSIVE = "SUBGROUP DISPARITY INCONCLUSIVE"


def point_label(epsilon: float | None) -> str:
    """Stable identifier for a ladder point, used in filenames and tables."""
    if epsilon is None:
        return NON_PRIVATE_LABEL
    return f"eps{epsilon:g}"


# --------------------------------------------------------------------------- #
# Privacy accounting
# --------------------------------------------------------------------------- #


def solve_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    epochs: int,
    delta: float = DELTA,
) -> float:
    """Noise multiplier reaching ``target_epsilon`` under RDP accounting.

    Requested epsilon is an *input*; the achieved epsilon read back from the
    accountant after training is the reported result.
    """
    from opacus.accountants.utils import get_noise_multiplier

    return float(
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant=ACCOUNTANT,
        )
    )


def epsilon_for_noise(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float = DELTA,
) -> float:
    """Achieved epsilon for a fixed noise multiplier, via the RDP accountant."""
    from dp.dpsgd import epsilon_for_noise_multiplier

    return float(
        epsilon_for_noise_multiplier(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=steps,
            target_delta=delta,
        )
    )


# --------------------------------------------------------------------------- #
# Training -- identical recipe at every point, only the noise changes
# --------------------------------------------------------------------------- #


def train_ladder_model(
    X: np.ndarray,
    y: np.ndarray,
    noise_multiplier: float | None,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[object, dict[str, float | int | str | None]]:
    """Train one target or shadow model at a given noise level.

    ``noise_multiplier=None`` trains the non-private control: no Opacus, no
    clipping, no noise. Everything else -- architecture, optimiser, epochs,
    batch size, initialisation seed and batch ordering seed -- is identical
    across all ladder points, so the noise multiplier is the only difference.
    """
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
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    model = build_mlp(X.shape[1], HIDDEN_UNITS, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    metadata: dict[str, float | int | str | None] = {
        "train_size": int(len(dataset)),
        "epochs": int(epochs),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": int(seed),
        "noise_multiplier": None,
        "achieved_epsilon": None,
        "max_grad_norm": None,
        "accountant": None,
        "sampling": None,
        "private": noise_multiplier is not None,
    }

    engine = None
    if noise_multiplier is not None:
        from opacus import PrivacyEngine

        engine = PrivacyEngine(accountant=ACCOUNTANT)
        model, optimizer, loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=MAX_GRAD_NORM,
            poisson_sampling=True,
        )
        metadata.update(
            {
                "noise_multiplier": float(noise_multiplier),
                "max_grad_norm": MAX_GRAD_NORM,
                "accountant": ACCOUNTANT,
                "sampling": "poisson",
            }
        )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()

    if engine is not None:
        metadata["achieved_epsilon"] = float(engine.get_epsilon(DELTA))
    return model, metadata


def predict_probability(model, X: np.ndarray) -> np.ndarray:
    """Positive-class probabilities from a (possibly Opacus-wrapped) model."""
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


# --------------------------------------------------------------------------- #
# Multiplicity adjustment
# --------------------------------------------------------------------------- #


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment.

    Sorts ascending, scales the i-th smallest by ``m - i``, enforces monotone
    non-decreasing adjusted values, clips at 1, and restores the input order.
    NaNs are passed through untouched so a missing point cannot silently
    inflate or deflate its neighbours.
    """
    values = np.asarray(list(p_values), dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(values))
    adjusted = np.full(values.shape, np.nan, dtype=float)
    if finite_idx.size == 0:
        return adjusted.tolist()

    finite = values[finite_idx]
    order = np.argsort(finite, kind="stable")
    m = finite.size
    running = 0.0
    scaled = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        candidate = (m - rank) * finite[idx]
        running = max(running, candidate)
        scaled[idx] = min(1.0, running)
    adjusted[finite_idx] = scaled
    return adjusted.tolist()


# --------------------------------------------------------------------------- #
# Pre-registered classification
# --------------------------------------------------------------------------- #


def memorisation_present(seed_results: Sequence[dict[str, object]]) -> bool:
    """Whether the target memorised, as metadata rather than a gate.

    A point counts as memorising when the pre-registered criterion holds on
    every seed. Unchanged numerically from the original registration; only its
    role changed under Amendment 1.
    """
    if not seed_results:
        return False
    return all(bool(r["memorisation_gate_passed"]) for r in seed_results)


def aggregate_detected(seed_results: Sequence[dict[str, object]]) -> tuple[bool, str]:
    """Apply the pre-registered aggregate detection rule.

    Thresholds are numerically identical to the original registration: mean
    offline AUC >= 0.55, AUC > 0.5 on every seed, and the Holm-adjusted
    permutation null rejected on at least two of three seeds.
    """
    completed = [r for r in seed_results if r["attack_status"] == "completed"]
    if not completed:
        return False, "no attack completed"

    aucs = np.asarray(
        [float(r["offline"]["aggregate"]["auc"]) for r in completed], dtype=float
    )
    adjusted = np.asarray(
        [
            float(r["offline"]["permutation"]["aggregate_auc"].get("p_value_holm", np.nan))
            for r in completed
        ],
        dtype=float,
    )
    mean_auc = float(np.nanmean(aucs))
    significant = int(np.sum(adjusted < ALPHA))
    above_chance = bool(np.all(aucs > 0.5))
    # "2 of 3" at the pre-registered three seeds; reduced-seed smoke runs would
    # otherwise be unsatisfiable.
    required = min(MIN_SIGNIFICANT_SEEDS, len(completed))

    if mean_auc >= DETECT_MEAN_AUC and above_chance and significant >= required:
        return True, (
            f"mean offline-LiRA AUC {mean_auc:.4f} (>= {DETECT_MEAN_AUC}), above 0.5 on "
            f"every seed, Holm-adjusted permutation null rejected on "
            f"{significant}/{len(completed)} seeds (>= {required} required)"
        )

    reasons = []
    if mean_auc < DETECT_MEAN_AUC:
        reasons.append(f"mean AUC {mean_auc:.4f} < {DETECT_MEAN_AUC}")
    if not above_chance:
        reasons.append("at least one seed at or below chance")
    if significant < required:
        reasons.append(
            f"Holm-adjusted rejection on only {significant}/{len(completed)} seeds "
            f"({required} required)"
        )
    return False, "; ".join(reasons)


def classify_ladder_point(seed_results: Sequence[dict[str, object]]) -> tuple[str, str]:
    """Classify one ladder point by combining detection and memorisation.

    Under Amendment 1 these are two independent binary decisions, and every
    point receives one of four statuses. No point is ever labelled private,
    safe or verified: an undetected point means *not detected at this attack
    power and cohort size*.
    """
    detected, detection_basis = aggregate_detected(seed_results)
    memorised = memorisation_present(seed_results)

    if detected and memorised:
        status = DETECTABLE_WITH_MEMORISATION
        basis = f"Detected ({detection_basis}); target memorised on every seed."
    elif detected and not memorised:
        status = DETECTABLE_LOW_MEMORISATION
        basis = (
            f"Detected ({detection_basis}) even though measured memorisation is below "
            f"the {MEMORISATION_GATE} criterion on at least one seed."
        )
    elif memorised:
        status = UNDETECTABLE_WITH_MEMORISATION
        basis = (
            f"Target memorised on every seed, yet not detected: {detection_basis}. "
            "Not detected at this attack power -- not a privacy claim."
        )
    else:
        status = UNDETECTABLE_LOW_MEMORISATION
        basis = (
            f"Not detected ({detection_basis}) and measured memorisation is below the "
            f"{MEMORISATION_GATE} criterion on at least one seed."
        )
    return status, basis


def classify_subgroup(seed_results: Sequence[dict[str, object]]) -> tuple[str, str]:
    """Apply the pre-registered signed-contrast subgroup rule."""
    completed = [r for r in seed_results if r["attack_status"] == "completed"]
    if not completed:
        return SUBGROUP_INCONCLUSIVE, "No attack completed, so no subgroup comparison."

    signed = np.asarray(
        [float(r["offline"]["signed_subgroup_difference"]) for r in completed], dtype=float
    )
    signed_p = np.asarray(
        [
            float(
                r["offline"]["permutation"]["signed_difference"].get(
                    "p_value_holm", np.nan
                )
            )
            for r in completed
        ],
        dtype=float,
    )
    gap_p = np.asarray(
        [
            float(r["offline"]["permutation"]["absolute_gap"].get("p_value_holm", np.nan))
            for r in completed
        ],
        dtype=float,
    )
    resolutions = [float(r["subgroup_resolution"]) for r in completed]
    resolution = float(max(resolutions)) if resolutions else float("inf")

    nonzero = signed[signed != 0.0]
    stable = bool(nonzero.size and np.all(np.sign(nonzero) == np.sign(nonzero[0])))
    significant = int(np.sum((signed_p < ALPHA) | (gap_p < ALPHA)))
    resolvable = bool(signed.size and np.all(np.abs(signed) >= resolution))
    # The subgroup rule is NOT relaxed for reduced-seed runs: pre-registered
    # criterion 4 forbids single-seed effects outright, so a one-seed run can
    # never support a disparity claim.
    required = MIN_SIGNIFICANT_SEEDS
    carrying = int(np.sum(np.abs(signed) >= resolution))
    multi_seed = carrying >= required

    if stable and significant >= required and resolvable and multi_seed:
        direction = "male" if float(np.mean(signed)) > 0 else "female"
        return (
            SUBGROUP_SUPPORTED,
            f"Signed male-female TPR@1% contrast keeps one sign across seeds "
            f"({direction} more exposed), adjusted permutation p<{ALPHA} on "
            f"{significant}/{len(completed)} seeds, and every effect exceeds the "
            f"{resolution:.4f} one-person resolution.",
        )

    reasons = []
    if not stable:
        reasons.append("the signed direction is not stable across non-zero seeds")
    if significant < required:
        reasons.append(
            f"the adjusted permutation null is rejected on only "
            f"{significant}/{len(completed)} seeds ({required} required)"
        )
    if not resolvable:
        reasons.append(
            f"at least one contrast is within the {resolution:.4f} one-person resolution"
        )
    elif not multi_seed:
        reasons.append("the effect is carried by a single seed")
    return SUBGROUP_UNSUPPORTED, "; ".join(reasons).capitalize() + "."


def detectability_frontier(points: Sequence[dict[str, object]]) -> dict[str, object]:
    """Bracket the transition, or report the pattern as unstable.

    Points are ordered by descending achieved epsilon (non-private first, as the
    least noisy). Detection is expected to be monotone -- detectable at low
    noise, lost at high noise. Any detectable point appearing *below* an
    undetectable one breaks that, and is reported as non-monotonic rather than
    smoothed into a single threshold.
    """
    ordered = sorted(
        points,
        key=lambda p: (
            float("inf")
            if p["requested_epsilon"] is None
            else float(p["requested_epsilon"])
        ),
        reverse=True,
    )
    # Amendment 1: every point is attacked, so every point is classifiable.
    classified = list(ordered)
    flags = [p["decision"] in DETECTABLE_STATUSES for p in classified]

    # Monotone means: every detectable point precedes every undetectable one.
    non_monotonic = any(
        flags[i] and not flags[i - 1] for i in range(1, len(flags))
    )

    result: dict[str, object] = {
        "ordered_points": [
            {
                "label": p["label"],
                "requested_epsilon": p["requested_epsilon"],
                "achieved_epsilon": p.get("achieved_epsilon"),
                "noise_multiplier": p.get("noise_multiplier"),
                "decision": p["decision"],
            }
            for p in ordered
        ],
        "non_monotonic": bool(non_monotonic),
        "last_detectable": None,
        "first_undetectable": None,
        "bracket": None,
        "stable": not non_monotonic,
    }
    if non_monotonic:
        result["summary"] = (
            "Detection is NON-MONOTONIC across the ladder: a lower-epsilon point is "
            "detectable while a higher-epsilon point is not. No single threshold is "
            "reported; the frontier is classified as UNSTABLE."
        )
        return result

    detectable = [p for p, f in zip(classified, flags) if f]
    undetectable = [p for p, f in zip(classified, flags) if not f]
    if detectable:
        result["last_detectable"] = detectable[-1]["label"]
    if undetectable:
        result["first_undetectable"] = undetectable[0]["label"]

    if detectable and undetectable:
        low = detectable[-1]
        high = undetectable[0]
        low_eps = low.get("achieved_epsilon") or low["requested_epsilon"]
        high_eps = high.get("achieved_epsilon") or high["requested_epsilon"]
        result["bracket"] = {
            "detectable_at": low["label"],
            "detectable_epsilon": low_eps,
            "not_detectable_at": high["label"],
            "not_detectable_epsilon": high_eps,
        }
        result["summary"] = (
            f"detectable at epsilon = {low_eps if low_eps is None else f'{float(low_eps):.4f}'} "
            f"({low['label']}); not detectable at epsilon = "
            f"{high_eps if high_eps is None else f'{float(high_eps):.4f}'} ({high['label']}); "
            "therefore the empirical transition lies between them. Six points do not "
            "support interpolating a precise threshold."
        )
    elif detectable:
        finite_classified = [
            p for p in classified if p["requested_epsilon"] is not None
        ]
        if not finite_classified:
            result["summary"] = (
                "Only the non-private point could be classified; every finite privacy "
                "point failed the memorisation gate, so the frontier cannot be "
                "bracketed. This is a statement about the targets, not about privacy."
            )
        else:
            result["summary"] = (
                "Leakage remains detectable at every classified ladder point, including "
                "the noisiest tested. The transition lies below the lowest epsilon "
                "tested."
            )
    else:
        result["summary"] = (
            "No classified ladder point shows detectable leakage, including the "
            "non-private control. The frontier cannot be bracketed; this indicates an "
            "attack-power problem rather than a privacy result."
        )
    return result



# --------------------------------------------------------------------------- #
# Generalisation-gap association analysis
# --------------------------------------------------------------------------- #

#: Verbatim caveat required on every report carrying this analysis.
ASSOCIATION_ONLY_STATEMENT = (
    "These analyses measure association only. DP noise changes both "
    "generalisation and leakage, so this experiment does not identify an effect "
    "of DP on membership leakage independently of its effect on memorisation."
)

GAP_METRICS = ("bce_gap", "auc_gap", "accuracy_gap")
LEAKAGE_METRICS = ("lira_auc", "tpr_at_1pct")


def collect_observations(summaries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """One record per seed-level observation, including low-memorisation points.

    Every observation is retained. Under Amendment 1 there are no skipped
    points, and none may be filtered out here either -- dropping the
    low-memorisation end is exactly the truncation the amendment exists to
    prevent.
    """
    observations: list[dict[str, object]] = []
    for summary in summaries:
        for result in summary["seed_results"]:
            metrics = result.get("offline")
            memo = result["memorisation"]
            observations.append(
                {
                    "label": result["label"],
                    "seed": int(result["seed"]),
                    "requested_epsilon": result["requested_epsilon"],
                    "achieved_epsilon": result.get("achieved_epsilon"),
                    "noise_multiplier": result.get("noise_multiplier"),
                    "private": result["requested_epsilon"] is not None,
                    "train_auc": memo["train_auc"],
                    "test_auc": memo["test_auc"],
                    "train_bce": memo["train_loss"],
                    "test_bce": memo["test_loss"],
                    "train_accuracy": memo["train_accuracy"],
                    "test_accuracy": memo["test_accuracy"],
                    "auc_gap": memo["auc_gap"],
                    "bce_gap": memo["loss_gap"],
                    "accuracy_gap": memo["accuracy_gap"],
                    "memorisation_threshold_passed": bool(result["memorisation_gate_passed"]),
                    "lira_auc": (
                        float(metrics["aggregate"]["auc"]) if metrics else float("nan")
                    ),
                    "tpr_at_1pct": (
                        float(metrics["aggregate"][f"tpr_at_fpr_{HEADLINE_FPR}"])
                        if metrics
                        else float("nan")
                    ),
                }
            )
    return observations


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, NaN when it is undefined."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    from scipy import stats

    xr = stats.rankdata(x[mask])
    yr = stats.rankdata(y[mask])
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def gap_associations(
    observations: Sequence[dict[str, object]],
    reps: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Spearman associations with seed-stratified bootstrap intervals.

    Resampling is stratified by target seed, so each replicate keeps the same
    number of observations per seed and the interval reflects variation across
    ladder points rather than across an arbitrary pooled sample.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted({int(o["seed"]) for o in observations})
    by_seed = {s: [o for o in observations if int(o["seed"]) == s] for s in seeds}

    results: dict[str, dict[str, float]] = {}
    for gap in GAP_METRICS:
        for leak in LEAKAGE_METRICS:
            x = np.asarray([float(o[gap]) for o in observations], dtype=float)
            y = np.asarray([float(o[leak]) for o in observations], dtype=float)
            observed = _spearman(x, y)

            draws = np.empty(reps, dtype=float)
            for rep in range(reps):
                sampled: list[dict[str, object]] = []
                for s in seeds:
                    pool = by_seed[s]
                    idx = rng.integers(0, len(pool), size=len(pool))
                    sampled.extend(pool[i] for i in idx)
                draws[rep] = _spearman(
                    np.asarray([float(o[gap]) for o in sampled], dtype=float),
                    np.asarray([float(o[leak]) for o in sampled], dtype=float),
                )
            finite = draws[np.isfinite(draws)]
            low, high = (
                np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
            )
            results[f"{gap}__{leak}"] = {
                "gap_metric": gap,
                "leakage_metric": leak,
                "n_observations": int(np.sum(np.isfinite(x) & np.isfinite(y))),
                "spearman": observed,
                "reps": int(reps),
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
    return results


# --------------------------------------------------------------------------- #
# One ladder point
# --------------------------------------------------------------------------- #


def prepare_seed_arrays(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Leakage-safe split and preprocessing for one target seed.

    Depends only on the seed, so every ladder point sees identical partitions.
    """
    from dp.pipeline import build_preprocessor
    from dp.tasks import get_task, prepare_task_data

    data = prepare_task_data(df, get_task(TASK_NAME), random_state=seed)
    preprocessor = build_preprocessor(data.X_train)
    return {
        "X_train": np.asarray(preprocessor.fit_transform(data.X_train), dtype=np.float32),
        "X_val": np.asarray(preprocessor.transform(data.X_val), dtype=np.float32),
        "X_test": np.asarray(preprocessor.transform(data.X_test), dtype=np.float32),
        "y_train": data.y_train,
        "y_val": data.y_val,
        "y_test": data.y_test,
        "groups_train": data.sensitive_train,
        "groups_test": data.sensitive_test,
    }


def run_seed(
    arrays: dict[str, np.ndarray],
    requested_epsilon: float | None,
    seed: int,
    num_shadows: int,
    bootstrap_reps: int,
    permutation_reps: int,
    epochs_override: int | None,
    smoke: bool,
    log,
) -> dict[str, object]:
    """Train one target plus its shadows at one ladder point, then attack."""
    from dp.attacks import ONLINE_VARIANTS, lira_offline_attack, lira_online_attack

    epochs = EPOCHS if epochs_override is None else int(epochs_override)
    # Smoke mode exists so CI can exercise the whole code path in minutes. It
    # lowers the Gaussian floor, which the pre-registered configuration never
    # does, so every artifact it produces is stamped `smoke: true` and the
    # report refuses to present it as a pre-registered result.
    min_observations = 2 if smoke else MIN_GAUSSIAN_OBSERVATIONS

    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    X_attack = np.vstack([X_train, X_test])
    y_attack = np.concatenate([y_train, y_test])
    groups_attack = np.concatenate([arrays["groups_train"], arrays["groups_test"]]).astype(str)
    membership = np.concatenate(
        [np.ones(len(y_train), dtype=int), np.zeros(len(y_test), dtype=int)]
    )
    # Cohort seeded by target seed only -> identical at every ladder point.
    attack_idx, cohort_counts = make_equalised_attack_cohort(groups_attack, membership, seed)

    sample_rate = min(BATCH_SIZE, len(y_train)) / len(y_train)
    steps = epochs * int(np.ceil(len(y_train) / min(BATCH_SIZE, len(y_train))))
    noise_multiplier = None
    if requested_epsilon is not None:
        noise_multiplier = solve_noise_multiplier(
            requested_epsilon, sample_rate, epochs, DELTA
        )

    target, target_meta = train_ladder_model(
        X_train, y_train, noise_multiplier, seed=seed, epochs=epochs
    )
    if int(target_meta["train_size"]) != len(y_train):
        raise AssertionError("target train size does not match the training partition")

    metrics = memorisation_metrics(
        y_train,
        predict_probability(target, X_train),
        y_test,
        predict_probability(target, X_test),
    )
    gate_passed = memorisation_gate_passed(metrics)

    result: dict[str, object] = {
        "label": point_label(requested_epsilon),
        "requested_epsilon": requested_epsilon,
        "seed": int(seed),
        "delta": DELTA,
        "accountant": ACCOUNTANT if requested_epsilon is not None else None,
        "sampling": "poisson" if requested_epsilon is not None else None,
        "noise_multiplier": noise_multiplier,
        "achieved_epsilon": target_meta["achieved_epsilon"],
        "accountant_epsilon": (
            epsilon_for_noise(noise_multiplier, sample_rate, steps, DELTA)
            if noise_multiplier is not None
            else None
        ),
        "max_grad_norm": MAX_GRAD_NORM if requested_epsilon is not None else None,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "architecture": f"Linear(d,{'->'.join(map(str, HIDDEN_UNITS))}->1) + ReLU",
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "cohort_counts": cohort_counts,
        "num_attack_examples": int(len(attack_idx)),
        "cohort_index_digest": int(np.sum(attack_idx.astype(np.int64) * 2654435761) % (2**31)),
        "memorisation": metrics,
        "memorisation_gate": MEMORISATION_GATE,
        "memorisation_gate_passed": bool(gate_passed),
        "smoke": bool(smoke),
        "attack_status": "pending",
        "offline": None,
        "online": None,
        "online_raw_loss": None,
        "online_logit_confidence": None,
        "online_available": False,
    }

    log(
        f"[{result['label']} seed={seed}] noise={noise_multiplier} "
        f"achieved_eps={target_meta['achieved_epsilon']} "
        f"train_auc={metrics['train_auc']:.4f} test_auc={metrics['test_auc']:.4f} "
        f"auc_gap={metrics['auc_gap']:.4f} loss_gap={metrics['loss_gap']:.4f} "
        f"memorisation={'present' if gate_passed else 'low'} (metadata only)"
    )
    # Amendment 1: the attack runs at every point and seed. Memorisation is
    # recorded as explanatory metadata, never as a condition on measurement --
    # discarding low-memorisation observations would truncate the sample on a
    # variable correlated with the outcome.

    X_pool = np.vstack([X_train, X_val, X_test])
    y_pool = np.concatenate([y_train, y_val, y_test])
    attack_pool_idx = attack_idx.copy()
    attack_pool_idx[attack_pool_idx >= len(y_train)] += len(y_val)

    # Schedule seeded by target seed only -> identical at every ladder point.
    schedule_seed = seed + 10_000
    train_sets, excluded_counts = balanced_shadow_train_sets(
        y_pool, train_size=len(y_train), num_shadows=num_shadows, seed=schedule_seed
    )
    result["shadow_schedule_digest"] = int(
        np.sum(np.concatenate(train_sets).astype(np.int64) * 2654435761) % (2**31)
    )

    target_losses = per_example_bce(
        y_attack[attack_idx], predict_probability(target, X_attack)[attack_idx]
    )
    out_losses: list[list[float]] = [[] for _ in attack_pool_idx]
    in_losses: list[list[float]] = [[] for _ in attack_pool_idx]

    for shadow_id, train_idx in enumerate(train_sets):
        if len(train_idx) != len(y_train):
            raise AssertionError("shadow and target train sizes differ")
        # Shadow base seed depends on target seed and shadow id only.
        shadow_model, shadow_meta = train_ladder_model(
            X_pool[train_idx],
            y_pool[train_idx],
            noise_multiplier,
            seed=schedule_seed + shadow_id + 1,
            epochs=epochs,
        )
        if int(shadow_meta["train_size"]) != int(target_meta["train_size"]):
            raise AssertionError("shadow and target train sizes differ")
        if noise_multiplier is not None:
            if not np.isclose(
                float(shadow_meta["noise_multiplier"]), float(noise_multiplier), atol=1e-12
            ):
                raise AssertionError("shadow noise multiplier differs from target")
            if not np.isclose(
                float(shadow_meta["achieved_epsilon"]),
                float(target_meta["achieved_epsilon"]),
                rtol=1e-6,
            ):
                raise AssertionError(
                    "shadow achieved epsilon differs from target beyond accounting tolerance"
                )

        in_mask = np.zeros(len(y_pool), dtype=bool)
        in_mask[train_idx] = True
        losses = per_example_bce(
            y_pool[attack_pool_idx], predict_probability(shadow_model, X_pool[attack_pool_idx])
        )
        for local_idx, is_in in enumerate(in_mask[attack_pool_idx]):
            (in_losses if is_in else out_losses)[local_idx].append(float(losses[local_idx]))

        if (shadow_id + 1) % 8 == 0 or shadow_id == 0:
            log(f"[{result['label']} seed={seed}] shadow {shadow_id + 1}/{num_shadows}")

    min_out = min(len(v) for v in out_losses)
    min_in = min(len(v) for v in in_losses)
    if min_out != int(excluded_counts[attack_pool_idx].min()):
        raise AssertionError("recorded OUT losses do not match the schedule")
    if min_out < min_observations:
        raise RuntimeError(
            f"offline LiRA needs >= {min_observations} OUT observations; got {min_out}"
        )

    out_matrix = np.vstack([np.asarray(v[:min_out], dtype=float) for v in out_losses])
    attack_groups = groups_attack[attack_idx]
    attack_membership = membership[attack_idx]
    member_counts = {
        group: int(((attack_groups == group) & (attack_membership == 1)).sum())
        for group in np.unique(attack_groups)
    }
    resolution = 1.0 / max(min(member_counts.values()), 1)

    offline = lira_offline_attack(
        target_losses, attack_membership, out_matrix, fprs=ATTACK_FPRS
    )
    # Per-example records are the raw design. They are what makes a later
    # *paired* between-recipe comparison possible at all: the same cohort rows,
    # in the same order, at every ladder point for a given target seed. Summary
    # statistics alone would discard that pairing irrecoverably.
    result["per_example"] = [
        {
            "label": result["label"],
            "requested_epsilon": requested_epsilon,
            "seed": int(seed),
            "cohort_position": int(position),
            "example_index": int(example_index),
            "group": str(attack_groups[position]),
            "membership": int(attack_membership[position]),
            "target_loss": float(target_losses[position]),
            "offline_score": float(offline.scores[position]),
            "out_observations": int(len(out_losses[position])),
            "in_observations": int(len(in_losses[position])),
        }
        for position, example_index in enumerate(attack_idx)
    ]
    result.update(
        {
            "attack_status": "completed",
            "num_shadows": int(num_shadows),
            "min_out_observations": int(min_out),
            "min_in_observations": int(min_in),
            "subgroup_member_counts": member_counts,
            "subgroup_resolution": resolution,
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

    if min_in >= min_observations:
        in_matrix = np.vstack([np.asarray(v[:min_in], dtype=float) for v in in_losses])
        result["online_available"] = True
        for offset, variant in enumerate(ONLINE_VARIANTS):
            online = lira_online_attack(
                target_losses,
                attack_membership,
                in_matrix,
                out_matrix,
                fprs=ATTACK_FPRS,
                variant=variant,
            )
            result[variant] = attack_metrics(
                attack_membership,
                attack_groups,
                online.scores,
                bootstrap_reps,
                seed + 30_000 + 1_000 * offset,
                permutation_reps,
            )
            if variant == "online_raw_loss":
                for record, score in zip(result["per_example"], online.scores):
                    record["online_raw_loss_score"] = float(score)
        # Back-compatible alias: "online" is the raw-loss variant.
        result["online"] = result["online_raw_loss"]
    else:
        result["online_unavailable_reason"] = (
            f"only {min_in} IN observations per example; {min_observations} required"
        )

    log(
        f"[{result['label']} seed={seed}] offline AUC="
        f"{result['offline']['aggregate']['auc']:.4f} "
        f"TPR@1%={result['offline']['aggregate'][f'tpr_at_fpr_{HEADLINE_FPR}']:.4f} "
        f"perm_p={result['offline']['permutation']['aggregate_auc']['p_value']:.4f}"
    )
    return result


def run_point(args: argparse.Namespace) -> None:
    """Run every seed for a single ladder point and write its artifact."""
    import torch

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = None if args.epsilon is None or args.epsilon <= 0 else float(args.epsilon)
    label = point_label(requested)
    log_path = output_dir / f"{label}.log"
    handle = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        handle.write(message + "\n")
        handle.flush()

    log(f"=== ladder point {label} (requested epsilon={requested}) ===")
    df = pd.read_csv(root / "data" / "insurance.csv")
    seed_results = []
    for seed in args.seeds:
        arrays = prepare_seed_arrays(df, seed)
        seed_results.append(
            run_seed(
                arrays,
                requested,
                seed,
                num_shadows=args.shadows,
                bootstrap_reps=args.bootstrap_reps,
                permutation_reps=args.permutation_reps,
                epochs_override=args.epochs,
                smoke=bool(args.smoke),
                log=log,
            )
        )

    payload = {
        "experiment": "detectability_noise_ladder_point",
        "label": label,
        "requested_epsilon": requested,
        "task": TASK_NAME,
        "sensitive_attribute": SENSITIVE_ATTRIBUTE,
        "delta": DELTA,
        "seeds": [int(s) for s in args.seeds],
        "num_shadows": int(args.shadows),
        "bootstrap_reps": int(args.bootstrap_reps),
        "permutation_reps": int(args.permutation_reps),
        "epochs": int(args.epochs) if args.epochs is not None else EPOCHS,
        "smoke": bool(args.smoke),
        "seed_results": seed_results,
    }
    out_path = output_dir / f"{label}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    log(f"Wrote {out_path}")
    handle.close()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def apply_holm_across_points(points: list[dict[str, object]]) -> None:
    """Holm-adjust permutation p-values across ladder points, within each seed.

    Each (seed, attack variant, statistic) family is adjusted over the ladder
    points, which is the multiplicity actually introduced by the ladder.
    """
    statistics = ("aggregate_auc", "aggregate_tpr", "signed_difference", "absolute_gap")
    seeds = sorted({int(r["seed"]) for p in points for r in p["seed_results"]})
    for seed in seeds:
        for variant in ("offline", "online"):
            for statistic in statistics:
                entries = []
                for point in points:
                    for result in point["seed_results"]:
                        if int(result["seed"]) != seed:
                            continue
                        metrics = result.get(variant)
                        if not metrics:
                            continue
                        entry = metrics["permutation"].get(statistic)
                        if entry is not None:
                            entries.append(entry)
                if not entries:
                    continue
                adjusted = holm_adjust([float(e["p_value"]) for e in entries])
                for entry, value in zip(entries, adjusted):
                    entry["p_value_holm"] = float(value)


def verify_pairing(points: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Check that every ladder point shares one cohort and schedule per seed.

    The pre-registration asserts that cohort indices and shadow inclusion
    schedules derive from the target seed alone. That assertion is only worth
    anything if it is checked against the artifacts that were actually produced,
    so aggregation refuses to proceed when two points disagree for a seed.

    Raises:
        ValueError: if any seed carries more than one cohort digest, schedule
            digest, cohort size or training size across the supplied points.
    """
    fields = (
        "cohort_index_digest",
        "shadow_schedule_digest",
        "num_attack_examples",
        "train_size",
    )
    seen: dict[int, dict[str, dict[object, list[str]]]] = {}
    for point in points:
        for result in point["seed_results"]:
            seed = int(result["seed"])
            per_seed = seen.setdefault(seed, {field: {} for field in fields})
            missing = [field for field in fields if result.get(field) is None]
            if missing:
                raise ValueError(
                    f"ladder point {result['label']!r} seed {seed} is missing "
                    f"{', '.join(missing)}, so its pairing cannot be verified"
                )
            for field in fields:
                value = result.get(field)
                per_seed[field].setdefault(value, []).append(str(result["label"]))

    problems: list[str] = []
    for seed in sorted(seen):
        for field in fields:
            values = seen[seed][field]
            if len(values) > 1:
                detail = "; ".join(
                    f"{value} at {', '.join(labels)}" for value, labels in values.items()
                )
                problems.append(f"seed {seed}: {field} differs across points -- {detail}")
    if problems:
        raise ValueError(
            "Ladder points are not paired, so no cross-point comparison is valid:\n  "
            + "\n  ".join(problems)
        )
    return {
        str(seed): {field: next(iter(seen[seed][field])) for field in fields}
        for seed in sorted(seen)
    }


def summarise_points(points: list[dict[str, object]]) -> list[dict[str, object]]:
    """Per-point summary plus the pre-registered decisions."""
    summaries = []
    for point in points:
        seed_results = point["seed_results"]
        completed = [r for r in seed_results if r["attack_status"] == "completed"]
        decision, basis = classify_ladder_point(seed_results)
        subgroup_verdict, subgroup_basis = classify_subgroup(seed_results)
        achieved = [
            float(r["achieved_epsilon"])
            for r in seed_results
            if r.get("achieved_epsilon") is not None
        ]
        noises = [
            float(r["noise_multiplier"])
            for r in seed_results
            if r.get("noise_multiplier") is not None
        ]
        summaries.append(
            {
                "label": point["label"],
                "requested_epsilon": point["requested_epsilon"],
                "achieved_epsilon": float(np.mean(achieved)) if achieved else None,
                "noise_multiplier": float(np.mean(noises)) if noises else None,
                "seeds": [int(r["seed"]) for r in seed_results],
                "seeds_attacked": len(completed),
                "gate_passed_seeds": int(
                    sum(1 for r in seed_results if r["memorisation_gate_passed"])
                ),
                "mean_auc_gap": float(
                    np.mean([r["memorisation"]["auc_gap"] for r in seed_results])
                ),
                "mean_loss_gap": float(
                    np.mean([r["memorisation"]["loss_gap"] for r in seed_results])
                ),
                "mean_offline_auc": (
                    float(np.mean([r["offline"]["aggregate"]["auc"] for r in completed]))
                    if completed
                    else None
                ),
                "mean_offline_tpr_1pct": (
                    float(
                        np.mean(
                            [
                                r["offline"]["aggregate"][f"tpr_at_fpr_{HEADLINE_FPR}"]
                                for r in completed
                            ]
                        )
                    )
                    if completed
                    else None
                ),
                "min_holm_p": (
                    float(
                        np.nanmin(
                            [
                                r["offline"]["permutation"]["aggregate_auc"].get(
                                    "p_value_holm", np.nan
                                )
                                for r in completed
                            ]
                        )
                    )
                    if completed
                    else None
                ),
                "decision": decision,
                "basis": basis,
                "subgroup_verdict": subgroup_verdict,
                "subgroup_basis": subgroup_basis,
                "seed_results": seed_results,
            }
        )
    return summaries


def build_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    """One CSV row per ladder point, seed and attack variant."""
    rows = []
    for summary in summaries:
        for result in summary["seed_results"]:
            variants = [v for v in ("offline", "online") if result.get(v)] or [None]
            for variant in variants:
                row: dict[str, object] = {
                    "label": result["label"],
                    "requested_epsilon": result["requested_epsilon"],
                    "achieved_epsilon": result.get("achieved_epsilon"),
                    "noise_multiplier": result.get("noise_multiplier"),
                    "seed": result["seed"],
                    "attack": f"lira-{variant}" if variant else "none",
                    "attack_status": result["attack_status"],
                    "decision": summary["decision"],
                    "memorisation_gate_passed": result["memorisation_gate_passed"],
                    "train_size": result["train_size"],
                    "train_auc": result["memorisation"]["train_auc"],
                    "test_auc": result["memorisation"]["test_auc"],
                    "auc_gap": result["memorisation"]["auc_gap"],
                    "loss_gap": result["memorisation"]["loss_gap"],
                    "accuracy_gap": result["memorisation"]["accuracy_gap"],
                    "min_out_observations": result.get("min_out_observations"),
                    "min_in_observations": result.get("min_in_observations"),
                    "subgroup_resolution": result.get("subgroup_resolution"),
                }
                if variant:
                    metrics = result[variant]
                    agg = metrics["aggregate"]
                    perm = metrics["permutation"]
                    boot = metrics["bootstrap"]
                    row.update(
                        {
                            "lira_auc": agg["auc"],
                            "tpr_at_fpr_0.001": agg["tpr_at_fpr_0.001"],
                            "tpr_at_fpr_0.01": agg["tpr_at_fpr_0.01"],
                            "tpr_at_fpr_0.1": agg["tpr_at_fpr_0.1"],
                            "detected_members_at_1pct": agg["operating_point"][
                                "detected_members"
                            ],
                            "false_positives_at_1pct": agg["operating_point"][
                                "false_positives"
                            ],
                            "permutation_null_auc_mean": perm["aggregate_auc"]["null_mean"],
                            "permutation_null_auc_ci_low": perm["aggregate_auc"][
                                "null_ci95_low"
                            ],
                            "permutation_null_auc_ci_high": perm["aggregate_auc"][
                                "null_ci95_high"
                            ],
                            "permutation_p_auc": perm["aggregate_auc"]["p_value"],
                            "permutation_p_auc_holm": perm["aggregate_auc"].get(
                                "p_value_holm"
                            ),
                            "permutation_p_tpr": perm["aggregate_tpr"]["p_value"],
                            "permutation_p_tpr_holm": perm["aggregate_tpr"].get(
                                "p_value_holm"
                            ),
                            "bootstrap_tpr_ci_low": boot["aggregate"]["ci95_low"],
                            "bootstrap_tpr_ci_high": boot["aggregate"]["ci95_high"],
                            "signed_gap": metrics["signed_subgroup_difference"],
                            "absolute_gap": metrics["absolute_subgroup_gap"],
                            "permutation_p_signed_holm": perm["signed_difference"].get(
                                "p_value_holm"
                            ),
                            "permutation_p_absolute_holm": perm["absolute_gap"].get(
                                "p_value_holm"
                            ),
                        }
                    )
                    for group, gm in metrics["subgroups"].items():
                        row[f"{group}_auc"] = gm["auc"]
                        row[f"{group}_tpr_at_fpr_0.01"] = gm["tpr_at_fpr_0.01"]
                        row[f"{group}_detected_members"] = gm["operating_point"][
                            "detected_members"
                        ]
                        row[f"{group}_false_positives"] = gm["operating_point"][
                            "false_positives"
                        ]
                        row[f"{group}_tpr_ci_low"] = boot[group]["ci95_low"]
                        row[f"{group}_tpr_ci_high"] = boot[group]["ci95_high"]
                        row[f"{group}_permutation_p"] = perm[group]["p_value"]
                rows.append(row)
    return rows


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "-"


def render_markdown(payload: dict[str, object]) -> str:
    """Render the pre-registered report."""
    summaries: list[dict[str, object]] = payload["points"]
    frontier: dict[str, object] = payload["frontier"]
    lines = ["# Worst-case neural auditability ladder", ""]
    if payload.get("smoke"):
        lines += [
            "> **SMOKE RUN -- NOT A PRE-REGISTERED RESULT.** Produced with a lowered "
            "Gaussian observation floor and reduced budgets to exercise the pipeline. "
            "No number below may be cited as a finding.",
            "",
        ]
    lines += [
        "",
        "**Auditability ladder for a deliberately memorising neural target.** This is "
        "not the universal DP-SGD detectability frontier: it characterises one "
        "intentionally attackable architecture on one dataset, chosen because the "
        "attack-power control showed it is attackable at all.",
        "",
        "Context from the attack-power control that motivates this design:",
        "",
        "| Control | Outcome |",
        "|---|---|",
        "| Matched-capacity MLP (mirrors the DP target recipe) | attack inconclusive -- "
        "generalised rather than memorised |",
        "| Deliberately memorising MLP | aggregate leakage detected |",
        "| This ladder | effect of DP noise on an intentionally attackable neural "
        "target |",
        "",
        "## 1. Pre-registered question and rules",
        "",
        "> At what privacy-noise level does aggregate natural membership leakage cease "
        "to be reproducibly detectable on the insurance `high_cost` task?",
        "",
        "Rules were fixed in `docs/NOISE_LADDER_PREREGISTRATION.md` and the adversary in "
        "`docs/THREAT_MODEL.md`, both committed before this experiment ran.",
        "",
        "**Detection rule** (unchanged from the original registration): mean "
        f"offline-LiRA AUC >= {DETECT_MEAN_AUC}, AUC > 0.5 on every seed, and the "
        f"Holm-adjusted permutation null rejected on >= {MIN_SIGNIFICANT_SEEDS} of 3 "
        f"seeds at alpha = {ALPHA}. Holm adjustment runs across ladder points within "
        "each seed.",
        "",
        "**Amendment 1**: every ladder point and seed receives the complete "
        "matched-shadow attack. Memorisation is explanatory metadata, never an "
        "execution gate, because discarding low-memorisation observations would "
        "truncate the sample on a variable correlated with the outcome. Points are "
        "classified by combining two independent decisions:",
        "",
        f"* `{DETECTABLE_WITH_MEMORISATION}`",
        f"* `{DETECTABLE_LOW_MEMORISATION}`",
        f"* `{UNDETECTABLE_WITH_MEMORISATION}`",
        f"* `{UNDETECTABLE_LOW_MEMORISATION}`",
        "",
        "No point is ever called private, safe or verified.",
        "",
        "## 2. Threat model summary",
        "",
        "The adversary knows the architecture, optimiser, batch size, epochs, clipping "
        "norm, preprocessing, task definition, source distribution and (at finite "
        "points) the noise multiplier and claimed (epsilon, delta). They do not know "
        "the target's membership set. They train matched shadows on the same pool. "
        "Membership is record-level inclusion; the sensitive attribute is `sex`. "
        "Offline LiRA is primary, online LiRA secondary and reported only with >= 10 IN "
        "and >= 10 OUT observations per example.",
        "",
        "## 3. Fixed experimental design",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Task | `{payload['task']}` |",
        f"| Sensitive attribute | `{payload['sensitive_attribute']}` |",
        f"| Architecture | `Linear(d,{'->'.join(map(str, HIDDEN_UNITS))}->1)` with ReLU |",
        f"| Epochs | {payload.get('epochs', EPOCHS)} |",
        f"| Batch size | {BATCH_SIZE} |",
        f"| Optimiser | Adam, lr = {LEARNING_RATE} |",
        f"| Clipping norm | {MAX_GRAD_NORM} (fixed, public) |",
        f"| delta | {DELTA} |",
        f"| Accountant | {ACCOUNTANT.upper()} |",
        "| Sampling | Poisson |",
        f"| Seeds | {', '.join(str(s) for s in payload['seeds'])} |",
        f"| Shadows per point per seed | {payload['num_shadows']} |",
        f"| Bootstrap / permutation replicates | {payload['bootstrap_reps']} / "
        f"{payload['permutation_reps']} |",
        "",
        "Only the noise multiplier changes between finite points. Cohort indices, "
        "shadow schedules and model seeds derive from the target seed alone, so points "
        "are paired.",
        "",
        "## 4. Privacy accounting",
        "",
        "| Point | Requested epsilon | Noise multiplier | Achieved epsilon | Clipping |",
        "|---|---|---:|---:|---:|",
    ]
    for summary in summaries:
        requested = (
            NON_PRIVATE_LABEL
            if summary["requested_epsilon"] is None
            else f"{float(summary['requested_epsilon']):g}"
        )
        lines.append(
            f"| {summary['label']} | {requested} | {_fmt(summary['noise_multiplier'])} | "
            f"{_fmt(summary['achieved_epsilon'])} | "
            f"{'-' if summary['noise_multiplier'] is None else MAX_GRAD_NORM} |"
        )
    lines += [
        "",
        "Requested epsilon is an input. The achieved epsilon read back from the RDP "
        "accountant after training is the reported figure.",
        "",
        "## 5. Memorisation metrics",
        "",
        "| Point | Seed | Train AUC | Test AUC | AUC gap | Train BCE | Test BCE | "
        "Loss gap | Acc gap | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        for result in summary["seed_results"]:
            m = result["memorisation"]
            lines.append(
                f"| {result['label']} | {result['seed']} | {_fmt(m['train_auc'])} | "
                f"{_fmt(m['test_auc'])} | {_fmt(m['auc_gap'])} | {_fmt(m['train_loss'])} | "
                f"{_fmt(m['test_loss'])} | {_fmt(m['loss_gap'])} | "
                f"{_fmt(m['accuracy_gap'])} | "
                f"{'PASS' if result['memorisation_gate_passed'] else 'FAIL'} |"
            )

    lines += [
        "",
        "## 6. Aggregate LiRA results",
        "",
        "| Point | Seed | Attack | LiRA AUC | TPR@0.1% | TPR@1% | TPR@10% | "
        "Detected @1% | FP @1% | OUT | IN |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for result in summary["seed_results"]:
            if result["attack_status"] != "completed":
                lines.append(
                    f"| {result['label']} | {result['seed']} | - | "
                    f"**{result['attack_status']}** | - | - | - | - | - | - | - |"
                )
                continue
            for variant in ("offline", "online"):
                metrics = result.get(variant)
                if not metrics:
                    lines.append(
                        f"| {result['label']} | {result['seed']} | lira-{variant} | "
                        f"**unavailable: {result.get('online_unavailable_reason', 'n/a')}** "
                        "| - | - | - | - | - | - | - |"
                    )
                    continue
                agg = metrics["aggregate"]
                op = agg["operating_point"]
                lines.append(
                    f"| {result['label']} | {result['seed']} | lira-{variant} | "
                    f"{_fmt(agg['auc'])} | {_fmt(agg['tpr_at_fpr_0.001'])} | "
                    f"{_fmt(agg['tpr_at_fpr_0.01'])} | {_fmt(agg['tpr_at_fpr_0.1'])} | "
                    f"{op['detected_members']}/{op['n_members']} | "
                    f"{op['false_positives']}/{op['n_nonmembers']} | "
                    f"{result['min_out_observations']} | {result['min_in_observations']} |"
                )

    lines += [
        "",
        "## 7. Permutation and bootstrap results",
        "",
        "Permutation nulls reshuffle membership within each sex, preserving exact cell "
        "counts with scores fixed. Holm adjustment is across ladder points within each "
        "seed.",
        "",
        "| Point | Seed | Attack | AUC | Null AUC 95% | p | Holm p | TPR@1% | p | "
        "Holm p | Bootstrap TPR 95% |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        for result in summary["seed_results"]:
            for variant in ("offline", "online"):
                metrics = result.get(variant)
                if not metrics:
                    continue
                agg, perm, boot = (
                    metrics["aggregate"],
                    metrics["permutation"],
                    metrics["bootstrap"],
                )
                lines.append(
                    f"| {result['label']} | {result['seed']} | lira-{variant} | "
                    f"{_fmt(agg['auc'])} | [{_fmt(perm['aggregate_auc']['null_ci95_low'])}, "
                    f"{_fmt(perm['aggregate_auc']['null_ci95_high'])}] | "
                    f"{_fmt(perm['aggregate_auc']['p_value'])} | "
                    f"{_fmt(perm['aggregate_auc'].get('p_value_holm'))} | "
                    f"{_fmt(agg['tpr_at_fpr_0.01'])} | "
                    f"{_fmt(perm['aggregate_tpr']['p_value'])} | "
                    f"{_fmt(perm['aggregate_tpr'].get('p_value_holm'))} | "
                    f"[{_fmt(boot['aggregate']['ci95_low'])}, "
                    f"{_fmt(boot['aggregate']['ci95_high'])}] |"
                )

    lines += [
        "",
        "## 8. Detectability decision per ladder point",
        "",
        "| Point | Achieved epsilon | Noise | Gate passed | Mean offline AUC | "
        "Mean TPR@1% | Min Holm p | Decision | Basis |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | {_fmt(summary['achieved_epsilon'])} | "
            f"{_fmt(summary['noise_multiplier'])} | "
            f"{summary['gate_passed_seeds']}/{len(summary['seeds'])} | "
            f"{_fmt(summary['mean_offline_auc'])} | "
            f"{_fmt(summary['mean_offline_tpr_1pct'])} | "
            f"{_fmt(summary['min_holm_p'])} | **{summary['decision']}** | "
            f"{summary['basis']} |"
        )

    lines += ["", "## 9. Detectability-frontier bracket", ""]
    if frontier["non_monotonic"]:
        lines += [
            "**FRONTIER UNSTABLE -- NON-MONOTONIC.**",
            "",
            str(frontier["summary"]),
            "",
            "The full ordered pattern is reported rather than a manufactured threshold:",
            "",
            "| Point | Achieved epsilon | Decision |",
            "|---|---:|---|",
        ]
        for entry in frontier["ordered_points"]:
            lines.append(
                f"| {entry['label']} | {_fmt(entry['achieved_epsilon'])} | "
                f"{entry['decision']} |"
            )
    else:
        lines.append(str(frontier["summary"]))
        if frontier["bracket"]:
            bracket = frontier["bracket"]
            lines += [
                "",
                "```text",
                f"detectable at epsilon = {_fmt(bracket['detectable_epsilon'])} "
                f"({bracket['detectable_at']})",
                f"not detectable at epsilon = {_fmt(bracket['not_detectable_epsilon'])} "
                f"({bracket['not_detectable_at']})",
                "therefore the empirical transition lies between them",
                "```",
            ]
    lines.append("")
    lines.append(
        "No interpolation is performed across six points, and no point is described as "
        "private on the basis of non-detection."
    )

    lines += [
        "",
        "## 10. Secondary subgroup analysis",
        "",
        "Primary statistic: signed contrast `male TPR@1% - female TPR@1%`. The absolute "
        "max-minus-min gap is descriptive only -- it is positively biased under the "
        "null. `resolution = 1 / subgroup_member_count` is the smallest movement a "
        "single member can produce.",
        "",
        "| Point | Seed | Female TPR@1% | Male TPR@1% | Signed gap | Absolute gap "
        "(descriptive) | Resolution | Signed Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for result in summary["seed_results"]:
            metrics = result.get("offline")
            if not metrics:
                continue
            groups = metrics["subgroups"]

            def group_tpr(name: str) -> object:
                for candidate, values in groups.items():
                    if candidate.lower() == name:
                        return values["tpr_at_fpr_0.01"]
                return None

            lines.append(
                f"| {result['label']} | {result['seed']} | "
                f"{_fmt(group_tpr('female'))} | {_fmt(group_tpr('male'))} | "
                f"{_fmt(metrics['signed_subgroup_difference'])} | "
                f"{_fmt(metrics['absolute_subgroup_gap'])} | "
                f"{_fmt(result.get('subgroup_resolution'))} | "
                f"{_fmt(metrics['permutation']['signed_difference'].get('p_value_holm'))} |"
            )
    lines += [
        "",
        "| Point | Subgroup verdict | Basis |",
        "|---|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | **{summary['subgroup_verdict']}** | "
            f"{summary['subgroup_basis']} |"
        )

    # ---- Generalisation-gap association ------------------------------------ #
    associations = payload.get("gap_associations") or {}
    observations = payload.get("observations") or []
    lines += [
        "",
        "## 10b. Generalisation gap and measured leakage",
        "",
        f"**{ASSOCIATION_ONLY_STATEMENT}**",
        "",
        f"Every seed-level observation is retained ({len(observations)} in total), "
        "including those below the memorisation criterion. Spearman rank "
        "correlations, with seed-stratified bootstrap intervals:",
        "",
        "| Generalisation gap | Leakage metric | n | Spearman rho | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for entry in associations.values():
        lines.append(
            f"| {entry['gap_metric']} | {entry['leakage_metric']} | "
            f"{entry['n_observations']} | {_fmt(entry['spearman'])} | "
            f"[{_fmt(entry['ci95_low'])}, {_fmt(entry['ci95_high'])}] |"
        )
    if not associations:
        lines.append("| - | - | - | - | - |")
    lines += [
        "",
        "Figures `lira_auc_vs_bce_gap.png`, `lira_auc_vs_auc_gap.png`, "
        "`lira_auc_vs_accuracy_gap.png` and `tpr_vs_bce_gap.png` plot every "
        "observation, distinguishing non-private from DP-SGD and "
        "threshold-passing from low-memorisation points.",
        "",
        "No causal, mediation or \"beyond generalisation\" claim is made or implied.",
    ]

    # ---- Conclusion -------------------------------------------------------- #
    non_private = next(
        (s for s in summaries if s["requested_epsilon"] is None), None
    )
    finite = [s for s in summaries if s["requested_epsilon"] is not None]
    detectable_finite = [s for s in finite if s["decision"] in DETECTABLE_STATUSES]
    supported = [s for s in summaries if s["subgroup_verdict"] == SUBGROUP_SUPPORTED]

    lines += ["", "## 11. Conclusion", ""]

    lines += ["### 1. Does the attack detect aggregate leakage without DP?", ""]
    if non_private is None:
        lines.append("The non-private point was not run, so this cannot be answered here.")
    elif non_private["decision"] in DETECTABLE_STATUSES:
        lines.append(
            f"Yes. The non-private target reaches mean offline-LiRA AUC "
            f"{_fmt(non_private['mean_offline_auc'])} with the Holm-adjusted permutation "
            "null rejected on the required number of seeds. The instrument works."
        )
    else:
        lines.append(
            f"**No.** The non-private target memorised (mean AUC gap "
            f"{_fmt(non_private['mean_auc_gap'])}) yet aggregate leakage was not "
            "detectable. Every finite point below is therefore uninterpretable as a "
            "privacy result -- this is an attack-power failure."
        )

    lines += ["", "### 2. At which DP noise levels is aggregate leakage detectable?", ""]
    if detectable_finite:
        names = ", ".join(
            f"{s['label']} (achieved eps {_fmt(s['achieved_epsilon'])})"
            for s in detectable_finite
        )
        lines.append(f"Detectable at: {names}.")
    else:
        lines.append(
            "At none of the finite privacy points tested. Detection survives only in the "
            "non-private control, if at all."
        )

    lines += ["", "### 3. Where does detection disappear?", ""]
    lines.append(str(frontier["summary"]))

    lines += ["", "### 4. Are subgroup differences supported?", ""]
    if supported:
        names = ", ".join(s["label"] for s in supported)
        lines.append(
            f"Supported at: {names} -- stable signed direction, adjusted permutation "
            "rejection on the required seeds, and an effect exceeding one-person "
            "resolution."
        )
    else:
        lines.append(
            "**No.** No ladder point satisfies the pre-registered signed-contrast rule. "
            "Aggregate detection, where it occurs, is reported as aggregate leakage only "
            "and is never read as subgroup disparity."
        )

    lines += ["", "### 5. Is the insurance dataset adequate for aggregate auditing?", ""]
    if non_private is not None and non_private["decision"] in DETECTABLE_STATUSES:
        lines.append(
            "Yes, at non-private capacity, and "
            + (
                "at the finite privacy points listed above."
                if detectable_finite
                else "only there -- every finite point tested falls below detectability."
            )
        )
    else:
        lines.append(
            "Not demonstrated. Without a detectable non-private control the pipeline "
            "cannot support aggregate auditing claims on this dataset."
        )

    lines += [
        "",
        "### 6. Is it adequate for small subgroup-leakage comparisons?",
        "",
    ]
    resolutions = [
        float(r["subgroup_resolution"])
        for s in summaries
        for r in s["seed_results"]
        if r.get("subgroup_resolution") is not None
    ]
    if resolutions:
        lines.append(
            f"**No.** Subgroup TPR@1% FPR moves in steps of {max(resolutions):.4f} "
            f"(one member out of {int(round(1 / max(resolutions)))}), and at a 1% FPR the "
            "operating point admits roughly one false positive per group. The "
            "measurement grid is coarser than the effects a subgroup claim would need "
            "to resolve, independently of how much noise is applied."
        )
    else:
        lines.append(
            "**No.** No completed attack produced a subgroup cohort large enough to "
            "assess."
        )

    lines += [
        "",
        "---",
        "",
        "Non-detection is reported as *not detected at this power*, never as privacy, "
        "safety or verification of a DP guarantee. No subgroup claim is made from "
        "unstable or single-seed results.",
    ]
    return "\n".join(lines) + "\n"


def make_figures(summaries: list[dict[str, object]], output_dir: Path) -> list[str]:
    """Publication-readable figures. No smoothing, no interpolation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def axis_positions() -> tuple[list[int], list[str]]:
        labels = []
        for summary in summaries:
            if summary["requested_epsilon"] is None:
                labels.append("non-private\n(separate category)")
            else:
                achieved = summary["achieved_epsilon"]
                labels.append(
                    f"eps={float(summary['requested_epsilon']):g}\nachieved="
                    f"{achieved:.2f}" if achieved is not None else
                    f"eps={float(summary['requested_epsilon']):g}"
                )
        return list(range(len(summaries))), labels

    positions, labels = axis_positions()
    written: list[str] = []

    def finish(fig, ax, path: Path, title: str, ylabel: str) -> None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Ladder point (non-private shown as a separate category)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        # A visual break: the non-private point is not an arbitrary finite epsilon.
        if summaries and summaries[0]["requested_epsilon"] is None:
            ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

    # -- aggregate AUC ---------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, seed in enumerate(sorted({r["seed"] for s in summaries for r in s["seed_results"]})):
        xs, ys = [], []
        for pos, summary in zip(positions, summaries):
            for result in summary["seed_results"]:
                if result["seed"] == seed and result.get("offline"):
                    xs.append(pos + (offset - 1) * 0.08)
                    ys.append(result["offline"]["aggregate"]["auc"])
        ax.scatter(xs, ys, s=28, alpha=0.75, label=f"seed {seed}")
    means_x = [p for p, s in zip(positions, summaries) if s["mean_offline_auc"] is not None]
    means_y = [s["mean_offline_auc"] for s in summaries if s["mean_offline_auc"] is not None]
    ax.scatter(means_x, means_y, marker="_", s=420, color="black", label="across-seed mean")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="AUC = 0.5 (chance)")
    ax.axhline(
        DETECT_MEAN_AUC, color="darkorange", linestyle="-.", linewidth=1,
        label=f"pre-registered {DETECT_MEAN_AUC}",
    )
    finish(fig, ax, output_dir / "detectability_auc.png",
           "Offline-LiRA aggregate AUC across the noise ladder", "Offline-LiRA ROC-AUC")

    # -- aggregate TPR@1% ------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, seed in enumerate(sorted({r["seed"] for s in summaries for r in s["seed_results"]})):
        xs, ys = [], []
        for pos, summary in zip(positions, summaries):
            for result in summary["seed_results"]:
                if result["seed"] == seed and result.get("offline"):
                    xs.append(pos + (offset - 1) * 0.08)
                    ys.append(result["offline"]["aggregate"]["tpr_at_fpr_0.01"])
        ax.scatter(xs, ys, s=28, alpha=0.75, label=f"seed {seed}")
    mx = [p for p, s in zip(positions, summaries) if s["mean_offline_tpr_1pct"] is not None]
    my = [s["mean_offline_tpr_1pct"] for s in summaries if s["mean_offline_tpr_1pct"] is not None]
    ax.scatter(mx, my, marker="_", s=420, color="black", label="across-seed mean")
    ax.axhline(0.01, color="red", linestyle="--", linewidth=1, label="1% random TPR baseline")
    finish(fig, ax, output_dir / "detectability_tpr_1pct.png",
           "Offline-LiRA TPR at 1% FPR across the noise ladder", "TPR @ 1% FPR")

    # -- memorisation gap -------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pos, summary in zip(positions, summaries):
        for result in summary["seed_results"]:
            ax.scatter(pos - 0.1, result["memorisation"]["auc_gap"], color="tab:blue", s=26)
            ax.scatter(pos + 0.1, result["memorisation"]["accuracy_gap"], color="tab:green", s=26)
    ax.scatter([], [], color="tab:blue", label="train-test AUC gap")
    ax.scatter([], [], color="tab:green", label="train-test accuracy gap")
    ax.axhline(MEMORISATION_GATE, color="darkorange", linestyle="-.", linewidth=1,
               label=f"memorisation gate = {MEMORISATION_GATE}")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
    finish(fig, ax, output_dir / "memorisation_gap.png",
           "Target memorisation across the noise ladder", "Generalisation gap")

    # -- signed subgroup contrast ------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 4.5))
    resolution = None
    for offset, seed in enumerate(sorted({r["seed"] for s in summaries for r in s["seed_results"]})):
        xs, ys = [], []
        for pos, summary in zip(positions, summaries):
            for result in summary["seed_results"]:
                if result["seed"] == seed and result.get("offline"):
                    xs.append(pos + (offset - 1) * 0.08)
                    ys.append(result["offline"]["signed_subgroup_difference"])
                    resolution = result.get("subgroup_resolution", resolution)
        ax.scatter(xs, ys, s=28, alpha=0.75, label=f"seed {seed}")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1, label="no disparity")
    if resolution:
        ax.axhspan(-resolution, resolution, color="grey", alpha=0.2,
                   label=f"+/- one-person resolution ({resolution:.4f})")
    finish(fig, ax, output_dir / "subgroup_signed_contrast.png",
           "Signed subgroup contrast (male - female TPR@1% FPR)",
           "male TPR@1% - female TPR@1%")
    return written



def make_gap_figures(
    observations: Sequence[dict[str, object]], output_dir: Path
) -> list[str]:
    """Scatter every seed-level observation of leakage against generalisation gap.

    All observations appear, including those below the memorisation criterion --
    they are the point of the amended design. Non-private and DP-SGD
    observations use different markers; threshold-passing and low-memorisation
    observations use filled and hollow faces.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("bce_gap", "lira_auc", "lira_auc_vs_bce_gap.png",
         "test - train BCE", "Offline-LiRA AUC"),
        ("auc_gap", "lira_auc", "lira_auc_vs_auc_gap.png",
         "train - test ROC-AUC", "Offline-LiRA AUC"),
        ("accuracy_gap", "lira_auc", "lira_auc_vs_accuracy_gap.png",
         "train - test accuracy", "Offline-LiRA AUC"),
        ("bce_gap", "tpr_at_1pct", "tpr_vs_bce_gap.png",
         "test - train BCE", "Offline-LiRA TPR @ 1% FPR"),
    ]
    written: list[str] = []
    for gap, leak, filename, xlabel, ylabel in specs:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for private, marker, family in ((False, "s", "non-private"), (True, "o", "DP-SGD")):
            for passed, face in ((True, "full"), (False, "none")):
                subset = [
                    o
                    for o in observations
                    if bool(o["private"]) is private
                    and bool(o["memorisation_threshold_passed"]) is passed
                ]
                if not subset:
                    continue
                ax.scatter(
                    [float(o[gap]) for o in subset],
                    [float(o[leak]) for o in subset],
                    marker=marker,
                    s=44,
                    facecolors="none" if face == "none" else None,
                    edgecolors="tab:blue" if private else "tab:red",
                    color=None if face == "none" else ("tab:blue" if private else "tab:red"),
                    label=(
                        f"{family}, "
                        f"{'memorisation present' if passed else 'low memorisation'}"
                    ),
                )
        if leak == "lira_auc":
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="AUC = 0.5")
            ax.axhline(
                DETECT_MEAN_AUC, color="darkorange", linestyle="-.", linewidth=1,
                label=f"detection floor {DETECT_MEAN_AUC}",
            )
        else:
            ax.axhline(
                0.01, color="grey", linestyle="--", linewidth=1, label="1% random TPR"
            )
        ax.axvline(
            MEMORISATION_GATE, color="green", linestyle=":", linewidth=1,
            label=f"memorisation criterion {MEMORISATION_GATE}",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {xlabel} (association only)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
        written.append(filename)
    return written


def run_aggregate(args: argparse.Namespace) -> None:
    """Combine ladder-point artifacts into the final report."""
    root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "noise_ladder.log"
    handle = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        handle.write(message + "\n")
        handle.flush()

    expected = [point_label(e) for e in LADDER_EPSILONS] if args.expect_all else None
    found: dict[str, dict[str, object]] = {}
    for path in sorted(input_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("experiment") != "detectability_noise_ladder_point":
            continue
        found[str(payload["label"])] = payload
        log(f"Loaded ladder point {payload['label']} from {path}")

    if expected is not None:
        missing = [label for label in expected if label not in found]
        if missing:
            handle.close()
            raise SystemExit(
                "Aggregation failed: missing ladder-point artifacts for "
                f"{', '.join(missing)}. Every requested point must complete before the "
                "frontier can be reported."
            )

    order = [point_label(e) for e in LADDER_EPSILONS]
    points = [found[label] for label in order if label in found]
    if not points:
        handle.close()
        raise SystemExit("Aggregation failed: no ladder-point artifacts found.")

    try:
        pairing = verify_pairing(points)
    except ValueError as error:
        log(str(error))
        handle.close()
        raise SystemExit(str(error)) from error
    log(f"Pairing verified across {len(points)} points for seeds {sorted(pairing)}")

    apply_holm_across_points(points)
    summaries = summarise_points(points)
    frontier = detectability_frontier(summaries)
    observations = collect_observations(summaries)
    associations = gap_associations(
        observations, reps=int(points[0].get("bootstrap_reps", 1000)), seed=7
    )

    # Per-example rows go to their own CSV: they are the design record a later
    # paired between-recipe comparison needs, and they would swamp the summary
    # JSON if inlined.
    per_example_rows = [
        record
        for summary in summaries
        for result in summary["seed_results"]
        for record in (result.get("per_example") or [])
    ]
    if per_example_rows:
        pd.DataFrame(per_example_rows).to_csv(output_dir / "per_example.csv", index=False)
        log(f"Wrote {len(per_example_rows)} per-example rows to per_example.csv")
    else:
        log(
            "WARNING: no per-example records found. These artifacts predate "
            "per-example persistence and cannot support a paired between-recipe test."
        )
    for summary in summaries:
        for result in summary["seed_results"]:
            result.pop("per_example", None)

    payload = {
        "experiment": "detectability_noise_ladder",
        "pairing": pairing,
        "per_example_rows": len(per_example_rows),
        "task": TASK_NAME,
        "sensitive_attribute": SENSITIVE_ATTRIBUTE,
        "delta": DELTA,
        "seeds": points[0]["seeds"],
        "num_shadows": points[0]["num_shadows"],
        "bootstrap_reps": points[0]["bootstrap_reps"],
        "permutation_reps": points[0]["permutation_reps"],
        "epochs": points[0].get("epochs", EPOCHS),
        "smoke": any(bool(p.get("smoke")) for p in points),
        "alpha": ALPHA,
        "points": summaries,
        "frontier": frontier,
        "observations": observations,
        "gap_associations": associations,
        "association_only_statement": ASSOCIATION_ONLY_STATEMENT,
    }

    (output_dir / "noise_ladder.json").write_text(json.dumps(payload, indent=2, default=float))
    pd.DataFrame(build_rows(summaries)).to_csv(output_dir / "noise_ladder.csv", index=False)
    (output_dir / "noise_ladder.md").write_text(render_markdown(payload))
    figures = make_figures(summaries, output_dir)
    figures += make_gap_figures(observations, output_dir)
    pd.DataFrame(observations).to_csv(output_dir / "observations.csv", index=False)

    for summary in summaries:
        log(f"[{summary['label']}] {summary['decision']} -- {summary['basis']}")
    log(f"Frontier: {frontier['summary']}")
    log(f"Figures: {', '.join(figures)}")
    log(f"Report: {output_dir / 'noise_ladder.md'}")
    handle.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    point = sub.add_parser("point", help="run one ladder point across all seeds")
    point.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="requested epsilon; omit or pass <= 0 for the non-private point",
    )
    point.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    point.add_argument("--shadows", type=int, default=DEFAULT_SHADOWS)
    point.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    point.add_argument("--permutation-reps", type=int, default=DEFAULT_PERMUTATION_REPS)
    point.add_argument("--epochs", type=int, default=None, help="smoke-test override")
    point.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "CI smoke mode: lowers the Gaussian observation floor so a few shadows "
            "suffice. Stamps the artifact as smoke; never a pre-registered result."
        ),
    )
    point.add_argument("--output-dir", default="reports/detectability_noise_ladder/points")
    point.set_defaults(func=run_point)

    aggregate = sub.add_parser("aggregate", help="combine ladder points into the report")
    aggregate.add_argument("--input-dir", default="reports/detectability_noise_ladder/points")
    aggregate.add_argument("--output-dir", default="reports/detectability_noise_ladder")
    aggregate.add_argument(
        "--expect-all",
        action="store_true",
        help="fail unless every requested ladder point is present",
    )
    aggregate.set_defaults(func=run_aggregate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
