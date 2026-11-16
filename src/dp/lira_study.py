"""Design and analysis for the prospective real-data LiRA validation study.

Protocol: ``docs/LIRA_VALIDATION_PROTOCOL.md`` (version 1). This module holds
everything that does not train a model: the fixed public encoding, the
membership design, score construction on top of
:func:`dp.attacks.lira_offline_attack`, and the predeclared statistics.

The frozen constants below are the protocol. Changing any of them changes
``protocol_hash`` and makes saved measurements unusable for the confirmatory
analysis.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .attacks import lira_offline_attack, loss_threshold_attack

PROTOCOL_VERSION = 1
STUDY_SEED = 20261117
PILOT_SEED = 999  # pilot measurements are never analysed as confirmatory

# --- data: fixed public encoding, identical for every model -----------------
NUMERIC_BOUNDS = {"age": (18.0, 64.0), "bmi": (15.0, 55.0), "children": (0.0, 5.0)}
REGIONS = ("northeast", "northwest", "southeast", "southwest")
#: Fixed label threshold (the median of ``charges`` over the whole public
#: table). It is a constant of the protocol, not fitted per training set, so it
#: is the same function of the pool for members and non-members.
HIGH_COST_THRESHOLD = 9382.033
N_RECORDS = 1338
TRAIN_SIZE = N_RECORDS // 2

# --- models ------------------------------------------------------------------
HIDDEN_UNITS = (128, 128, 64)
EPOCHS = 400
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
CONDITIONS = ("nonprivate", "clip_only", "dp_eps8", "dp_eps2")
REQUESTED_EPSILON = {"dp_eps8": 8.0, "dp_eps2": 2.0}

# --- design ------------------------------------------------------------------
N_TARGETS = 16
N_SHADOWS = 128  # 64 complementary pairs
SHADOW_COUNTS = (22, 64, 128)  # 11 / 32 / 64 OUT observations per example
STATISTICS = ("raw_loss", "logit_confidence")
VARIANCES = ("per_example", "global")

# --- predeclared analysis ------------------------------------------------------
BASELINE = {"statistic": "raw_loss", "variance": "per_example", "shadows": 22}
PRIMARY = {"statistic": "logit_confidence", "variance": "per_example", "shadows": 128}
PRIMARY_CONDITION = "dp_eps8"
ALPHA = 0.05
FPRS = (0.001, 0.01, 0.1)
HEADLINE_FPR = 0.01


def protocol_dict() -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "study_seed": STUDY_SEED,
        "numeric_bounds": NUMERIC_BOUNDS,
        "regions": REGIONS,
        "high_cost_threshold": HIGH_COST_THRESHOLD,
        "n_records": N_RECORDS,
        "train_size": TRAIN_SIZE,
        "hidden_units": HIDDEN_UNITS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "delta": DELTA,
        "conditions": CONDITIONS,
        "requested_epsilon": REQUESTED_EPSILON,
        "n_targets": N_TARGETS,
        "n_shadows": N_SHADOWS,
        "shadow_counts": SHADOW_COUNTS,
        "statistics": STATISTICS,
        "variances": VARIANCES,
        "baseline": BASELINE,
        "primary": PRIMARY,
        "primary_condition": PRIMARY_CONDITION,
        "alpha": ALPHA,
        "fprs": FPRS,
    }


def protocol_hash() -> str:
    blob = json.dumps(protocol_dict(), sort_keys=True, default=list)
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def encode_fixed(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise public encoding: no statistic is learned from any training set.

    Numeric columns are mapped to [-1, 1] with fixed bounds (values outside are
    clipped), ``sex`` and ``smoker`` to {0, 1}, ``region`` to a fixed one-hot.
    """
    columns = []
    for name, (lo, hi) in NUMERIC_BOUNDS.items():
        values = np.clip(df[name].to_numpy(dtype=float), lo, hi)
        columns.append(2.0 * (values - lo) / (hi - lo) - 1.0)
    columns.append((df["sex"].to_numpy() == "male").astype(float))
    columns.append((df["smoker"].to_numpy() == "yes").astype(float))
    region = df["region"].to_numpy()
    if not np.isin(region, REGIONS).all():
        raise ValueError("unexpected region outside the fixed schema")
    columns.extend((region == r).astype(float) for r in REGIONS)
    X = np.column_stack(columns).astype(np.float32)
    y = (df["charges"].to_numpy(dtype=float) > HIGH_COST_THRESHOLD).astype(np.float32)
    return X, y


def duplicate_groups(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Group id per record; records with identical features and label share one."""
    keys = [row.tobytes() + bytes([int(label)]) for row, label in zip(X, y)]
    lookup: dict[bytes, int] = {}
    return np.array([lookup.setdefault(k, len(lookup)) for k in keys])


# --------------------------------------------------------------------------- #
# Membership design
# --------------------------------------------------------------------------- #


def _rng(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, *stream])


def membership_matrix(seed: int, n: int = N_RECORDS, n_targets: int = N_TARGETS,
                      n_shadows: int = N_SHADOWS) -> np.ndarray:
    """Boolean ``(n_targets + n_shadows, n)`` inclusion matrix.

    Rows ``0..n_targets-1`` are targets: independent uniformly random halves.
    The remaining rows are shadows in complementary pairs (a random half and
    its complement), so within any even-length prefix of shadows every record
    is OUT of exactly half of them. Shadows are drawn independently of targets.
    """
    if n % 2 or n_shadows % 2:
        raise ValueError("n and n_shadows must be even")
    half = n // 2
    rows = np.zeros((n_targets + n_shadows, n), dtype=bool)
    for t in range(n_targets):
        rows[t, _rng(seed, 0, t).permutation(n)[:half]] = True
    for pair in range(n_shadows // 2):
        order = _rng(seed, 1, pair).permutation(n)
        rows[n_targets + 2 * pair, order[:half]] = True
        rows[n_targets + 2 * pair + 1, order[half:]] = True
    return rows


def model_seed(seed: int, model_index: int) -> int:
    """Initialisation and batch-order seed; shared by all conditions (pairing)."""
    return int(_rng(seed, 2, model_index).integers(0, 2**31 - 1))


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


def out_matrix(shadow_losses: np.ndarray, shadow_members: np.ndarray, shadows: int) -> np.ndarray:
    """``(n, shadows / 2)`` OUT losses per record from the first ``shadows`` shadows."""
    members = shadow_members[:shadows]
    out_counts = (~members).sum(axis=0)
    if not (out_counts == shadows // 2).all():
        raise ValueError("design is not balanced for this shadow count")
    order = np.argsort(members, axis=0, kind="stable")[: shadows // 2]  # OUT rows first
    return np.take_along_axis(shadow_losses[:shadows], order, axis=0).T


def configurations() -> list[dict[str, object]]:
    return [
        {"statistic": s, "variance": v, "shadows": k}
        for k, s, v in itertools.product(SHADOW_COUNTS, STATISTICS, VARIANCES)
    ]


def config_name(config: dict[str, object]) -> str:
    if config.get("attack") == "loss_threshold":
        return "loss-threshold"
    return f"{config['statistic']}|{config['variance']}|K={config['shadows']}"


def null_auc_sd(n_members: int, n_nonmembers: int) -> float:
    return float(np.sqrt((n_members + n_nonmembers + 1) / (12.0 * n_members * n_nonmembers)))


def tpr_interval(tpr: float, n_members: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Clopper-Pearson interval for one target's TPR at a fixed threshold.

    Conditional on the threshold; it ignores threshold-selection variability
    and is therefore descriptive.
    """
    k = int(round(tpr * n_members))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n_members - k + 1))
    hi = 1.0 if k == n_members else float(stats.beta.ppf(1 - alpha / 2, k + 1, n_members - k))
    return lo, hi


def score_target(target_losses: np.ndarray, membership: np.ndarray, shadow_losses: np.ndarray,
                 shadow_members: np.ndarray, config: dict[str, object]):
    if config.get("attack") == "loss_threshold":
        return loss_threshold_attack(
            target_losses[membership == 1], target_losses[membership == 0], fprs=FPRS
        )
    return lira_offline_attack(
        target_losses,
        membership,
        out_matrix(shadow_losses, shadow_members, int(config["shadows"])),
        fprs=FPRS,
        statistic=str(config["statistic"]),
        variance=str(config["variance"]),
    )


# --------------------------------------------------------------------------- #
# Inference with the target model as the unit
# --------------------------------------------------------------------------- #


def mean_t_interval(values: Sequence[float], alpha: float = ALPHA) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    if v.size < 2 or v.std(ddof=1) == 0:
        return {"mean": mean, "ci_low": float("nan"), "ci_high": float("nan"), "n": int(v.size)}
    half = float(stats.t.ppf(1 - alpha / 2, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n": int(v.size)}


def signflip_pvalue(differences: Sequence[float], max_exact: int = 20, reps: int = 100_000,
                    seed: int = 0) -> float:
    """Two-sided paired sign-flip test of mean difference zero, exact when small."""
    d = np.asarray(differences, dtype=float)
    observed = abs(d.mean())
    if d.size <= max_exact:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=d.size)))
        null = np.abs(signs @ d) / d.size
        return float(np.mean(null >= observed - 1e-15))
    rng = np.random.default_rng(seed)
    null = np.abs(rng.choice((-1.0, 1.0), size=(reps, d.size)) @ d) / d.size
    return float((1 + np.sum(null >= observed - 1e-15)) / (reps + 1))


def stouffer_detection(aucs: Sequence[float], n_members: int, n_nonmembers: int) -> dict[str, float]:
    """One-sided combined test of no score-membership association.

    Each target's membership vector is an independent uniformly random half,
    so under the null the target AUCs are independent with mean 1/2 and the
    exact permutation variance, whatever shadows the targets share.
    """
    z = (np.asarray(aucs, dtype=float) - 0.5) / null_auc_sd(n_members, n_nonmembers)
    combined = float(z.sum() / np.sqrt(z.size))
    return {"z": combined, "p_one_sided": float(stats.norm.sf(combined))}


def holm(p_values: Sequence[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (p.size - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def detection_power(true_auc: float, n_targets: int = N_TARGETS, n_per_class: int = TRAIN_SIZE,
                    alpha: float = ALPHA, sims: int = 2000, seed: int = 0,
                    tau: float = 0.0) -> float:
    """Simulated power of :func:`stouffer_detection` under binormal scores."""
    from .ladder_rule_power import simulate_seed_aucs

    rng = np.random.default_rng(seed)
    true = np.clip(true_auc + tau * rng.standard_normal((sims, n_targets)), 1e-4, 1 - 1e-4)
    aucs = simulate_seed_aucs(true, n_per_class, rng)
    z = (aucs - 0.5) / null_auc_sd(n_per_class, n_per_class)
    return float(np.mean(stats.norm.sf(z.sum(axis=1) / np.sqrt(n_targets)) < alpha))
