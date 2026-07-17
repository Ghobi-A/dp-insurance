"""Classification, calibration and group-fairness metric computation.

All functions operate on plain NumPy arrays of true labels, predicted
scores (probability of class 1) and, where relevant, thresholded 0/1
predictions.  Threshold selection is the caller's responsibility and must
use the validation partition, never the test partition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .fairness import demographic_parity_difference, equalized_odds_difference

logger = logging.getLogger(__name__)

#: Groups smaller than this are flagged as unstable for fairness metrics.
SMALL_GROUP_THRESHOLD = 30


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error with equal-width probability bins.

    ECE = sum over bins of (bin size / n) * |mean confidence - accuracy in
    bin|, using the predicted probability of class 1 as confidence for the
    positive class.  Empty bins contribute nothing.

    Args:
        y_true: 0/1 labels.
        y_score: Predicted probability of class 1.
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        ECE in [0, 1].

    Raises:
        ValueError: If inputs are empty or mismatched in length.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        raise ValueError("y_true must not be empty")
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-inclusive final bin so scores of exactly 1.0 are counted.
    bin_ids = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_score[mask].mean() - y_true[mask].mean())
    return float(ece)


def select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pick the decision threshold maximising F1 on the given partition.

    Call this with *validation* labels and scores only; using the test set
    here would leak the final evaluation data into a tuned decision.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    candidates = np.unique(np.round(y_score, 6))
    if candidates.size > 200:
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute the standard metric panel at a pre-selected threshold.

    Returns a dict with roc_auc, pr_auc, accuracy, precision, recall, f1,
    macro_f1, brier and ece. Degenerate single-class test partitions yield
    NaN for ranking metrics rather than raising.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": expected_calibration_error(y_true, y_score),
    }


@dataclass(frozen=True)
class GroupFairnessReport:
    """Group-fairness summary for a thresholded classifier.

    Attributes:
        demographic_parity_difference: Max-minus-min positive rate gap.
        equalized_odds_difference: Worst-case TPR/FPR gap.
        group_tpr: Per-group true-positive rate (NaN if no positives).
        group_fpr: Per-group false-positive rate (NaN if no negatives).
        group_sizes: Number of test examples per group.
        small_groups: Groups below :data:`SMALL_GROUP_THRESHOLD`, whose
            rates should be treated as unstable.
    """

    demographic_parity_difference: float
    equalized_odds_difference: float
    group_tpr: dict[str, float]
    group_fpr: dict[str, float]
    group_sizes: dict[str, int]
    small_groups: tuple[str, ...]


def group_fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> GroupFairnessReport:
    """Compute group fairness metrics with small-group warnings.

    Groups with fewer than :data:`SMALL_GROUP_THRESHOLD` members are listed
    in ``small_groups`` and a warning is logged; their rates are still
    reported but should be presented with qualification.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    group_tpr: dict[str, float] = {}
    group_fpr: dict[str, float] = {}
    group_sizes: dict[str, int] = {}
    small: list[str] = []
    for g in np.unique(sensitive):
        mask = sensitive == g
        key = str(g)
        group_sizes[key] = int(mask.sum())
        pos = mask & (y_true == 1)
        neg = mask & (y_true == 0)
        group_tpr[key] = float(y_pred[pos].mean()) if pos.any() else float("nan")
        group_fpr[key] = float(y_pred[neg].mean()) if neg.any() else float("nan")
        if group_sizes[key] < SMALL_GROUP_THRESHOLD:
            small.append(key)
    if small:
        logger.warning(
            "Fairness metrics unstable for small groups (n < %d): %s",
            SMALL_GROUP_THRESHOLD,
            ", ".join(small),
        )
    return GroupFairnessReport(
        demographic_parity_difference=demographic_parity_difference(y_pred, sensitive),
        equalized_odds_difference=equalized_odds_difference(y_true, y_pred, sensitive),
        group_tpr=group_tpr,
        group_fpr=group_fpr,
        group_sizes=group_sizes,
        small_groups=tuple(small),
    )
