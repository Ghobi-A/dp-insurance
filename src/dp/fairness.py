"""Group-fairness metrics for binary classifiers.

Both metrics follow the max-minus-min convention across groups (matching
fairlearn's definitions), so they generalise beyond two groups and are always
non-negative.  They are pure post-processing of model predictions: computing
them on the output of a differentially private model consumes no additional
privacy budget.
"""

from __future__ import annotations

import numpy as np


def _group_rates(values: np.ndarray, group: np.ndarray) -> dict:
    # Mean of `values` within each group; empty groups cannot occur since
    # groups are derived from `group` itself.
    return {g: float(values[group == g].mean()) for g in np.unique(group)}


def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Difference between the largest and smallest group positive-prediction rates.

    A value of 0 means every group receives positive predictions at the same
    rate; 0.10 means a 10-percentage-point gap between the most- and
    least-selected groups.

    Args:
        y_pred: Binary predictions (0/1).
        sensitive: Group membership for each prediction (any hashable values).

    Returns:
        max_g P(ŷ=1 | g) − min_g P(ŷ=1 | g), in [0, 1].

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    if len(y_pred) == 0:
        raise ValueError("y_pred must not be empty")
    if len(y_pred) != len(sensitive):
        raise ValueError("y_pred and sensitive must have the same length")
    rates = _group_rates(y_pred, sensitive)
    return max(rates.values()) - min(rates.values())


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Worst-case gap in true-positive or false-positive rates across groups.

    Computes the TPR gap (max − min across groups, among the true positives)
    and the FPR gap (among the true negatives) and returns the larger of the
    two.  A value of 0 means the classifier's error profile is identical
    across groups.  Groups with no positives (or no negatives) are skipped
    for the corresponding rate.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred: Binary predictions (0/1).
        sensitive: Group membership for each observation.

    Returns:
        max(TPR gap, FPR gap), in [0, 1].

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty")
    if not (len(y_true) == len(y_pred) == len(sensitive)):
        raise ValueError("y_true, y_pred and sensitive must have the same length")

    gaps = []
    for label in (1, 0):  # label==1 → TPR over positives; label==0 → FPR over negatives
        mask = y_true == label
        rates = [
            float(y_pred[mask & (sensitive == g)].mean())
            for g in np.unique(sensitive)
            if (mask & (sensitive == g)).any()
        ]
        if len(rates) >= 2:
            gaps.append(max(rates) - min(rates))
    return max(gaps) if gaps else 0.0
