"""Paired AUC influence-function bounds with independent-cluster sandwich variance.

Asymptotic research intervals, not finite-sample or arbitrary-shift certificates.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t


def auc_influence(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if len(y) != len(scores) or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid binary outcomes/scores")
    positive, negative = scores[y == 1], scores[y == 0]
    if min(len(positive), len(negative)) < 20:
        raise ValueError("at least 20 observations of each class required")
    neg, pos = np.sort(negative), np.sort(positive)
    vpos = (np.searchsorted(neg, positive, side="left")
            + np.searchsorted(neg, positive, side="right")) / (2 * len(neg))
    vneg = 1 - (np.searchsorted(pos, negative, side="left")
                + np.searchsorted(pos, negative, side="right")) / (2 * len(pos))
    auc = float(vpos.mean())
    influence = np.empty(len(y))
    influence[y == 1] = (vpos - auc) / len(positive)
    influence[y == 0] = (vneg - auc) / len(negative)
    return auc, influence


def cluster_se(influence, groups):
    groups = np.asarray(groups)
    if len(groups) != len(influence):
        raise ValueError("group length mismatch")
    unique, index = np.unique(groups, return_inverse=True)
    if len(unique) < 30:
        raise ValueError("at least 30 independent clusters required")
    sums = np.bincount(index, weights=influence)
    variance = len(unique) / (len(unique) - 1) * np.sum((sums - sums.mean()) ** 2)
    return float(np.sqrt(variance)), len(unique)


def paired_bounds(y, reference, candidate, groups, alpha=0.05, family=1):
    if not 0 < alpha < 1 or not isinstance(family, int) or family < 1:
        raise ValueError("invalid confidence/family")
    a0, i0 = auc_influence(y, reference)
    a1, i1 = auc_influence(y, candidate)
    gap_se, clusters = cluster_se(i0 - i1, groups)
    auc_se, _ = cluster_se(i1, groups)
    z = float(t.ppf(1 - alpha / family, clusters - 1))
    supported = 0 < a0 < 1 and 0 < a1 < 1  # boundary degeneracy cannot certify decisions
    return {"reference_auc": a0, "auc": a1, "gap": a0 - a1,
            "gap_se": gap_se, "auc_se": auc_se, "clusters": clusters,
            "gap_upper": min(1.0, a0 - a1 + z * gap_se) if supported else 1.0,
            "auc_lower": max(0.0, a1 - z * auc_se) if supported else 0.0,
            "bounds_supported": supported, "alpha": alpha, "family": family,
            "method": "paired cluster influence sandwich; asymptotic Bonferroni bounds"}
