"""Conservative bounded-differences AUC sensitivity under explicit assumptions.

Not an ACS survey certificate. Concentration is conditional on the fixed label
vector and cluster membership, independent score-generating clusters, and
fixed trained models. Marginal-AUC interpretation additionally requires a
common class-conditional score law across clusters under that conditioning.
"""
from __future__ import annotations

import math

import numpy as np

from .auc_uncertainty import auc_influence


def cluster_radius(y, groups, *, alpha=.05, family=1):
    y, groups = np.asarray(y), np.asarray(groups)
    if (y.ndim != 1 or groups.shape != y.shape or not np.isin(y, [0, 1]).all()
            or not 0 < alpha < 1 or not isinstance(family, int) or family < 1):
        raise ValueError("invalid labels/groups/confidence family")
    _, index = np.unique(groups, return_inverse=True)
    positive = np.bincount(index, weights=y)
    negative = np.bincount(index, weights=1-y)
    p, n = positive.sum(), negative.sum()
    if min(p, n) < 20:
        raise ValueError("at least 20 observations of each class required")
    # Union of all positive-negative rank pairs touching household h.
    # Same-household pairs must be counted once, not twice.
    within = positive * negative / (p*n)
    sensitivity = positive/p + negative/n - within
    radius = math.sqrt(.5 * float(sensitivity @ sensitivity) * math.log(family/alpha))
    return radius, float(within.sum()), len(positive)


def concentration_bounds(y, reference, candidate, groups, *, alpha=.05, family=1):
    """McDiarmid one-sided bounds plus worst-case within-cluster-pair bias.

    Candidate kernel has range [0,1], paired-gap kernel [-1,1]. Given the
    common conditional marginal law assumption, within-cluster bias is at
    most w for AUC and 2w for a paired gap. All assumptions are material;
    bounds on real ACS data are sensitivity analyses, not certification.
    """
    a0, _ = auc_influence(y, reference)
    a1, _ = auc_influence(y, candidate)
    radius, within, clusters = cluster_radius(y, groups, alpha=alpha, family=family)
    return {"reference_auc": a0, "auc": a1, "gap": a0-a1,
            "auc_lower": max(0., a1-radius-within),
            "gap_upper": min(1., a0-a1+2*radius+2*within),
            "auc_radius": radius+within, "gap_radius": 2*radius+2*within,
            "within_pair_mass": within, "clusters": clusters,
            "alpha": alpha, "family": family,
            "method": "cluster bounded differences plus within-pair bias; assumption-dependent sensitivity",
            "acs_coverage_certified": False}
