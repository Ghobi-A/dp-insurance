"""Canary constructions for empirical privacy auditing.

A *canary* is a crafted record inserted into (or withheld from) a training set
so that a membership-inference attack can be scored against ground truth.  The
strength of an audit depends heavily on how memorable the canary is: a bound is
only ever as tight as the worst case the attack actually probes (Nasr et al.,
"Adversary Instantiation", IEEE S&P 2021).  This module provides three
constructions that probe different failure modes, so an audit can test more
than the single label-flip canary the notebook originally used:

``make_label_flip_canaries``
    Real feature rows with their labels flipped.  An *outlier in label space*:
    the point is in-distribution but its label contradicts the pattern, so a
    model can only fit it by memorising.  This is the classic auditing device
    (Carlini et al. 2019; Jagielski et al. 2020).

``make_feature_outlier_canaries``
    Synthetic rows placed far from the data cloud in random, mutually
    near-orthogonal directions — an *outlier in feature space*.  Each point is
    unique and isolated, so a high-capacity model that memorises it is exposed,
    while independence keeps one canary's inclusion from leaking through a
    near-duplicate neighbour.

``make_duplicated_canaries``
    A distinctive record repeated ``k`` times, the worst-case memorisation
    probe: duplication amplifies the record's influence on the model ``k``-fold.
    Under DP this audits *group* privacy for a group of size ``k``, whose budget
    degrades to ≈ ``k·ε`` (group-privacy property), so its audit must be read
    against that larger budget, not the single-record ε.

Each constructor returns a :class:`CanarySet` of feature rows and labels; the
audit driver (see :func:`dp.audit.audit_membership_scores`) draws the inclusion
coins, trains once, scores every canary, and inverts the count into an ε lower
bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CanarySet:
    """A pool of canaries to audit.

    Attributes:
        features: Shape ``(n, d)``. One feature row per canary.
        labels: Shape ``(n,)``. The (audited) label of each canary.
        duplication: How many times each *included* canary is repeated in the
            training set.  ``1`` for single-record canaries; ``k`` marks a
            group-privacy probe whose audit is read against ≈ ``k·ε``.
        kind: Name of the construction, for reporting.
    """

    features: np.ndarray
    labels: np.ndarray
    duplication: int
    kind: str

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be a 2-D (n, d) array")
        if self.labels.shape[0] != self.features.shape[0]:
            raise ValueError("features and labels must align on axis 0")
        if self.duplication < 1:
            raise ValueError("duplication must be >= 1")

    def __len__(self) -> int:
        return self.features.shape[0]


def make_label_flip_canaries(
    features: np.ndarray,
    labels: np.ndarray,
    num_canaries: int,
    random_state: int | None = None,
) -> CanarySet:
    """Draw ``num_canaries`` distinct rows and flip their (binary) labels.

    An outlier in *label* space: the feature row is in-distribution but its
    label contradicts the model's learned pattern, so fitting it requires
    memorisation.  Rows are sampled *without replacement* so no two canaries
    coincide — near-duplicate canaries would let an excluded canary inherit an
    included neighbour's memorisation and corrupt the audit.

    Args:
        features: Pool of feature rows to draw from, shape ``(N, d)``.
        labels: Binary labels (0/1) aligned with ``features``.
        num_canaries: Number of canaries to draw (``<= N``).
        random_state: Seed for the row selection.

    Returns:
        A :class:`CanarySet` with flipped labels and ``duplication == 1``.

    Raises:
        ValueError: If ``num_canaries`` exceeds the pool size or labels are not
            binary.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if num_canaries > features.shape[0]:
        raise ValueError("num_canaries cannot exceed the pool size")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary (0/1) to flip")

    rng = np.random.default_rng(random_state)
    idx = rng.choice(features.shape[0], size=num_canaries, replace=False)
    return CanarySet(
        features=features[idx].copy(),
        labels=(1 - labels[idx]).astype(int),
        duplication=1,
        kind="label-flip",
    )


def make_feature_outlier_canaries(
    num_canaries: int,
    num_features: int,
    random_state: int | None = None,
    radius: float = 6.0,
) -> CanarySet:
    """Create synthetic canaries that are outliers in *feature* space.

    Each canary is a random unit vector scaled to ``radius``, so on
    standardised features (unit variance) it sits ~``radius`` standard
    deviations from the origin — far outside the data cloud.  In high dimension
    the directions are near-orthogonal, so the points are mutually isolated:
    an excluded canary lands nowhere near an included one, which keeps the
    membership signal clean.  Labels are assigned by a fair coin.

    Args:
        num_canaries: Number of canaries to create.
        num_features: Feature dimension ``d`` (match the preprocessed matrix).
        random_state: Seed for the directions and labels.
        radius: Distance from the origin in standardised units.

    Returns:
        A :class:`CanarySet` with ``duplication == 1``.

    Raises:
        ValueError: If ``num_features`` or ``radius`` is not positive.
    """
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    rng = np.random.default_rng(random_state)
    directions = rng.standard_normal(size=(num_canaries, num_features))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return CanarySet(
        features=directions * radius,
        labels=rng.integers(0, 2, size=num_canaries).astype(int),
        duplication=1,  # single-record; see make_duplicated_canaries for groups
        kind="feature-outlier",
    )


def make_duplicated_canaries(
    num_canaries: int,
    num_features: int,
    duplication: int = 8,
    random_state: int | None = None,
    radius: float = 6.0,
) -> CanarySet:
    """Create feature-outlier canaries to be repeated ``duplication`` times.

    The worst-case memorisation probe: repeating an included canary
    ``duplication`` times multiplies its gradient contribution, so its
    influence on the trained model is amplified ``duplication``-fold.  Because
    DP protects a *group* of ``duplication`` identical records only at a budget
    that grows to ≈ ``duplication · ε`` (the group-privacy property), an audit
    that finds more leakage here does not by itself violate the single-record
    ε; it must be compared against the inflated group budget.

    Args:
        num_canaries: Number of distinct canaries.
        num_features: Feature dimension ``d``.
        duplication: Repetition count ``k`` for each included canary (``>= 1``).
        random_state: Seed for the directions and labels.
        radius: Distance from the origin in standardised units.

    Returns:
        A :class:`CanarySet` whose ``duplication`` field records ``k``.

    Raises:
        ValueError: If ``duplication`` < 1 or geometry parameters are invalid.
    """
    if duplication < 1:
        raise ValueError("duplication must be >= 1")
    base = make_feature_outlier_canaries(
        num_canaries, num_features, random_state=random_state, radius=radius
    )
    return CanarySet(
        features=base.features,
        labels=base.labels,
        duplication=duplication,
        kind="duplicated",
    )
