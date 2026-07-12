"""Tests for the canary constructions used by the empirical audit.

Each construction probes a different memorisation failure mode; these tests pin
down the structural properties an audit relies on — distinctness (so an excluded
canary does not inherit an included neighbour's memorisation), correct label
flipping, feature-space isolation, and duplication bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pytest

from dp.canaries import (
    CanarySet,
    make_duplicated_canaries,
    make_feature_outlier_canaries,
    make_label_flip_canaries,
)


def _pool(n=200, d=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n, d))
    y = rng.integers(0, 2, size=n)
    return x, y


def test_label_flip_flips_labels_and_keeps_features():
    x, y = _pool()
    cs = make_label_flip_canaries(x, y, num_canaries=50, random_state=0)
    assert cs.kind == "label-flip"
    assert cs.duplication == 1
    assert cs.features.shape == (50, x.shape[1])
    # Each canary's label is the flip of some real row it copied.
    for row, lab in zip(cs.features, cs.labels):
        matches = np.all(np.isclose(x, row), axis=1)
        assert matches.any()
        assert lab == 1 - y[matches.argmax()]


def test_label_flip_rows_are_distinct():
    x, y = _pool()
    cs = make_label_flip_canaries(x, y, num_canaries=120, random_state=1)
    # No two canaries share a feature row (sampled without replacement).
    unique = np.unique(cs.features, axis=0)
    assert unique.shape[0] == len(cs)


def test_label_flip_rejects_oversized_or_nonbinary():
    x, y = _pool(n=30)
    with pytest.raises(ValueError, match="pool size"):
        make_label_flip_canaries(x, y, num_canaries=31)
    with pytest.raises(ValueError, match="binary"):
        make_label_flip_canaries(x, np.full(30, 2), num_canaries=5)


def test_feature_outlier_radius_and_separation():
    cs = make_feature_outlier_canaries(80, num_features=12, random_state=0, radius=6.0)
    assert cs.kind == "feature-outlier"
    assert cs.duplication == 1
    norms = np.linalg.norm(cs.features, axis=1)
    assert np.allclose(norms, 6.0)
    # Outliers are mutually far apart: nearest-neighbour distance is large
    # relative to a single feature's scale (near-orthogonal in high dimension).
    gram = cs.features @ cs.features.T
    np.fill_diagonal(gram, -np.inf)
    max_cos = gram.max() / 36.0  # divide by radius^2
    assert max_cos < 0.9
    assert set(np.unique(cs.labels)).issubset({0, 1})


def test_feature_outlier_validates_geometry():
    with pytest.raises(ValueError, match="num_features"):
        make_feature_outlier_canaries(10, num_features=0)
    with pytest.raises(ValueError, match="radius"):
        make_feature_outlier_canaries(10, num_features=4, radius=0.0)


def test_duplicated_records_duplication_factor():
    cs = make_duplicated_canaries(40, num_features=8, duplication=8, random_state=2)
    assert cs.kind == "duplicated"
    assert cs.duplication == 8
    assert cs.features.shape == (40, 8)
    with pytest.raises(ValueError, match="duplication"):
        make_duplicated_canaries(10, num_features=4, duplication=0)


def test_canaryset_validates_shapes():
    with pytest.raises(ValueError, match="2-D"):
        CanarySet(np.zeros(5), np.zeros(5), 1, "bad")
    with pytest.raises(ValueError, match="align"):
        CanarySet(np.zeros((5, 3)), np.zeros(4), 1, "bad")
    with pytest.raises(ValueError, match="duplication"):
        CanarySet(np.zeros((5, 3)), np.zeros(5), 0, "bad")
    assert len(CanarySet(np.zeros((7, 3)), np.zeros(7), 1, "ok")) == 7
