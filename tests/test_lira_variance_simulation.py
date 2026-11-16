"""Structural tests for the LiRA variance simulation.

These check the generator and bookkeeping only. Which variance rule wins is a
research outcome reported by the simulation, not an invariant asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from dp import lira_variance_simulation as sim
from dp.attacks import logit_confidence_from_bce


def test_bce_mapping_is_positive_and_inverts_the_logit_transform():
    logits = np.linspace(-8, 8, 33)
    losses = sim.bce_from_logit_confidence(logits)
    assert (losses > 0).all()
    np.testing.assert_allclose(logit_confidence_from_bce(losses), logits, atol=1e-9)


@pytest.mark.parametrize("scenario", sim.SCENARIOS)
def test_simulated_losses_are_valid_and_balanced(scenario):
    target, membership, shadow = sim.simulate_losses(scenario, 100, 7, np.random.default_rng(0))
    assert target.shape == (100,) and shadow.shape == (100, 7)
    assert (target > 0).all() and (shadow > 0).all()
    assert membership.sum() == 50


def test_scenarios_differ_only_in_variance_structure():
    def spreads(scenario):
        _, _, shadow = sim.simulate_losses(scenario, 400, 200, np.random.default_rng(1))
        return logit_confidence_from_bce(shadow).std(axis=1, ddof=1)

    homo, hetero = spreads("homoscedastic"), spreads("heteroscedastic")
    assert homo.std() / homo.mean() < 0.1
    assert hetero.std() / hetero.mean() > 0.4


def test_comparison_is_reproducible_and_complete():
    first = sim.compare_variance_rules("homoscedastic", 5, n=80, replicates=5, seed=3)
    again = sim.compare_variance_rules("homoscedastic", 5, n=80, replicates=5, seed=3)
    assert first == again
    assert first["auc_diff_global_minus_per_example"] == pytest.approx(
        first["auc_global"] - first["auc_per_example"]
    )
    assert all(np.isfinite(v) for v in first.values() if isinstance(v, float))


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        sim.simulate_losses("other", 10, 3, np.random.default_rng(0))
