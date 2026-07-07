"""Privacy audits as regression tests.

These tests treat the empirical ε lower bound as an executable contract: a
correctly implemented mechanism must not audit *above* its claimed ε (beyond
statistical noise), and a non-private mechanism must audit far above 0.  The
final test demonstrates the payoff — the audit catches the exact class of bug
that voided DP-SGD in an earlier revision of this project (adding no noise),
which type checks and API-level unit tests do not catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dp.audit import (
    audit_scalar_mechanism,
    epsilon_lower_bound_binomial,
    epsilon_lower_bound_clopper_pearson,
)
from dp.mechanisms import add_laplace_noise


# --- binomial one-run estimator -------------------------------------------

def test_binomial_perfect_privacy_yields_zero():
    # Exactly chance-level guessing (half correct) rules out no epsilon.
    result = epsilon_lower_bound_binomial(num_correct=500, num_guesses=1000)
    assert result.epsilon_lower_bound == 0.0


def test_binomial_perfect_attack_is_large():
    # Every guess correct out of 1000 => very high confidence of large epsilon.
    result = epsilon_lower_bound_binomial(num_correct=1000, num_guesses=1000)
    assert result.epsilon_lower_bound > 3.0


def test_binomial_monotone_in_correct_guesses():
    # More correct guesses can only raise the lower bound.
    lows = [
        epsilon_lower_bound_binomial(num_correct=c, num_guesses=1000).epsilon_lower_bound
        for c in (600, 700, 800, 900)
    ]
    assert lows == sorted(lows)
    assert all(b >= a for a, b in zip(lows, lows[1:]))


def test_binomial_bound_is_sound_for_randomized_response():
    # Simulate (ε, 0)-DP randomized response — the worst case for this bound —
    # and confirm the audit almost never over-states the true epsilon.
    true_eps = 1.0
    p = np.exp(true_eps) / (np.exp(true_eps) + 1)
    rng = np.random.default_rng(0)
    r = 2000
    over = 0
    trials = 40
    for _ in range(trials):
        correct = int(rng.binomial(r, p))
        lb = epsilon_lower_bound_binomial(
            correct, r, confidence=0.95
        ).epsilon_lower_bound
        if lb > true_eps:
            over += 1
    # A 95%-confidence lower bound should exceed the truth in <~5% of runs.
    assert over <= trials * 0.15


def test_binomial_input_validation():
    with pytest.raises(ValueError, match="num_guesses"):
        epsilon_lower_bound_binomial(0, 0)
    with pytest.raises(ValueError, match="num_correct"):
        epsilon_lower_bound_binomial(11, 10)
    with pytest.raises(ValueError, match="confidence"):
        epsilon_lower_bound_binomial(5, 10, confidence=1.5)


# --- clopper-pearson multi-run estimator ----------------------------------

def test_clopper_pearson_no_leakage_is_zero():
    # Attacker at chance: FPR ~ FNR ~ 0.5 -> bound collapses to 0.
    result = epsilon_lower_bound_clopper_pearson(
        false_positives=500,
        false_negatives=500,
        num_positive_trials=1000,
        num_negative_trials=1000,
    )
    assert result.epsilon_lower_bound == 0.0


def test_clopper_pearson_strong_attack_is_positive():
    # Near-perfect attacker: very low FPR and FNR -> large epsilon.
    result = epsilon_lower_bound_clopper_pearson(
        false_positives=5,
        false_negatives=5,
        num_positive_trials=1000,
        num_negative_trials=1000,
    )
    assert result.epsilon_lower_bound > 2.0


def test_clopper_pearson_validation():
    with pytest.raises(ValueError, match="trial counts"):
        epsilon_lower_bound_clopper_pearson(0, 0, 0, 0)
    with pytest.raises(ValueError, match="false_positives"):
        epsilon_lower_bound_clopper_pearson(20, 0, 10, 10)


# --- end-to-end mechanism audits ------------------------------------------

def _laplace_scalar(epsilon):
    def mech(x, rs):
        return add_laplace_noise(
            pd.DataFrame({"a": [x]}), epsilon=epsilon, sensitivity=1.0, random_state=rs
        ).iloc[0, 0]

    return mech


def test_audit_respects_correct_laplace_claim():
    # A correctly implemented ε=1 Laplace mechanism (sensitivity 1, canary gap
    # 1) must not audit far above its claim.
    result = audit_scalar_mechanism(
        _laplace_scalar(epsilon=1.0),
        value_in=1.0,
        value_out=0.0,
        num_guesses=4000,
        confidence=0.95,
        random_state=0,
    )
    # Sound lower bound: should sit at or below the claim (with slack for noise).
    assert result.epsilon_lower_bound <= 1.5


def test_audit_flags_non_private_mechanism():
    # A "mechanism" that returns the raw value adds NO noise — the attacker
    # separates the canary perfectly and the audit must report large epsilon.
    def broken(x, rs):
        return x  # bug: no noise, no privacy

    result = audit_scalar_mechanism(
        broken, value_in=1.0, value_out=0.0, num_guesses=2000, random_state=0
    )
    assert result.epsilon_lower_bound > 5.0
    assert result.violates(epsilon_claim=1.0)


def test_audit_detects_undernoised_mechanism():
    # Claiming ε=0.5 but actually sampling at ε=8 (16x too little noise) must
    # be caught: the empirical lower bound exceeds the claim.
    claimed = 0.5
    actual = audit_scalar_mechanism(
        _laplace_scalar(epsilon=8.0),
        value_in=1.0,
        value_out=0.0,
        num_guesses=4000,
        confidence=0.95,
        random_state=1,
    )
    assert actual.violates(epsilon_claim=claimed)
