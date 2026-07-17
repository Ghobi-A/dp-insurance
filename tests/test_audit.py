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
from sklearn.tree import DecisionTreeClassifier

from dp.audit import (
    audit_membership_scores,
    audit_scalar_mechanism,
    epsilon_lower_bound_binomial,
    epsilon_lower_bound_clopper_pearson,
    one_run_model_audit,
)
from dp.canaries import make_label_flip_canaries
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


# --- one-run auditor from membership scores -------------------------------

def test_audit_membership_scores_perfect_separation_is_large():
    # A score that ranks every member above every non-member => all guesses
    # correct => a large lower bound, just like the scalar no-noise audit.
    rng = np.random.default_rng(0)
    included = rng.integers(0, 2, size=1000)
    scores = included + rng.normal(0, 1e-3, size=1000)  # perfectly ordered
    result = audit_membership_scores(scores, included)
    assert result.epsilon_lower_bound > 2.5


def test_audit_membership_scores_chance_is_zero():
    # Scores independent of membership => attacker at chance => bound is 0.
    rng = np.random.default_rng(1)
    included = rng.integers(0, 2, size=2000)
    scores = rng.normal(0, 1, size=2000)
    result = audit_membership_scores(scores, included)
    assert result.epsilon_lower_bound == 0.0


def test_audit_membership_scores_abstention_reports_budget():
    # guess_fraction commits guesses on the most confident subset only.
    rng = np.random.default_rng(2)
    included = rng.integers(0, 2, size=1000)
    scores = included + rng.normal(0, 1e-3, size=1000)
    result = audit_membership_scores(scores, included, guess_fraction=0.4)
    assert result.details["guess_fraction"] == 0.4
    assert result.num_guesses == 2 * int(0.4 * 1000 / 2)


def test_audit_membership_scores_validation():
    with pytest.raises(ValueError, match="same length"):
        audit_membership_scores(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError, match="binary"):
        audit_membership_scores(np.zeros(5), np.full(5, 2))
    with pytest.raises(ValueError, match="guess_fraction"):
        audit_membership_scores(np.zeros(6), np.zeros(6), guess_fraction=0.0)
    with pytest.raises(ValueError, match="too few"):
        audit_membership_scores(np.array([0.0]), np.array([1]))


# --- one-run model audit: the strengthened privacy regression test --------

def _blobs(n, d, sep, seed):
    """Two well-separated Gaussian blobs; a learner predicts the true label."""
    rng = np.random.default_rng(seed)
    half = n // 2
    features = np.vstack([
        rng.normal(-sep, 1.0, size=(half, d)),
        rng.normal(sep, 1.0, size=(half, d)),
    ])
    labels = np.concatenate([np.zeros(half), np.ones(half)]).astype(int)
    return features, labels


def _bce(y_true, prob):
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return -(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))


def test_one_run_audit_catches_memorising_learner():
    # The strengthened auditor, run against a NON-private learner that
    # memorises mislabelled canaries, must certify a large ε lower bound — the
    # model-level analogue of the scalar no-noise regression test. A private
    # model that stopped memorising would drive this bound to ~0, so the
    # assertion is a live privacy contract, not a tautology.
    base_x, base_y = _blobs(600, 8, sep=3.0, seed=1)
    pool_x, pool_y = _blobs(400, 8, sep=3.0, seed=101)
    canaries = make_label_flip_canaries(pool_x, pool_y, 300, random_state=1)

    def train_and_score(x_aug, y_aug, canary_features):
        tree = DecisionTreeClassifier(random_state=0).fit(x_aug, y_aug)
        return -_bce(canaries.labels, tree.predict_proba(canary_features)[:, 1])

    result = one_run_model_audit(
        canaries, base_x, base_y, train_and_score,
        delta=0.0, guess_fraction=0.5, random_state=1,
    )
    assert result.epsilon_lower_bound > 1.5
    assert result.violates(epsilon_claim=1.0)
    assert result.details["canary_kind"] == "label-flip"


def test_one_run_audit_chance_attack_is_zero():
    # The SAME canaries and base data, but a scorer that ignores membership,
    # must not manufacture leakage: a sound audit reports ~0.
    base_x, base_y = _blobs(600, 8, sep=3.0, seed=1)
    pool_x, pool_y = _blobs(400, 8, sep=3.0, seed=101)
    canaries = make_label_flip_canaries(pool_x, pool_y, 300, random_state=1)
    noise = np.random.default_rng(7)

    def train_and_score(x_aug, y_aug, canary_features):
        return noise.standard_normal(canary_features.shape[0])

    result = one_run_model_audit(
        canaries, base_x, base_y, train_and_score,
        delta=0.0, guess_fraction=0.5, random_state=1,
    )
    assert result.epsilon_lower_bound == 0.0


def test_one_run_model_audit_validates_score_length():
    base_x, base_y = _blobs(200, 4, sep=2.0, seed=0)
    pool_x, pool_y = _blobs(100, 4, sep=2.0, seed=9)
    canaries = make_label_flip_canaries(pool_x, pool_y, 40, random_state=0)
    with pytest.raises(ValueError, match="one score per canary"):
        one_run_model_audit(
            canaries, base_x, base_y,
            lambda x_aug, y_aug, cf: np.zeros(cf.shape[0] - 1),
            random_state=0,
        )
