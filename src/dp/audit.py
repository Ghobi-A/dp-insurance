"""Empirical privacy auditing: statistical lower bounds on epsilon.

A differential-privacy *proof* is an upper bound on the privacy loss.  An
*audit* is the complementary object: a high-confidence **lower** bound on ε
derived from how well an attacker can actually distinguish neighbouring
datasets.  If a mechanism claims (ε_claim, δ)-DP but an audit reports a lower
bound ε_lb > ε_claim, the claim is provably violated — the implementation has
a bug, or the analysis is wrong.  This turns "is this code actually private?"
into a testable question (see ``tests/test_audit.py``).

Two estimators are provided:

``epsilon_lower_bound_binomial``
    The one-run auditor of Steinke, Nasr & Jagielski, "Privacy Auditing with
    One (1) Training Run" (NeurIPS 2023).  Given ``r`` guesses of which ``v``
    are correct, the number of correct guesses is stochastically dominated by
    ``Binomial(r, e^ε/(e^ε+1))`` (up to an O(δ) term), so the largest ε whose
    upper-tail p-value stays below β is a (1−β)-confidence lower bound.

``epsilon_lower_bound_clopper_pearson``
    The multi-run auditor of Jagielski, Ullman & Oprea, "Auditing Differentially
    Private Machine Learning: How Private is Private SGD?" (NeurIPS 2020).  From
    many independent trials it estimates the attacker's false-positive and
    false-negative rates, brackets each with a Clopper–Pearson interval, and
    inverts the (ε, δ)-DP privacy region to obtain ε_lb.

Both audits are *sound*: they can only certify leakage that is actually
present.  They never over-claim privacy, so a loose audit (ε_lb far below
ε_claim) means "not detected", not "proven safe".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize, stats

from .canaries import CanarySet


@dataclass(frozen=True)
class AuditResult:
    """Outcome of an empirical privacy audit.

    Attributes:
        epsilon_lower_bound: High-confidence lower bound on the true ε.
        confidence: 1 − β, the confidence level of the bound.
        num_guesses: Number of decisions the attacker committed to (r).
        num_correct: Number of correct decisions (v).
        method: Which estimator produced the bound.
        details: Estimator-specific diagnostics (rates, intervals, ...).
    """

    epsilon_lower_bound: float
    confidence: float
    num_guesses: int
    num_correct: int
    method: str
    details: dict

    def violates(self, epsilon_claim: float) -> bool:
        """True if the audit's lower bound exceeds a claimed ε (with margin 0)."""
        return self.epsilon_lower_bound > epsilon_claim


def epsilon_lower_bound_binomial(
    num_correct: int,
    num_guesses: int,
    delta: float = 0.0,
    confidence: float = 0.95,
) -> AuditResult:
    """One-run ε lower bound from correct-guess counts (Steinke et al. 2023).

    The auditor inserts ``num_guesses`` independent canaries, each randomly
    included or excluded, and after training guesses each canary's membership.
    Under (ε, δ)-DP the number of correct guesses ``v`` satisfies

        P[correct ≥ v] ≤ P[Binomial(r, e^ε/(e^ε+1)) ≥ v] + O(δ)

    (Theorem 5.2).  We ignore the O(δ) term for δ = 0 (the pure-DP case this
    project audits) and, for δ > 0, apply the standard conservative correction
    of crediting the attacker with up to ``r · δ`` "free" correct guesses,
    which never over-states the bound.  The reported ε_lb is the largest ε
    whose upper-tail p-value is still ≤ β = 1 − ``confidence``; since that tail
    is monotincreasing in ε, the root is unique and found by bisection.

    Args:
        num_correct: Number of correct membership guesses (v), 0 ≤ v ≤ r.
        num_guesses: Number of guesses committed to (r > 0). Abstentions
            (T_i = 0) should be excluded from this count.
        delta: The δ of the (ε, δ)-DP claim being audited. 0 for pure DP.
        confidence: Confidence level 1 − β for the lower bound.

    Returns:
        An :class:`AuditResult`. ``epsilon_lower_bound`` is 0.0 when the data
        are consistent with perfect privacy (no ε can be ruled out).

    Raises:
        ValueError: If inputs are out of range.
    """
    if num_guesses <= 0:
        raise ValueError("num_guesses must be positive")
    if not (0 <= num_correct <= num_guesses):
        raise ValueError("num_correct must satisfy 0 <= num_correct <= num_guesses")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    if not (0 <= delta < 1):
        raise ValueError("delta must be in [0, 1)")

    beta = 1.0 - confidence
    # Conservative δ correction: discount correct guesses the attacker could
    # have made "for free" on the δ-failure event (Steinke et al., §5).
    effective_correct = num_correct - delta * num_guesses

    def tail_pvalue(epsilon: float) -> float:
        p = np.exp(epsilon) / (np.exp(epsilon) + 1.0)
        # P[Binomial(r, p) >= v] = sf(v - 1); use ceil for the discounted count.
        v = int(np.ceil(effective_correct))
        return float(stats.binom.sf(v - 1, num_guesses, p))

    # If even ε → 0 (p = 0.5) is not surprising, nothing can be ruled out.
    if tail_pvalue(0.0) > beta:
        epsilon_lb = 0.0
    else:
        # tail_pvalue is increasing in epsilon; find where it crosses beta.
        hi = 1.0
        while tail_pvalue(hi) <= beta:
            hi *= 2.0
            if hi > 1e6:  # numerical ceiling; effectively "unbounded leakage"
                break
        epsilon_lb = optimize.brentq(lambda e: tail_pvalue(e) - beta, 0.0, hi)

    return AuditResult(
        epsilon_lower_bound=epsilon_lb,
        confidence=confidence,
        num_guesses=num_guesses,
        num_correct=num_correct,
        method="binomial-one-run",
        details={"delta": delta, "accuracy": num_correct / num_guesses},
    )


def _clopper_pearson(count: int, total: int, alpha: float) -> tuple[float, float]:
    """Two-sided Clopper–Pearson interval for a binomial proportion."""
    lo = 0.0 if count == 0 else stats.beta.ppf(alpha / 2, count, total - count + 1)
    hi = 1.0 if count == total else stats.beta.ppf(1 - alpha / 2, count + 1, total - count)
    return float(lo), float(hi)


def epsilon_lower_bound_clopper_pearson(
    false_positives: int,
    false_negatives: int,
    num_positive_trials: int,
    num_negative_trials: int,
    delta: float = 0.0,
    confidence: float = 0.95,
) -> AuditResult:
    """Multi-run ε lower bound via Clopper–Pearson (Jagielski et al. 2020).

    An attacker runs a membership test across many trials on neighbouring
    datasets, producing empirical false-positive rate (FPR) and false-negative
    rate (FNR).  Each rate is bracketed by a Clopper–Pearson interval at the
    given confidence, and the (ε, δ)-DP privacy region is inverted using the
    *upper* confidence limits (the attacker-favourable end):

        ε_lb = max( log((1 − δ − FPR⁺)/FNR⁺),  log((1 − δ − FNR⁺)/FPR⁺) )

    where FPR⁺, FNR⁺ are the upper Clopper–Pearson limits.  Non-positive or
    undefined arguments to the logarithm collapse to a 0 contribution, so
    ε_lb ≥ 0 always.

    Args:
        false_positives: Trials where an excluded canary was guessed "in".
        false_negatives: Trials where an included canary was guessed "out".
        num_positive_trials: Trials where the canary was included (denominator
            of FNR).
        num_negative_trials: Trials where the canary was excluded (denominator
            of FPR).
        delta: δ of the (ε, δ)-DP claim being audited.
        confidence: Confidence level 1 − β for the interval (two-sided).

    Returns:
        An :class:`AuditResult` summarising the bound and the rate intervals.

    Raises:
        ValueError: If counts are out of range.
    """
    if num_positive_trials <= 0 or num_negative_trials <= 0:
        raise ValueError("trial counts must be positive")
    if not (0 <= false_positives <= num_negative_trials):
        raise ValueError("false_positives out of range")
    if not (0 <= false_negatives <= num_positive_trials):
        raise ValueError("false_negatives out of range")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")

    alpha = 1.0 - confidence
    _, fpr_hi = _clopper_pearson(false_positives, num_negative_trials, alpha)
    _, fnr_hi = _clopper_pearson(false_negatives, num_positive_trials, alpha)

    def region(a: float, b: float) -> float:
        # log((1 - delta - a) / b), guarded to a 0 contribution when invalid.
        num = 1.0 - delta - a
        if b <= 0.0 or num <= 0.0:
            return 0.0
        return float(np.log(num / b))

    epsilon_lb = max(0.0, region(fpr_hi, fnr_hi), region(fnr_hi, fpr_hi))

    total = num_positive_trials + num_negative_trials
    num_correct = (num_positive_trials - false_negatives) + (
        num_negative_trials - false_positives
    )
    return AuditResult(
        epsilon_lower_bound=epsilon_lb,
        confidence=confidence,
        num_guesses=total,
        num_correct=num_correct,
        method="clopper-pearson-multi-run",
        details={
            "fpr_upper": fpr_hi,
            "fnr_upper": fnr_hi,
            "delta": delta,
        },
    )


def audit_membership_scores(
    scores: np.ndarray,
    included: np.ndarray,
    delta: float = 0.0,
    confidence: float = 0.95,
    guess_fraction: float = 1.0,
) -> AuditResult:
    """One-run ε lower bound from real-valued membership scores.

    This generalises :func:`audit_scalar_mechanism` from a scalar sampler to any
    attack that emits a per-canary membership *score* (higher = more member-like)
    against known inclusion labels — for instance the loss of each canary under
    a single trained model, or a LiRA statistic.  It is the auditing side of the
    membership-inference attacks in :mod:`dp.attacks`: strengthen the attack, run
    it against canaries whose inclusion you control, and this turns the result
    into a lower bound on ε.

    Following Steinke et al. (2023), the auditor commits a hard guess only where
    it is most confident: it sorts by score, guesses "in" for the top ``k`` and
    "out" for the bottom ``k`` (``k = ⌊guess_fraction · m / 2⌋``, abstaining on
    the middle), counts correct guesses, and feeds ``(correct, 2k)`` to
    :func:`epsilon_lower_bound_binomial`.  The guessing rule depends only on the
    scores, never on the inclusion labels, so the binomial bound applies.

    Args:
        scores: Shape ``(m,)``. Membership score per canary (higher = more
            member-like).
        included: Shape ``(m,)``. 1 if the canary was in training, else 0.
        delta: δ of the (ε, δ)-DP claim being audited.
        confidence: Confidence level 1 − β for the lower bound.
        guess_fraction: Fraction of canaries to commit guesses on, split evenly
            between the most and least member-like.  1.0 guesses on all of them;
            a smaller value trades coverage for a more confident subset.

    Returns:
        An :class:`AuditResult` from the binomial one-run estimator, with the
        realised guessing budget and attacker accuracy in ``details``.

    Raises:
        ValueError: On shape mismatch, non-binary ``included``, a
            ``guess_fraction`` outside ``(0, 1]``, or too few canaries to commit
            even a single in/out guess pair.
    """
    scores = np.asarray(scores, dtype=float)
    included = np.asarray(included)
    if scores.shape != included.shape or scores.ndim != 1:
        raise ValueError("scores and included must be 1-D arrays of the same length")
    if not set(np.unique(included)).issubset({0, 1}):
        raise ValueError("included must be binary (0/1)")
    if not (0.0 < guess_fraction <= 1.0):
        raise ValueError("guess_fraction must be in (0, 1]")

    m = scores.shape[0]
    k = int(np.floor(guess_fraction * m / 2.0))
    if k < 1:
        raise ValueError("too few canaries to commit an in/out guess pair")

    order = np.argsort(scores)  # ascending: low score = out-like, high = in-like
    out_idx = order[:k]
    in_idx = order[m - k:]
    correct = int((included[in_idx] == 1).sum() + (included[out_idx] == 0).sum())

    result = epsilon_lower_bound_binomial(
        num_correct=correct,
        num_guesses=2 * k,
        delta=delta,
        confidence=confidence,
    )
    result.details["guess_fraction"] = guess_fraction
    result.details["num_canaries"] = m
    return result


def audit_scalar_mechanism(
    mechanism: Callable[[float, int | None], float],
    value_in: float = 1.0,
    value_out: float = 0.0,
    num_guesses: int = 2000,
    delta: float = 0.0,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> AuditResult:
    """Audit a 1-D additive-noise mechanism by membership inference on a canary.

    Builds ``num_guesses`` independent instances, each drawing the mechanism's
    output on either ``value_in`` or ``value_out`` (chosen by a fair coin), and
    guesses the input by thresholding at the midpoint.  The correct-guess count
    feeds :func:`epsilon_lower_bound_binomial`.  This is the smallest possible
    end-to-end audit: it exercises a real sampler rather than a formula, so it
    catches implementation bugs (wrong scale, missing noise) that unit tests on
    the intended API would miss.

    ``mechanism`` must accept ``(value, random_state)`` and return a noised
    scalar.  For example, to audit the Laplace mechanism at its claimed ε::

        audit_scalar_mechanism(
            lambda x, rs: add_laplace_noise(
                pd.DataFrame({"a": [x]}), epsilon=claim, sensitivity=1.0,
                random_state=rs,
            ).iloc[0, 0],
        )

    Args:
        mechanism: Callable ``(value, random_state) -> float``.
        value_in: Canary-present input (differs from ``value_out`` by the
            sensitivity the mechanism was calibrated for).
        value_out: Canary-absent input.
        num_guesses: Number of independent trials.
        delta: δ of the claim being audited.
        confidence: Confidence level for the lower bound.
        random_state: Seed for the inclusion coins and per-trial mechanism
            seeds (full reproducibility).

    Returns:
        An :class:`AuditResult` from the binomial one-run estimator.
    """
    rng = np.random.default_rng(random_state)
    included = rng.integers(0, 2, size=num_guesses)  # 1 = in, 0 = out
    seeds = rng.integers(0, 2**31 - 1, size=num_guesses)
    threshold = 0.5 * (value_in + value_out)

    correct = 0
    for i in range(num_guesses):
        x = value_in if included[i] else value_out
        out = mechanism(x, int(seeds[i]))
        guess_in = out >= threshold if value_in >= value_out else out <= threshold
        if int(guess_in) == included[i]:
            correct += 1

    return epsilon_lower_bound_binomial(
        num_correct=correct,
        num_guesses=num_guesses,
        delta=delta,
        confidence=confidence,
    )


def one_run_model_audit(
    canaries: CanarySet,
    base_features: np.ndarray,
    base_labels: np.ndarray,
    train_and_score: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    delta: float = 0.0,
    confidence: float = 0.95,
    guess_fraction: float = 1.0,
    random_state: int | None = None,
) -> AuditResult:
    """Audit a trained model in a single run using membership canaries.

    This is the model-level counterpart of :func:`audit_scalar_mechanism`: it
    realises the one-run auditor of Steinke, Nasr & Jagielski (2023) against a
    real learner and a real membership attack.  Each canary is independently
    included (fair coin); included canaries are appended to the base training
    set (repeated ``canaries.duplication`` times), the caller's
    ``train_and_score`` trains one model on the augmented data and returns a
    membership score for every canary, and :func:`audit_membership_scores`
    inverts the score/label agreement into an ε lower bound.

    Keeping training and scoring in the ``train_and_score`` callback makes the
    audit agnostic to the mechanism: pass a DP-SGD trainer to audit DP-SGD, or a
    plain learner to check that the audit has power where memorisation is real.
    Because the auditor only sees scores, it cannot over-state privacy — a loose
    bound means "not detected at this power", not "proven private".

    Args:
        canaries: The pool of canaries to include/exclude and score.
        base_features: Real training features the canaries are added to.
        base_labels: Labels aligned with ``base_features``.
        train_and_score: Callable ``(X_aug, y_aug, canary_features) -> scores``
            that trains one model on the augmented data and returns a membership
            score per canary (higher = more member-like).
        delta: δ of the (ε, δ)-DP claim being audited.
        confidence: Confidence level 1 − β for the lower bound.
        guess_fraction: Passed through to :func:`audit_membership_scores`.
        random_state: Seed for the inclusion coins.

    Returns:
        An :class:`AuditResult`; ``details`` also carries the realised
        ``included`` mask and the canary ``kind``/``duplication`` for reporting.
    """
    base_features = np.asarray(base_features, dtype=float)
    base_labels = np.asarray(base_labels)
    rng = np.random.default_rng(random_state)
    included = rng.integers(0, 2, size=len(canaries))

    in_features = canaries.features[included == 1]
    in_labels = canaries.labels[included == 1]
    if canaries.duplication > 1 and in_features.shape[0] > 0:
        in_features = np.repeat(in_features, canaries.duplication, axis=0)
        in_labels = np.repeat(in_labels, canaries.duplication)

    aug_features = np.vstack([base_features, in_features])
    aug_labels = np.concatenate([base_labels, in_labels])

    scores = np.asarray(train_and_score(aug_features, aug_labels, canaries.features), dtype=float)
    if scores.shape[0] != len(canaries):
        raise ValueError("train_and_score must return one score per canary")

    result = audit_membership_scores(
        scores,
        included,
        delta=delta,
        confidence=confidence,
        guess_fraction=guess_fraction,
    )
    result.details["canary_kind"] = canaries.kind
    result.details["duplication"] = canaries.duplication
    result.details["included_fraction"] = float(included.mean())
    return result
