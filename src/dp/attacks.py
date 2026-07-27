"""Membership-inference attacks for measuring *practical* privacy leakage.

A theoretical ε bounds the worst case; a membership-inference attack (MIA)
measures what an adversary actually recovers from a specific trained model.
Reporting both — the ε from the accountant and the attack's success — is now
standard practice for evaluating private ML, because a model can have a loose
ε yet leak little in practice (or, if buggy, leak despite a tight ε).

The modern MIA literature (Carlini et al., "Membership Inference Attacks From
First Principles", S&P 2022) argues that *average-case* metrics like attack
accuracy or AUC are misleading: a privacy violation is the confident
identification of *some* members, so attacks must be judged by their
true-positive rate at low false-positive rate (TPR @ low FPR).  This module
implements two attacks and reports both AUC and TPR at fixed FPRs.

``loss_threshold_attack``
    The baseline of Yeom et al. (CSF 2018): members tend to have lower loss
    than non-members, so per-example loss is itself a membership score.

``lira_offline_attack``
    Offline Likelihood-Ratio Attack (Carlini et al. 2022).  Calibrates each
    example's observed loss against a Gaussian fit to the losses of *shadow*
    models trained without that example, yielding a per-example likelihood
    ratio.  The offline variant uses only "out" shadow models, which is the
    practical setting when the target's training set is unknown.

``lira_online_attack``
    Online Likelihood-Ratio Attack (Carlini et al. 2022, §IV).  The stronger
    adversary: it fits *two* Gaussians per example — one to the losses of
    shadow models trained *with* the example ("in") and one to those trained
    *without* it ("out") — and scores membership by the log-likelihood ratio
    between them.  Using the "in" distribution as well as the "out" distribution
    is what makes the online attack the field's most powerful membership test;
    the price is that the adversary must be able to train shadow models that
    contain the target example.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class AttackResult:
    """Membership-inference attack outcome.

    Attributes:
        auc: ROC-AUC of membership scores (0.5 = no leakage detected).
        tpr_at_fpr: Mapping {target_fpr: achieved_tpr} — the headline metric.
        scores: Per-example membership scores (higher = more member-like).
        membership: Ground-truth membership labels used to score the attack.
        method: Attack name.
    """

    auc: float
    tpr_at_fpr: dict[float, float]
    scores: np.ndarray
    membership: np.ndarray
    method: str


def _tpr_at_fpr(membership: np.ndarray, scores: np.ndarray, fprs) -> dict[float, float]:
    fpr, tpr, _ = roc_curve(membership, scores)
    out: dict[float, float] = {}
    for target in fprs:
        # Highest TPR achievable without exceeding the target FPR.
        allowed = fpr <= target
        out[float(target)] = float(tpr[allowed].max()) if allowed.any() else 0.0
    return out


def loss_threshold_attack(
    member_losses: np.ndarray,
    nonmember_losses: np.ndarray,
    fprs=(0.001, 0.01, 0.1),
) -> AttackResult:
    """Yeom-style loss-threshold membership inference.

    Uses ``-loss`` as the membership score (low loss ⇒ likely member) and
    scores the attack by ROC-AUC and TPR at the requested FPRs.

    Args:
        member_losses: Per-example losses for training members.
        nonmember_losses: Per-example losses for held-out non-members.
        fprs: False-positive rates at which to report TPR.

    Returns:
        An :class:`AttackResult`.

    Raises:
        ValueError: If either loss array is empty.
    """
    member_losses = np.asarray(member_losses, dtype=float)
    nonmember_losses = np.asarray(nonmember_losses, dtype=float)
    if member_losses.size == 0 or nonmember_losses.size == 0:
        raise ValueError("both member and non-member losses must be non-empty")

    scores = np.concatenate([-member_losses, -nonmember_losses])
    membership = np.concatenate(
        [np.ones(member_losses.size), np.zeros(nonmember_losses.size)]
    )
    return AttackResult(
        auc=float(roc_auc_score(membership, scores)),
        tpr_at_fpr=_tpr_at_fpr(membership, scores, fprs),
        scores=scores,
        membership=membership,
        method="loss-threshold",
    )


#: Recognised per-example statistics for the online attack.
ONLINE_VARIANTS = ("online_raw_loss", "online_logit_confidence")


def logit_confidence_from_bce(losses: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map per-example binary cross-entropy to a logit-confidence scale.

    For a binary cross-entropy ``L``, the model's probability on the *true*
    label is ``p = exp(-L)``. Carlini et al. (2022) fit their Gaussians to
    ``logit(p) = log(p / (1 - p))`` rather than to ``p`` or to the loss itself,
    because probabilities pile up against 0 and 1 and raw losses are
    heavy-tailed, while the logit scale is far closer to Gaussian and has
    roughly homogeneous per-example variance.

    ``p`` is clipped away from both endpoints, so a zero loss (``p = 1``) or a
    very large loss (``p -> 0``) yields a large finite score rather than an
    infinity. The mapping is strictly increasing in ``p`` and therefore strictly
    *decreasing* in the loss: a lower loss gives a higher logit confidence.

    Args:
        losses: Any shape. Per-example binary cross-entropy, non-negative.
        eps: Clip applied to ``p`` on both sides.

    Returns:
        An array of the same shape, finite everywhere.
    """
    losses = np.asarray(losses, dtype=float)
    p = np.clip(np.exp(-losses), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def lira_offline_attack(
    target_losses: np.ndarray,
    membership: np.ndarray,
    shadow_out_losses: np.ndarray,
    fprs=(0.001, 0.01, 0.1),
    eps: float = 1e-12,
) -> AttackResult:
    """Offline Likelihood-Ratio Attack (Carlini et al. 2022).

    For each target example we have its loss under the target model
    (``target_losses``) and a sample of its losses under shadow models that did
    *not* train on it (``shadow_out_losses[i]``).  Fitting a Gaussian to the
    "out" losses, the membership score is the left-tail probability of the
    observed loss — i.e. how surprisingly low the loss is relative to a model
    that never saw the example.  Lower-than-expected loss ⇒ likely member.

    Args:
        target_losses: Shape ``(n,)``. Loss of each example under the target
            model.
        membership: Shape ``(n,)``. 1 if the example was in the target's
            training set, else 0. Used only to score the attack.
        shadow_out_losses: Shape ``(n, k)``. For each example, ``k`` losses
            from shadow models trained without it.
        fprs: False-positive rates at which to report TPR.
        eps: Floor on the fitted standard deviation for numerical stability.

    Returns:
        An :class:`AttackResult`.

    Raises:
        ValueError: On shape mismatches or fewer than 2 shadow samples.

    Reference:
        Carlini, Chien, Nasr, Song, Terzis, Tramèr (2022). "Membership
        Inference Attacks From First Principles." IEEE S&P.
    """
    target_losses = np.asarray(target_losses, dtype=float)
    membership = np.asarray(membership)
    shadow_out_losses = np.asarray(shadow_out_losses, dtype=float)

    n = target_losses.shape[0]
    if membership.shape[0] != n or shadow_out_losses.shape[0] != n:
        raise ValueError("target_losses, membership, shadow_out_losses must align on axis 0")
    if shadow_out_losses.ndim != 2 or shadow_out_losses.shape[1] < 2:
        raise ValueError("shadow_out_losses must be (n, k) with k >= 2 shadow models")

    mu_out = shadow_out_losses.mean(axis=1)
    sigma_out = shadow_out_losses.std(axis=1, ddof=1)
    sigma_out = np.maximum(sigma_out, eps)

    # Standardised deviation; a loss far below the "out" mean is member-like.
    z = (target_losses - mu_out) / sigma_out
    scores = -z  # higher score = more member-like

    return AttackResult(
        auc=float(roc_auc_score(membership, scores)),
        tpr_at_fpr=_tpr_at_fpr(membership, scores, fprs),
        scores=scores,
        membership=membership,
        method="lira-offline",
    )


def lira_online_attack(
    target_losses: np.ndarray,
    membership: np.ndarray,
    shadow_in_losses: np.ndarray,
    shadow_out_losses: np.ndarray,
    fprs=(0.001, 0.01, 0.1),
    eps: float = 1e-12,
    variant: str = "online_raw_loss",
) -> AttackResult:
    """Online Likelihood-Ratio Attack (Carlini et al. 2022, §IV).

    The online attack fits, per example, a Gaussian to the losses of shadow
    models trained *with* the example (``shadow_in_losses[i]``) and another to
    the losses of shadow models trained *without* it (``shadow_out_losses[i]``).
    The membership score is the log-likelihood ratio of the observed target
    loss under the two fits,

        score_i = log N(loss_i; μ_in, σ_in²) − log N(loss_i; μ_out, σ_out²),

    which is the most powerful test between the two simple hypotheses (Neyman–
    Pearson).  Unlike :func:`lira_offline_attack`, which only models the "out"
    distribution, the online variant calibrates against the "in" distribution
    too, so it recovers more signal at low FPR — at the cost of requiring shadow
    models that contain the target example.

    Args:
        target_losses: Shape ``(n,)``. Loss of each example under the target
            model.
        membership: Shape ``(n,)``. 1 if the example was in the target's
            training set, else 0. Used only to score the attack.
        shadow_in_losses: Shape ``(n, k_in)``. For each example, ``k_in`` losses
            from shadow models trained *with* it.
        shadow_out_losses: Shape ``(n, k_out)``. For each example, ``k_out``
            losses from shadow models trained *without* it.
        fprs: False-positive rates at which to report TPR.
        eps: Floor on the fitted standard deviations for numerical stability.
        variant: Which per-example statistic the Gaussians are fitted to.
            ``"online_raw_loss"`` (default) preserves the original behaviour and
            fits to the raw cross-entropy losses. ``"online_logit_confidence"``
            first maps each loss through :func:`logit_confidence_from_bce`, the
            variance-stabilising transform Carlini et al. apply before fitting.
            Raw BCE is heavy-tailed and its per-example spread varies widely, so
            a likelihood ratio divided by a badly-estimated per-example sigma can
            rank acceptably on average while calibrating poorly in the extreme
            tail — which is where TPR at low FPR is read.

    Returns:
        An :class:`AttackResult`. ``method`` records which variant was used.

    Raises:
        ValueError: On shape mismatches, fewer than 2 shadow samples on either
            side, or an unknown ``variant``.

    Reference:
        Carlini, Chien, Nasr, Song, Terzis, Tramèr (2022). "Membership
        Inference Attacks From First Principles." IEEE S&P, §IV ("online" LiRA).
    """
    target_losses = np.asarray(target_losses, dtype=float)
    membership = np.asarray(membership)
    shadow_in_losses = np.asarray(shadow_in_losses, dtype=float)
    shadow_out_losses = np.asarray(shadow_out_losses, dtype=float)

    n = target_losses.shape[0]
    aligned = (
        membership.shape[0] == n
        and shadow_in_losses.shape[0] == n
        and shadow_out_losses.shape[0] == n
    )
    if not aligned:
        raise ValueError(
            "target_losses, membership, shadow_in_losses, shadow_out_losses must align on axis 0"
        )
    if shadow_in_losses.ndim != 2 or shadow_in_losses.shape[1] < 2:
        raise ValueError("shadow_in_losses must be (n, k) with k >= 2 shadow models")
    if shadow_out_losses.ndim != 2 or shadow_out_losses.shape[1] < 2:
        raise ValueError("shadow_out_losses must be (n, k) with k >= 2 shadow models")
    if variant not in ONLINE_VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(ONLINE_VARIANTS)}"
        )

    if variant == "online_logit_confidence":
        # Applied identically to target, IN and OUT so the three are comparable.
        target_losses = logit_confidence_from_bce(target_losses)
        shadow_in_losses = logit_confidence_from_bce(shadow_in_losses)
        shadow_out_losses = logit_confidence_from_bce(shadow_out_losses)

    mu_in = shadow_in_losses.mean(axis=1)
    sigma_in = np.maximum(shadow_in_losses.std(axis=1, ddof=1), eps)
    mu_out = shadow_out_losses.mean(axis=1)
    sigma_out = np.maximum(shadow_out_losses.std(axis=1, ddof=1), eps)

    # log N(x; mu, sigma) up to the shared constant, which cancels in the ratio.
    def log_gaussian(mu, sigma):
        return -np.log(sigma) - 0.5 * ((target_losses - mu) / sigma) ** 2

    scores = log_gaussian(mu_in, sigma_in) - log_gaussian(mu_out, sigma_out)

    return AttackResult(
        auc=float(roc_auc_score(membership, scores)),
        tpr_at_fpr=_tpr_at_fpr(membership, scores, fprs),
        scores=scores,
        membership=membership,
        # The raw-loss variant keeps its historical name so existing callers and
        # stored results stay valid; only the new transform gets a new label.
        method=(
            "lira-online"
            if variant == "online_raw_loss"
            else "lira-online-logit-confidence"
        ),
    )
