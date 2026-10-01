"""Conservative paired binary yield intervals, supplementary to frozen t summary.

Bonferroni combines exact binomial intervals for discordant outcome probabilities.
This concerns independent RNG/seed cases on a fixed archive/environment split,
not independent states and not validation-AUC coverage.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta


def binomial_interval(k, n, alpha=.05):
    if not isinstance(k, int) or not isinstance(n, int) or not 0 <= k <= n or n < 1 or not 0 < alpha < 1:
        raise ValueError('invalid binomial count/confidence')
    lower = float(beta.ppf(alpha / 2, k, n-k+1)) if k else 0.
    upper = float(beta.ppf(1-alpha / 2, k+1, n-k)) if k < n else 1.
    return lower, upper


def exact_paired_yield(a, b, alpha=.05):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('aligned nonempty binary outcomes required')
    n = len(a)
    wins = int(((a == 1) & (b == 0)).sum())
    losses = int(((a == 0) & (b == 1)).sum())
    win_lo, win_hi = binomial_interval(wins, n, alpha/2)
    loss_lo, loss_hi = binomial_interval(losses, n, alpha/2)
    return {'n_seed_units': n, 'discordant_wins': wins, 'discordant_losses': losses,
            'difference': (wins-losses)/n, 'ci95_lower': win_lo-loss_hi,
            'ci95_upper': win_hi-loss_lo, 'alpha': alpha,
            'method': 'Bonferroni difference of Clopper-Pearson discordance intervals',
            'scope': 'independent binary seed cases conditional on fixed archives/environments; supplementary analysis'}
