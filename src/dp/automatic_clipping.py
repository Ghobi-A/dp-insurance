"""Published AUTO-S all-parameter gradient normalization, not a new algorithm."""
from __future__ import annotations

import math


def normalize_grad_samples(parameters, stability=0.01):
    """Apply g_i/(||g_i||_2+gamma) across all parameters before DP aggregation.

    Caller retains Opacus unit sensitivity, Gaussian noise and its accountant.
    This deliberately supports one backward call per optimizer step only.
    """
    import torch

    if not math.isfinite(stability) or stability <= 0:
        raise ValueError("positive finite stability required")
    parameters = [p for p in parameters if p.requires_grad]
    samples = [p.grad_sample for p in parameters]
    if not samples or any(not isinstance(g, torch.Tensor) for g in samples):
        raise ValueError("one tensor grad_sample per trainable parameter required")
    n = len(samples[0])
    if any(len(g) != n for g in samples):
        raise ValueError("inconsistent sample dimensions")
    if n == 0:
        return
    norm = torch.stack([g.reshape(n, -1).norm(2, dim=1) for g in samples], dim=1).norm(2, dim=1)
    if not torch.isfinite(norm).all():
        raise ValueError("nonfinite per-sample gradient")
    factor = 1 / (norm + stability)
    for g in samples:
        g.mul_(factor.reshape((n,) + (1,) * (g.ndim - 1)))
