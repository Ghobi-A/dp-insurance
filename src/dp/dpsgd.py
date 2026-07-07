"""DP-SGD utilities using Opacus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:  # pragma: no cover - optional dependency
    import torch
    from opacus import PrivacyEngine
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    PrivacyEngine = None


@dataclass(frozen=True)
class DPSGDSetup:
    model: object
    optimizer: object
    data_loader: object
    privacy_engine: object


def opacus_available() -> bool:
    """Return True if Opacus and torch are installed."""
    return torch is not None and PrivacyEngine is not None


def make_private_with_epsilon(
    model: object,
    optimizer: object,
    data_loader: object,
    epochs: int,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
) -> DPSGDSetup:
    """Wrap a model/optimizer/data loader with Opacus for DP-SGD.

    Raises:
        RuntimeError: If Opacus/torch are not available.
    """
    if not opacus_available():
        raise RuntimeError("Opacus and torch are required for DP-SGD utilities.")
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    return DPSGDSetup(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        privacy_engine=privacy_engine,
    )


def epsilon_for_noise_multiplier(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    target_delta: float,
) -> float:
    """Compute epsilon for a given noise multiplier using RDP accounting.

    Uses Opacus's :class:`RDPAccountant` directly, replaying ``steps``
    Poisson-subsampled Gaussian mechanism steps at the given
    ``noise_multiplier`` and ``sample_rate``, then converting to (ε, δ)-DP.
    """
    if not opacus_available():
        raise RuntimeError("Opacus and torch are required for DP-SGD utilities.")
    from opacus.accountants import RDPAccountant

    accountant = RDPAccountant()
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    return accountant.get_epsilon(delta=target_delta)


def train_dp_sgd(
    model: object,
    optimizer: object,
    data_loader: Iterable,
    loss_fn: object,
    epochs: int,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
) -> DPSGDSetup:
    """Train a model with DP-SGD, returning the private setup."""
    setup = make_private_with_epsilon(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    # Step the *wrapped* optimizer: stepping the original one would update the
    # weights without per-sample clipping or noise, voiding the DP guarantee.
    for _ in range(epochs):
        for features, labels in setup.data_loader:
            setup.optimizer.zero_grad()
            outputs = setup.model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            setup.optimizer.step()
    return setup
