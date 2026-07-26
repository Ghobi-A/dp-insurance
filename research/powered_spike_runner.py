"""Run the corrected spike with balanced fixed-size shadow coverage.

The main spike harness already matches target/shadow training size. This runner
replaces its independent shadow-set sampling with an evenly spaced cyclic design
so every attack example receives a usable number of OUT observations without
changing the training size, privacy target, accountant or model recipe.
"""

from __future__ import annotations

import numpy as np

import two_config_spike as spike


def balanced_shadow_train_sets(
    y_pool: np.ndarray,
    train_size: int,
    num_shadows: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Create exact-size shadow sets with near-uniform OUT coverage."""
    n = len(y_pool)
    if train_size >= n:
        raise ValueError("shadow train_size must be smaller than the shadow pool")
    if num_shadows < 2:
        raise ValueError("num_shadows must be at least two")

    exclude_size = n - train_size
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    excluded_counts = np.zeros(n, dtype=int)
    train_sets: list[np.ndarray] = []

    for shadow_id in range(num_shadows):
        start = (shadow_id * n) // num_shadows
        positions = (start + np.arange(exclude_size)) % n
        excluded_idx = permutation[positions]
        excluded_counts[excluded_idx] += 1

        in_mask = np.ones(n, dtype=bool)
        in_mask[excluded_idx] = False
        train_idx = np.flatnonzero(in_mask)
        if len(train_idx) != train_size:
            raise AssertionError("balanced schedule changed shadow train size")
        if np.unique(y_pool[train_idx]).size != 2:
            raise RuntimeError("balanced schedule produced a one-class shadow set")
        train_sets.append(train_idx)

    return train_sets, excluded_counts


def collect_shadow_out_losses_balanced(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    attack_idx: np.ndarray,
    target_train_size: int,
    target_noise_multiplier: float,
    recipe: spike.Recipe,
    num_shadows: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float | int | str]], int]:
    """Train size-matched shadows with balanced OUT observations."""
    out_losses: list[list[float]] = [[] for _ in range(len(attack_idx))]
    metadata: list[dict[str, float | int | str]] = []
    train_sets, excluded_counts = balanced_shadow_train_sets(
        y_pool,
        train_size=target_train_size,
        num_shadows=num_shadows,
        seed=seed,
    )

    for shadow_id, train_idx in enumerate(train_sets):
        in_mask = np.zeros(len(y_pool), dtype=bool)
        in_mask[train_idx] = True

        model, meta = spike.train_private_model(
            X_pool[train_idx],
            y_pool[train_idx],
            recipe,
            seed=seed + shadow_id + 1,
        )
        if int(meta["train_size"]) != target_train_size:
            raise AssertionError("shadow and target train sizes differ")
        if not np.isclose(
            float(meta["noise_multiplier"]),
            target_noise_multiplier,
            rtol=0.0,
            atol=1e-10,
        ):
            raise AssertionError(
                "matched shadow has different noise multiplier: "
                f"target={target_noise_multiplier}, shadow={meta['noise_multiplier']}"
            )

        meta["shadow_id"] = shadow_id
        metadata.append(meta)
        losses = spike.per_example_bce(
            y_pool[attack_idx],
            spike.predict_probability(model, X_pool[attack_idx]),
        )
        attack_out = ~in_mask[attack_idx]
        for local_idx in np.flatnonzero(attack_out):
            out_losses[local_idx].append(float(losses[local_idx]))

        print(
            f"[{recipe.name}] shadow {shadow_id + 1}/{num_shadows} "
            f"n={meta['train_size']} achieved_eps={meta['achieved_epsilon']:.4f} "
            f"noise={meta['noise_multiplier']:.4f}",
            flush=True,
        )

    expected_min = int(excluded_counts[attack_idx].min())
    min_out = min(len(values) for values in out_losses)
    if min_out != expected_min:
        raise AssertionError(
            f"recorded OUT losses ({min_out}) do not match schedule ({expected_min})"
        )
    if min_out < 8:
        raise RuntimeError(
            f"LiRA requires at least eight OUT observations; minimum was {min_out}"
        )

    matrix = np.vstack(
        [np.asarray(values[:min_out], dtype=float) for values in out_losses]
    )
    return matrix, metadata, min_out


if __name__ == "__main__":
    spike.collect_shadow_out_losses = collect_shadow_out_losses_balanced
    spike.main()
