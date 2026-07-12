"""Strengthened-adversary stress test of the DP-SGD ε ≈ 2.13 claim.

This standalone, reproducible experiment tries to *break* the project's headline
privacy claim rather than re-confirm it.  It exercises the library added for the
stress test (``dp.attacks.lira_online_attack``, ``dp.canaries``,
``dp.audit.audit_membership_scores`` / ``one_run_model_audit``) and prints every
number quoted in ``reports/empirical_privacy_report.md`` §5.  Run headless with::

    python notebooks/strengthened_adversary.py

It is deliberately *not* wired into the notebook CI job (each DP-SGD audit is a
full training run); the library it drives is covered by the pytest suite, which
does run in CI.  All randomness is seeded, so the printed numbers are stable.

Three parts:

1. **Stronger LiRA.**  More shadow models (32→128), online vs offline
   likelihood ratios, and *architecture mismatch* (the adversary does not know
   the target's model class), scored by TPR at low FPR (Carlini et al. 2022).
2. **New canaries + one-run DP-SGD audit.**  Label-flip, feature-space outlier,
   and duplicated (group) canaries fed through the Steinke et al. (2023) one-run
   auditor against a properly-noised DP-SGD model and a no-noise ("broken") one.
3. **Audit power check.**  The same auditor on an unbounded tree that genuinely
   memorises, to show the machinery *does* certify leakage where it exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from dp.attacks import lira_offline_attack, lira_online_attack, loss_threshold_attack
from dp.audit import one_run_model_audit
from dp.canaries import (
    CanarySet,
    make_duplicated_canaries,
    make_feature_outlier_canaries,
    make_label_flip_canaries,
)
from dp.constants import RANDOM_STATE
from dp.pipeline import build_preprocessor, split_dataset

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from opacus import PrivacyEngine

    TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH = False


REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIM_EPS = 2.13
DELTA = 1e-5


def per_example_bce(y_true: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Numerically stable per-example binary cross-entropy loss."""
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return -(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))


def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess the insurance dataset exactly as the notebook does."""
    df = pd.read_csv(REPO_ROOT / "data" / "insurance.csv")
    split = split_dataset(df, target="smoker")
    pre = build_preprocessor(split.X_train)
    x_train = pre.fit_transform(split.X_train)
    x_test = pre.transform(split.X_test)
    y_train = (split.y_train == "yes").astype(int).to_numpy()
    y_test = (split.y_test == "yes").astype(int).to_numpy()
    return x_train, x_test, y_train, y_test


def _banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# Part 1 — stronger LiRA on the natural task.
# ---------------------------------------------------------------------------
def _collect_shadow_losses(builder, x_all, y_all, num_shadows, seed):
    """Return per-example lists of IN and OUT shadow losses over `num_shadows`."""
    rng = np.random.default_rng(seed)
    n = x_all.shape[0]
    in_losses: list[list[float]] = [[] for _ in range(n)]
    out_losses: list[list[float]] = [[] for _ in range(n)]
    for k in range(num_shadows):
        in_mask = rng.random(n) < 0.5
        shadow = builder(k).fit(x_all[in_mask], y_all[in_mask])
        losses = per_example_bce(y_all, shadow.predict_proba(x_all)[:, 1])
        for i in range(n):
            (in_losses if in_mask[i] else out_losses)[i].append(float(losses[i]))
    return in_losses, out_losses


def _rectangular(list_of_lists) -> np.ndarray:
    width = int(min(len(row) for row in list_of_lists))
    return np.vstack([np.asarray(row)[:width] for row in list_of_lists])


def run_lira(x_train, x_test, y_train, y_test) -> None:
    _banner("PART 1 — stronger LiRA on the natural task (target: unbounded tree)")
    x_all = np.vstack([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    membership = np.zeros(x_all.shape[0], dtype=int)
    membership[: len(y_train)] = 1

    target = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(x_train, y_train)
    target_losses = per_example_bce(y_all, target.predict_proba(x_all)[:, 1])

    base = loss_threshold_attack(target_losses[membership == 1], target_losses[membership == 0])
    print(f"loss-threshold baseline        AUC={base.auc:.3f}  "
          f"TPR@1%={base.tpr_at_fpr[0.01]:.3f}  TPR@0.1%={base.tpr_at_fpr[0.001]:.3f}")

    architectures = {
        "tree (matched)": lambda k: DecisionTreeClassifier(random_state=k),
        "shallow tree (mismatch)": lambda k: DecisionTreeClassifier(max_depth=3, random_state=k),
        "logistic reg (mismatch)": lambda k: LogisticRegression(max_iter=1000, random_state=k),
    }
    for arch, builder in architectures.items():
        for num_shadows in (32, 64, 128):
            in_losses, out_losses = _collect_shadow_losses(
                builder, x_all, y_all, num_shadows, RANDOM_STATE
            )
            out_mat = _rectangular(out_losses)
            off = lira_offline_attack(target_losses, membership, out_mat)
            msg = (f"[{arch:24s} K={num_shadows:3d}] offline AUC={off.auc:.3f} "
                   f"TPR@1%={off.tpr_at_fpr[0.01]:.3f} TPR@0.1%={off.tpr_at_fpr[0.001]:.3f}")
            if min(len(row) for row in in_losses) >= 2:
                on = lira_online_attack(
                    target_losses, membership, _rectangular(in_losses), out_mat
                )
                msg += (f" | online AUC={on.auc:.3f} "
                        f"TPR@1%={on.tpr_at_fpr[0.01]:.3f} TPR@0.1%={on.tpr_at_fpr[0.001]:.3f}")
            print(msg)


# ---------------------------------------------------------------------------
# Part 2 — one-run DP-SGD audit against each canary construction.
# ---------------------------------------------------------------------------
def _dpsgd_scorer(canary_labels, noise_multiplier, epochs=15, private=True):
    """Build a `train_and_score` callback for `one_run_model_audit`.

    Trains a small MLP (matching the notebook's DP-SGD network) on the
    canary-augmented data and scores each canary by −loss (low loss ⇒ member).
    The accountant ε for the run is stashed in the returned `state` dict.
    """
    state: dict = {}

    def train_and_score(x_aug, y_aug, canary_features):
        torch.manual_seed(RANDOM_STATE)
        x_t = torch.tensor(x_aug, dtype=torch.float32)
        y_t = torch.tensor(y_aug, dtype=torch.float32)
        net = nn.Sequential(nn.Linear(x_aug.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        loader = DataLoader(TensorDataset(x_t, y_t), batch_size=64, shuffle=True)
        if private:
            engine = PrivacyEngine()
            net, opt, loader = engine.make_private(
                module=net, optimizer=opt, data_loader=loader,
                noise_multiplier=noise_multiplier, max_grad_norm=1.0,
            )
        net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                nn.BCEWithLogitsLoss()(net(xb).squeeze(1), yb).backward()
                opt.step()
        net.eval()
        if private:
            try:
                state["eps"] = float(engine.get_epsilon(delta=DELTA))
            except Exception:
                state["eps"] = float("inf")
        else:
            state["eps"] = float("inf")
        with torch.no_grad():
            canary_t = torch.tensor(canary_features, dtype=torch.float32)
            prob = torch.sigmoid(net(canary_t).squeeze(1)).numpy()
        return -per_example_bce(canary_labels, prob)

    return train_and_score, state


def _audit_dpsgd(canaries: CanarySet, x_train, y_train, noise_multiplier, private, seed):
    scorer, state = _dpsgd_scorer(canaries.labels, noise_multiplier, private=private)
    result = one_run_model_audit(
        canaries, x_train, y_train, scorer,
        delta=DELTA, confidence=0.95, random_state=seed,
    )
    return result, state["eps"]


def run_dpsgd_audit(x_train, x_test, y_train, y_test) -> None:
    _banner(f"PART 2 — one-run auditor vs DP-SGD (claim ε ≈ {CLAIM_EPS}) by canary type")
    d = x_train.shape[1]
    seed = RANDOM_STATE
    n_flip = min(500, len(y_test))
    canary_sets = {
        "label-flip": make_label_flip_canaries(x_test, y_test, n_flip, random_state=1),
        "feature-outlier": make_feature_outlier_canaries(500, d, random_state=1),
        "duplicated x8 (group)": make_duplicated_canaries(300, d, duplication=8, random_state=1),
    }

    print("-- properly-noised DP-SGD (noise_multiplier = 2.0) --")
    for name, cs in canary_sets.items():
        result, eps = _audit_dpsgd(cs, x_train, y_train, 2.0, True, seed)
        group = eps * cs.duplication
        acc = result.details["accuracy"]
        lb = result.epsilon_lower_bound
        print(f"[{name:22s} n={len(cs):3d}] acct_ε={eps:5.2f} group_ε≈{group:6.2f}  "
              f"attack_acc={acc:.3f}  ε_lb={lb:.3f}  "
              f"violates_{CLAIM_EPS}={result.violates(CLAIM_EPS)}")

    print("-- broken DP-SGD: noise turned off (noise_multiplier = 0) --")
    for name, cs in canary_sets.items():
        result, eps = _audit_dpsgd(cs, x_train, y_train, 0.0, True, seed)
        acc = result.details["accuracy"]
        lb = result.epsilon_lower_bound
        print(f"[{name:22s} n={len(cs):3d}] acct_ε={'inf':>5}             "
              f"attack_acc={acc:.3f}  ε_lb={lb:.3f}  "
              f"violates_{CLAIM_EPS}={result.violates(CLAIM_EPS)}")


# ---------------------------------------------------------------------------
# Part 3 — the auditor has power where memorisation is real.
# ---------------------------------------------------------------------------
def run_power_check(x_train, x_test, y_train, y_test) -> None:
    _banner("PART 3 — audit power on a memorising learner (unbounded tree)")

    def make_scorer(labels):
        def train_and_score(x_aug, y_aug, canary_features):
            tree = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(x_aug, y_aug)
            return -per_example_bce(labels, tree.predict_proba(canary_features)[:, 1])
        return train_and_score

    for seed in (0, 1, 2):
        cs = make_label_flip_canaries(x_test, y_test, 250, random_state=seed)
        result = one_run_model_audit(
            cs, x_train, y_train, make_scorer(cs.labels), delta=DELTA, random_state=seed
        )
        acc = result.details["accuracy"]
        lb = result.epsilon_lower_bound
        print(f"[tree, label-flip n=250 seed={seed}] attack_acc={acc:.3f}  "
              f"ε_lb={lb:.3f}  clears_1.0={lb > 1.0}")


def main() -> None:
    x_train, x_test, y_train, y_test = load_features()
    print(f"preprocessed train={x_train.shape} test={x_test.shape} "
          f"positive_rate={y_train.mean():.3f}")
    run_lira(x_train, x_test, y_train, y_test)
    if TORCH:
        run_dpsgd_audit(x_train, x_test, y_train, y_test)
    else:  # pragma: no cover - optional dependency
        print("\n[torch/opacus unavailable — skipping DP-SGD audit]")
    run_power_check(x_train, x_test, y_train, y_test)
    print("\nDone.")


if __name__ == "__main__":
    main()
