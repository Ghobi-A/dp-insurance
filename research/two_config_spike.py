"""Powered two-configuration iso-epsilon subgroup-LiRA spike.

This remains a go/no-go experiment rather than a benchmark grid. It corrects the
first spike by:

* using the learnable ``high_cost`` task;
* refusing to run an attack when target ROC-AUC is below a hard gate;
* training every shadow on exactly the target training-set size, so the DP-SGD
  sampling rate and calibrated noise multiplier match the target;
* using at least 32 shadows;
* repeating three target seeds; and
* reporting stratified-bootstrap confidence intervals for the subgroup
  TPR@1% FPR gap Delta.

Run from the repository root:

    python research/two_config_spike.py --shadows 32 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dp.attacks import lira_offline_attack
from dp.pipeline import build_preprocessor
from dp.tasks import get_task, prepare_task_data


TARGET_EPSILON = 8.0
DELTA = 1e-5
TASK_NAME = "high_cost"
ATTACK_FPRS = (0.001, 0.01, 0.1)
HARD_AUC_GATE = 0.70
PREFERRED_AUC = 0.75
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_SHADOWS = 32
DEFAULT_BOOTSTRAP_REPS = 1000


@dataclass(frozen=True)
class Recipe:
    name: str
    batch_size: int
    epochs: int
    max_grad_norm: float
    hidden_units: int = 32
    learning_rate: float = 1e-2


RECIPES = (
    Recipe(
        name="small_batch_long_tight_clip",
        batch_size=64,
        epochs=30,
        max_grad_norm=0.5,
    ),
    Recipe(
        name="large_batch_short_loose_clip",
        batch_size=256,
        epochs=5,
        max_grad_norm=2.0,
    ),
)


def per_example_bce(y_true: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Numerically stable per-example binary cross-entropy."""
    prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    y_true = np.asarray(y_true, dtype=float)
    return -(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))


def build_model(n_features: int, hidden_units: int, seed: int) -> nn.Module:
    """Build the target/shadow architecture."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(n_features, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 1),
    )


def train_private_model(
    X: np.ndarray,
    y: np.ndarray,
    recipe: Recipe,
    seed: int,
) -> tuple[nn.Module, dict[str, float | int | str]]:
    """Train one RDP-accounted, Poisson-sampled DP-SGD model at epsilon=8."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    X = np.asarray(X[order], dtype=np.float32)
    y = np.asarray(y[order], dtype=np.float32)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(recipe.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    model = build_model(X.shape[1], recipe.hidden_units, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.learning_rate)
    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=recipe.epochs,
        target_epsilon=TARGET_EPSILON,
        target_delta=DELTA,
        max_grad_norm=recipe.max_grad_norm,
        poisson_sampling=True,
    )

    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(recipe.epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()

    metadata: dict[str, float | int | str] = {
        **asdict(recipe),
        "target_epsilon": TARGET_EPSILON,
        "achieved_epsilon": float(engine.get_epsilon(DELTA)),
        "delta": DELTA,
        "noise_multiplier": float(optimizer.noise_multiplier),
        "accountant": "rdp",
        "sampling": "poisson",
        "seed": seed,
        "train_size": int(len(dataset)),
    }
    return model, metadata


def predict_probability(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


def make_equalised_attack_cohort(
    groups: np.ndarray,
    membership: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Equalise every group x membership cell to the smallest cell size."""
    rng = np.random.default_rng(seed)
    unique_groups = sorted(str(value) for value in np.unique(groups))
    string_groups = groups.astype(str)
    cells: dict[tuple[str, int], np.ndarray] = {}
    for group in unique_groups:
        for member in (0, 1):
            idx = np.flatnonzero((string_groups == group) & (membership == member))
            if idx.size == 0:
                raise RuntimeError(f"empty attack cell group={group!r}, membership={member}")
            cells[(group, member)] = idx

    per_cell = min(len(idx) for idx in cells.values())
    selected: list[int] = []
    counts: dict[str, int] = {}
    for (group, member), idx in cells.items():
        chosen = rng.choice(idx, size=per_cell, replace=False)
        selected.extend(chosen.tolist())
        counts[f"{group}|member={member}"] = int(per_cell)

    return np.asarray(sorted(selected), dtype=int), counts


def _fixed_size_shadow_indices(
    y_pool: np.ndarray,
    train_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample exactly ``train_size`` rows while retaining both target classes."""
    if train_size >= len(y_pool):
        raise ValueError("shadow train_size must be smaller than the shadow pool")
    for _ in range(1000):
        idx = rng.choice(len(y_pool), size=train_size, replace=False)
        if np.unique(y_pool[idx]).size == 2:
            return np.asarray(idx, dtype=int)
    raise RuntimeError("could not draw a two-class fixed-size shadow training set")


def collect_shadow_out_losses(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    attack_idx: np.ndarray,
    target_train_size: int,
    target_noise_multiplier: float,
    recipe: Recipe,
    num_shadows: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float | int | str]], int]:
    """Train size-matched shadows and retain losses when each attack row is OUT."""
    rng = np.random.default_rng(seed)
    out_losses: list[list[float]] = [[] for _ in range(len(attack_idx))]
    metadata: list[dict[str, float | int | str]] = []

    for shadow_id in range(num_shadows):
        train_idx = _fixed_size_shadow_indices(y_pool, target_train_size, rng)
        in_mask = np.zeros(len(y_pool), dtype=bool)
        in_mask[train_idx] = True

        model, meta = train_private_model(
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
                "size-matched shadow has a different calibrated noise multiplier: "
                f"target={target_noise_multiplier}, shadow={meta['noise_multiplier']}"
            )

        meta["shadow_id"] = shadow_id
        metadata.append(meta)

        losses = per_example_bce(
            y_pool[attack_idx],
            predict_probability(model, X_pool[attack_idx]),
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

    min_out = min(len(values) for values in out_losses)
    if min_out < 4:
        raise RuntimeError(
            f"LiRA needs a usable OUT distribution; minimum was {min_out}. "
            "Increase --shadows."
        )
    matrix = np.vstack([np.asarray(values[:min_out], dtype=float) for values in out_losses])
    return matrix, metadata, min_out


def tpr_at_fpr(membership: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Highest TPR attainable without exceeding ``target_fpr``."""
    fpr, tpr, _ = roc_curve(membership, scores)
    allowed = fpr <= target_fpr
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def stratified_bootstrap_delta(
    groups: np.ndarray,
    membership: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
    reps: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap Delta while preserving every group x membership cell size."""
    rng = np.random.default_rng(seed)
    groups = groups.astype(str)
    unique_groups = sorted(np.unique(groups))
    cells = {
        (group, member): np.flatnonzero((groups == group) & (membership == member))
        for group in unique_groups
        for member in (0, 1)
    }
    if any(len(idx) == 0 for idx in cells.values()):
        raise RuntimeError("bootstrap requires non-empty group x membership cells")

    deltas = np.empty(reps, dtype=float)
    for rep in range(reps):
        group_tprs = []
        for group in unique_groups:
            member_idx = rng.choice(
                cells[(group, 1)], size=len(cells[(group, 1)]), replace=True
            )
            nonmember_idx = rng.choice(
                cells[(group, 0)], size=len(cells[(group, 0)]), replace=True
            )
            idx = np.concatenate([member_idx, nonmember_idx])
            group_tprs.append(tpr_at_fpr(membership[idx], scores[idx], target_fpr))
        deltas[rep] = max(group_tprs) - min(group_tprs)

    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "reps": int(reps),
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def run_recipe_seed(
    recipe: Recipe,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    groups_test: np.ndarray,
    num_shadows: int,
    bootstrap_reps: int,
    auc_gate: float,
    seed: int,
) -> dict[str, object]:
    """Run one recipe/target-seed cell, with an AUC gate before shadow training."""
    X_attack = np.vstack([X_train, X_test])
    y_attack = np.concatenate([y_train, y_test])
    groups_attack = np.concatenate([groups_train, groups_test]).astype(str)
    membership = np.concatenate(
        [np.ones(len(y_train), dtype=int), np.zeros(len(y_test), dtype=int)]
    )
    attack_idx, cohort_counts = make_equalised_attack_cohort(
        groups_attack, membership, seed
    )

    target, target_meta = train_private_model(X_train, y_train, recipe, seed=seed)
    target_test_prob = predict_probability(target, X_test)
    target_test_auc = float(roc_auc_score(y_test, target_test_prob))
    preferred_auc_met = target_test_auc >= PREFERRED_AUC

    base_result: dict[str, object] = {
        "seed": int(seed),
        "recipe": asdict(recipe),
        "target": target_meta,
        "target_test_auc": target_test_auc,
        "auc_gate": float(auc_gate),
        "preferred_auc": PREFERRED_AUC,
        "preferred_auc_met": preferred_auc_met,
        "cohort_counts": cohort_counts,
        "num_attack_examples": int(len(attack_idx)),
        "attack_status": "pending",
    }

    print(
        f"[{recipe.name} seed={seed}] target AUC={target_test_auc:.4f} "
        f"(gate={auc_gate:.2f}, preferred={PREFERRED_AUC:.2f})",
        flush=True,
    )
    if target_test_auc < auc_gate:
        base_result["attack_status"] = "skipped_target_auc_below_gate"
        base_result["subgroups"] = {}
        base_result["delta_tpr_at_fpr_0.01"] = None
        base_result["bootstrap_delta"] = None
        return base_result

    X_pool = np.vstack([X_train, X_val, X_test])
    y_pool = np.concatenate([y_train, y_val, y_test])

    attack_pool_idx = attack_idx.copy()
    train_n = len(y_train)
    attack_pool_idx[attack_pool_idx >= train_n] += len(y_val)

    target_prob_attack = predict_probability(target, X_attack)
    target_losses = per_example_bce(
        y_attack[attack_idx], target_prob_attack[attack_idx]
    )
    shadow_out, shadow_meta, min_out = collect_shadow_out_losses(
        X_pool,
        y_pool,
        attack_pool_idx,
        target_train_size=len(y_train),
        target_noise_multiplier=float(target_meta["noise_multiplier"]),
        recipe=recipe,
        num_shadows=num_shadows,
        seed=seed + 10_000,
    )

    attack_groups = groups_attack[attack_idx]
    attack_membership = membership[attack_idx]
    subgroup_results: dict[str, dict[str, float | int]] = {}
    all_scores = np.empty(len(attack_idx), dtype=float)

    for group in sorted(np.unique(attack_groups)):
        group_mask = attack_groups == group
        result = lira_offline_attack(
            target_losses[group_mask],
            attack_membership[group_mask],
            shadow_out[group_mask],
            fprs=ATTACK_FPRS,
        )
        all_scores[group_mask] = result.scores
        subgroup_results[str(group)] = {
            "n": int(group_mask.sum()),
            "members": int(attack_membership[group_mask].sum()),
            "nonmembers": int((attack_membership[group_mask] == 0).sum()),
            "auc": float(result.auc),
            "tpr_at_fpr_0.001": float(result.tpr_at_fpr[0.001]),
            "tpr_at_fpr_0.01": float(result.tpr_at_fpr[0.01]),
            "tpr_at_fpr_0.1": float(result.tpr_at_fpr[0.1]),
        }

    tprs = [metrics["tpr_at_fpr_0.01"] for metrics in subgroup_results.values()]
    observed_delta = float(max(tprs) - min(tprs))
    bootstrap = stratified_bootstrap_delta(
        attack_groups,
        attack_membership,
        all_scores,
        target_fpr=0.01,
        reps=bootstrap_reps,
        seed=seed + 20_000,
    )

    base_result.update(
        {
            "attack_status": "completed",
            "num_shadows": int(num_shadows),
            "min_out_shadow_losses_per_example": int(min_out),
            "subgroups": subgroup_results,
            "delta_tpr_at_fpr_0.01": observed_delta,
            "bootstrap_delta": bootstrap,
            "shadow_train_size": int(len(y_train)),
            "shadow_metadata": shadow_meta,
        }
    )
    return base_result


def summarize_recipe(recipe: Recipe, seed_results: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate completed target seeds without hiding skipped cells."""
    completed = [
        result for result in seed_results if result["attack_status"] == "completed"
    ]
    deltas = np.asarray(
        [result["delta_tpr_at_fpr_0.01"] for result in completed], dtype=float
    )
    aucs = np.asarray([result["target_test_auc"] for result in seed_results], dtype=float)
    summary: dict[str, object] = {
        "recipe": asdict(recipe),
        "target_seeds": int(len(seed_results)),
        "completed_attacks": int(len(completed)),
        "skipped_attacks": int(len(seed_results) - len(completed)),
        "target_auc_mean": float(aucs.mean()),
        "target_auc_min": float(aucs.min()),
        "target_auc_max": float(aucs.max()),
    }
    if len(deltas):
        summary.update(
            {
                "delta_mean": float(deltas.mean()),
                "delta_median": float(np.median(deltas)),
                "delta_min": float(deltas.min()),
                "delta_max": float(deltas.max()),
            }
        )
    else:
        summary.update(
            {
                "delta_mean": None,
                "delta_median": None,
                "delta_min": None,
                "delta_max": None,
            }
        )
    return summary


def render_markdown(payload: dict[str, object]) -> str:
    """Render an examiner-readable result summary."""
    lines = [
        "# Powered two-configuration iso-epsilon subgroup-LiRA spike",
        "",
        f"Task: `{payload['task']}`.",
        f"Target privacy: epsilon={TARGET_EPSILON}, delta={DELTA}.",
        f"Hard attack gate: target test ROC-AUC >= {payload['auc_gate']:.2f}.",
        f"Preferred target quality: ROC-AUC >= {PREFERRED_AUC:.2f}.",
        (
            "Shadows are trained on exactly the target training-set size, with the "
            "same recipe, accountant, epsilon and delta."
        ),
        "Attack cohort: equal counts in every sex x membership cell.",
        "Headline metric: subgroup offline-LiRA TPR at 1% FPR and max-min gap Delta.",
        "",
        "| Recipe | Seed | Achieved epsilon | Noise | Test AUC | Attack | Group | LiRA AUC | TPR@1% | Delta | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]

    for result in payload["seed_results"]:
        target = result["target"]
        if result["attack_status"] != "completed":
            lines.append(
                f"| {result['recipe']['name']} | {result['seed']} | "
                f"{target['achieved_epsilon']:.4f} | {target['noise_multiplier']:.4f} | "
                f"{result['target_test_auc']:.4f} | **SKIPPED: AUC gate** | - | - | - | - | - |"
            )
            continue

        bootstrap = result["bootstrap_delta"]
        ci = f"[{bootstrap['ci95_low']:.4f}, {bootstrap['ci95_high']:.4f}]"
        first = True
        for group, metrics in result["subgroups"].items():
            delta = f"{result['delta_tpr_at_fpr_0.01']:.4f}" if first else ""
            ci_cell = ci if first else ""
            lines.append(
                f"| {result['recipe']['name']} | {result['seed']} | "
                f"{target['achieved_epsilon']:.4f} | {target['noise_multiplier']:.4f} | "
                f"{result['target_test_auc']:.4f} | completed | {group} | "
                f"{metrics['auc']:.4f} | {metrics['tpr_at_fpr_0.01']:.4f} | "
                f"{delta} | {ci_cell} |"
            )
            first = False

    lines.extend(
        [
            "",
            "## Across-seed summary",
            "",
            "| Recipe | Seeds | Attacks completed | Mean target AUC | Min target AUC | Mean Delta | Delta range |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for summary in payload["recipe_summaries"]:
        mean_delta = (
            "-" if summary["delta_mean"] is None else f"{summary['delta_mean']:.4f}"
        )
        delta_range = (
            "-"
            if summary["delta_min"] is None
            else f"[{summary['delta_min']:.4f}, {summary['delta_max']:.4f}]"
        )
        lines.append(
            f"| {summary['recipe']['name']} | {summary['target_seeds']} | "
            f"{summary['completed_attacks']} | {summary['target_auc_mean']:.4f} | "
            f"{summary['target_auc_min']:.4f} | {mean_delta} | {delta_range} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A non-zero point estimate is not called a signal when its bootstrap interval "
            "includes zero or when it is driven by a single seed. The experiment only "
            "supports expansion when target models learn the task and the subgroup gap "
            "is stable across seeds.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadows", type=int, default=DEFAULT_SHADOWS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--auc-gate", type=float, default=HARD_AUC_GATE)
    parser.add_argument("--output-dir", default="reports/spike")
    args = parser.parse_args()

    if args.shadows < DEFAULT_SHADOWS:
        raise ValueError(f"Use at least {DEFAULT_SHADOWS} shadows.")
    if len(args.seeds) < 3:
        raise ValueError("Use at least three target seeds.")
    if args.bootstrap_reps < 200:
        raise ValueError("Use at least 200 bootstrap replicates.")
    if not (0.5 <= args.auc_gate <= 1.0):
        raise ValueError("--auc-gate must be in [0.5, 1.0]")

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "insurance.csv")

    payload: dict[str, object] = {
        "experiment": "powered_two_config_iso_epsilon_subgroup_lira_spike",
        "target_epsilon": TARGET_EPSILON,
        "delta": DELTA,
        "task": TASK_NAME,
        "sensitive_attribute": "sex",
        "auc_gate": float(args.auc_gate),
        "preferred_auc": PREFERRED_AUC,
        "seeds": [int(seed) for seed in args.seeds],
        "num_shadows": int(args.shadows),
        "bootstrap_reps": int(args.bootstrap_reps),
        "seed_results": [],
        "recipe_summaries": [],
    }

    for recipe in RECIPES:
        recipe_seed_results = []
        for seed in args.seeds:
            print(f"\n=== {recipe.name} | target seed {seed} ===", flush=True)
            data = prepare_task_data(df, get_task(TASK_NAME), random_state=seed)
            preprocessor = build_preprocessor(data.X_train)
            X_train = np.asarray(
                preprocessor.fit_transform(data.X_train), dtype=np.float32
            )
            X_val = np.asarray(preprocessor.transform(data.X_val), dtype=np.float32)
            X_test = np.asarray(preprocessor.transform(data.X_test), dtype=np.float32)

            result = run_recipe_seed(
                recipe,
                X_train,
                data.y_train,
                X_val,
                data.y_val,
                X_test,
                data.y_test,
                data.sensitive_train,
                data.sensitive_test,
                num_shadows=args.shadows,
                bootstrap_reps=args.bootstrap_reps,
                auc_gate=args.auc_gate,
                seed=seed,
            )
            recipe_seed_results.append(result)
            payload["seed_results"].append(result)

        payload["recipe_summaries"].append(
            summarize_recipe(recipe, recipe_seed_results)
        )

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "two_config_spike.json"
    md_path = output_dir / "two_config_spike.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(render_markdown(payload))

    print("\n" + md_path.read_text(), flush=True)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
