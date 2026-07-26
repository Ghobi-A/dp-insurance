"""Two-configuration iso-epsilon subgroup-LiRA spike.

This is deliberately a go/no-go experiment, not a benchmark grid. It trains two
contrasting DP-SGD recipes at the same target epsilon on the insurance dataset,
runs offline LiRA, equalises the attack cohort across sex and membership status,
and reports the subgroup TPR@1% FPR gap Delta.

Run from the repository root:

    python research/two_config_spike.py --shadows 12
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dp.attacks import lira_offline_attack
from dp.pipeline import build_preprocessor
from dp.tasks import get_task, prepare_task_data


TARGET_EPSILON = 8.0
DELTA = 1e-5
ATTACK_FPRS = (0.001, 0.01, 0.1)


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
    prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    y_true = np.asarray(y_true, dtype=float)
    return -(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))


def build_model(n_features: int, hidden_units: int, seed: int) -> nn.Module:
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
    selected = []
    counts: dict[str, int] = {}
    for (group, member), idx in cells.items():
        chosen = rng.choice(idx, size=per_cell, replace=False)
        selected.extend(chosen.tolist())
        counts[f"{group}|member={member}"] = int(per_cell)

    selected_idx = np.asarray(sorted(selected), dtype=int)
    return selected_idx, counts


def collect_shadow_out_losses(
    X_all: np.ndarray,
    y_all: np.ndarray,
    attack_idx: np.ndarray,
    recipe: Recipe,
    num_shadows: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float | int | str]], int]:
    """Train shadows and retain each example's losses only when it was OUT."""
    rng = np.random.default_rng(seed)
    out_losses: list[list[float]] = [[] for _ in range(len(attack_idx))]
    metadata: list[dict[str, float | int | str]] = []

    for shadow_id in range(num_shadows):
        mask = rng.random(len(y_all)) < 0.5
        while np.unique(y_all[mask]).size < 2 or mask.sum() < 2:
            mask = rng.random(len(y_all)) < 0.5

        model, meta = train_private_model(
            X_all[mask],
            y_all[mask],
            recipe,
            seed=seed + shadow_id + 1,
        )
        meta["shadow_id"] = shadow_id
        metadata.append(meta)

        losses = per_example_bce(y_all[attack_idx], predict_probability(model, X_all[attack_idx]))
        attack_out = ~mask[attack_idx]
        for local_idx in np.flatnonzero(attack_out):
            out_losses[local_idx].append(float(losses[local_idx]))

        print(
            f"[{recipe.name}] shadow {shadow_id + 1}/{num_shadows} "
            f"achieved_eps={meta['achieved_epsilon']:.4f} "
            f"noise={meta['noise_multiplier']:.4f}",
            flush=True,
        )

    min_out = min(len(values) for values in out_losses)
    if min_out < 2:
        raise RuntimeError(
            f"LiRA needs at least two OUT shadows per example; minimum was {min_out}. "
            "Increase --shadows."
        )
    matrix = np.vstack([np.asarray(values[:min_out], dtype=float) for values in out_losses])
    return matrix, metadata, min_out


def run_recipe(
    recipe: Recipe,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    groups_test: np.ndarray,
    num_shadows: int,
    seed: int,
) -> dict[str, object]:
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    groups_all = np.concatenate([groups_train, groups_test]).astype(str)
    membership = np.concatenate(
        [np.ones(len(y_train), dtype=int), np.zeros(len(y_test), dtype=int)]
    )

    attack_idx, cohort_counts = make_equalised_attack_cohort(groups_all, membership, seed)

    target, target_meta = train_private_model(X_train, y_train, recipe, seed=seed)
    target_prob_all = predict_probability(target, X_all)
    target_losses = per_example_bce(y_all[attack_idx], target_prob_all[attack_idx])
    target_test_auc = float(roc_auc_score(y_test, target_prob_all[len(y_train) :]))

    shadow_out, shadow_meta, min_out = collect_shadow_out_losses(
        X_all,
        y_all,
        attack_idx,
        recipe,
        num_shadows=num_shadows,
        seed=seed + 10_000,
    )

    subgroup_results: dict[str, dict[str, float | int]] = {}
    attack_groups = groups_all[attack_idx]
    attack_membership = membership[attack_idx]
    for group in sorted(np.unique(attack_groups)):
        group_mask = attack_groups == group
        result = lira_offline_attack(
            target_losses[group_mask],
            attack_membership[group_mask],
            shadow_out[group_mask],
            fprs=ATTACK_FPRS,
        )
        subgroup_results[str(group)] = {
            "n": int(group_mask.sum()),
            "members": int(attack_membership[group_mask].sum()),
            "auc": float(result.auc),
            "tpr_at_fpr_0.001": float(result.tpr_at_fpr[0.001]),
            "tpr_at_fpr_0.01": float(result.tpr_at_fpr[0.01]),
            "tpr_at_fpr_0.1": float(result.tpr_at_fpr[0.1]),
        }

    tprs = [metrics["tpr_at_fpr_0.01"] for metrics in subgroup_results.values()]
    delta_tpr_1pct = float(max(tprs) - min(tprs))

    return {
        "recipe": asdict(recipe),
        "target": target_meta,
        "target_test_auc": target_test_auc,
        "cohort_counts": cohort_counts,
        "num_attack_examples": int(len(attack_idx)),
        "num_shadows": int(num_shadows),
        "min_out_shadow_losses_per_example": int(min_out),
        "subgroups": subgroup_results,
        "delta_tpr_at_fpr_0.01": delta_tpr_1pct,
        "shadow_metadata": shadow_meta,
    }


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Two-configuration iso-epsilon subgroup-LiRA spike",
        "",
        f"Target privacy: epsilon={TARGET_EPSILON}, delta={DELTA}.",
        "Attack cohort: equal counts in every sex x membership cell.",
        "Headline metric: subgroup offline-LiRA TPR at 1% FPR and max-min gap Delta.",
        "",
        "| Recipe | Achieved epsilon | Noise multiplier | Test ROC-AUC | Group | LiRA AUC | TPR@1% FPR |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for result in payload["results"]:
        target = result["target"]
        for group, metrics in result["subgroups"].items():
            lines.append(
                f"| {result['recipe']['name']} | {target['achieved_epsilon']:.4f} | "
                f"{target['noise_multiplier']:.4f} | {result['target_test_auc']:.4f} | "
                f"{group} | {metrics['auc']:.4f} | {metrics['tpr_at_fpr_0.01']:.4f} |"
            )
        lines.append(
            f"| **{result['recipe']['name']} Delta** |  |  |  | **max-min** |  | "
            f"**{result['delta_tpr_at_fpr_0.01']:.4f}** |"
        )

    deltas = [result["delta_tpr_at_fpr_0.01"] for result in payload["results"]]
    lines.extend(
        [
            "",
            "## Go/no-go readout",
            "",
            f"Delta values: {', '.join(f'{value:.4f}' for value in deltas)}.",
            f"Absolute between-recipe Delta separation: {abs(deltas[0] - deltas[1]):.4f}.",
            "",
            "This spike is directional only: one target seed and a small shadow ensemble. "
            "A visible separation justifies a powered repeated-seed experiment; a null result "
            "means the current dataset/attack/config contrast did not resolve the phenomenon.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadows", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="reports/spike")
    args = parser.parse_args()

    if args.shadows < 6:
        raise ValueError("Use at least six shadows; 12 is the intended spike setting.")

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "insurance.csv")
    data = prepare_task_data(df, get_task("smoker_without_charges"), random_state=args.seed)

    preprocessor = build_preprocessor(data.X_train)
    X_train = np.asarray(preprocessor.fit_transform(data.X_train), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(data.X_test), dtype=np.float32)

    payload: dict[str, object] = {
        "experiment": "two_config_iso_epsilon_subgroup_lira_spike",
        "target_epsilon": TARGET_EPSILON,
        "delta": DELTA,
        "task": data.task.name,
        "sensitive_attribute": "sex",
        "seed": args.seed,
        "results": [],
    }

    for recipe in RECIPES:
        print(f"\n=== {recipe.name} ===", flush=True)
        payload["results"].append(
            run_recipe(
                recipe,
                X_train,
                data.y_train,
                X_test,
                data.y_test,
                data.sensitive_train,
                data.sensitive_test,
                num_shadows=args.shadows,
                seed=args.seed,
            )
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
