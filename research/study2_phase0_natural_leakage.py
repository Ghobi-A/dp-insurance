"""Study 2, Phase 0: natural-leakage feasibility gate on ACS/Folktables.

Phase 0 is a *gate*, not an experiment with an interesting outcome. It asks
whether a naturally (non-privately) trained model on the frozen ACSIncome slice
leaks membership strongly enough that a DP epsilon ladder built on top of it
could measure anything at all. If the answer is no, there is nothing for a
ladder to attenuate, and every downstream comparison would be a comparison of
noise with noise.

The continuation gate, frozen before any result exists:

1. mean online-LiRA ROC-AUC across the three target seeds >= 0.60,
2. online-LiRA ROC-AUC > 0.5 on *every* seed,
3. the stratified membership-permutation null rejected at alpha = 0.05 on all
   three seeds.

All three must hold. There is deliberately no reconsideration band: a result
such as mean AUC = 0.58 with 3/3 seeds significant is a **FAIL for
continuation** and is reported descriptively as measurable leakage below the
predeclared continuation margin -- not as a near-miss to be argued up.

The gate is evaluated **once**, on the frozen slice defined in
``research/study2_acs_slice.py``. A FAIL is terminal for the natural-leakage
Folktables/ACS branch: no other state, year, task, architecture, epoch budget
or model may be tried after seeing the result, no DP epsilon ladder is launched,
and the target is not tuned for more leakage. The preregistered fallback
direction is canary-based auditing, which is *identified* on failure and not
implemented or run until its own estimand and analysis plan are frozen
separately.

Online LiRA is the primary attack for this gate; offline LiRA is reported as a
secondary descriptive measure and never decides the gate.

Subcommands:
    ``seed``       run one target seed and persist its per-example records;
    ``aggregate``  combine all seeds, evaluate the gate, print PASS or FAIL.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The attack-power control is this repository's tested implementation of every
# statistic used below. Reuse it rather than growing a second framework.
from attack_power_control import (  # noqa: E402
    ATTACK_FPRS,
    HEADLINE_FPR,
    balanced_shadow_train_sets,
    build_mlp,
    memorisation_metrics,
    per_example_bce,
    stratified_membership_permutation_test,
    tpr_at_fpr,
)
from study2_acs_slice import (  # noqa: E402
    FEATURES,
    SAMPLING_SEED,
    SENSITIVE_ATTRIBUTE,
    load_slice,
    slice_metadata,
    verify_slice,
)

# --------------------------------------------------------------------------- #
# Frozen configuration
# --------------------------------------------------------------------------- #

STUDY = "study2-phase0-natural-leakage"
TARGET_SEEDS = (42, 43, 44)
HIDDEN_UNITS = (128, 128, 64)
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
EPOCHS = 60
NUM_SHADOWS = 64
PERMUTATION_REPS = 1000
ALPHA = 0.05

#: Target training-set size; the remainder of the frozen slice supplies
#: non-members and the shadow pool. Shadow training sets are the same size, so
#: shadow and target models are matched in every respect except which rows they
#: saw.
TRAIN_SIZE = 20_000

#: Cap per ``group x membership`` cell in the attack cohort. Equalising the four
#: cells keeps the aggregate AUC free of composition effects; the cap bounds the
#: cost of 1000 permutation replicates without materially affecting the
#: precision of an AUC estimated from 10,000 examples.
COHORT_PER_CELL = 2_500

#: LiRA needs a usable Gaussian fit per example on each side.
MIN_GAUSSIAN_OBSERVATIONS = 10

#: The primary attack for the gate. Offline LiRA is secondary and descriptive.
PRIMARY_ATTACK = "online"
SECONDARY_ATTACK = "offline"

#: Gate thresholds. Strict: no reconsideration band.
GATE_MEAN_AUC = 0.60
GATE_PER_SEED_AUC = 0.50

PASS = "PASS"
FAIL = "FAIL"

CONTINUATION_ON_FAIL = (
    "The natural-leakage Folktables/ACS branch of Study 2 is closed. Do not "
    "launch the DP epsilon ladder, do not tune the target for more leakage, and "
    "do not retry another state, year, task, architecture, epoch budget or "
    "model. The preregistered fallback direction is canary-based auditing, "
    "which is identified here only; it is not implemented or run until its own "
    "estimand and analysis plan are frozen separately."
)


def frozen_config(
    epochs: int = EPOCHS,
    shadows: int = NUM_SHADOWS,
    permutation_reps: int = PERMUTATION_REPS,
    smoke: bool = False,
) -> dict[str, object]:
    """The training and attack recipe, plus any deviation from the frozen one."""
    config = {
        "study": STUDY,
        "task": "ACSIncome",
        "sensitive_attribute": SENSITIVE_ATTRIBUTE,
        "architecture": "Linear(d,128)->ReLU->Linear(128,128)->ReLU->Linear(128,64)->ReLU->Linear(64,1)",
        "optimiser": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "regularisation": "none (no dropout, no weight decay, no early stopping)",
        "train_size": TRAIN_SIZE,
        "num_shadows": shadows,
        "target_seeds": list(TARGET_SEEDS),
        "sampling_seed": SAMPLING_SEED,
        "permutation_reps": permutation_reps,
        "alpha": ALPHA,
        "primary_attack": PRIMARY_ATTACK,
        "secondary_attack": SECONDARY_ATTACK,
        "cohort_per_cell": COHORT_PER_CELL,
        "smoke": bool(smoke),
    }
    deviations = {
        key: value
        for key, value, frozen in (
            ("epochs", epochs, EPOCHS),
            ("num_shadows", shadows, NUM_SHADOWS),
            ("permutation_reps", permutation_reps, PERMUTATION_REPS),
        )
        if value != frozen
    }
    config["deviates_from_frozen"] = deviations
    return config


def is_preregistered(config: dict[str, object]) -> bool:
    """True only for a run that may be read as a Phase 0 result."""
    return not config.get("smoke") and not config.get("deviates_from_frozen")


# --------------------------------------------------------------------------- #
# Design construction (pure; no torch needed, so the tests stay cheap)
# --------------------------------------------------------------------------- #


def standardise(features: np.ndarray) -> np.ndarray:
    """Z-score every column using the whole frozen slice.

    Deliberately fitted on the slice rather than on a split: the scaler is then
    a property of the frozen data, identical for the target and all 64 shadows,
    so it cannot itself become a channel that distinguishes them.
    """
    features = np.asarray(features, dtype=float)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    return (features - mean) / scale


def split_membership(n_rows: int, train_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Target training indices and the 0/1 membership vector over the slice.

    The permutation is seeded by the *target* seed, so each seed trains on a
    different subset of the same frozen slice -- the seeds are replicates of the
    experiment, not three readings of one draw.
    """
    if train_size >= n_rows:
        raise ValueError("train_size must be smaller than the slice")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_rows)
    train_idx = np.sort(permutation[:train_size])
    membership = np.zeros(n_rows, dtype=int)
    membership[train_idx] = 1
    return train_idx, membership


def equalised_cohort(
    groups: np.ndarray,
    membership: np.ndarray,
    seed: int,
    per_cell_cap: int = COHORT_PER_CELL,
) -> tuple[np.ndarray, dict[str, int]]:
    """Equal-sized ``group x membership`` cells, capped at ``per_cell_cap``.

    Equal cells mean the aggregate AUC cannot be moved by group composition,
    and the cap keeps the permutation null affordable at 1000 replicates.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups).astype(str)
    membership = np.asarray(membership).astype(int)
    cells: dict[tuple[str, int], np.ndarray] = {}
    for group in sorted(np.unique(groups)):
        for member in (0, 1):
            idx = np.flatnonzero((groups == group) & (membership == member))
            if idx.size == 0:
                raise RuntimeError(f"empty attack cell group={group!r}, membership={member}")
            cells[(group, member)] = idx

    per_cell = min(per_cell_cap, min(len(idx) for idx in cells.values()))
    selected: list[int] = []
    counts: dict[str, int] = {}
    for (group, member), idx in sorted(cells.items()):
        chosen = rng.choice(idx, size=per_cell, replace=False)
        selected.extend(chosen.tolist())
        counts[f"{group}|member={member}"] = int(per_cell)
    return np.asarray(sorted(selected), dtype=int), counts


def attack_summary(
    membership: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    permutation_reps: int,
    seed: int,
) -> dict[str, object]:
    """AUC, TPR at the reported FPRs, per-group AUC and the permutation null."""
    from sklearn.metrics import roc_auc_score

    membership = np.asarray(membership).astype(int)
    groups = np.asarray(groups).astype(str)
    scores = np.asarray(scores, dtype=float)

    permutation = stratified_membership_permutation_test(
        groups, membership, scores, HEADLINE_FPR, permutation_reps, seed
    )
    return {
        "auc": float(roc_auc_score(membership, scores)),
        **{f"tpr_at_fpr_{fpr}": tpr_at_fpr(membership, scores, fpr) for fpr in ATTACK_FPRS},
        "subgroup_auc": {
            group: float(roc_auc_score(membership[groups == group], scores[groups == group]))
            for group in sorted(np.unique(groups))
        },
        "permutation": permutation,
        "permutation_p_value": float(permutation["aggregate_auc"]["p_value"]),
    }


# --------------------------------------------------------------------------- #
# Gate evaluation (pure)
# --------------------------------------------------------------------------- #


def evaluate_gate(
    seed_results: Sequence[dict[str, object]],
    expected_seeds: Sequence[int] = TARGET_SEEDS,
) -> dict[str, object]:
    """Evaluate the three continuation criteria. Never partially evaluated.

    A missing seed is not a smaller experiment: the gate is defined over all
    three, so an incomplete set is an ``INCOMPLETE`` verdict rather than a gate
    decision taken on what happened to finish.
    """
    by_seed = {int(result["seed"]): result for result in seed_results}
    missing = [seed for seed in expected_seeds if seed not in by_seed]
    aucs = [float(by_seed[seed][PRIMARY_ATTACK]["auc"]) for seed in expected_seeds if seed in by_seed]
    p_values = [
        float(by_seed[seed][PRIMARY_ATTACK]["permutation_p_value"])
        for seed in expected_seeds
        if seed in by_seed
    ]
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    significant = [p < ALPHA for p in p_values]

    criteria = [
        {
            "id": "mean_auc",
            "description": f"mean online-LiRA ROC-AUC >= {GATE_MEAN_AUC}",
            "observed": mean_auc,
            "threshold": GATE_MEAN_AUC,
            "passed": bool(aucs) and mean_auc >= GATE_MEAN_AUC,
        },
        {
            "id": "per_seed_auc",
            "description": f"online-LiRA ROC-AUC > {GATE_PER_SEED_AUC} on every seed",
            "observed": aucs,
            "threshold": GATE_PER_SEED_AUC,
            "passed": bool(aucs) and all(auc > GATE_PER_SEED_AUC for auc in aucs),
        },
        {
            "id": "permutation",
            "description": f"membership-permutation null rejected at alpha={ALPHA} on all seeds",
            "observed": p_values,
            "threshold": ALPHA,
            "passed": bool(p_values) and all(significant),
        },
    ]

    if missing:
        verdict = "INCOMPLETE"
        rationale = (
            "Gate not evaluated: results missing for seed(s) "
            f"{', '.join(str(seed) for seed in missing)}. The gate is defined over "
            "all three target seeds and is never evaluated on a partial set."
        )
    elif all(item["passed"] for item in criteria):
        verdict = PASS
        rationale = (
            f"All three criteria met (mean online AUC {mean_auc:.4f}, "
            f"{sum(significant)}/{len(significant)} seeds significant)."
        )
    else:
        verdict = FAIL
        rationale = (
            f"Mean online AUC {mean_auc:.4f}; "
            f"{sum(significant)}/{len(significant)} seeds reject the permutation null. "
            + " ".join(
                f"Criterion {item['id']} failed."
                for item in criteria
                if not item["passed"]
            )
        )

    descriptive = None
    if verdict == FAIL and any(significant):
        descriptive = (
            f"Measurable membership leakage is present ({sum(significant)}/"
            f"{len(significant)} seeds reject the permutation null at alpha={ALPHA}, "
            f"mean online-LiRA AUC {mean_auc:.4f}) but below the predeclared "
            f"continuation margin of {GATE_MEAN_AUC}. This is reported "
            "descriptively and is not a basis for continuation."
        )

    return {
        "verdict": verdict,
        "gate": PASS if verdict == PASS else FAIL if verdict == FAIL else "INCOMPLETE",
        "mean_online_auc": mean_auc,
        "per_seed_online_auc": aucs,
        "per_seed_permutation_p": p_values,
        "seeds_significant": int(sum(significant)),
        "criteria": criteria,
        "missing_seeds": [int(seed) for seed in missing],
        "rationale": rationale,
        "descriptive_note": descriptive,
        "continuation": (
            "Continuation criteria met for the natural-leakage branch."
            if verdict == PASS
            else CONTINUATION_ON_FAIL
            if verdict == FAIL
            else "No continuation decision: the gate was not evaluated."
        ),
    }


# --------------------------------------------------------------------------- #
# Training and one-seed execution
# --------------------------------------------------------------------------- #


def train_target_model(X: np.ndarray, y: np.ndarray, seed: int, epochs: int, batch_size: int):
    """Train one non-private model with the frozen recipe."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    model = build_mlp(X.shape[1], HIDDEN_UNITS, seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            optimiser.step()
    return model


def predict_probability(model, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Positive-class probabilities, evaluated in chunks to bound memory."""
    import torch

    model.eval()
    X = np.asarray(X, dtype=np.float32)
    out = np.empty(len(X), dtype=float)
    with torch.no_grad():
        for start in range(0, len(X), chunk):
            batch = torch.from_numpy(X[start : start + chunk])
            out[start : start + chunk] = torch.sigmoid(model(batch).squeeze(-1)).cpu().numpy()
    return out


def run_seed(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
    epochs: int,
    shadows: int,
    permutation_reps: int,
    batch_size: int,
    train_size: int,
    smoke: bool,
    log,
) -> dict[str, object]:
    """Train one target and its matched shadow ensemble, then attack."""
    from dp.attacks import lira_offline_attack, lira_online_attack

    min_observations = 2 if smoke else MIN_GAUSSIAN_OBSERVATIONS
    n_rows = len(labels)
    train_idx, membership = split_membership(n_rows, train_size, seed)
    nonmember_idx = np.flatnonzero(membership == 0)
    cohort_idx, cohort_counts = equalised_cohort(groups, membership, seed)

    target = train_target_model(
        features[train_idx], labels[train_idx], seed=seed, epochs=epochs, batch_size=batch_size
    )
    target_prob = predict_probability(target, features)
    metrics = memorisation_metrics(
        labels[train_idx],
        target_prob[train_idx],
        labels[nonmember_idx],
        target_prob[nonmember_idx],
    )

    # Shadow schedule seeded by the target seed alone, so it is reproducible
    # from the seed and the frozen slice and nothing else.
    schedule_seed = seed + 10_000
    train_sets, excluded_counts = balanced_shadow_train_sets(
        labels, train_size=train_size, num_shadows=shadows, seed=schedule_seed
    )

    target_losses = per_example_bce(labels[cohort_idx], target_prob[cohort_idx])
    in_losses: list[list[float]] = [[] for _ in cohort_idx]
    out_losses: list[list[float]] = [[] for _ in cohort_idx]

    for shadow_id, shadow_train_idx in enumerate(train_sets):
        if len(shadow_train_idx) != train_size:
            raise AssertionError("shadow and target train sizes differ")
        shadow = train_target_model(
            features[shadow_train_idx],
            labels[shadow_train_idx],
            seed=schedule_seed + shadow_id + 1,
            epochs=epochs,
            batch_size=batch_size,
        )
        shadow_losses = per_example_bce(
            labels[cohort_idx], predict_probability(shadow, features[cohort_idx])
        )
        in_mask = np.zeros(n_rows, dtype=bool)
        in_mask[shadow_train_idx] = True
        for position, is_in in enumerate(in_mask[cohort_idx]):
            (in_losses if is_in else out_losses)[position].append(float(shadow_losses[position]))
        if shadow_id == 0 or (shadow_id + 1) % 8 == 0:
            log(f"[seed={seed}] shadow {shadow_id + 1}/{shadows}")

    min_in = min(len(v) for v in in_losses)
    min_out = min(len(v) for v in out_losses)
    if min_out != int(excluded_counts[cohort_idx].min()):
        raise AssertionError("recorded OUT observations do not match the shadow schedule")
    if min_in < min_observations or min_out < min_observations:
        raise RuntimeError(
            f"LiRA needs >= {min_observations} IN and OUT observations per example; "
            f"got IN={min_in}, OUT={min_out}"
        )

    in_matrix = np.vstack([np.asarray(v[:min_in], dtype=float) for v in in_losses])
    out_matrix = np.vstack([np.asarray(v[:min_out], dtype=float) for v in out_losses])
    cohort_membership = membership[cohort_idx]
    cohort_groups = groups[cohort_idx].astype(str)

    online = lira_online_attack(
        target_losses, cohort_membership, in_matrix, out_matrix, fprs=ATTACK_FPRS
    )
    offline = lira_offline_attack(target_losses, cohort_membership, out_matrix, fprs=ATTACK_FPRS)

    result: dict[str, object] = {
        "study": STUDY,
        "seed": int(seed),
        "smoke": bool(smoke),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_shadows": int(shadows),
        "train_size": int(train_size),
        "permutation_reps": int(permutation_reps),
        "cohort_counts": cohort_counts,
        "num_attack_examples": int(len(cohort_idx)),
        "min_in_observations": int(min_in),
        "min_out_observations": int(min_out),
        "memorisation": metrics,
        "attack_status": "completed",
        # Primary. The gate reads this and only this.
        "online": attack_summary(
            cohort_membership, cohort_groups, online.scores, permutation_reps, seed + 30_000
        ),
        # Secondary, descriptive. Never decides the gate.
        "offline": attack_summary(
            cohort_membership, cohort_groups, offline.scores, permutation_reps, seed + 40_000
        ),
        "per_example": [
            {
                "study": STUDY,
                "seed": int(seed),
                "cohort_position": int(position),
                "example_index": int(example_index),
                "group": str(cohort_groups[position]),
                "membership": int(cohort_membership[position]),
                "target_loss": float(target_losses[position]),
                "online_score": float(online.scores[position]),
                "offline_score": float(offline.scores[position]),
                "in_observations": int(len(in_losses[position])),
                "out_observations": int(len(out_losses[position])),
            }
            for position, example_index in enumerate(cohort_idx)
        ],
    }
    log(
        f"[seed={seed}] online AUC={result['online']['auc']:.4f} "
        f"p={result['online']['permutation_p_value']:.4f} | "
        f"offline AUC={result['offline']['auc']:.4f} "
        f"(train_auc={metrics['train_auc']:.4f} test_auc={metrics['test_auc']:.4f})"
    )
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def build_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    """One flat row per seed for the machine-readable CSV."""
    rows = []
    for result in payload["seeds"]:
        row = {
            "study": STUDY,
            "seed": result["seed"],
            "smoke": result["smoke"],
            "num_shadows": result["num_shadows"],
            "epochs": result["epochs"],
            "num_attack_examples": result["num_attack_examples"],
            "train_auc": result["memorisation"]["train_auc"],
            "test_auc": result["memorisation"]["test_auc"],
            "auc_gap": result["memorisation"]["auc_gap"],
            "loss_gap": result["memorisation"]["loss_gap"],
        }
        for attack in (PRIMARY_ATTACK, SECONDARY_ATTACK):
            summary = result[attack]
            row[f"{attack}_auc"] = summary["auc"]
            row[f"{attack}_tpr_at_fpr_{HEADLINE_FPR}"] = summary[f"tpr_at_fpr_{HEADLINE_FPR}"]
            row[f"{attack}_permutation_p"] = summary["permutation_p_value"]
        rows.append(row)
    return rows


def render_markdown(payload: dict[str, object]) -> str:
    """Human-readable gate report."""
    gate = payload["gate"]
    config = payload["config"]
    lines = [
        "# Study 2 - Phase 0: natural-leakage feasibility gate",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Slice: {config['task']} {payload['slice']['state']} "
        f"{payload['slice']['survey_year']} {payload['slice']['horizon']}, "
        f"{payload['slice']['rows']} rows, fingerprint `{payload['slice']['fingerprint'][:16]}`",
        f"- Sensitive attribute: `{config['sensitive_attribute']}`",
        f"- Recipe: {config['architecture']}, Adam lr={config['learning_rate']}, "
        f"batch={config['batch_size']}, epochs={config['epochs']}, "
        f"{config['regularisation']}",
        f"- Shadows per seed: {config['num_shadows']}; permutation replicates: "
        f"{config['permutation_reps']}",
        "- Primary attack: online LiRA. Secondary (descriptive): offline LiRA.",
        "",
    ]
    if not payload["preregistered"]:
        lines += [
            "> **Not a Phase 0 result.** This run is a smoke run or deviates from the "
            "frozen configuration "
            f"(`{json.dumps(config['deviates_from_frozen'])}`, smoke={config['smoke']}). "
            "Its numbers must not be read as the gate.",
            "",
        ]

    lines += [
        f"## Gate: {gate['gate']}",
        "",
        gate["rationale"],
        "",
        "| Criterion | Threshold | Observed | Result |",
        "| --- | --- | --- | --- |",
    ]
    for item in gate["criteria"]:
        observed = (
            ", ".join(_fmt(v) for v in item["observed"])
            if isinstance(item["observed"], list)
            else _fmt(item["observed"])
        )
        lines.append(
            f"| {item['description']} | {_fmt(item['threshold'])} | {observed} | "
            f"{'met' if item['passed'] else 'not met'} |"
        )

    lines += [
        "",
        "## Per-seed results",
        "",
        "| Seed | online AUC | online TPR@1%FPR | online perm p | offline AUC | "
        "offline perm p | train AUC | test AUC |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in payload["seeds"]:
        online, offline = result[PRIMARY_ATTACK], result[SECONDARY_ATTACK]
        lines.append(
            f"| {result['seed']} | {_fmt(online['auc'])} | "
            f"{_fmt(online[f'tpr_at_fpr_{HEADLINE_FPR}'])} | "
            f"{_fmt(online['permutation_p_value'])} | {_fmt(offline['auc'])} | "
            f"{_fmt(offline['permutation_p_value'])} | "
            f"{_fmt(result['memorisation']['train_auc'])} | "
            f"{_fmt(result['memorisation']['test_auc'])} |"
        )

    if gate.get("descriptive_note"):
        lines += ["", "## Descriptive reading", "", gate["descriptive_note"]]

    lines += ["", "## Continuation", "", gate["continuation"], ""]
    return "\n".join(lines)


def write_outputs(payload: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase0_gate.json").write_text(json.dumps(payload, indent=2) + "\n")

    rows = build_rows(payload)
    with (output_dir / "phase0_gate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["seed"])
        writer.writeheader()
        writer.writerows(rows)

    per_example = [record for result in payload["seeds"] for record in result.get("per_example", [])]
    if per_example:
        with (output_dir / "per_example.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_example[0]))
            writer.writeheader()
            writer.writerows(per_example)

    (output_dir / "phase0_gate.md").write_text(render_markdown(payload) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _prepare_arrays(slice_path: Path, smoke: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    frame, metadata = load_slice(slice_path)
    if smoke or metadata.get("smoke"):
        metadata = {**slice_metadata(frame), "smoke": True}
    else:
        metadata = {**verify_slice(frame, metadata or None), "smoke": False}
    features = standardise(frame[list(FEATURES)].to_numpy(dtype=float))
    labels = frame["y"].to_numpy(dtype=int)
    groups = frame["group"].astype(str).to_numpy()
    return features, labels, groups, metadata


def run_seed_command(args: argparse.Namespace) -> int:
    log = print
    features, labels, groups, slice_meta = _prepare_arrays(args.slice_path, args.smoke)
    # The frozen slice is 50,000 rows, so this is exactly TRAIN_SIZE there; the
    # min() only bites on a smaller smoke slice, which is never a Phase 0 result.
    train_size = args.train_size or min(TRAIN_SIZE, len(labels) // 2)
    config = frozen_config(args.epochs, args.shadows, args.permutation_reps, args.smoke)
    result = run_seed(
        features,
        labels,
        groups,
        seed=args.seed,
        epochs=args.epochs,
        shadows=args.shadows,
        permutation_reps=args.permutation_reps,
        batch_size=args.batch_size,
        train_size=train_size,
        smoke=args.smoke,
        log=log,
    )
    result["config"] = config
    result["slice"] = slice_meta
    result["preregistered"] = is_preregistered(config)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["python"] = platform.python_version()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"seed_{args.seed}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    log(f"wrote {destination}")
    return 0


def aggregate_command(args: argparse.Namespace) -> int:
    seed_files = sorted(args.input_dir.glob("seed_*.json"))
    results = [json.loads(path.read_text()) for path in seed_files]
    results.sort(key=lambda result: int(result["seed"]))
    if not results:
        raise SystemExit(f"no seed_*.json results under {args.input_dir}")

    fingerprints = {str(result["slice"]["fingerprint"]) for result in results}
    if len(fingerprints) != 1:
        raise SystemExit(
            "seed results were produced from different data slices; the gate is "
            "defined on a single frozen slice and cannot be aggregated across them"
        )

    gate = evaluate_gate(results)
    if args.expect_all and gate["verdict"] == "INCOMPLETE":
        # Aggregate only after every seed completes: a gate decision taken on a
        # partial set is not the preregistered gate.
        write_outputs(
            {
                "study": STUDY,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "config": results[0]["config"],
                "slice": results[0]["slice"],
                "preregistered": all(result["preregistered"] for result in results),
                "seeds": results,
                "gate": gate,
            },
            args.output_dir,
        )
        print(gate["rationale"])
        print(f"CONTINUATION GATE: {gate['gate']}")
        return 1

    payload = {
        "study": STUDY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": results[0]["config"],
        "slice": results[0]["slice"],
        "preregistered": all(result["preregistered"] for result in results),
        "seeds": results,
        "gate": gate,
    }
    write_outputs(payload, args.output_dir)

    print(render_markdown(payload))
    # The single machine-readable line the workflow greps for.
    print(f"CONTINUATION GATE: {gate['gate']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="run one target seed")
    seed.add_argument("--slice", dest="slice_path", type=Path, required=True)
    seed.add_argument("--seed", type=int, required=True)
    seed.add_argument("--epochs", type=int, default=EPOCHS)
    seed.add_argument("--shadows", type=int, default=NUM_SHADOWS)
    seed.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    seed.add_argument("--train-size", type=int, default=None)
    seed.add_argument("--permutation-reps", type=int, default=PERMUTATION_REPS)
    seed.add_argument("--output-dir", type=Path, required=True)
    seed.add_argument("--smoke", action="store_true")

    aggregate = sub.add_parser("aggregate", help="combine seeds and evaluate the gate")
    aggregate.add_argument("--input-dir", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument(
        "--expect-all",
        action="store_true",
        help="fail if any of the three frozen target seeds is missing",
    )

    args = parser.parse_args(argv)
    if args.command == "seed":
        return run_seed_command(args)
    return aggregate_command(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
