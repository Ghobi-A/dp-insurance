"""Prospective real-data LiRA validation on the insurance ``high_cost`` task.

Protocol: ``docs/LIRA_VALIDATION_PROTOCOL.md``. Design and statistics live in
``dp.lira_study``; this script trains models and writes artefacts.

Train (resumable; one small ``.npz`` per model, existing files are reused)::

    python research/lira_real_data_study.py train --output-dir reports/lira_validation/v1

Analyse saved measurements::

    python research/lira_real_data_study.py analyse --input-dir reports/lira_validation/v1

``--pilot`` uses the pilot seed and a separate directory. Pilot output is
stamped and refused by the confirmatory analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from dp import lira_study as ls

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "insurance.csv"


def runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "torch", "opacus", "scikit-learn", "pandas", "scipy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def load_data() -> tuple[np.ndarray, np.ndarray, str]:
    raw = DATA_PATH.read_bytes()
    df = pd.read_csv(DATA_PATH)
    if len(df) != ls.N_RECORDS:
        raise RuntimeError("insurance.csv does not have the protocol's record count")
    X, y = ls.encode_fixed(df)
    return X, y, hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def solve_noise_multiplier(epsilon: float, sample_rate: float, steps: int) -> float:
    """Noise multiplier for the mechanism that will actually run (q, steps)."""
    from opacus.accountants.utils import get_noise_multiplier

    return float(
        get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=ls.DELTA,
            sample_rate=sample_rate,
            steps=steps,
            accountant="rdp",
        )
    )


def train_and_measure(X: np.ndarray, y: np.ndarray, members: np.ndarray, condition: str,
                      seed: int, epochs: int) -> tuple[np.ndarray, dict[str, object]]:
    """Train one model and return its signed logit margin on every record.

    ``margin = (2y - 1) * logit``; the per-example loss is ``softplus(-margin)``.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    Xt, yt = torch.from_numpy(X[members]), torch.from_numpy(y[members])
    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=ls.BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    layers: list[nn.Module] = []
    width = X.shape[1]
    for units in ls.HIDDEN_UNITS:
        layers += [nn.Linear(width, units), nn.ReLU()]
        width = units
    model = nn.Sequential(*layers, nn.Linear(width, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=ls.LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    meta: dict[str, object] = {"condition": condition, "seed": int(seed), "epochs": int(epochs),
                               "train_size": int(members.sum()), "noise_multiplier": None,
                               "achieved_epsilon": None, "sample_rate": None}

    engine = None
    if condition != "nonprivate":
        from opacus import PrivacyEngine

        sample_rate = 1.0 / len(loader)
        noise = 0.0
        if condition in ls.REQUESTED_EPSILON:
            noise = solve_noise_multiplier(
                ls.REQUESTED_EPSILON[condition], sample_rate, epochs * len(loader)
            )
        engine = PrivacyEngine(accountant="rdp")
        model, optimizer, loader = engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=noise, max_grad_norm=ls.MAX_GRAD_NORM, poisson_sampling=True,
        )
        meta.update(noise_multiplier=float(noise), sample_rate=float(loader.sample_rate),
                    max_grad_norm=ls.MAX_GRAD_NORM)

    model.train()
    steps = 0
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            if len(xb) == 0:  # empty Poisson batch: the accountant still steps
                optimizer.step()
                steps += 1
                continue
            loss_fn(model(xb).squeeze(-1), yb).backward()
            optimizer.step()
            steps += 1
    meta["steps"] = int(steps)
    if engine is not None and condition in ls.REQUESTED_EPSILON:
        meta["achieved_epsilon"] = float(engine.get_epsilon(ls.DELTA))

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).squeeze(-1).double().numpy()
    return (2.0 * y.astype(float) - 1.0) * logits, meta


def _model_path(output_dir: Path, condition: str, index: int) -> Path:
    return output_dir / "models" / condition / f"model_{index:03d}.npz"


def _validate_saved_model(
    path: Path, *, condition: str, index: int, expected_members: np.ndarray,
    seed: int, pilot: bool, epochs: int, data_hash: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fail closed on stale/corrupt checkpoints before resume or analysis.

    A filename is not evidence that a model belongs to this experiment. In
    particular, never silently reuse pilot models or files from another data
    snapshot, condition, model index, or frozen protocol.
    """
    try:
        with np.load(path, allow_pickle=False) as saved:
            margins = saved["margins"]
            members = saved["members"]
            meta = json.loads(str(saved["meta"]))
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        raise SystemExit(f"invalid saved model {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SystemExit(f"invalid saved model {path}: metadata is not an object")

    required = {
        "condition": condition,
        "model_index": index,
        "design_seed": seed,
        "pilot": pilot,
        "epochs": epochs,
        "seed": ls.model_seed(seed, index),
        "protocol_hash": ls.protocol_hash(),
        "data_sha256": data_hash,
        "train_size": int(expected_members.sum()),
    }
    for key, value in required.items():
        if meta.get(key) != value or (key == "pilot" and type(meta.get(key)) is not bool):
            raise SystemExit(
                f"saved model {path} has mismatched {key}; refusing to reuse it"
            )
    if members.shape != expected_members.shape or members.dtype != np.dtype(bool):
        raise SystemExit(f"saved model {path} has invalid membership shape/type")
    if not np.array_equal(members, expected_members):
        raise SystemExit(f"saved model {path} has mismatched membership design")
    if (margins.shape != (ls.N_RECORDS,)
            or not np.issubdtype(margins.dtype, np.number)
            or not np.isfinite(margins).all()):
        raise SystemExit(f"saved model {path} has invalid/nonfinite margins")
    return margins, members, meta


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quarantine(output_dir: Path, path: Path, reason: str) -> Path:
    """Move a rejected checkpoint aside and log it. Nothing is ever deleted."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = output_dir / "rejected" / path.parent.name / f"{path.stem}.{stamp}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"original_path": str(path.relative_to(output_dir)),
              "quarantined_path": str(target.relative_to(output_dir)),
              "sha256": _sha256(path), "reason": reason, "utc": stamp}
    path.replace(target)
    with (output_dir / "rejected_artifacts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return target


def run_inventory(args: argparse.Namespace) -> None:
    """Read-only audit of every expected checkpoint; writes artifact_inventory.json."""
    input_dir = Path(args.input_dir)
    manifest = json.loads((input_dir / "run_manifest.json").read_text())
    pilot = bool(manifest["pilot"])
    seed = ls.PILOT_SEED if pilot else ls.STUDY_SEED
    _, _, data_hash = load_data()
    design = ls.membership_matrix(seed)
    entries, counts = [], {"valid": 0, "missing": 0, "invalid": 0}
    for condition in ls.CONDITIONS:
        for index in range(ls.N_TARGETS + ls.N_SHADOWS):
            path = _model_path(input_dir, condition, index)
            entry: dict[str, object] = {"condition": condition, "model_index": index}
            if not path.exists():
                entry["status"] = "missing"
            else:
                entry["sha256"] = _sha256(path)
                try:
                    _, _, meta = _validate_saved_model(
                        path, condition=condition, index=index, expected_members=design[index],
                        seed=seed, pilot=pilot, epochs=int(manifest["epochs"]),
                        data_hash=data_hash,
                    )
                    entry.update(status="valid", seconds=meta["seconds"],
                                 achieved_epsilon=meta["achieved_epsilon"])
                except SystemExit as exc:
                    entry.update(status="invalid", reason=str(exc))
            counts[str(entry["status"])] += 1
            entries.append(entry)
    rejected_log = input_dir / "rejected_artifacts.jsonl"
    rejected = ([json.loads(line) for line in rejected_log.read_text().splitlines() if line]
                if rejected_log.exists() else [])
    stray = sorted(str(p.relative_to(input_dir)) for p in (input_dir / "models").rglob("*")
                   if p.is_file() and not p.name.startswith("model_") or p.name.endswith(".tmp.npz"))
    payload = {"protocol_hash": ls.protocol_hash(), "data_sha256": data_hash, "pilot": pilot,
               "expected_models": len(entries), "counts": counts, "rejected": rejected,
               "stray_files": stray, "models": entries}
    (input_dir / "artifact_inventory.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"counts": counts, "rejected": len(rejected), "stray_files": stray}))
    if args.require_complete and (counts["missing"] or counts["invalid"]):
        raise SystemExit("inventory is not complete and valid")


def _train_one(job: tuple[str, int, str, int, bool, int, str]) -> tuple[str, int, float]:
    condition, index, output_dir, seed, pilot, epochs, data_hash = job
    import torch

    torch.set_num_threads(1)
    path = _model_path(Path(output_dir), condition, index)
    X, y, _ = load_data()
    members = ls.membership_matrix(seed)[index]
    started = time.time()
    margins, meta = train_and_measure(X, y, members, condition, ls.model_seed(seed, index), epochs)
    meta.update(model_index=int(index), design_seed=int(seed), pilot=bool(pilot),
                protocol_hash=ls.protocol_hash(), data_sha256=data_hash,
                seconds=round(time.time() - started, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, margins=margins, members=members, meta=json.dumps(meta))
    tmp.replace(path)
    return condition, index, meta["seconds"]


def run_train(args: argparse.Namespace) -> None:
    import multiprocessing as mp

    output_dir = Path(args.output_dir)
    seed = ls.PILOT_SEED if args.pilot else ls.STUDY_SEED
    epochs = args.epochs if args.epochs is not None else ls.EPOCHS
    if not args.pilot and (epochs != ls.EPOCHS or args.models is not None):
        raise SystemExit("epochs/models overrides are pilot-only")
    total = ls.N_TARGETS + ls.N_SHADOWS
    indices = range(total) if args.models is None else range(args.models)
    _, _, data_hash = load_data()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    run_identity = {
        "protocol_hash": ls.protocol_hash(), "design_seed": seed,
        "pilot": bool(args.pilot), "epochs": epochs, "data_sha256": data_hash,
    }
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except (ValueError, OSError) as exc:
            raise SystemExit(f"invalid existing run manifest: {exc}") from exc
        for key, value in run_identity.items():
            if existing_manifest.get(key) != value:
                raise SystemExit(f"existing run manifest has mismatched {key}; refusing resume")
    else:
        manifest_path.write_text(json.dumps({
            "protocol": ls.protocol_dict(), **run_identity,
            "runtime": runtime_versions(), "workers": args.workers,
        }, indent=2, default=list))

    # Validate existing artefacts rather than trusting filenames. This also
    # detects mixed old/new runs before launching any new training workers.
    design = ls.membership_matrix(seed)
    jobs = []
    for condition in args.conditions:
        for index in indices:
            path = _model_path(output_dir, condition, index)
            if path.exists():
                try:
                    _validate_saved_model(
                        path, condition=condition, index=index,
                        expected_members=design[index], seed=seed, pilot=bool(args.pilot),
                        epochs=epochs, data_hash=data_hash,
                    )
                    continue
                except SystemExit as exc:
                    if not args.quarantine_invalid:
                        raise
                    _quarantine(output_dir, path, str(exc))
            jobs.append((condition, index, str(output_dir), seed,
                         bool(args.pilot), epochs, data_hash))
    print(f"{len(jobs)} models to train ({total * len(args.conditions) - len(jobs)} reused)",
          flush=True)
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for done, (condition, index, seconds) in enumerate(pool.imap_unordered(_train_one, jobs), 1):
            if done % 8 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {condition} model {index} {seconds}s "
                      f"elapsed {time.time() - started:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def load_condition(input_dir: Path, condition: str, allow_pilot: bool):
    total = ls.N_TARGETS + ls.N_SHADOWS
    margins, members, metas = [], [], []
    _, _, data_hash = load_data()
    seed = None
    pilot = None
    epochs = None
    design = None
    for index in range(total):
        path = _model_path(input_dir, condition, index)
        # Read only the first metadata record to establish whether this is
        # a declared pilot; the shared validator checks each complete file.
        if index == 0:
            try:
                with np.load(path, allow_pickle=False) as saved:
                    first_meta = json.loads(str(saved["meta"]))
            except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
                raise SystemExit(f"invalid saved model {path}: {exc}") from exc
            if not isinstance(first_meta, dict) or type(first_meta.get("pilot")) is not bool:
                raise SystemExit(f"invalid saved model {path}: missing pilot marker")
            pilot = first_meta["pilot"]
            if pilot and not allow_pilot:
                raise SystemExit("pilot measurements cannot enter the confirmatory analysis")
            seed = ls.PILOT_SEED if pilot else ls.STUDY_SEED
            epochs = first_meta.get("epochs") if pilot else ls.EPOCHS
            design = ls.membership_matrix(seed)
        margin, member, meta = _validate_saved_model(
            path, condition=condition, index=index, expected_members=design[index],
            seed=seed, pilot=pilot, epochs=epochs, data_hash=data_hash,
        )
        margins.append(margin)
        members.append(member)
        metas.append(meta)
    return np.asarray(margins), np.asarray(members), metas


def analyse_condition(margins: np.ndarray, members: np.ndarray, groups: np.ndarray):
    losses = np.logaddexp(0.0, -margins)
    T = ls.N_TARGETS
    shadow_losses, shadow_members = losses[T:], members[T:]
    configs = [{"attack": "loss_threshold"}, *ls.configurations()]
    rows, scores_primary = [], {}
    for t in range(T):
        membership = members[t].astype(int)
        wrong = members[(t + 1) % T].astype(int)  # another target's independent coins
        for config in configs:
            result = ls.score_target(losses[t], membership, shadow_losses, shadow_members, config)
            control = ls.score_target(losses[t], wrong, shadow_losses, shadow_members, config)
            n_mem, n_non = int(membership.sum()), int((1 - membership).sum())
            row = {"target": t, "config": ls.config_name(config), "auc": result.auc,
                   "control_auc": control.auc, "n_members": n_mem, "n_nonmembers": n_non}
            for fpr in ls.FPRS:
                tpr = result.tpr_at_fpr[fpr]
                lo, hi = ls.tpr_interval(tpr, n_mem)
                row.update({f"tpr_at_{fpr}": tpr, f"tpr_at_{fpr}_lo": lo, f"tpr_at_{fpr}_hi": hi,
                            f"false_positives_allowed_at_{fpr}": int(np.floor(fpr * n_non))})
            rows.append(row)
            if config in (ls.PRIMARY, ls.BASELINE):
                scores_primary[(t, ls.config_name(config))] = (result.scores, result.membership)
    per_target = pd.DataFrame(rows)

    # Diagnostics (exploratory): Gaussian calibration, saturation, heteroscedasticity.
    out = ls.out_matrix(shadow_losses, shadow_members, ls.N_SHADOWS)
    from dp.attacks import logit_confidence_from_bce

    logit_out = logit_confidence_from_bce(out)
    sd_logit = logit_out.std(axis=1, ddof=1)
    k = out.shape[1]
    z_nonmember = []
    for t in range(T):
        non = ~members[t]
        stat = logit_confidence_from_bce(losses[t])
        z_nonmember.append(((stat - logit_out.mean(axis=1)) / np.maximum(sd_logit, 1e-12))[non])
    z_nonmember = np.concatenate(z_nonmember) / np.sqrt(1 + 1 / k)
    finite = z_nonmember[np.abs(z_nonmember) < 1e6]
    dup_sizes = np.bincount(groups)
    diagnostics = {
        "saturated_fraction_target_losses": float(np.mean(np.exp(-losses[:T]) > 1 - 1e-12)),
        "saturated_fraction_shadow_losses": float(np.mean(np.exp(-shadow_losses) > 1 - 1e-12)),
        "per_example_out_sd_logit_p10_p50_p90": np.percentile(sd_logit, [10, 50, 90]).tolist(),
        "per_example_out_sd_logit_ratio_p90_p10": float(
            np.percentile(sd_logit, 90) / max(np.percentile(sd_logit, 10), 1e-12)),
        "nonmember_z_mean": float(finite.mean()), "nonmember_z_sd": float(finite.std(ddof=1)),
        "nonmember_z_abs_gt_3_fraction": float(np.mean(np.abs(finite) > 3)),
        "nonmember_z_excluded_nonfinite_scale": int(z_nonmember.size - finite.size),
        "duplicate_records_in_groups_larger_than_one": int(dup_sizes[dup_sizes > 1].sum()),
        "target_train_accuracy": float(np.mean([(margins[t] > 0)[members[t]].mean() for t in range(T)])),
        "target_heldout_accuracy": float(np.mean([(margins[t] > 0)[~members[t]].mean() for t in range(T)])),
    }
    return per_target, diagnostics, scores_primary, losses


def summarise(per_target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in per_target.groupby("config", sort=False):
        group = group.sort_values("target")
        n_mem, n_non = int(group.n_members.iloc[0]), int(group.n_nonmembers.iloc[0])
        auc = ls.mean_t_interval(group.auc)
        tpr = ls.mean_t_interval(group[f"tpr_at_{ls.HEADLINE_FPR}"])
        det = ls.stouffer_detection(group.auc, n_mem, n_non)
        rows.append({
            "config": name, "n_targets": len(group), "members_per_target": n_mem,
            "nonmembers_per_target": n_non,
            "mean_auc": auc["mean"], "auc_ci_low": auc["ci_low"], "auc_ci_high": auc["ci_high"],
            "mean_tpr_at_1pct": tpr["mean"], "tpr_ci_low": tpr["ci_low"],
            "tpr_ci_high": tpr["ci_high"],
            "mean_tpr_at_0.1pct_descriptive": float(group["tpr_at_0.001"].mean()),
            "mean_control_auc": float(group.control_auc.mean()),
            "control_stouffer_p": ls.stouffer_detection(group.control_auc, n_mem, n_non)["p_one_sided"],
            "stouffer_z": det["z"], "stouffer_p": det["p_one_sided"],
        })
    return pd.DataFrame(rows)


def paired_contrast(per_target: pd.DataFrame, metric: str) -> dict[str, float]:
    a = per_target[per_target.config == ls.config_name(ls.PRIMARY)].sort_values("target")[metric]
    b = per_target[per_target.config == ls.config_name(ls.BASELINE)].sort_values("target")[metric]
    diff = a.to_numpy() - b.to_numpy()
    out = ls.mean_t_interval(diff)
    out["signflip_p_two_sided"] = ls.signflip_pvalue(diff)
    out["targets_improved"] = int((diff > 0).sum())
    return out


def vulnerable_records(losses, members, scores_primary, top: int = 10):
    """Exploratory: records whose primary score separates most by membership."""
    T, n = ls.N_TARGETS, losses.shape[1]
    name = ls.config_name(ls.PRIMARY)
    scores = np.vstack([scores_primary[(t, name)][0] for t in range(T)])
    scores = np.clip(scores, -50, 50)
    mem = members[:T]
    gap = np.full(n, np.nan)
    for i in range(n):
        if 2 <= mem[:, i].sum() <= T - 2:
            gap[i] = scores[mem[:, i], i].mean() - scores[~mem[:, i], i].mean()
    order = np.argsort(-np.nan_to_num(gap, nan=-np.inf))[:top]
    return [{"record": int(i), "mean_score_gap_member_minus_nonmember": float(gap[i]),
             "times_member": int(mem[:, i].sum())} for i in order]


def run_analyse(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    X, y, data_hash = load_data()
    groups = ls.duplicate_groups(X, y)
    manifest = json.loads((input_dir / "run_manifest.json").read_text())
    pilot = manifest.get("pilot")
    if type(pilot) is not bool:
        raise SystemExit("run manifest is missing a valid pilot marker")
    expected_identity = {
        "protocol_hash": ls.protocol_hash(),
        "data_sha256": data_hash,
        "design_seed": ls.PILOT_SEED if pilot else ls.STUDY_SEED,
    }
    if not pilot:
        expected_identity["epochs"] = ls.EPOCHS
    elif not args.allow_pilot:
        raise SystemExit("pilot measurements cannot enter the confirmatory analysis")
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise SystemExit(f"run manifest has mismatched {key}; analysis refused")
    results: dict[str, object] = {
        "protocol_hash": ls.protocol_hash(), "pilot": bool(manifest["pilot"]),
        "data_sha256": data_hash, "runtime": manifest["runtime"],
        "analysis_runtime": runtime_versions(), "conditions": {},
    }
    all_targets, all_summary = [], []
    for condition in args.conditions:
        margins, members, metas = load_condition(input_dir, condition, args.allow_pilot)
        per_target, diagnostics, scores_primary, losses = analyse_condition(margins, members, groups)
        summary = summarise(per_target)
        per_target.insert(0, "condition", condition)
        summary.insert(0, "condition", condition)
        all_targets.append(per_target)
        all_summary.append(summary)
        eps = [m["achieved_epsilon"] for m in metas if m["achieved_epsilon"] is not None]
        results["conditions"][condition] = {
            "achieved_epsilon_min_max": [min(eps), max(eps)] if eps else None,
            "noise_multiplier": metas[0]["noise_multiplier"],
            "sample_rate": metas[0]["sample_rate"], "steps": metas[0]["steps"],
            "train_seconds_total": float(sum(m["seconds"] for m in metas)),
            "paired_primary_minus_baseline_auc": paired_contrast(per_target, "auc"),
            "paired_primary_minus_baseline_tpr_at_1pct": paired_contrast(per_target, "tpr_at_0.01"),
            "diagnostics": diagnostics,
            "exploratory_vulnerable_records": vulnerable_records(losses, members, scores_primary),
        }
        np.savez_compressed(
            input_dir / f"scores_{condition}.npz",
            **{f"t{t}_{'primary' if name == ls.config_name(ls.PRIMARY) else 'baseline'}": s
               for (t, name), (s, _) in scores_primary.items()},
            membership=members[: ls.N_TARGETS],
        )
    summary = pd.concat(all_summary, ignore_index=True)
    for name in (ls.config_name(ls.PRIMARY), ls.config_name(ls.BASELINE)):
        mask = summary.config == name
        summary.loc[mask, "stouffer_p_holm_across_conditions"] = ls.holm(summary.loc[mask, "stouffer_p"])
    pd.concat(all_targets, ignore_index=True).to_csv(input_dir / "per_target.csv", index=False)
    summary.to_csv(input_dir / "summary.csv", index=False)
    (input_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(summary[["condition", "config", "mean_auc", "auc_ci_low", "auc_ci_high",
                   "mean_tpr_at_1pct", "mean_control_auc", "stouffer_p"]].to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--conditions", nargs="+", default=list(ls.CONDITIONS), choices=ls.CONDITIONS)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--pilot", action="store_true")
    train.add_argument("--epochs", type=int, default=None, help="pilot only")
    train.add_argument("--models", type=int, default=None, help="pilot only: first N models")
    train.add_argument("--quarantine-invalid", action="store_true",
                       help="move invalid checkpoints to rejected/ (logged) and retrain them; "
                            "default is to stop")
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--input-dir", required=True)
    inventory.add_argument("--require-complete", action="store_true")
    analyse = sub.add_parser("analyse")
    analyse.add_argument("--input-dir", required=True)
    analyse.add_argument("--conditions", nargs="+", default=list(ls.CONDITIONS), choices=ls.CONDITIONS)
    analyse.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args(argv)
    {"train": run_train, "inventory": run_inventory, "analyse": run_analyse}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
