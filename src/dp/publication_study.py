"""Fresh-seed clipping development and separate ACS external utility validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .acs_publication import prepare, sample_disjoint_households
from .auc_uncertainty import paired_bounds
from .proxy_shift import (
    Cohort,
    StudyConfig,
    metrics,
    proxy_environment,
    score,
    synthetic_cohort,
    train_model,
)


def select(rows, environments, tau, floor, cap, use_bounds):
    if set(rows.partition) != {"validation"}:
        raise ValueError("selection must not access test data")
    options = []
    for epsilon, part in rows[rows.environment.isin(environments)].groupby("requested_epsilon"):
        if set(part.environment) != set(environments) or part.environment.duplicated().any():
            raise ValueError("incomplete candidate environments")
        upper = part.gap_upper if use_bounds else part.gap
        lower = part.auc_lower if use_bounds else part.auc
        if part.achieved_epsilon.max() <= cap and (upper <= tau).all() and (lower >= floor).all():
            options.append((part.achieved_epsilon.max(), epsilon))
    return float(min(options)[1]) if options else None


def synthetic_environments(config, seed, strength):
    train = synthetic_cohort(config.n_train, np.random.SeedSequence([seed, 101]), config, strength)
    validation = synthetic_cohort(config.n_eval, np.random.SeedSequence([seed, 102]), config, strength)
    test = synthetic_cohort(config.n_eval, np.random.SeedSequence([seed, 103]), config, strength)
    groups = np.arange(config.n_eval).astype(str)
    val = {"source": (validation.X, validation.y, groups)}
    for level in config.stress_levels:
        val[f"proxy_{level:g}"] = (proxy_environment(validation, level, strength, "synthetic"),
                                  validation.y, groups)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 104]))
        X = validation.X.copy()
        columns = [i for i in range(config.dimensions) if i != 1]
        X[:, columns] += rng.normal(size=(config.n_eval, len(columns))) * config.generic_noise * (1 - level)
        val[f"generic_{level:g}"] = (np.clip(X, -1, 1), validation.y, groups)
    def load_test():
        return {"source": (test.X, test.y, groups), **{
            f"proxy_{level:g}": (proxy_environment(test, level, strength, "synthetic"), test.y, groups)
            for level in config.test_levels}}
    return train, val, load_test, {}


def acs_environments(config, seed, cache):
    def load(state, sizes):
        path = prepare(state, cache)
        with np.load(path, allow_pickle=False) as data:
            cohorts = sample_disjoint_households(data["X"], data["y"], data["groups"], sizes, seed)
        return cohorts, json.loads(path.with_suffix(".json").read_text())
    source, metadata = load("CA", (config.n_train, config.n_eval, config.n_eval))
    train, validation, source_test = source
    val = {"source": validation}
    provenance = {"CA": metadata}
    for state in ("OR", "WA"):
        cohorts, meta = load(state, (config.n_eval,))
        val[f"state_{state}"] = cohorts[0]
        provenance[state] = meta
    occupation = metadata["occupation_columns"]
    for level in config.stress_levels:
        X, y, groups = validation
        rng = np.random.default_rng(np.random.SeedSequence([seed, 105]))
        # Whole occupation one-hot block zeroed, not a scalar category code.
        changed = X.copy()
        removed = rng.random(len(y)) >= level
        changed[np.ix_(removed, occupation)] = 0
        val[f"proxy_{level:g}"] = changed, y, groups
    def load_test():
        environments = {"source": source_test}
        for state in ("NV", "AZ"):
            cohorts, meta = load(state, (config.n_eval,))
            environments[f"state_{state}"] = cohorts[0]
            provenance[state] = meta
        return environments
    return Cohort(train[0], train[1], np.zeros(len(train[1]))), val, load_test, provenance


def run(values: dict, output: Path, cache: Path):
    import torch

    code_hashes_at_launch = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("publication_study.py", "acs_publication.py", "auc_uncertainty.py", "proxy_shift.py")}

    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    mode = values["mode"]
    if mode not in {"synthetic_clipping", "acs_external"}:
        raise ValueError("invalid study mode")
    config = StudyConfig(**{k: tuple(v) if isinstance(v, list) else v
                            for k, v in values["training"].items()})
    config.validate()
    stage = values.get("stage", "development_external_pilot")
    if stage not in {"development_external_pilot", "prospective_confirmation"}:
        raise ValueError("invalid study stage")
    if stage == "prospective_confirmation" and (len(config.seeds) < 20 or min(config.seeds) < 100):
        raise ValueError("confirmation requires >=20 new seeds numbered 100+")
    if set(config.seeds) & set(range(5)):
        raise ValueError("use fresh seeds, not initial pilot seeds 0-4")
    clips = values["clip_norms"]
    if not clips or len(clips) != len(set(clips)) or any(not np.isfinite(c) or c <= 0 for c in clips):
        raise ValueError("invalid clip grid")
    alpha = values.get("alpha", 0.05)
    if not 0 < alpha < 1:
        raise ValueError("invalid alpha")
    if mode == "acs_external" and config.strengths != (1.0,):
        raise ValueError("ACS does not have synthetic strength levels")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(values, indent=2) + "\n")
    records, decisions, accounting, provenance = [], [], [], []
    for strength in config.strengths:
        for seed in config.seeds:
            train, validation, load_test, data_metadata = (
                synthetic_environments(config, seed, strength) if mode == "synthetic_clipping"
                else acs_environments(config, seed, cache))
            models, local_validation, frozen = {}, [], []
            family = 2 * len(config.epsilons) * len(clips) * len(validation)
            rules = {"source_only": ["source"],
                     "proxy_stress": ["source", *(f"proxy_{r:g}" for r in config.stress_levels)],
                     "generic_shift" if mode == "synthetic_clipping" else "natural_shift":
                         ["source", *(f"generic_{r:g}" for r in config.stress_levels)]
                         if mode == "synthetic_clipping" else ["source", "state_OR", "state_WA"]}
            for clip in clips:
                settings = replace(config, clip_norm=clip)
                reference, meta = train_model(train, seed, settings, "ordinary")
                accounting.append({"seed": seed, "strength": strength, "clip_norm": clip, **meta})
                models[clip, "ordinary", None] = reference
                clipped, meta = train_model(train, seed, settings, "clipped")
                models[clip, "clipped", None] = clipped
                accounting.append({"seed": seed, "strength": strength, "clip_norm": clip, **meta})
                baseline = {name: score(reference, cohort[0]) for name, cohort in validation.items()}
                for epsilon in config.epsilons:
                    model, meta = train_model(train, seed, settings, "dpsgd", epsilon)
                    models[clip, "dpsgd", epsilon] = model
                    accounting.append({"seed": seed, "strength": strength, "clip_norm": clip, **meta})
                    for environment, (X, y, groups) in validation.items():
                        bounds = paired_bounds(y, baseline[environment], score(model, X), groups,
                                               alpha=alpha, family=family)
                        local_validation.append({"seed": seed, "strength": strength, "clip_norm": clip,
                                                 "partition": "validation", "environment": environment,
                                                 "requested_epsilon": epsilon,
                                                 "achieved_epsilon": meta["achieved_epsilon"], **bounds})
                    print(f"{mode} seed={seed} strength={strength} clip={clip} epsilon={epsilon}", flush=True)
                rows = pd.DataFrame(local_validation)
                rows = rows[rows.clip_norm == clip]
                for rule, environments in rules.items():
                    for bounds in (False, True):
                        selected = select(rows, environments, config.tau, config.utility_floor,
                                          config.privacy_cap, bounds)
                        frozen.append({"seed": seed, "strength": strength, "clip_norm": clip,
                                       "rule": rule, "uncertainty_bounds": bounds,
                                       "selected_epsilon": selected})
            # Persist choices before loading external test data or evaluating test scores.
            frozen_path = output / f"frozen_seed{seed}_strength{strength:g}.json"
            frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
            frozen = json.loads(frozen_path.read_text())
            test = load_test()
            test_rows = []
            for clip in clips:
                source_gaps = {}
                for environment, (X, y, groups) in test.items():
                    ref = score(models[clip, "ordinary", None], X)
                    clipped_scores = score(models[clip, "clipped", None], X)
                    for epsilon in config.epsilons:
                        candidate = score(models[clip, "dpsgd", epsilon], X)
                        bound = paired_bounds(y, ref, candidate, groups, alpha=alpha, family=family)
                        clipped_auc = metrics(y, clipped_scores)["auc"]
                        row = {"seed": seed, "strength": strength, "clip_norm": clip,
                               "partition": "test", "environment": environment,
                               "requested_epsilon": epsilon, **bound,
                               "clipping_gap": bound["reference_auc"] - clipped_auc,
                               "noise_gap": clipped_auc - bound["auc"],
                               "pr_auc": metrics(y, candidate)["pr_auc"]}
                        if environment == "source":
                            source_gaps[epsilon] = row
                        row["total_interaction"] = row["gap"] - source_gaps[epsilon]["gap"]
                        row["clipping_interaction"] = row["clipping_gap"] - source_gaps[epsilon]["clipping_gap"]
                        row["noise_interaction"] = row["noise_gap"] - source_gaps[epsilon]["noise_gap"]
                        test_rows.append(row)
            for choice in frozen:
                for environment in test:
                    if environment == "source":
                        continue
                    selected = choice["selected_epsilon"]
                    item = {**choice, "environment": environment, "abstained": selected is None,
                            "failed": None, "loss_violation": None, "floor_violation": None}
                    if selected is not None:
                        row = next(r for r in test_rows if r["clip_norm"] == choice["clip_norm"]
                                   and r["environment"] == environment and r["requested_epsilon"] == selected)
                        item.update(gap=row["gap"], auc=row["auc"],
                                    failed=row["gap"] > config.tau or row["auc"] < config.utility_floor,
                                    loss_violation=max(0, row["gap"] - config.tau),
                                    floor_violation=max(0, config.utility_floor - row["auc"]))
                    decisions.append(item)
            records.extend(local_validation + test_rows)
            provenance.append({"seed": seed, "data": data_metadata})
    frame, decision_frame = pd.DataFrame(records), pd.DataFrame(decisions)
    frame.to_csv(output / "metrics.csv", index=False)
    decision_frame.to_csv(output / "decisions.csv", index=False)
    manifest = {"status": stage, "protocol": values,
                "accounting": accounting, "data": provenance,
                "code_hashes_at_launch": code_hashes_at_launch,
                "runtime": {"python": platform.python_version(),
                            **{name: importlib.metadata.version(name) for name in
                               ("torch", "opacus", "numpy", "pandas", "scipy", "scikit-learn")}},
                "code_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("publication_study.py", "acs_publication.py", "auc_uncertainty.py", "proxy_shift.py")},
                "training_config": asdict(config),
                "inference": "asymptotic cluster influence bounds; finite validation family only; not arbitrary-shift certified"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    lines = ["# Publication-stage run", "", f"Mode: {mode}. Stage: {stage}.",
             "", "| Clipping norm | Rule | Bounds | Selected cases | Abstained cases | Failed evaluations / selected evaluations |",
             "|---|---|---|---:|---:|---:|"]
    for (clip, rule, bounds), group in decision_frame.groupby(["clip_norm", "rule", "uncertainty_bounds"]):
        cases = group.drop_duplicates(["seed", "strength"])
        selected = group[~group.abstained]
        lines.append(f"| {clip} | {rule} | {bounds} | {(~cases.abstained).sum()} | "
                     f"{cases.abstained.sum()} | {int(selected.failed.sum()) if len(selected) else 0}/{len(selected)} |")
    lines += ["", "Evaluations share models; seeds are the training replication unit.",
              "Bounds are asymptotic with household-cluster variance on ACS and a Bonferroni",
              "validation family across budgets, clipping norms, environments and endpoints.",
              "They describe validation uncertainty, not guarantees in unseen states.",
              "ACS is a fixed hashed-representation, unweighted-income prediction variant.",
              "Natural state shift changes many variables; occupation masking is hypothetical.", ""]
    (output / "report.md").write_text("\n".join(lines))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/dp-publication-acs"))
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
