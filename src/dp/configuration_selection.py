"""Public-data utility-selection benchmark; no end-to-end private search claim.

Validation freezes a common ordinary reference and all policy decisions before
external scoring. Coverage means recommended model/cohort cases, not records.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from .acs_publication import prepare, sample_disjoint_households
from .auc_uncertainty import paired_bounds
from .proxy_shift import Cohort, StudyConfig, metrics, score, train_model
from .publication_study import synthetic_environments


def select_configuration(rows, environments, *, tau, floor, cap, use_bounds,
                         margin=0.0, allowed=None):
    """Minimum achieved epsilon, then best worst validation AUC, then stable ID."""
    if rows.empty or set(rows.partition) != {"validation"}:
        raise ValueError("selection requires nonempty validation rows only")
    if not environments or len(set(environments)) != len(environments):
        raise ValueError("invalid validation environments")
    if not np.isfinite([tau, floor, cap, margin]).all() or min(tau, cap) <= 0 or margin < 0:
        raise ValueError("invalid constraints")
    options = []
    candidates = rows if allowed is None else rows[rows.candidate_id.isin(allowed)]
    for candidate_id, part in candidates.groupby("candidate_id"):
        part = part[part.environment.isin(environments)]
        if set(part.environment) != set(environments) or part.environment.duplicated().any():
            raise ValueError("incomplete/duplicate candidate environments")
        required = ["achieved_epsilon", "gap", "auc", "gap_upper", "auc_lower"]
        if not np.isfinite(part[required].to_numpy()).all():
            raise ValueError("nonfinite candidate metrics")
        upper = part.gap_upper if use_bounds else part.gap
        lower = part.auc_lower if use_bounds else part.auc
        if (part.achieved_epsilon.max() <= cap and (upper <= tau - margin).all()
                and (lower >= floor + margin).all()):
            options.append((float(part.achieved_epsilon.max()), -float(lower.min()), candidate_id))
    return min(options)[2] if options else None


def acs_environments(config, seed, cache, data):
    year, representation = data["year"], data["representation"]
    def load(state, sizes):
        path = prepare(state, cache, year=year, representation=representation)
        with np.load(path, allow_pickle=False) as archive:
            cohorts = sample_disjoint_households(archive["X"], archive["y"], archive["groups"], sizes, seed)
        return cohorts, json.loads(path.with_suffix(".json").read_text())
    source, metadata = load(data["source"], (config.n_train, config.n_eval, config.n_eval))
    train, validation, source_test = source
    provenance = {data["source"]: metadata}
    val = {"source": validation}
    for state in data["validation"]:
        cohorts, meta = load(state, (config.n_eval,))
        val[f"state_{state}"] = cohorts[0]
        provenance[state] = meta
    for level in config.stress_levels:
        X, y, groups = validation
        rng = np.random.default_rng(np.random.SeedSequence([seed, 105]))
        changed = X.copy()
        changed[np.ix_(rng.random(len(y)) >= level, metadata["occupation_columns"])] = 0
        val[f"proxy_{level:g}"] = changed, y, groups
    def load_test():
        tests = {"source": source_test}
        for state in data["test"]:
            cohorts, meta = load(state, (config.n_eval,))
            tests[f"state_{state}"] = cohorts[0]
            provenance[state] = meta
        return tests
    return Cohort(train[0], train[1], np.zeros(len(train[1]))), val, load_test, provenance


def summarize(decisions):
    """Collapse environments before estimating seed/case-level yield and failure."""
    keys = ["seed", "strength", "policy", "uncertainty_bounds", "safety_margin"]
    cases = []
    for key, group in decisions.groupby(keys):
        abstained = bool(group.abstained.iloc[0])
        if group.abstained.nunique() != 1:
            raise ValueError("inconsistent abstention within case")
        cases.append(dict(zip(keys, key), selected=not abstained,
                          failed_any=bool(group.failed.eq(True).any()) if not abstained else False,
                          successful=bool(not abstained and not group.failed.eq(True).any()),
                          worst_auc=float(group.auc.min()) if not abstained else np.nan))
    cases = pd.DataFrame(cases)
    summaries = []
    for key, group in cases.groupby(keys[2:]):
        selected = group[group.selected]
        summaries.append(dict(zip(keys[2:], key), n_cases=len(group),
                              coverage=float(group.selected.mean()),
                              successful_yield=float(group.successful.mean()),
                              failed_yield=float((group.selected & group.failed_any).mean()),
                              conditional_failure=float(selected.failed_any.mean()) if len(selected) else None,
                              selected_worst_auc=float(selected.worst_auc.mean()) if len(selected) else None))
    return cases, pd.DataFrame(summaries)


def paired_primary(cases, primary):
    rows = cases[(cases.uncertainty_bounds == primary["bounds"])
                 & (cases.safety_margin == primary["margin"])]
    if rows.duplicated(["seed", "strength", "policy"]).any():
        raise ValueError("duplicate primary case")
    pairs = rows.pivot(index=["seed", "strength"], columns="policy", values="successful")
    a, b = primary["policy_a"], primary["policy_b"]
    if a not in pairs or b not in pairs or pairs[[a, b]].isna().any().any():
        raise ValueError("incomplete primary pairs")
    # Average strengths within seed: independent unit is a seed, not environment.
    values = (pairs[a].astype(float) - pairs[b].astype(float)).groupby(level="seed").mean()
    n, mean = len(values), float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    radius = float(t.ppf(.975, n - 1) * se) if n > 1 else None
    return {"endpoint": "successful recommendations / all cases; all external environments must pass",
            "contrast": f"{a} minus {b}", "n_seed_units": n, "difference": mean,
            "ci95_lower": mean - radius if radius is not None else None,
            "ci95_upper": mean + radius if radius is not None else None,
            "interval_supported": bool(n >= 20 and se is not None and se > 0),
            "interval_method": "paired seed-mean t approximation; degenerate/small-n intervals unsupported",
            "scope": "fixed environment split; public-data selection, not a composed DP mechanism"}


def validate_protocol(values):
    config = StudyConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in values["training"].items()})
    config.validate()
    if values["mode"] not in {"synthetic", "acs"}:
        raise ValueError("invalid mode")
    recipes = values["recipes"]
    if not recipes or len({r["id"] for r in recipes}) != len(recipes):
        raise ValueError("invalid recipe IDs")
    for r in recipes:
        if r["architecture"] not in {"mlp", "logistic"}:
            raise ValueError("invalid architecture")
        replace(config, epochs=r["epochs"], learning_rate=r["learning_rate"]).validate()
    if values["fixed_recipe"] not in {r["id"] for r in recipes}:
        raise ValueError("missing fixed recipe")
    clips = values["clip_norms"]
    if not clips or len(set(clips)) != len(clips) or not np.isfinite(clips).all() or min(clips) <= 0:
        raise ValueError("invalid clipping grid")
    if values["fixed_clip"] not in clips:
        raise ValueError("missing fixed clip")
    margins = values["safety_margins"]
    if not margins or len(set(margins)) != len(margins) or not np.isfinite(margins).all() or min(margins) < 0:
        raise ValueError("invalid safety margins")
    if values["mode"] == "acs":
        d = values["data"]
        if config.strengths != (1.0,) or not d["validation"] or not d["test"]:
            raise ValueError("invalid ACS task/split")
        states = [d["source"], *d["validation"], *d["test"]]
        if len(set(states)) != len(states):
            raise ValueError("source/validation/test states must be disjoint")
    if values.get("stage") == "prospective_external_replication":
        if len(config.seeds) < 20 or min(config.seeds) < 200:
            raise ValueError("replication needs >=20 fresh seeds 200+")
    return config


def run(values, output: Path, cache: Path):
    import torch
    config = validate_protocol(values)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    protocol_text = json.dumps(values, indent=2) + "\n"
    (output / "protocol.json").write_text(protocol_text)
    source_files = ["configuration_selection.py", "proxy_shift.py", "acs_publication.py",
                    "auc_uncertainty.py", "publication_study.py"]
    hashes = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source_files}
    records, decisions, accounting, provenance = [], [], [], []
    for strength in config.strengths:
        for seed in config.seeds:
            train, validation, load_test, metadata = (
                synthetic_environments(config, seed, strength) if values["mode"] == "synthetic"
                else acs_environments(config, seed, cache, values["data"]))
            models, descriptors, ordinary = {}, {}, {}
            for recipe in values["recipes"]:
                settings = replace(config, learning_rate=recipe["learning_rate"], epochs=recipe["epochs"])
                args = {"architecture": recipe["architecture"], "hidden_units": recipe.get("hidden_units", 16)}
                ref, meta = train_model(train, seed, settings, "ordinary", **args)
                ordinary[recipe["id"]] = ref
                accounting.append(dict(seed=seed, strength=strength, recipe=recipe["id"], **meta))
                for clip in values["clip_norms"]:
                    settings = replace(settings, clip_norm=clip)
                    clipped, meta = train_model(train, seed, settings, "clipped", **args)
                    models[recipe["id"], clip, "clipped"] = clipped
                    accounting.append(dict(seed=seed, strength=strength, recipe=recipe["id"], clip_norm=clip, **meta))
                    for epsilon in config.epsilons:
                        model, meta = train_model(train, seed, settings, "dpsgd", epsilon, **args)
                        candidate_id = f'{recipe["id"]}:C{clip:g}:E{epsilon:g}'
                        models[candidate_id] = model
                        descriptors[candidate_id] = dict(candidate_id=candidate_id, recipe=recipe["id"],
                                                        architecture=recipe["architecture"], clip_norm=clip,
                                                        requested_epsilon=epsilon, achieved_epsilon=meta["achieved_epsilon"])
                        accounting.append(dict(seed=seed, strength=strength, candidate_id=candidate_id, **meta))
                        print(f'{values["mode"]} seed={seed} {candidate_id}', flush=True)
            X, y, _ = validation["source"]
            reference_id = min(ordinary, key=lambda r: (-metrics(y, score(ordinary[r], X))["auc"], r))
            reference = ordinary[reference_id]
            family = 2 * len(descriptors) * len(validation) * len(ordinary)
            local = []
            for env, (X, y, groups) in validation.items():
                ref = score(reference, X)
                for candidate_id, descriptor in descriptors.items():
                    bound = paired_bounds(y, ref, score(models[candidate_id], X), groups, family=family)
                    local.append(dict(seed=seed, strength=strength, partition="validation", environment=env,
                                      reference_recipe=reference_id, **descriptor, **bound))
            rows = pd.DataFrame(local)
            shifted = ([f"generic_{v:g}" for v in config.stress_levels] if values["mode"] == "synthetic"
                       else [f"state_{s}" for s in values["data"]["validation"]])
            environments = {"source": ["source"], "shift": ["source", *shifted],
                            "proxy": ["source", *[f"proxy_{v:g}" for v in config.stress_levels]]}
            fixed = [c for c, d in descriptors.items() if d["recipe"] == values["fixed_recipe"]
                     and d["clip_norm"] == values["fixed_clip"]]
            clip_candidates = [c for c, d in descriptors.items() if d["recipe"] == values["fixed_recipe"]]
            logistic = [c for c, d in descriptors.items() if d["architecture"] == "logistic"]
            policies = {"epsilon_source": (fixed, "source"), "epsilon_shift": (fixed, "shift"),
                        "joint_clip_source": (clip_candidates, "source"),
                        "joint_clip_shift": (clip_candidates, "shift"),
                        "joint_recipe_source": (list(descriptors), "source"),
                        "joint_recipe_shift": (list(descriptors), "shift"),
                        "joint_recipe_proxy": (list(descriptors), "proxy")}
            if logistic:
                policies["logistic_shift"] = logistic, "shift"
            frozen = []
            for policy, (allowed, env_rule) in policies.items():
                for use_bounds in (False, True):
                    for margin in values["safety_margins"]:
                        selected = select_configuration(rows, environments[env_rule], tau=config.tau,
                                                        floor=config.utility_floor, cap=config.privacy_cap,
                                                        use_bounds=use_bounds, margin=margin, allowed=allowed)
                        frozen.append(dict(seed=seed, strength=strength, policy=policy,
                                           uncertainty_bounds=use_bounds, safety_margin=margin,
                                           selected_candidate=selected, n_search_candidates=len(allowed)))
            path = output / f"frozen_seed{seed}_strength{strength:g}.json"
            path.write_text(json.dumps({"reference_recipe": reference_id, "choices": frozen}, indent=2) + "\n")
            frozen = json.loads(path.read_text())["choices"]
            tests = load_test()  # first external load/score occurs after decision freeze
            test_rows = []
            for env, (X, y, groups) in tests.items():
                ref = score(reference, X)
                for candidate_id, d in descriptors.items():
                    matched_auc = metrics(y, score(ordinary[d["recipe"]], X))["auc"]
                    clipped_auc = metrics(y, score(models[d["recipe"], d["clip_norm"], "clipped"], X))["auc"]
                    bound = paired_bounds(y, ref, score(models[candidate_id], X), groups, family=family)
                    test_rows.append(dict(seed=seed, strength=strength, partition="test", environment=env,
                                          reference_recipe=reference_id, **d, **bound,
                                          optimization_gap=bound["reference_auc"] - matched_auc,
                                          clipping_gap=matched_auc - clipped_auc, noise_gap=clipped_auc - bound["auc"]))
            for choice in frozen:
                for env in tests:
                    if env == "source":
                        continue
                    selected = choice["selected_candidate"]
                    item = dict(**choice, environment=env, abstained=selected is None,
                                failed=None, gap=None, auc=None, loss_violation=None, floor_violation=None)
                    if selected is not None:
                        row = next(r for r in test_rows if r["candidate_id"] == selected and r["environment"] == env)
                        item.update(gap=row["gap"], auc=row["auc"],
                                    failed=row["gap"] > config.tau or row["auc"] < config.utility_floor,
                                    loss_violation=max(0, row["gap"] - config.tau),
                                    floor_violation=max(0, config.utility_floor - row["auc"]))
                    decisions.append(item)
            records.extend(local + test_rows)
            provenance.append({"seed": seed, "data": metadata})
            # Checkpoints are partial evidence, never a completed-run declaration.
            pd.DataFrame(records).to_csv(output / "metrics.csv", index=False)
            pd.DataFrame(decisions).to_csv(output / "decisions.csv", index=False)
    cases, summary = summarize(pd.DataFrame(decisions))
    cases.to_csv(output / "cases.csv", index=False)
    summary.to_csv(output / "coverage_failure.csv", index=False)
    primary = paired_primary(cases, values["primary"])
    (output / "primary_result.json").write_text(json.dumps(primary, indent=2) + "\n")
    manifest = {"status": "completed", "stage": values["stage"], "training_config": asdict(config),
                "protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
                "code_hashes_at_launch": hashes, "accounting": accounting, "data": provenance,
                "runtime": {n: importlib.metadata.version(n) for n in ["torch", "opacus", "numpy", "pandas", "scipy"]},
                "privacy_boundary": "public-data benchmark; per-candidate record DP only; tuning and release not composed",
                "reference": "common ordinary model selected using source-validation AUC only",
                "coverage": "model/cohort recommendations, not selective record predictions"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text("# Configuration-selection results\n\nStage: " + values["stage"]
                                     + "\n\n" + summary.to_markdown(index=False)
                                     + "\n\nPrimary paired comparison:\n\n```json\n"
                                     + json.dumps(primary, indent=2) + "\n```\n\n"
                                     + "Coverage/failure curves are descriptive; margins were fixed before scoring. "
                                     + "Zero selected-case failures with low coverage do not establish superiority. "
                                     + "Seeds reuse fixed archives and states; no arbitrary-shift guarantee.\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/dp-config-acs"))
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
