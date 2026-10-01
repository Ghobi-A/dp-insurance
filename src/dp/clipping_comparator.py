"""Supplementary matched AUTO-S comparator after the original ACS outcomes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .auc_uncertainty import paired_bounds
from .cluster_concentration import concentration_bounds
from .configuration_selection import acs_environments, select_configuration, summarize
from .proxy_shift import StudyConfig, score, train_model
from .publication_study import synthetic_environments
from .selection_inference import exact_paired_yield


def run(values, output, cache):
    import torch

    config = StudyConfig(**{k: tuple(v) if isinstance(v, list) else v
                           for k, v in values["training"].items()})
    config.validate()
    if values["mode"] not in {"acs", "synthetic"} or config.strengths != (1.,):
        raise ValueError("single-strength ACS/synthetic comparator required")
    if values["mode"] == "acs":
        data = values["data"]
        states = [data["source"], *data["validation"], *data["test"]]
        if not data["validation"] or not data["test"] or len(states) != len(set(states)):
            raise ValueError("disjoint nonempty environments required")
        if len(config.seeds) < 20 or min(config.seeds) < 300:
            raise ValueError("supplementary ACS uses 20 fresh seeds 300+")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.dumps(values, indent=2) + "\n"
    (output / "protocol.json").write_text(protocol)
    code_files = ["clipping_comparator.py", "proxy_shift.py", "automatic_clipping.py",
                  "configuration_selection.py", "acs_publication.py", "auc_uncertainty.py",
                  "cluster_concentration.py", "selection_inference.py", "publication_study.py"]
    hashes = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
              for n in code_files}
    records, decisions, accounting, provenance = [], [], [], []
    variants = [("standard_c1", "standard", 1.), ("standard_c5", "standard", 5.),
                ("auto_s", "auto_s", 1.)]
    for seed in config.seeds:
        train, validation, load_test, metadata = (
            acs_environments(config, seed, cache, values["data"]) if values["mode"] == "acs"
            else synthetic_environments(config, seed, 1.))
        # Proxy environments are not part of this published-clipping comparison.
        validation = {e: cohort for e, cohort in validation.items() if not e.startswith("proxy_")}
        reference, meta = train_model(train, seed, config, "ordinary")
        accounting.append(dict(seed=seed, variant="ordinary", **meta))
        clipped, models, descriptors = {}, {}, {}
        for variant, mode, clip in variants:
            settings = replace(config, clip_norm=clip)
            args = dict(clipping_mode=mode, stability=values["stability"])
            clipped[variant], meta = train_model(train, seed, settings, "clipped", **args)
            accounting.append(dict(seed=seed, variant=variant, **meta))
            for epsilon in config.epsilons:
                candidate_id = f'{variant}:E{epsilon:g}'
                models[candidate_id], meta = train_model(train, seed, settings, "dpsgd", epsilon, **args)
                descriptors[candidate_id] = dict(candidate_id=candidate_id, variant=variant,
                                                clip_norm=clip, requested_epsilon=epsilon,
                                                achieved_epsilon=meta["achieved_epsilon"])
                accounting.append(dict(seed=seed, variant=variant, candidate_id=candidate_id, **meta))
                print(f'{values["mode"]} seed={seed} {candidate_id}', flush=True)
        family = 2 * len(models) * len(validation)  # one fixed ordinary reference
        local = []
        for env, (X, y, groups) in validation.items():
            ref = score(reference, X)
            for candidate_id, d in descriptors.items():
                scores = score(models[candidate_id], X)
                sandwich = paired_bounds(y, ref, scores, groups, family=family)
                conservative = concentration_bounds(y, ref, scores, groups, family=family)
                for method, bounds in [("sandwich", sandwich), ("concentration", conservative)]:
                    local.append(dict(seed=seed, strength=1., partition="validation",
                                      environment=env, bound_rule=method, **d, **bounds))
        rows = pd.DataFrame(local)
        allowed = {name: [c for c, d in descriptors.items() if d["variant"] == name]
                   for name, _, _ in variants}
        allowed["standard_joint"] = allowed["standard_c1"] + allowed["standard_c5"]
        allowed["combined"] = list(models)
        environments = {"source": ["source"], "shift": list(validation)}
        frozen = []
        for name, bank in allowed.items():
            for env_rule, envs in environments.items():
                for method in ("point", "sandwich", "concentration"):
                    part = rows[rows.bound_rule == ("sandwich" if method == "point" else method)]
                    for margin in values["safety_margins"]:
                        selected = select_configuration(part, envs, tau=config.tau,
                                                        floor=config.utility_floor,
                                                        cap=config.privacy_cap,
                                                        use_bounds=method != "point", margin=margin,
                                                        allowed=bank)
                        frozen.append(dict(seed=seed, strength=1., policy=f'{name}_{env_rule}',
                                           bound_rule=method, uncertainty_bounds=method != "point",
                                           safety_margin=margin, selected_candidate=selected,
                                           n_search_candidates=len(bank)))
        path = output / f'frozen_seed{seed}.json'
        path.write_text(json.dumps({"choices": frozen, "reference": "fixed matched ordinary MLP",
                                    "family": family}, indent=2) + "\n")
        frozen = json.loads(path.read_text())["choices"]
        tests = load_test()  # external scoring follows every choice's disk freeze
        test_rows = []
        for env, (X, y, groups) in tests.items():
            ref = score(reference, X)
            for candidate_id, d in descriptors.items():
                result = paired_bounds(y, ref, score(models[candidate_id], X), groups, family=family)
                matched = paired_bounds(y, ref, score(clipped[d["variant"]], X), groups, family=family)
                test_rows.append(dict(seed=seed, strength=1., partition="test", environment=env,
                                      bound_rule="evaluation", **d, **result,
                                      clipping_gap=matched["gap"],
                                      noise_gap=result["gap"]-matched["gap"]))
        for choice in frozen:
            for env in tests:
                if env == "source":
                    continue
                candidate = choice["selected_candidate"]
                item = dict(**choice, environment=env, abstained=candidate is None,
                            failed=None, auc=None, gap=None)
                if candidate is not None:
                    r = next(r for r in test_rows if r["candidate_id"] == candidate and r["environment"] == env)
                    item.update(auc=r["auc"], gap=r["gap"],
                                failed=r["gap"] > config.tau or r["auc"] < config.utility_floor)
                decisions.append(item)
        records.extend(local + test_rows)
        provenance.append(dict(seed=seed, data=metadata))
        pd.DataFrame(records).to_csv(output / "metrics.csv", index=False)
        pd.DataFrame(decisions).to_csv(output / "decisions.csv", index=False)
    summaries, all_cases = [], []
    decisions = pd.DataFrame(decisions)
    for method, part in decisions.groupby("bound_rule"):
        cases, summary = summarize(part)
        all_cases.append(cases.assign(bound_rule=method))
        summaries.append(summary.assign(bound_rule=method))
    cases, summary = pd.concat(all_cases), pd.concat(summaries)
    cases.to_csv(output / "cases.csv", index=False)
    summary.to_csv(output / "coverage_failure.csv", index=False)
    contrasts = []
    for method in ("point", "sandwich", "concentration"):
        pairs = cases[(cases.bound_rule == method) & (cases.safety_margin == 0)].pivot(
            index="seed", columns="policy", values="successful")
        for a, b in [("auto_s_source", "standard_c1_source"),
                     ("auto_s_source", "standard_joint_source"),
                     ("combined_shift", "combined_source")]:
            contrasts.append(dict(bound_rule=method, contrast=f'{a} minus {b}',
                                  **exact_paired_yield(pairs[a].astype(int), pairs[b].astype(int))))
    (output / "supplementary_contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    manifest = {"status": "completed", "stage": values["stage"], "training_config": asdict(config),
                "protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest(),
                "code_hashes_at_launch": hashes, "accounting": accounting, "data": provenance,
                "runtime": {n: importlib.metadata.version(n) for n in ["torch", "opacus", "numpy", "pandas", "scipy"]},
                "privacy_boundary": "per-candidate record DP; public research search/release not composed",
                "scope": "supplementary fixed-recipe comparison; known ACS split; no tuned superiority or certified AUC claim"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text("# Supplementary AUTO-S comparator\n\n"
                                     + summary.to_markdown(index=False)
                                     + "\n\nFixed matched recipe, not a reproduction of published vision/NLP benchmarks. "
                                     + "Concentration rules on ACS are assumption-dependent sensitivities.\n")
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
