"""Exploratory utility-recommendation study; public-data research, not a DP release.

Independent train/validation/test partitions, fixed row-wise preparation,
matched Poisson sampling, ordinary/clipping-only/noisy controls and held-out
proxy failures. See docs/PROXY_SHIFT_STUDY.md before interpreting results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


@dataclass(frozen=True)
class StudyConfig:
    dataset: str = "synthetic"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    strengths: tuple[float, ...] = (0.6, 0.75, 0.9, 0.98)
    epsilons: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    stress_levels: tuple[float, ...] = (0.5, 0.75)
    test_levels: tuple[float, ...] = (0.0, 0.25)
    n_train: int = 1000
    n_eval: int = 2000
    dimensions: int = 8
    stable_signal: float = 0.7
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 0.1
    clip_norm: float = 1.0
    delta: float = 1e-5
    tau: float = 0.03
    utility_floor: float = 0.7
    privacy_cap: float = 8.0
    generic_noise: float = 0.5

    def validate(self) -> None:
        if self.dataset not in {"synthetic", "insurance"}:
            raise ValueError("dataset must be synthetic or insurance")
        for name in ("seeds", "strengths", "epsilons", "stress_levels", "test_levels"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and unique")
        if any(not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if any(not math.isfinite(x) or not 0.5 <= x <= 1 for x in self.strengths):
            raise ValueError("strengths must be in [0.5, 1]")
        if any(not math.isfinite(x) or not 0 <= x <= 1
               for x in (*self.stress_levels, *self.test_levels)):
            raise ValueError("proxy retention levels must be in [0, 1]")
        if set(self.stress_levels) & set(self.test_levels):
            raise ValueError("stress and held-out levels must be disjoint")
        if any(not math.isfinite(e) or e <= 0 for e in self.epsilons):
            raise ValueError("epsilons must be finite and positive")
        if min(self.n_train, self.n_eval) < 32 or self.dimensions < 2:
            raise ValueError("partitions need >=32 rows and >=2 dimensions")
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(self.batch_size, int) or not 1 <= self.batch_size <= self.n_train:
            raise ValueError("invalid batch size")
        if not 0 < self.delta < 1 / self.n_train:
            raise ValueError("delta must be positive and smaller than 1/n_train")
        for name in ("learning_rate", "clip_norm", "privacy_cap"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        for name in ("stable_signal", "generic_noise", "tau"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"invalid {name}")
        if not 0.5 <= self.utility_floor <= 1:
            raise ValueError("utility_floor must be in [0.5, 1]")
        if self.dataset == "insurance" and self.strengths != (1.0,):
            raise ValueError("insurance requires strengths=(1.0,); no synthetic correlation claim")


@dataclass
class Cohort:
    X: np.ndarray
    y: np.ndarray
    proxy_uniform: np.ndarray
    proxy_index: int = 1


def synthetic_cohort(n: int, seed: int, config: StudyConfig, strength: float) -> Cohort:
    """Row-wise bounded generation with fixed balanced label prior, no fitted scaler."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.float32)
    X = np.tanh(rng.normal(size=(n, config.dimensions))).astype(np.float32)
    X[:, 0] = np.tanh(config.stable_signal * (2 * y - 1) + rng.normal(size=n))
    uniform = rng.random(n)
    X[:, 1] = np.where(uniform < strength, 2 * y - 1, 1 - 2 * y)
    return Cohort(X, y, uniform)


def proxy_environment(cohort: Cohort, retention: float, strength: float,
                      dataset: str, *, stable_control: bool = False) -> np.ndarray:
    X = cohort.X.copy()
    if stable_control:
        return X  # identical source signal retained: exact source-difficulty control
    if dataset == "synthetic":
        reliability = 0.5 + retention * (strength - 0.5)
        X[:, cohort.proxy_index] = np.where(
            cohort.proxy_uniform < reliability, 2 * cohort.y - 1, 1 - 2 * cohort.y
        )
    else:
        # Documented zero-imputation policy on fixed-normalised charges, not natural drift.
        X[cohort.proxy_uniform >= retention, cohort.proxy_index] = 0.0
    return X


def insurance_cohorts(path: Path, seed: int) -> tuple[Cohort, Cohort, Cohort, dict]:
    """Public insurance case, fixed bounds/categories; no target-fitted preparation."""
    frame = pd.read_csv(path)
    required = ["age", "bmi", "children", "charges", "sex", "region", "smoker"]
    if frame[required].isna().any().any():
        raise ValueError("insurance input has missing values")
    domains = {"sex": ("female", "male"),
               "region": ("northeast", "northwest", "southeast", "southwest"),
               "smoker": ("no", "yes")}
    for name, values in domains.items():
        if not frame[name].isin(values).all():
            raise ValueError(f"unexpected public {name} category")
    bounds = {"age": 100.0, "charges": 100000.0, "bmi": 80.0, "children": 10.0}
    numeric = frame[list(bounds)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("insurance numeric features must be finite")
    X = np.clip(numeric / np.array(list(bounds.values())), 0, 1)
    for name in ("sex", "region"):
        X = np.column_stack([X, *(frame[name].to_numpy() == value
                                 for value in domains[name])])
    y = (frame.smoker.to_numpy() == "yes").astype(np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    a, b = int(0.6 * len(y)), int(0.8 * len(y))
    cohorts = []
    for ix in (indices[:a], indices[a:b], indices[b:]):
        if len(np.unique(y[ix])) != 2:
            raise ValueError("both labels required in each public partition")
        cohorts.append(Cohort(X[ix].astype(np.float32), y[ix], rng.random(len(ix))))
    return (*cohorts, {"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "public_bounds": bounds, "public_categories": domains,
                      "partition_sizes": [len(c.y) for c in cohorts]})


def train_model(cohort: Cohort, seed: int, config: StudyConfig,
                condition: str, epsilon: float | None = None, *,
                architecture: str = "mlp", hidden_units: int = 16):
    """Same initialisation, Poisson sample stream and normalised SGD for all controls."""
    import torch
    from opacus import PrivacyEngine
    from opacus.data_loader import DPDataLoader

    torch.manual_seed(seed)
    if architecture == "logistic":
        model = torch.nn.Sequential(torch.nn.Linear(cohort.X.shape[1], 1))
    elif architecture == "mlp" and isinstance(hidden_units, int) and hidden_units > 0:
        model = torch.nn.Sequential(torch.nn.Linear(cohort.X.shape[1], hidden_units),
                                    torch.nn.Tanh(), torch.nn.Linear(hidden_units, 1))
    else:
        raise ValueError("invalid architecture/hidden units")
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(cohort.X),
                                            torch.from_numpy(cohort.y))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(seed + 100000),
    )
    engine = None
    if condition == "ordinary":
        loader = DPDataLoader.from_data_loader(loader)
    elif condition in {"clipped", "dpsgd"}:
        engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        kwargs = dict(module=model, optimizer=optimizer, data_loader=loader,
                      max_grad_norm=config.clip_norm, poisson_sampling=True,
                      noise_generator=torch.Generator().manual_seed(seed + 200000))
        if condition == "dpsgd":
            model, optimizer, loader = engine.make_private_with_epsilon(
                **kwargs, target_epsilon=epsilon, target_delta=config.delta,
                epochs=config.epochs,
            )
        else:
            model, optimizer, loader = engine.make_private(**kwargs, noise_multiplier=0.0)
    else:
        raise ValueError(f"unknown condition {condition}")
    q = loader.sample_rate
    expected_batch = int(len(dataset) * q)
    steps, empty_steps, clipped_count, sample_count = 0, 0, 0, 0
    model.train()
    for _ in range(config.epochs):
        for X, y in loader:
            optimizer.zero_grad()
            logits = model(X).reshape(-1)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y,
                                                                         reduction="sum")
            # Opacus reconstructs per-example gradients from a batch-mean loss and
            # normalises clipped sums by the expected Poisson batch size. Ordinary
            # gradients use that same expected-size denominator directly.
            denominator = expected_batch if condition == "ordinary" else max(len(y), 1)
            (losses / denominator).backward()
            if condition != "ordinary" and len(y):
                norms = torch.stack([p.grad_sample.reshape(len(y), -1).norm(2, dim=1)
                                     for p in model.parameters()], dim=1).norm(2, dim=1)
                clipped_count += int((norms > config.clip_norm).sum())
                sample_count += len(y)
            optimizer.step()  # empty Poisson batches still receive noise/accounting
            steps += 1
            empty_steps += int(len(y) == 0)
    achieved = engine.get_epsilon(config.delta) if condition == "dpsgd" else None
    metadata = {"condition": condition, "requested_epsilon": epsilon,
                "achieved_epsilon": achieved, "delta": config.delta if achieved else None,
                "noise_multiplier": float(optimizer.noise_multiplier) if engine else 0.0,
                "sample_rate": q, "steps": steps, "empty_steps": empty_steps,
                "clip_rate": clipped_count / sample_count if sample_count else None,
                "secure_rng": False, "accountant": "rdp" if achieved else None,
                "privacy_scope": "per-model prepared-record add/remove accounting",
                "training_sha256": hashlib.sha256(cohort.X.tobytes() + cohort.y.tobytes()).hexdigest()}
    return model, metadata


def score(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X)).reshape(-1)).numpy()


def metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(y)) != 2:
        raise ValueError("both labels required for AUC")
    return {"auc": float(roc_auc_score(y, scores)),
            "pr_auc": float(average_precision_score(y, scores)),
            "balanced_accuracy": float(balanced_accuracy_score(y, scores >= 0.5))}


def choose_candidate(validation: pd.DataFrame, rule: str, config: StudyConfig) -> float | None:
    """Point-estimate exploratory selection; never access held-out rows here."""
    if set(validation.partition) != {"validation"}:
        raise ValueError("candidate selection requires validation rows only")
    environments = {"source_only": {"source"},
                    "generic_shift": {"source", *(
                        f"generic_{level:g}" for level in config.stress_levels)},
                    "proxy_stress": {"source", *(
                        f"proxy_{level:g}" for level in config.stress_levels)}}
    if rule not in environments:
        raise ValueError(f"unknown selection rule {rule}")
    wanted = environments[rule]
    rows = validation[validation.environment.isin(wanted)]
    if set(rows.environment) != wanted:
        raise ValueError("missing prespecified validation environment")
    candidates = []
    for requested, part in rows[rows.condition == "dpsgd"].groupby("requested_epsilon"):
        if set(part.environment) != wanted or part.environment.duplicated().any():
            raise ValueError("incomplete/duplicate candidate environments")
        if part.achieved_epsilon.max() > config.privacy_cap:
            continue
        if (part.gap <= config.tau).all() and (part.auc >= config.utility_floor).all():
            candidates.append((float(part.achieved_epsilon.max()), float(requested)))
    return min(candidates)[1] if candidates else None


def attach_gaps(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "strength", "partition", "variant", "environment"]
    reference = rows[rows.condition == "ordinary"][keys + ["auc"]].rename(
        columns={"auc": "reference_auc"})
    if reference.duplicated(keys).any():
        raise ValueError("duplicate reference models")
    result = rows.merge(reference, on=keys, validate="many_to_one")
    result["gap"] = result.reference_auc - result.auc
    # Source is from the same independent test cohort, never validation data.
    source_keys = ["seed", "strength", "partition", "variant", "condition", "requested_epsilon"]
    source = result[result.environment == "source"][source_keys + ["gap"]].rename(
        columns={"gap": "source_gap"})
    result = result.merge(source, on=source_keys, validate="many_to_one")
    result["gap_interaction"] = result.gap - result.source_gap
    return result


def seed_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Descriptive t intervals across independent training seeds, no row replication."""
    keys = ["strength", "variant", "environment", "condition", "requested_epsilon"]
    output = []
    for key, part in rows[rows.partition == "test"].groupby(keys, dropna=False):
        if part.seed.duplicated().any():
            raise ValueError("duplicate seed in summary cell")
        row = dict(zip(keys, key))
        row["n_seeds"] = len(part)
        for name in ("auc", "gap", "gap_interaction"):
            values = part[name].to_numpy()
            mean = float(values.mean())
            radius = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1)
                           / np.sqrt(len(values))) if len(values) > 1 else np.nan
            row.update({f"{name}_mean": mean, f"{name}_lower": mean - radius,
                        f"{name}_upper": mean + radius})
        output.append(row)
    return pd.DataFrame(output)


def evaluate_decisions(rows: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    output = []
    for (seed, strength), group in rows[rows.variant == "proxy"].groupby(["seed", "strength"]):
        validation = group[group.partition == "validation"]
        for rule in ("source_only", "generic_shift", "proxy_stress"):
            selected = choose_candidate(validation, rule, config)
            for level in config.test_levels:
                environment = f"proxy_{level:g}"
                row = {"seed": seed, "strength": strength, "rule": rule,
                       "environment": environment, "selected_epsilon": selected,
                       "abstained": selected is None, "gap": None, "auc": None,
                       "achieved_epsilon": None, "loss_violation": None,
                       "floor_violation": None, "failed": None}
                if selected is not None:
                    part = group[(group.partition == "test") & (group.environment == environment)
                                 & (group.condition == "dpsgd")
                                 & (group.requested_epsilon == selected)]
                    if len(part) != 1:
                        raise ValueError("missing/duplicate held-out decision outcome")
                    item = part.iloc[0]
                    row.update(gap=float(item.gap), auc=float(item.auc),
                               achieved_epsilon=float(item.achieved_epsilon),
                               loss_violation=max(0.0, float(item.gap) - config.tau),
                               floor_violation=max(0.0, config.utility_floor - float(item.auc)),
                               failed=bool(item.gap > config.tau or item.auc < config.utility_floor))
                output.append(row)
    return pd.DataFrame(output)


def run_study(config: StudyConfig, output_dir: Path, insurance_path: Path | None = None) -> dict:
    import torch

    config.validate()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be new or empty; preserve existing evidence")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records, accounting, preparation = [], [], []
    for strength in config.strengths:
        for seed in config.seeds:
            if config.dataset == "synthetic":
                train = synthetic_cohort(config.n_train, np.random.SeedSequence([seed, 1]), config, strength)
                validation = synthetic_cohort(config.n_eval, np.random.SeedSequence([seed, 2]), config, strength)
                test = synthetic_cohort(config.n_eval, np.random.SeedSequence([seed, 3]), config, strength)
            else:
                train, validation, test, prepared = insurance_cohorts(insurance_path, seed)
                preparation.append({"seed": seed, **prepared})
            jobs = [("ordinary", None), ("clipped", None), *(
                ("dpsgd", epsilon) for epsilon in config.epsilons)]
            for condition, epsilon in jobs:
                model, meta = train_model(train, seed, config, condition, epsilon)
                accounting.append({"seed": seed, "strength": strength, **meta})
                for partition, cohort, levels in (
                    ("validation", validation, config.stress_levels),
                    ("test", test, config.test_levels),
                ):
                    for variant in ("proxy", "stable_signal_control") if config.dataset == "synthetic" else ("proxy",):
                        environments = {"source": cohort.X}
                        for level in levels:
                            environments[f"proxy_{level:g}"] = proxy_environment(
                                cohort, level, strength, config.dataset,
                                stable_control=variant == "stable_signal_control")
                        if partition == "validation":
                            rng = np.random.default_rng(4000 + seed)
                            columns = [i for i in range(cohort.X.shape[1]) if i != cohort.proxy_index]
                            noise = rng.normal(size=(len(cohort.X), len(columns)))
                            for level in config.stress_levels:
                                generic = cohort.X.copy()
                                generic[:, columns] += noise * config.generic_noise * (1 - level)
                                environments[f"generic_{level:g}"] = np.clip(generic, -1, 1)
                        for environment, X in environments.items():
                            records.append({"seed": seed, "strength": strength,
                                            "partition": partition, "variant": variant,
                                            "environment": environment, "condition": condition,
                                            "requested_epsilon": epsilon,
                                            "achieved_epsilon": meta["achieved_epsilon"],
                                            **metrics(cohort.y, score(model, X))})
                print(f"{config.dataset} strength={strength:g} seed={seed} {condition} epsilon={epsilon}",
                      flush=True)
    rows = attach_gaps(pd.DataFrame(records))
    decisions = evaluate_decisions(rows, config)
    summary = seed_summary(rows)
    payload = {"schema_version": 1, "status": "exploratory_pilot",
               "config": asdict(config), "preparation": preparation,
               "accounting": accounting,
               "versions": {name: importlib.metadata.version(name)
                            for name in ("torch", "opacus", "numpy", "pandas", "scikit-learn")},
               "python": platform.python_version(),
               "study_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "privacy_release": "public/synthetic benchmark; not composed private search/release",
               "inference": "descriptive seed intervals conditional on evaluation cohorts; no confirmatory test"}
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        payload["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        payload["git_commit"] = None
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "metrics.csv", index=False)
    decisions.to_csv(output_dir / "decisions.csv", index=False)
    summary.to_csv(output_dir / "seed_summary.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    write_report(rows, decisions, config, output_dir)
    from .proxy_shift_components import analyse

    analyse(output_dir, output_dir / "components")
    return payload


def write_report(rows: pd.DataFrame, decisions: pd.DataFrame,
                 config: StudyConfig, output_dir: Path) -> None:
    lines = ["# Proxy-shift exploratory run", "", f"Dataset: `{config.dataset}`; seeds: {config.seeds}.",
             "", "No confirmatory significance or established novelty is claimed. Seed intervals",
             "are descriptive and do not integrate all evaluation-sample uncertainty.", "",
             "Selection uses validation point estimates and an achieved-epsilon cap.",
             "Held-out failures include loss-margin OR absolute-utility violations.", "",
             "| Rule | Decisions (seed × strength) | Selected | Abstained | Failed held-out evaluations / selected evaluations |",
             "|---|---:|---:|---:|---:|"]
    for rule, part in decisions.groupby("rule"):
        choices = part.drop_duplicates(["seed", "strength"])
        selected = part[~part.abstained]
        lines.append(f"| {rule} | {len(choices)} | {(~choices.abstained).sum()} | "
                     f"{choices.abstained.sum()} | {int(selected.failed.sum()) if len(selected) else 0}/{len(selected)} |")
    lines += ["", "Held-out evaluations share trained models; the denominator above is descriptive,",
              "not a count of independent training replicates. Failure is undefined for abstentions.",
              "", "| Held-out environment | Mean DP gap interaction | Range over training seeds/strengths/budgets |",
              "|---|---:|---:|"]
    part = rows[(rows.partition == "test") & (rows.variant == "proxy")
                & (rows.condition == "dpsgd") & (rows.environment != "source")]
    for environment, group in part.groupby("environment"):
        lines.append(f"| {environment} | {group.gap_interaction.mean():.4f} | "
                     f"{group.gap_interaction.min():.4f} to {group.gap_interaction.max():.4f} |")
    lines += ["", "The pooled interaction is a grid description, not a population effect or a test.",
              "Inspect `seed_summary.csv` by strength and budget before interpreting it.", "",
              "## Dataset scope", "",
              "Synthetic controls and insurance stress tests support a narrow exploratory study.",
              "Insurance masking is a hypothetical zero-imputation failure policy, not observed drift.",
              "Real external validation (prespecified ACS state/year environments) is required before",
              "claims that recommendations generalise. A second independent tabular domain belongs",
              "in future scope unless broad cross-domain claims are pursued now.", "",
              "The matched stable-signal condition has identical source data and preserves that",
              "signal in every synthetic test environment; it is a controlled counterfactual,",
              "not a new robust-training algorithm. Secure RNG is disabled for reproducible research.", ""]
    (output_dir / "report.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for environment, group in part.groupby("environment"):
        values = group.groupby("requested_epsilon").gap_interaction.mean()
        ax.plot(values.index, values.values, marker="o", label=environment)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(xlabel="Requested epsilon (achieved values in manifest)",
           ylabel="Held-out gap minus source gap", title="Exploratory grid means; no inferential error bars")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gap_interaction.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--insurance-path", type=Path, default=Path("data/insurance.csv"))
    args = parser.parse_args(argv)
    values = json.loads(args.config.read_text())
    for name in ("seeds", "strengths", "epsilons", "stress_levels", "test_levels"):
        if name in values:
            values[name] = tuple(values[name])
    run_study(StudyConfig(**values), args.output_dir, args.insurance_path)


if __name__ == "__main__":
    main()
