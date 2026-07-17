"""Multi-task benchmark runner and CLI.

Runs conventional baselines and (optionally) DP-SGD neural networks over
one or more :mod:`dp.tasks` benchmark tasks and multiple random seeds,
recording tidy per-seed results, repeated-seed summaries, per-example test
predictions and DP-SGD accounting metadata.

Usage::

    python -m dp.benchmark --task all --seeds 0 1 2 3 4 \
        --output-dir reports/generated

Leakage-safety invariants enforced here:

* Preprocessing is fitted on the training partition only.
* Decision thresholds are selected on the validation partition.
* Early stopping (neural) monitors validation loss.
* The test partition is used exactly once per run, for final metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .attacks import loss_threshold_attack
from .metrics import classification_metrics, group_fairness_report, select_threshold
from .pipeline import build_preprocessor
from .tasks import TASKS, TaskData, get_task, prepare_task_data
from .uncertainty import TIDY_COLUMNS, summarize_runs

logger = logging.getLogger(__name__)

MODEL_NAMES = ("dummy", "logistic", "svm", "gradient_boosting", "neural", "dpsgd")

#: Default target-epsilon grid: strong (<1), moderate (1-3) and weak (3-8)
#: privacy. The non-private reference is the ``neural`` model.
DEFAULT_EPSILONS = (0.5, 2.0, 6.0)
DEFAULT_DELTA = 1e-5
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

_DPSGD_EPOCHS = 15
_DPSGD_BATCH_SIZE = 128
_DPSGD_MAX_GRAD_NORM = 1.0
_NEURAL_EPOCHS = 60
_NEURAL_PATIENCE = 8
_HIDDEN_UNITS = 32


@dataclass
class RunRecord:
    """Accumulates tidy rows, predictions and DP-SGD metadata for one run."""

    tidy_rows: list[dict[str, object]] = field(default_factory=list)
    prediction_rows: list[pd.DataFrame] = field(default_factory=list)
    dpsgd_metadata: list[dict[str, object]] = field(default_factory=list)


def _torch_available() -> bool:
    try:  # pragma: no cover - trivial import guard
        import torch  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


def _opacus_available() -> bool:
    try:  # pragma: no cover - trivial import guard
        import opacus  # noqa: F401

        return _torch_available()
    except ImportError:  # pragma: no cover
        return False


def build_sklearn_model(name: str, X_train: pd.DataFrame, seed: int) -> Pipeline:
    """Build a leakage-safe sklearn Pipeline (preprocessing + classifier).

    The preprocessor is part of the Pipeline, so it is fitted only on
    whatever data ``fit`` receives (the training partition).

    Raises:
        ValueError: For names outside dummy/logistic/svm/gradient_boosting.
    """
    estimators: dict[str, Callable[[], object]] = {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "logistic": lambda: LogisticRegression(max_iter=2000, random_state=seed),
        "svm": lambda: SVC(kernel="rbf", probability=True, random_state=seed),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=seed),
    }
    if name not in estimators:
        raise ValueError(f"Unknown sklearn model {name!r}")
    return Pipeline(
        [("preprocess", build_preprocessor(X_train)), ("classifier", estimators[name]())]
    )


def _torch_arrays(data: TaskData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit preprocessing on train only; transform train/val/test."""
    preprocessor = build_preprocessor(data.X_train)
    X_train = np.asarray(preprocessor.fit_transform(data.X_train), dtype=np.float32)
    X_val = np.asarray(preprocessor.transform(data.X_val), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(data.X_test), dtype=np.float32)
    return X_train, X_val, X_test


def _build_mlp(n_features: int, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(n_features, _HIDDEN_UNITS),
        nn.ReLU(),
        nn.Linear(_HIDDEN_UNITS, 1),
    )


def _per_example_bce(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    import torch
    from torch import nn

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X)).squeeze(-1)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(y.astype(np.float32)), reduction="none"
        )
    return losses.numpy()


def _predict_proba_torch(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X)).squeeze(-1)).numpy()


def train_neural(
    data: TaskData, seed: int
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Train a non-private MLP with validation-based early stopping.

    Returns the model, per-partition score arrays (train/val/test) and a
    metadata dict.  Preprocessing is fitted on the training partition only.
    """
    import torch
    from torch import nn

    X_train, X_val, X_test = _torch_arrays(data)
    model = _build_mlp(X_train.shape[1], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(data.y_train.astype(np.float32))
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(data.y_val.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=_DPSGD_BATCH_SIZE, shuffle=True, generator=generator
    )

    best_val, best_state, patience, epochs_run = np.inf, None, 0, 0
    for epoch in range(_NEURAL_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv).squeeze(-1), yv))
        epochs_run = epoch + 1
        if val_loss < best_val - 1e-5:
            best_val, patience = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= _NEURAL_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {"epochs_run": epochs_run, "early_stopped": epochs_run < _NEURAL_EPOCHS}
    return (
        model,
        _predict_proba_torch(model, X_train),
        _predict_proba_torch(model, X_val),
        _predict_proba_torch(model, X_test),
        metadata,
    )


def train_dpsgd(
    data: TaskData, seed: int, target_epsilon: float, delta: float
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Train an MLP with Opacus DP-SGD at a target (epsilon, delta).

    The achieved epsilon is read back from the privacy accountant after
    training; no epsilon values are hardcoded.  Returns the model,
    train/val/test scores and an accounting-metadata dict.
    """
    import torch
    from opacus import PrivacyEngine
    from torch import nn

    X_train, X_val, X_test = _torch_arrays(data)
    torch.manual_seed(seed)
    model = _build_mlp(X_train.shape[1], seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loss_fn = nn.BCEWithLogitsLoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(data.y_train.astype(np.float32))
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=_DPSGD_BATCH_SIZE, shuffle=True, generator=generator
    )
    engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=_DPSGD_EPOCHS,
        target_epsilon=target_epsilon,
        target_delta=delta,
        max_grad_norm=_DPSGD_MAX_GRAD_NORM,
    )
    for _ in range(_DPSGD_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()

    achieved_epsilon = float(engine.get_epsilon(delta))
    metadata = {
        "target_epsilon": float(target_epsilon),
        "achieved_epsilon": achieved_epsilon,
        "delta": float(delta),
        "noise_multiplier": float(optimizer.noise_multiplier),
        "max_grad_norm": float(_DPSGD_MAX_GRAD_NORM),
        "batch_size": int(_DPSGD_BATCH_SIZE),
        "epochs": int(_DPSGD_EPOCHS),
        "seed": int(seed),
        "accountant": "rdp",
    }
    return (
        model,
        _predict_proba_torch(model, X_train),
        _predict_proba_torch(model, X_val),
        _predict_proba_torch(model, X_test),
        metadata,
    )


def _record_run(
    record: RunRecord,
    data: TaskData,
    model_label: str,
    privacy_mechanism: str,
    epsilon: float,
    delta: float,
    seed: int,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    """Select the threshold on validation, evaluate on test, append rows."""
    threshold = select_threshold(data.y_val, val_scores)
    metrics = classification_metrics(data.y_test, test_scores, threshold)
    y_pred = (test_scores >= threshold).astype(int)
    fairness = group_fairness_report(data.y_test, y_pred, data.sensitive_test)
    metrics["threshold"] = threshold
    metrics["demographic_parity_difference"] = fairness.demographic_parity_difference
    metrics["equalized_odds_difference"] = fairness.equalized_odds_difference
    for g, v in fairness.group_tpr.items():
        metrics[f"tpr_group_{g}"] = v
    for g, v in fairness.group_fpr.items():
        metrics[f"fpr_group_{g}"] = v
    for g, v in fairness.group_sizes.items():
        metrics[f"group_size_{g}"] = float(v)
    if extra_metrics:
        metrics.update(extra_metrics)
    for metric, value in metrics.items():
        record.tidy_rows.append(
            {
                "task": data.task.name,
                "model": model_label,
                "privacy_mechanism": privacy_mechanism,
                "epsilon": epsilon,
                "delta": delta,
                "seed": seed,
                "metric": metric,
                "value": value,
            }
        )
    record.prediction_rows.append(
        pd.DataFrame(
            {
                "task": data.task.name,
                "model": model_label,
                "privacy_mechanism": privacy_mechanism,
                "epsilon": epsilon,
                "seed": seed,
                "y_true": data.y_test,
                "y_score": test_scores,
                "group": data.sensitive_test,
            }
        )
    )


def run_single_seed(
    df: pd.DataFrame,
    task_name: str,
    models: Sequence[str],
    seed: int,
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    delta: float = DEFAULT_DELTA,
    record: RunRecord | None = None,
) -> RunRecord:
    """Run all requested models on one task for one seed.

    The seed controls the train/val/test split, model initialisation and
    training randomness, so repeated-seed intervals capture all three
    sources of variation.
    """
    record = record if record is not None else RunRecord()
    task = get_task(task_name)
    data = prepare_task_data(df, task, random_state=seed)

    for name in models:
        if name in ("dummy", "logistic", "svm", "gradient_boosting"):
            model = build_sklearn_model(name, data.X_train, seed)
            model.fit(data.X_train, data.y_train)
            val_scores = model.predict_proba(data.X_val)[:, 1]
            test_scores = model.predict_proba(data.X_test)[:, 1]
            _record_run(
                record, data, name, "none", float("inf"), float("nan"), seed,
                val_scores, test_scores,
            )
        elif name == "neural":
            if not _torch_available():
                logger.warning("Skipping neural model: torch is not installed")
                continue
            model, train_scores, val_scores, test_scores, _ = train_neural(data, seed)
            mia = _neural_mia_metrics(model, data)
            _record_run(
                record, data, name, "none", float("inf"), float("nan"), seed,
                val_scores, test_scores, extra_metrics=mia,
            )
        elif name == "dpsgd":
            if not _opacus_available():
                logger.warning("Skipping DP-SGD: torch/opacus are not installed")
                continue
            for target_eps in epsilons:
                model, train_scores, val_scores, test_scores, meta = train_dpsgd(
                    data, seed, target_eps, delta
                )
                meta["task"] = task.name
                record.dpsgd_metadata.append(meta)
                mia = _neural_mia_metrics(model, data)
                extra = dict(mia)
                extra["achieved_epsilon"] = meta["achieved_epsilon"]
                extra["noise_multiplier"] = meta["noise_multiplier"]
                _record_run(
                    record, data, "dpsgd", "dpsgd",
                    float(meta["achieved_epsilon"]), delta, seed,
                    val_scores, test_scores, extra_metrics=extra,
                )
        else:
            raise ValueError(f"Unknown model {name!r}. Known: {', '.join(MODEL_NAMES)}")
        logger.info("Finished task=%s model=%s seed=%d", task_name, name, seed)
    return record


def _neural_mia_metrics(model: object, data: TaskData) -> dict[str, float]:
    """Loss-threshold membership-inference metrics for a torch model.

    Practical attack performance only — chance-level results do not prove
    the formal guarantee; see docs/EXPERIMENT_DESIGN.md.
    """
    X_train, _, X_test = _torch_arrays(data)
    train_losses = _per_example_bce(model, X_train, data.y_train)
    test_losses = _per_example_bce(model, X_test, data.y_test)
    result = loss_threshold_attack(train_losses, test_losses)
    return {
        "mia_loss_threshold_auc": result.auc,
        "mia_tpr_at_fpr_0.01": result.tpr_at_fpr[0.01],
    }


def run_benchmark(
    df: pd.DataFrame,
    tasks: Sequence[str],
    models: Sequence[str],
    seeds: Sequence[int],
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    delta: float = DEFAULT_DELTA,
) -> RunRecord:
    """Run the full benchmark grid (tasks x models x seeds)."""
    record = RunRecord()
    for task_name in tasks:
        for seed in seeds:
            run_single_seed(
                df, task_name, models, seed,
                epsilons=epsilons, delta=delta, record=record,
            )
    return record


def export_results(record: RunRecord, output_dir: str | Path) -> dict[str, Path]:
    """Write tidy results, summary, predictions and DP-SGD metadata.

    Files are written under ``output_dir`` with fixed names; only these
    benchmark-owned files are ever (over)written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tidy = pd.DataFrame(record.tidy_rows, columns=list(TIDY_COLUMNS))
    summary = summarize_runs(tidy)
    paths = {
        "tidy": out / "tidy_results.csv",
        "summary": out / "summary.csv",
        "predictions": out / "predictions.csv",
        "dpsgd_metadata": out / "dpsgd_metadata.json",
    }
    tidy.to_csv(paths["tidy"], index=False)
    summary.to_csv(paths["summary"], index=False)
    if record.prediction_rows:
        pd.concat(record.prediction_rows, ignore_index=True).to_csv(
            paths["predictions"], index=False
        )
    paths["dpsgd_metadata"].write_text(json.dumps(record.dpsgd_metadata, indent=2))
    logger.info("Wrote benchmark outputs to %s", out)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dp.benchmark",
        description="Run the multi-task DP insurance benchmark.",
    )
    parser.add_argument(
        "--task",
        default="all",
        help=f"Task name or 'all'. Known tasks: {', '.join(sorted(TASKS))}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_NAMES),
        help=f"Models to run. Known: {', '.join(MODEL_NAMES)}",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Random seeds (each controls split + training).",
    )
    parser.add_argument(
        "--epsilons", nargs="+", type=float, default=list(DEFAULT_EPSILONS),
        help="Target epsilon grid for DP-SGD.",
    )
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--data", default="data/insurance.csv", help="Dataset CSV path.")
    parser.add_argument(
        "--output-dir", default="reports/generated", help="Directory for result files."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.task == "all":
        tasks = sorted(TASKS)
    elif args.task in TASKS:
        tasks = [args.task]
    else:
        print(
            f"error: unknown task {args.task!r}. Known tasks: {', '.join(sorted(TASKS))}",
            file=sys.stderr,
        )
        return 2
    unknown = [m for m in args.models if m not in MODEL_NAMES]
    if unknown:
        print(
            f"error: unknown model(s) {unknown}. Known models: {', '.join(MODEL_NAMES)}",
            file=sys.stderr,
        )
        return 2
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"error: dataset not found at {data_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(data_path)
    logger.info(
        "Running benchmark: tasks=%s models=%s seeds=%s epsilons=%s",
        tasks, args.models, args.seeds, args.epsilons,
    )
    record = run_benchmark(
        df, tasks, args.models, args.seeds, epsilons=args.epsilons, delta=args.delta
    )
    paths = export_results(record, args.output_dir)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
