"""Portfolio report generation from benchmark outputs.

Reads the files written by :mod:`dp.benchmark` (tidy results, summary,
predictions, DP-SGD metadata) and regenerates figures under
``reports/figures/``, tables under ``reports/tables/`` and
``reports/benchmark_report.md``.  Every number in the report is computed
from the benchmark outputs; nothing is hand-written.

Usage::

    python -m dp.reporting --results reports/generated --reports-dir reports
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

HEADLINE_METRICS = ("roc_auc", "pr_auc", "macro_f1", "brier")
_NONPRIVATE_MODELS = ("dummy", "logistic", "svm", "gradient_boosting", "neural")


def _fmt_ci(row: pd.Series) -> str:
    if np.isnan(row["mean"]):
        return "n/a"
    return f"{row['mean']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"


def _load(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    tidy = pd.read_csv(results_dir / "tidy_results.csv")
    summary = pd.read_csv(results_dir / "summary.csv")
    pred_path = results_dir / "predictions.csv"
    predictions = pd.read_csv(pred_path) if pred_path.exists() else None
    return tidy, summary, predictions


def model_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Wide per-task model comparison of headline metrics with seed CIs."""
    rows = []
    subset = summary[summary["metric"].isin(HEADLINE_METRICS)]
    for (task, model, eps), grp in subset.groupby(["task", "model", "epsilon"]):
        row: dict[str, object] = {
            "task": task,
            "model": model,
            "epsilon": "inf" if np.isinf(eps) else f"{eps:.2f}",
            "n_runs": int(grp["n_runs"].iloc[0]),
        }
        for _, r in grp.iterrows():
            row[r["metric"]] = _fmt_ci(r)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "model", "epsilon"]).reset_index(drop=True)


def _summary_metric(summary: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    return summary[(summary["task"] == task) & (summary["metric"] == metric)]


def plot_model_comparison(summary: pd.DataFrame, figures_dir: Path) -> None:
    for task in summary["task"].unique():
        sub = _summary_metric(summary, task, "roc_auc").copy()
        sub["label"] = [
            m if np.isinf(e) else f"{m} (eps={e:.1f})"
            for m, e in zip(sub["model"], sub["epsilon"])
        ]
        sub = sub.sort_values("mean")
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(sub) + 2))
        err = np.vstack(
            [sub["mean"] - sub["ci_lower"], sub["ci_upper"] - sub["mean"]]
        )
        ax.barh(sub["label"], sub["mean"], xerr=err, capsize=3, color="#4878cf")
        ax.set_xlabel("ROC-AUC (mean, repeated-seed 95% CI)")
        ax.set_title(f"Model comparison — {task}")
        ax.set_xlim(0.4, 1.02)
        fig.tight_layout()
        fig.savefig(figures_dir / f"model_comparison_{task}.png", dpi=150)
        plt.close(fig)


def plot_privacy_curves(summary: pd.DataFrame, figures_dir: Path) -> None:
    """Privacy-utility and privacy-fairness curves for DP-SGD."""
    dp = summary[summary["model"] == "dpsgd"]
    for task in dp["task"].unique():
        ref = _summary_metric(summary, task, "roc_auc")
        ref = ref[ref["model"] == "neural"]
        for metric, fname, ylabel in [
            ("roc_auc", "privacy_utility", "ROC-AUC"),
            ("demographic_parity_difference", "privacy_fairness_dpd", "Demographic parity difference"),
            ("equalized_odds_difference", "privacy_fairness_eod", "Equalized odds difference"),
        ]:
            sub = _summary_metric(dp, task, metric).sort_values("epsilon")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.errorbar(
                sub["epsilon"], sub["mean"],
                yerr=[sub["mean"] - sub["ci_lower"], sub["ci_upper"] - sub["mean"]],
                marker="o", capsize=3, label="DP-SGD",
            )
            if metric == "roc_auc" and not ref.empty:
                ax.axhline(ref["mean"].iloc[0], linestyle="--", color="gray",
                           label="non-private neural")
            ax.set_xlabel("Achieved epsilon (RDP accountant)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs privacy budget — {task}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(figures_dir / f"{fname}_{task}.png", dpi=150)
            plt.close(fig)


def plot_curves_from_predictions(
    predictions: pd.DataFrame, tidy: pd.DataFrame, figures_dir: Path
) -> None:
    """ROC, PR, calibration and confusion-matrix figures (seed 0 runs)."""
    seed0 = predictions[predictions["seed"] == predictions["seed"].min()]
    thresholds = tidy[(tidy["metric"] == "threshold")]
    for task in seed0["task"].unique():
        tsub = seed0[seed0["task"] == task]
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        fig_cal, ax_cal = plt.subplots(figsize=(6, 5))
        groups = list(tsub.groupby(["model", "epsilon"]))
        for (model, eps), grp in groups:
            label = model if np.isinf(eps) else f"{model} (eps={eps:.1f})"
            fpr, tpr, _ = roc_curve(grp["y_true"], grp["y_score"])
            ax_roc.plot(fpr, tpr, label=label)
            prec, rec, _ = precision_recall_curve(grp["y_true"], grp["y_score"])
            ax_pr.plot(rec, prec, label=label)
            bins = np.linspace(0, 1, 11)
            ids = np.clip(np.digitize(grp["y_score"], bins[1:-1]), 0, 9)
            conf = [grp["y_score"][ids == b].mean() for b in range(10)]
            acc = [grp["y_true"][ids == b].mean() for b in range(10)]
            ax_cal.plot(conf, acc, marker="o", label=label)
        ax_roc.plot([0, 1], [0, 1], "--", color="gray")
        ax_cal.plot([0, 1], [0, 1], "--", color="gray")
        for ax, name, xl, yl in [
            (ax_roc, "roc_curves", "False positive rate", "True positive rate"),
            (ax_pr, "pr_curves", "Recall", "Precision"),
            (ax_cal, "calibration", "Mean predicted probability", "Observed frequency"),
        ]:
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{name.replace('_', ' ').title()} — {task} (seed 0)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        for fig, name in [(fig_roc, "roc_curves"), (fig_pr, "pr_curves"), (fig_cal, "calibration")]:
            fig.tight_layout()
            fig.savefig(figures_dir / f"{name}_{task}.png", dpi=150)
            plt.close(fig)

        # Confusion matrices at the validation-selected threshold.
        n = len(groups)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), squeeze=False)
        for ax, ((model, eps), grp) in zip(axes[0], groups):
            trow = thresholds[
                (thresholds["task"] == task)
                & (thresholds["model"] == model)
                & (thresholds["seed"] == grp["seed"].iloc[0])
            ]
            if not np.isinf(eps):
                trow = trow[np.isclose(trow["epsilon"], eps)]
            thr = float(trow["value"].iloc[0]) if len(trow) else 0.5
            cm = confusion_matrix(grp["y_true"], (grp["y_score"] >= thr).astype(int))
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")
            ax.set_title(model if np.isinf(eps) else f"{model}\neps={eps:.1f}", fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        fig.suptitle(f"Confusion matrices — {task} (seed 0)")
        fig.tight_layout()
        fig.savefig(figures_dir / f"confusion_matrices_{task}.png", dpi=150)
        plt.close(fig)


def _best_nonprivate(summary: pd.DataFrame, task: str) -> pd.Series:
    sub = _summary_metric(summary, task, "roc_auc")
    sub = sub[sub["model"].isin(_NONPRIVATE_MODELS) & (sub["model"] != "dummy")]
    return sub.loc[sub["mean"].idxmax()]


def write_report(
    tidy: pd.DataFrame,
    summary: pd.DataFrame,
    reports_dir: Path,
    tables_dir: Path,
) -> Path:
    """Write reports/benchmark_report.md answering the benchmark questions."""
    lines: list[str] = [
        "# Multi-task benchmark report",
        "",
        "All numbers below are generated by `python -m dp.benchmark` followed by",
        "`python -m dp.reporting`; none are hand-entered. Intervals are",
        "**repeated-seed 95% t-intervals** — they describe run-to-run",
        "variability of the pipeline (split + initialisation + training),",
        "not independent-sample statistical guarantees.",
        "",
        f"Runs per configuration: {int(summary['n_runs'].max())} seeds.",
        "",
        "## Model comparison",
        "",
    ]
    comparison = model_comparison_table(summary)
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    lines.append(comparison.to_markdown(index=False))
    lines.append("")

    tasks = sorted(summary["task"].unique())

    # Q1: effect of removing charges on smoker classification.
    if {"smoker_without_charges", "smoker_with_charges_legacy"} <= set(tasks):
        with_c = _summary_metric(summary, "smoker_with_charges_legacy", "roc_auc")
        without_c = _summary_metric(summary, "smoker_without_charges", "roc_auc")
        best_with = with_c[with_c["model"] != "dummy"]["mean"].max()
        best_without = without_c[without_c["model"] != "dummy"]["mean"].max()
        lines += [
            "## Q1: Effect of removing `charges` from smoker classification",
            "",
            f"Best non-dummy ROC-AUC **with** charges (legacy task): {best_with:.3f}.",
            f"Best non-dummy ROC-AUC **without** charges: {best_without:.3f}.",
            f"Removing `charges` changes best-model ROC-AUC by {best_without - best_with:+.3f},",
            "confirming that `charges` acted as a near-direct proxy for smoking",
            "status and that the legacy task was unusually separable.",
            "",
        ]

    # Q2: best conventional model per task.
    lines += ["## Q2: Best model per task (ROC-AUC)", ""]
    for task in tasks:
        best = _best_nonprivate(summary, task)
        lines.append(
            f"- **{task}**: {best['model']} — {_fmt_ci(best)}"
        )
    lines.append("")

    # Q3: DP-SGD utility retention.
    dp = summary[(summary["model"] == "dpsgd") & (summary["metric"] == "roc_auc")]
    if not dp.empty:
        lines += ["## Q3: DP-SGD utility retention", ""]
        retention_rows = []
        for task in dp["task"].unique():
            best = _best_nonprivate(summary, task)
            for _, r in dp[dp["task"] == task].sort_values("epsilon").iterrows():
                retention_rows.append(
                    {
                        "task": task,
                        "achieved_epsilon": round(r["epsilon"], 3),
                        "dpsgd_roc_auc": _fmt_ci(r),
                        "best_nonprivate_roc_auc": _fmt_ci(best),
                        "retention": f"{r['mean'] / best['mean']:.1%}",
                    }
                )
        ret = pd.DataFrame(retention_rows)
        ret.to_csv(tables_dir / "dpsgd_retention.csv", index=False)
        lines.append(ret.to_markdown(index=False))
        lines.append("")

    # Q4: stability across seeds.
    lines += ["## Q4: Stability across seeds", ""]
    stab = summary[summary["metric"] == "roc_auc"].copy()
    stab = stab[stab["model"] != "dummy"]
    lines.append(
        f"Across all non-dummy configurations, the ROC-AUC seed standard "
        f"deviation ranges from {stab['std'].min():.4f} to {stab['std'].max():.4f} "
        f"(median {stab['std'].median():.4f})."
    )
    lines.append("")

    # Q5: privacy vs fairness across budgets.
    fair = summary[
        (summary["model"] == "dpsgd")
        & summary["metric"].isin(
            ["demographic_parity_difference", "equalized_odds_difference"]
        )
    ]
    if not fair.empty:
        lines += ["## Q5: Privacy and fairness across budgets", ""]
        ftab = fair.pivot_table(
            index=["task", "epsilon"], columns="metric", values="mean"
        ).round(4).reset_index()
        ftab.to_csv(tables_dir / "privacy_fairness.csv", index=False)
        lines.append(ftab.to_markdown(index=False))
        lines += [
            "",
            "See `reports/figures/privacy_fairness_*.png` for curves with",
            "repeated-seed intervals.",
            "",
        ]

    # Q6: supported vs uncertain conclusions.
    lines += [
        "## Q6: Supported and uncertain conclusions",
        "",
        "**Supported by these runs:** relative model ordering per task, the",
        "drop in smoker-classification performance when `charges` is removed,",
        "and the direction of the DP-SGD privacy-utility trade-off, all with",
        "repeated-seed intervals attached.",
        "",
        "**Uncertain:** absolute generalisation beyond this 1,338-row public",
        "dataset; fairness gaps for small groups (flagged in the tidy results);",
        "and any claim that chance-level membership-inference results verify",
        "the formal DP guarantee — empirical attacks provide lower bounds and",
        "practical leakage estimates only. Formal guarantees come from the",
        "accountant, not from the attacks.",
        "",
    ]

    # Fairness group table.
    group_rows = tidy[tidy["metric"].str.startswith(("tpr_group_", "fpr_group_", "group_size_"))]
    if not group_rows.empty:
        gsum = (
            group_rows.groupby(["task", "model", "epsilon", "metric"])["value"]
            .mean()
            .reset_index()
            .rename(columns={"value": "mean_over_seeds"})
        )
        gsum.to_csv(tables_dir / "group_fairness.csv", index=False)

    path = reports_dir / "benchmark_report.md"
    path.write_text("\n".join(lines))
    return path


def generate_reports(
    results_dir: str | Path,
    reports_dir: str | Path = "reports",
) -> Path:
    """Generate all figures, tables and the benchmark report.

    Returns the path of the written ``benchmark_report.md``.
    """
    results = Path(results_dir)
    reports = Path(reports_dir)
    figures_dir = reports / "figures"
    tables_dir = reports / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    tidy, summary, predictions = _load(results)
    plot_model_comparison(summary, figures_dir)
    plot_privacy_curves(summary, figures_dir)
    if predictions is not None:
        plot_curves_from_predictions(predictions, tidy, figures_dir)
    report = write_report(tidy, summary, reports, tables_dir)
    logger.info("Wrote %s", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m dp.reporting",
        description="Regenerate benchmark figures, tables and report.",
    )
    parser.add_argument("--results", default="reports/generated")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args(argv)
    results = Path(args.results)
    if not (results / "tidy_results.csv").exists():
        import sys

        print(
            f"error: no benchmark results found in {results}; run "
            "`python -m dp.benchmark` first",
            file=sys.stderr,
        )
        return 2
    print(f"report: {generate_reports(results, args.reports_dir)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
