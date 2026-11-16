"""Tables and figures for the real-data LiRA validation study.

Reads only artefacts written by ``lira_real_data_study.py analyse``; every
number in the generated report traces to ``summary.csv``, ``per_target.csv``
or ``results.json`` in the same directory.

    python research/lira_validation_report.py --input-dir reports/lira_validation/v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from dp import lira_study as ls

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # validated categorical order
MARKERS = ("o", "s", "D", "^")  # secondary encoding: identity is never colour alone
CONDITION_LABELS = {"nonprivate": "Non-private", "clip_only": "Clipping only",
                    "dp_eps8": "DP-SGD ε≈8", "dp_eps2": "DP-SGD ε≈2"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def figure_attack_comparison(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    series = [("loss-threshold", "Loss threshold"),
              (ls.config_name(ls.BASELINE), "Baseline LiRA (raw loss, 11 OUT)"),
              (ls.config_name(ls.PRIMARY), "Primary LiRA (logit, 64 OUT)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    panels = [("mean_auc", "auc_ci_low", "auc_ci_high", "Mean attack AUC", 0.5),
              ("mean_tpr_at_1pct", "tpr_ci_low", "tpr_ci_high", "Mean TPR at 1% FPR", 0.01)]
    for ax, (mean, lo, hi, title, chance) in zip(axes, panels):
        _style(ax)
        for j, (config, label) in enumerate(series):
            rows = summary[summary.config == config].set_index("condition").loc[list(ls.CONDITIONS)]
            x = [i + (j - 1) * 0.2 for i in range(len(rows))]
            ax.errorbar(x, rows[mean], yerr=[rows[mean] - rows[lo], rows[hi] - rows[mean]],
                        fmt=MARKERS[j], color=SERIES[j], markersize=7, linewidth=2, capsize=0,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate("chance", (len(ls.CONDITIONS) - 0.55, chance), color=MUTED, fontsize=8,
                    va="bottom", ha="right")
        ax.set_xticks(range(len(ls.CONDITIONS)), [CONDITION_LABELS[c] for c in ls.CONDITIONS])
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.suptitle("Membership-inference attacks on 16 target models per condition "
                 "(mean, 95% t-interval over targets)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_factorial(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    combos = [(s, v) for s in ls.STATISTICS for v in ls.VARIANCES]
    fig, axes = plt.subplots(1, len(ls.CONDITIONS), figsize=(12, 4), facecolor=SURFACE)
    for ax, condition in zip(axes, ls.CONDITIONS):
        _style(ax)
        sub = summary[summary.condition == condition].set_index("config")
        for j, (stat, var) in enumerate(combos):
            y = [sub.loc[f"{stat}|{var}|K={k}", "mean_auc"] for k in ls.SHADOW_COUNTS]
            ax.plot(range(len(y)), y, color=SERIES[j], marker=MARKERS[j], linewidth=2,
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"{stat.replace('_', ' ')}, {var.replace('_', '-')} variance")
        ax.set_xticks(range(len(ls.SHADOW_COUNTS)), [f"{k // 2}" for k in ls.SHADOW_COUNTS])
        ax.set_title(CONDITION_LABELS[condition], loc="left", color=INK, fontsize=10)
        ax.set_xlabel("OUT shadow observations per record", color=MUTED, fontsize=9)
    axes[0].set_ylabel("Mean attack AUC", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK, ncol=4,
               loc="lower center")
    fig.suptitle("Exploratory: all twelve offline LiRA configurations (y-axes differ by panel)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _fmt(x: float, digits: int = 4) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}"


def _p(x: float) -> str:
    return "n/a" if pd.isna(x) else (f"{x:.1e}" if x < 1e-3 else f"{x:.4f}")


def render(input_dir: Path) -> str:
    summary = pd.read_csv(input_dir / "summary.csv")
    results = json.loads((input_dir / "results.json").read_text())
    names = {"loss-threshold": "Loss threshold", ls.config_name(ls.BASELINE): "Baseline LiRA",
             ls.config_name(ls.PRIMARY): "Primary LiRA"}
    lines = [
        "# Real-data LiRA validation — generated results",
        "",
        "Generated by `research/lira_validation_report.py` from `summary.csv`, "
        "`per_target.csv` and `results.json` in this directory. Protocol: "
        "`docs/LIRA_VALIDATION_PROTOCOL.md`. Protocol hash "
        f"`{results['protocol_hash']}`. Data SHA-256 `{results['data_sha256']}`.",
        "",
        f"Runtime: {json.dumps(results['runtime'])}",
        "",
        "Each target has 669 members and 669 non-members; n = 16 targets per condition. "
        "Intervals are 95% t-intervals over targets. A non-detection is not evidence of privacy.",
        "",
        "## Conditions and accounting",
        "",
        "| Condition | Noise multiplier | Sample rate | Steps | Achieved ε (δ=1e-5) | Train acc. | Held-out acc. |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition in ls.CONDITIONS:
        c = results["conditions"][condition]
        d = c["diagnostics"]
        eps = c["achieved_epsilon_min_max"]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {c['noise_multiplier'] if c['noise_multiplier'] is not None else 'none'} "
            f"| {_fmt(c['sample_rate']) if c['sample_rate'] else 'n/a (shuffled)'} | {c['steps']} "
            f"| {'none' if eps is None else _fmt(eps[1])} | {_fmt(d['target_train_accuracy'], 3)} "
            f"| {_fmt(d['target_heldout_accuracy'], 3)} |")
    lines += ["", "## Predeclared attacks", "",
              "| Condition | Attack | Mean AUC [95% CI] | Mean TPR at 1% FPR [95% CI] | "
              "Detection p (Holm) | Mismatched-label control AUC |", "|---|---|---|---|---|---|"]
    for condition in ls.CONDITIONS:
        for config, label in names.items():
            r = summary[(summary.condition == condition) & (summary.config == config)].iloc[0]
            holm = r.get("stouffer_p_holm_across_conditions")
            p = _p(holm) if not pd.isna(holm) else f"{_p(r.stouffer_p)} (unadjusted)"
            lines.append(
                f"| {CONDITION_LABELS[condition]} | {label} | {_fmt(r.mean_auc)} "
                f"[{_fmt(r.auc_ci_low)}, {_fmt(r.auc_ci_high)}] | {_fmt(r.mean_tpr_at_1pct)} "
                f"[{_fmt(r.tpr_ci_low)}, {_fmt(r.tpr_ci_high)}] | {p} | {_fmt(r.mean_control_auc)} |")
    lines += ["", "## Primary minus baseline, paired by target", "",
              "| Condition | Metric | Mean difference [95% CI] | Sign-flip p | Targets improved |",
              "|---|---|---|---|---|"]
    for condition in ls.CONDITIONS:
        for key, label in (("paired_primary_minus_baseline_auc", "AUC"),
                           ("paired_primary_minus_baseline_tpr_at_1pct", "TPR at 1% FPR")):
            c = results["conditions"][condition][key]
            role = " **(primary endpoint)**" if condition == ls.PRIMARY_CONDITION and label == "AUC" else ""
            lines.append(
                f"| {CONDITION_LABELS[condition]} | {label}{role} | {c['mean']:+.4f} "
                f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {_p(c['signflip_p_two_sided'])} "
                f"| {c['targets_improved']}/{c['n']} |")
    lines += ["", "## Exploratory: all configurations (mean AUC / mean TPR at 1% FPR)", "",
              "| Configuration | " + " | ".join(CONDITION_LABELS[c] for c in ls.CONDITIONS) + " |",
              "|---|" + "---|" * len(ls.CONDITIONS)]
    for config in summary.config.unique():
        cells = []
        for condition in ls.CONDITIONS:
            r = summary[(summary.condition == condition) & (summary.config == config)].iloc[0]
            cells.append(f"{_fmt(r.mean_auc)} / {_fmt(r.mean_tpr_at_1pct)}")
        lines.append(f"| `{config}` | " + " | ".join(cells) + " |")
    lines += ["", "## Exploratory diagnostics", "",
              "| Condition | Saturated target losses | OUT-sd ratio p90/p10 (logit) | "
              "Non-member z mean / sd | |z|>3 | Mean TPR at 0.1% FPR, primary (descriptive) |",
              "|---|---|---|---|---|---|"]
    for condition in ls.CONDITIONS:
        d = results["conditions"][condition]["diagnostics"]
        r = summary[(summary.condition == condition)
                    & (summary.config == ls.config_name(ls.PRIMARY))].iloc[0]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {_fmt(d['saturated_fraction_target_losses'])} "
            f"| {_fmt(d['per_example_out_sd_logit_ratio_p90_p10'], 2)} "
            f"| {_fmt(d['nonmember_z_mean'], 3)} / {_fmt(d['nonmember_z_sd'], 3)} "
            f"| {_fmt(d['nonmember_z_abs_gt_3_fraction'])} | {_fmt(r['mean_tpr_at_0.1pct_descriptive'])} |")
    lines += ["",
              "Under a correct Gaussian OUT model the non-member z has mean 0, sd about 1 and "
              "|z|>3 about 0.003. TPR at 0.1% FPR allows zero false positives among 669 "
              "non-members and is descriptive only.", "",
              "![Attack comparison](attack_comparison.png)", "",
              "![Exploratory factorial](factorial_auc.png)", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir)
    summary = pd.read_csv(input_dir / "summary.csv")
    figure_attack_comparison(summary, input_dir / "attack_comparison.png")
    figure_factorial(summary, input_dir / "factorial_auc.png")
    (input_dir / "RESULTS.md").write_text(render(input_dir))
    print(f"wrote {input_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    sys.exit(main())
