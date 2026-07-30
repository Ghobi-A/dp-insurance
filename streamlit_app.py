"""Interactive results explorer for the differential-privacy benchmark."""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dp.dashboard import (
    AUDIT_METRICS,
    FAIRNESS_METRICS,
    METRIC_LABELS,
    MODEL_LABELS,
    TASK_LABELS,
    UTILITY_METRICS,
    available_metrics,
    best_nonprivate,
    dp_budget_row,
    group_rate_rows,
    headline_statistics,
    load_summary,
    metric_rows,
)

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "reports" / "generated" / "summary.csv"

st.set_page_config(
    page_title="DP Insurance Benchmark",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1240px; padding-top: 1.8rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 14px;}
      .small-note {font-size: .92rem; opacity: .78;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_summary(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return load_summary(path)


def format_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def interval_tooltips(metric: str) -> list[alt.Tooltip]:
    return [
        alt.Tooltip("configuration:N", title="Configuration"),
        alt.Tooltip("epsilon_label:N", title="Achieved ε"),
        alt.Tooltip("mean:Q", title=METRIC_LABELS.get(metric, metric), format=".3f"),
        alt.Tooltip("ci_lower:Q", title="CI lower", format=".3f"),
        alt.Tooltip("ci_upper:Q", title="CI upper", format=".3f"),
        alt.Tooltip("n_runs:Q", title="Seeds"),
    ]


def privacy_curve(frame: pd.DataFrame, metric: str, baseline: pd.Series | None = None) -> alt.Chart:
    y_title = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    base = alt.Chart(frame).encode(
        x=alt.X("epsilon:Q", title="Achieved privacy budget ε", scale=alt.Scale(zero=False)),
        color=alt.Color("model_label:N", title="Model"),
        tooltip=interval_tooltips(metric),
    )
    line = base.mark_line(point=True).encode(
        y=alt.Y("mean:Q", title=y_title, scale=alt.Scale(zero=False))
    )
    error = base.mark_errorbar(ticks=True).encode(
        y=alt.Y("ci_lower:Q", title=y_title, scale=alt.Scale(zero=False)),
        y2="ci_upper:Q",
    )
    chart: alt.Chart = line + error
    if baseline is not None:
        ref = pd.DataFrame(
            {
                "baseline": [float(baseline["mean"])],
                "label": [str(baseline["model_label"])],
            }
        )
        rule = alt.Chart(ref).mark_rule(strokeDash=[7, 5]).encode(
            y="baseline:Q",
            tooltip=[
                alt.Tooltip("label:N", title="Reference"),
                alt.Tooltip("baseline:Q", title=y_title, format=".3f"),
            ],
        )
        chart = chart + rule
    return chart.properties(height=430).interactive()


def comparison_chart(frame: pd.DataFrame, metric: str) -> alt.Chart:
    order = frame.sort_values("mean", ascending=False)["configuration"].tolist()
    base = alt.Chart(frame).encode(
        y=alt.Y("configuration:N", sort=order, title=None),
        tooltip=interval_tooltips(metric),
    )
    bars = base.mark_bar().encode(
        x=alt.X("mean:Q", title=METRIC_LABELS.get(metric, metric), scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", legend=None),
    )
    error = base.mark_errorbar(ticks=True).encode(x="ci_lower:Q", x2="ci_upper:Q")
    return (bars + error).properties(height=max(320, 36 * len(frame)))


def render_data_table(frame: pd.DataFrame) -> None:
    columns = [
        "task_label",
        "configuration",
        "mean",
        "ci_lower",
        "ci_upper",
        "n_runs",
    ]
    available = [column for column in columns if column in frame.columns]
    st.dataframe(frame[available], use_container_width=True, hide_index=True)


try:
    summary = cached_summary(str(SUMMARY_PATH), SUMMARY_PATH.stat().st_mtime_ns)
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.code(
        "python -m dp.benchmark --task all --seeds 0 1 2 3 4 --output-dir reports/generated\n"
        "python -m dp.reporting --results reports/generated --reports-dir reports"
    )
    st.stop()

stats = headline_statistics(summary)

st.title("Differential Privacy in Machine Learning")
st.subheader("An interactive insurance benchmark for privacy, utility, calibration and fairness")
st.caption(
    "Read-only explorer over versioned benchmark outputs. No model training, data upload, "
    "or insurance pricing is performed in this app."
)

with st.sidebar:
    st.header("Benchmark scope")
    st.markdown(
        "**Dataset:** public 1,338-row teaching dataset  \n"
        "**Evaluation:** leakage-safe train/validation/test pipeline  \n"
        "**Uncertainty:** repeated-seed 95% t-intervals  \n"
        "**Privacy:** achieved ε from the Opacus RDP accountant"
    )
    st.divider()
    st.markdown("[View source repository](https://github.com/Ghobi-A/dp-insurance)")
    st.caption("Results shown here are research/portfolio evidence, not clinical or actuarial advice.")

overview_tab, utility_tab, proxy_tab, fairness_tab, audit_tab, methods_tab = st.tabs(
    ["Overview", "Privacy–utility", "Proxy leakage", "Fairness", "Privacy audits", "Methodology"]
)

with overview_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Legacy smoker ROC-AUC",
        f"{stats['legacy_roc_auc']:.3f}",
        help="Best non-private, non-dummy model when charges is available.",
    )
    c2.metric(
        "Proxy-safe smoker ROC-AUC",
        f"{stats['corrected_roc_auc']:.3f}",
        delta=f"{stats['proxy_drop']:+.3f}",
        help="Best non-private result after removing charges.",
    )
    c3.metric(
        "Strict-DP high-cost ROC-AUC",
        f"{stats['high_cost_dp_roc_auc']:.3f}",
        delta=f"{stats['high_cost_retention']:.1%} retained",
        help=f"DP-SGD at achieved ε={stats['high_cost_dp_epsilon']:.2f} against the best non-private model.",
    )
    c4.metric(
        "Fairness-gap change",
        "n/a" if np.isnan(stats["fairness_change"]) else f"{stats['fairness_change']:+.3f}",
        help="Change in DP-SGD demographic-parity difference from the strictest to loosest budget on the proxy-safe smoker task.",
    )

    st.markdown("### What the benchmark establishes")
    left, right = st.columns(2)
    with left:
        st.info(
            "**The original smoker task was dominated by a proxy.** Removing `charges` "
            "reduces the best ROC-AUC from near-perfect discrimination to approximately chance-level performance."
        )
        st.success(
            "**DP-SGD can preserve useful signal.** On high-cost classification, the strictest "
            "tested budget retains most of the strongest non-private model's ROC-AUC."
        )
    with right:
        st.warning(
            "**Privacy and fairness move together.** Noise can suppress both predictive signal "
            "and measured group disparity; a smaller gap is not automatically a fairer useful model."
        )
        st.info(
            "**Empirical attacks are diagnostics, not certificates.** Formal privacy comes from "
            "the accountant; attack results provide practical leakage evidence and lower bounds."
        )

    task = st.selectbox(
        "Inspect a task",
        options=sorted(summary["task"].unique()),
        format_func=lambda value: TASK_LABELS.get(value, value),
        key="overview_task",
    )
    rows = metric_rows(summary, task=task, metric="roc_auc")
    st.altair_chart(comparison_chart(rows, "roc_auc"), use_container_width=True)
    with st.expander("Show result rows"):
        render_data_table(rows)

with utility_tab:
    st.markdown("### Privacy–utility explorer")
    col_task, col_metric, col_baseline = st.columns([1.25, 1, 1.25])
    task = col_task.selectbox(
        "Task",
        options=sorted(summary["task"].unique()),
        format_func=lambda value: TASK_LABELS.get(value, value),
        key="utility_task",
    )
    utility_metrics = available_metrics(summary, UTILITY_METRICS)
    metric = col_metric.selectbox(
        "Metric",
        options=utility_metrics,
        format_func=lambda value: METRIC_LABELS.get(value, value),
        key="utility_metric",
    )
    baseline_options = [
        model
        for model in ("logistic", "svm", "gradient_boosting", "neural")
        if not metric_rows(summary, task=task, metric=metric, models=(model,)).empty
    ]
    default_baseline = best_nonprivate(summary, task, metric)["model"]
    baseline_model = col_baseline.selectbox(
        "Non-private reference",
        options=baseline_options,
        index=baseline_options.index(default_baseline),
        format_func=lambda value: MODEL_LABELS.get(value, value),
        key="utility_baseline",
    )

    dp_rows = metric_rows(summary, task=task, metric=metric, models=("dpsgd",))
    dp_rows = dp_rows[np.isfinite(dp_rows["epsilon"])].sort_values("epsilon")
    baseline = metric_rows(summary, task=task, metric=metric, models=(baseline_model,)).iloc[0]
    st.altair_chart(privacy_curve(dp_rows, metric, baseline), use_container_width=True)

    strict = dp_budget_row(summary, task, metric, strictest=True)
    loose = dp_budget_row(summary, task, metric, strictest=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Strictest tested ε", f"{strict['epsilon']:.3f}")
    c2.metric("Metric at strictest ε", format_value(metric, strict["mean"]))
    c3.metric("Metric at loosest ε", format_value(metric, loose["mean"]))
    st.caption(
        "For ROC-AUC, PR-AUC and macro F1, higher is better. For Brier score and ECE, lower is better. "
        "Intervals describe pipeline run-to-run variation across seeds."
    )
    with st.expander("Show plotted data"):
        render_data_table(dp_rows)

with proxy_tab:
    st.markdown("### Proxy leakage demonstration")
    st.write(
        "The legacy smoker classifier can use insurance charges, which are tightly associated with smoking. "
        "The corrected task removes that feature before splitting and evaluation."
    )
    proxy_metric = st.selectbox(
        "Metric",
        options=[m for m in ("roc_auc", "pr_auc", "macro_f1") if m in set(summary["metric"])],
        format_func=lambda value: METRIC_LABELS.get(value, value),
        key="proxy_metric",
    )
    models = ("logistic", "svm", "gradient_boosting", "neural")
    legacy = metric_rows(
        summary,
        task="smoker_with_charges_legacy",
        metric=proxy_metric,
        models=models,
    )
    corrected = metric_rows(
        summary,
        task="smoker_without_charges",
        metric=proxy_metric,
        models=models,
    )
    proxy_rows = pd.concat([legacy, corrected], ignore_index=True)
    proxy_rows["task_short"] = proxy_rows["task"].map(
        {
            "smoker_with_charges_legacy": "Charges available",
            "smoker_without_charges": "Charges removed",
        }
    )
    chart = (
        alt.Chart(proxy_rows)
        .mark_bar()
        .encode(
            x=alt.X("model_label:N", title=None, axis=alt.Axis(labelAngle=-20)),
            xOffset="task_short:N",
            y=alt.Y("mean:Q", title=METRIC_LABELS.get(proxy_metric, proxy_metric), scale=alt.Scale(zero=False)),
            color=alt.Color("task_short:N", title="Task design"),
            tooltip=[
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("task_short:N", title="Task"),
                alt.Tooltip("mean:Q", title=METRIC_LABELS.get(proxy_metric, proxy_metric), format=".3f"),
                alt.Tooltip("ci_lower:Q", title="CI lower", format=".3f"),
                alt.Tooltip("ci_upper:Q", title="CI upper", format=".3f"),
            ],
        )
        .properties(height=430)
    )
    st.altair_chart(chart, use_container_width=True)
    legacy_best = legacy.loc[legacy["mean"].idxmax()]
    corrected_best = corrected.loc[corrected["mean"].idxmax()]
    st.metric(
        "Best-model change after removing charges",
        f"{corrected_best['mean'] - legacy_best['mean']:+.3f}",
        help=f"{legacy_best['model_label']} vs {corrected_best['model_label']} on {METRIC_LABELS.get(proxy_metric, proxy_metric)}.",
    )
    st.caption(
        "This is a benchmark-design finding, not evidence that the remaining features can support a useful smoker classifier."
    )

with fairness_tab:
    st.markdown("### Privacy–fairness explorer")
    fairness_tasks = [
        task
        for task in sorted(summary["task"].unique())
        if not metric_rows(
            summary,
            task=task,
            metric="demographic_parity_difference",
            models=("dpsgd",),
        ).empty
    ]
    task = st.selectbox(
        "Task",
        options=fairness_tasks,
        format_func=lambda value: TASK_LABELS.get(value, value),
        key="fairness_task",
    )
    fairness_metric = st.radio(
        "Gap metric",
        options=available_metrics(summary, FAIRNESS_METRICS),
        format_func=lambda value: METRIC_LABELS.get(value, value),
        horizontal=True,
    )
    fairness_rows = metric_rows(summary, task=task, metric=fairness_metric, models=("dpsgd",))
    fairness_rows = fairness_rows[np.isfinite(fairness_rows["epsilon"])].sort_values("epsilon")
    st.altair_chart(privacy_curve(fairness_rows, fairness_metric), use_container_width=True)

    rate = st.radio("Group-rate detail", options=["tpr", "fpr"], format_func=str.upper, horizontal=True)
    group_rows = group_rate_rows(summary, task, rate)
    if group_rows.empty:
        st.info("Group-rate rows are unavailable for this task.")
    else:
        group_rows["configuration"] = group_rows["group"]
        group_chart = (
            alt.Chart(group_rows)
            .mark_line(point=True)
            .encode(
                x=alt.X("epsilon:Q", title="Achieved privacy budget ε", scale=alt.Scale(zero=False)),
                y=alt.Y("mean:Q", title=f"{rate.upper()} by group", scale=alt.Scale(zero=False)),
                color=alt.Color("group:N", title="Group"),
                tooltip=[
                    alt.Tooltip("group:N", title="Group"),
                    alt.Tooltip("epsilon:Q", title="Achieved ε", format=".3f"),
                    alt.Tooltip("mean:Q", title=rate.upper(), format=".3f"),
                    alt.Tooltip("ci_lower:Q", title="CI lower", format=".3f"),
                    alt.Tooltip("ci_upper:Q", title="CI upper", format=".3f"),
                ],
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(group_chart, use_container_width=True)
    st.warning(
        "A smaller measured gap can result from a model becoming less informative for everyone. "
        "Interpret fairness jointly with utility, group rates and sample sizes."
    )

with audit_tab:
    st.markdown("### Empirical privacy diagnostics")
    a1, a2, a3 = st.columns(3)
    a1.info("**Formal accounting**\n\nThe RDP accountant provides the claimed upper-bound guarantee at δ=1e-5.")
    a2.info("**Auditor lower bounds**\n\nCanary-based tests try to prove the implementation leaks at least some ε.")
    a3.info("**Attack performance**\n\nMembership-inference metrics estimate practical distinguishability for a chosen attack.")

    audit_metrics = available_metrics(summary, AUDIT_METRICS)
    if not audit_metrics:
        st.info("No membership-inference summary metrics were found.")
    else:
        col1, col2 = st.columns(2)
        task = col1.selectbox(
            "Task",
            options=sorted(summary["task"].unique()),
            format_func=lambda value: TASK_LABELS.get(value, value),
            key="audit_task",
        )
        metric = col2.selectbox(
            "Attack metric",
            options=audit_metrics,
            format_func=lambda value: METRIC_LABELS.get(value, value),
            key="audit_metric",
        )
        audit_rows = metric_rows(summary, task=task, metric=metric, models=("dpsgd",))
        audit_rows = audit_rows[np.isfinite(audit_rows["epsilon"])].sort_values("epsilon")
        chart = privacy_curve(audit_rows, metric)
        if metric == "mia_loss_threshold_auc":
            chance = alt.Chart(pd.DataFrame({"chance": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(
                y="chance:Q",
                tooltip=[alt.Tooltip("chance:Q", title="Chance AUC", format=".1f")],
            )
            chart = chart + chance
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Chance-level attack performance means this attack did not distinguish members from non-members; "
            "it does not verify the formal differential-privacy guarantee."
        )
        with st.expander("Show attack result rows"):
            render_data_table(audit_rows)

with methods_tab:
    st.markdown("### Experimental design")
    st.markdown(
        """
        1. **Leakage-safe task construction:** preprocessing is fit on the training partition only.  
        2. **Validation-only decisions:** thresholds, early stopping and tuning use validation data.  
        3. **Single test evaluation:** the test partition is touched once per run.  
        4. **Repeated seeds:** reported intervals quantify split, initialisation and training variability.  
        5. **Achieved privacy budgets:** ε is taken from the Opacus RDP accountant rather than copied from a requested target.  
        6. **Defence in depth:** formal accounting is supplemented with canaries, membership-inference attacks and privacy regression tests.
        """
    )
    st.markdown("### Important limitations")
    st.markdown(
        """
        - The dataset is a small public teaching dataset, not real clinical or actuarial evidence.
        - Repeated-seed intervals are not independent-sample confidence guarantees.
        - Fairness analysis is limited to a binary `sex` attribute and can be unstable for small groups.
        - The proxy-safe smoker task has little remaining predictive signal.
        - Empirical attacks on a small dataset have limited power to certify small ε values.
        """
    )
    st.code(
        "python -m dp.benchmark --task all --seeds 0 1 2 3 4 --output-dir reports/generated\n"
        "python -m dp.reporting --results reports/generated --reports-dir reports\n"
        "streamlit run streamlit_app.py",
        language="bash",
    )
