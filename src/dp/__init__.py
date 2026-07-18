"""Differential privacy tooling for the insurance dataset."""

from .attacks import (
    AttackResult,
    lira_offline_attack,
    lira_online_attack,
    loss_threshold_attack,
)
from .audit import (
    AuditResult,
    audit_membership_scores,
    audit_scalar_mechanism,
    epsilon_lower_bound_binomial,
    epsilon_lower_bound_clopper_pearson,
    one_run_model_audit,
)
from .benchmark import run_benchmark, run_single_seed
from .canaries import (
    CanarySet,
    make_duplicated_canaries,
    make_feature_outlier_canaries,
    make_label_flip_canaries,
)
from .constants import RANDOM_STATE
from .evaluation import plot_privacy_utility, plot_roc_curves, privacy_utility_sweep
from .fairness import demographic_parity_difference, equalized_odds_difference
from .mechanisms import (
    add_gaussian_noise,
    add_laplace_noise,
    apply_randomized_response,
    calibrate_analytic_gaussian_sigma,
    exponential_mechanism,
    randomized_response,
)
from .metrics import (
    classification_metrics,
    expected_calibration_error,
    group_fairness_report,
    select_threshold,
)
from .models import build_decision_tree_model, build_model_registry, build_svm_model
from .pipeline import (
    apply_bounded_feature_noise,
    clip_numeric,
    compute_clip_bounds,
    load_dataset,
    preprocess_split,
    split_dataset,
)
from .tasks import TASKS, BenchmarkTask, TaskData, get_task, prepare_task_data
from .uncertainty import (
    bootstrap_confidence_interval,
    seed_confidence_interval,
    summarize_runs,
)

__all__ = [
    "TASKS",
    "BenchmarkTask",
    "TaskData",
    "bootstrap_confidence_interval",
    "classification_metrics",
    "expected_calibration_error",
    "get_task",
    "group_fairness_report",
    "prepare_task_data",
    "run_benchmark",
    "run_single_seed",
    "seed_confidence_interval",
    "select_threshold",
    "summarize_runs",
    "RANDOM_STATE",
    "AttackResult",
    "AuditResult",
    "CanarySet",
    "add_gaussian_noise",
    "add_laplace_noise",
    "apply_bounded_feature_noise",
    "apply_randomized_response",
    "audit_membership_scores",
    "audit_scalar_mechanism",
    "calibrate_analytic_gaussian_sigma",
    "clip_numeric",
    "compute_clip_bounds",
    "demographic_parity_difference",
    "epsilon_lower_bound_binomial",
    "epsilon_lower_bound_clopper_pearson",
    "equalized_odds_difference",
    "exponential_mechanism",
    "lira_offline_attack",
    "lira_online_attack",
    "loss_threshold_attack",
    "make_duplicated_canaries",
    "make_feature_outlier_canaries",
    "make_label_flip_canaries",
    "one_run_model_audit",
    "build_decision_tree_model",
    "build_model_registry",
    "build_svm_model",
    "load_dataset",
    "plot_privacy_utility",
    "plot_roc_curves",
    "preprocess_split",
    "privacy_utility_sweep",
    "randomized_response",
    "split_dataset",
]
