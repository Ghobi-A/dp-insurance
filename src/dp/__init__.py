"""Differential privacy tooling for the insurance dataset."""

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
from .models import build_decision_tree_model, build_model_registry, build_svm_model
from .pipeline import (
    apply_bounded_feature_noise,
    clip_numeric,
    compute_clip_bounds,
    load_dataset,
    preprocess_split,
    split_dataset,
)

__all__ = [
    "RANDOM_STATE",
    "add_gaussian_noise",
    "add_laplace_noise",
    "apply_bounded_feature_noise",
    "apply_randomized_response",
    "calibrate_analytic_gaussian_sigma",
    "clip_numeric",
    "compute_clip_bounds",
    "demographic_parity_difference",
    "equalized_odds_difference",
    "exponential_mechanism",
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
