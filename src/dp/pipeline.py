"""Training pipeline with leakage-safe preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import RANDOM_STATE
from .mechanisms import add_gaussian_noise, add_laplace_noise


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the insurance dataset from a repo-root-relative path."""
    return pd.read_csv(Path(path))


def split_dataset(
    df: pd.DataFrame,
    target: str = "smoker",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    """Split the dataset into train/test partitions."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric/categorical columns."""
    numeric_features = df.select_dtypes(include="number").columns
    categorical_features = df.select_dtypes(exclude="number").columns
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )


def compute_clip_bounds(
    df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Compute per-column clip bounds for numeric columns from quantiles.

    .. warning::
        Quantiles computed from the private data are themselves
        data-dependent, so bounds derived this way leak a small amount of
        information that is not covered by the ε of the noise added later.
        For a rigorous end-to-end guarantee, prefer fixed, publicly known
        domain bounds (e.g. age ∈ [18, 65]) and pass them to
        :func:`clip_numeric` / :func:`apply_bounded_feature_noise` directly.
    """
    numeric = df.select_dtypes(include="number")
    return pd.DataFrame(
        {"lower": numeric.quantile(lower_q), "upper": numeric.quantile(upper_q)}
    )


def clip_numeric(df: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """Clip numeric columns to per-column ``[lower, upper]`` bounds.

    ``bounds`` must be indexed by column name with ``lower`` and ``upper``
    columns, as produced by :func:`compute_clip_bounds`.
    """
    clipped = df.copy()
    for column in bounds.index:
        clipped[column] = clipped[column].clip(
            bounds.loc[column, "lower"], bounds.loc[column, "upper"]
        )
    return clipped


def apply_bounded_feature_noise(
    df: pd.DataFrame,
    bounds: pd.DataFrame,
    mechanism: str = "laplace",
    epsilon: float = 1.0,
    delta: float = 1e-5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Release all numeric columns under a single ε (or (ε, δ)) budget.

    Each numeric column listed in ``bounds`` is clipped to its bounds and
    rescaled to [0, 1], so that changing one record moves each of the d
    rescaled columns by at most 1.  The joint sensitivity of the full-row
    release is therefore d in L1 (Laplace) and √d in L2 (Gaussian).  Noise
    calibrated to that joint sensitivity is added to every column, and values
    are mapped back to their original scale.  The stated ``epsilon`` covers
    the release of *all* numeric columns together — no extra composition
    accounting is needed.

    This replaces the earlier approach of using the maximum column range as a
    shared sensitivity, which both under-accounted the joint release and let
    wide columns (e.g. ``charges``) dictate the noise scale for narrow ones
    (e.g. ``children``).

    Args:
        df: Input DataFrame.  Columns not listed in ``bounds`` (e.g.
            categorical columns) pass through unchanged.
        bounds: Per-column ``lower``/``upper`` bounds indexed by column name.
            Use fixed public domain bounds where possible (see
            :func:`compute_clip_bounds` for the caveat on data-derived bounds).
        mechanism: ``"laplace"`` (ε-DP) or ``"gaussian"`` ((ε, δ)-DP).
        epsilon: Total privacy budget for the joint release.
        delta: Failure probability (Gaussian only).
        random_state: Integer seed for reproducibility.

    Returns:
        Copy of ``df`` with calibrated noise added to the bounded numeric
        columns, in the original column order and scale.

    Raises:
        ValueError: If ``mechanism`` is unknown or any bound is degenerate
            (``upper`` ≤ ``lower``).
    """
    widths = bounds["upper"] - bounds["lower"]
    if (widths <= 0).any():
        bad = list(widths[widths <= 0].index)
        raise ValueError(f"Degenerate clip bounds (upper <= lower) for: {bad}")

    columns = list(bounds.index)
    unit = (clip_numeric(df, bounds)[columns] - bounds["lower"]) / widths

    n_cols = len(columns)
    if mechanism == "laplace":
        noisy_unit = add_laplace_noise(
            unit, epsilon=epsilon, sensitivity=float(n_cols), random_state=random_state
        )
    elif mechanism == "gaussian":
        noisy_unit = add_gaussian_noise(
            unit,
            epsilon=epsilon,
            delta=delta,
            sensitivity=float(np.sqrt(n_cols)),
            random_state=random_state,
        )
    else:
        raise ValueError("Only 'laplace' and 'gaussian' mechanisms are supported.")

    noisy = df.copy()
    noisy[columns] = noisy_unit * widths + bounds["lower"]
    return noisy


def apply_feature_noise(
    df: pd.DataFrame,
    mechanism: str = "laplace",
    epsilon: float = 0.1,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Apply feature-level noise to numeric columns only.

    ``sensitivity`` is used as-is for every numeric column, so for the stated
    ``epsilon`` to cover the joint release of all columns it must be the
    whole-row sensitivity (L1 for Laplace, L2 for Gaussian).  Prefer
    :func:`apply_bounded_feature_noise`, which derives this automatically
    from per-column clip bounds.
    """
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")

    if mechanism == "laplace":
        noisy_num = add_laplace_noise(
            numeric,
            epsilon=epsilon,
            sensitivity=sensitivity,
            random_state=random_state,
        )
    elif mechanism == "gaussian":
        noisy_num = add_gaussian_noise(
            numeric,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            random_state=random_state,
        )
    else:
        raise ValueError("Only 'laplace' and 'gaussian' mechanisms are supported.")

    return pd.concat([noisy_num, categorical], axis=1)[df.columns]


def preprocess_split(split: DatasetSplit) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit preprocessing on train only, then transform train/test."""
    preprocessor = build_preprocessor(split.X_train)
    X_train = preprocessor.fit_transform(split.X_train)
    X_test = preprocessor.transform(split.X_test)
    return X_train, X_test, preprocessor
