"""Benchmark task definitions with leakage-safe data preparation.

Each :class:`BenchmarkTask` names a prediction target and the features that
must be excluded from the predictor set.  :func:`prepare_task_data` turns a
raw DataFrame into a train/validation/test split with the target encoded as
0/1, guaranteeing that any data-dependent target construction (e.g. the
high-cost charge threshold) uses only the training partition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import RANDOM_STATE

logger = logging.getLogger(__name__)

#: Fraction of the full dataset reserved for the final, untouched test set.
DEFAULT_TEST_SIZE = 0.2
#: Fraction of the *remaining* (non-test) data used for validation.
DEFAULT_VAL_SIZE = 0.2
#: Training-only quantile used to define the high-cost target.
HIGH_COST_QUANTILE = 0.5


@dataclass(frozen=True)
class BenchmarkTask:
    """A single benchmark task: what to predict and what to exclude.

    Attributes:
        name: Unique task identifier used on the CLI and in result tables.
        target_column: Column the 0/1 target is derived from.
        excluded_features: Columns removed from the predictor set (the
            target column itself is always removed).
        task_type: Currently always ``"binary_classification"``.
        description: One-line human-readable description.
        positive_label: Raw value mapped to class 1 for categorical targets.
        legacy: True for easy-reference benchmarks that are not headline
            results (e.g. smoker prediction with ``charges`` available).
    """

    name: str
    target_column: str
    excluded_features: tuple[str, ...]
    task_type: str
    description: str
    positive_label: str | None = None
    legacy: bool = False


@dataclass(frozen=True)
class TaskData:
    """Leakage-safe train/validation/test partitions for one task."""

    task: BenchmarkTask
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    #: Sensitive-attribute values aligned with each partition (for fairness).
    sensitive_train: np.ndarray
    sensitive_val: np.ndarray
    sensitive_test: np.ndarray
    #: Task metadata, e.g. the training-only high-cost threshold.
    metadata: Mapping[str, float | str] = field(default_factory=dict)


TASKS: dict[str, BenchmarkTask] = {
    "smoker_without_charges": BenchmarkTask(
        name="smoker_without_charges",
        target_column="smoker",
        excluded_features=("charges",),
        task_type="binary_classification",
        description=(
            "Predict smoking status without access to insurance charges, "
            "which otherwise act as a near-direct proxy for smoking."
        ),
        positive_label="yes",
    ),
    "high_cost": BenchmarkTask(
        name="high_cost",
        target_column="charges",
        excluded_features=("charges",),
        task_type="binary_classification",
        description=(
            "Predict whether charges exceed the training-set median. "
            "The threshold is computed from the training partition only and "
            "raw charges are removed from the feature set."
        ),
    ),
    "smoker_with_charges_legacy": BenchmarkTask(
        name="smoker_with_charges_legacy",
        target_column="smoker",
        excluded_features=(),
        task_type="binary_classification",
        description=(
            "LEGACY easy-reference benchmark: predict smoking status with "
            "charges available. Unusually separable; not a headline result."
        ),
        positive_label="yes",
        legacy=True,
    ),
}


def get_task(name: str) -> BenchmarkTask:
    """Look up a task by name, raising a clear error on unknown names."""
    try:
        return TASKS[name]
    except KeyError as exc:
        known = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task {name!r}. Known tasks: {known}") from exc


def _encode_target(task: BenchmarkTask, raw: pd.Series) -> np.ndarray:
    if task.positive_label is not None:
        return (raw == task.positive_label).to_numpy(dtype=int)
    return raw.to_numpy(dtype=int)


def prepare_task_data(
    df: pd.DataFrame,
    task: BenchmarkTask,
    random_state: int = RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    sensitive_feature: str = "sex",
) -> TaskData:
    """Create leakage-safe train/validation/test partitions for a task.

    The final test set is split off first and never used for any
    data-dependent decision.  For the ``high_cost`` task the charge
    threshold is the ``HIGH_COST_QUANTILE`` quantile (the median) of
    ``charges`` computed on the *training partition only*, then applied to
    the validation and test partitions.

    Args:
        df: Raw insurance DataFrame.
        task: Task definition.
        random_state: Seed controlling both split stages.
        test_size: Fraction of rows held out as the final test set.
        val_size: Fraction of the non-test rows used for validation.
        sensitive_feature: Column carried alongside each partition for
            group-fairness evaluation (it remains a predictor too).

    Returns:
        A :class:`TaskData` with 0/1 targets and aligned sensitive values.

    Raises:
        ValueError: If required columns are missing from ``df``.
    """
    required = {task.target_column, sensitive_feature, *task.excluded_features}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    metadata: dict[str, float | str] = {}
    if task.name == "high_cost":
        # Split rows before the target exists; the threshold below uses only
        # the training partition, so no full-dataset statistic leaks into the
        # target definition. The median threshold makes classes near-balanced,
        # so unstratified splitting is acceptable here.
        rest, test = train_test_split(df, test_size=test_size, random_state=random_state)
        train, val = train_test_split(rest, test_size=val_size, random_state=random_state)
        threshold = float(train[task.target_column].quantile(HIGH_COST_QUANTILE))
        metadata["threshold"] = threshold
        metadata["threshold_quantile"] = HIGH_COST_QUANTILE
        logger.info(
            "Task %s: training-only charge threshold = %.2f (q=%.2f)",
            task.name,
            threshold,
            HIGH_COST_QUANTILE,
        )
        y_train = (train[task.target_column] > threshold).to_numpy(dtype=int)
        y_val = (val[task.target_column] > threshold).to_numpy(dtype=int)
        y_test = (test[task.target_column] > threshold).to_numpy(dtype=int)
    else:
        strat = df[task.target_column]
        rest, test = train_test_split(
            df, test_size=test_size, stratify=strat, random_state=random_state
        )
        train, val = train_test_split(
            rest,
            test_size=val_size,
            stratify=rest[task.target_column],
            random_state=random_state,
        )
        y_train = _encode_target(task, train[task.target_column])
        y_val = _encode_target(task, val[task.target_column])
        y_test = _encode_target(task, test[task.target_column])

    drop = sorted({task.target_column, *task.excluded_features})
    X_train = train.drop(columns=drop)
    X_val = val.drop(columns=drop)
    X_test = test.drop(columns=drop)

    return TaskData(
        task=task,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        sensitive_train=train[sensitive_feature].to_numpy(),
        sensitive_val=val[sensitive_feature].to_numpy(),
        sensitive_test=test[sensitive_feature].to_numpy(),
        metadata=metadata,
    )
