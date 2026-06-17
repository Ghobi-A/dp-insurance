"""Differential privacy noise mechanisms.

This module provides four DP mechanisms covering the most common use cases:

- ``add_laplace_noise``      — ε-DP for numeric features (pure DP, no δ)
- ``add_gaussian_noise``     — (ε, δ)-DP for numeric features (approximate DP)
- ``randomized_response``    — local ε-DP for binary categorical values
- ``apply_randomized_response`` — convenience wrapper for multiple columns
- ``exponential_mechanism``  — ε-DP for discrete/categorical outputs

Choosing a mechanism:
    Use Laplace when you need a pure DP guarantee and your sensitivity is
    measured in L1 norm.  Use Gaussian when a small δ is acceptable and you
    want lower noise variance (useful for high-dimensional data).  Use
    randomized response for binary categorical features where local DP is
    required (no trusted aggregator).  Use the exponential mechanism when the
    output is discrete and a utility score over candidates can be defined.

Privacy budget guidance:
    ε < 1      — strict privacy; feature-level noise collapses utility (see
                 findings_report.md).  DP-SGD is the recommended alternative.
    1 ≤ ε ≤ 5  — moderate privacy; feature-level Laplace remains viable.
    ε > 10     — loose privacy; noise impact on accuracy is negligible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import get_rng


def add_laplace_noise(
    data: pd.DataFrame,
    epsilon: float = 0.1,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add Laplace noise to all numeric columns, providing ε-differential privacy.

    The Laplace mechanism draws noise from Lap(0, sensitivity/ε) and adds it
    independently to every numeric cell.  It satisfies pure ε-DP with no
    failure probability δ, making it the standard choice when a tight,
    unconditional privacy guarantee is required.

    When to use:
        - Numeric (continuous or integer) features.
        - You need a pure ε-DP guarantee without a δ term.
        - ε ≥ 1.0 for reasonable utility (see project findings: ROC-AUC drops
          to ≈ 0.43 at ε = 0.01 on this dataset).
        - The data will be released or used in a downstream model, not just
          queried once (prefer query-level mechanisms for one-shot releases).

    Sensitivity calculation:
        Sensitivity is the maximum change in a column value when one row
        changes.  For clipped features the L1 sensitivity equals the clip
        bound.  Example: if age is clipped to [0, 100], sensitivity = 100 for
        that column.  When multiple numeric columns are perturbed independently,
        each receives the *same* scale = sensitivity/ε, so set sensitivity to
        the tightest meaningful per-column bound (or clip first).

        Practical recipe::

            clip_bound = df["age"].quantile(0.99)   # e.g. 65.0
            df["age"] = df["age"].clip(upper=clip_bound)
            noisy = add_laplace_noise(df, epsilon=1.0, sensitivity=clip_bound)

    Production considerations:
        - Always clip numeric columns before calling this function.  Without
          clipping, sensitivity is unbounded and the DP guarantee is void.
        - Under basic composition, applying this across k independent queries
          consumes ε·k total budget.  Use advanced composition (e.g. RDP) if
          running many queries.
        - The noise is added in-place to a copy; the original DataFrame is not
          mutated.
        - Do not reuse the same ε budget across training and evaluation data
          derived from the same individuals.

    Args:
        data: Input DataFrame.  Only ``dtype`` numeric columns are perturbed;
            categorical columns pass through unchanged.
        epsilon: Privacy budget (ε > 0).  Smaller values mean stronger privacy
            and more noise.
        sensitivity: L1 sensitivity of the query — the maximum absolute change
            in any single column value when one record changes.  Must reflect
            the actual clip bound applied before calling this function.
        random_state: Integer seed for reproducibility.  ``None`` uses the
            global NumPy RNG.

    Returns:
        Copy of ``data`` with Laplace noise added to numeric columns.

    Raises:
        ValueError: If ``epsilon`` is not positive.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    scale = sensitivity / epsilon
    rng = get_rng(random_state)
    noisy = data.copy()
    num_cols = data.select_dtypes(include="number").columns
    if len(num_cols):
        noise = rng.laplace(0, scale, size=(len(data), len(num_cols)))
        noisy[num_cols] = data[num_cols] + noise
    return noisy


def add_gaussian_noise(
    data: pd.DataFrame,
    epsilon: float = 0.1,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Add Gaussian noise to all numeric columns, providing (ε, δ)-differential privacy.

    Noise is drawn from N(0, σ²) where σ is derived from the analytic Gaussian
    mechanism: σ = sensitivity · √(2 · ln(1.25/δ)) / ε.  This satisfies
    (ε, δ)-DP, meaning privacy holds with probability 1 − δ.  For equal ε,
    Gaussian noise has lower variance than Laplace in high dimensions (L2 vs
    L1 sensitivity), but the δ failure probability must be accepted.

    When to use:
        - Numeric features where a small probability of privacy failure (δ) is
          acceptable under your threat model.
        - High-dimensional settings where L2 sensitivity is tighter than L1.
        - You are composing many mechanisms and using RDP or zCDP accounting,
          which handles Gaussian noise more efficiently than Laplace.
        - ε ≥ 1.0 for usable utility — in this project Gaussian at ε = 1.0
          yields ROC-AUC ≈ 0.86 (vs 0.978 for Laplace), so Laplace is
          generally preferred at feature level.

    Sensitivity calculation:
        For the Gaussian mechanism, sensitivity should be the L2 sensitivity
        (Euclidean norm of the worst-case change across all columns).  For a
        single column clipped to [−b, b], L2 sensitivity = b.  For d columns
        with independent clip bounds b_1 … b_d, the joint L2 sensitivity is
        √(b_1² + … + b_d²) if you treat the full row as a single vector query;
        or use per-column bounds with separate calls if columns are released
        independently.

        Example::

            clip_bound = 1.0   # after standard-scaling, values ≈ in [−3, 3]
            noisy = add_gaussian_noise(df, epsilon=1.0, delta=1e-5,
                                       sensitivity=clip_bound)

    Production considerations:
        - Choose δ ≪ 1/n where n is the dataset size.  A common rule of thumb
          is δ = 1e-5 for datasets with tens of thousands of records.  δ = 0.01
          is almost certainly too large for any production deployment.
        - Gaussian noise is not suitable when a pure DP guarantee is required
          (e.g. by regulation).  Use Laplace in that case.
        - The σ formula used here is the classic analytic bound; tighter bounds
          (e.g. from Balle & Wang 2018) can reduce noise for the same (ε, δ).
        - Like Laplace, clip before calling and account for composition across
          multiple releases.

    Args:
        data: Input DataFrame.  Only ``dtype`` numeric columns are perturbed.
        epsilon: Privacy budget (ε > 0).
        delta: Failure probability (0 < δ < 1).  Typical value: 1e-5.
        sensitivity: L2 sensitivity of the query.
        random_state: Integer seed for reproducibility.

    Returns:
        Copy of ``data`` with Gaussian noise added to numeric columns.

    Raises:
        ValueError: If ``epsilon`` is not positive or ``delta`` is not in (0, 1).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be between 0 and 1")
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    rng = get_rng(random_state)
    noisy = data.copy()
    num_cols = data.select_dtypes(include="number").columns
    if len(num_cols):
        noise = rng.normal(0, sigma, size=(len(data), len(num_cols)))
        noisy[num_cols] = data[num_cols] + noise
    return noisy


def randomized_response(
    series: pd.Series,
    epsilon: float = 1.0,
    random_state: int | None = None,
) -> pd.Series:
    """Apply the randomized response mechanism to a binary categorical Series.

    Each value is kept with probability p = e^ε / (e^ε + 1) and flipped to the
    other value with probability 1 − p.  This satisfies ε-local differential
    privacy (local DP): privacy is guaranteed even without a trusted data
    curator, because each individual perturbs their own value before sharing.

    When to use:
        - Binary categorical features (e.g. yes/no, true/false, 0/1).
        - Local DP settings where individuals cannot trust a central server.
        - Survey data collection where participants self-report sensitive
          attributes (classic application: "Did you commit crime X?").
        - ε ∈ [1, 3] is typical for local DP deployments; lower ε makes the
          majority of responses random, destroying signal.

    Sensitivity:
        Randomized response has inherent bounded sensitivity — the output
        domain is fixed to the two observed values, so no explicit sensitivity
        parameter is needed.  The flip probability is fully determined by ε.

        Proof of ε-LDP: for any pair of true values v, v′ and any output o,
        Pr[M(v) = o] / Pr[M(v′) = o] = e^ε / 1 = e^ε. ∎

    Production considerations:
        - Works only on binary series.  For k > 2 categories, use the
          k-ary randomized response or RAPPOR mechanisms instead.
        - Local DP typically requires a larger ε than central DP to achieve
          comparable utility, because noise is added before aggregation rather
          than to aggregate statistics.
        - NaN values are propagated unchanged.  Ensure missing values are
          handled before calling.
        - Composition: applying randomized response to k binary columns with
          the same ε costs ε · k under basic composition.

    Args:
        series: Binary pandas Series.  Must have exactly two distinct non-NaN
            values; raises ``ValueError`` otherwise.
        epsilon: Privacy budget (ε > 0).  Higher values keep more true values.
        random_state: Integer seed for reproducibility.

    Returns:
        New Series with the same index and dtype, values perturbed in place.

    Raises:
        ValueError: If ``epsilon`` ≤ 0 or ``series`` is not binary.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = series.dropna().unique()
    if len(values) != 2:
        raise ValueError("randomized_response expects a binary series")
    value_a, value_b = values
    rng = get_rng(random_state)
    prob_keep = np.exp(epsilon) / (np.exp(epsilon) + 1)
    keep_mask = rng.random(len(series)) < prob_keep
    flipped = series.copy()
    flip_values = np.where(flipped == value_a, value_b, value_a)
    flipped = flipped.where(keep_mask, flip_values)
    return flipped


def apply_randomized_response(
    df: pd.DataFrame,
    columns: list[str],
    epsilon: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Apply randomized response independently to multiple binary columns.

    Convenience wrapper around :func:`randomized_response` that iterates over
    ``columns`` and applies independent perturbation to each.  Each column
    consumes the full ``epsilon`` budget, so under basic composition the total
    privacy cost is ε · len(columns).  If budget is a concern, consider
    splitting ε across columns or using advanced composition bounds.

    When to use:
        - Multiple binary columns in the same DataFrame need local DP.
        - You want a single call with a consistent ε across all columns.
        - Columns are treated as independent queries (no joint sensitivity
          across columns is required by your threat model).

    Production considerations:
        - Each column gets an independent random draw; per-column seeds are
          derived deterministically from ``random_state`` to ensure full
          reproducibility while preserving independence between columns.
        - If columns are correlated (e.g. one-hot encoded from the same
          categorical), perturbing them independently may produce inconsistent
          output (e.g. two "yes" values for a one-hot variable).  Re-encode
          after perturbation or use the exponential mechanism for multi-class
          categoricals instead.

    Args:
        df: Input DataFrame containing ``columns``.
        columns: List of column names to perturb.  Each must be a binary
            series.  An empty list returns an unmodified copy.
        epsilon: Privacy budget applied to each column independently.
        random_state: Integer seed for reproducibility.

    Returns:
        Copy of ``df`` with randomized response applied to ``columns``.
    """
    if not columns:
        return df.copy()
    rng = get_rng(random_state)
    noisy = df.copy()
    for column in columns:
        seed = None if random_state is None else int(rng.integers(0, 1_000_000))
        noisy[column] = randomized_response(noisy[column], epsilon=epsilon, random_state=seed)
    return noisy


def exponential_mechanism(
    candidates: list,
    utility_scores: np.ndarray,
    epsilon: float,
    sensitivity: float = 1.0,
    random_state: int | None = None,
):
    """Select a candidate using the exponential mechanism, providing ε-DP.

    Samples from ``candidates`` with probability proportional to
    exp(ε · score / (2 · sensitivity)).  Candidates with higher utility scores
    are exponentially more likely to be selected, while the randomisation
    provides ε-differential privacy with respect to the global L1 sensitivity
    of the utility function.

    When to use:
        - The output must be drawn from a discrete set (e.g. selecting a model
          hyperparameter, a query column, a histogram bin, or a category).
        - The utility of each candidate can be scored as a function of the
          dataset (e.g. count of matching records, a loss value, a frequency).
        - You need pure ε-DP (no δ) for a non-numeric output type.
        - Avoid when the candidate set is continuous or very large — the
          mechanism requires enumerating all candidates and their scores.

    Sensitivity calculation:
        Sensitivity is the maximum change in *any single candidate's* utility
        score when one record in the dataset is added or removed.  For
        count-based utilities (e.g. score[c] = number of records with value c),
        adding or removing one record changes exactly one count by 1, so
        sensitivity = 1.  For average-based utilities over clipped values in
        [0, b], sensitivity = b / n where n is the dataset size.

        Example — privately select the most common age bucket::

            buckets = ["18-30", "31-45", "46-60", "61+"]
            counts  = np.array([df["age_bucket"].value_counts()[b]
                                for b in buckets])
            chosen  = exponential_mechanism(buckets, counts, epsilon=1.0,
                                            sensitivity=1.0)

    Production considerations:
        - The mechanism is ε-DP regardless of the number of candidates,
          making it safe to use with large discrete sets.
        - Log-space computation is used internally to prevent floating-point
          overflow when ε · score is large.
        - Sensitivity must reflect the *global* worst-case, not the typical
          case.  Underestimating sensitivity breaks the DP guarantee.
        - Composing k exponential mechanism calls under basic composition costs
          ε · k total budget.
        - For repeated selection (e.g. choosing k items from the same set),
          use the Report Noisy Max or Sparse Vector Technique to reduce
          composition cost.

    Args:
        candidates: Ordered list of outputs to select from.  Any Python type.
        utility_scores: Real-valued array of length len(candidates).  Higher
            values indicate more preferred outputs.
        epsilon: Privacy budget (ε > 0).  Larger values favour high-utility
            candidates more strongly.
        sensitivity: Global L1 sensitivity of the utility function — the
            maximum change in any candidate's score when one record changes.
        random_state: Integer seed for reproducibility.

    Returns:
        The selected element from ``candidates``.

    Raises:
        ValueError: If ``epsilon`` or ``sensitivity`` are not positive, if
            ``candidates`` is empty, or if ``candidates`` and
            ``utility_scores`` have different lengths.

    Examples:
        >>> import numpy as np
        >>> from dp.mechanisms import exponential_mechanism
        >>> candidates = ["low", "medium", "high"]
        >>> scores = np.array([1.0, 5.0, 3.0])
        >>> exponential_mechanism(candidates, scores, epsilon=1.0, random_state=0)
        'medium'
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    scores = np.asarray(utility_scores, dtype=float)
    if len(candidates) != len(scores):
        raise ValueError(
            f"candidates and utility_scores must have the same length "
            f"(got {len(candidates)} and {len(scores)})"
        )
    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")

    # Subtract max before exponentiation for numerical stability.
    log_weights = epsilon * scores / (2.0 * sensitivity)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    probabilities = weights / weights.sum()

    rng = get_rng(random_state)
    index = rng.choice(len(candidates), p=probabilities)
    return candidates[index]
