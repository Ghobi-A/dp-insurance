import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from scipy.stats import norm

from dp.mechanisms import (
    add_gaussian_noise,
    add_laplace_noise,
    apply_randomized_response,
    calibrate_analytic_gaussian_sigma,
    exponential_mechanism,
    randomized_response,
)


def _check_noise(func):
    data = pd.DataFrame(np.zeros((10, 5)))
    out1 = func(data, random_state=0)
    out2 = func(data, random_state=0)
    assert out1.shape == data.shape
    pdt.assert_frame_equal(out1, out2)
    out3 = func(data, random_state=1)
    assert not out1.equals(out3)


def _check_noise_mixed(func, dtype=float, check_dtype=False):
    data = pd.DataFrame({"num": np.zeros(10, dtype=dtype), "cat": ["x"] * 10})
    out1 = func(data, random_state=0)
    # Non-numeric columns should remain unchanged
    pdt.assert_series_equal(out1["cat"], data["cat"])
    out2 = func(data, random_state=0)
    pdt.assert_frame_equal(out1, out2)
    out3 = func(data, random_state=1)
    assert not out1["num"].equals(out3["num"])
    if check_dtype:
        assert out1["num"].dtype == data["num"].dtype


def test_laplace_noise():
    _check_noise(add_laplace_noise)


def test_gaussian_noise():
    _check_noise(add_gaussian_noise)


def test_laplace_noise_mixed_types():
    _check_noise_mixed(add_laplace_noise)


def test_gaussian_noise_mixed_types():
    _check_noise_mixed(add_gaussian_noise)


def test_laplace_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        add_laplace_noise(pd.DataFrame({"a": [1.0]}), epsilon=0.0)


def test_gaussian_invalid_epsilon_and_delta():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="epsilon"):
        add_gaussian_noise(df, epsilon=-1.0)
    with pytest.raises(ValueError, match="delta"):
        add_gaussian_noise(df, epsilon=1.0, delta=0.0)
    with pytest.raises(ValueError, match="delta"):
        add_gaussian_noise(df, epsilon=1.0, delta=1.0)


def test_laplace_noise_scale():
    # Empirical std of Laplace(0, b) noise should be close to b * sqrt(2).
    data = pd.DataFrame(np.zeros((20000, 1)))
    epsilon, sensitivity = 1.0, 2.0
    noisy = add_laplace_noise(data, epsilon=epsilon, sensitivity=sensitivity, random_state=0)
    expected_std = (sensitivity / epsilon) * np.sqrt(2)
    assert abs(noisy[0].mean()) < 0.1
    assert abs(noisy[0].std() - expected_std) / expected_std < 0.05


def _analytic_gaussian_delta(sigma, epsilon, sensitivity):
    # (ε, δ) achieved by N(0, σ²): Balle & Wang (2018), Theorem 8.
    a = sensitivity / (2 * sigma)
    b = epsilon * sigma / sensitivity
    return norm.cdf(a - b) - np.exp(epsilon) * norm.cdf(-a - b)


@pytest.mark.parametrize("epsilon", [0.1, 1.0, 3.0, 10.0])
def test_analytic_gaussian_sigma_is_tight(epsilon):
    delta, sensitivity = 1e-5, 1.0
    sigma = calibrate_analytic_gaussian_sigma(epsilon, delta, sensitivity)
    # Returned sigma satisfies the (ε, δ) condition...
    assert _analytic_gaussian_delta(sigma, epsilon, sensitivity) <= delta * (1 + 1e-6)
    # ...and is minimal: slightly less noise would violate it.
    assert _analytic_gaussian_delta(sigma * 0.99, epsilon, sensitivity) > delta


def test_analytic_gaussian_never_worse_than_classic():
    # For ε ≤ 1 (where the classic bound is valid), analytic σ must be ≤ classic σ.
    for epsilon in [0.1, 0.5, 1.0]:
        delta = 1e-5
        classic = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        analytic = calibrate_analytic_gaussian_sigma(epsilon, delta, 1.0)
        assert analytic <= classic


def test_analytic_gaussian_sigma_invalid_inputs():
    with pytest.raises(ValueError, match="epsilon"):
        calibrate_analytic_gaussian_sigma(0.0, 1e-5)
    with pytest.raises(ValueError, match="delta"):
        calibrate_analytic_gaussian_sigma(1.0, 0.0)
    with pytest.raises(ValueError, match="sensitivity"):
        calibrate_analytic_gaussian_sigma(1.0, 1e-5, sensitivity=0.0)


def test_randomized_response_binary():
    series = pd.Series(["yes", "no", "yes", "no"])
    out1 = randomized_response(series, epsilon=1.0, random_state=0)
    out2 = randomized_response(series, epsilon=1.0, random_state=0)
    assert set(out1.unique()) <= {"yes", "no"}
    pdt.assert_series_equal(out1, out2)


def test_apply_randomized_response_columns():
    df = pd.DataFrame({"flag": ["yes", "no", "yes"], "keep": [1, 2, 3]})
    out = apply_randomized_response(df, ["flag"], epsilon=0.5, random_state=0)
    assert set(out["flag"].unique()) <= {"yes", "no"}
    pdt.assert_series_equal(out["keep"], df["keep"])


def test_randomized_response_rejects_non_binary():
    with pytest.raises(ValueError, match="binary"):
        randomized_response(pd.Series(["a", "b", "c"]), epsilon=1.0)


def test_randomized_response_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        randomized_response(pd.Series(["yes", "no"]), epsilon=0.0)


def test_randomized_response_keep_probability():
    # With ε = 1, each value is kept with p = e/(e+1) ≈ 0.731.
    series = pd.Series(["yes"] * 20000)
    # Force binary domain by appending one "no" and dropping it from the check.
    series = pd.concat([series, pd.Series(["no"])], ignore_index=True)
    out = randomized_response(series, epsilon=1.0, random_state=0)
    kept = (out.iloc[:-1] == "yes").mean()
    expected = np.exp(1) / (np.exp(1) + 1)
    assert abs(kept - expected) < 0.02


def test_exponential_mechanism_returns_candidate():
    candidates = ["a", "b", "c"]
    scores = np.array([1.0, 3.0, 2.0])
    result = exponential_mechanism(candidates, scores, epsilon=1.0, random_state=0)
    assert result in candidates


def test_exponential_mechanism_reproducible():
    candidates = ["x", "y", "z"]
    scores = np.array([0.5, 2.0, 1.0])
    r1 = exponential_mechanism(candidates, scores, epsilon=2.0, random_state=42)
    r2 = exponential_mechanism(candidates, scores, epsilon=2.0, random_state=42)
    assert r1 == r2


def test_exponential_mechanism_prefers_high_utility():
    # With large epsilon, the highest-scoring candidate should dominate.
    candidates = ["low", "high"]
    scores = np.array([0.0, 100.0])
    selections = [
        exponential_mechanism(candidates, scores, epsilon=10.0, random_state=i)
        for i in range(50)
    ]
    assert selections.count("high") > selections.count("low")


def test_exponential_mechanism_uniform_at_equal_scores():
    # Equal scores → each candidate equally likely.
    candidates = list(range(4))
    scores = np.ones(4)
    counts = {c: 0 for c in candidates}
    for i in range(400):
        result = exponential_mechanism(candidates, scores, epsilon=1.0, random_state=i)
        counts[result] += 1
    # Each bucket should receive roughly 100 ± a generous margin.
    for c in candidates:
        assert 50 < counts[c] < 200, f"Candidate {c} count {counts[c]} looks skewed"


def test_exponential_mechanism_invalid_epsilon():
    import pytest
    with pytest.raises(ValueError, match="epsilon"):
        exponential_mechanism(["a"], np.array([1.0]), epsilon=0.0)


def test_exponential_mechanism_invalid_sensitivity():
    import pytest
    with pytest.raises(ValueError, match="sensitivity"):
        exponential_mechanism(["a"], np.array([1.0]), epsilon=1.0, sensitivity=-1.0)


def test_exponential_mechanism_length_mismatch():
    import pytest
    with pytest.raises(ValueError, match="same length"):
        exponential_mechanism(["a", "b"], np.array([1.0]), epsilon=1.0)
