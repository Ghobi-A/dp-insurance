import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from dp.pipeline import (
    apply_bounded_feature_noise,
    apply_feature_noise,
    build_preprocessor,
    clip_numeric,
    compute_clip_bounds,
    preprocess_split,
    split_dataset,
)


def _sample_df(size: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 65, size=size),
            "charges": rng.normal(10_000, 5_000, size=size),
            "sex": rng.choice(["male", "female"], size=size),
            "smoker": rng.choice(["yes", "no"], size=size),
        }
    )


def test_compute_clip_bounds_shape_and_order():
    df = _sample_df()
    bounds = compute_clip_bounds(df)
    assert list(bounds.columns) == ["lower", "upper"]
    assert set(bounds.index) == {"age", "charges"}
    assert (bounds["upper"] > bounds["lower"]).all()


def test_clip_numeric_respects_bounds():
    df = _sample_df()
    bounds = compute_clip_bounds(df, lower_q=0.1, upper_q=0.9)
    clipped = clip_numeric(df, bounds)
    for column in bounds.index:
        assert clipped[column].min() >= bounds.loc[column, "lower"]
        assert clipped[column].max() <= bounds.loc[column, "upper"]
    # Categorical columns untouched.
    pdt.assert_series_equal(clipped["sex"], df["sex"])


@pytest.mark.parametrize("mechanism", ["laplace", "gaussian"])
def test_apply_bounded_feature_noise_basic(mechanism):
    df = _sample_df()
    bounds = compute_clip_bounds(df)
    noisy = apply_bounded_feature_noise(
        df, bounds, mechanism=mechanism, epsilon=1.0, random_state=0
    )
    assert list(noisy.columns) == list(df.columns)
    pdt.assert_series_equal(noisy["sex"], df["sex"])
    assert not noisy["age"].equals(df["age"])
    # Reproducible.
    again = apply_bounded_feature_noise(
        df, bounds, mechanism=mechanism, epsilon=1.0, random_state=0
    )
    pdt.assert_frame_equal(noisy, again)


def test_apply_bounded_feature_noise_scale_calibration():
    # In unit space each column receives Laplace(0, d/ε) noise; after mapping
    # back, column noise std should be (d/ε)·√2·width.
    size = 20000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.uniform(0, 10, size), "b": rng.uniform(0, 100, size)})
    bounds = pd.DataFrame({"lower": {"a": 0.0, "b": 0.0}, "upper": {"a": 10.0, "b": 100.0}})
    epsilon, d = 2.0, 2
    noisy = apply_bounded_feature_noise(df, bounds, epsilon=epsilon, random_state=1)
    noise = noisy["b"] - df["b"]
    expected_std = (d / epsilon) * np.sqrt(2) * 100.0
    assert abs(noise.std() - expected_std) / expected_std < 0.05


def test_apply_bounded_feature_noise_rejects_degenerate_bounds():
    df = _sample_df()
    bounds = pd.DataFrame({"lower": {"age": 30.0}, "upper": {"age": 30.0}})
    with pytest.raises(ValueError, match="Degenerate"):
        apply_bounded_feature_noise(df, bounds)


def test_apply_bounded_feature_noise_rejects_unknown_mechanism():
    df = _sample_df()
    bounds = compute_clip_bounds(df)
    with pytest.raises(ValueError, match="mechanisms"):
        apply_bounded_feature_noise(df, bounds, mechanism="bernoulli")


def test_apply_feature_noise_preserves_columns():
    df = _sample_df()
    noisy = apply_feature_noise(df, mechanism="laplace", epsilon=1.0, random_state=0)
    assert list(noisy.columns) == list(df.columns)
    pdt.assert_series_equal(noisy["smoker"], df["smoker"])


def test_preprocess_split_fits_on_train_only():
    df = _sample_df(size=200)
    split = split_dataset(df, target="smoker")
    X_train, X_test, preprocessor = preprocess_split(split)
    assert X_train.shape[0] == len(split.X_train)
    assert X_test.shape[0] == len(split.X_test)
    # Scaler statistics must come from the training partition alone.
    scaler = preprocessor.named_transformers_["num"]
    expected = split.X_train.select_dtypes(include="number").mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-10)


def test_build_preprocessor_handles_unseen_categories():
    df = _sample_df()
    preprocessor = build_preprocessor(df)
    preprocessor.fit(df)
    unseen = df.head(3).copy()
    unseen["sex"] = "other"
    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 3
