import numpy as np
import pytest

from dp.fairness import demographic_parity_difference, equalized_odds_difference


def test_demographic_parity_known_gap():
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    group = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    # P(ŷ=1 | a) = 0.5, P(ŷ=1 | b) = 0.25
    assert demographic_parity_difference(y_pred, group) == pytest.approx(0.25)


def test_demographic_parity_zero_when_equal():
    y_pred = np.array([1, 0, 1, 0])
    group = np.array(["a", "a", "b", "b"])
    assert demographic_parity_difference(y_pred, group) == 0.0


def test_demographic_parity_multiple_groups():
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    group = np.array(["a", "a", "b", "b", "c", "c"])
    # rates: a=1.0, b=0.5, c=0.0 → max-min = 1.0
    assert demographic_parity_difference(y_pred, group) == pytest.approx(1.0)


def test_demographic_parity_validates_inputs():
    with pytest.raises(ValueError, match="empty"):
        demographic_parity_difference(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="length"):
        demographic_parity_difference(np.array([1, 0]), np.array(["a"]))


def test_equalized_odds_known_gap():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    group = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    # TPR: a=1.0, b=0.5 → gap 0.5. FPR: a=0.5, b=0.0 → gap 0.5.
    assert equalized_odds_difference(y_true, y_pred, group) == pytest.approx(0.5)


def test_equalized_odds_zero_for_identical_behaviour():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    group = np.array(["a", "a", "b", "b"])
    assert equalized_odds_difference(y_true, y_pred, group) == 0.0


def test_equalized_odds_skips_groups_without_label():
    # Group "b" has no true positives; TPR gap cannot be computed for it,
    # but the FPR gap still can.
    y_true = np.array([1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    group = np.array(["a", "a", "b", "b"])
    # FPR: a=0.0, b=0.5 → 0.5
    assert equalized_odds_difference(y_true, y_pred, group) == pytest.approx(0.5)


def test_equalized_odds_validates_inputs():
    with pytest.raises(ValueError, match="empty"):
        equalized_odds_difference(np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError, match="length"):
        equalized_odds_difference(np.array([1]), np.array([1, 0]), np.array(["a", "b"]))
