import numpy as np
import pytest

from dp.attacks import lira_offline_attack, loss_threshold_attack


def test_loss_threshold_detects_separation():
    rng = np.random.default_rng(0)
    # Members have systematically lower loss than non-members.
    member = rng.normal(0.2, 0.1, size=500).clip(min=0)
    nonmember = rng.normal(1.0, 0.3, size=500).clip(min=0)
    result = loss_threshold_attack(member, nonmember)
    assert result.auc > 0.9
    assert result.tpr_at_fpr[0.1] > 0.5


def test_loss_threshold_no_leakage_is_chance():
    rng = np.random.default_rng(1)
    member = rng.normal(0.5, 0.2, size=1000)
    nonmember = rng.normal(0.5, 0.2, size=1000)
    result = loss_threshold_attack(member, nonmember)
    assert abs(result.auc - 0.5) < 0.05


def test_loss_threshold_reports_all_requested_fprs():
    result = loss_threshold_attack(
        np.array([0.1, 0.2]), np.array([0.9, 1.0]), fprs=(0.01, 0.05, 0.5)
    )
    assert set(result.tpr_at_fpr) == {0.01, 0.05, 0.5}


def test_loss_threshold_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        loss_threshold_attack(np.array([]), np.array([1.0]))


def test_lira_offline_beats_baseline_on_calibratable_data():
    rng = np.random.default_rng(2)
    n, k = 400, 32
    # Per-example difficulty: baseline loss varies a lot across examples, which
    # confounds a raw threshold but LiRA calibrates away.
    difficulty = rng.normal(1.0, 0.5, size=n).clip(min=0.1)
    membership = rng.integers(0, 2, size=n)
    # Shadow "out" losses centre on each example's difficulty.
    shadow_out = rng.normal(difficulty[:, None], 0.15, size=(n, k)).clip(min=0)
    # Target losses: members shifted down by a fixed margin from difficulty.
    target = difficulty + rng.normal(0, 0.15, size=n) - membership * 0.4
    target = target.clip(min=0)

    lira = lira_offline_attack(target, membership, shadow_out)
    baseline = loss_threshold_attack(target[membership == 1], target[membership == 0])
    assert lira.auc > 0.5
    # Calibration should help versus the raw-threshold baseline on this data.
    assert lira.auc >= baseline.auc - 0.02


def test_lira_offline_shape_validation():
    with pytest.raises(ValueError, match="align"):
        lira_offline_attack(
            np.zeros(5), np.zeros(4), np.zeros((5, 8))
        )
    with pytest.raises(ValueError, match="k >= 2"):
        lira_offline_attack(np.zeros(5), np.zeros(5), np.zeros((5, 1)))
