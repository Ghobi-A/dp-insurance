"""Tests for the prospective real-data LiRA validation design and statistics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp import lira_study as ls
from dp.attacks import lira_offline_attack

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "lira_real_data_study", ROOT / "research" / "lira_real_data_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _runner()


def test_fixed_encoding_is_rowwise_and_matches_protocol_constants():
    df = pd.read_csv(ROOT / "data" / "insurance.csv")
    X, y = ls.encode_fixed(df)
    assert X.shape == (ls.N_RECORDS, 9) and X[:, :3].min() >= -1 and X[:, :3].max() <= 1
    # Row-wise: encoding a subset equals the subset of the encoding.
    subset = df.iloc[100:140]
    np.testing.assert_array_equal(ls.encode_fixed(subset)[0], X[100:140])
    assert float(df.charges.median()) == pytest.approx(ls.HIGH_COST_THRESHOLD)
    assert y.sum() == ls.N_RECORDS // 2


def test_duplicate_groups_flag_identical_records():
    X = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    assert ls.duplicate_groups(X, np.array([1, 1, 0])).tolist() == [0, 0, 1]
    assert ls.duplicate_groups(X, np.array([1, 0, 0])).tolist() == [0, 1, 2]


def test_membership_design_is_balanced_reproducible_and_complementary():
    m = ls.membership_matrix(ls.PILOT_SEED)
    T = ls.N_TARGETS
    assert m.shape == (T + ls.N_SHADOWS, ls.N_RECORDS)
    assert (m.sum(axis=1) == ls.TRAIN_SIZE).all()
    np.testing.assert_array_equal(m, ls.membership_matrix(ls.PILOT_SEED))
    assert not np.array_equal(m, ls.membership_matrix(ls.STUDY_SEED))
    shadows = m[T:]
    np.testing.assert_array_equal(shadows[0::2], ~shadows[1::2])
    for k in ls.SHADOW_COUNTS:
        assert ((~shadows[:k]).sum(axis=0) == k // 2).all()
    # Targets are distinct training sets and not copies of any shadow.
    assert len({row.tobytes() for row in m}) == len(m)


def test_out_matrix_contains_exactly_the_out_shadow_losses():
    m = ls.membership_matrix(3, n=40, n_targets=2, n_shadows=8)
    shadows = m[2:]
    losses = np.arange(8 * 40, dtype=float).reshape(8, 40)
    out = ls.out_matrix(losses, shadows, 8)
    assert out.shape == (40, 4)
    for i in range(40):
        assert sorted(out[i]) == sorted(losses[~shadows[:, i], i])
    with pytest.raises(ValueError, match="balanced"):
        ls.out_matrix(losses, shadows, 3)


def test_score_target_reuses_the_library_attack():
    rng = np.random.default_rng(0)
    m = ls.membership_matrix(5, n=60, n_targets=1, n_shadows=128)
    losses = rng.exponential(0.5, (128, 60))
    target = rng.exponential(0.5, 60)
    result = ls.score_target(target, m[0].astype(int), losses, m[1:], ls.PRIMARY)
    direct = lira_offline_attack(
        target, m[0].astype(int), ls.out_matrix(losses, m[1:], 128),
        fprs=ls.FPRS, statistic="logit_confidence", variance="per_example",
    )
    np.testing.assert_array_equal(result.scores, direct.scores)
    yeom = ls.score_target(target, m[0].astype(int), losses, m[1:], {"attack": "loss_threshold"})
    assert yeom.method == "loss-threshold"


def test_configuration_grid_contains_primary_and_baseline_once():
    configs = ls.configurations()
    assert len(configs) == 12
    assert configs.count(ls.PRIMARY) == 1 and configs.count(ls.BASELINE) == 1
    assert ls.config_name(ls.BASELINE) == "raw_loss|per_example|K=22"


def test_signflip_and_interval_and_holm():
    assert ls.signflip_pvalue([1.0] * 5) == pytest.approx(2 / 32)
    assert ls.signflip_pvalue([1.0, -1.0, 1.0, -1.0]) == 1.0
    ci = ls.mean_t_interval([1.0, 2.0, 3.0])
    assert ci["mean"] == 2.0 and ci["ci_low"] < 2.0 < ci["ci_high"]
    assert np.isnan(ls.mean_t_interval([1.0, 1.0])["ci_low"])
    assert ls.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_tpr_interval_is_clopper_pearson():
    lo, hi = ls.tpr_interval(0.0, 100)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 100))
    lo, hi = ls.tpr_interval(0.1, 200)
    assert lo < 0.1 < hi


def test_stouffer_detection_is_calibrated_under_random_membership():
    """Null control: scores unrelated to membership must not be detected."""
    rng = np.random.default_rng(1)
    rejections = 0
    for _ in range(400):
        aucs = []
        for _ in range(8):
            scores = rng.standard_normal(120)
            members = rng.permutation(120) < 60
            ranks = scores.argsort().argsort() + 1
            aucs.append((ranks[members].sum() - 60 * 61 / 2) / 3600)
        rejections += ls.stouffer_detection(aucs, 60, 60)["p_one_sided"] < 0.05
    assert 0.02 <= rejections / 400 <= 0.09


def test_detection_power_null_and_alternative():
    assert ls.detection_power(0.5, sims=400, seed=2) <= 0.09
    assert ls.detection_power(0.52, sims=400, seed=3) >= 0.95


def test_protocol_hash_tracks_constants(monkeypatch):
    before = ls.protocol_hash()
    assert before == ls.protocol_hash() and len(before) == 64
    monkeypatch.setattr(ls, "N_TARGETS", ls.N_TARGETS + 1)
    assert ls.protocol_hash() != before


@pytest.mark.parametrize("condition", ls.CONDITIONS)
def test_train_and_measure_runs_each_condition(condition):
    pytest.importorskip("torch")
    pytest.importorskip("opacus")
    rng = np.random.default_rng(4)
    X = rng.standard_normal((128, 9)).astype(np.float32)
    y = (rng.random(128) < 0.5).astype(np.float32)
    members = np.arange(128) < 64
    margins, meta = runner.train_and_measure(X, y, members, condition, seed=7, epochs=2)
    assert margins.shape == (128,) and np.isfinite(margins).all()
    assert meta["steps"] == 2 and meta["train_size"] == 64
    if condition in ls.REQUESTED_EPSILON:
        assert meta["sample_rate"] == 1.0
        assert meta["achieved_epsilon"] <= ls.REQUESTED_EPSILON[condition] * 1.01
    elif condition == "clip_only":
        assert meta["noise_multiplier"] == 0.0 and meta["achieved_epsilon"] is None
    else:
        assert meta["noise_multiplier"] is None


def test_confirmatory_run_rejects_overrides_and_pilot_measurements(tmp_path):
    with pytest.raises(SystemExit, match="pilot-only"):
        runner.main(["train", "--output-dir", str(tmp_path), "--epochs", "1"])
    members = ls.membership_matrix(ls.PILOT_SEED)
    for index in range(ls.N_TARGETS + ls.N_SHADOWS):
        path = tmp_path / "models" / "nonprivate" / f"model_{index:03d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"pilot": True, "protocol_hash": "x", "design_seed": ls.PILOT_SEED}
        np.savez(path, margins=np.zeros(ls.N_RECORDS), members=members[index], meta=json.dumps(meta))
    with pytest.raises(SystemExit, match="pilot measurements"):
        runner.load_condition(tmp_path, "nonprivate", allow_pilot=False)


def test_analysis_on_synthetic_margins_recovers_planted_signal():
    """Analysis bookkeeping: planted member advantage is detected, control is not."""
    rng = np.random.default_rng(5)
    members = ls.membership_matrix(ls.PILOT_SEED)
    difficulty = rng.normal(2.0, 1.0, ls.N_RECORDS)
    margins = difficulty + 0.5 * rng.standard_normal(members.shape) + 0.4 * members
    X, y = ls.encode_fixed(pd.read_csv(ROOT / "data" / "insurance.csv"))
    per_target, diagnostics, _, _ = runner.analyse_condition(
        margins, members, ls.duplicate_groups(X, y)
    )
    summary = runner.summarise(per_target).set_index("config")
    primary = summary.loc[ls.config_name(ls.PRIMARY)]
    assert primary.mean_auc > 0.6 and primary.stouffer_p < 1e-6
    assert abs(primary.mean_control_auc - 0.5) < 0.02 and primary.control_stouffer_p > 0.001
    assert primary.members_per_target == ls.TRAIN_SIZE
    assert diagnostics["duplicate_records_in_groups_larger_than_one"] >= 2
    contrast = runner.paired_contrast(per_target, "auc")
    assert contrast["n"] == ls.N_TARGETS
