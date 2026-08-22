"""Tests for Study 2 Phase 0: the frozen slice and the continuation gate.

The gate logic is where a mistake would be worst -- a strict, once-only,
terminal decision -- so it is tested exhaustively against the four ways it can
be reached, including the case the preregistration singles out: measurable
leakage that still fails continuation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))

import study2_acs_slice as acs  # noqa: E402
import study2_phase0_natural_leakage as phase0  # noqa: E402

# --------------------------------------------------------------------------- #
# Frozen slice
# --------------------------------------------------------------------------- #


def test_frozen_constants_match_the_preregistration():
    assert (acs.TASK, acs.SURVEY_YEAR, acs.HORIZON, acs.STATE) == (
        "ACSIncome",
        2018,
        "1-Year",
        "CA",
    )
    assert acs.SENSITIVE_ATTRIBUTE == "SEX"
    assert (acs.SAMPLE_ROWS, acs.SAMPLING_SEED) == (50_000, 20260822)
    assert phase0.TARGET_SEEDS == (42, 43, 44)
    assert phase0.HIDDEN_UNITS == (128, 128, 64)
    assert (phase0.LEARNING_RATE, phase0.BATCH_SIZE, phase0.EPOCHS) == (1e-3, 512, 60)
    assert (phase0.NUM_SHADOWS, phase0.PERMUTATION_REPS, phase0.ALPHA) == (64, 1000, 0.05)
    assert (phase0.GATE_MEAN_AUC, phase0.GATE_PER_SEED_AUC) == (0.60, 0.50)
    assert phase0.PRIMARY_ATTACK == "online"


def _raw_pums(rows: int = 60_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "AGEP": rng.integers(10, 90, rows),
            "COW": rng.integers(1, 9, rows),
            "SCHL": rng.integers(1, 25, rows),
            "MAR": rng.integers(1, 6, rows),
            "OCCP": rng.integers(10, 9800, rows),
            "POBP": rng.integers(1, 60, rows),
            "RELP": rng.integers(0, 18, rows),
            "WKHP": rng.integers(0, 80, rows),
            "SEX": rng.integers(1, 3, rows),
            "RAC1P": rng.integers(1, 10, rows),
            "PINCP": rng.integers(0, 200_000, rows),
            "PWGTP": rng.integers(1, 100, rows),
        }
    )
    return frame


def test_eligibility_filter_matches_the_acsincome_definition():
    raw = _raw_pums()
    kept = acs.eligible_rows(raw)
    assert (kept["AGEP"] > 16).all()
    assert (kept["PINCP"] > 100).all()
    assert (kept["WKHP"] > 0).all()
    assert (kept["PWGTP"] >= 1).all()
    # Nothing eligible was dropped.
    dropped = raw.drop(index=kept.index)
    assert not (
        (dropped["AGEP"] > 16)
        & (dropped["PINCP"] > 100)
        & (dropped["WKHP"] > 0)
        & (dropped["PWGTP"] >= 1)
    ).any()


def test_build_slice_is_deterministic_and_exactly_fifty_thousand_rows():
    raw = _raw_pums()
    first = acs.build_slice(raw)
    second = acs.build_slice(raw)
    assert len(first) == acs.SAMPLE_ROWS
    assert list(first.columns) == list(acs.FEATURES) + ["y", "group"]
    pd.testing.assert_frame_equal(first, second)
    assert acs.slice_fingerprint(first) == acs.slice_fingerprint(second)


def test_build_slice_refuses_a_pool_smaller_than_the_frozen_sample():
    with pytest.raises(ValueError, match="eligible rows"):
        acs.build_slice(_raw_pums(rows=1_000))


def test_verify_slice_rejects_a_changed_slice():
    frame = acs.build_slice(_raw_pums())
    metadata = acs.verify_slice(frame)
    assert metadata["rows"] == acs.SAMPLE_ROWS

    tampered = frame.copy()
    tampered.loc[0, "AGEP"] = tampered.loc[0, "AGEP"] + 1
    with pytest.raises(ValueError, match="fingerprint"):
        acs.verify_slice(tampered, metadata)

    with pytest.raises(ValueError, match="rows"):
        acs.verify_slice(frame.iloc[:-1], metadata)


def test_verify_slice_rejects_reordered_rows():
    frame = acs.build_slice(_raw_pums())
    metadata = acs.verify_slice(frame)
    reordered = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="fingerprint"):
        acs.verify_slice(reordered, metadata)


def test_smoke_slice_is_stamped_and_never_mistaken_for_the_frozen_slice(tmp_path):
    output = tmp_path / "slice.csv"
    frame, metadata = acs.materialise(output, tmp_path / "cache", smoke=True)
    assert metadata["smoke"] is True
    assert len(frame) != acs.SAMPLE_ROWS
    assert acs.main(["verify", "--slice", str(output)]) == 0


# --------------------------------------------------------------------------- #
# Design construction
# --------------------------------------------------------------------------- #


def test_split_membership_is_seed_dependent_and_exact():
    train_a, membership_a = phase0.split_membership(1_000, 400, seed=42)
    train_b, _ = phase0.split_membership(1_000, 400, seed=43)
    assert len(train_a) == 400
    assert membership_a.sum() == 400
    assert membership_a[train_a].all()
    assert not np.array_equal(train_a, train_b)
    # Reproducible from the seed alone.
    assert np.array_equal(train_a, phase0.split_membership(1_000, 400, seed=42)[0])


def test_equalised_cohort_balances_every_group_by_membership_cell():
    rng = np.random.default_rng(0)
    groups = np.where(rng.random(2_000) < 0.4, "female", "male")
    membership = (rng.random(2_000) < 0.3).astype(int)
    idx, counts = phase0.equalised_cohort(groups, membership, seed=42, per_cell_cap=50)
    assert set(counts.values()) == {50}
    assert len(idx) == 200
    for group in ("male", "female"):
        for member in (0, 1):
            cell = (groups[idx] == group) & (membership[idx] == member)
            assert cell.sum() == 50


def test_standardise_leaves_zero_variance_columns_finite():
    features = np.column_stack([np.arange(10.0), np.ones(10)])
    scaled = phase0.standardise(features)
    assert np.isfinite(scaled).all()
    assert scaled[:, 1].std() == 0


# --------------------------------------------------------------------------- #
# The continuation gate
# --------------------------------------------------------------------------- #


def _seed_result(seed: int, auc: float, p_value: float) -> dict:
    return {
        "seed": seed,
        "smoke": False,
        "epochs": phase0.EPOCHS,
        "num_shadows": phase0.NUM_SHADOWS,
        "num_attack_examples": 10_000,
        "memorisation": {
            "train_auc": 0.9,
            "test_auc": 0.8,
            "auc_gap": 0.1,
            "loss_gap": 0.1,
        },
        "online": {
            "auc": auc,
            f"tpr_at_fpr_{phase0.HEADLINE_FPR}": 0.02,
            "permutation_p_value": p_value,
        },
        "offline": {
            "auc": auc - 0.02,
            f"tpr_at_fpr_{phase0.HEADLINE_FPR}": 0.015,
            "permutation_p_value": p_value,
        },
    }


def test_gate_passes_only_when_all_three_criteria_hold():
    gate = phase0.evaluate_gate(
        [_seed_result(42, 0.64, 0.001), _seed_result(43, 0.61, 0.001), _seed_result(44, 0.62, 0.001)]
    )
    assert gate["gate"] == phase0.PASS
    assert all(item["passed"] for item in gate["criteria"])
    assert gate["descriptive_note"] is None


def test_measurable_leakage_below_the_margin_is_still_a_fail():
    # The case the preregistration singles out: AUC 0.58, 3/3 significant.
    gate = phase0.evaluate_gate(
        [_seed_result(42, 0.58, 0.001), _seed_result(43, 0.58, 0.001), _seed_result(44, 0.58, 0.001)]
    )
    assert gate["gate"] == phase0.FAIL
    assert gate["seeds_significant"] == 3
    # It is nevertheless reported descriptively rather than as "no leakage".
    assert "below the predeclared continuation margin" in gate["descriptive_note"]
    assert "canary" in gate["continuation"]
    assert "epsilon ladder" in gate["continuation"]


def test_high_mean_auc_fails_if_one_seed_is_not_significant():
    gate = phase0.evaluate_gate(
        [_seed_result(42, 0.70, 0.001), _seed_result(43, 0.70, 0.001), _seed_result(44, 0.70, 0.20)]
    )
    assert gate["gate"] == phase0.FAIL
    failed = [item["id"] for item in gate["criteria"] if not item["passed"]]
    assert failed == ["permutation"]


def test_high_mean_auc_fails_if_one_seed_is_at_or_below_chance():
    gate = phase0.evaluate_gate(
        [_seed_result(42, 0.85, 0.001), _seed_result(43, 0.85, 0.001), _seed_result(44, 0.50, 0.001)]
    )
    assert gate["gate"] == phase0.FAIL
    assert "per_seed_auc" in [item["id"] for item in gate["criteria"] if not item["passed"]]


def test_gate_is_incomplete_rather_than_decided_on_a_partial_seed_set():
    gate = phase0.evaluate_gate([_seed_result(42, 0.9, 0.001), _seed_result(43, 0.9, 0.001)])
    assert gate["verdict"] == "INCOMPLETE"
    assert gate["missing_seeds"] == [44]
    assert "No continuation decision" in gate["continuation"]


def test_failure_continuation_text_closes_the_branch_without_launching_the_ladder():
    gate = phase0.evaluate_gate(
        [_seed_result(seed, 0.51, 0.9) for seed in phase0.TARGET_SEEDS]
    )
    assert gate["gate"] == phase0.FAIL
    text = gate["continuation"].lower()
    assert "do not launch the dp epsilon ladder" in text
    assert "do not tune the target" in text
    assert "not implemented or run until its own estimand" in text


# --------------------------------------------------------------------------- #
# Reporting and provenance
# --------------------------------------------------------------------------- #


def _payload(gate_results) -> dict:
    return {
        "study": phase0.STUDY,
        "generated_at": "2026-08-22T00:00:00+00:00",
        "config": phase0.frozen_config(),
        "slice": {
            "state": "CA",
            "survey_year": 2018,
            "horizon": "1-Year",
            "rows": 50_000,
            "fingerprint": "0" * 64,
        },
        "preregistered": True,
        "seeds": gate_results,
        "gate": phase0.evaluate_gate(gate_results),
    }


def test_markdown_report_states_the_gate_and_every_criterion():
    payload = _payload([_seed_result(seed, 0.58, 0.001) for seed in phase0.TARGET_SEEDS])
    markdown = phase0.render_markdown(payload)
    assert "## Gate: FAIL" in markdown
    assert "mean online-LiRA ROC-AUC >= 0.6" in markdown
    assert "Descriptive reading" in markdown


def test_smoke_and_deviating_runs_are_never_presented_as_phase_0_results():
    frozen = phase0.frozen_config()
    assert phase0.is_preregistered(frozen)
    assert not phase0.is_preregistered(phase0.frozen_config(smoke=True))
    assert not phase0.is_preregistered(phase0.frozen_config(epochs=5))

    payload = _payload([_seed_result(seed, 0.9, 0.001) for seed in phase0.TARGET_SEEDS])
    payload["preregistered"] = False
    payload["config"] = phase0.frozen_config(epochs=5, smoke=True)
    assert "Not a Phase 0 result" in phase0.render_markdown(payload)


def test_write_outputs_emits_json_csv_and_markdown(tmp_path):
    payload = _payload([_seed_result(seed, 0.62, 0.001) for seed in phase0.TARGET_SEEDS])
    for result in payload["seeds"]:
        result["per_example"] = [
            {
                "study": phase0.STUDY,
                "seed": result["seed"],
                "cohort_position": 0,
                "example_index": 3,
                "group": "male",
                "membership": 1,
                "target_loss": 0.1,
                "online_score": 1.2,
                "offline_score": 0.8,
                "in_observations": 32,
                "out_observations": 32,
            }
        ]
    phase0.write_outputs(payload, tmp_path)
    stored = json.loads((tmp_path / "phase0_gate.json").read_text())
    assert stored["gate"]["gate"] == phase0.PASS
    assert (tmp_path / "phase0_gate.csv").read_text().splitlines()[0].startswith("study,seed")
    assert "## Gate: PASS" in (tmp_path / "phase0_gate.md").read_text()
    # Per-example records are the raw design; losing them is unrecoverable.
    assert (tmp_path / "per_example.csv").exists()


def test_aggregate_refuses_to_mix_slices(tmp_path):
    inputs = tmp_path / "in"
    inputs.mkdir()
    for offset, seed in enumerate(phase0.TARGET_SEEDS):
        result = _seed_result(seed, 0.7, 0.001)
        result["config"] = phase0.frozen_config()
        result["preregistered"] = True
        result["slice"] = {"fingerprint": f"{offset}" * 64, "state": "CA"}
        (inputs / f"seed_{seed}.json").write_text(json.dumps(result))
    args = ["aggregate", "--input-dir", str(inputs), "--output-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="different data slices"):
        phase0.main(args)


@pytest.mark.slow
def test_seed_command_runs_end_to_end_in_smoke_mode(tmp_path):
    pytest.importorskip("torch")
    slice_path = tmp_path / "slice.csv"
    acs.materialise(slice_path, tmp_path / "cache", smoke=True)
    out = tmp_path / "seeds"
    assert (
        phase0.main(
            [
                "seed",
                "--slice",
                str(slice_path),
                "--seed",
                "42",
                "--epochs",
                "2",
                "--shadows",
                "4",
                "--permutation-reps",
                "20",
                "--output-dir",
                str(out),
                "--smoke",
            ]
        )
        == 0
    )
    stored = json.loads((out / "seed_42.json").read_text())
    assert stored["attack_status"] == "completed"
    assert stored["preregistered"] is False
    assert stored["online"]["auc"] > 0
    assert len(stored["per_example"]) == stored["num_attack_examples"]
