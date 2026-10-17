import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from dp.acs_publication import prepare_frame, sample_disjoint_households
from dp.auc_uncertainty import auc_influence, paired_bounds
from dp.publication_diagnostics import (
    cluster_bootstrap,
    confirmation_summary,
    plan_confirmation,
    primary_values,
)
from dp.publication_study import run, select


def test_auc_influences_match_pairwise_kernel_with_ties():
    y = np.array([0, 1] * 50)
    scores = np.random.default_rng(4).integers(0, 5, len(y))
    auc, influence = auc_influence(y, scores)
    positive, negative = scores[y == 1], scores[y == 0]
    kernel = (positive[:, None] > negative).astype(float) + 0.5 * (positive[:, None] == negative)
    assert auc == pytest.approx(kernel.mean())
    assert auc == pytest.approx(roc_auc_score(y, scores))
    np.testing.assert_allclose(influence[y == 1], (kernel.mean(axis=1) - auc) / 50)
    np.testing.assert_allclose(influence[y == 0], (kernel.mean(axis=0) - auc) / 50)
    assert influence.sum() == pytest.approx(0)


def test_household_bootstrap_preserves_paired_scores():
    rng = np.random.default_rng(81)
    y = np.tile([0, 1], 100)
    scores = y + rng.normal(size=len(y))
    groups = np.repeat(np.arange(100), 2)
    result = cluster_bootstrap(y, scores, scores, groups, replicates=99)
    assert result['replicates_valid'] == 99
    assert result['marginal_gap_upper'] == 0


def test_prospective_planning_and_complete_pair_requirement(tmp_path):
    rows = pd.DataFrame([
        dict(seed=seed, clip_norm=clip, total_interaction=(seed / 1000 if clip == 1 else 0),
             partition='test', environment='proxy_0', strength=0.9, requested_epsilon=2)
        for seed in range(10, 15) for clip in [1, 5]
    ])
    rows.to_csv(tmp_path / 'metrics.csv', index=False)
    base = tmp_path / 'base.json'
    base.write_text(json.dumps({'training': {}}))
    target = tmp_path / 'confirmation.json'
    result = plan_confirmation(tmp_path, base, target)
    assert result['n_seeds'] >= 20
    protocol = json.loads(target.read_text())
    assert min(protocol['training']['seeds']) == 100
    with pytest.raises(ValueError, match='incomplete'):
        primary_values(rows.iloc[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        primary_values(pd.concat([rows, rows.iloc[[0]]]))
    (tmp_path / 'protocol.json').write_text(target.read_text())
    with pytest.raises(ValueError, match='incomplete'):
        confirmation_summary(tmp_path)


def test_paired_covariance_and_cluster_duplication():
    rng = np.random.default_rng(8)
    y = np.tile([0, 1], 100)
    scores = rng.normal(size=200) + y
    groups = np.arange(200).astype(str)
    original = paired_bounds(y, scores, scores, groups)
    assert original['gap_se'] == 0
    assert original['gap_upper'] == 0
    duplicated = paired_bounds(np.repeat(y, 2), np.repeat(scores, 2),
                               np.repeat(scores, 2), np.repeat(groups, 2))
    assert duplicated['auc_se'] == pytest.approx(original['auc_se'])
    conservative = paired_bounds(y, scores, scores, groups, family=20)
    assert conservative['auc_lower'] < original['auc_lower']
    boundary = paired_bounds(y, y, y, groups)
    assert not boundary['bounds_supported']
    assert boundary['auc_lower'] == 0


def acs_frame():
    rng = np.random.default_rng(8)
    n = 300
    return pd.DataFrame({
        'SERIALNO': np.repeat(np.arange(100).astype(str), 3), 'SPORDER': np.tile([1, 2, 3], 100),
        'AGEP': rng.integers(17, 90, n), 'COW': 1, 'SCHL': 20, 'MAR': 1,
        'OCCP': rng.integers(0, 10000, n), 'POBP': 10, 'RELP': 0, 'WKHP': 40,
        'SEX': 1, 'RAC1P': 1, 'PINCP': rng.choice([20000, 80000], n), 'PWGTP': 1,
    })


def test_fixed_acs_encoding_and_household_separation():
    raw = acs_frame()
    X, y, groups, metadata = prepare_frame(raw)
    assert X.shape[1] == len(metadata['feature_names'])
    assert len(metadata['occupation_columns']) == 32
    original_first = X[0].copy()
    changed = raw.copy()
    changed.loc[1, 'AGEP'] = 500
    X_changed, _, _, _ = prepare_frame(changed)
    np.testing.assert_array_equal(original_first, X_changed[0])
    cohorts = sample_disjoint_households(X, y, groups, (90, 60, 60), 10)
    for i in range(3):
        for j in range(i):
            assert not set(cohorts[i][2]) & set(cohorts[j][2])
    with pytest.raises(ValueError, match='duplicate'):
        prepare_frame(pd.concat([raw, raw.iloc[[0]]]))


def test_bounds_change_decision_and_test_rows_rejected():
    rows = pd.DataFrame([
        {'partition': 'validation', 'requested_epsilon': eps, 'achieved_epsilon': eps,
         'environment': 'source', 'gap': 0.01, 'auc': 0.8,
         'gap_upper': 0.04 if eps == 2 else 0.02, 'auc_lower': 0.75}
        for eps in [2, 8]
    ])
    assert select(rows, ['source'], 0.03, 0.7, 8, False) == 2
    assert select(rows, ['source'], 0.03, 0.7, 8, True) == 8
    with pytest.raises(ValueError, match='test'):
        select(rows.assign(partition='test'), ['source'], 0.03, 0.7, 8, True)


def test_publication_run_smoke_and_frozen_choices(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('opacus')
    config = {'mode': 'synthetic_clipping', 'clip_norms': [1],
              'training': {'seeds': [10], 'strengths': [0.9], 'epsilons': [2],
                           'n_train': 128, 'n_eval': 128, 'epochs': 1, 'batch_size': 32}}
    manifest = run(config, tmp_path / 'study', tmp_path / 'cache')
    assert len(manifest['accounting']) == 3
    choices = json.loads((tmp_path / 'study/frozen_seed10_strength0.9.json').read_text())
    assert len(choices) == 6
    rows = pd.read_csv(tmp_path / 'study/metrics.csv')
    test = rows[rows.partition == 'test']
    np.testing.assert_allclose(test.total_interaction,
                               test.clipping_interaction + test.noise_interaction, atol=1e-12)
    config['training']['seeds'] = [0]
    with pytest.raises(ValueError, match='fresh'):
        run(config, tmp_path / 'invalid', tmp_path / 'cache')
