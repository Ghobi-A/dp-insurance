import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp.cluster_concentration import cluster_radius, concentration_bounds


def test_household_bounded_differences_include_cross_class_pairs_once():
    y = np.tile([0, 1], 40)
    groups = np.repeat(np.arange(40), 2)
    radius, within, clusters = cluster_radius(y, groups, family=20)
    c = 1/40 + 1/40 - 1/1600
    assert radius == pytest.approx(np.sqrt(.5*40*c*c*np.log(20/.05)))
    assert within == pytest.approx(1/40)
    assert clusters == 40
    # Changing arbitrary scores inside one household changes no more than
    # the union of all rank pairs that touch that household.
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(11)
    original = rng.normal(size=len(y))
    modified = original.copy()
    modified[:2] = [100, -100]
    assert abs(roc_auc_score(y, original)-roc_auc_score(y, modified)) <= c
    # Splitting dependent households into rows falsely shrinks the radius.
    row_radius, row_within, _ = cluster_radius(y, np.arange(len(y)), family=20)
    assert row_radius < radius and row_within == 0


def test_concentration_bounds_are_invariant_to_row_order_and_widen_with_family():
    rng = np.random.default_rng(7)
    y = np.tile([0, 1], 40)
    groups = np.repeat(np.arange(40), 2)
    ref, scores = rng.normal(size=(2, len(y))) + y
    a = concentration_bounds(y, ref, scores, groups)
    b = concentration_bounds(y, ref, scores, groups, family=360)
    assert b['gap_upper'] >= a['gap_upper']
    assert b['auc_lower'] <= a['auc_lower']
    perm = rng.permutation(len(y))
    shuffled = concentration_bounds(y[perm], ref[perm], scores[perm], groups[perm])
    for key, value in a.items():
        if isinstance(value, float):
            assert shuffled[key] == pytest.approx(value, abs=1e-14)
        else:
            assert shuffled[key] == value
    assert not a['acs_coverage_certified']
    with pytest.raises(ValueError, match='invalid'):
        cluster_radius(y, groups, family=0)


def test_auto_s_uses_global_norm_and_preserves_direction_and_zero_gradients():
    torch = pytest.importorskip('torch')
    from dp.automatic_clipping import normalize_grad_samples
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(1))
    a.grad_sample = torch.tensor([[3., 0.], [0., 0.], [.001, 0.]])
    b.grad_sample = torch.tensor([[4.], [0.], [0.]])
    normalize_grad_samples([a, b], stability=.01)
    torch.testing.assert_close(a.grad_sample[0], torch.tensor([3/5.01, 0]))
    torch.testing.assert_close(b.grad_sample[0], torch.tensor([4/5.01]))
    assert a.grad_sample[2, 0] > .001  # AUTO-S can amplify small gradients
    norms = torch.sqrt((a.grad_sample*a.grad_sample).sum(1) + (b.grad_sample*b.grad_sample).sum(1))
    assert (norms < 1).all() and norms[1] == 0
    with pytest.raises(ValueError, match='stability'):
        normalize_grad_samples([a, b], stability=0)


def test_auto_s_one_step_matches_direct_per_example_update():
    torch = pytest.importorskip('torch')
    pytest.importorskip('opacus')
    from opacus import PrivacyEngine

    from dp.automatic_clipping import normalize_grad_samples
    X = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])
    y = torch.tensor([0., 1., 1.])
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    base = torch.optim.SGD(model.parameters(), lr=.1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=3)
    private, optimizer, _ = PrivacyEngine().make_private(module=model, optimizer=base,
        data_loader=loader, max_grad_norm=1., noise_multiplier=0., poisson_sampling=False)
    torch.nn.functional.binary_cross_entropy_with_logits(private(X).reshape(-1), y).backward()
    normalize_grad_samples(private.parameters())
    optimizer.step()
    error = .5-y
    individual = torch.cat([error[:, None]*X, error[:, None]], dim=1)
    expected = -.1*(individual/(individual.norm(dim=1, keepdim=True)+.01)).mean(0)
    actual = torch.cat([private._module.weight.reshape(-1), private._module.bias])
    torch.testing.assert_close(actual, expected)


def test_training_comparator_matches_noise_accounting_and_rejects_wrong_sensitivity():
    pytest.importorskip('torch')
    pytest.importorskip('opacus')
    from dataclasses import replace

    from dp.proxy_shift import StudyConfig, synthetic_cohort, train_model
    config = StudyConfig(n_train=64, epochs=1, batch_size=32)
    cohort = synthetic_cohort(64, 2, config, .9)
    _, ordinary = train_model(cohort, 3, config, 'dpsgd', 2)
    _, auto = train_model(cohort, 3, config, 'dpsgd', 2, clipping_mode='auto_s')
    for name in ['achieved_epsilon', 'noise_multiplier', 'steps', 'sample_rate', 'training_sha256']:
        assert auto[name] == ordinary[name]
    with pytest.raises(ValueError, match='unit sensitivity'):
        train_model(cohort, 3, replace(config, clip_norm=5), 'dpsgd', 2, clipping_mode='auto_s')


def test_comparator_freezes_every_rule_before_external_score(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('opacus')
    import dp.clipping_comparator as study
    original = study.synthetic_environments
    def guarded(*args):
        train, validation, load, metadata = original(*args)
        def test():
            frozen = json.loads((tmp_path/'run/frozen_seed300.json').read_text())
            assert len(frozen['choices']) == 30
            assert frozen['family'] == 36
            return load()
        return train, validation, test, metadata
    monkeypatch.setattr(study, 'synthetic_environments', guarded)
    values = json.loads(Path('research/configuration_selection/clipping_smoke.json').read_text())
    manifest = study.run(values, tmp_path/'run', tmp_path/'cache')
    assert manifest['status'] == 'completed' and len(manifest['accounting']) == 10
    outcomes = pd.read_csv(tmp_path/'run/coverage_failure.csv')
    assert set(outcomes.bound_rule) == {'point', 'sandwich', 'concentration'}
    assert len(outcomes) == 30
    with pytest.raises(ValueError, match='new or empty'):
        study.run(values, tmp_path/'run', tmp_path/'cache')


def test_fresh_calibration_concentration_smoke():
    from dp.calibration_followup import calibration
    result = calibration(repetitions=2, n_candidates=2, n_environments=2, n_references=2)
    assert result['family'] == 16 and result['seed'] == 20261002
    assert all(s['methods']['concentration']['simultaneous_coverage'] == 1 for s in result['scenarios'])
