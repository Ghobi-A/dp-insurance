import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dp.acs_publication import prepare_frame
from dp.configuration_selection import (
    paired_primary,
    run,
    select_configuration,
    summarize,
    validate_protocol,
)


def candidates():
    return pd.DataFrame([
        dict(partition='validation', candidate_id=c, environment=e,
             achieved_epsilon=eps, gap=gap, gap_upper=gap+.005,
             auc=auc, auc_lower=auc-.01)
        for c, eps, gap, auc in [('fixed', 2., .05, .76), ('joint', 2., .01, .81),
                                 ('large_epsilon', 8., .0, .85)]
        for e in ['source', 'shift']
    ])


def test_joint_selection_finds_same_epsilon_better_recipe_and_respects_cap():
    rows = candidates()
    kwargs = dict(tau=.03, floor=.7, cap=8, use_bounds=True)
    assert select_configuration(rows, ['source','shift'], **kwargs) == 'joint'
    assert select_configuration(rows, ['source'], allowed=['fixed'], **kwargs) is None
    assert select_configuration(rows, ['source'], margin=.02, **kwargs) == 'large_epsilon'
    kwargs['cap'] = 2
    assert select_configuration(rows, ['source'], margin=.02, **kwargs) is None
    with pytest.raises(ValueError, match='validation'):
        select_configuration(rows.assign(partition='test'), ['source'], **kwargs)
    with pytest.raises(ValueError, match='incomplete'):
        select_configuration(rows.iloc[:-1], ['source','shift'], **kwargs)
    with pytest.raises(ValueError, match='nonfinite'):
        select_configuration(rows.assign(gap=np.nan), ['source'], **kwargs)


def test_stable_tie_break_and_row_order_invariance():
    rows = candidates()
    rows.loc[rows.candidate_id == 'fixed', ['gap','gap_upper','auc','auc_lower']] = [.01,.015,.81,.8]
    kwargs = dict(tau=.03, floor=.7, cap=8, use_bounds=True)
    assert select_configuration(rows, ['source'], **kwargs) == 'fixed'
    assert select_configuration(rows.sample(frac=1), ['source'], **kwargs) == 'fixed'


def test_abstention_is_not_success_and_environments_are_not_replications():
    rows = pd.DataFrame([
        dict(seed=s, strength=1., policy=p, uncertainty_bounds=True, safety_margin=0.,
             environment=e, abstained=p=='abstain', failed=(p=='select' and s==1 and e=='b'),
             auc=.8 if p=='select' else None)
        for s in [1,2] for p in ['select','abstain'] for e in ['a','b']
    ])
    cases, summary = summarize(rows)
    select = summary[summary.policy == 'select'].iloc[0]
    abstain = summary[summary.policy == 'abstain'].iloc[0]
    assert select.n_cases == 2
    assert select.coverage == 1 and select.successful_yield == .5
    assert abstain.coverage == 0 and abstain.successful_yield == 0
    result = paired_primary(cases, dict(bounds=True, margin=0., policy_a='select', policy_b='abstain'))
    assert result['n_seed_units'] == 2 and result['difference'] == .5
    assert not result['interval_supported']
    with pytest.raises(ValueError, match='incomplete'):
        paired_primary(cases[cases.policy != 'select'], dict(bounds=True, margin=0., policy_a='select', policy_b='abstain'))


def test_protocol_fresh_seeds_and_disjoint_states():
    values = json.loads(Path('research/configuration_selection/acs_replication.json').read_text())
    assert len(validate_protocol(values).seeds) == 20
    values['training']['seeds']=[10]
    with pytest.raises(ValueError, match='fresh'):
        validate_protocol(values)
    values['stage']='pilot'
    values['data']['test']=['CA']
    with pytest.raises(ValueError, match='disjoint'):
        validate_protocol(values)


def test_digits_distinguish_hash_collision_and_are_rowwise():
    raw = pd.DataFrame({'SERIALNO':['a','b'], 'SPORDER':[1,1], 'AGEP':[40,40],
                        'COW':[1,1], 'SCHL':[20,20], 'MAR':[1,1], 'OCCP':[100,132],
                        'POBP':[10,10], 'RELP':[0,0], 'WKHP':[40,40], 'SEX':[1,1],
                        'RAC1P':[1,1], 'PINCP':[20000,80000], 'PWGTP':[1,1]})
    hashed, _, _, _ = prepare_frame(raw)
    digits, _, _, metadata = prepare_frame(raw, year=2017, representation='digits')
    np.testing.assert_array_equal(hashed[0], hashed[1])
    assert not np.array_equal(digits[0],digits[1])
    assert metadata['year'] == 2017
    alone, _, _, _ = prepare_frame(raw.iloc[:1],year=2017,representation='digits')
    np.testing.assert_array_equal(digits[0],alone[0])


def test_end_to_end_frozen_before_external_score_and_decomposition(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('opacus')
    import dp.configuration_selection as study
    original = study.synthetic_environments
    def guarded(*args):
        train, validation, loader, metadata = original(*args)
        def external():
            frozen = json.loads((tmp_path/'run/frozen_seed200_strength0.9.json').read_text())
            assert frozen['choices'] and frozen['reference_recipe']
            return loader()
        return train,validation,external,metadata
    monkeypatch.setattr(study,'synthetic_environments',guarded)
    values = json.loads(Path('research/configuration_selection/smoke.json').read_text())
    manifest=run(values,tmp_path/'run',tmp_path/'cache')
    assert manifest['status']=='completed'
    assert len(manifest['accounting'])==21
    metrics=pd.read_csv(tmp_path/'run/metrics.csv')
    test=metrics[metrics.partition=='test']
    np.testing.assert_allclose(test.gap,test.optimization_gap+test.clipping_gap+test.noise_gap,atol=1e-12)
    assert metrics.family.nunique()==1
    # All possible selected ordinary references are in the confidence family.
    assert metrics.family.iloc[0]==2*12*5*3
    with pytest.raises(ValueError,match='new or empty'):
        run(values,tmp_path/'run',tmp_path/'cache')


def test_complete_family_calibration_smoke():
    from dp.selection_calibration import calibration
    result = calibration(repetitions=2, n_candidates=2, n_environments=2, n_references=2)
    assert result['family'] == 16
    assert len(result['scenarios']) == 3
    assert all(0 <= s['simultaneous_coverage'] <= 1 for s in result['scenarios'])


def test_exact_paired_yield_handles_degenerate_and_null_cases():
    from dp.selection_inference import exact_paired_yield
    gain = exact_paired_yield(np.ones(20),np.zeros(20))
    assert gain['difference'] == 1 and .5 < gain['ci95_lower'] < 1
    assert gain['ci95_upper'] == 1
    null = exact_paired_yield(np.ones(20),np.ones(20))
    assert null['ci95_lower'] < 0 < null['ci95_upper']
    reverse = exact_paired_yield(np.zeros(20),np.ones(20))
    assert reverse['ci95_upper'] == pytest.approx(-gain['ci95_lower'])
    with pytest.raises(ValueError,match='binary'):
        exact_paired_yield([.5],[1.])
