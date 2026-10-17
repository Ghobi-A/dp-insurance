"""Read-only consistency checks for saved ACS decision-study evidence.

These checks do not establish execution order, data authenticity, or a clean
training reproduction. They reject incomplete or internally inconsistent tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .configuration_selection import summarize
from .proxy_shift import StudyConfig


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_table(actual, expected, keys):
    _require(set(actual.columns) == set(expected.columns), 'summary columns differ')
    actual = actual.sort_values(keys).reset_index(drop=True)[expected.columns]
    expected = expected.sort_values(keys).reset_index(drop=True)
    try:
        # CSV and in-memory summaries represent abstentions with different
        # null scalars (NaN/None/pd.NA), especially after pandas 3 concatenation.
        # Require identical missingness, then compare every nonmissing value.
        pd.testing.assert_frame_equal(actual.isna(), expected.isna())
        for column in expected:
            present = expected[column].notna()
            pd.testing.assert_series_equal(
                actual.loc[present, column].reset_index(drop=True),
                expected.loc[present, column].reset_index(drop=True),
                check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
            )
    except AssertionError as error:
        raise ValueError('derived summary differs from decisions') from error


def verify_run(folder: Path, source: Path, *, revision: str | None = None):
    manifest = json.loads((folder / 'manifest.json').read_text())
    protocol_path = folder / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    _require(manifest['status'] == 'completed', 'run is incomplete')
    _require(hashlib.sha256(protocol_path.read_bytes()).hexdigest() == manifest['protocol_sha256'],
             'protocol hash differs')
    for name, digest in manifest['code_hashes_at_launch'].items():
        content = ((source / name).read_bytes() if revision is None else
                   subprocess.check_output(['git', 'show', f'{revision}:src/dp/{name}'],
                                           cwd=source.parent.parent))
        _require(hashlib.sha256(content).hexdigest() == digest,
                 f'launch source hash differs: {name}')
    seeds = protocol['training']['seeds']
    config = StudyConfig(**protocol['training'])
    comparator = 'stability' in protocol
    keys = ['seed', 'policy', 'uncertainty_bounds', 'safety_margin']
    if comparator:
        keys.append('bound_rule')
    frozen = []
    for seed in seeds:
        name = f'frozen_seed{seed}.json' if comparator else f'frozen_seed{seed}_strength1.json'
        choices = json.loads((folder / name).read_text())['choices']
        _require(all(c['seed'] == seed for c in choices), 'frozen seed differs')
        frozen.extend(choices)
    frozen = pd.DataFrame(frozen)
    decisions = pd.read_csv(folder / 'decisions.csv')
    metrics = pd.read_csv(folder / 'metrics.csv')
    envs = [f'state_{s}' for s in protocol['data']['test']]
    _require(not frozen.duplicated(keys).any(), 'duplicate frozen choice')
    _require(not decisions.duplicated(keys + ['environment']).any(), 'duplicate decision')
    expected = frozen.merge(pd.DataFrame({'environment': envs}), how='cross')
    # An outer join detects missing environments and extra case keys. Reindexing
    # only existing environments silently drops those discrepancies.
    joined = expected.merge(decisions, on=keys + ['environment'], how='outer',
                            suffixes=('_frozen', '_actual'), indicator=True,
                            validate='one_to_one')
    _require(joined['_merge'].eq('both').all(), 'missing or unexpected decision case/environment')
    _require(joined.selected_candidate_frozen.fillna('ABSTAIN').equals(
        joined.selected_candidate_actual.fillna('ABSTAIN')), 'selected candidate differs from freeze')
    _require(decisions.abstained.equals(decisions.selected_candidate.isna()),
             'abstention disagrees with candidate')
    test = metrics[metrics.partition == 'test']
    metric_keys = ['seed', 'environment', 'candidate_id']
    _require(not test.duplicated(metric_keys).any(), 'duplicate external metric')
    for row in decisions.itertuples():
        if row.abstained:
            _require(pd.isna(row.failed), 'abstention has failure outcome')
            continue
        match = test[(test.seed == row.seed) & (test.environment == row.environment)
                     & (test.candidate_id == row.selected_candidate)]
        _require(len(match) == 1, 'selected external metric missing')
        metric = match.iloc[0]
        _require(np.isfinite([row.auc, row.gap]).all()
                 and np.allclose([row.auc, row.gap], [metric.auc, metric.gap],
                                 rtol=1e-12, atol=1e-12), 'decision metric differs')
        failed = metric.gap > config.tau or metric.auc < config.utility_floor
        _require(row.failed == failed, 'failure verdict differs from external metric')
    cases, summaries = [], []
    parts = decisions.groupby('bound_rule') if comparator else [(None, decisions)]
    for method, part in parts:
        c, s = summarize(part)
        if comparator:
            c, s = c.assign(bound_rule=method), s.assign(bound_rule=method)
        cases.append(c)
        summaries.append(s)
    _same_table(pd.read_csv(folder / 'cases.csv'), pd.concat(cases), keys)
    _same_table(pd.read_csv(folder / 'coverage_failure.csv'), pd.concat(summaries), keys[1:])
    accounting = pd.DataFrame(manifest['accounting'])
    _require(set(accounting.seed) == set(seeds), 'accounting seeds differ')
    _require(accounting.groupby('seed').size().eq(10 if comparator else 21).all(),
             'model fit count differs')
    private = accounting[accounting.condition == 'dpsgd']
    _require(((private.achieved_epsilon > 0)
              & (private.achieved_epsilon <= private.requested_epsilon)
              & (private.delta == config.delta)
              & (private.accountant == 'rdp')).all(), 'invalid per-model accounting')
    return {'run': folder.name, 'fits': len(accounting), 'decisions': len(decisions),
            'scope': 'saved-artifact consistency; no full training reproduction'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    for name in ('acs_replication', 'acs_digits', 'clipping_hashed', 'clipping_digits'):
        print(json.dumps(verify_run(args.root / 'reports/configuration_selection' / name,
                                   args.root / 'src/dp',
                                   revision=('3871f7c5ddf0e1303ac5a93fa999fbe5a0c61332'
                                             if name.startswith('acs_') else None))))


if __name__ == '__main__':
    main()
