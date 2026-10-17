"""Saved evidence must fail closed on missing cases and inconsistent outcomes."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from dp.artifact_validation import verify_run

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/dp'
ORIGINAL_REVISION = '3871f7c5ddf0e1303ac5a93fa999fbe5a0c61332'


@pytest.mark.parametrize('name', ['acs_replication', 'acs_digits', 'clipping_hashed', 'clipping_digits'])
def test_saved_acs_evidence_is_internally_consistent(name):
    result = verify_run(ROOT / 'reports/configuration_selection' / name, SOURCE,
                        revision=ORIGINAL_REVISION if name.startswith('acs_') else None)
    assert result['fits'] == (420 if name.startswith('acs_') else 200)


@pytest.mark.parametrize('corruption', ['missing_environment', 'extra_case', 'wrong_outcome', 'wrong_summary'])
def test_saved_evidence_rejects_corruption(tmp_path, corruption):
    folder = tmp_path / 'study'
    shutil.copytree(ROOT / 'reports/configuration_selection/clipping_digits', folder)
    path = folder / ('coverage_failure.csv' if corruption == 'wrong_summary' else 'decisions.csv')
    rows = pd.read_csv(path)
    if corruption == 'missing_environment':
        rows = rows[rows.environment != 'state_UT']
    elif corruption == 'extra_case':
        extra = rows.iloc[[0]].assign(seed=999)
        rows = pd.concat([rows, extra])
    elif corruption == 'wrong_outcome':
        ix = rows.index[~rows.abstained][0]
        rows.loc[ix, 'failed'] = not rows.loc[ix, 'failed']
    else:
        rows.loc[0, 'successful_yield'] = .12345
    rows.to_csv(path, index=False)
    with pytest.raises(ValueError):
        verify_run(folder, SOURCE)
