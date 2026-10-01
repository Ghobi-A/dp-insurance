"""Build supplementary figures/tables and verify completed run artifacts."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[2]
    report = root/'reports/configuration_selection'
    summaries, checks = [], []
    for encoding in ['hashed', 'digits']:
        folder = report/f'clipping_{encoding}'
        manifest = json.loads((folder/'manifest.json').read_text())
        assert manifest['status'] == 'completed'
        assert len(manifest['accounting']) == 200
        assert sorted({r['seed'] for r in manifest['accounting']}) == list(range(300,320))
        for name, expected in manifest['code_hashes_at_launch'].items():
            assert hashlib.sha256((root/'src/dp'/name).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest() == manifest['protocol_sha256']
        frozen = []
        for seed in range(300,320):
            choices = json.loads((folder/f'frozen_seed{seed}.json').read_text())
            assert choices['family'] == 36 and len(choices['choices']) == 60
            frozen.extend(choices['choices'])
        keys = ['seed','policy','bound_rule','safety_margin']
        frozen = pd.DataFrame(frozen).set_index(keys)
        decisions = pd.read_csv(folder/'decisions.csv')
        assert set(decisions.environment) == {'state_CO', 'state_UT'}
        for _, group in decisions.groupby('environment'):
            actual = group.set_index(keys).reindex(frozen.index)
            assert actual.index.equals(frozen.index)
            assert actual.selected_candidate.fillna('ABSTAIN').equals(frozen.selected_candidate.fillna('ABSTAIN'))
        private = [r for r in manifest['accounting'] if r['condition'] == 'dpsgd']
        assert len(private) == 120
        assert all(0 < r['achieved_epsilon'] <= r['requested_epsilon'] <= 8 for r in private)
        assert all(r['delta'] == 1e-5 and r['accountant'] == 'rdp' for r in private)
        # All method variants use the same record sample and Poisson schedules.
        for seed in range(300,320):
            a = [r for r in manifest['accounting'] if r['seed'] == seed]
            assert len(a) == 10
            assert len({r['training_sha256'] for r in a}) == 1
            assert len({(r['steps'], r['sample_rate']) for r in a}) == 1
        metrics = pd.read_csv(folder/'metrics.csv')
        test = metrics[metrics.partition == 'test']
        np.testing.assert_allclose(test.gap, test.clipping_gap+test.noise_gap, atol=1e-12)
        summary = pd.read_csv(folder/'coverage_failure.csv').assign(representation=encoding)
        assert set(summary.bound_rule) == {'point','sandwich','concentration'}
        assert len(summary) == 60 and (summary.n_cases == 20).all()
        summaries.append(summary)
        checks.append(dict(representation=encoding,status='passed',n_fits=200,
                           private_candidates=120,seed_cases=20,
                           checks=['launch source hashes','protocol hash','accounting cap/delta',
                                   'matched training streams','frozen choice/decision consistency',
                                   'external environment set','decomposition identity'],
                           scope='artifact/invariant checks; no full rerun or scientific certification'))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(report/'clipping_comparison.csv',index=False)
    (report/'followup_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    policies = ['standard_c1_source','standard_joint_source','auto_s_source','combined_source','combined_shift']
    labels = ['Fixed C=1','Joint C=1/5','AUTO-S','Combined\nsource','Combined\nshift']
    fig, axes = plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax, representation in zip(axes, ['hashed','digits']):
        x = np.arange(len(policies))
        for offset, (method,color) in enumerate([('point','#4878A8'),('sandwich','#59A14F'),('concentration','#C66A48')]):
            part=summary[(summary.representation==representation)&summary.safety_margin.eq(0)&(summary.bound_rule==method)].set_index('policy')
            ax.bar(x+(offset-1)*.24,part.loc[policies].successful_yield,.22,label=method,color=color)
        ax.set_xticks(x,labels)
        ax.set_ylim(0,1.08)
        ax.set_title(f'{representation.capitalize()} encoding; 20 seed cases')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Successful recommendations / all cases')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False)
    fig.suptitle('Supplementary matched clipping comparison; margin 0',fontsize=12)
    fig.tight_layout(rect=(0,.07,1,.95))
    for suffix in ['svg','png']:
        fig.savefig(report/f'clipping_comparison.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    cal=json.loads((report/'calibration_followup.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4.4))
    for offset, (method,color) in enumerate([('sandwich','#4878A8'),('concentration','#C66A48')]):
        xs=np.arange(3)+(offset-.5)*.3
        values=np.array([s['methods'][method]['simultaneous_coverage'] for s in cal['scenarios']])
        intervals=np.array([s['methods'][method]['coverage_ci95'] for s in cal['scenarios']])
        ax.bar(xs,values,.28,color=color,label=method)
        ax.errorbar(xs,values,yerr=np.stack([values-intervals[:,0],intervals[:,1]-values]),fmt='none',color='#222222',capsize=4)
    ax.axhline(.95,color='#222222',linestyle='--',linewidth=1,label='Nominal 95%')
    ax.set_xticks(np.arange(3),['Prevalence .10\n100 households','Prevalence .30\n250 households','Prevalence .50\n250 households'])
    ax.set_ylim(.75,1.025)
    ax.set_ylabel('Simultaneous coverage; exact binomial 95% intervals')
    ax.set_title('Fresh known-Gaussian calibration; 360 endpoints')
    ax.legend(loc='lower right',frameon=False)
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for suffix in ['svg','png']:
        fig.savefig(report/f'calibration_followup.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    table = summary[summary.safety_margin.eq(0)].copy()
    for source,target in [('coverage','selected'),('successful_yield','successful')]:
        table[target] = (table[source]*table.n_cases).round().astype(int).astype(str)+'/20'
    table = table[['representation','bound_rule','policy','selected','successful']]
    (report/'clipping_comparison_table.md').write_text(table.to_markdown(index=False)+'\n')
    for filename in ['clipping_comparison.svg', 'calibration_followup.svg']:
        path = report/filename
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
