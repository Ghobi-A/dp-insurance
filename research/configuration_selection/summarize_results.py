"""Descriptive coverage comparisons and figures; never tunes a selection rule."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dp.selection_inference import exact_paired_yield


def build(root):
    summary = pd.read_csv(root / 'coverage_failure.csv')
    cases = pd.read_csv(root / 'cases.csv')
    protocol = json.loads((root / 'protocol.json').read_text())
    primary = protocol['primary']
    rows = cases[(cases.uncertainty_bounds == primary['bounds'])
                 & (cases.safety_margin == primary['margin'])]
    if rows.strength.nunique() != 1:
        raise ValueError('exact binary analysis needs one strength per seed')
    pairs = rows.pivot(index='seed', columns='policy', values='successful')
    supplement = exact_paired_yield(pairs[primary['policy_a']].astype(int),
                                    pairs[primary['policy_b']].astype(int))
    (root / 'supplementary_exact_yield.json').write_text(json.dumps(supplement, indent=2) + '\n')
    comparisons = []
    baseline = summary[summary.policy == 'epsilon_source']
    for _, a in baseline.iterrows():
        matches = summary[(summary.policy != 'epsilon_source')
                          & (summary.uncertainty_bounds == a.uncertainty_bounds)
                          & np.isclose(summary.coverage, a.coverage)]
        for _, b in matches.iterrows():
            comparisons.append(dict(baseline='epsilon_source', policy=b.policy,
                                    bounds=a.uncertainty_bounds, coverage=a.coverage,
                                    baseline_margin=a.safety_margin, policy_margin=b.safety_margin,
                                    success_yield_difference=b.successful_yield-a.successful_yield,
                                    failure_difference=b.conditional_failure-a.conditional_failure))
    pd.DataFrame(comparisons).to_csv(root / 'exact_coverage_comparisons.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {'epsilon_source':'#d05d44', 'joint_clip_shift':'#398f9d',
              'joint_recipe_source':'#8b6fc0', 'joint_recipe_shift':'#3a9566',
              'logistic_shift':'#ae8335'}
    for policy, color in colors.items():
        data = summary[(summary.policy == policy) & summary.uncertainty_bounds].sort_values('coverage')
        axes[0].plot(data.coverage, data.conditional_failure, 'o-', label=policy, color=color)
        axes[1].plot(data.coverage, data.successful_yield, 'o-', color=color)
    for ax in axes:
        ax.set_xlim(-.03, 1.03)
        ax.set_ylim(-.03, 1.03)
        ax.set_xlabel('Recommendation coverage')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Failure among selected cases')
    axes[1].set_ylabel('Successful recommendations / all cases')
    axes[0].legend(fontsize=7)
    fig.suptitle('Fixed safety-margin sweep; bounded rules; descriptive seed/cohort outcomes')
    fig.savefig(root / 'coverage_failure.svg')
    plt.close(fig)


if __name__ == '__main__':
    for directory in ['acs_replication', 'acs_digits']:
        build(Path('reports/configuration_selection') / directory)
