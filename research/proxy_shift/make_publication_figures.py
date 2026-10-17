"""Reproduce descriptive seed-level figures from committed publication-stage CSVs."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import t

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'reports/publication_stage'


def interval(values):
    values = np.asarray(values)
    return values.mean(), t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))


def main():
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    figures = RESULTS / 'figures'
    figures.mkdir(exist_ok=True)
    rows = pd.read_csv(RESULTS / 'confirmation/metrics.csv')
    primary = rows[(rows.partition == 'test') & (rows.environment == 'proxy_0')]
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    for offset, (column, label, color) in enumerate([
        ('total_interaction', 'Total', '#263c70'),
        ('clipping_interaction', 'Clipping-associated', '#23846d'),
        ('noise_interaction', 'Added noise', '#bd7524'),
    ]):
        estimates = [interval(primary.loc[primary.clip_norm == clip, column]) for clip in [1, 5]]
        ax.errorbar(np.arange(2) + (offset - 1) * 0.15,
                    [v[0] for v in estimates], yerr=[v[1] for v in estimates],
                    fmt='o', capsize=4, color=color, label=label)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set(xticks=[0, 1], xticklabels=['Clipping norm 1', 'Clipping norm 5'],
           ylabel='Shift minus source AUC gap',
           title='Fresh-seed synthetic confirmation: ε=2, strength=0.90')
    ax.legend(frameon=False)
    fig.savefig(figures / 'synthetic_components.svg')
    plt.close(fig)
    rows = pd.read_csv(RESULTS / 'acs_external/metrics.csv')
    rows = rows[(rows.partition == 'test') & (rows.requested_epsilon == 2)]
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    for offset, (clip, label, color) in enumerate([
        (1, 'DP norm 1', '#bd7524'), (5, 'DP norm 5', '#23846d'),
        (5, 'Ordinary reference', '#263c70'),
    ]):
        column = 'reference_auc' if offset == 2 else 'auc'
        estimates = [interval(rows.loc[(rows.clip_norm == clip) & (rows.environment == env), column])
                     for env in ['source', 'state_AZ', 'state_NV']]
        ax.errorbar(np.arange(3) + (offset - 1) * 0.15,
                    [v[0] for v in estimates], yerr=[v[1] for v in estimates],
                    fmt='o', capsize=4, color=color, label=label)
    ax.set(xticks=[0, 1, 2], xticklabels=['CA source test', 'AZ external', 'NV external'],
           ylabel='AUC', title='ACS external pilot: ε=2, five training/cohort seeds')
    ax.legend(frameon=False)
    fig.savefig(figures / 'acs_utility.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
