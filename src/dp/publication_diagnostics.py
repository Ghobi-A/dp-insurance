"""Calibration, prospective seed planning and primary synthetic contrast reporting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.metrics import roc_auc_score

from .auc_uncertainty import paired_bounds


def calibration(repetitions=400, seed=20260930):
    rng = np.random.default_rng(seed)
    true_auc0 = float(norm.cdf(1 / np.sqrt(2 * 1.49)))
    true_auc1 = float(norm.cdf(0.8 / np.sqrt(2 * 1.49)))
    covers_gap, covers_floor = 0, 0
    for _ in range(repetitions):
        groups = np.repeat(np.arange(250), 4)
        y = rng.integers(0, 2, len(groups))
        shared = np.repeat(rng.normal(0, 0.7, 250), 4)
        reference = y + shared + rng.normal(size=len(y))
        candidate = 0.8 * y + shared + rng.normal(size=len(y))
        bounds = paired_bounds(y, reference, candidate, groups)
        covers_gap += bounds['gap_upper'] >= true_auc0 - true_auc1
        covers_floor += bounds['auc_lower'] <= true_auc1
    return {'repetitions': repetitions, 'simulation_seed': seed,
            'nominal_marginal_one_sided_coverage': 0.95,
            'gap_coverage': covers_gap / repetitions, 'floor_coverage': covers_floor / repetitions,
            'monte_carlo_standard_error_at_nominal': math.sqrt(0.95 * 0.05 / repetitions),
            'limitations': 'one Gaussian clustered DGP; marginal bounds only; not proof of ACS or simultaneous coverage'}


def cluster_bootstrap(y, reference, candidate, groups, replicates=999, seed=20260930):
    rng = np.random.default_rng(seed)
    unique, codes = np.unique(groups, return_inverse=True)
    gaps, aucs = [], []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[codes]
        if not all(weights[np.asarray(y) == label].sum() > 0 for label in (0, 1)):
            continue
        a0 = roc_auc_score(y, reference, sample_weight=weights)
        a1 = roc_auc_score(y, candidate, sample_weight=weights)
        gaps.append(a0 - a1)
        aucs.append(a1)
    if len(gaps) < 0.95 * replicates:
        raise ValueError('too many bootstrap replicates without both classes')
    return {'replicates_requested': replicates, 'replicates_valid': len(gaps),
            'marginal_gap_upper': float(np.quantile(gaps, 0.95)),
            'marginal_auc_lower': float(np.quantile(aucs, 0.05)),
            'method': 'paired household percentile bootstrap; marginal/asymptotic sensitivity only'}


def primary_values(rows, metric='total_interaction'):
    subset = rows[(rows.partition == 'test') & (rows.environment == 'proxy_0')
                  & (rows.strength == 0.9) & (rows.requested_epsilon == 2)]
    if subset.duplicated(['seed', 'clip_norm']).any():
        raise ValueError('duplicate primary seed/norm')
    pairs = subset.pivot(index='seed', columns='clip_norm', values=metric)
    if not {1.0, 5.0}.issubset(pairs.columns) or pairs[[1.0, 5.0]].isna().any().any():
        raise ValueError('incomplete primary paired contrast')
    return (pairs[1.0] - pairs[5.0]).to_numpy()


def plan_confirmation(input_dir: Path, config_path: Path, output_path: Path):
    rows = pd.read_csv(input_dir / 'metrics.csv')
    values = primary_values(rows)
    if len(values) < 2:
        raise ValueError('planning requires at least two paired seeds')
    sd = float(values.std(ddof=1))
    minimum_effect = 0.01
    proposed = max(20, math.ceil(((norm.ppf(0.975) + norm.ppf(0.8)) * sd / minimum_effect) ** 2))
    config = json.loads(config_path.read_text())
    config.update(stage='prospective_confirmation', clip_norms=[1.0, 5.0])
    config['training'].update(seeds=list(range(100, 100 + proposed)), strengths=[0.9],
                              epsilons=[2.0], n_eval=4000)
    config['primary'] = {'contrast': 'clip1_minus_clip5 total gap interaction',
                         'epsilon': 2.0, 'strength': 0.9, 'test_retention': 0.0,
                         'practical_effect': minimum_effect, 'alpha_two_sided': 0.05,
                         'power_planning_target': 0.8, 'development_paired_sd': sd,
                         'n_seeds': proposed,
                         'planning_limitation': 'normal approximation from five development seeds; not guaranteed achieved power'}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError('confirmation config already exists')
    output_path.write_text(json.dumps(config, indent=2) + '\n')
    return config['primary']


def confirmation_summary(input_dir: Path):
    rows = pd.read_csv(input_dir / 'metrics.csv')
    protocol = json.loads((input_dir / 'protocol.json').read_text())
    if protocol.get('stage') != 'prospective_confirmation':
        raise ValueError('not a prospectively labelled confirmation run')
    values = primary_values(rows)
    if len(values) != protocol['primary']['n_seeds'] or len(values) < 20:
        raise ValueError('incomplete prospective seed set')
    if set(rows.seed) != set(protocol['training']['seeds']):
        raise ValueError('seed set does not match prospective protocol')
    n, mean = len(values), float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    radius = float(t.ppf(0.975, n - 1) * se)
    result = {'n_independent_seeds': n, 'mean_primary_contrast': mean,
              'ci95_lower': mean - radius, 'ci95_upper': mean + radius,
              'two_sided_p': float(2 * t.sf(abs(mean / se), n - 1)) if se else None,
              'practical_effect_margin': 0.01,
              'scope': 'specified synthetic DGP and recipe; not general ACS transport or novel mechanism proof'}
    (input_dir / 'primary_result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def acs_bootstrap_check(config_path: Path, cache: Path, output_path: Path):
    """Predetermined seed/norm/budget sensitivity check; not new replication."""
    from dataclasses import replace

    import torch

    from .proxy_shift import StudyConfig, score, train_model
    from .publication_study import acs_environments

    torch.set_num_threads(1)
    values = json.loads(config_path.read_text())
    config = StudyConfig(**{k: tuple(v) if isinstance(v, list) else v
                            for k, v in values['training'].items()})
    config = replace(config, clip_norm=1.0)
    train, validation, _, _ = acs_environments(config, 10, cache)
    reference, _ = train_model(train, 10, config, 'ordinary')
    model, _ = train_model(train, 10, config, 'dpsgd', 2.0)
    X, y, groups = validation['source']
    ref_scores, candidate_scores = score(reference, X), score(model, X)
    result = {'cohort': 'ACS CA source validation', 'seed': 10, 'clip_norm': 1,
              'epsilon': 2, 'additional_diagnostic_fits': 2,
              'influence_marginal': paired_bounds(y, ref_scores, candidate_scores, groups),
              'bootstrap_marginal': cluster_bootstrap(y, ref_scores, candidate_scores, groups),
              'scope': 'one predetermined cell; marginal bounds only; no simultaneous coverage claim'}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError('bootstrap output already exists')
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-output', type=Path)
    parser.add_argument('--development-dir', type=Path)
    parser.add_argument('--base-config', type=Path)
    parser.add_argument('--confirmation-config', type=Path)
    parser.add_argument('--confirmation-dir', type=Path)
    parser.add_argument('--acs-bootstrap-config', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=Path('/tmp/dp-publication-acs'))
    parser.add_argument('--bootstrap-output', type=Path)
    args = parser.parse_args()
    if args.calibration_output:
        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_output.write_text(json.dumps(calibration(), indent=2) + '\n')
    if args.development_dir:
        print(json.dumps(plan_confirmation(args.development_dir, args.base_config,
                                            args.confirmation_config), indent=2))
    if args.confirmation_dir:
        print(json.dumps(confirmation_summary(args.confirmation_dir), indent=2))
    if args.acs_bootstrap_config:
        print(json.dumps(acs_bootstrap_check(args.acs_bootstrap_config, args.cache_dir,
                                            args.bootstrap_output), indent=2))


if __name__ == '__main__':
    main()
