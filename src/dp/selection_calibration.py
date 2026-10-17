"""Simultaneous AUC-bound calibration under known clustered Gaussian DGPs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .auc_uncertainty import paired_bounds


def calibration(repetitions=200, seed=20261001, n_candidates=12, n_environments=5,
                n_references=3):
    if repetitions < 1 or min(n_candidates, n_environments, n_references) < 1:
        raise ValueError("positive repetitions and family dimensions required")
    scenarios = [(0.1, 100, 1.0), (0.3, 250, 0.7), (0.5, 250, 1.2)]
    rng = np.random.default_rng(seed)
    output = []
    family = 2 * n_candidates * n_environments * n_references
    for prevalence, n_groups, shared_sd in scenarios:
        covers, valid, unsupported, class_failures = 0, 0, 0, 0
        for _ in range(repetitions):
            sizes = rng.integers(2, 7, n_groups)
            groups = np.repeat(np.arange(n_groups), sizes)
            # Beta-binomial labels produce within-household label correlation.
            household_p = rng.beta(prevalence * 5, (1 - prevalence) * 5, n_groups)
            y = (rng.random(len(groups)) < household_p[groups]).astype(int)
            if min(np.bincount(y, minlength=2)) < 20:
                class_failures += 1
                continue
            all_cover = True
            for env in range(n_environments):
                shared = rng.normal(0, shared_sd, n_groups)[groups]
                common = rng.normal(size=len(y))
                references = []
                for r in range(n_references):
                    mean = 1 + .08 * r - .12 * env
                    scores = mean * y + shared + .6 * common + .8 * rng.normal(size=len(y))
                    references.append((scores, float(norm.cdf(mean / math.sqrt(2 * (shared_sd ** 2 + 1))))))
                for candidate in range(n_candidates):
                    mean = .65 + .04 * candidate - .1 * env
                    scores = mean * y + shared + .6 * common + .8 * rng.normal(size=len(y))
                    true_auc = float(norm.cdf(mean / math.sqrt(2 * (shared_sd ** 2 + 1))))
                    for reference, true_ref in references:
                        bound = paired_bounds(y, reference, scores, groups, family=family)
                        unsupported += int(not bound['bounds_supported'])
                        all_cover &= (bound['gap_upper'] >= true_ref - true_auc and
                                      bound['auc_lower'] <= true_auc)
            covers += int(all_cover)
            valid += 1
        if valid < .95 * repetitions:
            raise ValueError("too many invalid-class calibration samples")
        output.append({'class_prevalence': prevalence, 'n_households': n_groups,
                       'shared_score_sd': shared_sd, 'repetitions_valid': valid,
                       'class_failures': class_failures, 'unsupported_bounds': unsupported,
                       'simultaneous_coverage': covers / valid,
                       'monte_carlo_se': math.sqrt((covers / valid) * (1 - covers / valid) / valid)})
    return {'simulation_seed': seed, 'repetitions_per_scenario': repetitions,
            'family': family, 'n_candidates': n_candidates, 'n_environments': n_environments,
            'n_references': n_references, 'nominal_simultaneous_coverage': .95,
            'scenarios': output,
            'scope': 'known Gaussian marginal AUCs; unequal cluster sizes and dependent labels/scores; not ACS survey or finite-sample validation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse to overwrite calibration')
    result = calibration(args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
