"""Fresh supplementary calibration of sandwich and conservative cluster bounds."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .auc_uncertainty import paired_bounds
from .cluster_concentration import cluster_radius
from .selection_inference import binomial_interval


def calibration(repetitions=200, seed=20261002, n_candidates=12, n_environments=5,
                n_references=3):
    if repetitions < 1 or min(n_candidates, n_environments, n_references) < 1:
        raise ValueError("positive calibration dimensions required")
    family = 2*n_candidates*n_environments*n_references
    rng = np.random.default_rng(seed)
    outputs = []
    for prevalence, n_groups, shared_sd in [(0.1, 100, 1.), (0.3, 250, .7), (0.5, 250, 1.2)]:
        covered = {"sandwich": 0, "concentration": 0}
        widths = {m: [] for m in covered}
        valid = 0
        class_failures = 0
        for _ in range(repetitions):
            groups = np.repeat(np.arange(n_groups), rng.integers(2, 7, n_groups))
            p = rng.beta(prevalence*5, (1-prevalence)*5, n_groups)
            y = (rng.random(len(groups)) < p[groups]).astype(int)
            if min(np.bincount(y, minlength=2)) < 20:
                class_failures += 1
                continue
            radius, within, _ = cluster_radius(y, groups, family=family)
            widths["concentration"].append(2*(radius+within))
            cover = dict.fromkeys(covered, True)
            local_widths = []
            for env in range(n_environments):
                shared = rng.normal(0, shared_sd, n_groups)[groups]
                common = rng.normal(size=len(y))
                refs = []
                for r in range(n_references):
                    mean = 1+.08*r-.12*env
                    scores = mean*y+shared+.6*common+.8*rng.normal(size=len(y))
                    refs.append((scores, float(norm.cdf(mean/math.sqrt(2*(shared_sd**2+1))))))
                for c in range(n_candidates):
                    mean = .65+.04*c-.1*env
                    scores = mean*y+shared+.6*common+.8*rng.normal(size=len(y))
                    true_auc = float(norm.cdf(mean/math.sqrt(2*(shared_sd**2+1))))
                    for ref, true_ref in refs:
                        b = paired_bounds(y, ref, scores, groups, family=family)
                        cover["sandwich"] &= b["gap_upper"] >= true_ref-true_auc and b["auc_lower"] <= true_auc
                        cover["concentration"] &= (b["gap"]+2*radius+2*within >= true_ref-true_auc
                                                   and b["auc"]-radius-within <= true_auc)
                        local_widths.append(b["gap_upper"]-b["gap"])
            widths["sandwich"].append(float(np.mean(local_widths)))
            for method in covered:
                covered[method] += int(cover[method])
            valid += 1
        if valid < .95*repetitions:
            raise ValueError("too many invalid-class calibration samples")
        outputs.append(dict(class_prevalence=prevalence, n_households=n_groups,
                            shared_score_sd=shared_sd, valid_repetitions=valid,
                            class_failures=class_failures,
                            methods={m: {"covered": k, "simultaneous_coverage": k/valid,
                                         "coverage_ci95": binomial_interval(k, valid),
                                         "mean_gap_radius": float(np.mean(widths[m]))}
                                     for m, k in covered.items()}))
        print(f'completed prevalence={prevalence} households={n_groups}', flush=True)
    return dict(seed=seed, repetitions_per_scenario=repetitions, family=family,
                n_candidates=n_candidates, n_environments=n_environments, n_references=n_references,
                nominal_simultaneous_coverage=.95, scenarios=outputs,
                scope="known conditional marginal Gaussian score laws and independent clusters; not ACS design or arbitrary-shift coverage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refuse to overwrite calibration")
    values = json.loads(args.config.read_text())
    result = calibration(**values)
    result["protocol_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    result["code_hashes"] = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                             for n in ["calibration_followup.py", "auc_uncertainty.py", "cluster_concentration.py"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
