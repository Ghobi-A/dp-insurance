# Proxy-shift exploratory run

Dataset: `insurance`; seeds: (0, 1, 2, 3, 4).

No confirmatory significance or established novelty is claimed. Seed intervals
are descriptive and do not integrate all evaluation-sample uncertainty.

Selection uses validation point estimates and an achieved-epsilon cap.
Held-out failures include loss-margin OR absolute-utility violations.

| Rule | Decisions (seed × strength) | Selected | Abstained | Failed held-out evaluations / selected evaluations |
|---|---:|---:|---:|---:|
| generic_shift | 5 | 0 | 5 | 0/0 |
| proxy_stress | 5 | 0 | 5 | 0/0 |
| source_only | 5 | 0 | 5 | 0/0 |

Held-out evaluations share trained models; the denominator above is descriptive,
not a count of independent training replicates. Failure is undefined for abstentions.

| Held-out environment | Mean DP gap interaction | Range over training seeds/strengths/budgets |
|---|---:|---:|
| proxy_0 | -0.0442 | -0.0919 to 0.0028 |
| proxy_0.25 | -0.0405 | -0.0749 to 0.0023 |

The pooled interaction is a grid description, not a population effect or a test.
Inspect `seed_summary.csv` by strength and budget before interpreting it.

## Dataset scope

Synthetic controls and insurance stress tests support a narrow exploratory study.
Insurance masking is a hypothetical zero-imputation failure policy, not observed drift.
Real external validation (prespecified ACS state/year environments) is required before
claims that recommendations generalise. A second independent tabular domain belongs
in future scope unless broad cross-domain claims are pursued now.

The matched stable-signal condition has identical source data and preserves that
signal in every synthetic test environment; it is a controlled counterfactual,
not a new robust-training algorithm. Secure RNG is disabled for reproducible research.
