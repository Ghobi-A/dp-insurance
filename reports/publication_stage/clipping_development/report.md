# Publication-stage development run

Mode: synthetic_clipping. No confirmatory result is claimed.

| Clipping norm | Rule | Bounds | Selected cases | Abstained cases | Failed evaluations / selected evaluations |
|---|---|---|---:|---:|---:|
| 0.5 | generic_shift | False | 10 | 0 | 12/20 |
| 0.5 | generic_shift | True | 10 | 0 | 12/20 |
| 0.5 | proxy_stress | False | 7 | 3 | 6/14 |
| 0.5 | proxy_stress | True | 7 | 3 | 6/14 |
| 0.5 | source_only | False | 10 | 0 | 12/20 |
| 0.5 | source_only | True | 10 | 0 | 12/20 |
| 1.0 | generic_shift | False | 10 | 0 | 5/20 |
| 1.0 | generic_shift | True | 10 | 0 | 5/20 |
| 1.0 | proxy_stress | False | 10 | 0 | 5/20 |
| 1.0 | proxy_stress | True | 10 | 0 | 5/20 |
| 1.0 | source_only | False | 10 | 0 | 5/20 |
| 1.0 | source_only | True | 10 | 0 | 5/20 |
| 2.0 | generic_shift | False | 10 | 0 | 5/20 |
| 2.0 | generic_shift | True | 10 | 0 | 5/20 |
| 2.0 | proxy_stress | False | 10 | 0 | 5/20 |
| 2.0 | proxy_stress | True | 10 | 0 | 5/20 |
| 2.0 | source_only | False | 10 | 0 | 5/20 |
| 2.0 | source_only | True | 10 | 0 | 5/20 |
| 5.0 | generic_shift | False | 10 | 0 | 5/20 |
| 5.0 | generic_shift | True | 10 | 0 | 5/20 |
| 5.0 | proxy_stress | False | 10 | 0 | 5/20 |
| 5.0 | proxy_stress | True | 10 | 0 | 5/20 |
| 5.0 | source_only | False | 10 | 0 | 5/20 |
| 5.0 | source_only | True | 10 | 0 | 5/20 |

Evaluations share models; seeds are the training replication unit.
Bounds are asymptotic with household-cluster variance on ACS and a Bonferroni
validation family across budgets, clipping norms, environments and endpoints.
They describe validation uncertainty, not guarantees in unseen states.
ACS is a fixed hashed-representation, unweighted-income prediction variant.
Natural state shift changes many variables; occupation masking is hypothetical.
