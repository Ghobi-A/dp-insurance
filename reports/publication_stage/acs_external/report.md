# Publication-stage development run

Mode: acs_external. No confirmatory result is claimed.

| Clipping norm | Rule | Bounds | Selected cases | Abstained cases | Failed evaluations / selected evaluations |
|---|---|---|---:|---:|---:|
| 1.0 | natural_shift | False | 3 | 2 | 1/6 |
| 1.0 | natural_shift | True | 0 | 5 | 0/0 |
| 1.0 | proxy_stress | False | 2 | 3 | 1/4 |
| 1.0 | proxy_stress | True | 0 | 5 | 0/0 |
| 1.0 | source_only | False | 3 | 2 | 1/6 |
| 1.0 | source_only | True | 0 | 5 | 0/0 |
| 5.0 | natural_shift | False | 5 | 0 | 0/10 |
| 5.0 | natural_shift | True | 5 | 0 | 0/10 |
| 5.0 | proxy_stress | False | 5 | 0 | 0/10 |
| 5.0 | proxy_stress | True | 5 | 0 | 0/10 |
| 5.0 | source_only | False | 5 | 0 | 0/10 |
| 5.0 | source_only | True | 5 | 0 | 0/10 |

Evaluations share models; seeds are the training replication unit.
Bounds are asymptotic with household-cluster variance on ACS and a Bonferroni
validation family across budgets, clipping norms, environments and endpoints.
They describe validation uncertainty, not guarantees in unseen states.
ACS is a fixed hashed-representation, unweighted-income prediction variant.
Natural state shift changes many variables; occupation masking is hypothetical.
