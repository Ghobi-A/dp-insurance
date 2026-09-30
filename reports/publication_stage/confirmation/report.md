# Publication-stage run

Mode: synthetic_clipping. Stage: prospective_confirmation.

| Clipping norm | Rule | Bounds | Selected cases | Abstained cases | Failed evaluations / selected evaluations |
|---|---|---|---:|---:|---:|
| 1.0 | generic_shift | False | 20 | 0 | 20/40 |
| 1.0 | generic_shift | True | 20 | 0 | 20/40 |
| 1.0 | proxy_stress | False | 20 | 0 | 20/40 |
| 1.0 | proxy_stress | True | 20 | 0 | 20/40 |
| 1.0 | source_only | False | 20 | 0 | 20/40 |
| 1.0 | source_only | True | 20 | 0 | 20/40 |
| 5.0 | generic_shift | False | 20 | 0 | 19/40 |
| 5.0 | generic_shift | True | 20 | 0 | 19/40 |
| 5.0 | proxy_stress | False | 20 | 0 | 19/40 |
| 5.0 | proxy_stress | True | 20 | 0 | 19/40 |
| 5.0 | source_only | False | 20 | 0 | 19/40 |
| 5.0 | source_only | True | 20 | 0 | 19/40 |

Evaluations share models; seeds are the training replication unit.
Bounds are asymptotic with household-cluster variance on ACS and a Bonferroni
validation family across budgets, clipping norms, environments and endpoints.
They describe validation uncertainty, not guarantees in unseen states.
ACS is a fixed hashed-representation, unweighted-income prediction variant.
Natural state shift changes many variables; occupation masking is hypothetical.
