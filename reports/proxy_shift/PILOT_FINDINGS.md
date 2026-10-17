# Proxy-shift implementation and exploratory findings — 30 September 2026

The new study is executable end to end on synthetic and public insurance data.
It does not establish novelty or complete the confirmatory/external study.
Protocol: [docs/PROXY_SHIFT_STUDY.md](../../docs/PROXY_SHIFT_STUDY.md).

## Executed evidence

The fixed initial pilot trains 120 synthetic models (80 DP plus 40 controls)
and 30 insurance models (20 DP plus 10 controls). A separately named,
post-pilot insurance optimisation diagnostic trains another 30 models. Total:
180 independently fitted condition/seed/budget models, including 120 DP models.
Conditions within a seed share initialisation/sampling and are paired, not
180 independent statistical replicates. All cells and seeds are retained.

| Run | Source-validation mean ordinary AUC | Mean DP AUC across the budget grid | Source-only selected cases | Proxy-stress selected cases |
|---|---:|---:|---:|---:|
| Synthetic pilot, 15 epochs | 0.9220 | 0.9174 | 20/20 | 16/20 |
| Insurance pilot, 15 epochs | 0.7001 | 0.6405 | 0/5 | 0/5 |
| Insurance training diagnostic, 100 epochs / LR 0.3 | 0.9861 | 0.9694 | 5/5 | 1/5 |

These are descriptive averages across the stated grid; differences are not
confirmatory significance tests. The longer insurance recipe changed epochs
and learning rate together after seeing the weak source performance. It
checks optimisation adequacy, not a new independent test of the hypothesis.

## Decision failures

Synthetic source-only and generic-shift rules select the same epsilon-1 setting
on all 20 seed/strength cases. Across two held-out proxy levels, each has
18/40 failed evaluations: eight exceed the 0.03 AUC-loss allowance, fourteen
fall below the 0.70 absolute AUC floor, with overlap. Proxy stress abstains on
four cases, selects epsilon 1 on the other sixteen, and has 10/32 failed
evaluations: none exceed the loss allowance, ten fail the absolute floor.

**The improvement is entirely from selective abstention, not a better epsilon
choice.** The eight excluded evaluations correspond to four seed/strength
cases. Do not describe 40 environment evaluations as 40 independent models,
or report 0/32 margin violations while concealing ten absolute-utility failures.
Generic and proxy validation severities are not information-equivalent; the
comparison cannot establish superiority of proxy-specific validation generally.

The first insurance recipe never selects a candidate. Its conditional failure
rate is undefined, not zero. In the longer diagnostic, source-only/generic
rules select on all five seeds but all ten held-out evaluations fail the
absolute floor; two also fail the AUC-loss allowance. Proxy stress selects
on one seed and abstains on four; both selected held-out evaluations still
fail the floor. Thus stress validation on moderate failures does not guarantee
utility under more severe proxy failure.

## Clipping is the main synthetic interaction

The primary total-gap interaction is defined relative to ordinary training.
Clipping-only controls allow an exact algebraic decomposition:

`total gap change = clipping gap change + added-noise gap change`.

| Synthetic held-out proxy retention | Mean total interaction | Clipping component | Added-noise component |
|---|---:|---:|---:|
| 0 | 0.014073 | 0.013974 | 0.000099 |
| 0.25 | 0.009933 | 0.009907 | 0.000026 |

These pooled grid means are dominated by clipping under this fixed recipe.
They do not support a noise-driven fragility claim. A near-zero mean does not
establish equivalence or exclude individual seed/budget effects. Existing
literature already links stable features and DP utility; this control analysis
is a methodological requirement, not itself evidence of novel discovery.

The insurance total interaction is negative on average, including the longer
diagnostic. Severe masking harms both ordinary and DP utility; the DP-relative
gap can shrink even as absolute utility becomes unacceptable. This supports
reporting both loss margins and absolute utility, rather than treating a
smaller relative gap as a successful recommendation.

## Next study gate and dataset decision

1. On a distinct public development cohort, assess training adequacy and
   clipping sensitivity (for example, norms 0.5/1/2/5), with matched noise and
   unnoised controls. Preserve these pilot results; freeze a new protocol
   before confirmatory outcomes. Do not search for a positive result by
   adjusting loss margins, proxy severities or seeds after observing failures.
2. Define primary seed-level contrasts, practical margins, evaluation-sample
   uncertainty and a powered sample size. Selection here is a point-estimate
   pilot, not a confidence-certified recommendation procedure.
3. Add ACSIncome state/year environments as main-study external validation
   before a claim of recommendation transportability. Insurance remains the
   MSc continuity case; synthetic data provide controlled interventions.
   Real ACS natural-shift results have **not** been run by this change.
4. A third independent tabular domain can remain future scope for a narrow
   synthetic/insurance/ACS paper. It is needed now only if the desired claim
   is broad reliability across unrelated domains. Another random split of
   insurance or an injected proxy on ACS is not natural external validation.

The current evidence warrants an exploratory benchmark and a revised mechanism
hypothesis centred on clipping/optimisation. It is insufficient for a strong
novel methods-paper claim. The next protocol should explicitly compare against
the stable-feature and spurious-correlation work cited in the study document.

## Reproducibility and outputs

Each run contains the complete config/accounting manifest, per-cell metrics,
validation-selected decisions, seed summaries, plot and report. Component
analysis adds its own code/input hashes and preserves the raw training outputs.
The training manifests retain their original code hash and git state; the
component analysis was added after the first runs. Current new runs produce
components automatically. To analyse an earlier run without retraining:

```bash
python -m dp.proxy_shift_components \
  --input-dir reports/proxy_shift/synthetic_pilot \
  --output-dir reports/proxy_shift/new_component_analysis
```

* [Synthetic run](synthetic_pilot/report.md), [component analysis](synthetic_pilot/components/report.md).
* [Initial insurance run](insurance_pilot/report.md), [components](insurance_pilot/components/report.md).
* [Insurance diagnostic](insurance_training_diagnostic/report.md), [components](insurance_training_diagnostic/components/report.md).

Checks: 350 non-slow tests passed after installing the repository's declared
dependencies; five existing slow tests were deselected. The new 11 study tests
cover validation-only selection, actual-budget caps, invalid protocol rejection,
fixed preparation, empty-batch accounting, inactive-clipping agreement,
deterministic repetition and clipping/noise decomposition. No ACS feasibility
workflow or old leakage gate was changed/dispatched.
