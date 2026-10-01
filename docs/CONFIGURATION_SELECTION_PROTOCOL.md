# Configuration-selection extension: frozen external protocol

Dated 1 October 2026. Read [the literature assessment](CONFIGURATION_SELECTION_LITERATURE.md)
first: this is a candidate empirical contribution, not confirmed novelty.
The configs are frozen before external scoring. A Git commit is a prospective
execution record, not an independently registered protocol.

## Independent external split

ACSIncome 2017: CA source; OR/WA validation; **CO/UT held-out tests**. This uses
an unused year and two states absent from the 2018 pilot. It is a new
geographic replication in another year, not a pure temporal-transfer test.
The target remains nominal income >$50,000. Survey weights do not enter the
classifier. Whole source households are partitioned disjointly.

Twenty fresh seeds 200–219 resample the fixed archives. Seeds are training/cohort
replications, not independent states or independent survey populations. A
minimum of 20 seeds supports a bounded comparison; the binary successful-yield
endpoint has not been prospectively powered from the older AUC contrast.
No failure-rate precision or guaranteed achieved power is promised.

## Fixed comparison

Privacy candidates epsilon 2/8, delta 1e-5; norms 1/5. Three recipes:
16-unit tanh MLP / 50 epochs / LR .1; 32-unit tanh MLP / 100 epochs / LR .05;
DP logistic regression / 100 epochs / LR .1. All use matched Poisson streams,
3000 training rows, 2000 evaluation rows, batch 256. Candidate search is
small and deliberately fixed; this is not a convergence or exhaustive HPO claim.

One ordinary model per recipe is selected by **source validation AUC only**
as the common reference, before external loading/scoring. Comparisons to
that reference cannot pass simply because a weak matched ordinary model was
used. Optimization-reference, clipping and noise gaps sum exactly to the
reported total gap. Components need not be positive and are not causal effects.

Policies: epsilon-only fixed recipe/norm, joint clip fixed recipe, and joint
recipe/architecture, each under source-only or natural-shift validation;
proxy-stress joint search and logistic-only shift search are secondary.
The union-bound family includes all ordinary reference choices as well as
all candidate/environment gap/floor endpoints. Same family for every policy
supports a controlled comparison but makes restricted policies conservative.

The rule chooses the smallest achieved epsilon among passing candidates,
then the best worst-environment validation lower AUC (point AUC for unbounded
rules), then deterministic ID. Actual budgets and candidate counts are logged.
No held-out metric participates in that choice.

AUC-loss tolerance .03 and absolute floor .70 stay fixed. Safety margins
0/.005/.01/.02 tighten the constraints to trace descriptive coverage/failure
curves. Point and asymptotic simultaneous-bound rules are both retained.

**Primary:** bounded joint-recipe natural-shift minus bounded epsilon-only
source rule, safety margin 0, successful recommendation yield over all cases.
A case succeeds only if it selects and both CO/UT satisfy both utility targets.
Abstention contributes zero success. Paired seed-mean 95% t intervals are
approximate; small-n/degenerate intervals are explicitly unsupported.
Ten percentage points is the proposed practical yield difference. Other
factorial contrasts and the margin curves are descriptive, not extra primary
hypothesis tests. Report coverage and failure jointly, even for the primary.

## Representation and calibration

Primary encoding: earlier fixed public modulo hash. Prespecified sensitivity:
public decimal-digit one-hot codes for occupation/birthplace, with explicit
out-of-range bucket. This avoids within-range modulo collisions but imposes
a different compositional inductive bias; it is not a causal hash-collision
experiment or the standard Folktables representation.

Calibration covers the complete candidate/reference/environment bound family
under varied class imbalance, household sizes and within-household label/score
dependence. Known Gaussian marginal AUCs supply a simultaneous coverage check;
this does not establish ACS survey-design validity or finite-sample coverage.

## Run and interpret

```bash
python -m dp.configuration_selection --config research/configuration_selection/acs_replication.json --output-dir reports/configuration_selection/acs_replication
python -m dp.configuration_selection --config research/configuration_selection/acs_digits.json --output-dir reports/configuration_selection/acs_digits
python -m dp.selection_calibration --output reports/configuration_selection/calibration.json
```

Raw records stay outside Git. Each run retains launch code hashes, input
archive hashes, accounting, protocol hash, decisions frozen on disk before
external loading, metrics and seed-level cases. Output paths cannot overwrite
existing evidence. Partial runs have no completed manifest.

All data/validation/research outputs are public. The privacy cap describes
each candidate training model: search and ordinary controls are not composed
into a private release. Fixed public representation prevents a learned private
scaler; household partitioning does not provide household DP.

Before submission: inspect these outcomes against the literature gate, add
broader training/adaptive-clipping comparators if needed, and seek expert
review. No outcome justifies asserting a first-ever method.
