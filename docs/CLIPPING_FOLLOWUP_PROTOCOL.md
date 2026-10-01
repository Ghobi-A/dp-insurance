# Supplementary clipping comparator and calibration response — 1 October 2026

This amendment follows inspection of the 2017 ACS outcomes and the 88%
calibration failure. It is not a new primary confirmatory test, an independent
domain, or a remedy retroactively applied to the frozen original decisions.
All original evidence remains available. New choices are frozen before each
seed's external scoring, but the same state/year outcomes are already known.

## Published comparator, frozen recipe

Implement Bu et al. (NeurIPS 2023) AUTO-S Eq. 4.1 and Appendix K:
all-parameter per-example gradients become `g / (||g|| + .01)`, with unit
sensitivity. This is the published rule, not our algorithm. Opacus retains
Poisson sampling, expected-batch normalization, Gaussian noise and RDP
accounting. The following ordinary unit clipping is inactive for normalized
gradients. No library installation is modified.

Compare standard norms 1/5 and AUTO-S at epsilon 2/8. All share the base
16-unit tanh MLP, 50 epochs, LR .1, batch 256, train 3000, eval 2000,
delta 1e-5. One fixed matched ordinary model and three noise-free transformed
controls give ten fits per seed. Seeds 300–319 are new; hashed and digit
representations are both retained, giving 400 fits. Keep the 2017
CA/OR/WA/CO/UT split and the gap .03 / floor .70 constraints unchanged.
Do not tune the published rule's stability, LR or epoch count after outcomes.

Policies: fixed norm 1, fixed norm 5, standard joint norm search, AUTO-S only,
and combined search, each with source-only or source+OR/WA validation. Compare
point, original asymptotic sandwich and conservative concentration rules,
with fixed margins 0/.01. Family is 2*6*3*1 = 36 for all rules, including
restricted ones. This differs from the original 360-endpoint three-recipe
study; compare algorithms within this supplementary run, not its coverage
directly to the original run. External success requires both CO and UT to pass;
abstention is zero successful yield. Paired exact yield contrasts are
descriptive supplementary analyses with no multiplicity-adjusted claims.

This is a controlled training-rule comparison, not a tuned performance contest,
reproduction of the original vision/NLP experiments, or private-HPO system.
Panda et al.'s private HPO would require its adaptive trial schedule and
composed private selection accounting; a simplified fixed grid is not that
published algorithm. Consequently no private-HPO superiority claim is made.

## Calibration response and its assumptions

Fresh simulation seed 20261002, 200 repetitions in each of the original three
Gaussian scenarios, all 360 endpoints. Retain the sandwich rule unchanged.
Compare a standard McDiarmid bounded-differences sensitivity, not a new
inference method and not a fitted inflation of the old standard errors.

Conditional on labels and household membership, let P/N be class counts and
p_h/n_h the counts in household h. Replacing its scores can change AUC by at
most `c_h = p_h/P + n_h/N - p_h*n_h/(P*N)`. The candidate AUC kernel has range
[0,1]; the paired-gap kernel has range [-1,1]. Thus, for each one-sided
endpoint at alpha/family, the conditional-expectation radii are
`r = sqrt(.5*sum(c_h^2)*log(family/alpha))` and `2r`.

For a common class-conditional score law across households **after conditioning
on the entire label vector**, cross-household AUC has a common marginal target.
Within-household positive/negative pairs can have different expectations. With
`w = sum(p_h*n_h)/(P*N)`, their worst-case bias is at most w for candidate AUC
and 2w for the gap. Use lower AUC `AUC-r-w` and upper gap `gap+2r+2w`, truncated
only at their parameter-domain endpoints. The union bound needs no independent
candidate or environment errors. Independent score-generating households,
fixed models independent of validation score randomness, and the common
conditional marginal law are material assumptions. The Gaussian calibration
DGP satisfies them because label generation and score noise are independent.

Real ACS survey data need not satisfy these conditional laws or independent
household sampling. The ACS concentration rule is therefore explicitly an
**assumption-dependent sensitivity**, not a survey, finite-population or
unseen-state certificate. Wider bounds may abstain everywhere. Report that
cost rather than relax the margin until a useful result appears. Successful
Gaussian calibration does not validate real-data assumptions.

Run configs and code are committed before launching these supplementary runs.
Do not overwrite old evidence. Manifests retain data/code/protocol hashes,
runtime versions and all per-model accounting. This narrows the manuscript to
descriptive decision auditing; calibrated operational certification remains
outside its supported contribution.

Sources: [AUTO-S paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/8249b30d877c91611fd8c7aa6ac2b5fe-Paper-Conference.pdf),
[bounded differences background](https://arxiv.org/abs/1212.5796).
