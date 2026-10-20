# Methodological proposals — require human review before any implementation

These follow from [`STATISTICAL_REVIEW.md`](STATISTICAL_REVIEW.md). None is
implemented. Each would be a **new, prospectively frozen protocol or
amendment**, written before outcomes are seen. None may reinterpret or rerun a
frozen confirmatory study under a changed rule. The owner (and ideally an
external statistician) should decide which, if any, to pursue.

## P1. A powered confirmatory test of residual aggregate leakage at finite ε

*Motivation.* The frozen ladder rule has 4–32% power at the observed effect
sizes. An exploratory pooled-seed analysis suggests small residual leakage at
every finite ε.

*Proposal.* Freeze, before running, a new design on fresh target seeds with:
one primary point (for example ε = 8), the seed as the unit, a primary
statistic of mean seed AUC minus 0.5 with a one-sided test calibrated by
within-seed membership permutation and combined across seeds, and a seed count
chosen from `dp.ladder_rule_power`-style simulation for a stated minimum
effect (for example AUC 0.53 at 80% power). Report the estimate and interval,
not only a binary verdict.

*Cost/risk.* About 33 DP-SGD fits per seed. A positive result would show that
leakage is detectable at that ε for this deliberately memorising target only;
it would not bound ε or generalise to other models.

## P2. Stronger, literature-faithful attack for future runs

Use 64–128 shadows, logit-scaled confidence, and global (pooled) variance
offline LiRA when per-example OUT counts are small; report TPR at low FPR with
exact binomial intervals. Freeze it as a new primary for *new* runs only. The
ACS Phase 0 frozen raw-loss online primary stays unchanged unless a dated
pre-result amendment says otherwise.

## P3. Isolate noise from clipping and sampling in the ladder endpoint

Add a clipped, Poisson-sampled, zero-noise endpoint (σ = 0 with C = 1) so the
non-private-to-ε=32 bracket compares like with like. Without it, the bracket
cannot be attributed to noise.

## P4. A valid equal-leakage (subgroup) test

Candidates, each needing calibration and power checks by simulation under
*non-zero* common leakage:

* seed-level paired contrast of group AUCs (or TPR at a fixed FPR) with
  sign-flip or t inference over many target seeds;
* a parametric bootstrap that fits a common membership-signal model across
  groups and resamples scores under it.

Membership-label permutation must not be reused for this purpose.

## P5. Pin the numerical environment for research workflows

Pin `torch`, `opacus`, `numpy` and Python in research workflows (a lockfile or
exact versions) and keep the new `runtime` record. Treat cross-environment
variation as a source of uncertainty, or report it explicitly, when a decision
sits near a threshold.

## P6. End-to-end raw-record privacy for insurance (only if claimed)

Freeze public scaling bounds, the categorical schema and a public `high_cost`
threshold before sampling private records, or account for private
preprocessing and compose it with training. Otherwise keep every insurance
claim explicitly conditional on prepared records.

## P7. Per-example follow-up of the seed-42 anomaly (exploratory)

Inspect the highest-scoring members in seed 42 across ε (features, label,
OUT-loss dispersion), to decide whether the persistent signal comes from a few
vulnerable records or from the calibration step. Label any finding as
exploratory and hypothesis-generating for P1.
