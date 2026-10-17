# Independent correctness review of PR #55 — 2026-10-17

Reviewed original head `5547134b4fbc34993b8c00b1bc4a190251d7eed8` and
brought the follow-up PR #54 fixes into this stacked branch.

## Implementation and mathematical assessment

- ACS fixed representations are row-wise, with public numeric bounds,
  categorical schemas and fixed modulo/digit encodings. Source cohorts are
  household-disjoint. The sampler truncates the final assigned household to
  meet row budgets; this is not household-weighted or survey-design inference.
- Model initialization, Poisson sampling generators, expected-batch
  normalization and noise generators are matched across controls. Empty
  Poisson batches still step the accountant. The decomposition is an
  algebraic matched-control comparison, not separate causal effects.
- AUTO-S matches Bu et al., NeurIPS 2023, Eq. 4.1 / Appendix K: global
  per-example gradient normalization with stability .01 and unit sensitivity.
  Its direction/norm and direct update tests pass. Its frozen LR/schedule
  does not establish performance relative to optimally tuned AUTO-S.
- The AUC influence calculation uses midranks and paired covariance; cluster
  sums preserve household dependence in the sandwich approximation. The
  confidence family covers both endpoints, all validation environments,
  candidates and selectable references. The approximation is not calibrated
  in the rare-label simulation. No operational certification follows.
- The concentration sensitivity counts all rank pairs touching a household
  once, with a doubled radius for the gap and explicit within-household-pair
  bias. Its interpretation requires the stated conditional marginal score
  laws and independent score-generating clusters. ACS survey observations
  do not establish those assumptions.
- Choices are written before external scoring in the runner. Saved files
  alone cannot independently prove that execution order. Seed-level paired
  intervals are conditional on fixed archives/states, not uncertainty across
  new populations. Degenerate t intervals are marked unsupported; the
  supplementary exact discordance intervals avoid treating zero variance
  as certainty. Supplementary contrasts are not multiplicity-adjusted claims.

## Confirmed verification and CI defects fixed

The old follow-up checker iterated only over environments present in
`decisions.csv`, then reindexed to frozen keys. An entirely missing external
state, or extra case keys discarded by reindexing, could escape that check.
It also trusted derived success tables. The new read-only validator requires
an exact outer-joined case/environment set, checks frozen choices, recomputes
failure flags from external metrics, and reconstructs case/coverage tables.
Tests deliberately remove an entire state, append an unexpected seed, flip
an outcome and corrupt a summary. All four complete saved runs are checked.
The old follow-up script now invokes the stronger validator first.

Original replication launch sources are recovered from committed revision
`3871f7c5ddf0e1303ac5a93fa999fbe5a0c61332`, and every recorded source hash is
checked. Current AUTO-S sources match their launch hashes. Historical hashes
are preserved rather than rewritten to pretend the later comparator code
was used for the original 840 fits.

The repository-wide workflow previously ran only for PRs targeting `main`.
PR #55 targets the #54 branch, so only study-specific CI covered it. The full
workflow now includes that stacked target. CI fetches history for historical
source verification, and the study workflow explicitly validates saved ACS
artifacts. A successful smoke workflow remains distinct from a real ACS rerun.

## Research conclusions and release boundary

Saved evidence contains 420 + 420 original and 200 + 200 supplementary fits.
Recomputed decision summaries match the tracked tables. Both original joint
source-only and shift-aware rules succeed in 20/20 cases per representation;
that does not isolate shift-aware benefit. The supplementary digit result's
one-case gain has a paired interval containing zero. AUTO-S's fixed-recipe
result is a limited comparison. No claim of novelty, reliably calibrated
sandwich certification, unseen-state generalization, or tuned superiority
is supported.

Per-candidate RDP accounting is for prepared-record training, with public
research data and non-secure seeded RNG. It does not compose the search,
ordinary controls, training diagnostics, evaluation or release, and does not
establish household DP. A failed attack is not a privacy guarantee.

## Reproduction and remaining gates

Run `python -m dp.artifact_validation` from a full checkout to validate all
four saved runs without rewriting results. These checks establish internal
consistency, not provenance attestation or a full 1,240-fit reproduction.
The original final-head CI study job 110183148635 passed its invariant tests
and all synthetic smoke stages; its logs were inspected during this review.
Local full-suite and fresh calibration outcomes are recorded in the PR review.

Before a submission: clean independent reproduction from verified Census
archives; human author/scientific review of assumptions and contribution;
remaining primary-source literature followups; final anonymous packaging.
The historical insurance PDF remains superseded. The ACS manuscript/PDF's
scientific conclusions were not strengthened by this review. No merge,
submission, real Phase 0 run or new ACS tuning experiment was performed.

## Final CI compatibility correction — 17 October 2026

Final-head CI exposed a pandas 3 compatibility bug in the new validator:
concatenated all-abstention summaries retain `None`, whereas CSV parses them
as `NaN`. Pandas 2's testing helper accepted them as equivalent; pandas 3
rejected them. The checker now requires identical missing-value masks and
compares every nonmissing value at the existing numeric tolerance. Regression
tests cover None/NaN/pd.NA and reject both a changed null position and a
changed numeric value. No saved experiment or scientific result is changed.
PR #54 passed every final-head workflow and was merged first. PR #55 is
retargeted to main; its final corrected head must pass CI before merging.
