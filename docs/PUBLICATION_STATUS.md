# Publication correctness status — 2026-09-30

The historical PDF in `paper/` is superseded. Its proxy-heavy smoker task and
broad privacy/utility conclusions do not describe the corrected benchmark.
There is no tracked editable manuscript source. This repository is not yet a
submission-ready paper, and these corrections do not establish novelty.

## Privacy boundary

Opacus accounts DP-SGD on prepared records. The insurance pipeline learns
scaling statistics, categorical vocabularies and the high-cost label threshold
from training data without a privacy budget. A raw-record change can therefore
alter multiple prepared records. Holding that preparation fixed defines a
different, conditional privacy claim; train-only fitting does not repair this.
The insurance shadows also share the target-fitted transform, so their attack
does not audit an independently refitted raw-data pipeline.

For an end-to-end claim, freeze a public task threshold, feature schema and
scaling bounds before sampling private records, or account for private
preparation and compose it with training. Predictions, evaluation metrics and
research logs need their own release analysis if their underlying data are
private. The bounded numeric-noise helper preserves categorical fields, row
count and order: it protects numeric fields conditional on those unchanged
fields and fixed public bounds, not a full insurance record.

ACS Phase 0 instead uses a fixed public universe transformed before membership
randomisation. Preserve that conditional game; do not replace it with an
unaccounted target-fitted transform.

## Inference corrections

Within-group membership permutation tests the absence of score-membership
association. It destroys aggregate leakage and does **not** simulate equal
subgroup leakage when both groups leak. Signed contrasts and bootstrap
intervals remain descriptive. The control and ladder classifiers now return
an inconclusive disparity verdict even when these association p-values are
small. Historical supported/unsupported subgroup verdicts in generated
artifacts are superseded by this correction; neither equality nor disparity
has been established. Aggregate detection decisions remain applicable.

The separate, model-level paired recipe-contrast test is unchanged. It answers
a between-recipe redistribution question under its own sign-exchangeability
assumptions; it does not validate the old within-recipe subgroup test.

The one-run auditor now uses the approximate-DP correction in
[Steinke, Nasr and Jagielski, Corollary 5.4 and Appendix D](https://arxiv.org/abs/2305.08846).
It counts all independently randomised canaries before abstention. Crediting
`r * delta` free correct guesses was not this bound. Positive delta requires
the total candidate count in the count-only API; score and scalar wrappers
provide it. The published examples with and without abstention are regression
tests. This is a statistical confidence bound for the specified canary game,
not evidence of privacy from a failed attack. Duplicated canaries test group
privacy, not a single-record claim.

## Completed insurance ladder

The full manual run completed successfully on 2026-08-21:
[Actions run 32441370755](https://github.com/Ghobi-A/dp-insurance/actions/runs/32441370755),
commit `d8550bf89465ac13f23776aec01f5e345289f515`, artifact `9433303328`
(`detectability-noise-ladder`). The downloaded aggregate JSON was inspected;
this table records the existing run, not a new experiment.

| Requested epsilon | Mean achieved epsilon | Mean primary offline AUC | Aggregate rule |
|---|---:|---:|---|
| Non-private | — | 0.5745 | Pass |
| 32 | 31.0307 | 0.5390 | Fail |
| 16 | 15.5427 | 0.5439 | Fail |
| 8 | 7.7749 | 0.5503 | Fail |
| 4 | 3.8939 | 0.5431 | Fail |
| 2 | 1.9432 | 0.5322 | Fail |

The frozen rule requires mean AUC >= 0.55, every seed AUC > 0.5 and
Holm-adjusted aggregate p < 0.05 on at least two of three seeds. Epsilon 8
meets the mean criterion but rejects on only one seed. No finite point passes;
the insurance iso-epsilon follow-up must not run under this design.
Non-detection does not demonstrate privacy, subgroup equality, or absence of
leakage. The non-private endpoint also changes clipping and sampling relative
to finite points, so it does not isolate a noise-only transition.

## Work required before a paper

1. Recover or create editable manuscript and bibliography sources and replace
   the historical PDF with claims matching the current evidence.
2. Specify one privacy adjacency, release boundary and attack estimand. For
   raw-record claims, implement public/fixed or accounted private preparation.
3. Validate attack calibration and power independently of the confirmatory
   data. Any primary attack change must be an explicit pre-result amendment;
   this PR preserves ACS's frozen raw-loss online LiRA primary.
4. If pursuing subgroup inference, specify an equal-leakage test valid under
   nonzero aggregate leakage and validate its calibration and power. Retain
   the seed-level unit for trained-model recipe comparisons.
5. Run the already frozen ACS Phase 0 once when ready. A failure closes that
   natural-leakage branch; this PR does not dispatch it or retune the gate.
6. Establish the contribution against existing disparate membership-inference
   and DP utility literature before claiming a novel result. A benchmark or
   power-limited negative result may be useful, but novelty is not demonstrated
   by code completeness or by attaching a new title.
