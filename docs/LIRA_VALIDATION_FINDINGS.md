# Real-data LiRA validation — findings (protocol v1)

Executed 16 November 2026 under [`LIRA_VALIDATION_PROTOCOL.md`](LIRA_VALIDATION_PROTOCOL.md),
frozen and pushed (commit `571e384`) before any confirmatory model was trained.
Every number below is generated in
[`reports/lira_validation/v1/RESULTS.md`](../reports/lira_validation/v1/RESULTS.md)
from `summary.csv`, `per_target.csv` and `results.json`. Prepared with AI
assistance; not externally reviewed; not submitted anywhere.

## What was run

576 models on the public insurance table (`high_cost`, fixed public encoding):
16 targets and 128 shadows in each of four conditions. Each target has 669
members and 669 non-members. All 576 checkpoints passed identity validation
(protocol hash, data hash, condition, index, seeds, membership, finite
margins); none was rejected or retrained (`artifact_inventory.json`).

| Condition | Achieved ε (δ = 1e-5) | Train / held-out accuracy |
|---|---|---|
| Non-private | none | 0.991 / 0.901 |
| Clipping only (σ = 0) | none | 0.984 / 0.906 |
| DP-SGD | 7.995 | 0.923 / 0.922 |
| DP-SGD | 1.993 | 0.852 / 0.842 |

An independent RDP replay of (σ, q = 1/11, 4,400 steps) reproduces both ε
values to 1e-10.

## Primary endpoint: not met

At DP-SGD ε ≈ 8, the literature-faithful attack (logit confidence, per-example
variance, 64 OUT shadows) minus the existing baseline (raw loss, 11 OUT
shadows), paired over 16 targets:

**ΔAUC = +0.0015, 95% CI [−0.0022, +0.0052], sign-flip p = 0.38.**

No improvement is established. The interval excludes an AUC gain larger than
about 0.005 in this setting; it does not show the attacks are equivalent in
general. The secondary contrasts agree: TPR at 1% FPR −0.0013 [−0.0044, +0.0018]
at ε ≈ 8, and no contrast in any condition has an interval excluding zero.

## Detection: leakage is detected under DP-SGD

| Condition | Baseline LiRA mean AUC [95% CI] | Primary LiRA mean AUC [95% CI] | Holm p (baseline / primary) |
|---|---|---|---|
| Non-private | 0.5732 [0.5652, 0.5813] | 0.5642 [0.5527, 0.5756] | 1.7e-76 / 2.9e-59 |
| Clipping only | 0.5629 [0.5544, 0.5714] | 0.5689 [0.5598, 0.5780] | 5.5e-57 / 5.4e-68 |
| DP-SGD ε ≈ 8 | 0.5316 [0.5241, 0.5391] | 0.5330 [0.5261, 0.5400] | 1.3e-15 / 5.7e-17 |
| DP-SGD ε ≈ 2 | 0.5256 [0.5169, 0.5343] | 0.5263 [0.5189, 0.5337] | 4.6e-11 / 1.4e-11 |

Both predeclared attacks reject "no score–membership association" in every
condition. At ε ≈ 8 all 16 targets have primary AUC above 0.5 (0.510–0.561).
The loss-threshold reference is weaker everywhere (0.512 under DP-SGD).

This is small aggregate leakage from a deliberately memorising model. It is
fully compatible with the formal guarantees: (ε, δ)-DP at ε = 2 permits far
more attack advantage than AUC 0.53. It is not an ε estimate and not a
violation.

**What changed relative to the earlier ladder.** The gain in sensitivity comes
from the design — 16 targets with 669 members each, randomised membership and
a pooled test — not from the attack variant. The earlier frozen ladder's
finite-ε "fail" verdicts stand as recorded under its own rule; this study shows
that a powered design detects leakage at comparable budgets on the same task.
The two studies differ in training-set size (669 vs 856), encoding and noise
calibration, so their AUCs are not directly comparable.

## Clipping versus noise

Clipping and Poisson sampling alone leave leakage near the non-private level
(0.563–0.569 vs 0.564–0.573). The drop to 0.53 appears only when noise is
added. This is a descriptive comparison of matched conditions, not a causal
decomposition.

## Low-FPR results are weak and uncertain

Mean TPR at 1% FPR under DP-SGD is 0.011–0.015 (chance is at most 0.010), with
target-level intervals that mostly include 0.010. Per target, 1% FPR allows 6
false positives; individual TPRs range 0.004–0.028 at ε ≈ 8. No null reference
distribution was computed for this statistic, so it is descriptive. TPR at 0.1%
FPR allows zero false positives among 669 non-members and is reported only as
description (0.001–0.002 under DP-SGD). **The study does not show confident
identification of individual members under DP-SGD.**

## Null controls

Scoring each target against another target's membership gives mean AUC
0.492–0.508 across all 52 condition–attack cells. Two cells have unadjusted
p < 0.05 (loss threshold at ε ≈ 8 and ε ≈ 2, p = 0.026 and 0.042; these two
share seeds and the same label pairing), against about 2.6 expected by chance.
No predeclared LiRA control cell is below 0.15. The controls do not indicate
label leakage in the pipeline.

## Exploratory findings (hypothesis-generating only)

* **Shadow count and variance rule barely matter here.** Under DP-SGD all
  twelve LiRA configurations give mean AUC 0.518–0.534. Pooled variance with
  raw loss is the weakest at ε ≈ 2 (0.518–0.520).
* **The logit statistic saturates on memorising models.** In the non-private
  and clipping-only conditions 43% and 55% of target losses map to the clipped
  probability bound, and per-example OUT standard deviations span thirteen
  orders of magnitude. That plausibly explains why the primary attack trails
  the baseline without noise (−0.009 AUC, CI includes zero). Under DP-SGD
  nothing saturates, the OUT spread ratio (p90/p10) is 1.7–2.1, and non-member
  z-scores are close to standard normal (sd 0.95–1.01; mean +0.13 at ε ≈ 2).
* **High-risk records.** Records 806 and 1057 separate most strongly by
  membership without noise; under DP-SGD the largest gaps are an order of
  magnitude smaller and involve different records. Not tested.

## Threats to validity

1. One small public dataset, one architecture chosen to memorise. Nothing here
   generalises to other models or data.
2. Targets share a pool and a shadow set. The detection test remains valid
   (independent membership vectors), but intervals over targets understate
   uncertainty about a new dataset or shadow set.
3. Offline LiRA only; online LiRA and stronger attacks could find more.
4. The primary attack's numerical saturation without noise is a property of
   the predeclared statistic, left unchanged.
5. No attack-based ε lower bound: the design's fixed-size membership and
   within-model dependence violate the available auditors' assumptions.
6. ε is per model on encoded records. The 576-model release is not composed.
7. CPU training is not bit-reproducible across torch builds (versions are in
   `run_manifest.json`). A rerun should reproduce conclusions, not digits.
8. Locally prospective, not independently preregistered. A 20-epoch pilot on a
   different seed was seen before the freeze.

## Reproduction

```bash
python research/lira_real_data_study.py inventory --input-dir reports/lira_validation/v1 --require-complete
python research/lira_real_data_study.py analyse --input-dir reports/lira_validation/v1
python research/lira_validation_report.py --input-dir reports/lira_validation/v1
```

These regenerate every table and figure from the committed per-model
measurements in under a minute. Retraining from scratch (`train --output-dir
<fresh dir>`) took 2.85 CPU-hours here (about 70 minutes on 4 threads).

## Before any external submission

1. Human expert review of the design, the Stouffer argument and this write-up.
2. Replication on a second dataset (the frozen ACS Phase 0 is the natural one)
   and a non-memorising model.
3. An unsaturated statistic (the stored signed margins) and online LiRA,
   frozen as a new protocol.
4. A design that supports an ε lower bound (independent inclusion coins).
5. A null reference for TPR at low FPR, and enough non-members to resolve 0.1%.
6. Pinned environment and an independent rerun.
