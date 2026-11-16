# Real-data LiRA validation — prospective protocol, version 1

**Frozen 17 November 2026, before any confirmatory model was trained.**
Protocol hash (`dp.lira_study.protocol_hash()`):
`326d8849cbb7dd7952975d7c42deed711c32a77226e26c83cac7c66d3797b4f1`.
The constants in `src/dp/lira_study.py` are the protocol; the analysis refuses
measurements made under a different hash. This is a locally prospective
protocol committed and pushed before the run, not an independently timestamped
public preregistration.

This is a **new study**. It does not rerun, reinterpret or replace the frozen
noise ladder, the attack-power control or ACS Phase 0, and their conclusions
stand as recorded.

## 1. Question

Does a literature-faithful offline LiRA attack detect membership leakage in
real data more effectively than the repository's existing baseline attack,
especially under DP-SGD?

## 2. Target and data

* **Dataset:** `data/insurance.csv` (1,338 public records; SHA-256 recorded in
  every artefact). Chosen because the existing baseline attack and its
  positive controls were run on it, so old and new attacks are comparable, and
  because 576 models fit the available compute. ACS Phase 0 has its own
  frozen, not-yet-run protocol and is deliberately left untouched.
* **Task:** `high_cost` — `charges > 9382.033`. The threshold is a fixed
  protocol constant (the median over the whole public table), not fitted per
  training set.
* **Preprocessing:** row-wise fixed public encoding (`dp.lira_study.encode_fixed`):
  age, BMI and children scaled to [-1, 1] with bounds [18, 64], [15, 55],
  [0, 5]; sex and smoker as {0, 1}; region as a fixed one-hot. No statistic is
  learned from any training set, so no transform depends on membership.
* **Duplicates:** records identical in features and label are kept and counted.
  A duplicate can be a member and a non-member at once, which can only dilute
  measured leakage.

## 3. Membership design

* Every model trains on exactly 669 of the 1,338 records.
* **16 target models:** independent uniformly random halves. Members are the
  target's training records; non-members are the other 669. Membership is
  therefore randomised per record with probability 1/2.
* **128 shadow models:** 64 complementary pairs (a random half and its
  complement), drawn independently of every target. For the first K shadows
  (K even) every record is OUT of exactly K/2.
* Shadows are never targets. Shadow training sets necessarily overlap target
  training sets (all draw from one pool); that is the standard LiRA design and
  is symmetric for members and non-members. Only OUT shadows of a record are
  used to score it.
* Design seed 20261117. Model *i* uses the same initialisation and batch-order
  seed in every condition.

## 4. Models and conditions

MLP 9→128→128→64→1 with ReLU, Adam (lr 1e-3), batch 64, 400 epochs: the
deliberately memorising recipe of the earlier studies. It is a worst-case
audit target, not a recommended model.

| Condition | Clipping | Sampling | Noise |
|---|---|---|---|
| `nonprivate` | none | shuffled batches | none |
| `clip_only` | per-example, C = 1 | Poisson, q = 1/11 | σ = 0 (no DP guarantee) |
| `dp_eps8` | C = 1 | Poisson, q = 1/11 | σ solved for ε = 8 |
| `dp_eps2` | C = 1 | Poisson, q = 1/11 | σ solved for ε = 2 |

Accounting: Opacus RDP, δ = 1e-5, add/remove adjacency on encoded records. σ
is solved for the **executed** rate and step count (q = 1/11, 4,400 steps);
achieved ε is read from the engine and reported. ε is a per-model guarantee.
It does not cover the 576-model release, and it is an upper bound on
worst-case loss, not a prediction of attack success. `clip_only` separates the
effect of clipping and sampling from that of noise.

## 5. Attacks

All LiRA variants call `dp.attacks.lira_offline_attack` (PR #58).

* Factorial, 12 configurations: statistic {raw loss, logit confidence} ×
  variance {per-example, pooled} × shadows K ∈ {22, 64, 128} (11, 32, 64 OUT
  observations per record).
* **Baseline (the existing attack):** raw loss, per-example variance, K = 22.
  This mirrors the 11 OUT observations of the earlier studies.
* **Primary (literature-faithful):** logit confidence, per-example variance,
  K = 128. Carlini et al. fit per-example Gaussians on logit-scaled confidence
  and use a pooled variance only when shadows are few.
* **Reference:** Yeom loss-threshold attack.
* The pilot seen before this freeze (§9) did not change the primary choice,
  which was fixed in code beforehand.

## 6. Predeclared analysis

The unit of analysis is the **target model** (n = 16 per condition).

1. **Primary endpoint.** At `dp_eps8`: mean over targets of
   AUC(primary) − AUC(baseline), paired by target, with a 95% t-interval and an
   exact two-sided sign-flip test over the 16 targets (α = 0.05).
   *Success:* interval above zero. *Otherwise:* "no improvement established" —
   not "no difference".
2. **Secondary endpoints.** The same contrast for TPR at 1% FPR, and both
   contrasts in the other three conditions. Unadjusted, labelled secondary.
3. **Detection.** For baseline and primary, per condition: one-sided Stouffer
   combination of target AUCs against 1/2 using the exact permutation variance
   (669 vs 669), Holm-adjusted over the four conditions. A rejection means
   scores are associated with membership for some target. A non-rejection is a
   non-detection at this power, never evidence of privacy.
4. **Everything else is exploratory:** the other ten configurations, pooled
   versus per-example variance, shadow-count trends, calibration diagnostics
   and high-risk records.

**Reported for every configuration:** mean AUC and mean TPR at 1% FPR with
t-intervals over targets; per-target values with members/non-members counts;
per-target Clopper–Pearson TPR intervals (conditional on the threshold,
descriptive).

**FPR support.** With 669 non-members per target, 1% FPR allows 6 false
positives and is estimable. 0.1% FPR allows none, so TPR there has resolution
1/669 and is reported only as descriptive, never tested.

**Null controls.**
* *Mismatched membership:* each target is scored against another target's
  independent membership vector. Real scores, null labels: mean AUC must be
  near 0.5 and the Stouffer test must not reject systematically.
* *Simulation:* `tests/test_lira_study.py` checks the Stouffer test's size
  under random membership.

**Planned sensitivity** (binormal scores, 16 targets × 669/669, α = 0.05,
4,000 simulations; `dp.lira_study.detection_power`): per-target null SD of AUC
is 0.0158.

| True mean AUC | 0.500 | 0.505 | 0.5075 | 0.510 | 0.515 | 0.520 |
|---|---|---|---|---|---|---|
| P(detect) | 0.05 | 0.36 | 0.60 | 0.82 | 0.98 | 1.00 |

Power for the paired contrast depends on the unknown between-target SD of the
difference; it is reported after the fact as the interval width.

## 7. Known limitations fixed in advance

* Targets share one data pool and one shadow set, so across-target intervals
  understate uncertainty about a new dataset or a fresh shadow set. The
  detection test stays valid because target membership vectors are independent.
* One small public dataset, one architecture, one sensitive-free aggregate
  question. No subgroup analysis.
* Offline LiRA only. An attack-based ε lower bound is **not** computed:
  membership here is a fixed-size half, not independent fair coins, and
  per-record outcomes within one model are not independent trials, so the
  one-run and Clopper–Pearson auditors' assumptions do not hold.
* ε covers encoded records per model. With this protocol's fixed public
  encoding, an encoded record is a row-wise function of one raw record. That
  differs from the earlier insurance pipeline, whose fitted preprocessing has
  no raw-record guarantee; nothing here changes that earlier limitation.
* CPU training is not bit-reproducible across torch builds; versions are
  recorded.

## 8. Execution

```bash
pip install -e ".[experimental,dev]"
python research/lira_real_data_study.py train --output-dir reports/lira_validation/v1
python research/lira_real_data_study.py analyse --input-dir reports/lira_validation/v1
python research/lira_validation_report.py --input-dir reports/lira_validation/v1
```

Training is resumable: each model writes one `.npz` (signed logit margin on all
1,338 records, membership, metadata) and existing files are reused. 576 models.

## 9. Pilot (not confirmatory)

Run before the freeze on pilot seed 999, in a scratch directory, never analysed
as evidence: (a) four full-length models per condition for timing — about 10 s
(`nonprivate`) to 30 s (DP) per model on one CPU thread, about 70 minutes for
the study on 4 threads; achieved ε 7.995 and 1.993; (b) all 576 models at 20
epochs to exercise the analysis code. The pilot's attack table was seen. It
was not used to choose the primary configuration, and no protocol constant was
changed after it.

## 10. Compute plan

The full protocol (128 shadows) fits this environment and the stated local
hardware (Ryzen 7 5700X: roughly 20–25 minutes on 16 threads; the GPU is not
needed for a 27k-parameter MLP and Opacus per-sample gradients are CPU-bound
here). Nothing is deferred.
