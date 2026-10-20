# Independent statistical review — 20 October 2026

Scope: privacy adjacency and release boundaries, membership-inference
implementations and inference, power, multiplicity and interpretation across
the current protocols, code, tests, saved artefacts and the ACS decision-audit
manuscript (`paper/submission/review_manuscript.tex`, `paper/proxy_shift_manuscript.md`).
The historical PDF in `paper/` was treated as historical only.

This review was prepared with AI assistance (Claude Code) at the repository
owner's request. It is not an external human expert review, and it does not
replace the statistician/ML-researcher check listed in
[`SUBMISSION_REVIEW.md`](SUBMISSION_REVIEW.md). No frozen confirmatory protocol
was retuned, no new confirmatory experiment was run, and nothing here describes
the paper as submitted.

## 1. What the evidence supports

| Claim | Verdict | Basis |
|---|---|---|
| Joint configuration search restores successful ACS recommendations where a restricted ε-only rule fails (20/20 vs 0/20 hashed, 20/20 vs 1/20 digits) | **Supported, descriptive** | Recomputed from saved artefacts by `python -m dp.artifact_validation`; search sizes differ, so no efficiency claim |
| Shift-aware validation improves external success | **Not supported** | Source-only joint search also 20/20; supplementary digit gain is one case, interval includes zero |
| Fresh-seed synthetic primary contrast is positive (0.008275 AUC, 95% CI [0.004092, 0.012458]) | **Supported for the specified generator** | Reproduced exactly from `reports/publication_stage/confirmation/metrics.csv`; interval straddles the 0.01 practical margin |
| Sandwich bounds certify recommendations | **Not supported** | Rare-label simultaneous coverage 173/199 (0.869, exact CI [0.814, 0.913]) |
| Per-candidate DP-SGD meets its ε cap | **Supported, conditional** | Opacus RDP under add/remove adjacency on *prepared* records; see §2 |
| End-to-end raw-record DP for insurance models | **Not supported** | Data-dependent preprocessing and label threshold are outside the accountant |
| Non-private memorising targets leak detectably (offline LiRA AUC 0.558–0.599) | **Supported** | Positive controls; permutation null rejected on 3/3 seeds |
| DP-SGD at ε ≤ 32 removes detectable aggregate leakage on insurance | **Not supported** | The frozen rule fails every finite point, but its power at the observed effect sizes is 4–32%; an exploratory pooled-seed test rejects no-association at every finite ε (§4) |
| Subgroup (sex) disparity or equality of leakage | **Neither established** | No equal-leakage test exists; membership permutation tests association only |
| Fairness gap grows with ε on the smoker task | **Not supported** | Paired 5-seed difference 0.15, p ≈ 0.09; classifier near chance. README corrected |
| Novel method | **Not claimed, not supported** | Consistent with `SUBMISSION_REVIEW.md` |

## 2. Privacy adjacency and release boundaries

**What Opacus accounts.** Each DP-SGD fit is (ε, δ)-DP with respect to add/remove
of one *prepared* tuple `(T_D(x), 1[charges > m_D])`, where `T_D` is the scaler
and one-hot vocabulary fitted on the training partition `D` and `m_D` is the
training median of `charges`. The guarantee holds conditionally on `T_D` and `m_D`
being fixed public constants.

**What raw-record adjacency would require.** Adding or removing one raw record
changes `T_D` (every standardised feature of every record moves) and can move
`m_D` to the adjacent order statistic (flipping at most one label). The two
prepared datasets then differ in essentially all `n` rows, so group privacy
yields no useful bound: the accounted ε says nothing finite about raw-record
privacy of the insurance pipeline. Train-only fitting prevents evaluation
leakage; it does not create privacy. The threat model's adversary is also given
the fitted preprocessing, which is itself a function of private data.

**What is released without accounting.** Predictions, evaluation metrics,
shadow-model diagnostics, research logs, repeated fits across seeds and
ε levels, and any selection among candidates. None is composed.

**ACS.** Fixed public bounds, schemas and encodings make prepared-record and
raw-row adjacency coincide for one fit. Household-disjoint partitions do not
give household-level DP, and the many-candidate selection is not accounted.

**Executed ε.** In the ladder the noise multiplier was solved with
`q = 64/856` and `epochs/q` steps, but Opacus's Poisson loader ran at
`q = 1/14` for `14 × 400` steps. The achieved ε read from the engine
(31.03, 15.54, 7.77, 3.89, 1.94) is correct for what ran and was reproduced
exactly here; it is about 3% below each request. The secondary
`accountant_epsilon` field replayed the calibration rate and therefore
*overstated* ε (32.96 for the ε=32 point). Fixed (§8).

## 3. Membership-inference implementations

* `lira_offline_attack`: per-example Gaussian on raw BCE from OUT shadows,
  one-sided score `-z`. Correct as implemented. With 32 shadows and
  856/1338 inclusion, each example has only 11 OUT observations, so the
  per-example σ is noisy; Carlini et al. recommend a global variance in this
  regime and fit on logit-scaled confidence. The attack is therefore weaker
  than the reference attack. The committed online variant has higher AUC but
  *lower* TPR@1% than offline in the power control, consistent with poorly
  estimated per-example variances.
* The membership game is valid for a permutation null: the `high_cost` split
  is a simple random split, so under no leakage member and non-member scores
  are exchangeable. Cohorts are equalised by sex × membership; permutation is
  within sex.
* `stratified_membership_permutation_test` tests *no association between score
  and membership*. It is a valid test of aggregate detection. It is not a test
  of equal subgroup leakage (correctly stated in code and status docs).
* Bootstrap percentile intervals on TPR at 0.1% and 1% FPR are unreliable: with
  about 256 non-members, FPR ≤ 0.1% means zero false positives, and per-group
  resolution is 1/128. Treat them as descriptive.
* One-run auditor (`audit.py`): checked against Steinke, Nasr and Jagielski,
  Corollary 5.4 and Appendix D. The δ correction
  `2mδ·max_i (f(v−i) − f(v))/i` is implemented via cumulative pmf mass, with
  the maximum correctly restricted to `i ≤ v`. Guess budgets are fixed
  independently of the inclusion bits. No defect found.
* Analytic Gaussian calibration: checked numerically at four (ε, δ, Δ)
  settings; δ(σ) matches the target to 1e-12.

## 4. Power and detection sensitivity

The independent simulation (`python -m dp.ladder_rule_power`, results in
`reports/statistical_review/ladder_rule_power.json`) reimplements the frozen
aggregate rule under an equal-variance binormal score model with 256 members
and 256 non-members per seed, three seeds and Holm over six points. The
reimplementation reproduces all six recorded ladder decisions and agrees with
`aggregate_detected` on 300 random inputs (tests).

| True per-seed AUC | 0.50 | 0.52 | 0.53 | 0.54 | 0.55 | 0.56 | 0.57 | 0.58 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|
| P(frozen rule detects), τ=0 | 0.000 | 0.003 | 0.037 | 0.115 | 0.317 | 0.616 | 0.847 | 0.962 | 1.000 |
| P(frozen rule detects), τ=0.02 | 0.001 | 0.019 | 0.064 | 0.162 | 0.332 | 0.549 | 0.760 | 0.890 | 0.986 |

(4,000 simulations per cell; τ is between-seed SD of true AUC.)

* The rule is extremely conservative: its size under the null is about 0, and
  it reaches 80% power only near AUC 0.567. That is above the non-private
  ladder's own seed 44 (0.558).
* Observed finite-ε means (0.532–0.550) fall where the rule detects 4–32% of the
  time. **"Fail" is the expected outcome even if true leakage were AUC ≈ 0.54
  at every finite ε.**
* *Exploratory, post hoc, not preregistered:* the three seeds are independent,
  while points within a seed are not (Spearman correlation of offline scores
  between finite points 0.14–0.94). Pooling seeds per point by Stouffer z,
  Holm over the five finite points, gives adjusted p = 0.008, 0.006, 0.002,
  0.006 and 0.014 for ε = 32, 16, 8, 4 and 2 (Fisher: 0.014–0.025;
  `reports/statistical_review/ladder_pooled_exploratory.json`). Most of this
  comes from seed 42, whose offline AUC is 0.547–0.592 at every point despite
  zero or negative generalisation gaps. Seeds 43/44 are positive at 9 of 10
  finite points but individually non-significant. This rejects "no seed's scores
  are associated with membership". It is not a pooled effect size and it cannot
  change the frozen decision. It does mean the defensible reading is
  *small residual aggregate leakage is plausible at every tested finite ε; the
  frozen rule was not powered to detect it*, not *no leakage at finite ε*.
* Power-control interpretation: detecting AUC ≈ 0.57–0.60 on non-private
  targets does not establish power against DP-sized effects. The attack-power
  report previously said the DP spike's null was "informative about the DP-SGD
  configurations"; corrected (§8).

## 5. Leakage, invalid comparisons, multiplicity, pseudoreplication, deviations

* **Confounded endpoint.** The ladder's non-private point uses no clipping and
  shuffled batches; finite points use clipping C=1 and Poisson sampling. The
  frozen frontier text "the empirical transition lies between non-private and
  ε=32" therefore brackets a change in clipping, sampling *and* noise, measured
  by a low-power rule. Present it only with both caveats.
* **Pseudoreplication.** Five finite points within a seed share split, cohort,
  shadow schedule and seeds, so they are not five confirmations. The
  `gap_associations` bootstrap pools 18 correlated seed-point observations and
  resamples within seeds of six; its intervals are descriptive only. In the ACS
  study, seeds on a fixed archive split are algorithm/cohort replicates
  (correctly disclosed), and two external states share one model.
* **Multiplicity.** The ladder's Holm family (six points within seed) is
  appropriate for the frozen decision. Supplementary ACS contrasts are
  unadjusted and correctly labelled descriptive. No issue found in the primary.
* **Numerical reproducibility.** Workflows install torch unpinned. Retraining
  seed 42's non-private point under torch 2.14.1/Python 3.13 reproduced every
  design hash but moved offline AUC from 0.5859 to 0.6078. That is about 0.9 null
  SDs, the same order as the ε=8 margin to the 0.55 threshold.
* **Artefact retention.** Per-seed ladder evidence existed only in a GitHub
  Actions artifact that expires; the repository had only point means. Archived
  with checksums (§8).
* **Protocol deviations.** None found in the executed confirmatory runs. The
  ε-calibration mismatch (§2) is an implementation detail of the frozen run,
  disclosed via achieved ε; it is not a retuning.

## 6. Negative results and subgroup comparisons

* Ladder: non-detection is correctly not called privacy in the status docs. It
  should also not be called absence of detectable leakage (§4).
* Spike (`reports/spike/CONCLUSION.md`): "a null for subgroup privacy
  redistribution" rests on bootstrap intervals including zero and an unstable
  direction. That is absence of evidence, at a resolution of one member per
  cell. The methodological note already says the spike could not support a
  subgroup claim; it also cannot support equivalence.
* Attack-power report: the regenerated report no longer renders historical
  "SUBGROUP DISPARITY UNSUPPORTED" verdicts. Those labels came from
  association p-values.
* ACS manuscript: the negative findings (shift-aware search adds no yield;
  sandwich undercoverage; AUTO-S abstention under one schedule) are reported
  with appropriate scope. The primary interval straddling 0.01 is described
  correctly as neither above-margin nor equivalent.

## 7. Reproductions

| Target | Command | Outcome |
|---|---|---|
| Full test suite | `python -m pytest -q` | 396 passed before changes; 417 after (including slow tests) |
| ACS/AUTO-S saved artefacts (1,240 fits) | `python -m dp.artifact_validation` | All four runs consistent (420/420/200/200 fits) |
| Primary synthetic contrast | paired t on `test`/`proxy_0` `total_interaction`, norm 1 − norm 5 | 0.008275 [0.004092, 0.012458], p = 0.000556: exact. Wilcoxon p = 3.6e-5; 18/20 positive |
| Discordance intervals | `dp.selection_inference.exact_paired_yield` | [0.6065, 1.0000] and [0.5239, 0.9994]: exact |
| Calibration exact CIs | Clopper–Pearson on 173/199, 191/200, 194/200 | Matches manuscript |
| Ladder achieved ε | Opacus RDP replay at executed q = 1/14, T = 5600 | 31.0307, 15.5427, 7.7749, 3.8939, 1.9432: exact |
| Ladder decisions | `python research/detectability_noise_ladder.py aggregate --input-dir <downloaded points> --output-dir <fresh>` | All six decisions and the frontier text reproduced |
| Ladder non-private, seed 42, retrain | `python research/detectability_noise_ladder.py point --seeds 42 --output-dir <fresh>` | Hashes identical; AUC 0.6078 vs 0.5859 (environment difference, §5) |
| Ladder ε=8, seed 42, retrain | `python research/detectability_noise_ladder.py point --epsilon 8 --seeds 42 --output-dir <fresh>` | See PR description |

## 8. Severity-ranked findings

### Verified defects (fixed in this branch)

1. **High — ladder evidence at risk and environment-dependent.** Per-seed and
   per-example ladder results were held only in an expiring Actions artifact,
   and artefacts did not record library versions. *Fix:* archived the aggregate
   artefact under `reports/detectability_noise_ladder/` with SHA-256
   provenance and recovered CI versions. Ladder point artefacts now record a
   `runtime` block. Test: `test_runtime_versions_record_numerical_environment`.
2. **Medium — attack-power report rendered superseded subgroup verdicts and
   overstated what the positive control implies for DP.** *Fix:*
   `reported_subgroup_verdict` maps historical SUPPORTED/UNSUPPORTED labels to
   INCONCLUSIVE with the superseded label named. Section 2/3 wording no longer
   implies a disparity test or DP-level power. Added `--render-only`, and the
   committed report was regenerated from unchanged JSON. Tests:
   `test_historical_subgroup_labels_are_reported_as_inconclusive`,
   `test_render_only_rewrites_markdown_from_saved_json`, updated report test.
3. **Medium — README fairness trend overstated.** Reworded with the paired
   five-seed result.
4. **Low — ladder `accountant_epsilon` replayed the wrong sampling rate.**
   *Fix:* record executed `sample_rate`/`steps` and replay those; calibration and
   executed rates are both stored. Frozen noise calibration unchanged. Test:
   `test_accountant_cross_check_replays_the_executed_mechanism`.
5. **Low — stale test counts in status docs.** Replaced with pointers.

### Interpretation risks (documented, frozen outputs unchanged)

6. **High — ladder "Fail" read as no leakage at finite ε** (§4). The frozen
   decision stands, and the iso-ε follow-up must still not run under this design.
7. **Medium — frontier bracket confounds clipping/sampling with noise** (§5).
8. **Medium — attack weaker than the reference LiRA** (11 OUT shadows,
   per-example σ, raw loss) (§3).
9. **Low — percentile bootstrap at extreme FPR; pooled gap-association
   intervals** (§3, §5).

### Unresolved questions

* Why does seed 42 show offline AUC 0.55–0.59 at every ε with a zero or negative
  generalisation gap and *negative* loss-threshold AUC (0.45 at ε=2), while
  seeds 43/44 do not? Candidates: a few highly vulnerable records in that
  split, interaction with the difficulty calibration, or chance. A per-example
  inspection of the top-scoring members is the next step.
* Does residual aggregate leakage at ε ≤ 8 replicate in a fresh, prospectively
  powered design?
* What equal-leakage test is valid under non-zero aggregate leakage with seeds
  as units?

Methodological proposals that need human decision are listed separately in
[`METHODOLOGICAL_PROPOSALS.md`](METHODOLOGICAL_PROPOSALS.md).
