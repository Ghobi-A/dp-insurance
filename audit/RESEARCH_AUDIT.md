# Research Audit — dp-insurance

**Scope:** Audit and research-planning pass only. No code, notebooks, README, reports, dependencies, CI, or generated outputs were modified. Anything not verifiable from repository contents is marked `unverified`.

**Auditor role:** senior differential-privacy researcher / applied ML scientist / reproducibility engineer.
**Date:** 2026-07-04. **Commit audited:** `bc3778c`.

---

## 1. Verdict

**NEEDS MATERIAL REVISION**

The engineering hygiene (packaging, tests, CI scaffolding, docstrings) is well above typical MSc-portfolio level, and the self-critical framing is genuinely good. But the central scientific comparison — feature-level perturbation vs DP-SGD at "the same ε" — is not formally valid as implemented, several headline numbers are mutually contradictory across documents and traceable to no stored artefact, and every result comes from a single seed, single split, and single noise draw. The project's conclusion is probably directionally correct, but as it stands it cannot be defended against a knowledgeable reviewer.

---

## 2. Executive summary

1. **The two arms do not protect the same thing, so their ε values are not comparable.** The feature-perturbation arm noises only the 4 numeric columns of the training set; `sex`, `region`, and — critically — the **label `smoker` itself** (the sensitive attribute the whole project is about) are used in the clear. Under standard record-level adjacency this arm provides **no finite ε at all**; its stated ε holds only under a much weaker adjacency where everything except numeric attributes is public. DP-SGD's ε (Opacus, add/remove adjacency, example-level) protects the entire record including the label. The headline table ("Laplace ε=2.51 → 0.891 vs DP-SGD ε≈2.13 → 0.994") compares budgets defined over different privacy units and adjacency relations.

2. **The feature-noise sweep is dominated by a sensitivity-calibration artefact.** The notebook sets one shared sensitivity = the *maximum* per-column range after clipping ≈ **47,265** (the `charges` range), and applies that scale to every numeric column. Age (range 46), BMI (≈28), and children (5) are pure noise at *every* tested ε; the entire privacy–utility curve measures only when noise drowns `charges`. The "feature-level DP collapses at low ε" finding is partly designed in.

3. **The task is nearly trivially separable via a target proxy.** `charges` (a consequence of smoking; smoker minimum charge 12,829 exceeds the non-smoker median 7,346) makes the classification easy (AUC ≈ 0.99), which both explains DP-SGD's flat utility curve and weakens the generality of "noise location matters more than quantity."

4. **Headline numbers are unverifiable and internally inconsistent.** No notebook version in git history stores outputs. The README executive table, `reports/findings_report.md`, `reports/findings_summary.md`, and the paper PDF disagree with each other (baseline 0.9878 vs 0.9948; Gaussian ε=10 at 0.756 vs 0.8586), and `findings_summary.md` attributes results to **ε = 1.0, a value that was never in the tested grid**.

5. **No matched non-private neural baseline, no repetitions, no uncertainty.** DP-SGD is compared to an SVM; the actual "cost of privacy" for the neural network is unmeasured. Every metric is a single draw. Fairness is computed once, on the non-private baseline only, at a threshold tuned on the test set — yet the summary claims "fairness gaps persist regardless of privacy mechanism," which was never measured.

The good news: the fix is narrower than a rewrite. The dataset, the two-arm design, and most of the code structure survive. What must change is the *definition* of the comparison (matched privacy units, per-column sensitivity, public clip bounds), a matched baseline, and a seed-repeated, artefact-generating harness.

---

## 3. Findings

### Critical

**C1 — Privacy-unit mismatch between the compared arms (invalidates the headline comparison).**
`notebooks/dp_privacy_insurance.ipynb` cells 10–11 (`add_dp_noise`, `evaluate_dp_features`): noise is applied only to `df.select_dtypes(include="number")`; categorical columns and `y_train` pass through untouched. The released training object is (noised numerics, raw `sex`/`region`, raw `smoker` labels). Under replace-one adjacency over full records, this mechanism has unbounded privacy loss — the label is released in the clear. DP-SGD (cell 16) protects the full example under add/remove adjacency via Opacus. The two ε axes are therefore incomparable, and the README's Key Finding ("location of noise injection matters more than quantity") is confounded with "what is protected differs." Also note the adjacency mismatch itself (bounded/replace-one for a fixed-size data release vs unbounded/add-remove in Opacus accounting) — worth roughly a factor of 2 in ε even after the unit is fixed.

**C2 — Sensitivity mis-calibration: one shared max-range scale for all columns.**
`notebooks/dp_privacy_insurance.ipynb` cell 11: `sensitivity = (bounds["upper"] - bounds["lower"]).max()` ≈ 47,265 (train 1%–99% quantile range of `charges`; verified from `data/insurance.csv`). This single value feeds `add_laplace_noise`/`add_gaussian_noise`, which apply scale = sensitivity/ε **per column** (`src/dp/mechanisms.py:99-105, 177-183`). At ε = 10 the Laplace scale is ≈ 4,727 added to `age` (range 46), `bmi` (≈28), `children` (5): those features carry zero signal at every tested ε. Additionally, for the Laplace vector release the correct L1 sensitivity is the *sum* of per-column ranges (≈47,344), not the max, so the claimed ε is formally understated by a small factor (≈1.002); for Gaussian the max ≈ the L2 norm here only because `charges` dominates. Correct design: per-column scales calibrated to per-column ranges with an explicit budget split (or clip in standardised space).

**C3 — Data-dependent preprocessing sits inside the claimed privacy boundary but is not accounted for.**
Clip bounds are the 1%/99% quantiles of the raw training data (`compute_clip_bounds(split.X_train)`, cell 11) — computed non-privately from the very data the ε claims to protect, and required to run/interpret the mechanism. The DP-SGD arm fits `StandardScaler` statistics non-privately on the same raw training data (cell 16, via `build_preprocessor`). In both arms the end-to-end release (parameters + model) is **not** (ε, δ)-DP as claimed. Why it matters: quantiles and per-feature means/stds can each leak individual records (e.g., an outlier's charge moves the 99% quantile). Fix: declare fixed, a-priori public bounds (justified from domain knowledge) or spend budget on private quantiles — and state the choice.

**C4 — Headline results are unverifiable and mutually contradictory.**
No notebook version in git history contains outputs (verified across all commits touching `notebooks/`). The README executive-summary numbers were introduced in README-only commits (`9624492`, `714101d`) with no accompanying artefact. Contradictions: baseline SVM 0.9878 (README) vs 0.994793 (`reports/findings_report.md`); Laplace ε=10 0.9680 vs 0.9780; Gaussian ε=10 0.7560 vs 0.8586. `reports/findings_summary.md` reports "Laplace ε = 1.0 → 0.9780" and "Gaussian ε = 1.0 → 0.8586" — **ε = 1.0 is not in the tested grid** (`np.logspace(-2, 1, 6)` = {0.01, 0.0398, 0.158, 0.631, 2.512, 10}); those are the ε=10 values misattributed. The same ε=1.0 error is baked into `src/dp/mechanisms.py:130-132` docstrings. All headline metrics: `unverified`.

### High

**H1 — No matched non-private neural baseline.**
DP-SGD (1 hidden layer × 32, Adam lr=1e-2, batch 64, 15 epochs; cell 16) is compared only against SVM/decision-tree baselines. "Within 0.08 pp of the unprotected baseline" (`findings_summary.md`) compares across model classes; the cost of privacy for *the network itself* is unmeasured. (Nit: with Adam this is DP-Adam, not DP-SGD.)

**H2 — Single seed, single split, single noise draw; and `get_rng(None)` silently reuses seed 42.**
`src/dp/constants.py:10-13`: `get_rng(None)` returns `default_rng(RANDOM_STATE)`, contradicting the mechanisms docstrings ("None uses the global NumPy RNG", `mechanisms.py:88-89, 165`). Consequences: (a) in `dp.evaluation.evaluate_models` with `random_state=None`, every ε in a sweep reuses the *same* underlying uniform draws — one noise realisation rescaled; (b) in `apply_randomized_response(random_state=None)` (`mechanisms.py:298`) every column receives seed 42 → identical flip masks across columns, violating the documented independence and the basic-composition claim in its own docstring. No experiment reports variance; AUC differences of 0.001–0.02 are currently uninterpretable.

**H3 — Task construction: `charges` is a near-perfect proxy for the target.**
Verified from the data: smoker minimum charge (12,829) exceeds the non-smoker median (7,346); non-smoker 90th percentile is 14,350. The task is close to one-feature separable, which (a) explains near-ceiling AUCs, (b) makes DP-SGD's flat curve *expected* (a dominant signal survives gradient noise easily), and (c) means the feature-DP curve mostly measures when `charges` drowns. Plausibility of the DP-SGD accounting itself checks out roughly (n≈1070, batch 64 → sample rate ≈0.06, ≈255 steps; nm=2.0 → ε≈2.13 at δ=1e-5 is plausible under RDP accounting — `unverified`, not re-run), so the flat utility is likely real but weakly generalisable.

**H4 — Fairness: test-set threshold tuning, baseline-only measurement, no uncertainty, unsupported "regardless of privacy" claim.**
Cell 20: Youden's J threshold chosen **on the test set**, metrics reported on the same test set. Fairness is computed only for the non-private SVM — never under feature noise or DP-SGD — yet `findings_summary.md` claims "Fairness gaps persist regardless of privacy mechanism" and "Privacy noise does not correct for pre-existing group disparities" (never measured). Test subgroup sizes: ≈268 test rows, ≈55 positives, ≈27–32 positives per sex group → TPR-difference CIs on the order of ±0.15–0.2, so EO diff 0.024 is statistically indistinguishable from substantial values. Base rates differ by sex in the raw data (23.5% vs 17.4% smokers), so demographic-parity difference partly reflects base rates rather than model behaviour. Currently descriptive only.

**H5 — README privacy claims not backed by any code.**
`README.md:229-237`: "The repository also includes: k-anonymity, l-diversity, t-closeness" — no such code exists anywhere in the repo (verified by search). `README.md:203-207` lists a threat model (re-identification, membership inference, attribute inference) with no empirical or formal support; the feature-DP arm does not even protect the label against attribute inference.

**H6 — The tested library path and the notebook's experimental path diverge; exported DP utilities contain a privacy-destroying bug.**
`dp.evaluation.evaluate_models`/`privacy_utility_sweep` (exported in `__init__.py`, covered by `tests/test_privacy_checks.py`) apply noise to **unclipped raw features with sensitivity=1.0 default** — a formally meaningless ε that violates the module's own docs ("Without clipping … the DP guarantee is void", `mechanisms.py:70-72`). The notebook re-implements clipping/noise/DP-SGD/fairness inline; so CI tests one experiment and the reports describe another. `pipeline.apply_feature_noise` is dead code. `src/dp/dpsgd.py:101-107` (`train_dp_sgd`) calls `optimizer.zero_grad()`/`optimizer.step()` on the **raw optimizer argument instead of `setup.optimizer`** — any caller would train with unclipped, un-noised gradients while believing DP-SGD ran, and `grad_sample` buffers would never be cleared. It is unused by the notebook and untested, but it is exported "DP utilities" API.

**H7 — Classical Gaussian mechanism used outside its validity range.**
`src/dp/mechanisms.py:177`: σ = Δ·√(2·ln(1.25/δ))/ε is the classical analytic bound, whose proof requires **ε < 1** (Dwork & Roth, Thm A.1; the docstring itself cites Balle & Wang 2018 for tighter bounds). The Gaussian sweep reports ε ∈ {2.512, 10} — a third of the Gaussian grid carries no established (ε, δ) guarantee under the implemented formula.

### Medium

**M1 — CI validates software integrity only.** `run-notebook.yml` runs pytest (shape/reproducibility smoke tests) and `jupyter nbconvert --execute` (no `--inplace`, output discarded, nothing asserted). README's "This ensures that all reported results remain reproducible" (`README.md:290`) is false. Test file name `test_privacy_checks.py` overstates: nothing in it checks a privacy property.

**M2 — Quickstart cannot reproduce the headline result.** `pip install -e .` does not install torch/opacus (they are in the `experimental` extra, `pyproject.toml:23-27`), so the DP-SGD cell silently skips (`opacus_available()` guard). README also tells users to "Add the dataset" even though `data/insurance.csv` is committed. `requirements.txt` and `pyproject.toml` disagree (seaborn only in the former; different pin styles).

**M3 — Phantom dependencies.** `tensorflow`, `keras`, `imbalanced-learn`, `fairlearn`, `scipy` are base dependencies but are imported nowhere in `src/`, `tests/`, or the notebook (fairlearn is listed while fairness metrics are hand-rolled). This inflates every install/CI run by gigabytes and signals unreviewed configuration.

**M4 — Evaluation hygiene.** All ~25 configurations are evaluated on the single 268-row test set from which the operating threshold is also selected; conclusions ("recommended operating point") are drawn from the same set. No validation split; no statement of how DP-SGD hyperparameters (architecture, lr, epochs, clip norm) were chosen — if they were tuned on this data, that tuning is an unaccounted privacy cost (acceptable in practice if declared; currently undeclared).

**M5 — Documentation inconsistencies.** `findings_report.md` §"Why Bernoulli/exponential/geometric noise were excluded" vs notebook cell 14 which excludes "randomized response and output perturbation" — different rationales for different lists. `CONTRIBUTING.md` claims coverage reports are written to `htmlcov/` via `[tool.pytest.ini_options]` (no such `addopts` exist) and prescribes statistical sanity tests that don't exist. δ = 1e-5 is fine (< 1/n ≈ 9×10⁻⁴) but justified nowhere.

### Low

- `dp.evaluation._encode_labels` maps classes by sorted order ("no"→0, "yes"→1 — happens to be right; fragile in general).
- `demographic_parity_diff`/`equalized_odds_diff` (cell 20) silently use only the first two groups if more exist.
- Notebook cell 16 refits the shared `preprocessor` object from cell 7 (order-dependent notebook state).
- `randomized_response` docstring "Pr[M(v)=o]/Pr[M(v′)=o] = e^ε / 1" — the correct ratio is p/(1−p) = e^ε (result right, derivation sloppy).
- `requires-python = ">=3.8"` while CONTRIBUTING mandates 3.10+ union syntax (works at runtime via `from __future__ import annotations`; CI tests only 3.9/3.10).
- Dataset provenance/licence (Kaggle "insurance.csv") not cited anywhere.
- Git hygiene: commits `2353156` ("Add new file 'a'") / `bc3778c` ("Delete paper/a").

---

## 4. Claim-to-evidence matrix

Verdict key: **Retain** / **Narrow** (keep with caveats) / **Correct** (number or statement wrong) / **Remove**. Every "Regenerate" means: re-run under the upgraded harness and cite the stored artefact (results CSV + config hash).

| # | Claim (location) | Evidence found | Verdict | Artefact required |
|---|---|---|---|---|
| 1 | Baseline SVM ROC-AUC **0.9878** (README exec table) | Contradicts findings_report (0.994793); no stored output anywhere; introduced in README-only commit | **Correct** | Seed-averaged baseline run in results CSV |
| 2 | Baseline SVM ROC-AUC **0.994793**, DT 0.917883 (findings_report §Baseline) | Plausible from code path; `unverified` (no outputs in any commit) | **Narrow → Regenerate** | Same |
| 3 | Laplace SVM ε=10 → **0.9680** (README) | Contradicts findings_report span max 0.9780 | **Correct** | Sweep artefact |
| 4 | Laplace SVM ε=2.51 → **0.8910** (README) | Within report's span; `unverified`; ε not comparable across arms (C1) | **Narrow → Regenerate** | Matched-unit sweep artefact |
| 5 | Gaussian SVM ε=10 → **0.7560** "Not Recommended" (README) | Contradicts report (0.8586); Gaussian σ formula invalid at ε>1 (H7) | **Correct + Narrow** | Re-run with valid Gaussian calibration (ε<1 or Balle–Wang) |
| 6 | Laplace/Gaussian spans 0.4329→0.9780 / 0.4732→0.8586 (findings_report §Failure) | `unverified`; magnitude driven by max-range shared sensitivity (C2) and single noise draw (H2) | **Narrow → Regenerate** | Per-column-calibrated sweep, ≥10 seeds, CIs |
| 7 | "Feature-level Laplace **ε = 1.0** → 0.9780"; "Gaussian **ε = 1.0** → 0.8586"; "viable only above ε ≈ 1.0" (findings_summary) | **ε=1.0 never tested** — grid is {0.01, 0.04, 0.16, 0.63, 2.51, 10}; numbers are the ε=10 values | **Correct** (factual error) | Include ε=1.0 in the new grid |
| 8 | Same ε=1.0 utility claims in `src/dp/mechanisms.py:130-132` docstring and ε-guidance table (`mechanisms.py:19-23`) | Same error, plus derived from artefact-driven curve | **Correct** | New sweep |
| 9 | DP-SGD table: nm {0.5,1,1.5,2} → ε {32.09, 6.38, 3.19, 2.13}, AUC ≈0.993–0.994 (findings_report; README row) | Code path exists (cell 16); accounting figures plausible (`unverified`); single seed; single run each | **Narrow → Regenerate** | Seeded DP-SGD runs with accountant metadata (sample rate, steps, accountant type, δ) |
| 10 | "DP-SGD … minimal degradation relative to the SVM baseline"; "within 0.08 pp of unprotected baseline" (findings_report, findings_summary, paper abstract) | Cross-model-class comparison (H1); no matched NN baseline exists | **Narrow** | Matched non-private NN baseline (addition A) |
| 11 | **Key Finding: "location of noise injection matters more than quantity of noise"** (README) | Confounded: different privacy units (C1), artefact sensitivity (C2), easy task (H3), one seed (H2) | **Narrow** — restate as the *research question*, claimable only after matched-unit redesign | Additions A–C below |
| 12 | "Correct sensitivity calibration through clipping and standardisation" (README §Evolution) | False: shared max-range scale (C2); noise applied before standardisation; bounds non-private (C3) | **Correct** | Per-column calibration + public bounds |
| 13 | "Leakage-safe preprocessing" (README) | True for scaler fit-on-train-only (`pipeline.py:95-100`); violated by non-private clip bounds (C3) and test-set threshold tuning (H4) | **Narrow** | Public-bounds decision + validation-split thresholding |
| 14 | "Formal privacy accounting" (README) | Holds for DP-SGD arm (Opacus); feature arm's end-to-end ε not formally valid (C1–C3, H7) | **Narrow** | Privacy-unit table (see §6) |
| 15 | "The project implements: Laplace, Gaussian, Exponential, Randomised Response, DP-SGD" (README §Overview) | All exist in `src/dp/`; exponential & RR never used in any experiment | **Retain** with "implemented as library, not used in experiments" — or use RR for label protection (see §5) | — |
| 16 | Threat model: re-identification / MIA / attribute inference (README §Privacy Model) | No attack evaluation; feature arm leaves the sensitive label public | **Remove** or reduce to "formal guarantee statement" only | None (or addition F, not recommended) |
| 17 | "Repository also includes k-anonymity, l-diversity, t-closeness" (README) | **No such code exists** | **Remove** | — |
| 18 | "CI … ensures that all reported results remain reproducible" (README §CI) | CI asserts nothing about results (M1); notebook outputs not stored | **Correct** | Addition H (artefact-generating pipeline + CI schema check) |
| 19 | Dataset: 1,338 rows, 7 cols, 79.52%/20.48% imbalance (findings_report §Dataset) | **Verified** from `data/insurance.csv` | **Retain** | — |
| 20 | Fairness: DP diff 0.1337, EO diff 0.0240 (findings_report, README, paper) | Code path exists (cell 20); `unverified`; test-set threshold; no CIs; ≈27–32 positives/group; baseline model only | **Narrow** to descriptive, with CIs and subgroup counts | Addition G |
| 21 | "Fairness gaps persist **regardless of privacy mechanism**"; "privacy noise does not correct disparities" (findings_summary §4) | Fairness never measured under any privacy mechanism | **Remove** until measured | Addition G (fairness across arms/ε) |
| 22 | "Deploy DP-SGD with noise multiplier ≥ 2.0 (ε≈2.13)" (findings_summary §Recommendation) | Deployment recommendation from a single-seed run on one easy dataset | **Remove** (replace with scoped conclusion) | Additions B–C |
| 23 | "Gradient averaging absorbs noise that would otherwise destroy individual feature signals" (findings_summary §2) | Mechanistic explanation, not evidenced; plausible | **Narrow** to hypothesis wording | Optional: addition D sheds indirect light |
| 24 | Paper abstract: "SVM falling below 0.6 at strict budgets"; "Gaussian unsuitable across the full tested range"; "degrade sharply below ε=1" (paper PDF) | Consistent with findings_report ranges but inherits every issue above incl. the ε=1 grid gap; `unverified` | **Narrow** (paper should be re-issued after regeneration or labelled as superseded) | Regenerated results |
| 25 | "Reproducible comparison" (README intro) | Quickstart skips DP-SGD silently (M2); no artefacts; notebook stripped of outputs | **Correct** | Addition H |

---

## 5. Research contribution (revised)

**What the project can honestly claim after the upgrade:**

> **Central claim.** In tabular healthcare classification, *where* differential-privacy noise is injected — into the input features (a private release of the training data, then ordinary training) versus into training gradients (DP-SGD) — changes the privacy–utility frontier by [measured amount] in ε at matched utility, **when the comparison is made fair**: same privacy unit (full record, including the label), same adjacency relation, same model class, same preprocessing inside a declared privacy boundary, and uncertainty quantified over seeds. Fairness and calibration effects are reported as secondary, power-limited analyses.

**Research question.** "Does the location of privacy-noise injection — input perturbation vs DP-SGD — materially alter the privacy–utility(–fairness) trade-off in tabular healthcare classification?" — retained, but the paper's distinctive value becomes *the construction of a valid comparison*, which almost no small studies get right. The audit trail (what a naive comparison gets wrong: unprotected labels, shared max-range sensitivity, unmatched baselines) is itself part of the contribution and should appear as a "pitfalls" subsection.

**Key design decision required (Day 3 below).** To put both arms on one ε axis, the input-perturbation arm must protect the whole record. Recommended: per-column Laplace/Gaussian on numeric features **plus randomised response on the label and categorical features** (already implemented and unused in `src/dp/mechanisms.py`), composed to a single per-record (ε, δ) under replace-one adjacency, with an explicit conversion/statement relative to Opacus's add/remove accounting. The fallback — narrowing the feature arm's claim to "protects numeric attributes only, everything else public" — keeps the arms incomparable and should be rejected for the headline comparison (it can be a discussion note).

**Fairness: keep, but demoted to secondary.** With ≈27–32 test positives per sex group, only large fairness effects are detectable. Recommendation: keep fairness in the research question as an explicitly power-limited secondary analysis — subgroup counts and bootstrap CIs in every table, and "no detectable difference" reported as such. Do not headline fairness deltas.

---

## 6. Proposed paper / report structure

1. **Abstract.** One matched comparison, one dataset, quantified: "At matched record-level privacy (ε ∈ [0.5, 10], δ = 1/(10n)), DP-SGD attains ROC-AUC X ± ci vs Y ± ci for full-record input perturbation; the non-private matched network attains Z. Calibration degrades [or not] under each arm. Fairness differences are within CI resolution given subgroup sizes."
2. **Threat model and privacy units** *(the table most current versions of this study lack)*: for each arm — released object, privacy unit, adjacency relation, what is public (clip bounds, scaler statistics, hyperparameters), accountant, δ and its justification.
3. **Methods.** Per-column sensitivity calibration and budget split; RR on label/categoricals with composition; DP-SGD setup (architecture, sampling, accountant, clipping norm); why classic Gaussian is restricted to ε<1 or replaced with Balle–Wang calibration.
4. **Experimental setup.** Data (provenance + licence), split protocol, seeds (≥10), metrics (ROC-AUC, PR-AUC, Brier/ECE), threshold selection on validation only, fixed hyperparameters declared a priori (with the caveat that no private tuning was performed).
5. **Results.**
   - **F1**: privacy–utility Pareto: ROC-AUC (and PR-AUC) vs ε, mean ± 95% CI bands, both arms; non-private matched baselines as horizontal reference bands.
   - **T1**: headline table at ε ∈ {0.5, 1, 3, 10}: AUC, PR-AUC, Brier, ECE per arm.
   - **F2**: calibration curves at one matched ε.
   - **T2**: fairness (DP diff, EO diff) with bootstrap CIs and subgroup n's, per arm and ε.
   - **T3** *(optional, addition D)*: DP-SGD ablation (clip norm × noise multiplier at fixed ε budget).
6. **Pitfalls subsection.** What the naive comparison got wrong (this audit, condensed).
7. **Limitations.** One small easy dataset with a dominant proxy feature; fairness underpowered; hyperparameters not privately tuned; preprocessing boundary choices; results may not transfer to harder tasks (state the `charges`-ablation observation if run).
8. **Reproducibility appendix.** Config files, exact commands, seed list, environment pins, artefact hashes; every number in the paper generated from a committed CSV by one command.

---

## 7. Minimal upgrade roadmap (Phase 2)

### Must Have (4 substantive additions — the cap)

**A. Matched non-private neural baseline** — *essential*
- **RQ answered:** what does privacy cost *for this model*? (Removes the SVM confound.)
- **Minimum experiment:** identical architecture/optimizer/schedule/preprocessing as the DP arm with the PrivacyEngine disabled; same seeds.
- **Files:** promote cell-16 logic into `src/dp/` (e.g. a `train_model(cfg, private: bool)` in a new `experiments.py` or rewritten `dpsgd.py` — fixing the `train_dp_sgd` raw-optimizer bug in the process); notebook becomes a caller.
- **Output:** the horizontal reference band in F1; the "cost of privacy" column in T1.
- **Validity threat if skipped:** every "near-baseline" claim remains cross-model-class.
- **Effort:** ~half a day. **Portfolio value:** high (reviewers look for exactly this).

**B. Matched-unit ε sweep with justified δ and consistent accounting** — *essential; this is the paper*
- **RQ answered:** the central question, made formally valid.
- **Minimum experiment:** (i) feature arm: public a-priori clip bounds; per-column sensitivity; explicit per-column budget split; RR on label + categoricals; compose to a single per-record (ε, δ); Gaussian restricted to ε<1 or recalibrated (Balle–Wang); (ii) DP-SGD arm: `make_private_with_epsilon` targeting the same ε grid (e.g. {0.5, 1, 2, 5, 10}), δ = 1/(10n) stated once; (iii) a privacy-unit/adjacency table, including the replace-one vs add/remove caveat.
- **Files:** `src/dp/mechanisms.py` (per-column calibration), `src/dp/pipeline.py` (public bounds; retire or fix `apply_feature_noise`), `src/dp/dpsgd.py`, `src/dp/evaluation.py` (align the library path with the real experiment — closes H6), config files.
- **Output:** F1 Pareto figure; T1.
- **Validity threat:** the replace-one vs add/remove adjacency mismatch must be stated (or converted); RR-on-label at very low ε will destroy label utility — that is a *finding*, not a bug, but must be explained.
- **Effort:** 1.5–2 days. **Portfolio value:** the core.

**C. Repeated-seed harness with uncertainty** — *essential*
- **RQ answered:** are the observed gaps real?
- **Minimum experiment:** ≥10 seeds varying split, noise draw, and model init; report mean ± sd and 95% CI for ROC-AUC, PR-AUC, Brier/ECE (and fairness, via G) per configuration; fix `get_rng(None)` semantics so seeds actually vary (closes H2).
- **Files:** new `src/dp/harness.py` (or `experiments.py`), `src/dp/constants.py`, tests asserting seed-to-seed variation and noise-scale statistics.
- **Output:** CI bands on every figure; a seed appendix table.
- **Validity threat:** with n_test = 268, AUC standard error is ~0.01–0.02 — differences smaller than that must be reported as indistinguishable.
- **Effort:** ~1 day. **Portfolio value:** high; converts anecdotes into results.

**H. Config-driven experiments + one-command regeneration** — *essential*
- **RQ answered:** none directly — it is what makes every other claim auditable (and permanently fixes the C4 class of contradictions).
- **Minimum design:** YAML/JSON configs per arm; `python -m dp.run_experiments` → `results/*.csv` (with config hash + git SHA columns); `python -m dp.build_report` → figures + the markdown tables embedded in the report; CI runs a tiny grid and asserts artefact schema + baseline AUC within a tolerance band; README/report numbers generated, never hand-typed.
- **Files:** `configs/`, `src/dp/run_experiments.py`, `src/dp/build_report.py`, `Makefile` or scripts, `run-notebook.yml`; move torch/opacus out of the `experimental` extra or make the quickstart install it (closes M2); drop phantom deps (M3).
- **Output:** the reproducibility appendix; "one command regenerates every number" is a headline portfolio feature.
- **Effort:** ~1 day. **Portfolio value:** high for an engineering-flavoured portfolio.

### Valuable (fold in cheaply once C exists; optional)

**G. Fairness as a caveated secondary analysis** — subgroup counts + bootstrap CIs for DP/EO diffs, computed **across arms and ε**, threshold selected on a validation split. Answers the fairness leg of the RQ honestly (and either substantiates or retires the currently unsupported "regardless of privacy mechanism" claim). Files: new `src/dp/fairness.py` (promoted from cell 20). Effort: ~half a day on top of C. Main threat: underpowered — must be labelled as such.

**E. Calibration (Brier + ECE)** — cheap columns in the harness metric set; technically justified because DP noise can preserve ranking (AUC) while damaging probability calibration — potentially the most interesting secondary result if AUC stays flat. Effort: hours. Threat: ECE binning instability at n=268 → use ≤10 bins and report Brier as primary.

**D. Small DP-SGD ablation** — clip norm × noise multiplier at fixed ε (via accountant), one epoch setting. Only if time remains; one factor at a time; supports the "why DP-SGD tolerates noise" discussion. Effort: ~half a day of compute/config. Threat: combinatorial creep — cap at a 3×3 grid.

*(Optional robustness variant inside B/C, not a separate addition: one run of both arms with `charges` excluded, to show the conclusion on a harder version of the task. One config line once H exists.)*

### Avoid

**F. Membership-inference evaluation — avoid.** With 1,338 records, an easy task, and a near-zero generalisation gap, loss-threshold MIA will sit at ~chance for *all* arms including the non-private baseline — an uninformative result that invites "empirically private" overclaims; a defensible LiRA-style protocol needs hundreds of shadow models and would dwarf the rest of the project. Note it as future work only.

---

## 8. Seven-day execution plan

| Day | Work (dependency-ordered) | Exit criterion |
|---|---|---|
| **1** | Privacy foundations: public clip bounds; per-column sensitivity + budget split in `src/dp/mechanisms.py`/`pipeline.py`; fix `get_rng(None)`; fix or delete `train_dp_sgd`; restrict/recalibrate Gaussian (ε<1 or Balle–Wang); statistical unit tests for noise scales | Tests assert correct per-column scale and seed-to-seed variation |
| **2** | Model arms in `src/`: matched NN baseline (A) + DP-SGD via `make_private_with_epsilon`; record accountant metadata (sample rate, steps, accountant, δ) | One function trains private/non-private from a config; smoke run reproduces ε ≈ target |
| **3** | Matched privacy unit (B): RR on label + categoricals, per-record composition; write the privacy-unit/adjacency table (paper §2). *This is the day with a real design decision — do it before running anything big* | A written, reviewable privacy statement per arm |
| **4** | Harness (C+H): configs, seed loop, `results/*.csv` with config hash; `build_report` generating tables/figures; CI tiny-grid schema check | `make reproduce` runs end-to-end on a reduced grid |
| **5** | Full run: 2 arms × ε grid × ≥10 seeds; metrics incl. Brier/ECE (E) and fairness with bootstrap CIs (G); generate F1/F2/T1/T2 | Artefacts committed; figures generated from artefacts only |
| **6** | Rewrite `reports/findings_report.md` + README executive summary strictly from generated artefacts; apply the claim-matrix verdicts (remove k-anonymity claim, threat-model section, ε=1.0 misattributions, "regardless of privacy" fairness claim); mark `findings_summary.md` and the paper PDF as superseded or regenerate | Every number in README/report traceable to a CSV row |
| **7** | Clean-clone reproduction test (fresh venv, pinned deps, `make reproduce`); dependency prune (drop TF/keras/imblearn/fairlearn or actually use fairlearn); limitations section; final self-review against this audit | A stranger can regenerate the headline table with one command |

Dependencies: Day 1 → 2 → 3 are strictly ordered (foundations before arms before unit-matching); Day 4 can start in parallel with Day 3; Days 5–7 are serial.

---

## 9. Do Not Add

- **Federated learning, homomorphic encryption, synthetic data, transformers/LLMs** — excluded by design (per project owner), and each would dilute the single contribution.
- **Membership-inference attacks (F)** — see above; ~chance results on this dataset prove nothing and risk overclaiming.
- **A second dataset** — would double every experiment for a generality claim the paper doesn't need; state single-dataset scope in Limitations and list as future work.
- **More model families (XGBoost, logistic regression, deeper nets)** — the contribution is *noise location*, not a model bake-off; the matched NN + existing SVM/DT references suffice.
- **Fairness mitigation techniques (post-processing EO, reweighing)** — a different paper; here fairness is measured, not fixed.
- **PATE, output perturbation, objective perturbation as extra arms** — each adds a new privacy unit to reconcile; two arms done rigorously beat four done loosely.
- **Hyperparameter search / AutoML** — introduces an unaccounted privacy cost and noise in the comparison; declare fixed, a-priori hyperparameters and caveat it.
- **DP feature selection, shuffle-model DP, tighter accountants comparison (RDP vs PRV)** — accountant choice is a footnote, not a section.
- **Exponential-mechanism experiments** — keep it as tested library code; wiring it into the pipeline answers no part of the research question.

---

*End of audit. Nothing in the repository was modified by this pass; this document is the only addition.*
