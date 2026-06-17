# Executive Summary: Differential Privacy on Smoking Classification

**Dataset:** 1,338 insurance records · 7 features · target: smoker status (79.5% no / 20.5% yes)
**Objective:** Quantify the privacy–utility trade-off across three DP strategies.

---

## Results at a Glance

| Mechanism | Privacy Budget | ROC-AUC | Verdict |
|---|---|---|---|
| No privacy (SVM baseline) | ∞ | 0.9948 | Reference |
| Feature-level Laplace | ε = 1.0 | 0.9780 | Degrades sharply below ε = 1 |
| Feature-level Gaussian | ε = 1.0 | 0.8586 | Not recommended |
| **DP-SGD (noise × 2.0)** | **ε ≈ 2.13** | **0.9940** | **Recommended** |

---

## Key Findings

**1. Feature-level noise fails at strict privacy budgets.**
At ε = 0.01, both Laplace (ROC-AUC ≈ 0.43) and Gaussian (ROC-AUC ≈ 0.47) mechanisms reduce performance to near-chance. Clipping bounds the numeric signal tightly; any tight privacy budget overwhelms it with noise. Feature-level DP is only viable above ε ≈ 1.0.

**2. DP-SGD preserves utility across a wide ε range.**
Across noise multipliers 0.5–2.0 (ε = 32.09 down to 2.13), DP-SGD holds ROC-AUC stable at 0.993–0.994 — within 0.1 percentage points of the unprotected baseline. Gradient averaging absorbs noise that would otherwise destroy individual feature signals.

**3. Strict privacy (ε ≈ 2.13) is achievable with negligible accuracy cost.**
The highest-noise DP-SGD run (multiplier = 2.0, ε = 2.13, δ = 1e-5) achieves ROC-AUC 0.9940 vs. 0.9948 unprotected — a 0.08 pp drop. This is the recommended operating point.

**4. Fairness gaps persist regardless of privacy mechanism.**
The baseline SVM shows a demographic parity difference of 0.134 (13.4 pp gap in positive prediction rates by sex) and equalized odds difference of 0.024. Privacy noise does not correct for pre-existing group disparities; fairness must be addressed separately.

---

## Recommendation

Deploy **DP-SGD with noise multiplier ≥ 2.0 (ε ≈ 2.13)** for this task. It meets strict differential privacy standards while preserving near-baseline accuracy. Feature-level noise is unsuitable for any deployment requiring ε < 1.0. Fairness auditing should accompany any privacy-protected model before production use.
