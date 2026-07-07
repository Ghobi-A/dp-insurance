# Executive Summary: Differential Privacy on Smoking Classification

**Dataset:** 1,338 insurance records · 7 features · target: smoker status (79.5% no / 20.5% yes)
**Objective:** Quantify the privacy–utility trade-off across feature-level DP and DP-SGD.

---

## Results at a Glance

| Mechanism | Privacy Budget | ROC-AUC | Verdict |
|---|---|---|---|
| No privacy (SVM baseline) | ∞ | 0.9948 | Reference |
| Feature-level Laplace (SVM) | ε = 2.51 | 0.5327 | Near chance — not viable |
| Feature-level Laplace (SVM) | ε = 10.0 | 0.6715 | Weak even at loose budgets |
| Feature-level Gaussian (SVM) | ε = 10.0, δ = 1e-5 | 0.5301 | Not viable |
| **DP-SGD (noise × 2.0)** | **ε ≈ 2.13, δ = 1e-5** | **0.9940** | **Recommended** |

All feature-level numbers use honest accounting: features are clipped to fixed public domain bounds, rescaled to [0, 1], and noised with the joint sensitivity of the full numeric release, so the stated ε covers all four numeric columns together. Gaussian noise is calibrated with the analytic Gaussian mechanism (Balle & Wang, 2018), valid at all ε.

---

## Key Findings

**1. With correct calibration, feature-level noise fails at every meaningful budget.**
ROC-AUC stays near 0.5 (chance) for ε ≤ 2.5 and reaches only ≈ 0.67 at the very loose ε = 10. The predictive signal is concentrated in `charges`; once noise is calibrated to a properly accounted joint release, that signal is destroyed. (An earlier version of this project reported much higher feature-level scores — those were an artifact of a sensitivity mis-calibration, since corrected.)

**2. DP-SGD preserves utility across a wide ε range.**
Across noise multipliers 0.5–2.0 (ε = 32.09 down to 2.13), DP-SGD holds ROC-AUC stable at 0.993–0.994 — within a fraction of a point of the unprotected baseline. Per-sample gradient clipping plus batch averaging absorbs noise that would destroy individual feature values.

**3. Strict privacy (ε ≈ 2.13) is achievable with negligible accuracy cost.**
The highest-noise DP-SGD run (multiplier 2.0, ε = 2.13, δ = 1e-5) achieves ROC-AUC 0.9940 vs. 0.9948 unprotected. At the same budget, feature-level Laplace manages 0.5327. This is the recommended operating point.

**4. Fairness gaps persist regardless of privacy mechanism.**
With the decision threshold tuned on training scores (Youden's J), the baseline SVM shows a demographic parity difference of 0.170 and an equalized odds difference of 0.200 by sex. Privacy noise does not correct pre-existing group disparities; fairness must be audited separately. Both metrics are post-processing and consume no privacy budget.

---

## Recommendation

Deploy **DP-SGD with noise multiplier ≥ 2.0 (ε ≈ 2.13, δ = 1e-5)** for this task. It meets strict differential privacy standards while preserving near-baseline accuracy. Feature-level noise is unsuitable for this task at any commonly used budget. Fairness auditing should accompany any privacy-protected model before production use.
