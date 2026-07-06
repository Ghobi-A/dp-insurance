# Findings Report

## Dataset characteristics

The insurance dataset contains **1,338 rows** and **7 columns**. Features are mixed-type: **3 categorical (object) columns**, **2 integer columns**, and **2 floating-point columns**. The target class **smoker** is imbalanced: **1,064 "no" (79.52%)** and **274 "yes" (20.48%)**. These proportions establish a substantial majority class that any evaluation must consider when interpreting ROC-AUC outcomes.

## Methodology note (revision)

An earlier version of this report was produced with two calibration errors that have since been corrected:

1. **Sensitivity was set to the maximum numeric column range** (≈ 47,400, dominated by `charges`) and applied as a shared noise scale to every column. This let one wide column dictate the noise added to narrow columns (`age`, `bmi`, `children`), and the reported ε did not account for the joint release of all columns. The pipeline now clips each column to **fixed, publicly known domain bounds**, rescales to [0, 1], and calibrates noise to the joint sensitivity of the full numeric row (L1 = d for Laplace, L2 = √d for Gaussian), so a single ε covers the entire release.
2. **Clip bounds were computed from quantiles of the private data**, which leaks information outside the DP accounting. Fixed public bounds (age ∈ [18, 65], BMI ∈ [15, 55], children ∈ [0, 5], charges ∈ [0, 65,000]) are used instead.

In addition, the Gaussian mechanism is now calibrated with the **analytic Gaussian mechanism** (Balle & Wang, 2018), which is valid for all ε > 0; the classic σ = Δ·√(2·ln(1.25/δ))/ε bound used previously is only valid for ε ≤ 1, so earlier Gaussian results at ε > 1 carried no formal guarantee.

The corrected feature-level results below are **substantially worse** than previously reported. This is expected: the earlier numbers reflected under-noised wide columns rather than a properly accounted ε.

## Baseline performance

Two baseline classifiers were trained with standard preprocessing (scaling numeric features and one-hot encoding categorical features). ROC-AUC on the held-out test set:

- **SVM:** 0.9948
- **Decision tree:** 0.9179

The SVM establishes a near-ceiling reference point for discrimination.

## Failure of feature-level Laplace/Gaussian noise

Feature-level perturbation was applied to the four numeric features (clipped to public bounds and rescaled) across ε ∈ {0.01, 0.0398, 0.158, 0.631, 2.512, 10}. ROC-AUC on the test set:

| ε | Laplace (SVM) | Laplace (tree) | Gaussian (SVM) | Gaussian (tree) |
| --- | --- | --- | --- | --- |
| 0.01 | 0.4471 | 0.4254 | 0.4808 | 0.5000 |
| 0.0398 | 0.4472 | 0.4769 | 0.4808 | 0.5000 |
| 0.158 | 0.4444 | 0.5000 | 0.4807 | 0.3889 |
| 0.631 | 0.5377 | 0.5067 | 0.4798 | 0.5109 |
| 2.512 | 0.5327 | 0.5211 | 0.5092 | 0.5585 |
| 10.0 | **0.6715** | 0.6428 | 0.5301 | 0.5357 |

With honest ε accounting, feature-level DP is close to chance (ROC-AUC ≈ 0.5) across the entire ε ≤ 2.5 range, and even at the very loose ε = 10 the best mechanism/model pair (Laplace + SVM) reaches only **0.67** — far below the 0.99 baseline. The signal for smoker status lives almost entirely in the `charges` feature; once that column receives noise calibrated to a properly accounted joint release, the discriminative structure is destroyed. This constitutes a practical failure of feature-level DP for this task at *any* commonly used privacy budget.

## Why Bernoulli / exponential / geometric noise were excluded

The notebook's experimental scope restricts feature-level perturbation to **Laplace and Gaussian mechanisms** and compares them to **DP-SGD training-time privacy**. Alternative distributions such as Bernoulli, exponential, or geometric noise were excluded to keep the experiments limited to (i) continuous feature perturbations with standard DP mechanisms and (ii) training-time privacy via DP-SGD, all evaluated uniformly through ROC-AUC.

## DP-SGD results and interpretation

DP-SGD was run with Opacus using noise multipliers {0.5, 1.0, 1.5, 2.0} (15 epochs, batch size 64, max gradient norm 1.0, δ = 1e-5). The privacy accountant reports:

| Noise multiplier | ε (δ = 1e-5) | ROC-AUC |
| --- | --- | --- |
| 0.5 | 32.09 | 0.9932 |
| 1.0 | 6.38 | 0.9927 |
| 1.5 | 3.19 | 0.9926 |
| 2.0 | 2.13 | 0.9940 |

Across a wide ε range (≈ 2.13–32.09), **ROC-AUC remains effectively constant at 0.993–0.994** — within a fraction of a percentage point of the non-private SVM baseline. The contrast with feature-level noise is stark: at a comparable budget (ε ≈ 2.1–2.5), DP-SGD scores 0.9940 while feature-level Laplace scores 0.5327. Injecting calibrated noise into per-sample-clipped gradients, where it is averaged over a batch and amortised over training, preserves utility that per-feature noise destroys.

## Fairness discussion

Fairness metrics were computed on the baseline SVM using **sex** as the protected attribute. The decision threshold maximises Youden's J statistic **on the training scores** (an earlier version selected the threshold on the test set, leaking test labels into the decision rule). Both metrics use the max-minus-min convention across groups:

- **Demographic parity (DP) difference:** 0.1702
- **Equalized odds (EO) difference:** 0.2000

A DP difference of 0.17 indicates a 17-percentage-point gap in positive prediction rates between sexes; the EO difference of 0.20 is driven by a gap in group-conditional error rates. These disparities persist despite near-ceiling discrimination, and privacy mechanisms do nothing to correct them — fairness must be audited and addressed separately. Because thresholding and fairness auditing are post-processing of model scores, neither consumes additional privacy budget.

## Final conclusion

With corrected sensitivity calibration and honest privacy accounting, feature-level Laplace and Gaussian perturbation fails on this task at every meaningful privacy budget: ROC-AUC sits near chance for ε ≤ 2.5 and reaches only ≈ 0.67 at ε = 10. In contrast, DP-SGD achieves **ε ≈ 2.13** while preserving **ROC-AUC ≈ 0.994**, essentially matching the non-private baseline. The revision strengthens the project's central claim — *where* noise is injected matters far more than *how much*: gradient-level noise under a formal accountant preserves utility that feature-level noise cannot. Fairness metrics reveal non-trivial group disparities regardless of the privacy mechanism, underscoring the need to evaluate privacy, utility, and fairness jointly.
