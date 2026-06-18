# Differential Privacy in Machine Learning

**I revisited a project from 2024 and asked whether I would still trust my own methodology. The answer was no.**

This repository contains a substantially revised version of my MSc dissertation work on differential privacy and machine learning. Revisiting the project with more experience revealed methodological issues in the original implementation, particularly around sensitivity calibration and privacy accounting. The pipeline was rebuilt with bounded sensitivity, leakage-safe preprocessing, ROC-AUC evaluation, and Differentially Private Stochastic Gradient Descent (DP-SGD) via Opacus.

The result is a reproducible comparison of feature-level differential privacy mechanisms against modern training-time privacy techniques on a healthcare classification task.

## Executive Summary

| Mechanism                 | Privacy Budget | ROC-AUC | Recommendation                  |
| ------------------------- | -------------- | ------- | ------------------------------- |
| No Privacy (Baseline SVM) | ∞              | 0.9878  | Reference Only                  |
| Feature Laplace (SVM)     | ε = 10.0       | 0.968   | Viable at loose privacy budgets |
| Feature Laplace (SVM)     | ε = 2.51       | 0.891   | Moderate utility loss           |
| Feature Gaussian (SVM)    | ε = 10.0       | 0.756   | Not Recommended                 |
| DP-SGD                    | ε ≈ 2.13       | 0.9940  | Recommended                     |

## Key Finding

**The location of noise injection matters more than the quantity of noise.**

Feature-level perturbation rapidly degrades utility as privacy budgets become stricter. DP-SGD achieves meaningful privacy guarantees while maintaining near-baseline model performance by injecting noise into gradients rather than directly into features.

## Why This Project Matters

Differential Privacy is increasingly relevant in:

* Healthcare analytics
* AI governance
* Responsible AI
* Regulatory compliance (GDPR, HIPAA)
* Privacy-preserving machine learning

This project investigates the practical cost of privacy and evaluates whether modern differential privacy techniques can preserve utility without sacrificing formal guarantees.

### For Data Governance Teams

This repository demonstrates:

* Differential Privacy implementation and evaluation
* Privacy-utility trade-off analysis
* Fairness auditing and demographic parity monitoring
* Reproducible ML workflows
* Automated testing and CI/CD
* Experimental design and statistical evaluation

### For ML Engineers

Key technical lessons:

* Why feature-level differential privacy struggles at low ε
* Why sensitivity calibration matters
* How clipping affects privacy guarantees
* Why DP-SGD outperforms feature perturbation
* Practical use of Opacus and privacy accounting
* ROC-AUC evaluation under privacy constraints

## Project Evolution

This project began as an MSc dissertation submitted in 2024.

The revised version introduces:

* Correct sensitivity calibration through clipping and standardisation
* Leakage-safe preprocessing
* ROC-AUC as the primary evaluation metric
* DP-SGD via Opacus
* Formal privacy accounting
* Fairness auditing
* Automated testing and GitHub Actions

The accompanying paper documents the revised methodology and findings in detail.
