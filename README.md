# Differential Privacy in Machine Learning

**I revisited a project from 2024 and asked whether I would still trust my own methodology. The answer was no.**

This repository contains a substantially revised version of my MSc dissertation work on differential privacy and machine learning. Revisiting the project with more experience revealed methodological issues in the original implementation, particularly around sensitivity calibration and privacy accounting.

Rather than treating the dissertation as finished, I rebuilt the pipeline with bounded sensitivity, leakage-safe preprocessing, ROC-AUC evaluation, formal privacy accounting, and Differentially Private Stochastic Gradient Descent (DP-SGD) via Opacus.

The result is a reproducible comparison of feature-level differential privacy mechanisms against modern training-time privacy techniques on a healthcare classification task.

---

## Executive Summary

| Mechanism                 | Privacy Budget | ROC-AUC | Recommendation                  |
| ------------------------- | -------------- | ------- | ------------------------------- |
| No Privacy (Baseline SVM) | ∞              | 0.9878  | Reference Only                  |
| Feature Laplace (SVM)     | ε = 10.0       | 0.9680  | Viable at loose privacy budgets |
| Feature Laplace (SVM)     | ε = 2.51       | 0.8910  | Moderate utility loss           |
| Feature Gaussian (SVM)    | ε = 10.0       | 0.7560  | Not Recommended                 |
| DP-SGD                    | ε ≈ 2.13       | 0.9940  | Recommended                     |

### Key Finding

**The location of noise injection matters more than the quantity of noise.**

Feature-level perturbation rapidly degrades utility as privacy budgets become stricter. DP-SGD maintains near-baseline performance while providing meaningful privacy guarantees because noise is injected into gradients rather than directly into features.

---

## Why This Project Matters

Differential Privacy is increasingly relevant in:

* Healthcare analytics
* AI governance
* Responsible AI
* Privacy-preserving machine learning
* Regulatory compliance (GDPR, HIPAA)
* Sensitive-data modelling

This project investigates the practical cost of privacy and evaluates whether modern differential privacy techniques can preserve utility without sacrificing formal guarantees.

---

### For Hiring Managers / Data Governance Teams

This repository demonstrates:

* Differential Privacy implementation and evaluation
* Privacy-utility trade-off analysis
* Fairness auditing and demographic parity monitoring
* Reproducible ML workflows
* Automated testing and CI/CD
* Experimental design and statistical evaluation
* Responsible AI thinking

---

### For ML Engineers

Key technical lessons:

* Why feature-level differential privacy struggles at low ε
* Why sensitivity calibration matters
* How clipping affects privacy guarantees
* Why DP-SGD outperforms feature perturbation
* Practical use of Opacus and privacy accounting
* ROC-AUC evaluation under privacy constraints
* Trade-offs between privacy, utility, and fairness

For a deeper theoretical background on differential privacy concepts, see the findings report:

`reports/findings_report.md`

---

## Project Evolution

This project began as an MSc dissertation submitted in 2024.

The revised version introduces:

* Correct sensitivity calibration through clipping and standardisation
* Leakage-safe preprocessing
* ROC-AUC as the primary evaluation metric
* DP-SGD via Opacus
* Formal privacy accounting
* Fairness auditing
* Automated testing
* GitHub Actions CI/CD

The accompanying paper documents the revised methodology and findings in detail.

---

## Overview

The included Jupyter notebook (`notebooks/dp_privacy_insurance.ipynb`) walks through the following steps:

### 1. Data Loading and Exploration

We begin with a health insurance dataset containing:

* Age
* Sex
* BMI
* Number of children
* Smoker status
* Region
* Medical charges

Exploratory analysis is performed to understand feature distributions and relationships.

### 2. Differential Privacy Mechanisms

The project implements:

* Laplace Mechanism
* Gaussian Mechanism
* Exponential Mechanism
* Randomised Response
* Differentially Private SGD (Opacus)

Noise functions accept a privacy budget (`epsilon`) and sensitivity parameters, computing required scales automatically.

### 3. Sensitivity Calibration

Features are:

* Standardised
* Clipped
* Assigned bounded sensitivity

This allows formal privacy guarantees and prevents unbounded noise calculations.

### 4. Machine Learning Models

The project evaluates:

* Support Vector Machines (SVM)
* Decision Trees
* Neural Networks (DP-SGD)

Models are compared under varying privacy budgets.

### 5. Fairness Evaluation

The project measures:

* Demographic Parity Difference
* Equalised Odds Difference

to assess how privacy-preserving techniques interact with group fairness.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Ghobi-A/dp-insurance.git
cd dp-insurance
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Add the dataset

Place:

```text
data/insurance.csv
```

in the repository root.

The project uses the public Health Insurance Cost dataset.

### 4. Run the notebook

```bash
jupyter notebook notebooks/dp_privacy_insurance.ipynb
```

Or execute directly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/dp_privacy_insurance.ipynb
```

---

## Privacy Model and Guarantees

This project implements formal Differential Privacy mechanisms.

Threat model:

* Re-identification attacks
* Membership inference attacks
* Attribute inference attacks

Mechanisms implemented:

### Laplace Mechanism

Provides pure ε-Differential Privacy.

### Gaussian Mechanism

Provides (ε,δ)-Differential Privacy.

### DP-SGD

Provides training-time privacy using:

* Gradient clipping
* Gaussian noise injection
* Privacy accounting via Opacus

Privacy budgets are reported explicitly and evaluated against model utility.

### Non-DP Baselines

The repository also includes:

* k-anonymity
* l-diversity
* t-closeness

These should be treated as heuristic privacy techniques rather than formal privacy guarantees.

---

## Exponential Mechanism

The exponential mechanism (`exponential_mechanism` in `src/dp/mechanisms.py`) selects from a discrete candidate set with probability proportional to:

```text
exp(ε · utility / (2 · sensitivity))
```

The implementation:

* Uses numerically stable log-space computation
* Supports arbitrary utility functions
* Is fully covered by the test suite

---

## Pipeline Architecture

Reusable pipeline components live in:

```text
src/dp/pipeline.py
```

These functions:

* Load the dataset
* Perform train-test splitting
* Fit preprocessing transformers on training data only
* Prevent data leakage
* Construct reproducible evaluation pipelines

---

## Continuous Integration

GitHub Actions automatically:

* Install dependencies
* Execute tests
* Run notebook validation
* Verify pipeline integrity

Workflow definitions are located in:

```text
.github/workflows/
```

This ensures that all reported results remain reproducible.

---

## Repository Structure

```text
dp-insurance/
│
├── notebooks/
├── src/
│   └── dp/
├── tests/
├── reports/
├── data/
├── .github/
│   └── workflows/
└── README.md
```

---

## Removing Private Materials

The original university submission and institution-specific materials are intentionally excluded from version control.

If you add:

* PDFs
* Sensitive datasets
* Private reports

update `.gitignore` accordingly.

---

## License

This project is licensed under the MIT License.

See:

```text
LICENSE
```

for details.
