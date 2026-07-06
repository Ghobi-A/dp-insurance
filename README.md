# Differential Privacy in Machine Learning

**I revisited a project from 2024 and asked whether I would still trust my own methodology. The answer was no — twice.**

This repository contains a substantially revised version of my MSc dissertation work on differential privacy and machine learning. Revisiting the project with more experience revealed methodological issues in the original implementation — and a second audit of the revision found more: sensitivity mis-calibration that silently inflated feature-level DP results, a Gaussian mechanism whose classic calibration is invalid above ε = 1, clip bounds derived from the private data itself, and a fairness threshold tuned on the test set.

The current version fixes all of these. The pipeline uses bounded sensitivity with honest joint-release accounting, fixed public domain bounds, the analytic Gaussian mechanism (Balle & Wang, 2018), leakage-safe preprocessing and thresholding, ROC-AUC evaluation, and Differentially Private Stochastic Gradient Descent (DP-SGD) via Opacus with formal privacy accounting. Every number below is produced by executing the notebook in CI.

---

## Executive Summary

| Mechanism | Privacy Budget | ROC-AUC | Recommendation |
| --- | --- | --- | --- |
| No Privacy (Baseline SVM) | ∞ | 0.9948 | Reference only |
| Feature Laplace (SVM) | ε = 2.51 | 0.5327 | Near chance — not viable |
| Feature Laplace (SVM) | ε = 10.0 | 0.6715 | Weak even at loose budgets |
| Feature Gaussian (SVM) | ε = 10.0, δ = 1e-5 | 0.5301 | Not viable |
| **DP-SGD** | **ε ≈ 2.13, δ = 1e-5** | **0.9940** | **Recommended** |

### Key Finding

**The location of noise injection matters far more than the quantity of noise.**

With honest privacy accounting, feature-level perturbation on this task is close to random guessing at every commonly used privacy budget. DP-SGD maintains near-baseline performance at ε ≈ 2.13 because noise is injected into per-sample-clipped gradients — where batch averaging and training dynamics absorb it — rather than directly into features.

### What the audit changed

An earlier revision reported feature-level Laplace at 0.978 ROC-AUC. That number was an artifact of two calibration errors, both fixed here:

1. **Sensitivity was the max column range (≈ 47,400 from `charges`), applied to every column.** The joint release of all numeric columns was not properly accounted, and one wide column dictated the noise on narrow ones. The pipeline now clips to fixed public bounds, rescales each column to [0, 1], and calibrates noise to the joint sensitivity of the full numeric row (L1 = d for Laplace, L2 = √d for Gaussian) — a single ε covers the whole release.
2. **Clip bounds were quantiles of the private data**, leaking information outside the DP guarantee. Fixed, data-independent domain bounds are used instead.

The corrected feature-level results are much worse — and that is the point: the original headline finding survives the audit only because DP-SGD's numbers were real, not because the feature-level baseline was fairly beaten.

---

## Why This Project Matters

Differential Privacy is increasingly relevant in:

* Healthcare analytics
* AI governance and Responsible AI
* Privacy-preserving machine learning
* Regulatory compliance (GDPR, HIPAA)
* Sensitive-data modelling

This project investigates the practical cost of privacy and evaluates whether modern differential privacy techniques can preserve utility without sacrificing formal guarantees.

---

### For Hiring Managers / Data Governance Teams

This repository demonstrates:

* Differential privacy implementation and evaluation, with mistakes found and corrected transparently
* Privacy-utility trade-off analysis with honest accounting
* Fairness auditing (demographic parity, equalized odds)
* Reproducible ML workflows — the notebook and all reported numbers re-execute in CI
* Automated testing (48 tests, including statistical calibration checks and DP-SGD integration tests)

### For ML Engineers

Key technical lessons:

* Why feature-level differential privacy fails at practical ε on tasks where signal concentrates in one feature
* Why sensitivity must be accounted for the *joint* release, not per column
* Why clip bounds must be data-independent
* Why the classic Gaussian mechanism formula is invalid for ε > 1, and how the analytic Gaussian mechanism (Balle & Wang, 2018) fixes it
* Practical use of Opacus, RDP accounting, and post-processing invariance
* How threshold tuning on the test set silently corrupts fairness metrics

For the full analysis, see `reports/findings_report.md` and `reports/findings_summary.md`.

---

## Overview

The Jupyter notebook (`notebooks/dp_privacy_insurance.ipynb`) walks through:

1. **Data loading and exploration** — the public Health Insurance Cost dataset (1,338 records: age, sex, BMI, children, smoker, region, charges), with class-imbalance analysis.
2. **Baseline models** — SVM and decision tree with leakage-safe preprocessing (transformers fit on the training partition only).
3. **Feature-level DP** — Laplace (ε-DP) and Gaussian ((ε, δ)-DP) noise on numeric features, clipped to fixed public bounds and calibrated to the joint sensitivity of the full release; ROC-AUC swept across ε ∈ [0.01, 10].
4. **DP-SGD** — a small neural network trained with Opacus across noise multipliers 0.5–2.0, with ε reported by the RDP accountant.
5. **Fairness auditing** — demographic parity and equalized odds differences by sex, with the decision threshold tuned on training scores only.

---

## Library

Reusable components live in `src/dp/`:

| Module | Contents |
| --- | --- |
| `mechanisms.py` | Laplace, Gaussian (analytic calibration), randomized response, exponential mechanism |
| `pipeline.py` | Dataset loading/splitting, leakage-safe preprocessing, clipping, bounded joint-release noise |
| `dpsgd.py` | Opacus wrappers: private training loop, ε for a given noise multiplier via RDP accounting |
| `fairness.py` | Demographic parity and equalized odds differences (max−min across groups) |
| `evaluation.py` | Privacy–utility sweeps and plotting |

Example — release all numeric columns under a single ε:

```python
import pandas as pd
from dp import apply_bounded_feature_noise

bounds = pd.DataFrame({
    "lower": {"age": 18.0, "charges": 0.0},
    "upper": {"age": 65.0, "charges": 65_000.0},
})
noisy = apply_bounded_feature_noise(df, bounds, mechanism="laplace", epsilon=1.0)
```

---

## Quickstart

```bash
git clone https://github.com/Ghobi-A/dp-insurance.git
cd dp-insurance
pip install -e .                  # core (mechanisms, pipeline, fairness, evaluation)
pip install -e ".[experimental]"  # + torch/Opacus for the DP-SGD section
pip install -e ".[dev]"           # + pytest/jupyter
```

The dataset (`data/insurance.csv`, the public Health Insurance Cost dataset) is included in the repository.

Run the notebook:

```bash
jupyter notebook notebooks/dp_privacy_insurance.ipynb
# or execute headlessly:
jupyter nbconvert --to notebook --execute --inplace notebooks/dp_privacy_insurance.ipynb
```

Run the tests:

```bash
pytest
```

---

## Privacy Model and Guarantees

Threat model: re-identification, membership inference, and attribute inference attacks by an adversary who observes released features or the trained model.

Mechanisms implemented (`src/dp/mechanisms.py`):

* **Laplace mechanism** — pure ε-DP for numeric features (L1 sensitivity).
* **Gaussian mechanism** — (ε, δ)-DP, calibrated with the *analytic Gaussian mechanism* (Balle & Wang, 2018). Valid for all ε > 0 and strictly less noise than the classic σ = Δ·√(2·ln(1.25/δ))/ε bound, which only holds for ε ≤ 1.
* **Randomized response** — ε-local-DP for binary categorical values.
* **Exponential mechanism** — ε-DP selection from a discrete candidate set, computed in log-space for numerical stability.
* **DP-SGD** (`src/dp/dpsgd.py`) — training-time (ε, δ)-DP via per-sample gradient clipping, Gaussian noise, and Opacus RDP accounting.

Accounting conventions used throughout:

* Feature-level noise is calibrated to the **joint sensitivity of the full numeric release** (all columns under one ε), not per-column budgets.
* Clip bounds are **fixed and data-independent**; data-derived quantile bounds are supported (`compute_clip_bounds`) but explicitly documented as leaking outside the guarantee.
* Threshold tuning and fairness metrics are **post-processing** and consume no additional budget.

---

## Continuous Integration

GitHub Actions (`.github/workflows/run-notebook.yml`) runs on every push and pull request:

* Test suite with coverage across supported Python versions
* Full end-to-end notebook execution (including DP-SGD via CPU torch + Opacus)

This ensures all reported results remain reproducible.

---

## Repository Structure

```text
dp-insurance/
├── notebooks/          # end-to-end analysis notebook (executed in CI)
├── src/dp/             # reusable library (mechanisms, pipeline, dpsgd, fairness, evaluation)
├── tests/              # pytest suite
├── reports/            # findings report + executive summary
├── data/               # public insurance dataset
└── .github/workflows/  # CI definitions
```

---

## License

MIT — see `LICENSE`.
