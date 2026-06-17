# Differential Privacy in Machine Learning

Quick Answer: DP-SGD achieves ε ≈ 2.13 (strict privacy) with ROC-AUC ≈ 0.993 on smoking classification. Feature-level noise unsuitable for privacy budgets below ε = 1.0.

| Mechanism | Privacy Budget | Accuracy (ROC-AUC) | Recommendation |
|-----------|---|---|---|
| No privacy (baseline) | ∞ | 0.9948 | Reference only |
| Feature Laplace | ε = 1.0 | 0.9780 | Only for loose privacy (ε > 1.0) |
| Feature Gaussian | ε = 1.0 | 0.8586 | Not recommended |
| **DP-SGD** | **ε = 2.13** | **0.9940** | **✓ Recommended** |

**Key Insight:** Feature-level noise fails at strict privacy levels (ε < 1.0) because clipped numeric features have limited range; noise becomes dominant. DP-SGD, which adds noise during training, preserves signal through gradient averaging and is suitable for strict budgets.

### 🏢 For Hiring Managers / Data Governance Teams
This repository demonstrates:
- **Rigorous DP implementation** — correct sensitivity calculations, privacy accounting, composition limits
- **Privacy-utility tradeoff analysis** — quantified cost of privacy on model accuracy
- **Fairness-aware thinking** — monitoring demographic parity under privacy constraints
- **Reproducible ML practices** — tests, automated pipelines, transparent reporting

### 👨‍💻 For ML Engineers
Key technical insights:
- **Why feature-level DP fails at low ε:** Clipping bounds sensitivity + per-feature noise amplification
- **Why DP-SGD works better:** Noise added in gradient computation, averaged over batch
- **How to set sensitivity:** Quantile-based clipping with L1 bounds

For a deeper theoretical background on differential privacy concepts, see the
[findings report](reports/findings_report.md).

## Overview

The included Jupyter notebook (`notebooks/dp_privacy_insurance.ipynb`) walks
through the following steps:

1. **Data loading and exploration** –  We start with a sample insurance dataset containing demographic and health‑related features (age, sex, BMI, number of children, smoker status, region and charges).  Basic exploratory analysis is performed on numeric features.
2. **Noise injection** –  Laplace and Gaussian noise are added to the dataset at
   both random and fixed magnitudes to provide differential‑privacy guarantees.
   The noise functions accept a privacy budget ``epsilon`` (and optional
   ``sensitivity``), computing any required scale internally.
3. **Anonymisation** –  We anonymise certain quasi‑identifiers by grouping continuous features into buckets (e.g. converting raw age into age groups) and binarising the number of children.  This improves privacy through *k‑anonymity*, *l‑diversity* and *t‑closeness* without destroying utility.
4. **Machine‑learning models** –  Three classifiers are trained on the original and noised datasets: Support Vector Machines, a basic Neural Network and a Decision Tree.  The goal is to understand how privacy noise affects predictive performance.
5. **Fairness metrics** –  For each trained model we compute demographic‑parity and equal‑opportunity metrics to see how the addition of noise impacts fairness across sensitive groups.

The notebook is fully executable and makes no assumptions about prior infrastructure—open it in JupyterLab or VS Code and run the cells from top to bottom.

## Quickstart

1. **Clone this repository** and change into the directory:

   ```bash
   git clone https://github.com/your‑username/your‑repo.git
   cd your‑repo
   ```

2. **Install the project** in editable mode so that its modules and CLI are available system‑wide:

   ```bash
   pip install -e .
   ```

   This installs the package along with the required dependencies.

3. **Add the dataset**.  The notebook expects a CSV file named
   `data/insurance.csv` in the repository root.  You can supply your own dataset
   or download a public insurance dataset such as the one from the Kaggle
   medical‑cost dataset.

4. **Launch Jupyter** and run the notebook:

   ```bash
   jupyter notebook notebooks/dp_privacy_insurance.ipynb
   ```

Alternatively, you can execute the notebook from the command line using `nbconvert`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/dp_privacy_insurance.ipynb
```

## Privacy Model and Guarantees

This project uses additive Laplace and Gaussian noise as standard
differential-privacy mechanisms. The threat model assumes an attacker with
auxiliary information who attempts re-identification or attribute inference
from released data or model outputs. Under the stated sensitivity and epsilon
choices, the Laplace/Gaussian mechanisms provide formal DP guarantees. Other
anonymisation steps in this repo (e.g., k-anonymity, l-diversity, t-closeness)
are heuristic, non-DP baselines and should not be treated as formal
privacy guarantees.

The exponential mechanism (`exponential_mechanism` in `src/dp/mechanisms.py`)
is now implemented.  It selects from a discrete set of candidates with
probability proportional to `exp(ε · score / (2 · sensitivity))`, providing
ε-differential privacy with respect to the global L1 sensitivity of the
utility function.  The implementation uses numerically stable log-space
computation and is fully covered by the test suite.

## Pipeline entry point

Reusable pipeline components live in `src/dp/pipeline.py`. These functions load
the dataset, split it into train/test partitions, and build a preprocessing
transformer that is fit only on the training data to avoid leakage.

## Continuous integration

For convenience, this repository includes a GitHub Actions workflow that will
run the notebook on every push and pull request.  It sets up a Python
environment, installs the dependencies and executes
`notebooks/dp_privacy_insurance.ipynb` using `nbconvert`.  If the notebook runs
successfully, the workflow will finish with a green check mark.  You can find
the workflow definition in `.github/workflows/run‑notebook.yml`.

## Removing private materials

The original academic report (PDF) and any institution‑specific content are intentionally **not** tracked in this repository.  The `.gitignore` file excludes the uploaded dissertation to avoid committing it.  If you add your own private files (e.g. PDFs, datasets with personal information), update `.gitignore` accordingly.

## License

This project is licensed under the [MIT License](LICENSE).
