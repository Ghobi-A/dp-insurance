# Data card: Medical Cost Personal Dataset ("insurance.csv")

## Origin

The dataset is the widely used *Medical Cost Personal Dataset* popularised by
Brett Lantz's *Machine Learning with R* and redistributed on Kaggle
(`mirichoi0218/insurance`). It is a teaching dataset of simulated / composite
US health-insurance billing records, not a real clinical or actuarial extract.
A copy is versioned in this repository at `data/insurance.csv`.

## Size

1,338 rows, 7 columns, no missing values.

## Columns

| Column | Type | Description |
| --- | --- | --- |
| `age` | int | Age of the primary beneficiary (18-64) |
| `sex` | categorical | `male` / `female` (used as the sensitive attribute for fairness evaluation) |
| `bmi` | float | Body mass index |
| `children` | int | Number of dependants covered |
| `smoker` | categorical | `yes` / `no` |
| `region` | categorical | US region (`northeast`, `northwest`, `southeast`, `southwest`) |
| `charges` | float | Individual medical costs billed by the insurer |

## Target construction

Three benchmark targets are derived (see `src/dp/tasks.py`):

* **smoker_without_charges** — `smoker == "yes"`, with `charges` removed from
  the predictor set because it is a near-direct proxy for smoking status.
* **high_cost** — `charges > median(charges of the training partition)`.
  The threshold is computed **only from the training partition** of each
  split (never the full dataset), then applied to validation and test rows;
  raw `charges` is removed from the feature set.
* **smoker_with_charges_legacy** — the original dissertation task, kept only
  as an easy-reference benchmark; it is unusually separable.

## Limitations

* Small (1,338 rows): confidence intervals are wide and DP-SGD noise has a
  proportionally large effect.
* No documented collection methodology, consent process or time period.
* Only a binary `sex` attribute is available for fairness analysis; other
  demographic dimensions (race, income, disability) are absent, so fairness
  conclusions are necessarily narrow.
* Region categories are coarse US census regions.

## Demographic limitations

The dataset's demographic distribution is unverifiable and likely not
representative of any real insured population. Group fairness metrics
computed on it demonstrate *methodology*, not real-world disparities.

## Not clinical evidence

This dataset must not be treated as real clinical or actuarial evidence.
It is used here solely as a benchmark substrate for privacy-preserving
machine-learning methodology. No conclusion in this repository about
smoking, cost or demographics should be applied to real people.
