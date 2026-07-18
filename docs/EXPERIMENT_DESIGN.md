# Experiment design

This document specifies how the multi-task benchmark is constructed so that
every reported number is leakage-safe and reproducible.

## Tasks

Defined in `src/dp/tasks.py` as frozen `BenchmarkTask` dataclasses:

| Task | Target | Excluded features | Notes |
| --- | --- | --- | --- |
| `smoker_without_charges` | `smoker == yes` | `charges` | Headline task A. Stratified splits. |
| `high_cost` | `charges > training-set median` | `charges` | Headline task B. Threshold from the training partition only. |
| `smoker_with_charges_legacy` | `smoker == yes` | none | Legacy easy-reference benchmark; unusually separable. |

## Splitting

Each seed produces a fresh 64/16/20 train/validation/test split
(`test_size=0.2` first, then `val_size=0.2` of the remainder). Smoker tasks
stratify on the target at both stages. For `high_cost` the rows are split
*before* the target exists; the median threshold is then computed on the
training partition and applied to validation/test, so no full-dataset
statistic enters the target definition. Because a median threshold yields
near-balanced classes, unstratified splitting is acceptable there.

## Leakage controls

* Preprocessing (`StandardScaler` + `OneHotEncoder`) is fitted on the
  training partition only — inside a sklearn `Pipeline` for conventional
  models, and explicitly fit-on-train / transform-on-val-test for neural
  models.
* Decision thresholds maximise F1 **on the validation partition**.
* Neural early stopping monitors validation loss.
* The test partition is touched exactly once per run, for final metrics.
* Fairness metrics use the validation-selected threshold, never a
  test-tuned one.

## Repeated seeds and uncertainty

Each configuration runs over ≥5 deterministic seeds (CLI-configurable). A
seed controls the split, model initialisation and training randomness.
Summary tables report mean, standard deviation and a Student-t 95% interval
(`method="repeated_seed_t"` in `src/dp/uncertainty.py`). **These intervals
describe run-to-run variability of the pipeline; they are not
independent-sample statistical guarantees.** A percentile bootstrap over
test examples (`bootstrap_confidence_interval`) is available for
single-model example-level uncertainty and is labelled
`bootstrap_percentile`.

## Metrics

ROC-AUC and PR-AUC are the headline metrics (the tasks can be imbalanced,
so accuracy is reported but never headlined). Also recorded per run:
accuracy, precision, recall, F1, macro-F1, Brier score, expected
calibration error (10 equal-width bins), the selected threshold, fairness
metrics (demographic parity difference, equalized odds difference,
per-group TPR/FPR/sizes with small-group warnings), and for neural/DP-SGD
models a loss-threshold membership-inference AUC and TPR@1%FPR.

## DP-SGD configuration

The privacy grid targets ε ∈ {0.5, 2.0, 6.0} at δ = 1e-5 by default
(strong / moderate / weak privacy), with the non-private neural model as
the ε = ∞ reference. Opacus calibrates the noise multiplier to the target;
the **achieved** ε is read back from the RDP accountant after training and
is what appears in all result tables — no ε is hardcoded. Every run records
target ε, achieved ε, δ, noise multiplier, clipping norm (1.0), batch size
(128), epochs (15) and seed in `dpsgd_metadata.json`.

## Privacy auditing vs accounting

Three notions are kept distinct throughout:

1. **Formal accounting** — the RDP accountant's (ε, δ); the only actual
   guarantee.
2. **Empirical lower bounds** — the one-run auditor and canary-based audits
   in `src/dp/audit.py` / `src/dp/canaries.py` bound ε from below; a low
   empirical bound does *not* verify the formal claim.
3. **Practical attack performance** — LiRA and loss-threshold membership
   inference estimate realistic leakage; chance-level results mean the
   attacks failed, not that the guarantee is proven.

## Reproduction

```bash
pip install -e ".[experimental,dev]"
python -m dp.benchmark --task all --seeds 0 1 2 3 4 --output-dir reports/generated
python -m dp.reporting --results reports/generated --reports-dir reports
```

Outputs: tidy per-seed results (`tidy_results.csv`, schema
`task, model, privacy_mechanism, epsilon, delta, seed, metric, value`),
summary with intervals (`summary.csv`), per-example test predictions
(`predictions.csv`), DP-SGD accounting metadata (`dpsgd_metadata.json`),
figures under `reports/figures/`, tables under `reports/tables/` and
`reports/benchmark_report.md`.
