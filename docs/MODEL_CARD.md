# Model card: dp-insurance benchmark models

## Models covered

* `dummy` — `sklearn.dummy.DummyClassifier(strategy="prior")` (floor reference)
* `logistic` — `LogisticRegression` in a leakage-safe Pipeline
* `svm` — RBF-kernel `SVC` with probability estimates
* `gradient_boosting` — `HistGradientBoostingClassifier`
* `neural` — small MLP (1 hidden layer, 32 units), non-private, validation
  early stopping
* `dpsgd` — the same MLP trained with Opacus DP-SGD across a grid of target
  privacy budgets

## Intended use

Educational / portfolio demonstration of privacy-preserving ML methodology:
model comparison under leakage-safe evaluation, DP-SGD privacy-utility
trade-offs, empirical privacy auditing and group-fairness measurement on a
public teaching dataset.

## Out-of-scope uses

* Any real underwriting, pricing, clinical or eligibility decision.
* Inference about real individuals' smoking status or medical costs.
* Deployment of the trained weights anywhere.

## Evaluation tasks

See `docs/EXPERIMENT_DESIGN.md`. Headline tasks are
`smoker_without_charges` and `high_cost`; `smoker_with_charges_legacy` is an
easy-reference benchmark only.

## Privacy guarantees

Only the `dpsgd` model carries a formal (ε, δ)-DP guarantee, computed by the
Opacus RDP accountant and recorded per run in
`reports/generated/dpsgd_metadata.json` (target ε, achieved ε, δ, noise
multiplier, clipping norm, batch size, epochs, seed). All other models are
non-private. Empirical audits and membership-inference attacks provide
*lower bounds* and practical leakage estimates; they supplement but never
replace the formal accounting, and a failed attack does not prove the
guarantee correct.

## Fairness limitations

Fairness is evaluated only across the binary `sex` attribute (demographic
parity difference, equalized odds difference, per-group TPR/FPR/sizes).
Small groups are flagged as unstable. Absence of a measured gap on this
dataset says nothing about fairness on real populations.

## Known failure modes

* DP-SGD at strong privacy (ε < 1) can collapse towards chance on this
  small dataset, and run-to-run variance grows as ε shrinks.
* The F1-maximising decision threshold is selected on a small validation
  set and is itself noisy across seeds.
* SVM probability estimates rely on internal Platt scaling and can be
  poorly calibrated (see the calibration figures).
