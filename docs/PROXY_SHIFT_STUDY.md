# Proxy-shift utility study

## Research question and current scope

When a DP-SGD setting meets a utility-loss target on the source distribution,
does the same setting remain acceptable when a predictive proxy becomes less
reliable? This study continues the MSc's tabular privacy–utility work. It is
an exploratory implementation, not an established novel result or a finished
submission. The old membership-leakage study and its frozen ACS protocol are
separate; this runner does not modify their gates or primary attacks.

The implemented paths are controlled synthetic experiments and the public
insurance smoker task with hypothetical charge unavailability. Both run from
data generation/preparation through training, accounting, operating-point
selection, held-out evaluation, CSVs, metadata, plots and a report. New runs
also write `components/` with the clipping-versus-added-noise decomposition.
Executed findings and publication limitations are recorded in
[PILOT_FINDINGS.md](../reports/proxy_shift/PILOT_FINDINGS.md).

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[experimental,dev]'
python -m dp.proxy_shift --config research/proxy_shift/smoke.json \
  --output-dir reports/proxy_shift/smoke
python -m dp.proxy_shift --config research/proxy_shift/pilot.json \
  --output-dir reports/proxy_shift/synthetic_pilot
python -m dp.proxy_shift --config research/proxy_shift/insurance.json \
  --insurance-path data/insurance.csv \
  --output-dir reports/proxy_shift/insurance_pilot
```

Output directories must be empty/new to prevent overwriting evidence. The
manifest records the complete configuration, software versions, code hash,
git revision/dirty status, actual accountant budgets, noise, sample rate,
clipping frequency, step counts and insurance file fingerprint. No model or
private row predictions are published. Outputs are public/synthetic research
results; the whole search is not claimed to be a composed private release.

## Frozen exploratory configurations

Before examining pilot results: four synthetic source strengths (0.60, 0.75,
0.90, 0.98), epsilon targets 1/2/4/8, seeds 0–4, 1,000 training records and
2,000 records in each independent evaluation cohort, 15 epochs, batch 128,
SGD learning rate 0.1, clip norm 1, delta 1e-5. This trains 80 DP models and
40 matched unnoised/clipping-only controls. It is a pilot, not a statistical
preregistration or a powered confirmatory sample size.

The utility-loss allowance is 0.03 AUC, the absolute utility floor is 0.70
AUC, and the maximum allowed achieved per-model epsilon is 8. These are
illustrative research decision thresholds, not an approved privacy policy.
No seed, metric, proxy level or threshold may be silently retuned and reported
as the same experiment. New exploratory configurations need new names.

`insurance_training_diagnostic.json` is a **post-pilot exploratory amendment**:
100 epochs and learning rate 0.3, with every seed/budget retained and unchanged
decision thresholds. It checks whether the initial weak source performance
reflects insufficient optimisation. It is not confirmatory, and it cannot
validate a hypothesis against the already examined test cohort. Outputs must
use `reports/proxy_shift/insurance_training_diagnostic`, preserving the first
pilot. It is executed with the same command and its own config filename.

## Generative intervention and exact source control

Labels are independent Bernoulli(0.5). The stable feature is
`tanh(0.7 * (2*y - 1) + normal_noise)`. A binary proxy agrees with the label
with probability `strength`. Six further features are bounded noise. All
transforms are fixed row-wise functions; no scaler or category vocabulary is
fitted to training records.

Proxy retention `r` changes agreement probability to
`0.5 + r * (strength - 0.5)`. Source retention is 1; stress-validation
retentions are 0.5/0.75; unseen test retentions are 0/0.25. Source and shifted
conditions share evaluation labels/stable features/random uniforms: paired
comparisons change only the proxy coordinate. Independent random streams
construct train, validation and test data. The stable-signal counterpart has
identical source data and preserves the original signal at evaluation. It
controls source learnability exactly without another model fit. It is an
ideal counterfactual, not evidence for a new robust-learning algorithm.

## Insurance case

Predict smoker using age, charges, BMI, children and fixed one-hot sex/region
categories. Public bounds are age 100, charges 100,000, BMI 80 and children
10; numeric features are divided by those constants and clipped to [0,1].
Splits are an independent public random permutation (60/20/20), with both
classes required in each partition. No SMOTE, scaler fitting, median-derived
target or private feature selection is used.

A fixed per-row uniform selects charges to be zero-imputed as retention
decreases. This represents a particular missing-proxy policy; it is not
observed insurance drift, a causal intervention on smoking, or evidence that
charges will disappear in deployment. Zero-imputation itself contributes to
the failure. The literal no-charges task in the legacy benchmark is a
different feature-availability experiment.

## Matched controls and privacy boundary

Each strength/seed trains an ordinary model, a zero-noise per-example clipped
model and four DP-SGD models. Initial weights, architecture (16-unit tanh MLP),
SGD schedule, Poisson sample stream and expected-batch normalisation match.
Noise uses a separate random stream, so noise generation does not alter the
sampling schedule. Ordinary versus clipped separates clipping; clipped versus
DP separates added noise under this optimisation recipe. This is not an
independently tuned comparison of each method's best possible performance.

Opacus uses the RDP accountant and Poisson sampling. Empty Poisson batches
still execute the noise/accounting step; tests check this case. Requested
budgets and actual achieved budgets are recorded separately. Ordinary and
clipping-only controls have no DP guarantee. Secure RNG is disabled for
reproducible experiments: these checkpoints are not production releases.

The reported accountant describes per-model add/remove prepared-record
training, conditional on public partition selection and fixed preparation.
Insurance is already public research data. Do not describe this experiment
as protecting its published input file. Fixed preparation avoids the previous
target-fitted preprocessing problem, but does not compose an entire private
search, validation and metrics-release pipeline. Public-record duplicates,
user-level grouping and other adjacencies require their own analysis.

## Decision evaluation and uncertainty

`gap(e) = AUC_ordinary(e) - AUC_DP(e)`;
`interaction(e) = gap(e) - gap(source)` within the same evaluation cohort.
Both positive and negative interactions are possible. Ordinary shift damage
is not automatically DP-specific harm. An absolute utility floor prevents
claiming success merely because DP retains a useless reference model's AUC.

Three operating-point rules choose the smallest achieved-epsilon candidate
within the cap that meets the loss margin and utility floor on all their
validation environments:

* Source-only: source validation.
* Generic shift: source plus two fixed additive-noise environments on every
  non-proxy coordinate. Perturbation magnitudes are `0.5 * (1-retention)`.
* Proxy stress: source plus the two prespecified proxy-retention environments.

Generic and proxy rules have equal numbers of evaluations and share the
validation cohort; their perturbation severity/information is not guaranteed
equivalent. Source-only intentionally uses fewer environments. A proxy-rule
advantage requires further comparisons across severity/information budgets.
Candidate selection uses exploratory point estimates, not confidence-bound
certification. Each rule abstains if no candidate qualifies. Test outcomes
are queried only after selection, and test rows are rejected by the selector.
The runner materialises evaluation metrics for all candidates; the pure
selection function receives only validation rows and never chooses from test
outcomes. This is an exploratory automated comparison, not a sealed test set.
Report selected/abstained counts, loss and absolute-floor violations, achieved
budgets and failure rates conditional on selection. Abstention is not success.

Metrics include ROC-AUC, PR-AUC and balanced accuracy at a fixed score threshold
of 0.5. Descriptive t intervals use one observation per training seed per cell;
rows/environments are not independent training replicates. They do not fully
integrate finite-evaluation-sample uncertainty, adjust multiplicity or establish
equivalence. Pooled grid plots are descriptive, without inferential error bars.
The next confirmatory protocol needs seed counts from pilot variance, specified
primary contrasts, evaluation-cohort resampling and fresh seeds/environments.

## Literature boundary and publication gate

The intended distinction is reliability of utility-based recommendations,
not discovering stable features, spurious privacy leakage or privacy–utility
prediction for the first time. Closest primary sources:

* [Mahajan, Tople & Sharma (2021)](https://arxiv.org/abs/2110.03369): stable
  feature learning already improves utility at the same DP guarantee.
* [Zhang, Pang & Mauw (2025)](https://arxiv.org/abs/2505.20095): spurious
  groups, robust training, LiRA and DP-SGD utility/privacy comparisons.
* [Hod, Rosenblatt & Stoyanovich (2025)](https://arxiv.org/abs/2504.14368):
  surrogate tabular data and privacy–utility estimation for synthesis.
* [Jarin & Eshete / DP-UTIL (2022)](https://arxiv.org/abs/2112.12998):
  existing multi-location perturbation benchmarks.

Proceed only if a useful recommendation failure/interaction survives matched
controls and independent environments, or an adequately powered null answers
a concrete unresolved question. Do not claim novelty from code completeness.
If generic validation matches proxy stress, or failures are only ordinary
information loss, the proposed contribution needs narrowing.

## Do we need another dataset?

**For a broad paper, yes: real ACS external validation should be in the main
study, not indefinitely deferred to future work.** Synthetic interventions
identify the controlled mechanism; insurance establishes dissertation
continuity. Neither shows transport to naturally occurring shifts. ACS offers
state/year environments and an income task, with an independent public
feature/schema specification and held-out states/years. The existing frozen
50,000-row CA membership-audit slice is not an external-validation design.

[Ding et al., Retiring Adult (2021)](https://arxiv.org/abs/2108.04884) provides
the Folktables/ACS framework. A proposed next-stage design is source CA-2018,
validation OR-2018/WA-2018 and final NV-2018/AZ-2018, subject to an up-front
data/schema review. These are proposed new utility-study splits, not executed
experiments and not a claim that these states isolate one proxy mechanism.
Use a public fixed income threshold, fixed categories/bounds, and no tuning
on final states. Report selection uncertainty and an abstention rule before
running confirmatory comparisons. State/year choice must be fixed before
outcomes; the income definition and dollar comparability need documentation.

**A third independent tabular domain can be future scope** for the narrow
synthetic+insurance+ACS paper. It becomes a main-study requirement if the
claim is broad cross-domain predictive reliability. Adult is not a strong
independent addition to ACS, and substituting one random split is not external
shift validation. No second dataset was downloaded or its results invented.
