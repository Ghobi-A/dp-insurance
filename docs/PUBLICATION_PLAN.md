# Publication plan — next stage, 30 September 2026

**Execution update:** stages 1–4 below have run, including the 20 fresh-seed
synthetic primary comparison. The [findings](../reports/publication_stage/FINDINGS.md)
and [editable manuscript](../paper/proxy_shift_manuscript.md) separate completed
evidence from the remaining submission gate. ACS remains external pilot
evidence; the prospective confirmation applies to one synthetic contrast.

## Intended paper and limits

Working title: **Stress-Testing Utility Recommendations for DP-SGD in Tabular
Classification**. The candidate contribution is an empirical recommendation
benchmark with matched clipping/noise controls and external environments.
Neither stable-feature utility benefits nor clipping bias are new phenomena.
The manuscript must compare directly with Mahajan et al. (2021), Zhang et al.
(2025), DP-UTIL (2022), and the DP utility/disparate-impact literature.

The first pilot's pooled interaction was clipping-dominated. Improving its
training recipe and changing the dataset cannot establish novelty by itself.
A publishable benchmark needs a precise decision failure, comparisons that
support a useful conclusion, statistical calibration, external evidence and
a reproducible claim boundary. No outcome is promised in this protocol.

## Stage 1: fresh-seed clipping development

Run `research/proxy_shift/clipping_development.json`: strengths .75/.90,
clipping norms .5/1/2/5, epsilon 2/8, seeds 10–14, 3,000 training rows, 2,000
rows in independent validation/test cohorts, 50 epochs, SGD LR .1, batch 256,
delta 1e-5. The original loss margin .03 and AUC floor .70 remain unchanged.
This is development after the first pilot, not retrospective confirmation.

Compare total, clipping and added-noise interactions across norms. Record all
cells and test whether the original pattern survives improved optimisation.
Do not pick whichever norm or seed makes a publication story most attractive.

## Stage 2: ACS external pilot

Run `research/proxy_shift/acs_external.json`: source CA-2018; OR-2018 and
WA-2018 provide natural-shift validation; NV-2018 and AZ-2018 are held-out
external tests. This follows the state split proposed before these outcomes.
Use seeds 10–14, clipping norms 1/5, epsilon 2/8 and the same training schedule.

ACSIncome eligibility and income threshold >50,000 are fixed. Use bounded
numeric features, fixed public one-hot domains and public modulo hashing for
occupation/place-of-birth codes. This is a documented representation variant,
not the unmodified Folktables feature representation. Hash collisions can
affect utility; an alternative representation is a future sensitivity check.
The task is unweighted public-record classification, not a survey-weighted
population estimate. All states use 2018 nominal dollars.

Whole households are allocated to disjoint source train/validation/test
blocks. Occupation-block zero-masking supplies a hypothetical proxy stress
validation rule. It is not a causal explanation of natural state shifts.
All source choices are written to JSON before external test data are loaded
or scored. The older frozen CA membership-audit protocol remains separate.

## Stage 3: uncertainty and decision evaluation

Compare point-estimate and uncertainty-aware choices for source-only,
generic/natural-shift, and proxy-stress validation. Use paired AUC structural
influences with a cluster-sandwich variance. ACS clusters are households;
synthetic clusters are records. One-sided t bounds apply a Bonferroni family
over candidates, clipping norms, validation environments and the gap/floor
endpoints. At least 20 records/class and 30 clusters are required; degenerate
boundary AUCs cannot certify decisions. Record selection, abstention, budgets,
gap violations and absolute-floor violations separately.

These are **asymptotic** validation bounds conditional on trained models and
the sampled environments. They are not finite-sample guarantees, not a bound
on unseen-state failure, and not a new inference method. Validation cohorts
can share rows; the union bound does not require independence between
endpoints, but the cluster-variance approximation needs independent clusters.
Survey dependence, small classes and representation choices limit coverage.
The next confirmatory study must evaluate empirical interval coverage and
cluster-bootstrap sensitivity, not assume nominal coverage from unit tests.

## Stage 4: confirmatory protocol and power

After development, freeze one primary clipping contrast (provisionally 1 vs 5),
one primary epsilon (2), one primary interaction and a practical effect margin
of .01 AUC. Estimate paired seed-level variance on development data. Use a
prospective paired-mean sample-size calculation at alpha .05 / power .80 as a
planning approximation, with a minimum of 20 new training seeds. Fresh seeds
100+ and new source/validation/test cohorts must be independent of these runs.

This does not alone provide external-domain replication: five fixed states
cannot establish universal transportability. Freeze the cohort sampling,
confidence procedure and primary claim before confirmatory runs. Define a
meaningful equivalence margin if concluding a negligible noise interaction;
a nonsignificant test is not equivalence. Changes require a dated amendment.

## Stage 5: manuscript and release

Maintain an editable manuscript with methods, comparison to closest papers,
all positive/null findings, accountant/release scope, limitations and a
reproduction command. Use actual outcomes; label exploratory and confirmatory
results distinctly. Add a second independent domain only for broader
cross-domain claims. Insurance + controlled synthetic + ACS is adequate scope
for a narrow benchmark candidate, not proof of a general method.

The per-model accountant covers prepared-record training. Public research
evaluation/selection is not a composed private release. Household-safe
partitions and clustered utility inference do not turn row DP into household
DP. Dataset availability and utility do not determine an acceptable epsilon.

Submission gate: a distinct supported claim against prior work; calibrated
inference and prospective confirmatory evidence; adequate baseline training;
reproducible external results; manuscript claims matching every reported
number. If the new evidence reproduces established behaviour only, narrow to
a replication/negative-result benchmark rather than assert methods novelty.

## Run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[experimental,dev]'
python -m dp.publication_study --config research/proxy_shift/clipping_development.json \
  --output-dir reports/publication_stage/clipping_development
python -m dp.publication_study --config research/proxy_shift/acs_external.json \
  --cache-dir /tmp/dp-publication-acs \
  --output-dir reports/publication_stage/acs_external
python -m dp.publication_diagnostics \
  --calibration-output reports/publication_stage/interval_calibration.json
python -m dp.publication_diagnostics \
  --acs-bootstrap-config research/proxy_shift/acs_external.json \
  --bootstrap-output reports/publication_stage/acs_bootstrap.json
python -m dp.publication_diagnostics \
  --development-dir reports/publication_stage/clipping_development \
  --base-config research/proxy_shift/clipping_development.json \
  --confirmation-config research/proxy_shift/confirmation.json
# The committed confirmation.json already exists. For independent reproduction,
# compare a newly generated file at another path with that frozen protocol.
# Freeze/commit the protocol before starting confirmation.
python -m dp.publication_study --config research/proxy_shift/confirmation.json \
  --output-dir reports/publication_stage/confirmation
python -m dp.publication_diagnostics \
  --confirmation-dir reports/publication_stage/confirmation
python research/proxy_shift/make_publication_figures.py
```

Raw ACS archives and prepared records stay outside Git. The saved manifest
contains source URLs, checksums, representation, full config and accounting.
Output directories are never overwritten. Failed partial runs remain evidence
of failures; retries use a new output path and retain the original protocol.

Primary data source: [Folktables](https://github.com/socialfoundations/folktables)
and [Census 2018 PUMS](https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/).
Inference background: [DeLong structural components and clustered ROC analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC8586066/).
Known clipping/noise utility effects: [Bagdasaryan et al. (2019)](https://arxiv.org/abs/1905.12101).
