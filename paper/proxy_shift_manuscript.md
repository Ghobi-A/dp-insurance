# Stress-Testing Utility Recommendations for DP-SGD in Tabular Classification

**Editable research draft — updated 1 October 2026.** Author names and affiliations
are to be supplied by the author. This draft replaces neither the MSc thesis
nor its historical PDF. It reports an empirical benchmark, with no assertion
that a new DP mechanism or inference method has been invented.

## Abstract

Selecting a differentially private training budget using source-distribution
utility can obscure two distinct failures: a loss relative to a non-private
reference and an absolute performance floor under distribution shift. We
evaluate validation-only recommendations for DP-SGD, separating matched
ordinary, clipping-only and noisy training and testing frozen decisions in
held-out environments. Fresh-seed synthetic development shows that the
earlier clipping-dominated interaction depends strongly on the training
recipe and clipping norm. A five-state ACS income pilot shows that conservative
validation bounds can change recommendations to abstentions, while a larger
clipping norm improves source utility and enables selections that pass the
two external-state tests. These results motivate configuration-aware auditing
of utility recommendations; they do not establish universal benefits of proxy
stress testing. The primary fresh-seed synthetic contrast and its uncertainty
are reported below. A separate 20-seed ACS 2017 replication under two representations finds
that joint configuration search restores recommendations, but source-only
joint search has the same successful yield as shift-aware search. Broader
simultaneous calibration exposes undercoverage in a small imbalanced clustered
scenario. This is a candidate replication and negative-result benchmark; its
incremental contribution requires external scholarly review.

## 1. Research question and relationship to the MSc

The original MSc compared privacy-related perturbations on a small public
insurance table, including smoker prediction with a strong charges proxy.
The revised benchmark introduced accounted DP-SGD, leakage-safe evaluation
and empirical privacy auditing. The present study remains adjacent to that
work through tabular prediction, privacy–utility trade-offs and honest
measurement, but asks a narrower decision question:

> When a validation rule recommends a DP-SGD budget subject to a relative
> utility-loss margin and an absolute AUC floor, which recommendations survive
> held-out distribution shift, and how sensitive is the answer to clipping,
> training configuration and validation uncertainty?

This study concerns utility reliability. It does not infer membership privacy
from utility, equate a failed attack with absence of leakage, or revive the
unsupported sex-specific leakage hypothesis. The earlier LiRA/power work
remains a separate study. Insurance motivates the problem; controlled
synthetic data and ACS provide the current evidence.

## 2. Closest prior work and bounded contribution

Mahajan, Tople and Sharma [1] connect stable representations, distribution
generalization and membership inference, including utility differences at
the same formal DP guarantee. Therefore, discovering that stable features
can improve private utility is not a defensible novelty claim here.

Zhang, Pang and Mauw [2] study privacy disparities associated with spurious
groups and the distinction between prediction robustness and memorization.
Our utility recommendation evaluation does not reproduce their privacy
conclusions or claim to introduce spurious-feature privacy analysis.

DP-UTIL [3] compares utility and privacy across perturbation locations.
Bagdasaryan and Shmatikov [4] explain disparate accuracy effects of clipping
and noise. Consequently, broad mechanism comparisons and the existence of
clipping-related utility loss are established background.

The candidate incremental contribution is a reproducible **decision audit**:
frozen validation-only budget choices; explicit abstention and two failure
criteria; matched clipping/noise controls; paired clustered uncertainty;
development separated from a prospective synthetic primary comparison; and
external state tests. None of these ingredients alone is new. Their combination
may support a useful empirical benchmark, but a combination is not itself
proof of publication-level novelty. The results should be positioned as
recipe sensitivity and negative evidence against broad stress-test claims.

A subsequent targeted survey adds important constraints. Morsbach et al. [8]
already establish joint clipping/learning-rate effects. Panda et al. [9]
combine private HPO with OOD evaluation. DomainBed [10] treats model selection
as essential to domain generalization. Accuracy First [11] and Brownian Noise
Reduction [12] already optimize privacy subject to accuracy requirements.
Consequently, neither the selection problem nor a joint search is our method
novelty. The [full survey](../docs/CONFIGURATION_SELECTION_LITERATURE.md) maps
sixteen related sources and the remaining evidence boundary.

## 3. Methods

### 3.1 Models and training controls

The study uses a two-layer MLP (16 hidden units, tanh, one binary logit),
cross-entropy and SGD. All controls use matched initialization and Poisson
sampling streams with a common expected-batch gradient normalization.
The three conditions are ordinary training, per-record clipped training with
zero noise, and DP-SGD at requested ε values 2 or 8. Noise uses a separate,
fixed reproducible random stream. The RDP accountant records achieved ε,
δ=10⁻⁵, sample rate, noise multiplier, steps and empty batches per model.

Development uses 3,000 training rows, independent 2,000-row validation/test
cohorts, batch 256, 50 epochs and learning rate 0.1. Synthetic development
crosses strengths 0.75/0.90, clipping norms 0.5/1/2/5 and five seeds 10–14.
ACS crosses norms 1/5 and the same five seeds. These account for 160 and 40
model fits respectively, including ordinary controls repeated across norms.
Those repeated controls are not independent observations.

This is a controlled recipe comparison, not a claim that each configuration
has been independently optimized. No alternative architecture benchmark or
convergence guarantee is established. Raising the norm also raises absolute
noise for a given noise multiplier, so the contrast is a training-configuration
effect rather than an isolated causal effect of clipping.

### 3.2 Environments and data

The synthetic generator and all parameters are available in
`src/dp/proxy_shift.py` and the JSON protocols. Train, validation and test
streams are independent within seed. Validation proxy retention is 0.5/0.75;
test retention is 0/0.25. Generic perturbations provide a comparison validation
rule. Proxy intervention is controlled by construction, not inferred from
real-world correlations.

The external pilot downloads official 2018 one-year ACS person archives.
California supplies source train/validation/test partitions; Oregon and
Washington supply natural-shift validation; Nevada and Arizona supply
external tests. ACSIncome eligibility is age >16, income >100, working hours
>0 and person weight ≥1, with binary target income >$50,000 [5,6]. Classification
is unweighted and concerns public sampled records, not survey-weighted
population income estimates.

Encoding uses fixed public bounds for numeric features, fixed categorical
domains and deterministic modulo hashing for occupation and birthplace.
It differs from the standard Folktables representation. Households are
allocated wholly to distinct source partitions; the last household in a
partition can be truncated, with its unused records excluded from subsequent
partitions. Occupation-block zero masking is a hypothetical validation stress
test; it is not a causal explanation of state differences. State shifts can
change multiple variables and relationships simultaneously.

ACS training seeds resample from the same fixed state archives, so they are
algorithm/cohort replications rather than independent state replications.
The five states and one year restrict external validity.

### 3.3 Recommendations and held-out evaluation

For each configuration the recommendation chooses the smallest achieved ε
among candidate budgets satisfying both AUC loss ≤0.03 relative to the
ordinary reference and private AUC ≥0.70 in every specified validation
environment. Failure to meet these constraints yields abstention. Source-only,
proxy-stress and generic/natural-shift rules are compared using point estimates
or uncertainty bounds. Privacy caps are fixed by the protocols, not justified
as acceptable deployment privacy levels.

All choices are persisted before external test scoring; ACS external test
records are loaded after choices are written. Test failure occurs when either
the relative gap exceeds 0.03 or private AUC falls below 0.70. Counts are
reported alongside abstentions. Evaluations in two test environments share a
model and cannot be treated as independent Bernoulli trials. Increasing
abstention can reduce selected-case failures without improving budget choice.

### 3.4 Paired uncertainty and decomposition

AUC structural influences preserve ties and covariance between predictions
on the same labeled cohort. Variance sums record influences within households
for ACS, then uses a cluster sandwich estimate; synthetic records form their
own clusters. One-sided t bounds use a Bonferroni family across validation
environments, candidate budgets, clipping norms and both gap/floor endpoints.
At least 20 records per class and 30 clusters are required. Boundary AUCs with
degenerate variance are marked unsupported rather than certified.

These are asymptotic, conditional-on-model bounds for sampled validation
environments. They do not certify unseen-state utility or constitute a new
confidence procedure. Cluster independence is an approximation for ACS and
does not model all survey design dependence. A Gaussian clustered simulation
and a paired household bootstrap provide limited implementation/calibration
checks; the broader simultaneous check in Section 4.5 finds undercoverage in one
small imbalanced scenario and does not validate ACS survey-design coverage.

Let Aᵒ, Aᶜ, Aᵈ denote ordinary, clipped and DP AUC. The relative gap decomposes
exactly as (Aᵒ−Aᵈ)=(Aᵒ−Aᶜ)+(Aᶜ−Aᵈ). The corresponding shift interaction subtracts
the source gap from the shifted gap. Components are matched training
comparisons, not a general causal mediation analysis; added-noise components
can be negative.

### 3.5 Prospective primary comparison

After development, the protocol fixed one primary contrast: norm-1 minus
norm-5 **total gap interaction**, at ε=2, strength 0.90 and test retention 0.
The development paired SD was 0.0087875 AUC. Normal-approximation planning used
a 0.01 effect, two-sided α=0.05 and target power 0.80, with a minimum 20 fresh
seeds. The protocol uses seeds 100–119, 4,000 validation/test rows and unchanged
training settings. It was committed as `1feb9fe` before that run started.
This is a locally prospective protocol, not an independently timestamped
public preregistration. Five-seed variance planning is uncertain.

The primary estimate uses the paired seed mean and Student-t 95% interval.
All other confirmation-run summaries are descriptive. A non-significant test
does not imply equivalence; practical interpretation must use the interval
and the specified 0.01 margin. The tested population is the specified synthetic
generator/recipe, not arbitrary tabular tasks.

## 4. Results

### 4.1 Synthetic development

At norm 0.5 and ε=2, mean complete-proxy-removal interaction was approximately
0.02994 AUC, consisting of 0.02964 clipping-associated and 0.00030 added-noise
components (pooled over both development strengths for description only).
At norm 1 the corresponding mean was approximately 0.00093; at norm 5 it was
approximately −0.00027. These changes challenge a recipe-independent
clipping-dominance story. They also show why further mechanistic claims need
stronger comparisons than a single low-norm pilot.

Source-only selection failed in 12/20 held-out evaluations at norm 0.5 and
5/20 at norms 1, 2 and 5. Proxy stress at norm 0.5 selected seven of ten
seed/strength cases and failed in 6/14 evaluations; all higher-norm rules
selected ten cases and failed in 5/20. Point and bounded decisions were equal
in this development run. These counts do not demonstrate that proxy stress
generally improves transport. Absolute-floor violations can remain when the
ordinary reference also loses signal under shift.

### 4.2 ACS external pilot

At ε=2, ordinary/private source-test mean AUCs were 0.85578/0.82488 for norm 1
and 0.85578/0.84627 for norm 5. Norm-1 mean clipped-only gradient clipping rate
was approximately 54.9%, versus 1.69% for norm 5. Under norm 1, point source-only
and natural-shift rules each selected three seeds and failed in one of six
external evaluations; proxy stress selected two and failed in one of four.
Bounded selection abstained on all five seeds under every norm-1 rule.

Under norm 5, every rule and bound choice selected all five seeds and all ten
external evaluations passed. This is a pilot observation, not a population
failure-rate bound. Mean private AUC at ε=2 was 0.82875 in AZ and 0.81291 in NV;
the corresponding ordinary means were 0.84276 and 0.82428. The larger norm
reduces clipping-related source utility loss here while leaving some
added-noise loss. The external relative gaps do not uniformly amplify source
gaps; natural shift is not equivalent to synthetic proxy removal.

### 4.3 Interval checks

Across 400 Gaussian clustered simulations, marginal one-sided 95% coverage
was 0.950 for the gap upper bound and 0.960 for the AUC lower bound. Nominal
Monte Carlo standard error is 0.01090. This single DGP does not validate ACS
survey dependence, tail cases or simultaneous selection coverage.

For the predetermined ACS source-validation seed-10, norm-1, ε=2 cell, the
paired influence gap upper bound was 0.03419 and the 999-replicate paired
household percentile-bootstrap bound was 0.03436. Candidate AUC lower bounds
were 0.82184 and 0.82052. Both marginal gap bounds exceed the 0.03 tolerance.
This is a sensitivity check using two diagnostic retrains, not additional
independent experimental evidence or the multiplicity-adjusted selection rule.

### 4.4 Fresh-seed primary result

The exact primary estimate and interval are recorded in
`reports/publication_stage/confirmation/primary_result.json`. Across 20 fresh
seeds, the norm-1 minus norm-5 interaction was **0.008275 AUC**, with paired-mean
95% interval **[0.004092, 0.012458]** and two-sided p=0.000556. The positive
contrast is distinguishable from zero in the specified experiment. Its
interval straddles the planned 0.01 practical-effect margin; it establishes
neither an effect above that margin nor equivalence within that margin.

Descriptively, every rule selected all 20 seeds under both norms. Norm 1 failed
in 20/40 held-out evaluations and norm 5 in 19/40; proxy stress and bounded
selection made the same choices as source-only selection. These paired
environment counts are not the primary statistical test and do not establish
a recommendation-policy improvement. There were 120 confirmation fits,
including repeated ordinary controls.

### 4.5 Independent ACS 2017 configuration-selection extension

Before external scoring, a separate protocol fixed CA source, OR/WA validation
and previously unused CO/UT external tests. Twenty fresh seeds 200–219 were run
for each of the original hashed representation and a public decimal-digit
one-hot sensitivity. Three recipes (16-unit MLP/50 epochs/LR .1,
32-unit MLP/100 epochs/LR .05, logistic/100 epochs/LR .1), norms 1/5 and
epsilon 2/8 provide 420 fits per encoding, **840 in total**. Cohort sizes,
batch size and utility limits remain 3000/2000, 256, .03 gap and .70 floor.
The common ordinary reference is selected using source-validation AUC only.
All decisions and the reference are persisted before loading external tests.
The confidence family also covers every possible ordinary reference choice.

The primary endpoint is successful recommendation yield over all seed/cohort
cases; a selected case must satisfy both utility constraints in both external
states. Abstention is zero successful yield. At zero safety margin, bounded
fixed-recipe epsilon/source selection succeeds in 0/20 hashed and 1/20 digit
cases, versus 20/20 for joint-recipe/shift search. **Source-only joint clipping
and source-only joint recipe search also succeed in 20/20 under both encodings.**
The primary yield contrast is therefore +1.00 and +.95, but demonstrates no
additional success-yield benefit from shift-aware validation over competent
source-only configuration search. Search sizes differ; this is not a claim
of HPO efficiency or new selection methodology.

The original specified paired t summary is retained, with its degenerate
hashed interval flagged unsupported. A dated supplementary conservative paired
binary construction uses Bonferroni-combined Clopper–Pearson discordance
intervals: [.6065,1.0000] and [.5239,.9994]. These intervals concern independent
seed streams on a fixed split, not independent states or universal transport.
The supplementary estimator is standard and does not fix AUC-bound coverage.

The point fixed-recipe source policy succeeds in 10/20 hashed cases (ten
abstentions), and 17/20 digit cases (one abstention, two selected failures).
Representation changes recommendation behavior. Bounded logistic-only shift
selection also succeeds in all 20 cases under each encoding. Both source-only
and shift-aware joint policies continue to succeed in 20/20 at safety margin
.01. All predefined margin curves and exact-coverage comparisons are retained;
no threshold is optimized against test failures.

For the hashed base MLP at epsilon 2, mean source clipping/noise gaps are
.02884/.00136 at norm 1, versus .00010/.01153 at norm 5. This illustrates a
bias/noise balance under the same accounted privacy target. The stronger MLP
has similar ordinary utility, and does not eliminate configuration sensitivity.
These descriptive components are not a general causal mechanism claim.

The full-family diagnostic has 600 clustered Gaussian simulations with
12 candidates, five environments and three possible references (360
endpoints). Observed simultaneous coverage is .880/.945/.980 in three
predefined imbalance/cluster scenarios; Monte Carlo SEs are .0230/.0161/.0099.
The .880 result limits any claim of uniform nominal .95 calibration. The
`bounds_supported` flag checks computational nondegeneracy, not coverage.
The study uses explicitly asymptotic estimates and does not provide certified
utility. [Detailed findings and provenance](../reports/configuration_selection/FINDINGS.md)
retain the complete protocols, launch hashes, archive checksums and accounting.

## 5. Interpretation and submission boundary

The evidence currently supports checking clipping/training configuration
before telling a privacy-noise or proxy-robustness story, and reporting
abstention alongside selected-case failures. It does not establish a new
DP protection phenomenon or universally superior recommendation policy.
Related work already establishes the broad stable-feature and clipping/noise
issues. The defensible publication direction is a narrow reproducible
benchmark/negative-result contribution, subject to a reviewer finding the
decision evaluation increment useful.

The independently frozen ACS replication, representation sensitivity and
stronger recipe/logistic comparison have now been executed (Section 4.5).
They narrow the interpretation to configuration-sensitive recommendation
availability; shift-aware search adds no successful-yield improvement on the
examined splits. Before submission, assess scholarly novelty and venue fit
against the expanded literature and address the failed simultaneous calibration
if inference reliability is a central contribution. Automatic/adaptive clipping
and published private-HPO comparators remain important for any superiority claim.
ACS is already implemented and executed, so another unrelated dataset is
optional for this narrow claim. It becomes necessary if claiming cross-domain
generality. The tiny insurance table cannot provide that generality merely by
remaining in the paper. A stronger non-neural baseline, selected only on
validation data, would help establish that failures are not an undertrained
MLP artifact. No automated script can guarantee sufficient scholarly novelty.

## 6. Privacy scope, reproducibility and limitations

The Opacus accountant covers each prepared-record DP-SGD model under the
implemented Poisson sampling and add/remove adjacency. It does not account
for the many-model research release, public evaluation/selection or the
older insurance preprocessing. Fixed public ACS encoding avoids learning
feature domains from private records, but household-safe partitions and
household-cluster utility inference do not provide household DP. The study
uses public data and reproducible non-cryptographic randomness; it is not a
production privacy deployment or an acceptable-ε policy.

Configs, per-model accounting, frozen choices, metrics, decisions, archive
checksums and calibration results are committed. Raw ACS archives and prepared
records are excluded from Git. Runtime/provenance notes identify executed
revisions, including the development runner's stage-label metadata change
during execution. Reported source hashes captured at run completion must not
be mistaken for an immutable launch snapshot.

Commands and the prospective protocol are in `docs/PUBLICATION_PLAN.md`.
Reproduction requires a fresh output directory, the listed CPU runtime and
official data access. Seeds reproduce experimental streams conditional on
software/hardware; tests check invariants rather than certify scientific
conclusions. The historical PDF and unsupported demographic significance
claims must not be used as the submission manuscript.

## References

1. Mahajan, D., Tople, S., Sharma, A. (2021). *The Connection between
   Out-of-Distribution Generalization and Privacy of ML Models*.
   https://arxiv.org/abs/2110.03369
2. Zhang, C., Pang, J., Mauw, S. (2025). *Spurious Privacy Leakage in Neural
   Networks*. TMLR. https://arxiv.org/abs/2505.20095
3. Jarin, I., Eshete, B. (2022). *DP-UTIL: Comprehensive Utility Analysis of
   Differential Privacy in Machine Learning*. CODASPY.
   https://arxiv.org/abs/2112.12998
4. Bagdasaryan, E., Shmatikov, V. (2019). *Differential Privacy Has Disparate
   Impact on Model Accuracy*. https://arxiv.org/abs/1905.12101
5. Ding, F., Hardt, M., Miller, J., Schmidt, L. (2021). *Retiring Adult:
   New Datasets for Fair Machine Learning*.
   https://github.com/socialfoundations/folktables
6. U.S. Census Bureau. 2018 ACS one-year PUMS person archives.
   https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/
7. Ying, G., Maguire, M. G., Glynn, R. J., Rosner, B. (2022; online 2021).
   *Tutorial on Biostatistics: Receiver-Operating Characteristic (ROC) Analysis
   for Correlated Eye Data*. Ophthalmic Epidemiology, 29(2), 117–127.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8586066/

8. Morsbach, F., Reubold, J., Strufe, T. (2024). *R+R: Understanding
   Hyperparameter Effects in DP-SGD*. ACSAC. https://arxiv.org/abs/2411.02051
9. Panda, A., Tang, X., Mahloujifar, S., Sehwag, V., Mittal, P. (2024).
   *A New Linear Scaling Rule for Private Adaptive Hyperparameter Optimization*.
   ICML. https://proceedings.mlr.press/v235/panda24a.html
10. Gulrajani, I., Lopez-Paz, D. (2021). *In Search of Lost Domain Generalization*.
    ICLR. https://arxiv.org/abs/2007.01434
11. Ligett, K., Neel, S., Roth, A., Waggoner, B., Wu, Z. S. (2017).
    *Accuracy First: Selecting a Differential Privacy Level for Accuracy-Constrained ERM*.
    NeurIPS. https://arxiv.org/abs/1705.10829
12. Whitehouse, J., Wu, Z. S., Ramdas, A., Rogers, R. (2022).
    *Brownian Noise Reduction: Maximizing Privacy Subject to Accuracy Constraints*.
    NeurIPS. https://arxiv.org/abs/2206.07234
13. U.S. Census Bureau. 2017 ACS one-year PUMS person archives.
    https://www2.census.gov/programs-surveys/acs/data/pums/2017/1-Year/
