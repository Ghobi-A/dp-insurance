# Literature survey: utility-constrained DP configuration selection under shift

Survey date: 1 October 2026. This is a targeted novelty assessment, not a
systematic review with exhaustive database screening. The user requested
novelty confirmation; the available evidence **does not confirm originality**.
It rules out broad claims and identifies a narrower empirical question to test.

## Search and evidence boundary

Primary-source searches covered combinations of differential privacy, DP-SGD,
hyperparameter tuning, clipping, model selection, distribution shift, robust
utility, and selective prediction/coverage. Existing manuscript references
were rechecked alongside newer hyperparameter and privacy-variance work.
Search snippets are not treated as full-paper evidence. Full HTML methods
were inspected for Morsbach et al., Mahajan et al., DomainBed and the
OOD/epsilon-selection sections of Panda et al.; the NDSS
2026 paper's opening methodology/privacy argument was inspected. Other entries
below are scoped to their author abstracts/proceedings summaries. A targeted citation-chain check through Panda et al. and Jiang et al. added
Accuracy First and Brownian Noise Reduction. An exhaustive
citation-chain and full-text search remains required before a first-ever claim.

The following matrix identifies substantive overlap. These sources do not
prove that no other paper already implements the proposed benchmark.

| Primary source | Established contribution / overlap | Consequence for this study |
|---|---|---|
| Mahajan, Tople & Sharma (2021), [The Connection between Out-of-Distribution Generalization and Privacy of ML Models](https://arxiv.org/abs/2110.03369) | Stable-feature learning can improve utility at the same formal DP guarantee; OOD generalization and membership robustness need not agree. | Stable features helping private models and proxy robustness are not our novelty. |
| Morsbach, Reubold & Strufe (ACSAC 2024), [R+R: Understanding Hyperparameter Effects in DP-SGD](https://arxiv.org/abs/2411.02051) | Factorial replication across datasets, architectures and budgets; clipping/learning-rate interaction is important. | Joint clipping/learning-rate sensitivity and a hyperparameter sweep are established. |
| Hu et al. (ICML 2025), [Empirical Privacy Variance](https://proceedings.mlr.press/v267/hu25x.html) | Same formal guarantee can give different empirical memorization; utility-focused tuning can worsen empirical privacy; proposes selection heuristics. | Same epsilon/different empirical outcomes and privacy-aware tuning cannot be claimed as first observations. Our endpoint is external utility, not memorization. |
| Panda et al. (ICML 2024), [A New Linear Scaling Rule for Private Adaptive Hyperparameter Optimization](https://proceedings.mlr.press/v235/panda24a.html) | Privacy-accounted adaptive HPO; appendix explicitly evaluates distribution shifts and discusses choosing epsilon. | Private HPO plus OOD testing is already combined. A tabular version or joint search alone is insufficient novelty. Our narrower endpoint is a utility-constrained recommendation with explicit abstention/yield. |
| Papernot & Steinke (2022), [Hyperparameter Tuning with Renyi Differential Privacy](https://arxiv.org/abs/2110.03620) | Multiple training runs and tuning have privacy costs; gives search analyses. | Candidate accounting alone does not make our search/research release private. |
| Xiang et al. (NDSS 2026), [Revisiting Differentially Private Hyper-parameter Tuning](https://www.ndss-symposium.org/wp-content/uploads/2026-s447-paper.pdf) | Studies tightness of private selection bounds and improves white-box tuning analysis. | We do not contribute a private selection mechanism or improved search privacy bound. |
| Gulrajani & Lopez-Paz (ICLR 2021), [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434) | Explicit model-selection strategies, matched baselines and held-out domains are central; provides DomainBed. | Held-out selection benchmarking is not itself new. A DP-specific decision finding must add value beyond ordinary shift-aware selection. |
| Rabanser (2025 PhD thesis), [Uncertainty-Driven Reliability](https://arxiv.org/abs/2508.07556) | Selective prediction, DP-related uncertainty degradation and accuracy/coverage analysis. | Abstention or coverage reporting is not invented here. Our coverage unit is a model recommendation, rather than a record prediction. Abstract-level evidence only. |
| Zhang, Pang & Mauw (TMLR 2025), [Spurious Privacy Leakage in Neural Networks](https://arxiv.org/abs/2505.20095) | Spurious groups can differ in membership vulnerability; prediction robustness need not remove memorization. | Do not repackage spurious-feature privacy analysis as a new contribution. |
| Bagdasaryan & Shmatikov (2019), [Differential Privacy Has Disparate Impact on Model Accuracy](https://arxiv.org/abs/1905.12101) | Differential privacy can have unequal accuracy costs. | Broad unequal-utility/privacy-cost claims have precedent. |
| Ligett et al. (NeurIPS 2017), [Accuracy First](https://arxiv.org/abs/1705.10829) | Searches privacy levels to meet accuracy constraints, with noise-reduction/ex-post privacy analysis; includes logistic regression. | Choosing epsilon subject to utility targets is established and is not a new decision problem by itself. Abstract and citation-chain evidence. |
| Whitehouse et al. (NeurIPS 2022), [Brownian Noise Reduction](https://arxiv.org/abs/2206.07234) | Gaussian noise reduction and adaptive stopping meet accuracy requirements while controlling privacy loss. | Our public-data candidate search must not be presented as a new utility-first private release mechanism. Abstract and citation-chain evidence. |
| Jiang et al. (IEEE S&P 2025), [Meeting Utility Constraints in Differential Privacy](https://arxiv.org/abs/2412.10612) | A privacy-boosting noise-mechanism framework meets specified utility requirements. | Utility-constrained DP is established; our claim must concern recommendation reliability under shift, not the existence of a utility constraint. Introduction/related-work and abstract evidence. |
| Jarin & Eshete (CODASPY 2022), [DP-UTIL](https://arxiv.org/abs/2112.12998) | Comprehensive utility comparison of perturbation locations. | A privacy-utility benchmark needs a sharper decision endpoint than utility curves alone. |
| Bu et al. (NeurIPS 2023), [Automatic Clipping](https://arxiv.org/abs/2206.07136) | A clipping replacement eliminates tuning the threshold and provides convergence analysis. | We do not claim to solve clipping or introduce a new training algorithm. Adaptive/automatic clipping remains an important later comparator. |
| Ding et al. (NeurIPS 2021), [Retiring Adult](https://arxiv.org/abs/2108.04884) | ACS tasks span years/states for temporal and geographic evaluation. | ACS and geographic shift are established evaluation resources, not novel data. |

A newly indexed 2026 matched-accounting federated tabular benchmark
([publisher page](https://www.mdpi.com/2079-9292/15/16/3597)) also appeared in
search. Its page returned HTTP 429 during direct retrieval. It is a flagged
full-text follow-up, not evidence we have read its methods. Its abstract/snippet
suggests stronger tabular baselines and symmetric tuning should be checked.

## Verdict and candidate claim

**Method novelty: not established. Broad phenomenon novelty: contradicted by
prior work. Narrow empirical benchmark novelty: plausible but unresolved.**

The candidate question is:

> At a fixed per-candidate privacy cap, which validation-only selection rules
> produce useful DP-SGD configuration recommendations in unseen tabular
> environments, when both a common-reference utility gap and an absolute
> utility floor matter?

A distinctive result would identify a reproducible decision failure, show
whether it persists after fair representation/optimization comparisons, and
quantify successful recommendation yield and coverage without an abstention
artifact. The clipping/noise/optimization decomposition explains the failure;
it is accounting of matched controls, not a new causal inference method.

The implemented comparison separates expansion of the candidate grid from
choice of validation environments: fixed-recipe epsilon-only, joint clipping,
and joint recipe/architecture selection each have source and shift variants.
A proxy-stress variant and DP logistic-only policy are supporting comparisons.

The primary endpoint is **successful recommendations / all seed/cohort cases**,
requiring both external states to pass. Abstention never counts as success.
Conditional failure, selected utility and coverage curves accompany it.
Safety margins are fixed before external scoring; no threshold is selected
using test failures. This is a public-data benchmark, not an end-to-end
private tuning service. Search cost and number of candidates are explicit.

## Novelty gate after results

1. If joint tuning helps only by recovering the known clipping/learning-rate
   interaction and source-only selection performs equally well, narrow to a
   replication benchmark. Do not market joint search as a new method.
2. If natural-shift validation improves yield beyond source-only joint tuning,
   replicate on the independent year/state split and representation variant;
   require a useful effect with uncertainty and adequate coverage.
3. If all policies pass, the experiment demonstrates adequacy on that split,
   not a new selection failure. Report the null rather than changing states
   or utility margins until a desired finding appears.
4. If all policies abstain/fail, inspect baseline convergence on development
   data. Amend a future protocol transparently rather than tune held-out data.
5. Even a positive result needs a full-text/citation-chain scholarly review.
   A combination of existing ingredients is not proof of original research.

The original MSc connection remains tabular privacy-utility evaluation and
feature perturbation. Insurance is the motivating case; it cannot establish
cross-domain generality. ACS replication strengthens this narrow scope;
a third independent domain is necessary only for broader cross-domain claims.

## Full-text and citation-chain follow-up — 1 October 2026

After the initial outcomes, inspect the closest methods in their primary
full texts rather than treating the initial abstract matrix as sufficient.
This remains a targeted survey, not an exhaustive proof of absence.

| Paper and inspected locations | Comparison boundary after deeper inspection |
|---|---|
| Accuracy First: Sections 2.2–3, Algorithms 1–2 and the private ERM construction, [full text](https://arxiv.org/html/1705.10829v1) | Correlated noise reduction and InteractiveAboveThreshold account for stopping/utility queries. Our independent candidates and public AUC checks implement neither construction. A new epsilon-selection problem is not claimed. |
| Brownian Noise Reduction: Sections 1–3 and 5, [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/48aaa5ea741ae8430bd58e25917d267d-Paper-Conference.pdf) | Brownian release and ReducedAboveThreshold give adaptive privacy boundaries. Our per-model RDP reports are not an alternative private release mechanism. |
| Panda et al.: Sections 2–5, Algorithm 2, Appendix B.1/B.2, [full text](https://arxiv.org/html/2212.04486v3) | Their low-budget adaptive trials, scaling of LR/steps and composed search accounting differ from our fixed bank. OOD evaluation is already present; a fixed grid cannot be labelled a reproduction of that HPO. |
| R+R: Sections III-C/D, IV, V-C and conclusion, [full text](https://arxiv.org/html/2411.02051v1) | Their factorial replication establishes coupled clipping/LR effects. Our added endpoint is recommendation yield/abstention under explicit constraints; the underlying training interaction is established. |
| Automatic Clipping: Sections 3–4 and Appendix K, [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/8249b30d877c91611fd8c7aa6ac2b5fe-Paper-Conference.pdf) | AUTO-S has a published formula and an Opacus implementation recipe. The supplementary study uses that rule under a fixed matched tabular schedule, without claiming a new optimizer or reproducing its vision/NLP performance claims. |
| Koskela & Kulkarni (NeurIPS 2023): abstract, Sections 1–3 and tuning/privacy methodology, [proceedings PDF](https://papers.neurips.cc/paper_files/paper/2023/file/59b9582cd35f555ea8415030073e7b22-Paper-Conference.pdf) | Randomized tuning on a subsample and extrapolation have a specific RDP search analysis. A public-data tuning bank does not establish the same release guarantee. |

Panda et al.'s bibliography also identifies Hulkund et al., *Limits of
Algorithmic Stability for Distributional Generalization* (2023),
[OpenReview](https://openreview.net/forum?id=PoU_NgCStE5). Direct full-text
retrieval met a verification wall. Its result is not inferred solely from
Panda et al.'s characterization; retain it as an unresolved primary-source
check. The 2026 publisher follow-up above likewise remains unresolved.
Additional targeted searches combining ACS, automatic clipping, recommendation
abstention and utility-constrained selection did not resolve a first-ever
benchmark claim. Search non-detection is not evidence of absence.

The [supplementary protocol](CLIPPING_FOLLOWUP_PROTOCOL.md) chooses a published
clipping-rule comparison that directly fits the observed configuration issue.
Full private-HPO superiority is removed from scope, so implementing an
approximate adaptive search under the wrong privacy accounting is unnecessary.

**Updated gate:** pursue an empirical replication/decision audit, preserve the
null shift-validation finding and failed asymptotic calibration, and justify
what a reader learns beyond the established clipping interaction. External
human assessment of that contribution is still required; this survey cannot
confirm publication-level originality.
