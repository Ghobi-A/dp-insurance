# Submission assessment — 1 October 2026

## Recommended scope and venue assessment

Treat this as a reproducible tabular **decision audit / replication study**.
The contribution is the measured consequence of configuration, representation
and uncertainty rules for recommendation availability, with explicit external
success/failure and abstention. It is not a new privacy mechanism, optimizer,
uncertainty estimator, or demonstrated superiority of shift-aware selection.

TMLR is a plausible venue to assess, not a predicted acceptance. Its current
[acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html) focus on
supported claims and reader interest; method novelty is not necessary. Its
reproducibility example requires useful lessons beyond merely rerunning a
published idea. These rules allow this scope but do not establish that the
current paper satisfies the interest criterion. A domain expert should assess
whether the decision endpoint teaches enough beyond known tuning effects.

The [author guide](https://www.jmlr.org/tmlr/author-guide.html) requires its
LaTeX style, anonymized submission/supplements and complete author OpenReview
profiles. The provided review PDF is a draft, not submitted or under review.
The public GitHub repository identifies its owner; linking it directly is not
an anonymized supplementary release. Final anonymous packaging therefore
depends on the chosen submission route and author review.

The [FAQ](https://www.jmlr.org/tmlr/faq.html) permits LLM assistance while
retaining author responsibility. The editable draft, experiments and this
assessment were prepared with AI assistance. Ghobi and any coauthors need to
read, understand, correct and take responsibility for the scientific text,
methods and claims. Decide authorship, affiliations, conflicts/funding and
required disclosure based on actual contributions. Do not list an AI as an
author or suggest expert human review has already occurred.

## Claim-to-evidence register

| Proposed statement | Evidence and supported boundary |
|---|---|
| Joint configuration selection can recover useful recommendations | Original 2017 ACS runs give 20/20 successful cases per encoding; restricted epsilon-only gives 0/20 or 1/20. The bank is wider and restricted baseline is known-sensitive, so no search-efficiency or general superiority claim. |
| Shift-aware selection improves external success | Not established: original source-only joint search also gives 20/20. The supplementary digit run gives a one-case gain, with a paired interval including zero. Retain both findings without claiming general equivalence or superiority. |
| Clipping can explain a misleading privacy-noise story | Matched descriptive controls show recipe-sensitive clipping/noise components. This reproduces established training effects; components are not an isolated causal decomposition. |
| AUTO-S is a relevant published baseline | Matched supplementary comparison uses the cited rule, unchanged stability and schedule; its sandwich rules abstain for all cases. This is not a tuned algorithm ranking, a reproduction of the published task suite, or a private HPO implementation. |
| Validation confidence bounds reliably certify recommendations | Unsupported for the original sandwich rule: original rare-label coverage .880, fresh follow-up about .869. Keep it as an asymptotic heuristic and disclose the failure. |
| Conservative concentration has a finite-sample interpretation | Only under the explicit conditional laws and independent-cluster assumptions in the supplementary derivation. ACS coverage remains an assumption-dependent sensitivity; no survey or unseen-state certificate. |
| Seed-level intervals measure uncertainty | Conditional on the fixed archives/state split and independent RNG streams. Forty representation/seed runs are not forty independent populations; model fits are not statistical replications. |
| Each DP candidate meets the specified training cap | Per-model Opacus RDP under prepared-record add/remove adjacency. Research bank, selection, ordinary controls and output release are not jointly private. |
| The method is novel | Unsupported. A targeted primary-source review finds broad overlap; benchmark value and originality require scholarly assessment. |

## Concrete author review before submission

1. Read the actual paper and inspect the three decision-result tables, not only
   the positive primary contrast. Decide whether the supported replication
   lesson is worth submitting under this scope.
2. Have a statistician/ML researcher check the cluster influence approximation,
   concentration assumptions and distinction between seed uncertainty,
   survey sampling and unseen-state uncertainty. No review has been arranged
   or completed by this automation.
3. Resolve the inaccessible primary-source checks and follow citation chains
   around the closest methods. Avoid a first-ever claim unless justified.
4. Supply actual author information, contribution/disclosure statements,
   funding/conflicts and the relationship to the MSc. Confirm that the thesis
   is permitted prior dissemination under the chosen venue's rules.
5. Reproduce the smoke checks and inspect the run-verification output. A clean
   full reproduction is recommended before submission; a fresh output directory
   is required and raw Census archives must not be packaged.
6. Review the PDF and prepare a genuinely anonymous artifact for double-blind
   review. This repository's research draft and logs remain public evidence,
   not the final anonymized submission.

No numerical percentage of publication readiness is defensible. Code/run
completion is now reviewable; the remaining author/expert contribution review
cannot be replaced by more seeds or automatic novelty confirmation.
