# Case study: measuring the cost and limits of differential privacy, honestly

**Ghobi A. · MSc project extended into a reproducible research codebase · Python, PyTorch, Opacus, scikit-learn**

## Question

When a tabular model is trained with differentially private SGD, what does it
cost in utility, and what can an attacker still learn? The second, narrower
question: can a validation rule pick a privacy budget whose utility survives
a change of population?

## What I built

* A leakage-safe benchmark on a public insurance table. Splits are made before
  targets exist, every threshold and stopping rule is tuned on validation only,
  and Opacus DP-SGD is reported at the *achieved* ε from the RDP accountant.
* An empirical privacy-auditing stack: Yeom loss-threshold and offline/online
  LiRA membership attacks with matched shadow models; permutation nulls and
  Holm correction; and a one-run ε lower-bound auditor (Steinke, Nasr and
  Jagielski 2023, including the approximate-DP correction). The auditor runs as
  a regression test that fails the build if the DP-SGD noise path is broken.
* Pre-registered experiments: a non-private attack-power control, a six-point
  ε ladder (594 trained models), and an ACS Census decision audit with 1,240
  model fits on frozen protocols, with choices persisted before external test
  scoring.
* More than 400 automated tests, artefact validators that recompute every
  saved decision table, and CI workflows that rerun the studies.

## What the evidence supports

1. **The original headline task was rigged.** Smoker prediction reached AUC
   0.993 only because `charges` acted as a near-proxy for the label. Without
   it, the best model scored 0.537 [0.489, 0.584].
2. **Configuration matters more than the stress test.** On an independent
   2017 ACS split, joint clipping/recipe search produced successful
   recommendations in 20/20 seed cases, against 0–1/20 for an ε-only rule. A
   plain source-only search did just as well as the shift-aware one, so the
   gain comes from the wider search, not from the stress test.
3. **Uncertainty bounds can fail.** A cluster-robust AUC bound reached only
   86.9% simultaneous coverage (nominal 95%) in a rare-label simulation. The
   paper therefore reports abstention and removes any certification claim.
4. **Attacks detect leakage from memorising non-private models**
   (mean offline LiRA AUC 0.57–0.60; permutation p ≤ 0.013 on every seed).

## What it does not show — and why that matters

* **A failed attack is not privacy.** The frozen ladder rule detected leakage
  at no finite ε. An independent power simulation shows that rule detects a
  true AUC of 0.54 only about 12% of the time. An exploratory pooled analysis
  of the same data suggests small residual leakage at every ε from 2 to 32. The
  honest summary is "not powered to detect", not "private".
* **The ε applies to prepared records, not raw ones.** Scaling, category
  vocabularies and the cost threshold are learned from private data outside
  the accountant, so the insurance models have no end-to-end raw-record
  guarantee.
* **No subgroup (sex) leakage difference is established in either direction.**
  The permutation test used tests association, not equal leakage.
* **No novel method.** The contribution is a reproducible negative-result and
  decision-audit benchmark. It has not been submitted or peer reviewed.

## My contribution and what I would do next

I set the research questions, directed the experiments and froze the
protocols. The code, experiments and drafts were developed with AI coding
assistance, as disclosed in the repository, and I remain responsible for the
claims. The project also corrected earlier errors in my original MSc
implementation: a DP-SGD loop that added no
noise, mis-calibrated sensitivity, test-set threshold tuning, and invalid
subgroup verdicts. An independent statistical review
([`STATISTICAL_REVIEW.md`](STATISTICAL_REVIEW.md)) reproduced the headline
numbers and fixed reporting defects. It also identified the next steps: a
powered confirmatory leakage test, a stronger reference attack, a valid
equal-leakage test, and public preprocessing for end-to-end privacy.

**Skills shown:** differential privacy accounting, membership-inference
auditing, experimental design and pre-registration, permutation and simulation
inference, multiple-testing control, reproducible ML engineering, and clear
reporting of negative results.
