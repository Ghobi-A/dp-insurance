# Future Work

## Predicting the Cost of Privacy (candidate v3 direction)

**Research question:** To what extent can pre-training dataset and task
characteristics predict the utility degradation induced by DP-SGD?

### Motivating observation
DP-SGD at ε≈0.49 retained ~98% of non-private ROC-AUC on `high_cost`.
Why was privacy this cheap on this task specifically, and can that be
known in advance rather than discovered empirically per-dataset?

### Subquestions
- **RQ1 (Prediction):** Can task descriptors (n, d, n/d, class imbalance,
  non-private AUC, separability/margin, effective dimensionality,
  gradient norm distribution, clipping rate, subgroup prevalence,
  feature redundancy) predict ΔAUC(ε) on *unseen* datasets?
- **RQ2 (Mechanism):** Which characteristics drive the penalty?
  ΔAUC ≈ f(n, separability, gradient clipping, imbalance, d)
- **RQ3 (Decision):** Can predictions recommend a privacy budget
  satisfying max(privacy) subject to ΔAUC < threshold?

### Stronger version — controlled perturbation study
Rather than relying on confounded natural datasets, systematically vary:
- Sample size (n = 500 / 1000 / 2000 / 5000 / 10000)
- Class imbalance (50/50 → 95/5)
- Separability (synthetic tasks with known Bayes difficulty)
- Dimensionality (added noise dimensions)
- Sensitive-group size

Fit: ΔAUC = β0 + β1·log(n) + β2·ε⁻¹ + β3·separability + β4·imbalance
          + β5·d + β6·(ε⁻¹×n) + ...

### Novelty assessment (as of Aug 2026, unverified — needs full lit review)
- Privacy/fairness/utility Pareto frontiers: **not novel** (Yaghini et al.,
  FairDP-SGD/FairPATE already cover this)
- Explanation stability under DP: **active existing literature**, incremental
  at best (2026 privacy-explainability-utility trilemma survey covers this)
- **Predicting the DP-SGD privacy-utility curve / ε-knee from pre-training
  task descriptors, validated on held-out datasets: plausible gap**,
  not confirmed as novel — no full lit review done yet

### Status
Not started. Blocked behind the primary corrective experiment sequence
(subgroup-conditioned empirical leakage via LiRA — switch task, ↑shadow
count, iso-ε sweep) which remains the priority. This is v3 scope, not
a v2 addition.
