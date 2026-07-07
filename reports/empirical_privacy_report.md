# Empirical Privacy Report: Auditing, Attacks, and the Fairness Frontier

This report covers the empirical-privacy extension to the project. Where the
main findings report quantifies the *utility* cost of privacy, this one asks a
different question: **does the privacy actually hold, and what can an attacker
recover in practice?** All numbers are produced by executing
`notebooks/dp_privacy_insurance.ipynb` and are reproduced in CI.

## Motivation

A differential-privacy analysis produces an *upper* bound on the privacy loss
ε. It says nothing about whether the code implements the mechanism correctly —
and, as this project's own history shows (a DP-SGD training loop that stepped
the un-wrapped optimizer and added no noise), a guarantee-voiding bug can pass
type checks and unit tests. The complementary tool is an **audit**: a
high-confidence *lower* bound on ε estimated from attack success. If the
audited lower bound exceeds the claimed ε, the implementation is provably
leaking more than advertised.

## 1. Empirical auditing (one-run, Steinke et al. 2023)

We audit the Laplace mechanism by membership inference on independent canaries:
each of `r` canaries is randomly included or excluded, the mechanism is run,
and membership is guessed. Under (ε, 0)-DP the number of correct guesses is
stochastically dominated by `Binomial(r, e^ε/(e^ε+1))` (Theorem 5.2), so the
largest ε whose upper-tail p-value stays below β = 0.05 is a 95%-confidence
lower bound.

| Claimed ε | Empirical ε (95% lower bound) | Attacker accuracy | Sound (lb ≤ claim) |
| --- | --- | --- | --- |
| 0.5 | 0.403 | 0.611 | ✓ |
| 1.0 | 0.780 | 0.697 | ✓ |
| 2.0 | 1.453 | 0.820 | ✓ |
| 4.0 | 2.684 | 0.942 | ✓ |

The lower bound tracks the claim from below at every level, exactly as a sound
audit should: it recovers a large fraction of the true ε (the gap is the price
of a finite sample and a conservative confidence bound) without ever exceeding
it.

## 2. Auditing as a privacy regression test

The audit is wired into the test suite as an executable privacy contract
(`tests/test_audit.py`). The decisive test audits a mechanism that returns its
input unchanged — the "forgot to add noise" bug:

```
Broken mechanism empirical eps lower bound: 6.50
Attacker accuracy: 1.000
Violates an eps=1.0 claim? True
```

The attacker separates the canary perfectly, the audit reports ε ≥ 6.5, and the
`violates(1.0)` assertion fails the build. This is the class of bug that unit
tests on the *intended* API cannot catch, because the API surface is unchanged;
only the end-to-end statistical behaviour reveals it.

## 3. Membership inference: natural vs. canary leakage

Following Carlini et al. (S&P 2022), attacks are scored by TPR at low FPR, not
average accuracy.

**Natural attack** (members = training rows, non-members = test rows):

| Target | MIA AUC | TPR @ 1% FPR |
| --- | --- | --- |
| Decision tree (no privacy) | 0.522 | 0.00 |
| DP-SGD (ε ≈ 2.13) | 0.519 | 0.01 |

Natural membership leakage is **near chance for both models**. This is a
genuine finding, not a null result to hide: `smoker` is almost deterministic
given `charges`, so the models generalise rather than memorise, and there is
little membership signal to exploit. Leakage is a property of *memorisation*,
not of the mere presence of sensitive features.

**Canary attack** (60 mislabelled canaries injected into training):

```
MIA AUC        = 1.000
TPR @ 1% FPR   = 1.000
mean member loss     = 0.000  (memorised -> low)
mean non-member loss = 16.118 (not seen  -> high)
```

When memorisation *is* forced — by injecting training rows with randomly
flipped labels that a high-capacity model can only fit by memorising — the same
attack separates members from matched held-out controls perfectly. The
natural-vs-canary contrast is the honest scientific point and validates the
attack machinery on a real trained model.

**LiRA vs. loss threshold.** On the natural task, the offline Likelihood-Ratio
Attack (calibrating each example's loss against shadow models trained without
it) extracts more low-FPR signal than a raw threshold (AUC 0.547 vs 0.522;
TPR @ 1% FPR 0.021 vs 0.000), confirming that average-case metrics understate
worst-case risk even when they look like chance.

## 4. The privacy–utility–fairness frontier

Fairness is usually reported only for the non-private model. Measuring
demographic-parity (DP) and equalized-odds (EO) differences by sex as
feature-level Laplace ε varies exposes a three-way interaction:

| ε | ROC-AUC | DP difference | EO difference |
| --- | --- | --- | --- |
| 0.5 | 0.538 | 0.000 | 0.000 |
| 1.0 | 0.535 | 0.000 | 0.000 |
| 2.0 | 0.528 | 0.000 | 0.000 |
| 4.0 | 0.543 | 0.000 | 0.000 |
| 10.0 | 0.672 | 0.098 | 0.229 |

At tight ε the noise erases *all* signal — the classifier predicts almost
constantly, so both utility and the measured disparity collapse to zero. The
group gap re-emerges (EO difference 0.229) only as utility returns at ε = 10.
The apparent "fairness" at low ε is an artifact of a useless model, which is
why privacy, utility, and fairness must be read jointly: a single fairness
number without its accompanying utility is meaningless.

## Conclusion

The extension turns the project from one that *claims* privacy into one that
*verifies* it. The audit empirically lower-bounds ε and catches a
guarantee-voiding implementation bug that passes conventional tests; membership
inference shows that practical leakage tracks memorisation rather than ε alone;
and the fairness frontier shows privacy noise interacting with group disparity.
Together these support a reporting discipline — ε, empirical audit, attack
success at low FPR, and fairness, all together — rather than any single metric
in isolation.

## References

- Steinke, T., Nasr, M., & Jagielski, M. (2023). *Privacy Auditing with One (1)
  Training Run.* NeurIPS 2023 (Outstanding Paper).
- Jagielski, M., Ullman, J., & Oprea, A. (2020). *Auditing Differentially
  Private Machine Learning: How Private is Private SGD?* NeurIPS 2020.
- Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramèr, F. (2022).
  *Membership Inference Attacks From First Principles.* IEEE S&P 2022.
- Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). *Privacy Risk in
  Machine Learning: Analyzing the Connection to Overfitting.* IEEE CSF 2018.
- Bagdasaryan, E., Poursaeed, O., & Shmatikov, V. (2019). *Differential Privacy
  Has Disparate Impact on Model Accuracy.* NeurIPS 2019.
