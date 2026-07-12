# Empirical Privacy Report: Auditing, Attacks, and the Fairness Frontier

This report covers the empirical-privacy extension to the project. Where the
main findings report quantifies the *utility* cost of privacy, this one asks a
different question: **does the privacy actually hold, and what can an attacker
recover in practice?** The numbers in §1–§4 are produced by executing
`notebooks/dp_privacy_insurance.ipynb` and are reproduced in CI. The
strengthened-adversary stress test in §5 is produced by the standalone, seeded
script `notebooks/strengthened_adversary.py`; the library it drives is covered
by the pytest suite that runs in CI.

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

## 5. Stress-testing the DP-SGD ε ≈ 2.13 claim with a stronger adversary

Sections 1–4 establish that DP-SGD holds membership inference at chance and that
the audit is sound. This section does the opposite of confirming that result: it
gives the adversary more power and actively tries to break the ε ≈ 2.13 claim —
more shadow models, the stronger online LiRA, an adversary that does not know the
target's model class, and worst-case canary constructions the earlier sections
never tested. The headline is stated up front and defended below: **the claim
held, but the audit is too weak on this model to independently *certify* it — a
distinction we are careful not to blur.**

### 5.1 A stronger LiRA does not move natural leakage off chance

We strengthened the offline LiRA of §3 three ways: raised the shadow-model count
from 32 to 128, added the **online** likelihood-ratio attack (Carlini et al.
2022, §IV — it fits an "in" *and* an "out" Gaussian per example, the field's most
powerful membership test), and tested **architecture mismatch**, where the
adversary's shadow models are a different model class than the target
(`shallow tree`, `logistic regression` vs the target's unbounded tree).

| Shadow arch. | K | offline AUC | offline TPR@1% | online AUC | online TPR@1% |
| --- | --- | --- | --- | --- | --- |
| tree (matched) | 32 | 0.547 | 0.021 | 0.549 | 0.025 |
| tree (matched) | 64 | 0.546 | 0.032 | 0.546 | 0.035 |
| tree (matched) | 128 | 0.552 | 0.032 | 0.552 | 0.032 |
| shallow tree (mismatch) | 128 | 0.558 | 0.030 | 0.523 | 0.000 |
| logistic reg. (mismatch) | 128 | 0.507 | 0.040 | 0.481 | 0.000 |

Three plain observations, none of which helps the attacker:

- **More shadow models barely move the AUC** (matched: 0.547 → 0.552 from K = 32
  to 128). The signal is not undersampled — it is essentially absent, so paying
  for more shadow models buys nothing.
- **Online ≈ offline when the architecture matches.** Modelling the "in"
  distribution adds no usable low-FPR signal here.
- **Architecture mismatch degrades the attack, and it collapses *online* LiRA
  to chance or below** (AUC 0.46–0.52, TPR@1% ≈ 0). This is expected: online
  LiRA's power comes from calibrating against loss distributions that match the
  target's; when the adversary guesses the wrong model class, the "in" Gaussian
  is mis-specified and the extra modelling hurts rather than helps. The offline
  attack, which only models the "out" distribution, is more robust to mismatch
  but stays near chance (AUC ≤ 0.56).

The §3 finding survives a much stronger adversary: natural membership leakage on
this task is near chance (AUC ≤ 0.56, TPR@1% ≤ 0.04) under every strengthening we
tried. `smoker` is too separable to memorise exploitably, and no amount of
attack machinery manufactures signal that is not there.

### 5.2 Worst-case canaries and a one-run audit of DP-SGD itself

Section 1 audited only the scalar Laplace mechanism. Here the one-run auditor of
Steinke et al. (2023) is run against the **DP-SGD model itself**
(`dp.audit.one_run_model_audit`): insert canaries (each included by a fair coin),
train one private model, score each canary by its loss, guess membership, and
invert the correct-guess count into an ε lower bound. We test three canary
constructions — the §3 **label-flip** (outlier in label space) plus two the
project had not tried: a **feature-space outlier** (a point far outside the data
cloud) and a **duplicated** record repeated eight times (the worst-case
memorisation probe). Each is run against the properly-noised model
(noise_multiplier = 2.0) and a deliberately **broken** one (noise turned off).

| Canary | model | acct ε (run) | attacker acc | ε lower bound | violates 2.13? |
| --- | --- | --- | --- | --- | --- |
| label-flip | DP-SGD (noise on) | 2.00 | 0.478 | 0.000 | no |
| feature-outlier | DP-SGD (noise on) | 1.89 | 0.468 | 0.000 | no |
| duplicated ×8 | DP-SGD (noise on) | 1.41 (group ≈ 11.3) | 0.583 | 0.138 | no |
| label-flip | broken (no noise) | ∞ | 0.545 | 0.000 | no |
| feature-outlier | broken (no noise) | ∞ | 0.516 | 0.000 | no |
| duplicated ×8 | broken (no noise) | ∞ | 0.710 | 0.680 | no |

**No canary construction produced an audited lower bound above the claim.** On
the noised model, feature-outlier and label-flip canaries pin the bound at
exactly 0.0 — the noised gradients erase the per-example signal. The **duplicated
probe is the most sensitive** and is the only construction that separates the
no-noise model (ε_lb 0.68) from the noised one (ε_lb 0.14). But a duplicated
record audits *group* privacy for a group of eight, whose budget degrades to
≈ 8·ε ≈ 11 (the group-privacy property), so even 0.68 sits far below the relevant
budget and says nothing about the single-record ε ≈ 2.13 claim. The
`duplicated ×8` run's accountant ε is lower (1.41) only because inserting many
duplicated rows enlarges the dataset and dilutes the sampling rate.

### 5.3 The honest caveat: this audit is under-powered on this model

The result above is easy to over-read, so we state the limitation plainly.
**Even the broken, no-noise model audits at ε_lb ≤ 0.68 — far below 2.13.** To
certify ε > 2.13 through the binomial one-run bound, the attacker must be ~89%
accurate with 95% confidence; a 32-unit MLP trained for 15 epochs does not
memorise individual input-space canaries anywhere near that hard, and its
out-of-training loss on a canary is noisy. So the audit **fails to break** the
claim, which is a much weaker statement than **verifying** it: on a 1,070-row
dataset with this model and a loss-based score, the one-run auditor cannot
distinguish ε ≈ 2 from ε = ∞. A "not detected" is not a certificate.

The weakness is in this particular attack, not in the audit machinery. Pointed at
a target that genuinely memorises — an unbounded decision tree with the same
mislabelled canaries — the identical auditor certifies ε_lb ≈ 1.1–1.7 (attacker
accuracy 0.80–0.88), and the scalar no-noise mechanism of §2 still audits at
ε ≥ 6.5. The power exists; it is the DP-SGD MLP scored by loss that resists this
input-space attack. Closing the gap would require the gradient-space "Dirac"
canaries used by the state-of-the-art one-run DP-SGD audits, which this
loss-based attack does not implement — a concrete direction for future work, not
a claim we can currently support.

### 5.4 Verdict

The ε ≈ 2.13 headline **held** against the strengthened adversary, and the number
does not need revising: no larger shadow pool, stronger (online) attack,
architecture assumption, or worst-case canary construction produced an audited
lower bound above 2.13, and natural membership leakage stayed near chance. But
the finding is a negative result about the *adversary*, not a positive
certificate about the *mechanism*: on this small model the loss-based one-run
auditor cannot certify a bound as low as 2.13 even against a completely
non-private model, so it can only fail to refute the claim. We report the claim
as surviving, and we report — with equal prominence — that this particular audit
is not powerful enough to independently confirm it.

## Conclusion

The extension turns the project from one that *claims* privacy into one that
*verifies* it. The audit empirically lower-bounds ε and catches a
guarantee-voiding implementation bug that passes conventional tests; membership
inference shows that practical leakage tracks memorisation rather than ε alone;
and the fairness frontier shows privacy noise interacting with group disparity.
The strengthened-adversary stress test (§5) adds a note of caution to its own
tooling: a stronger attacker did not break the DP-SGD claim, but the one-run
auditor is too weak on this small model to independently certify it, so "not
broken" must not be mistaken for "verified." Together these support a reporting
discipline — ε, empirical audit, attack success at low FPR, and fairness, all
together, each with its power and limits stated — rather than any single metric
in isolation.

## References

- Steinke, T., Nasr, M., & Jagielski, M. (2023). *Privacy Auditing with One (1)
  Training Run.* NeurIPS 2023 (Outstanding Paper).
- Jagielski, M., Ullman, J., & Oprea, A. (2020). *Auditing Differentially
  Private Machine Learning: How Private is Private SGD?* NeurIPS 2020.
- Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramèr, F. (2022).
  *Membership Inference Attacks From First Principles.* IEEE S&P 2022.
- Nasr, M., Song, S., Thakurta, A., Papernot, N., & Carlini, N. (2021).
  *Adversary Instantiation: Lower Bounds for Differentially Private Machine
  Learning.* IEEE S&P 2021.
- Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). *Privacy Risk in
  Machine Learning: Analyzing the Connection to Overfitting.* IEEE CSF 2018.
- Bagdasaryan, E., Poursaeed, O., & Shmatikov, V. (2019). *Differential Privacy
  Has Disparate Impact on Model Accuracy.* NeurIPS 2019.
