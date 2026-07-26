# Threat model for membership-inference auditing

This document fixes the adversary model used by every membership-inference
experiment in this repository. It is written before the detectability noise
ladder is run, so that the experiment's interpretation is constrained by a
model chosen in advance rather than by whatever the results turn out to be.

## Asset

The **target training dataset** is private. It is a fixed subset of 856 records
drawn from the insurance benchmark pool, used to fit one target model for the
`high_cost` task.

## What the adversary knows

The adversary is given full knowledge of the training *recipe*, and no
knowledge of the training *set*. Specifically, they know:

* the model architecture, layer by layer;
* the optimiser family and its hyperparameters;
* the batch size;
* the number of epochs;
* the gradient-clipping norm;
* the preprocessing pipeline, including how it was fitted;
* the task definition, including the training-only high-cost threshold rule;
* the broad source distribution the training data was drawn from;
* under DP-SGD: the noise multiplier, the sampling scheme, the accountant and
  the claimed (ε, δ).

This is the standard white-box-recipe / black-box-weights setting used in the
LiRA literature. It is deliberately generous: a weaker adversary would produce
a weaker audit, and an audit is only useful as a lower bound.

## What the adversary does not know

* The exact membership set of the target's training data. This is the secret
  the attack tries to recover.
* The random seeds used for target initialisation and batch ordering.

## What the adversary can do

The adversary can draw auxiliary samples from the same benchmark pool and train
**matched shadow models** on them: same architecture, same optimiser, same
epochs, same batch size, same clipping norm, same noise multiplier, and exactly
the same training-set size as the target. They can train as many shadows as
their budget allows; these experiments use 32 per target seed.

The adversary cannot see the target's weights or gradients, and cannot
influence the target's training data.

## Membership definition

Membership is **record-level inclusion in the target's training set**. A record
is a member if that exact row was used to fit the target model, and a
non-member otherwise. Non-members are drawn from the held-out test partition.

There is no canary insertion in these experiments: the leakage under study is
*natural* memorisation of real records, not worst-case memorisation of crafted
ones. Canary-based auditing (see `src/dp/canaries.py` and `src/dp/audit.py`)
answers a different, worst-case question and is not part of this threat model.

## Sensitive attribute

The sensitive attribute is `sex`, taking values `female` and `male`. It is both
a model feature and the grouping variable for subgroup analysis. Attack cohorts
are equalised so that every `sex × membership` cell holds the same number of
records.

## Attacks

* **Offline LiRA is the primary attack.** Each example's target loss is
  standardised against a Gaussian fitted to its losses under shadow models
  trained *without* it. This is the realistic setting: the adversary does not
  know which records the target trained on, so they cannot condition on it.
* **Online LiRA is secondary.** It additionally fits an IN distribution per
  example, which requires shadow models that contain the target example. It is
  reported **only when every attacked example has at least 10 IN and at least
  10 OUT shadow observations**. No Gaussian is ever fitted from fewer than 10
  observations.

Where the two disagree, the offline result is the one the pre-registered
decision rules act on.

## Three separate objects

These are kept distinct throughout, and conclusions about one are never
transferred to another:

1. **Formal DP accounting** — the (ε, δ) an accountant certifies. An upper
   bound on worst-case privacy loss, over all datasets and all adversaries.
2. **Empirical attack performance** — what this specific adversary recovers
   from this specific model. A lower bound on leakage, and only for the attack
   actually run.
3. **Subgroup disparity** — whether leakage differs between `female` and
   `male`. This is a *comparison* of empirical attack performance, and is
   never inferred from aggregate attack success.

## The central caveat

**Failure to detect leakage is not proof that leakage is absent, and is not
verification of a DP guarantee.**

A null attack result is consistent with at least four different worlds: the
mechanism genuinely leaks little; the attack is too weak; the cohort is too
small to resolve the effect; or the operating point is too coarse. This
repository's experiments are designed to tell these apart where possible — that
is what the non-private positive controls and the permutation nulls are for —
but a non-detection is always reported as *not detected at this power*, never
as *private* or *safe*.

Symmetrically, a detected leak is a genuine finding: attacks only certify
leakage that is really present.
