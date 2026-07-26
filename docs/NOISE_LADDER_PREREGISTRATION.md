# Pre-registration: DP-SGD detectability noise ladder

**Committed before the full ladder workflow was run.** The decision rules below
were fixed in advance; the analysis code implements them mechanically, and the
report classifies each ladder point by applying them rather than by inspection.

## Research question

> At what privacy-noise level does aggregate natural membership leakage cease
> to be reproducibly detectable on the insurance `high_cost` task?

The contribution is the **empirical detectability frontier** — the bracket
between the lowest ε at which leakage is still detected and the adjacent point
where detection is lost. It is *not* another same-ε configuration comparison.

## What is already settled

These questions are closed and are not re-opened by this experiment:

* The original matched-ε DP-SGD subgroup spike produced no stable subgroup
  effect.
* Absolute max-minus-min subgroup gaps are **positively biased under the null**
  and must not be the primary inferential statistic. They are retained only as
  descriptive statistics.
* The corrected subgroup statistic is the signed contrast
  `male TPR@1% FPR − female TPR@1% FPR`.
* The matched-capacity non-private MLP did not memorise enough to support a
  useful attack.
* The deliberately memorising non-private MLP did leak: offline LiRA AUC
  ≈ 0.558–0.586, permutation-significant on all three seeds.
* The unbounded decision tree also leaked.
* Neither positive control produced a stable, reproducible sex-specific
  leakage difference.
* Aggregate membership leakage is therefore **measurable on this dataset when a
  model memorises**.

What remains unknown, and what this ladder measures: **where the attack loses
power as DP-SGD noise increases.**

## Fixed design

| Parameter | Value |
|---|---|
| Task | `high_cost` |
| Sensitive attribute | `sex` |
| δ | 1e-5 |
| Accountant | RDP |
| Sampling | Poisson |
| Seeds | 42, 43, 44 |
| Shadows per point per seed | 32 |
| Bootstrap replicates | 1000 |
| Permutation replicates | 1000 |
| Target/shadow training size | 856 (exactly matched) |
| Architecture | `Linear(d,128) → ReLU → Linear(128,128) → ReLU → Linear(128,64) → ReLU → Linear(64,1)` |
| Epochs | 400 |
| Batch size | 64 |
| Optimiser | Adam, lr = 1e-3 |
| Regularisation | none (no dropout, no weight decay, no early stopping) |
| Clipping norm | 1.0, fixed and public at every finite point |

**Only the noise multiplier changes between finite ladder points.** The
cohort indices, the shadow inclusion schedule, the shadow base seeds, the model
initialisation seeds and the batch-ordering seeds are all derived from the
target seed alone, never from the ladder point, so every point is paired.

Ladder points: **non-private**, ε = 32, 16, 8, 4, 2.

For each finite point the noise multiplier is solved from the requested ε via
the repository's accounting utilities, then training runs at that **fixed**
noise multiplier and the **achieved** ε is read back from the RDP accountant.
Requested ε is an input, not a result; the report shows requested ε, achieved ε
and noise multiplier side by side.

The architecture and optimiser are **not** retuned at each privacy level. The
object of study is how attack detectability changes with noise, not how well
each privacy level can be made to perform.

## Primary outcome

* **Aggregate offline-LiRA ROC-AUC** (primary),
* supported by **aggregate TPR@1% FPR**,
* assessed with **stratified membership-label permutation tests** — labels
  reshuffled within each sensitive group, preserving exact per-group member and
  non-member counts, with attack scores held fixed.

The subgroup outcome is **secondary**.

## Pre-registered aggregate detection rule

A ladder point is classified `AGGREGATE LEAKAGE DETECTABLE` only when **all** of:

1. The target passes the memorisation gate on all three seeds.
2. Mean offline-LiRA AUC across seeds ≥ 0.55.
3. Offline-LiRA AUC > 0.5 on every seed.
4. The stratified permutation null is rejected on at least two of three seeds.
5. p-values are **Holm-adjusted across ladder points within each seed**.
6. The adjusted significance threshold is 0.05.

`UNDETECTABLE AT CURRENT POWER` — the memorisation gate passes, but the
detection rule above is not met.

`TARGET DID NOT MEMORISE` — the memorisation gate fails. The point stays in the
report; it is never dropped.

**An undetected point is never labelled private, safe, or verified.**

### Memorisation gate

Passes when any of:

```
train AUC − test AUC   >= 0.03
test BCE − train BCE   >= 0.03
train acc − test acc   >= 0.03
```

## Pre-registered subgroup rule

The **signed contrast** `male TPR@1% − female TPR@1%` is the only inferential
subgroup statistic. A disparity is `SUPPORTED` only when all of:

1. The signed direction is the same across all non-zero seeds.
2. The signed-difference or absolute-gap permutation null is rejected on at
   least two of three seeds, after multiplicity adjustment.
3. The effect exceeds one-person TPR resolution, `1 / subgroup_member_count`.
4. The result is not driven by a single seed.

Otherwise: `UNSUPPORTED` (criteria assessable and not met) or `INCONCLUSIVE`
(no attack completed).

The absolute max-minus-min gap is retained **as a descriptive statistic only**.
Aggregate attack success is never interpreted as subgroup disparity.

## Detectability frontier

Reported as a **bracket**, not an interpolated threshold:

```
detectable at epsilon = X
not detectable at epsilon = Y
therefore the empirical transition lies between Y and X
```

Six points do not support interpolating a precise crossing. If detection is
**non-monotonic** in ε, the report flags the non-monotonicity explicitly,
declines to state a single threshold, and classifies the frontier as unstable.

## Coverage requirements

* At least 10 OUT observations per attacked example — offline LiRA otherwise
  aborts.
* At least 10 IN *and* 10 OUT per attacked example for online LiRA — otherwise
  online LiRA is reported as unavailable.
* No Gaussian is fitted from fewer than 10 observations.

## Assertions enforced in code

* Target and shadow training sizes match exactly.
* Target and shadow noise multipliers match exactly at finite points.
* Target and shadow achieved ε agree within accounting tolerance.
* Cohort indices are identical across ladder points for a given seed.
* Shadow inclusion schedules are identical across ladder points for a given
  seed.

## Deviations

Any deviation from this document must be recorded in the results report with
its reason. If exact ε calibration is not attainable, the nearest reproducible
noise multiplier is used and the difference is documented.

---

## Preregistration Amendment 1 — attack all ladder points

**Adopted before any corrected ladder run. The numerical detection thresholds
below are unchanged from the original registration; only the execution rule and
the classification vocabulary change.**

### What the original version specified

The original registration used the memorisation gate as an **execution gate**:
a ladder point whose target failed the gate was recorded as
`TARGET DID NOT MEMORISE` and the attack was never run against it. Only
gate-passing points produced LiRA metrics.

### Why that is wrong

DP noise suppresses memorisation and attack power together. Gating execution on
measured memorisation therefore removes observations **non-randomly, and
precisely where the noise is highest** — the high-ε-noise end of the ladder is
exactly where targets are least likely to clear the gate.

That has two consequences. First, the ladder would be silent about attack
behaviour in the regime the experiment exists to characterise. Second, and more
seriously, the relationship between measured generalisation and measured
leakage cannot be studied at all if every low-generalisation observation is
discarded before it is measured: the sample would be truncated on a variable
correlated with the outcome.

Those low-memorisation observations are **necessary data**, not skippable
cases.

### What changes

Every ladder point and every seed now receives the **complete matched-shadow
attack**, regardless of the memorisation gate:

* matched shadow training,
* offline LiRA,
* stratified membership permutation testing,
* stratified bootstrap intervals,
* subgroup metrics,
* online LiRA where IN/OUT coverage permits.

No result may carry the status `skipped_memorisation_gate`. Memorisation
becomes **explanatory metadata** recorded alongside every observation, not a
condition on whether the observation exists.

### What does not change

The aggregate detection rule is numerically identical to the original
registration:

* mean offline-LiRA AUC across seeds >= 0.55;
* offline-LiRA AUC > 0.5 on every seed;
* Holm-adjusted permutation null rejected on at least two of three seeds, at
  alpha = 0.05, adjusted across ladder points within each seed.

The memorisation criterion is also numerically unchanged — memorisation is
**present** when any of `AUC gap >= 0.03`, `BCE gap >= 0.03`, or
`accuracy gap >= 0.03` — but it is now read as a property of the target rather
than a precondition for measurement.

### Revised point classification

Detection and memorisation are two independent binary decisions, and each point
is classified by their combination:

```text
DETECTABLE WITH MEMORISATION
DETECTABLE DESPITE LOW MEASURED MEMORISATION
UNDETECTABLE DESPITE MEMORISATION
UNDETECTABLE WITH LOW MEMORISATION
```

No point is ever labelled private, safe or verified. An undetected point means
*not detected at this attack power and cohort size*, nothing more.

### Status of the first workflow run

Workflow run `30218898999` was launched automatically by the pull-request event
**before this amendment existed**, and was cancelled roughly two and a half
minutes into training. It produced no ladder-point results. It is retained as an
**exploratory / timing run only** and is not the pre-registered ladder. No
result in it may be cited as a finding.
