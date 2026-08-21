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

---

## Preregistration Amendment 2 — persist the paired design, and pre-register the between-recipe contrast

**Adopted before the full corrected ladder was run. No detection or subgroup
threshold changes. This amendment adds record-keeping, one aggregation-time
check, and the inferential statistic for the conditional follow-up.**

### 2a. Per-example records are persisted

Every attacked example now leaves a record carrying its ladder point, target
seed, cohort position, source index, sensitive group, membership label, target
loss, offline-LiRA score, and its IN/OUT observation counts. These are written
per ladder point in the point artifact and aggregated into `per_example.csv`.

The reason is design information, not convenience. Cohorts, shadow schedules,
target seeds and example ordering are deliberately paired across ladder points
and, later, across training recipes. Summary statistics discard that pairing
irrecoverably, and a paired analysis cannot be reconstructed after the fact.

### 2b. Pairing is verified at aggregation, not merely asserted

The original registration listed "cohort indices identical across ladder points"
and "shadow inclusion schedules identical across ladder points" under
*assertions enforced in code*. They were computed as modular index sums but
never compared — and a modular sum could not have verified them in any case,
being blind to ordering and to which shadow a record belongs to.

Pairing is now verified by **SHA-256 hashes over canonical, ordered
representations**:

* **Cohort hash** — `sha256("cohort|v2|" + comma-joined attack indices in their
  exact order)`. Reordering the same cohort members changes the hash.
* **Shadow-schedule hash** — each shadow contributes
  `"{position}:{size}:{comma-joined training indices in order}"`, the blocks are
  joined by `|`, and the whole is prefixed `"schedule|v2|{number of shadows}|"`.
  Moving one record from one shadow to another changes the hash even when the
  global multiset of indices is identical, because position, per-shadow size and
  the block boundaries are all part of the encoding.

Aggregation checks, per target seed and across every ladder point, that the
cohort hash, shadow-schedule hash, attack-cohort size and training size all
agree, and **fails loudly** otherwise. A point artifact that carries no hashes
cannot be verified and is rejected rather than assumed paired.

### 2c. The between-recipe contrast (conditional follow-up)

If — and only if — the ladder identifies a finite ε at which aggregate leakage
is detectable under the rule above, the corrective iso-epsilon comparison runs
two materially different DP-SGD recipes at that operating point, independently
calibrated to the same **achieved** ε.

The inferential statistic for that comparison is **not** `D_recipe ≠ 0`. It is
the difference of signed subgroup contrasts between recipes:

```
D_recipe  = male TPR@1% FPR − female TPR@1% FPR
ΔD        = D_recipe_A − D_recipe_B
```

The null hypothesis is **no recipe-dependent redistribution of subgroup
leakage**. Because the two recipes share target seeds, cohort rows and shadow
inclusion schedules, the null is generated by **exchanging the two recipes'
scores on the same example**, independently per example, with a two-sided
p-value under the `(1+k)/(reps+1)` correction. That keeps every
`group × membership` cell, the cohort geometry and the operating-point
discreteness exactly as observed; only the recipe labelling moves. The statistic
is computed per target seed and averaged across seeds.

A redistribution finding requires **all** of:

1. aggregate attack power established at that ε by the ladder rule above;
2. the two recipes' **achieved** ε matched within a predeclared tolerance of
   **0.05** (requested ε is never substituted for achieved ε);
3. the effect not driven by a single seed — at least two seeds carry a contrast
   at or above **that seed's own** one-person TPR resolution (see below);
4. the sign of ΔD reproducible across seeds;
5. the mean effect exceeding the **worst-case per-seed** resolution,
   `abs(mean ΔD) >= max(resolution_by_seed.values())`;
6. the paired permutation null rejected at α = 0.05 after multiplicity
   adjustment;
7. the **between-recipe** contrast supported — one recipe individually showing
   `D ≠ 0` is explicitly not sufficient.

### One-person resolution is per seed

Resolution is computed **within each target seed's own cohort**:

```
resolution_by_seed[seed] = 1 / min(male_member_count[seed], female_member_count[seed])
```

and criterion 3 compares `abs(ΔD_seed) >= resolution_by_seed[seed]` for each
seed independently. Criterion 5 uses the conservative
`abs(mean ΔD) >= max(resolution_by_seed.values())`.

The pooled-across-seeds member count is **not** used for either criterion and is
retained as a descriptive figure only. Pooling three seeds roughly triples the
member count and so understates the discrete grid each seed's contrast actually
moves on; an effect that clears the pooled resolution but no single seed's own
resolution is rejected. `resolution_by_seed` is persisted in the contrast output
and reported.

Failing any criterion yields `RECIPE REDISTRIBUTION UNSUPPORTED`; no finite
per-seed contrast at all yields `RECIPE REDISTRIBUTION INCONCLUSIVE`. A
supported result is stated as an empirical finding for this dataset, task,
architecture, adversary and pair of recipes. ε remains the formal global
worst-case guarantee and is never described as subgroup-specific.

---

## Preregistration Amendment 3 — the two iso-epsilon recipes, frozen before any ladder result

**Adopted before the authoritative detectability ladder was dispatched. No
ladder result existed when these definitions were written, so no recipe search
can have followed from seeing one.**

Amendment 2 §2c specified "two materially different DP-SGD recipes" without
saying which. That leaves room for a post-result recipe search, which would
invalidate the ΔD test however carefully the test itself is run. The two recipes
are therefore fixed here, in full, in advance. They are mirrored as
`FROZEN_RECIPES` in `research/recipe_contrast.py` so the frozen definitions and
the code cannot drift apart unnoticed.

### Held identical across both recipes

| Parameter | Value |
|---|---|
| Dataset / task | insurance `high_cost` |
| Sensitive attribute | `sex` |
| Architecture | `Linear(d,128) → ReLU → Linear(128,128) → ReLU → Linear(128,64) → ReLU → Linear(64,1)` |
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Regularisation | none (no dropout, no weight decay, no early stopping) |
| Sampling | Poisson |
| δ | 1e-5 |
| Accountant | RDP |
| Target train size | 856 (shadows exactly matched) |
| Shadows per recipe per seed | 32 |
| Target seeds | 42, 43, 44 |
| Attack cohort, shadow inclusion schedule, attack implementation, threat model, subgroup statistic | identical, and verified by the hashes in Amendment 2 §2b |

### Differing materially in training dynamics

| Parameter | **Recipe A** — large-batch, few steps, tight clipping | **Recipe B** — small-batch, many steps, loose clipping |
|---|---:|---:|
| Batch size | 256 | 32 |
| Epochs | 100 | 400 |
| Optimisation steps | 400 | 10,800 |
| Sample rate | 256/856 ≈ 0.2991 | 32/856 ≈ 0.0374 |
| Clipping norm | 0.5 | 4.0 |
| Noise multiplier | solved independently at the operating ε | solved independently at the operating ε |

The two differ by a factor of 27 in optimisation steps, a factor of 8 in batch
size and sample rate, and a factor of 8 in clipping norm — materially different
training dynamics, reached through the same architecture and the same downstream
attack.

### Order of operations — fixed, and not negotiable after the fact

```text
recipes fixed before ladder result
  -> ladder selects the operating epsilon using the AGGREGATE leakage rule only
  -> both recipes calibrated independently to that epsilon
  -> one pre-registered ΔD test
```

* The operating ε is **conditional** on the ladder and is read from the
  aggregate detectability rule alone (mean offline-LiRA AUC ≥ 0.55, AUC > 0.5 on
  every seed, Holm-adjusted permutation rejection on ≥ 2 of 3 seeds). Subgroup
  ladder results are **not** inspected to choose ε, and are not inspected to
  choose or modify the recipes.
* If no finite ladder point clears the aggregate rule, the follow-up does not
  run at all.
* At the chosen finite point, each recipe's noise multiplier is solved
  independently for its own sample rate and step count, and the two **achieved**
  ε values must differ by ≤ 0.05 (Amendment 2, criterion 2). Requested ε is
  never substituted for achieved ε; if the tolerance cannot be met, that is
  reported and the comparison is not presented as iso-epsilon.
* Exactly **one** pre-registered ΔD test is run. Neither recipe is retuned after
  its results are seen.
