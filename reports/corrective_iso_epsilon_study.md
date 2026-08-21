# Corrective iso-epsilon study

**Status: SCIENTIFIC RESULT PENDING.** The implementation is complete and
validated; the authoritative experiment has *not* been executed. Nothing in this
document is a finding. Sections 7–10 are placeholders that stay empty until the
corrected detectability ladder has run to completion on the dispatch workflow,
and the phases downstream of it are gated on that result.

| Phase | State |
|---|---|
| 1. Audit of the ladder implementation | complete — see §4 |
| 2. Full ladder reproducibly runnable | complete — command in §4 |
| 3. Detectability decision | **not run** (needs the full ladder) |
| 4. Conditional iso-epsilon follow-up | **not started** — gated on Phase 3 |
| 5. Mechanism diagnostics | **not started** — gated on Phase 4 |
| 6. This report | scaffolded; results pending |

---

## 1. Motivation

The candidate research claim under test is:

> Formal iso-ε does not necessarily imply iso-exposure across demographic
> subgroups: materially different DP-SGD training dynamics may redistribute
> empirically detectable membership leakage between subgroups while preserving
> essentially the same global (ε, δ) privacy guarantee.

It is a candidate, not a conclusion. The experiment is built so that it can
return a null, and the pre-registered decision rules are applied mechanically by
code rather than by inspection.

The claim cannot be tested at all until one prior question is answered: **is
natural membership leakage measurable at the privacy levels being used?** A
subgroup comparison run where the attack has no aggregate power measures
nothing, and any subgroup difference it reports is noise on a discrete grid.
That is the question the detectability ladder exists to answer, and it is why no
second dataset, dashboard or model family is added before this sequence
resolves.

## 2. Original null

`research/two_config_spike.py` ran a powered two-configuration iso-epsilon
subgroup LiRA experiment: `high_cost` task, sensitive attribute `sex`, target
ε ≈ 8 at δ = 1e-5, RDP accountant with Poisson sampling, two materially
different DP-SGD recipes, matched target and shadow training sizes, equalised
`sex × membership` attack cohorts, offline LiRA primary.

It found **no subgroup effect**. Results are in `reports/spike/`.

That null has two possible readings, and the experiment as run could not
separate them: DP-SGD really does equalise empirical exposure at this ε, or the
attack had no power to detect anything on this dataset regardless of DP.

## 3. Attack-power control

`research/attack_power_control.py` addressed the second reading with three
non-private targets: a matched-capacity MLP mirroring the DP recipe, a
deliberately memorising MLP, and an unbounded decision tree. Results are in
`reports/attack_power_control/`.

What it established: aggregate membership leakage **is** detectable on this
dataset when the target memorises — the memorising MLP reached offline-LiRA AUC
≈ 0.558–0.586 with the stratified permutation null rejected on all three seeds.
The original DP null therefore cannot simply be dismissed as a broken LiRA
implementation.

What it did **not** establish, and this matters for the claim: a stable
sex-specific leakage disparity. Neither positive control produced one. It also
did not establish anything about detectability *under DP noise* — every control
was non-private.

## 4. Corrected detectability ladder

The ladder (`research/detectability_noise_ladder.py`, pre-registered in
`docs/NOISE_LADDER_PREREGISTRATION.md`, adversary fixed in
`docs/THREAT_MODEL.md`) traces where the attack loses power as DP-SGD noise
increases: non-private, then ε = 32, 16, 8, 4, 2, changing **only the noise
multiplier** between finite points.

### Audit outcome

Fourteen mechanical properties were checked against the implementation. Twelve
held as written. Two did not, and were fixed before anything was run:

| # | Property | Outcome |
|---|---|---|
| 1 | Every point and seed receives the full LiRA attack; the memorisation gate is metadata only | held (Amendment 1) |
| 2 | No result can carry `skipped_memorisation_gate` | held; asserted in tests and in the workflow |
| 3 | Target and shadow training sizes identical | held; asserted per shadow at runtime |
| 4 | Target and shadows share architecture, optimiser, epochs, batch size, clipping norm, noise multiplier and sampling | held; noise multiplier and achieved ε re-checked per shadow |
| 5 | Achieved ε read from the accountant, never the requested value | held; both reported side by side |
| 6 | Cohort indices and shadow schedules identical across points per seed | **FIXED** — digests were computed but never compared |
| 7 | Offline LiRA primary | held |
| 8 | ≥ 10 OUT observations per attacked example | held; offline LiRA aborts otherwise |
| 9 | Online LiRA secondary, only with sufficient IN *and* OUT | held; reported unavailable otherwise |
| 10 | Permutation stratified within sensitive group, preserving cell counts | held |
| 11 | Holm correction across finite ladder points within each seed | held |
| 12 | Signed `male − female` TPR@1% contrast is the inferential statistic | held |
| 13 | Absolute gap descriptive only | held |
| 14 | No non-detection described as privacy, safety or verification | held; guarded by tests |

A third gap was not on the audit list but blocks Phase 4: **per-example attack
outputs were not persisted anywhere**, so the deliberately paired design (same
seeds, cohort rows and shadow schedules) could not survive into a paired
between-recipe analysis. Both fixes are recorded in Preregistration Amendment 2,
adopted before the run.

### Running the full ladder

The authoritative run is **manual dispatch only** — six points × three seeds ×
33 models at 400 epochs never starts from a push or a pull request. Pull
requests get lint, unit tests and a reduced two-point smoke run.

```bash
# GitHub Actions (authoritative):
#   Actions -> "Detectability noise ladder" -> Run workflow
#   seeds = "42 43 44", shadows = 32, bootstrap_reps = 1000, permutation_reps = 1000

# Equivalent locally, one point per invocation:
for eps in 0 32 16 8 4 2; do
  python research/detectability_noise_ladder.py point \
    --epsilon "$eps" --seeds 42 43 44 --shadows 32 \
    --bootstrap-reps 1000 --permutation-reps 1000 \
    --output-dir reports/detectability_noise_ladder/points
done

python research/detectability_noise_ladder.py aggregate \
  --input-dir reports/detectability_noise_ladder/points \
  --output-dir reports/detectability_noise_ladder \
  --expect-all
```

`--expect-all` makes aggregation fail loudly if any ladder point is missing.
Aggregation additionally refuses to proceed if the points are not paired.
Outputs: `noise_ladder.md`, `noise_ladder.json`, `noise_ladder.csv`,
`observations.csv`, `per_example.csv` and figures — requested ε, achieved ε and
noise multiplier preserved throughout.

### Result

**Not yet produced.** This environment has no PyTorch or Opacus installation and
no runner budget for hours of DP-SGD shadow training, so the ladder was not
executed here. No ladder numbers are quoted anywhere in this document, and none
were fabricated.

## 5. Selected operating privacy level

**Pending.** The operating ε is selected from the ladder's pre-registered
aggregate rule alone — mean offline-LiRA AUC ≥ 0.55 across seeds, AUC > 0.5 on
every seed, and the stratified permutation null rejected on ≥ 2 of 3 seeds at
Holm-adjusted α = 0.05 — and never from subgroup outcomes. The frontier is
reported as a bracket ("detectable at ε = X, not detectable at ε = Y"), with no
interpolated threshold; a non-monotonic ladder is reported as unstable instead.

If no finite ε clears that rule, the iso-epsilon follow-up **does not run**, and
that is itself the answer for Phase 4.

## 6. Corrected iso-epsilon experimental design

Gated on §5. When it runs, it holds fixed: dataset, `high_cost` task, sensitive
attribute `sex`, target architecture, target train size, attack cohort, shadow
inclusion schedule, number of shadows, target seeds, δ, accountant, threat
model, attack implementation and subgroup inferential statistic. The two recipes
differ materially in training dynamics (batch size / optimisation steps /
clipping regime) and are **independently calibrated to the same achieved ε**,
matched within the predeclared tolerance of 0.05. Requested ε is never
substituted for achieved ε.

Persisted per recipe, seed and example: requested ε, achieved ε, noise
multiplier, batch size, epochs and optimisation steps, clipping norm, sample
rate, utility metrics, train/test memorisation metrics, aggregate LiRA AUC and
TPR@1%, female and male LiRA AUC and TPR@1%, the signed male−female contrast,
permutation p-values, multiplicity-adjusted p-values, subgroup member counts and
one-person TPR resolution.

## 7. Aggregate leakage results

*Pending the ladder run.*

## 8. Subgroup-conditioned leakage results

*Pending the ladder run.*

## 9. Between-recipe subgroup redistribution test

The statistic is implemented and unit-tested in `research/recipe_contrast.py`;
only its application to real data is pending.

```
D_recipe = male TPR@1% FPR − female TPR@1% FPR
ΔD       = D_recipe_A − D_recipe_B
```

The null is **no recipe-dependent redistribution of subgroup leakage**. Since
the recipes share target seeds, cohort rows and shadow schedules, the null is
generated by exchanging the two recipes' scores on the same example,
independently per example — preserving every `group × membership` cell, the
cohort geometry and the operating-point discreteness — with a two-sided p-value
under the `(1+k)/(reps+1)` correction, computed per seed and averaged.

The seven pre-registered criteria for a redistribution finding are listed in
Preregistration Amendment 2, §2c. Criterion 7 is the one that distinguishes this
from the original design: the **between-recipe** contrast must itself be
supported; one recipe individually showing `D ≠ 0` is not sufficient.

Synthetic validation already performed: a null design (both recipes leaking
identically) does not produce a redistribution verdict, and an injected
redistribution (recipe A exposing male members, recipe B exposing female
members) is detected with the correct sign.

*Result on real data: pending.*

## 10. Mechanism diagnostics

Not justified, and therefore not implemented. Diagnostics — subgroup-conditioned
gradient-norm distributions, fraction of examples clipped by subgroup,
per-example loss distributions by membership × subgroup, train/test loss gaps by
subgroup, subgroup utility and calibration differences, optimisation exposures —
are added **only if** §9 produces a reproducible recipe-dependent effect. They
would be mechanism diagnostics, never additional hypothesis fishing, and never
silently upgraded into causal claims.

## 11. Limitations

* **Discreteness dominates the subgroup grid.** Subgroup TPR@1% FPR moves in
  steps of `1 / subgroup_member_count`, and at 1% FPR the operating point admits
  roughly one false positive per group. Effects smaller than one person are not
  resolvable, however many replicates are drawn.
* **One dataset, one task, one architecture, one adversary.** The ladder
  characterises a deliberately attackable neural target on the insurance
  `high_cost` task. It is not a universal DP-SGD detectability frontier.
* **Non-detection is not privacy.** An undetected point means *not detected at
  this attack power and cohort size*. No point is described as private, safe or
  verified, and no attack result verifies a formal guarantee.
* **Association, not causation.** DP noise changes generalisation and leakage
  together; the gap-versus-leakage analysis in the ladder report measures
  association only.
* **A ladder bracket is not a threshold.** Six points do not support
  interpolating a crossing point.
* **ε is not subgroup-specific.** Should redistribution be supported, it would
  be an empirical statement about *measured* exposure under two specific
  training recipes. ε remains the formal global worst-case guarantee.

## 12. Decision / next step

1. Dispatch the full detectability ladder (command in §4).
2. Apply the pre-registered aggregate rule; report the frontier as a bracket, or
   as unstable if non-monotonic.
3. If a finite ε shows demonstrably nonzero aggregate attack power, run the
   corrected iso-epsilon comparison at that ε with both recipes calibrated to
   the same achieved ε, then apply the paired between-recipe test in §9.
4. If no finite ε clears the aggregate rule, stop and report that: the
   iso-epsilon subgroup question is not answerable with this dataset, cohort
   size and adversary, and the honest output is a null.
5. Do not update the README headline claim until this sequence is complete.

**Verdict on the candidate claim: NOT YET TESTED.**
