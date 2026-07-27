# Methodological note on the powered two-configuration spike

**Added by the detectability-noise-ladder work. The spike's recorded results in
`two_config_spike.json`, `two_config_spike.md` and `CONCLUSION.md` are
unchanged; this note records how they should now be read.**

## Raw scores are not available

The spike computed per-example LiRA scores (`all_scores` in
`research/two_config_spike.py`) but never persisted them. The committed JSON
contains only aggregated per-subgroup metrics — `n`, `members`, `nonmembers`,
`auc` and TPR at each FPR — together with bootstrap Delta summaries and shadow
metadata. The GitHub Actions artifact (run 30212921872, artifact 8635033599)
holds that same payload, so it does not contain score-level data either.

Score-level permutation testing therefore **cannot be applied retrospectively**
to the spike. The spike was not re-run for this purpose.

## What the spike's Delta actually was

The spike's headline subgroup statistic was the **absolute max-minus-min gap**

```text
Delta = max_group_TPR@1% - min_group_TPR@1%
```

reported as differing by approximately 0.0027 in mean between the two DP-SGD
configurations.

That statistic is **positively biased under the null**. Taking a maximum minus
a minimum over noisy per-group estimates yields a positive expected value even
when no subgroup difference exists, because the max and min are selected
*after* seeing the noise. A small positive Delta is therefore not evidence of
a small positive effect, and a difference between two configurations' mean
Deltas is not evidence that they differ in subgroup leakage.

The corrected primary statistic — used in the attack-power control and in the
noise ladder — is the **signed contrast** `male TPR@1% − female TPR@1%`, which
is centred on zero under the null. The absolute gap is retained only as a
descriptive statistic.

## What this means for the spike's conclusion

The spike's conclusion — that the corrected high-cost spike is a **null** for
subgroup privacy redistribution across the two tested iso-ε recipes — is not
overturned by this note. Its stated basis is strengthened in one respect and
weakened in another:

* **Strengthened**: bootstrap intervals for Delta all included zero and the
  direction of the gap was unstable across seeds. Both remain true and both are
  consistent with no detectable subgroup effect.
* **Weakened**: the ~0.0027 mean-Delta difference between configurations was a
  **descriptive** quantity computed from a biased statistic without a
  score-level null. It cannot support a claim about subgroup leakage in either
  direction — neither that a small difference exists, nor that the
  configurations are equivalent.

The honest reading is: **the spike did not measure a subgroup effect, and its
Delta comparison was never capable of supporting a subgroup claim.** Any future
subgroup claim must come from signed contrasts assessed against a stratified
membership permutation null on retained per-example scores.

## A caution about borrowed null distributions

The attack-power control measured a non-zero permutation-null mean for the
absolute gap on its own cohorts. That figure belongs to **that** experiment —
its models, its cohort sizes, its score distributions. It is **not** the DP
spike's null distribution and must not be substituted for one. The bias
direction described above is a general property of a max-minus-min statistic;
the specific magnitude of that bias is experiment-dependent and, for the spike,
was never measured because the scores were not retained.

## Change adopted going forward

Experiments in this repository now retain what is needed to test their own
statistics: the noise ladder records per-point, per-seed permutation nulls
(observed value, null mean, null interval, raw and Holm-adjusted p-values) for
the aggregate AUC, aggregate TPR, each subgroup TPR, the signed contrast and
the absolute gap.
