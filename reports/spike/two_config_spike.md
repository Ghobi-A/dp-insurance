# Powered two-configuration iso-epsilon subgroup-LiRA spike

Task: `high_cost`.

- Target privacy: epsilon = 8.0, delta = 1e-5.
- RDP accountant with Poisson sampling.
- Hard attack gate: target test ROC-AUC >= 0.70.
- Three target seeds: 42, 43 and 44.
- 32 DP shadow models per recipe and target seed.
- Every shadow trained on exactly 856 rows, matching the target train size and calibrated noise multiplier.
- Balanced shadow schedule: every attack example had at least 11 OUT-shadow losses.
- Attack cohort equalised within every `sex x membership` cell.
- Headline metric: subgroup offline-LiRA TPR at 1% FPR and max-minus-min gap Delta.

| Recipe | Seed | Achieved epsilon | Noise | Test AUC | Group | LiRA AUC | TPR@1% | Delta | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| small_batch_long_tight_clip | 42 | 7.9937 | 1.2054 | 0.9587 | female | 0.5094 | 0.0156 | 0.0000 | [0.0000, 0.0547] |
| small_batch_long_tight_clip | 42 | 7.9937 | 1.2054 | 0.9587 | male | 0.5301 | 0.0156 |  |  |
| small_batch_long_tight_clip | 43 | 7.9937 | 1.2054 | 0.9147 | female | 0.5949 | 0.0000 | 0.0310 | [0.0000, 0.1010] |
| small_batch_long_tight_clip | 43 | 7.9937 | 1.2054 | 0.9147 | male | 0.5612 | 0.0310 |  |  |
| small_batch_long_tight_clip | 44 | 7.9937 | 1.2054 | 0.9504 | female | 0.5746 | 0.0076 | 0.0153 | [0.0000, 0.0840] |
| small_batch_long_tight_clip | 44 | 7.9937 | 1.2054 | 0.9504 | male | 0.5177 | 0.0229 |  |  |
| large_batch_short_loose_clip | 42 | 7.9999 | 1.0791 | 0.9255 | female | 0.5173 | 0.0000 | 0.0234 | [0.0000, 0.0703] |
| large_batch_short_loose_clip | 42 | 7.9999 | 1.0791 | 0.9255 | male | 0.5174 | 0.0234 |  |  |
| large_batch_short_loose_clip | 43 | 7.9999 | 1.0791 | 0.8824 | female | 0.5419 | 0.0310 | 0.0233 | [0.0000, 0.1163] |
| large_batch_short_loose_clip | 43 | 7.9999 | 1.0791 | 0.8824 | male | 0.5312 | 0.0078 |  |  |
| large_batch_short_loose_clip | 44 | 7.9999 | 1.0791 | 0.8872 | female | 0.5370 | 0.0000 | 0.0076 | [0.0000, 0.0458] |
| large_batch_short_loose_clip | 44 | 7.9999 | 1.0791 | 0.8872 | male | 0.5313 | 0.0076 |  |  |

## Across-seed summary

| Recipe | Mean target AUC | Min target AUC | Mean Delta | Delta range |
|---|---:|---:|---:|---|
| small_batch_long_tight_clip | 0.9413 | 0.9147 | 0.0154 | [0.0000, 0.0310] |
| large_batch_short_loose_clip | 0.8984 | 0.8824 | 0.0181 | [0.0076, 0.0234] |

The between-recipe difference in mean Delta is only 0.0027. Seed-wise, the loose-minus-tight Delta differences are +0.0234, -0.0078 and -0.0076, so the direction reverses across seeds.

## Honest readout

This is a null result for the proposed subgroup-redistribution claim on this dataset, task, protected attribute and pair of iso-epsilon recipes.

The target models learned the task well, so the null cannot be blamed on target utility. Shadow and target train sizes, sampling assumptions, privacy target and noise calibration were matched, and every example had 11 OUT observations. Nevertheless:

- every bootstrap interval for Delta includes zero;
- subgroup LiRA AUC is mostly close to chance;
- the more exposed group is not stable across seeds; and
- the two recipes have almost identical mean subgroup gaps despite materially different training recipes.

This result does not justify launching the full multi-dataset hyperparameter grid. The next defensible experiment is an attack-power control: run the same high-cost setup with a non-private/no-noise target and a deliberately higher-capacity model. If LiRA still remains near chance, natural membership leakage is not measurable on this dataset and the research should move to a larger dataset such as MEPS or ACS rather than adding more DP configurations.
