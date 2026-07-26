# Two-configuration iso-epsilon subgroup-LiRA spike

Target privacy: epsilon=8.0, delta=1e-5, RDP accountant, Poisson sampling.
Attack cohort: equal counts in every `sex x membership` cell (132 per cell; 528 total).
Attack: offline LiRA with 12 DP shadow models per recipe.
Headline metric: subgroup TPR at 1% FPR and max-min gap Delta.

| Recipe | Achieved epsilon | Noise multiplier | Test ROC-AUC | Group | LiRA AUC | TPR@1% FPR |
|---|---:|---:|---:|---|---:|---:|
| small_batch_long_tight_clip | 7.9937 | 1.2054 | 0.5536 | female | 0.5250 | 0.0076 |
| small_batch_long_tight_clip | 7.9937 | 1.2054 | 0.5536 | male | 0.5562 | 0.0076 |
| **small_batch_long_tight_clip Delta** |  |  |  | **max-min** |  | **0.0000** |
| large_batch_short_loose_clip | 7.9999 | 1.0791 | 0.5143 | female | 0.4815 | 0.0076 |
| large_batch_short_loose_clip | 7.9999 | 1.0791 | 0.5143 | male | 0.5526 | 0.0455 |
| **large_batch_short_loose_clip Delta** |  |  |  | **max-min** |  | **0.0379** |

## Readout

The observed between-recipe Delta separation is 0.0379: the tight-clipping recipe had no subgroup gap at 1% FPR, while the loose-clipping recipe had a male-minus-female gap of 0.0379.

This is not yet evidence for the research claim. Both target models have weak predictive utility, subgroup LiRA AUCs are near chance, and the 12-shadow ensemble left only two OUT losses for the least-covered attack examples. At 132 non-members per subgroup, TPR@1% FPR is quantised in increments of 1/132 = 0.0076; the loose-clipping result is therefore five detected male members versus one detected female member at the permitted FPR.

The result supports exactly one next experiment: repeat these same two recipes across multiple target seeds with a larger shadow ensemble and bootstrap intervals. It does not justify expanding to the full multi-dataset configuration grid yet.
