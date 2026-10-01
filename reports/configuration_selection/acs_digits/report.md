# Configuration-selection results

Stage: prespecified_representation_sensitivity

| policy              | uncertainty_bounds   |   safety_margin |   n_cases |   coverage |   successful_yield |   failed_yield |   conditional_failure |   selected_worst_auc |
|:--------------------|:---------------------|----------------:|----------:|-----------:|-------------------:|---------------:|----------------------:|---------------------:|
| epsilon_shift       | False                |           0     |        20 |       0.75 |               0.7  |           0.05 |             0.0666667 |             0.825851 |
| epsilon_shift       | False                |           0.005 |        20 |       0.25 |               0.25 |           0    |             0         |             0.826653 |
| epsilon_shift       | False                |           0.01  |        20 |       0.05 |               0.05 |           0    |             0         |             0.840799 |
| epsilon_shift       | False                |           0.02  |        20 |       0    |               0    |           0    |           nan         |           nan        |
| epsilon_shift       | True                 |           0     |        20 |       0    |               0    |           0    |           nan         |           nan        |
| epsilon_shift       | True                 |           0.005 |        20 |       0    |               0    |           0    |           nan         |           nan        |
| epsilon_shift       | True                 |           0.01  |        20 |       0    |               0    |           0    |           nan         |           nan        |
| epsilon_shift       | True                 |           0.02  |        20 |       0    |               0    |           0    |           nan         |           nan        |
| epsilon_source      | False                |           0     |        20 |       0.95 |               0.85 |           0.1  |             0.105263  |             0.82344  |
| epsilon_source      | False                |           0.005 |        20 |       0.7  |               0.65 |           0.05 |             0.0714286 |             0.826938 |
| epsilon_source      | False                |           0.01  |        20 |       0.25 |               0.25 |           0    |             0         |             0.829745 |
| epsilon_source      | False                |           0.02  |        20 |       0    |               0    |           0    |           nan         |           nan        |
| epsilon_source      | True                 |           0     |        20 |       0.05 |               0.05 |           0    |             0         |             0.834601 |
| epsilon_source      | True                 |           0.005 |        20 |       0    |               0    |           0    |           nan         |           nan        |
| epsilon_source      | True                 |           0.01  |        20 |       0    |               0    |           0    |           nan         |           nan        |
| epsilon_source      | True                 |           0.02  |        20 |       0    |               0    |           0    |           nan         |           nan        |
| joint_clip_shift    | False                |           0     |        20 |       1    |               1    |           0    |             0         |             0.829776 |
| joint_clip_shift    | False                |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.831331 |
| joint_clip_shift    | False                |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.83352  |
| joint_clip_shift    | False                |           0.02  |        20 |       1    |               1    |           0    |             0         |             0.843117 |
| joint_clip_shift    | True                 |           0     |        20 |       1    |               1    |           0    |             0         |             0.836274 |
| joint_clip_shift    | True                 |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.840909 |
| joint_clip_shift    | True                 |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.843117 |
| joint_clip_shift    | True                 |           0.02  |        20 |       0.9  |               0.9  |           0    |             0         |             0.843903 |
| joint_clip_source   | False                |           0     |        20 |       1    |               1    |           0    |             0         |             0.829669 |
| joint_clip_source   | False                |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.829669 |
| joint_clip_source   | False                |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.829669 |
| joint_clip_source   | False                |           0.02  |        20 |       1    |               1    |           0    |             0         |             0.841724 |
| joint_clip_source   | True                 |           0     |        20 |       1    |               1    |           0    |             0         |             0.830466 |
| joint_clip_source   | True                 |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.834997 |
| joint_clip_source   | True                 |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.841033 |
| joint_clip_source   | True                 |           0.02  |        20 |       1    |               1    |           0    |             0         |             0.843117 |
| joint_recipe_proxy  | False                |           0     |        20 |       1    |               1    |           0    |             0         |             0.829669 |
| joint_recipe_proxy  | False                |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.829669 |
| joint_recipe_proxy  | False                |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.829198 |
| joint_recipe_proxy  | False                |           0.02  |        20 |       1    |               1    |           0    |             0         |             0.842316 |
| joint_recipe_proxy  | True                 |           0     |        20 |       1    |               1    |           0    |             0         |             0.82863  |
| joint_recipe_proxy  | True                 |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.835478 |
| joint_recipe_proxy  | True                 |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.842814 |
| joint_recipe_proxy  | True                 |           0.02  |        20 |       1    |               1    |           0    |             0         |             0.843767 |
| joint_recipe_shift  | False                |           0     |        20 |       1    |               1    |           0    |             0         |             0.829776 |
| joint_recipe_shift  | False                |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.829856 |
| joint_recipe_shift  | False                |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.83026  |
| joint_recipe_shift  | False                |           0.02  |        20 |       1    |               1    |           0    |             0         |             0.843355 |
| joint_recipe_shift  | True                 |           0     |        20 |       1    |               1    |           0    |             0         |             0.83303  |
| joint_recipe_shift  | True                 |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.839126 |
| joint_recipe_shift  | True                 |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.843557 |
| joint_recipe_shift  | True                 |           0.02  |        20 |       1    |               1    |           0    |             0         |             0.843974 |
| joint_recipe_source | False                |           0     |        20 |       1    |               1    |           0    |             0         |             0.829669 |
| joint_recipe_source | False                |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.829669 |
| joint_recipe_source | False                |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.829669 |
| joint_recipe_source | False                |           0.02  |        20 |       1    |               1    |           0    |             0         |             0.839245 |
| joint_recipe_source | True                 |           0     |        20 |       1    |               1    |           0    |             0         |             0.829537 |
| joint_recipe_source | True                 |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.833202 |
| joint_recipe_source | True                 |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.839114 |
| joint_recipe_source | True                 |           0.02  |        20 |       1    |               1    |           0    |             0         |             0.843602 |
| logistic_shift      | False                |           0     |        20 |       1    |               0.95 |           0.05 |             0.05      |             0.831005 |
| logistic_shift      | False                |           0.005 |        20 |       1    |               0.95 |           0.05 |             0.05      |             0.831813 |
| logistic_shift      | False                |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.83711  |
| logistic_shift      | False                |           0.02  |        20 |       0.9  |               0.9  |           0    |             0         |             0.84213  |
| logistic_shift      | True                 |           0     |        20 |       1    |               1    |           0    |             0         |             0.840406 |
| logistic_shift      | True                 |           0.005 |        20 |       1    |               1    |           0    |             0         |             0.840825 |
| logistic_shift      | True                 |           0.01  |        20 |       1    |               1    |           0    |             0         |             0.84129  |
| logistic_shift      | True                 |           0.02  |        20 |       0.05 |               0.05 |           0    |             0         |             0.845676 |

Primary paired comparison:

```json
{
  "endpoint": "successful recommendations / all cases; all external environments must pass",
  "contrast": "joint_recipe_shift minus epsilon_source",
  "n_seed_units": 20,
  "difference": 0.95,
  "ci95_lower": 0.8453487972795845,
  "ci95_upper": 1.0546512027204153,
  "interval_supported": true,
  "interval_method": "paired seed-mean t approximation; degenerate/small-n intervals unsupported",
  "scope": "fixed environment split; public-data selection, not a composed DP mechanism"
}
```

Coverage/failure curves are descriptive; margins were fixed before scoring. Zero selected-case failures with low coverage do not establish superiority. Seeds reuse fixed archives and states; no arbitrary-shift guarantee.
