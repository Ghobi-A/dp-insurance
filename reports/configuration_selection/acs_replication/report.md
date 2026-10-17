# Configuration-selection results

Stage: prospective_external_replication

| policy              | uncertainty_bounds   |   safety_margin |   n_cases |   coverage |   successful_yield |   failed_yield |   conditional_failure |   selected_worst_auc |
|:--------------------|:---------------------|----------------:|----------:|-----------:|-------------------:|---------------:|----------------------:|---------------------:|
| epsilon_shift       | False                |           0     |        20 |       0.3  |               0.3  |              0 |                     0 |             0.817715 |
| epsilon_shift       | False                |           0.005 |        20 |       0.15 |               0.15 |              0 |                     0 |             0.820485 |
| epsilon_shift       | False                |           0.01  |        20 |       0.05 |               0.05 |              0 |                     0 |             0.824549 |
| epsilon_shift       | False                |           0.02  |        20 |       0    |               0    |              0 |                   nan |           nan        |
| epsilon_shift       | True                 |           0     |        20 |       0    |               0    |              0 |                   nan |           nan        |
| epsilon_shift       | True                 |           0.005 |        20 |       0    |               0    |              0 |                   nan |           nan        |
| epsilon_shift       | True                 |           0.01  |        20 |       0    |               0    |              0 |                   nan |           nan        |
| epsilon_shift       | True                 |           0.02  |        20 |       0    |               0    |              0 |                   nan |           nan        |
| epsilon_source      | False                |           0     |        20 |       0.5  |               0.5  |              0 |                     0 |             0.810397 |
| epsilon_source      | False                |           0.005 |        20 |       0.2  |               0.2  |              0 |                     0 |             0.815253 |
| epsilon_source      | False                |           0.01  |        20 |       0.1  |               0.1  |              0 |                     0 |             0.813031 |
| epsilon_source      | False                |           0.02  |        20 |       0    |               0    |              0 |                   nan |           nan        |
| epsilon_source      | True                 |           0     |        20 |       0    |               0    |              0 |                   nan |           nan        |
| epsilon_source      | True                 |           0.005 |        20 |       0    |               0    |              0 |                   nan |           nan        |
| epsilon_source      | True                 |           0.01  |        20 |       0    |               0    |              0 |                   nan |           nan        |
| epsilon_source      | True                 |           0.02  |        20 |       0    |               0    |              0 |                   nan |           nan        |
| joint_clip_shift    | False                |           0     |        20 |       1    |               1    |              0 |                     0 |             0.815902 |
| joint_clip_shift    | False                |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.816631 |
| joint_clip_shift    | False                |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.821315 |
| joint_clip_shift    | False                |           0.02  |        20 |       1    |               1    |              0 |                     0 |             0.829774 |
| joint_clip_shift    | True                 |           0     |        20 |       1    |               1    |              0 |                     0 |             0.822089 |
| joint_clip_shift    | True                 |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.829025 |
| joint_clip_shift    | True                 |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.829796 |
| joint_clip_shift    | True                 |           0.02  |        20 |       0.9  |               0.9  |              0 |                     0 |             0.830267 |
| joint_clip_source   | False                |           0     |        20 |       1    |               1    |              0 |                     0 |             0.815902 |
| joint_clip_source   | False                |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.815902 |
| joint_clip_source   | False                |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.817629 |
| joint_clip_source   | False                |           0.02  |        20 |       1    |               1    |              0 |                     0 |             0.827748 |
| joint_clip_source   | True                 |           0     |        20 |       1    |               1    |              0 |                     0 |             0.817629 |
| joint_clip_source   | True                 |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.82265  |
| joint_clip_source   | True                 |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.828544 |
| joint_clip_source   | True                 |           0.02  |        20 |       0.95 |               0.95 |              0 |                     0 |             0.830276 |
| joint_recipe_proxy  | False                |           0     |        20 |       1    |               1    |              0 |                     0 |             0.815902 |
| joint_recipe_proxy  | False                |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.815902 |
| joint_recipe_proxy  | False                |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.817575 |
| joint_recipe_proxy  | False                |           0.02  |        20 |       1    |               1    |              0 |                     0 |             0.826275 |
| joint_recipe_proxy  | True                 |           0     |        20 |       1    |               1    |              0 |                     0 |             0.817575 |
| joint_recipe_proxy  | True                 |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.819861 |
| joint_recipe_proxy  | True                 |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.82911  |
| joint_recipe_proxy  | True                 |           0.02  |        20 |       1    |               1    |              0 |                     0 |             0.83122  |
| joint_recipe_shift  | False                |           0     |        20 |       1    |               1    |              0 |                     0 |             0.815902 |
| joint_recipe_shift  | False                |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.816167 |
| joint_recipe_shift  | False                |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.819841 |
| joint_recipe_shift  | False                |           0.02  |        20 |       1    |               1    |              0 |                     0 |             0.831531 |
| joint_recipe_shift  | True                 |           0     |        20 |       1    |               1    |              0 |                     0 |             0.821279 |
| joint_recipe_shift  | True                 |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.828567 |
| joint_recipe_shift  | True                 |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.831259 |
| joint_recipe_shift  | True                 |           0.02  |        20 |       1    |               1    |              0 |                     0 |             0.83122  |
| joint_recipe_source | False                |           0     |        20 |       1    |               1    |              0 |                     0 |             0.815902 |
| joint_recipe_source | False                |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.815902 |
| joint_recipe_source | False                |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.817575 |
| joint_recipe_source | False                |           0.02  |        20 |       1    |               1    |              0 |                     0 |             0.826275 |
| joint_recipe_source | True                 |           0     |        20 |       1    |               1    |              0 |                     0 |             0.817575 |
| joint_recipe_source | True                 |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.819861 |
| joint_recipe_source | True                 |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.82911  |
| joint_recipe_source | True                 |           0.02  |        20 |       1    |               1    |              0 |                     0 |             0.83122  |
| logistic_shift      | False                |           0     |        20 |       1    |               1    |              0 |                     0 |             0.818801 |
| logistic_shift      | False                |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.822722 |
| logistic_shift      | False                |           0.01  |        20 |       1    |               1    |              0 |                     0 |             0.825449 |
| logistic_shift      | False                |           0.02  |        20 |       0.7  |               0.7  |              0 |                     0 |             0.828293 |
| logistic_shift      | True                 |           0     |        20 |       1    |               1    |              0 |                     0 |             0.828301 |
| logistic_shift      | True                 |           0.005 |        20 |       1    |               1    |              0 |                     0 |             0.828301 |
| logistic_shift      | True                 |           0.01  |        20 |       0.9  |               0.9  |              0 |                     0 |             0.828158 |
| logistic_shift      | True                 |           0.02  |        20 |       0    |               0    |              0 |                   nan |           nan        |

Primary paired comparison:

```json
{
  "endpoint": "successful recommendations / all cases; all external environments must pass",
  "contrast": "joint_recipe_shift minus epsilon_source",
  "n_seed_units": 20,
  "difference": 1.0,
  "ci95_lower": 1.0,
  "ci95_upper": 1.0,
  "interval_supported": false,
  "interval_method": "paired seed-mean t approximation; degenerate/small-n intervals unsupported",
  "scope": "fixed environment split; public-data selection, not a composed DP mechanism"
}
```

Coverage/failure curves are descriptive; margins were fixed before scoring. Zero selected-case failures with low coverage do not establish superiority. Seeds reuse fixed archives and states; no arbitrary-shift guarantee.
