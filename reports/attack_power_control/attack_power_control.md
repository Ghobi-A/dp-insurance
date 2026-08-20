# Attack-power control experiment

## 1. Experimental design

Task: `high_cost`. Sensitive attribute: `sex`.
This is a control for the powered two-configuration DP spike, which returned a null subgroup result. Every model here is **non-private**: no Opacus, no gradient clipping, no noise. If the pipeline cannot detect membership in an unprotected, deliberately memorising model, the null DP result says nothing about DP.
Seeds: 42, 43, 44.
Shadow models per control and seed: 32 (balanced fixed-size schedule, training size pinned to the target's).
Bootstrap replicates: 1000, resampled within each `sex x membership` cell.
Permutation replicates: 1000, with membership labels reshuffled within each sensitive group. Bootstrap intervals are kept, but they are not used as the hypothesis test against chance; the permutation null is.
Memorisation gate: the attack runs only when train-test ROC-AUC, test-train loss or train-test accuracy differs by at least 0.03. The old ROC-AUC >= 0.70 utility gate is not used on its own -- a model can generalise well and still be an unattackable membership target.

## 2. Training sizes and cohort counts

| Control | Seed | Train | Val | Test | Attack cohort | Cells (sex\|membership) |
|---|---:|---:|---:|---:|---:|---|
| matched_capacity_mlp | 42 | 856 | 214 | 268 | 512 | female|member=0=128, female|member=1=128, male|member=0=128, male|member=1=128 |
| matched_capacity_mlp | 43 | 856 | 214 | 268 | 516 | female|member=0=129, female|member=1=129, male|member=0=129, male|member=1=129 |
| matched_capacity_mlp | 44 | 856 | 214 | 268 | 524 | female|member=0=131, female|member=1=131, male|member=0=131, male|member=1=131 |
| memorising_mlp | 42 | 856 | 214 | 268 | 512 | female|member=0=128, female|member=1=128, male|member=0=128, male|member=1=128 |
| memorising_mlp | 43 | 856 | 214 | 268 | 516 | female|member=0=129, female|member=1=129, male|member=0=129, male|member=1=129 |
| memorising_mlp | 44 | 856 | 214 | 268 | 524 | female|member=0=131, female|member=1=131, male|member=0=131, male|member=1=131 |
| unbounded_decision_tree | 42 | 856 | 214 | 268 | 512 | female|member=0=128, female|member=1=128, male|member=0=128, male|member=1=128 |
| unbounded_decision_tree | 43 | 856 | 214 | 268 | 516 | female|member=0=129, female|member=1=129, male|member=0=129, male|member=1=129 |
| unbounded_decision_tree | 44 | 856 | 214 | 268 | 524 | female|member=0=131, female|member=1=131, male|member=0=131, male|member=1=131 |

Target and shadow training sizes are exactly matched; every cohort cell holds the same number of rows by construction.

## 3. Target and shadow architectures

| Control | Kind | Architecture | Training | Positive control |
|---|---|---|---|---|
| matched_capacity_mlp | mlp | `Linear(input_dim, 32 -> 1) with ReLU between layers` | Adam, lr=0.01, batch=64, epochs=30 | no |
| memorising_mlp | mlp | `Linear(input_dim, 128 -> 128 -> 64 -> 1) with ReLU between layers` | Adam, lr=0.001, batch=64, epochs=400 | yes |
| unbounded_decision_tree | tree | `DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2)` | fitted to purity | yes |

Shadow models use the identical architecture and recipe as their target.

## 4. Memorisation metrics

| Control | Seed | Train AUC | Test AUC | AUC gap | Train BCE | Test BCE | Loss gap | Train acc | Test acc | Acc gap | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| matched_capacity_mlp | 42 | 0.9766 | 0.9614 | 0.0152 | 0.1564 | 0.1690 | 0.0126 | 0.9486 | 0.9515 | -0.0029 | FAIL (attack skipped) |
| matched_capacity_mlp | 43 | 0.9748 | 0.9209 | 0.0539 | 0.1549 | 0.3114 | 0.1565 | 0.9533 | 0.9067 | 0.0466 | PASS |
| matched_capacity_mlp | 44 | 0.9710 | 0.9301 | 0.0409 | 0.1752 | 0.2449 | 0.0696 | 0.9451 | 0.9328 | 0.0123 | PASS |
| memorising_mlp | 42 | 0.9999 | 0.9407 | 0.0592 | 0.0153 | 0.7611 | 0.7458 | 0.9965 | 0.9030 | 0.0935 | PASS |
| memorising_mlp | 43 | 0.9999 | 0.9232 | 0.0767 | 0.0105 | 1.0720 | 1.0615 | 0.9977 | 0.8769 | 0.1208 | PASS |
| memorising_mlp | 44 | 0.9999 | 0.9387 | 0.0612 | 0.0267 | 0.7887 | 0.7620 | 0.9871 | 0.9067 | 0.0804 | PASS |
| unbounded_decision_tree | 42 | 1.0000 | 0.8801 | 0.1199 | 0.0016 | 1.9245 | 1.9229 | 0.9988 | 0.8806 | 0.1182 | PASS |
| unbounded_decision_tree | 43 | 1.0000 | 0.8752 | 0.1247 | 0.0032 | 2.0474 | 2.0442 | 0.9977 | 0.8694 | 0.1283 | PASS |
| unbounded_decision_tree | 44 | 1.0000 | 0.8946 | 0.1053 | 0.0032 | 1.6840 | 1.6807 | 0.9977 | 0.8955 | 0.1021 | PASS |

## 5. Aggregate LiRA results

| Control | Seed | Attack | LiRA AUC | TPR@0.1% | TPR@1% | TPR@10% | Detected members @1% | False positives @1% | OUT obs | IN obs |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| matched_capacity_mlp | 42 | - | **SKIPPED: skipped_memorisation_gate** | - | - | - | - | - | - | - |
| matched_capacity_mlp | 43 | lira-offline | 0.5283 | 0.0000 | 0.0000 | 0.0930 | 0/258 | 0/258 | 11 | 20 |
| matched_capacity_mlp | 43 | lira-online | 0.5132 | 0.0039 | 0.0039 | 0.1008 | 1/258 | 0/258 | 11 | 20 |
| matched_capacity_mlp | 44 | lira-offline | 0.5083 | 0.0115 | 0.0153 | 0.1031 | 4/262 | 1/262 | 11 | 20 |
| matched_capacity_mlp | 44 | lira-online | 0.5045 | 0.0038 | 0.0191 | 0.0916 | 5/262 | 1/262 | 11 | 20 |
| memorising_mlp | 42 | lira-offline | 0.5859 | 0.0469 | 0.0547 | 0.2070 | 14/256 | 2/256 | 11 | 20 |
| memorising_mlp | 42 | lira-online | 0.6420 | 0.0078 | 0.0078 | 0.2031 | 2/256 | 0/256 | 11 | 20 |
| memorising_mlp | 43 | lira-offline | 0.5794 | 0.0310 | 0.0581 | 0.1512 | 15/258 | 2/258 | 11 | 20 |
| memorising_mlp | 43 | lira-online | 0.6255 | 0.0116 | 0.0116 | 0.1318 | 3/258 | 0/258 | 11 | 20 |
| memorising_mlp | 44 | lira-offline | 0.5582 | 0.0344 | 0.0725 | 0.1450 | 19/262 | 2/262 | 11 | 20 |
| memorising_mlp | 44 | lira-online | 0.5602 | 0.0038 | 0.0076 | 0.1183 | 2/262 | 2/262 | 11 | 20 |
| unbounded_decision_tree | 42 | lira-offline | 0.5985 | 0.0195 | 0.0391 | 0.1953 | 10/256 | 2/256 | 11 | 20 |
| unbounded_decision_tree | 42 | lira-online | 0.5905 | 0.0195 | 0.0391 | 0.1953 | 10/256 | 2/256 | 11 | 20 |
| unbounded_decision_tree | 43 | lira-offline | 0.6170 | 0.0194 | 0.0775 | 0.2016 | 20/258 | 2/258 | 11 | 20 |
| unbounded_decision_tree | 43 | lira-online | 0.6204 | 0.0194 | 0.0736 | 0.1667 | 19/258 | 2/258 | 11 | 20 |
| unbounded_decision_tree | 44 | lira-offline | 0.5822 | 0.0382 | 0.0420 | 0.1489 | 11/262 | 1/262 | 11 | 20 |
| unbounded_decision_tree | 44 | lira-online | 0.5737 | 0.0382 | 0.0420 | 0.1489 | 11/262 | 1/262 | 11 | 20 |

## 6. Subgroup LiRA results

| Control | Seed | Attack | Female AUC | Male AUC | Female TPR@1% | Male TPR@1% | Signed diff (male-female) | Absolute gap |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| matched_capacity_mlp | 43 | lira-offline | 0.5562 | 0.5035 | 0.0078 | 0.0000 | -0.0078 | 0.0078 |
| matched_capacity_mlp | 43 | lira-online | 0.5475 | 0.4801 | 0.0078 | 0.0078 | 0.0000 | 0.0000 |
| matched_capacity_mlp | 44 | lira-offline | 0.4973 | 0.5211 | 0.0305 | 0.0305 | 0.0000 | 0.0000 |
| matched_capacity_mlp | 44 | lira-online | 0.5305 | 0.4798 | 0.0305 | 0.0229 | -0.0076 | 0.0076 |
| memorising_mlp | 42 | lira-offline | 0.5319 | 0.6361 | 0.0234 | 0.0859 | 0.0625 | 0.0625 |
| memorising_mlp | 42 | lira-online | 0.6123 | 0.6701 | 0.0078 | 0.0156 | 0.0078 | 0.0078 |
| memorising_mlp | 43 | lira-offline | 0.6024 | 0.5554 | 0.0620 | 0.0620 | 0.0000 | 0.0000 |
| memorising_mlp | 43 | lira-online | 0.5975 | 0.6574 | 0.0233 | 0.0078 | -0.0155 | 0.0155 |
| memorising_mlp | 44 | lira-offline | 0.5524 | 0.5643 | 0.0763 | 0.0687 | -0.0076 | 0.0076 |
| memorising_mlp | 44 | lira-online | 0.5900 | 0.5328 | 0.0076 | 0.0153 | 0.0076 | 0.0076 |
| unbounded_decision_tree | 42 | lira-offline | 0.5871 | 0.6133 | 0.0078 | 0.1172 | 0.1094 | 0.1094 |
| unbounded_decision_tree | 42 | lira-online | 0.5759 | 0.6078 | 0.0078 | 0.1094 | 0.1016 | 0.1016 |
| unbounded_decision_tree | 43 | lira-offline | 0.6169 | 0.6175 | 0.0310 | 0.0698 | 0.0388 | 0.0388 |
| unbounded_decision_tree | 43 | lira-online | 0.6223 | 0.6188 | 0.0310 | 0.0698 | 0.0388 | 0.0388 |
| unbounded_decision_tree | 44 | lira-offline | 0.5709 | 0.5933 | 0.0458 | 0.0458 | 0.0000 | 0.0000 |
| unbounded_decision_tree | 44 | lira-online | 0.5593 | 0.5878 | 0.0458 | 0.0458 | 0.0000 | 0.0000 |

## 7. Bootstrap 95% confidence intervals

Stratified within each `sex x membership` cell.

| Control | Seed | Attack | Aggregate TPR@1% | Female TPR@1% | Male TPR@1% | Signed diff | Absolute gap |
|---|---:|---|---|---|---|---|---|
| matched_capacity_mlp | 43 | lira-offline | [0.0000, 0.0271] | [0.0000, 0.0465] | [0.0000, 0.0465] | [-0.0388, 0.0388] | [0.0000, 0.0465] |
| matched_capacity_mlp | 43 | lira-online | [0.0000, 0.0310] | [0.0000, 0.0465] | [0.0000, 0.0543] | [-0.0388, 0.0465] | [0.0000, 0.0543] |
| matched_capacity_mlp | 44 | lira-offline | [0.0038, 0.0344] | [0.0000, 0.0613] | [0.0000, 0.0763] | [-0.0534, 0.0534] | [0.0000, 0.0611] |
| matched_capacity_mlp | 44 | lira-online | [0.0000, 0.0458] | [0.0000, 0.0687] | [0.0000, 0.0534] | [-0.0534, 0.0382] | [0.0000, 0.0534] |
| memorising_mlp | 42 | lira-offline | [0.0273, 0.0820] | [0.0000, 0.0547] | [0.0391, 0.2188] | [0.0078, 0.2031] | [0.0078, 0.2031] |
| memorising_mlp | 42 | lira-online | [0.0000, 0.0273] | [0.0000, 0.0469] | [0.0000, 0.0703] | [-0.0236, 0.0625] | [0.0000, 0.0703] |
| memorising_mlp | 43 | lira-offline | [0.0233, 0.0969] | [0.0233, 0.1318] | [0.0233, 0.1085] | [-0.0777, 0.0698] | [0.0000, 0.0853] |
| memorising_mlp | 43 | lira-online | [0.0039, 0.0349] | [0.0078, 0.0620] | [0.0000, 0.0930] | [-0.0543, 0.0698] | [0.0000, 0.0775] |
| memorising_mlp | 44 | lira-offline | [0.0191, 0.1069] | [0.0076, 0.1298] | [0.0229, 0.1069] | [-0.0840, 0.0763] | [0.0000, 0.0916] |
| memorising_mlp | 44 | lira-online | [0.0000, 0.0458] | [0.0000, 0.0994] | [0.0000, 0.0611] | [-0.0840, 0.0534] | [0.0000, 0.0840] |
| unbounded_decision_tree | 42 | lira-offline | [0.0117, 0.1016] | [0.0000, 0.1172] | [0.0703, 0.1875] | [-0.0078, 0.1797] | [0.0234, 0.1797] |
| unbounded_decision_tree | 42 | lira-online | [0.0117, 0.0938] | [0.0000, 0.0938] | [0.0625, 0.1875] | [0.0234, 0.1721] | [0.0312, 0.1721] |
| unbounded_decision_tree | 43 | lira-offline | [0.0155, 0.1279] | [0.0000, 0.1783] | [0.0310, 0.1318] | [-0.1085, 0.1085] | [0.0000, 0.1240] |
| unbounded_decision_tree | 43 | lira-online | [0.0155, 0.1240] | [0.0000, 0.1783] | [0.0310, 0.1240] | [-0.1085, 0.1008] | [0.0000, 0.1163] |
| unbounded_decision_tree | 44 | lira-offline | [0.0191, 0.0725] | [0.0076, 0.1145] | [0.0153, 0.0916] | [-0.0687, 0.0611] | [0.0000, 0.0689] |
| unbounded_decision_tree | 44 | lira-online | [0.0229, 0.0802] | [0.0153, 0.1069] | [0.0153, 0.0916] | [-0.0687, 0.0611] | [0.0000, 0.0763] |

## 8. Stratified permutation tests against the membership null

Membership labels are reshuffled within each sensitive group (1000 replicates), preserving each group's exact member and non-member counts while the attack scores stay fixed. Bootstrap intervals above describe the precision of an estimate; these p-values ask whether an estimate that large arises from finite-sample ROC variation alone. AUC, TPR and the absolute gap use one-sided tests; the signed difference is two-sided on absolute value. All p-values use the (1+k)/(reps+1) correction, so the smallest attainable value is 0.0010.

| Control | Seed | Attack | AUC (p) | Null AUC 95% | TPR@1% (p) | Null TPR 95% | Signed diff (p, two-sided) | Null signed 95% | Absolute gap (p) | Null gap 95% |
|---|---:|---|---|---|---|---|---|---|---|---|
| matched_capacity_mlp | 43 | lira-offline | 0.5283 (p=0.1379) | [0.4503, 0.5475] | 0.0000 (p=1.0000) | [0.0000, 0.0388] | -0.0078 (p=0.8212) | [-0.0388, 0.0465] | 0.0078 (p=0.8212) | [0.0000, 0.0543] |
| matched_capacity_mlp | 43 | lira-online | 0.5132 (p=0.2967) | [0.4528, 0.5517] | 0.0039 (p=0.8721) | [0.0000, 0.0310] | 0.0000 (p=1.0000) | [-0.0465, 0.0388] | 0.0000 (p=1.0000) | [0.0000, 0.0467] |
| matched_capacity_mlp | 44 | lira-offline | 0.5083 (p=0.3636) | [0.4506, 0.5526] | 0.0153 (p=0.3417) | [0.0000, 0.0344] | 0.0000 (p=1.0000) | [-0.0382, 0.0458] | 0.0000 (p=1.0000) | [0.0000, 0.0534] |
| matched_capacity_mlp | 44 | lira-online | 0.5045 (p=0.4106) | [0.4507, 0.5501] | 0.0191 (p=0.2088) | [0.0000, 0.0344] | -0.0076 (p=0.8032) | [-0.0458, 0.0458] | 0.0076 (p=0.8032) | [0.0000, 0.0534] |
| memorising_mlp | 42 | lira-offline | 0.5859 (p=0.0010) | [0.4515, 0.5537] | 0.0547 (p=0.0040) | [0.0000, 0.0352] | 0.0625 (p=0.0160) | [-0.0391, 0.0469] | 0.0625 (p=0.0160) | [0.0000, 0.0469] |
| memorising_mlp | 42 | lira-online | 0.6420 (p=0.0010) | [0.4453, 0.5506] | 0.0078 (p=0.7113) | [0.0000, 0.0352] | 0.0078 (p=0.8112) | [-0.0391, 0.0469] | 0.0078 (p=0.8112) | [0.0000, 0.0547] |
| memorising_mlp | 43 | lira-offline | 0.5794 (p=0.0020) | [0.4543, 0.5490] | 0.0581 (p=0.0020) | [0.0000, 0.0349] | 0.0000 (p=1.0000) | [-0.0388, 0.0390] | 0.0000 (p=1.0000) | [0.0000, 0.0467] |
| memorising_mlp | 43 | lira-online | 0.6255 (p=0.0010) | [0.4507, 0.5500] | 0.0116 (p=0.4945) | [0.0000, 0.0349] | -0.0155 (p=0.5185) | [-0.0388, 0.0388] | 0.0155 (p=0.5185) | [0.0000, 0.0465] |
| memorising_mlp | 44 | lira-offline | 0.5582 (p=0.0130) | [0.4533, 0.5509] | 0.0725 (p=0.0010) | [0.0000, 0.0344] | -0.0076 (p=0.8272) | [-0.0382, 0.0458] | 0.0076 (p=0.8272) | [0.0000, 0.0460] |
| memorising_mlp | 44 | lira-online | 0.5602 (p=0.0080) | [0.4509, 0.5489] | 0.0076 (p=0.6633) | [0.0000, 0.0344] | 0.0076 (p=0.8022) | [-0.0458, 0.0458] | 0.0076 (p=0.8022) | [0.0000, 0.0611] |
| unbounded_decision_tree | 42 | lira-offline | 0.5985 (p=0.0010) | [0.4616, 0.5383] | 0.0391 (p=0.0220) | [0.0000, 0.0352] | 0.1094 (p=0.0010) | [-0.0391, 0.0469] | 0.1094 (p=0.0010) | [0.0000, 0.0469] |
| unbounded_decision_tree | 42 | lira-online | 0.5905 (p=0.0010) | [0.4618, 0.5401] | 0.0391 (p=0.0190) | [0.0000, 0.0352] | 0.1016 (p=0.0010) | [-0.0391, 0.0391] | 0.1016 (p=0.0010) | [0.0000, 0.0469] |
| unbounded_decision_tree | 43 | lira-offline | 0.6170 (p=0.0010) | [0.4582, 0.5424] | 0.0775 (p=0.0010) | [0.0000, 0.0349] | 0.0388 (p=0.0729) | [-0.0388, 0.0388] | 0.0388 (p=0.0729) | [0.0000, 0.0543] |
| unbounded_decision_tree | 43 | lira-online | 0.6204 (p=0.0010) | [0.4574, 0.5403] | 0.0736 (p=0.0010) | [0.0000, 0.0349] | 0.0388 (p=0.0759) | [-0.0388, 0.0465] | 0.0388 (p=0.0759) | [0.0000, 0.0543] |
| unbounded_decision_tree | 44 | lira-offline | 0.5822 (p=0.0010) | [0.4619, 0.5401] | 0.0420 (p=0.0070) | [0.0000, 0.0344] | 0.0000 (p=1.0000) | [-0.0458, 0.0458] | 0.0000 (p=1.0000) | [0.0000, 0.0534] |
| unbounded_decision_tree | 44 | lira-online | 0.5737 (p=0.0010) | [0.4595, 0.5412] | 0.0420 (p=0.0100) | [0.0000, 0.0305] | 0.0000 (p=1.0000) | [-0.0458, 0.0458] | 0.0000 (p=1.0000) | [0.0000, 0.0534] |

Per-subgroup TPR@1% permutation p-values:

| Control | Seed | Attack | Female TPR@1% (p) | Male TPR@1% (p) |
|---|---:|---|---|---|
| matched_capacity_mlp | 43 | lira-offline | 0.0078 (p=0.7313) | 0.0000 (p=1.0000) |
| matched_capacity_mlp | 43 | lira-online | 0.0078 (p=0.7572) | 0.0078 (p=0.7493) |
| matched_capacity_mlp | 44 | lira-offline | 0.0305 (p=0.1908) | 0.0305 (p=0.2018) |
| matched_capacity_mlp | 44 | lira-online | 0.0305 (p=0.1738) | 0.0229 (p=0.3227) |
| memorising_mlp | 42 | lira-offline | 0.0234 (p=0.2987) | 0.0859 (p=0.0050) |
| memorising_mlp | 42 | lira-online | 0.0078 (p=0.7642) | 0.0156 (p=0.5255) |
| memorising_mlp | 43 | lira-offline | 0.0620 (p=0.0150) | 0.0620 (p=0.0240) |
| memorising_mlp | 43 | lira-online | 0.0233 (p=0.3247) | 0.0078 (p=0.7413) |
| memorising_mlp | 44 | lira-offline | 0.0763 (p=0.0040) | 0.0687 (p=0.0110) |
| memorising_mlp | 44 | lira-online | 0.0076 (p=0.7522) | 0.0153 (p=0.4805) |
| unbounded_decision_tree | 42 | lira-offline | 0.0078 (p=0.7672) | 0.1172 (p=0.0010) |
| unbounded_decision_tree | 42 | lira-online | 0.0078 (p=0.7652) | 0.1094 (p=0.0010) |
| unbounded_decision_tree | 43 | lira-offline | 0.0310 (p=0.2178) | 0.0698 (p=0.0060) |
| unbounded_decision_tree | 43 | lira-online | 0.0310 (p=0.1808) | 0.0698 (p=0.0090) |
| unbounded_decision_tree | 44 | lira-offline | 0.0458 (p=0.0609) | 0.0458 (p=0.0749) |
| unbounded_decision_tree | 44 | lira-online | 0.0458 (p=0.0609) | 0.0458 (p=0.0400) |

## 9. Decision table

| Control | Positive control | Seeds attacked | Mean memorisation gap (AUC) | Mean offline LiRA AUC | Mean permutation p (AUC) | Verdict | Basis |
|---|---|---:|---:|---:|---:|---|---|
| matched_capacity_mlp | no | 2 | 0.0367 | 0.5183 | 0.2507 | **ATTACK INCONCLUSIVE** | Mean aggregate LiRA AUC 0.5183 with the permutation null rejected on only 0/2 seeds. |
| memorising_mlp | yes | 3 | 0.0657 | 0.5745 | 0.0053 | **ATTACK DETECTS LEAKAGE** | Mean aggregate offline-LiRA AUC 0.5745, above chance on every seed; the stratified membership permutation null is rejected at p<0.05 on 3/3 seeds (3/3 bootstrap TPR@1% intervals also clear the 1% random baseline). |
| unbounded_decision_tree | yes | 3 | 0.1167 | 0.5992 | 0.0010 | **ATTACK DETECTS LEAKAGE** | Mean aggregate offline-LiRA AUC 0.5992, above chance on every seed; the stratified membership permutation null is rejected at p<0.05 on 3/3 seeds (3/3 bootstrap TPR@1% intervals also clear the 1% random baseline). |

Subgroup disparity is judged separately; an aggregate detection is never carried over into a disparity claim.

| Control | Subgroup verdict | Basis |
|---|---|---|
| matched_capacity_mlp | **SUBGROUP DISPARITY UNSUPPORTED** | The permutation null is rejected on only 0/2 seeds; at least one gap is within the 0.0078 discrete operating-point resolution. |
| memorising_mlp | **SUBGROUP DISPARITY UNSUPPORTED** | The signed direction is not stable across seeds; the permutation null is rejected on only 1/3 seeds; at least one gap is within the 0.0078 discrete operating-point resolution. |
| unbounded_decision_tree | **SUBGROUP DISPARITY UNSUPPORTED** | The permutation null is rejected on only 1/3 seeds; at least one gap is within the 0.0078 discrete operating-point resolution. |

## Conclusions

These three questions are answered separately. Aggregate attack success is not evidence of subgroup disparity, and neither is evidence about the DP results on its own.

### 1. Is aggregate membership leakage detectable?

Yes, on: memorising_mlp, unbounded_decision_tree. Mean aggregate LiRA AUC clears 0.55 with the stratified membership permutation null rejected at p<0.05 on at least 2 of the target seeds, in a consistent direction.

### 2. Does subgroup leakage differ by sex?

Not supported. No control satisfies all three requirements (stable signed direction across seeds, permutation rejection on at least 2 seeds, and a gap exceeding the operating-point resolution). Where an aggregate attack does succeed, that success is reported as aggregate leakage only.

### 3. Is the insurance dataset still suitable for the iso-epsilon subgroup research question?

Partially. The pipeline has measurable aggregate attack power at non-private capacity, so the powered spike's null is informative about the DP-SGD configurations rather than about a dead attack. But the subgroup contrast -- the actual research question -- is not resolvable here: subgroup TPR@1% FPR moves in units of one member, so the cohort is too small to separate a real disparity from operating-point resolution.

Go/no-go: **CONDITIONAL GO** -- aggregate auditing on the insurance dataset is sound, but the iso-epsilon *subgroup* comparison needs a larger dataset to reach usable resolution.

Attack AUC near 0.5 is reported here as a failure of attack power, not as evidence of privacy. No claim of subgroup privacy variance is made unless section 2 above explicitly supports it.
