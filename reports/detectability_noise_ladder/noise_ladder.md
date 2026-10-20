# Worst-case neural auditability ladder


**Auditability ladder for a deliberately memorising neural target.** This is not the universal DP-SGD detectability frontier: it characterises one intentionally attackable architecture on one dataset, chosen because the attack-power control showed it is attackable at all.

Context from the attack-power control that motivates this design:

| Control | Outcome |
|---|---|
| Matched-capacity MLP (mirrors the DP target recipe) | attack inconclusive -- generalised rather than memorised |
| Deliberately memorising MLP | aggregate leakage detected |
| This ladder | effect of DP noise on an intentionally attackable neural target |

## 1. Pre-registered question and rules

> At what privacy-noise level does aggregate natural membership leakage cease to be reproducibly detectable on the insurance `high_cost` task?

Rules were fixed in `docs/NOISE_LADDER_PREREGISTRATION.md` and the adversary in `docs/THREAT_MODEL.md`, both committed before this experiment ran.

**Detection rule** (unchanged from the original registration): mean offline-LiRA AUC >= 0.55, AUC > 0.5 on every seed, and the Holm-adjusted permutation null rejected on >= 2 of 3 seeds at alpha = 0.05. Holm adjustment runs across ladder points within each seed.

**Amendment 1**: every ladder point and seed receives the complete matched-shadow attack. Memorisation is explanatory metadata, never an execution gate, because discarding low-memorisation observations would truncate the sample on a variable correlated with the outcome. Points are classified by combining two independent decisions:

* `DETECTABLE WITH MEMORISATION`
* `DETECTABLE DESPITE LOW MEASURED MEMORISATION`
* `UNDETECTABLE DESPITE MEMORISATION`
* `UNDETECTABLE WITH LOW MEMORISATION`

No point is ever called private, safe or verified.

## 2. Threat model summary

The adversary knows the architecture, optimiser, batch size, epochs, clipping norm, preprocessing, task definition, source distribution and (at finite points) the noise multiplier and claimed (epsilon, delta). They do not know the target's membership set. They train matched shadows on the same pool. Membership is record-level inclusion; the sensitive attribute is `sex`. Offline LiRA is primary, online LiRA secondary and reported only with >= 10 IN and >= 10 OUT observations per example.

## 3. Fixed experimental design

| Parameter | Value |
|---|---|
| Task | `high_cost` |
| Sensitive attribute | `sex` |
| Architecture | `Linear(d,128->128->64->1)` with ReLU |
| Epochs | 400 |
| Batch size | 64 |
| Optimiser | Adam, lr = 0.001 |
| Clipping norm | 1.0 (fixed, public) |
| delta | 1e-05 |
| Accountant | RDP |
| Sampling | Poisson |
| Seeds | 42, 43, 44 |
| Shadows per point per seed | 32 |
| Bootstrap / permutation replicates | 1000 / 1000 |

Only the noise multiplier changes between finite points. Cohort indices, shadow schedules and model seeds derive from the target seed alone, so points are paired.

## 4. Privacy accounting

| Point | Requested epsilon | Noise multiplier | Achieved epsilon | Clipping |
|---|---|---:|---:|---:|
| non-private | non-private | - | - | - |
| eps32 | 32 | 1.3486 | 31.0307 | 1.0 |
| eps16 | 16 | 2.1112 | 15.5427 | 1.0 |
| eps8 | 8 | 3.5791 | 7.7749 | 1.0 |
| eps4 | 4 | 6.3965 | 3.8939 | 1.0 |
| eps2 | 2 | 11.8359 | 1.9432 | 1.0 |

Requested epsilon is an input. The achieved epsilon read back from the RDP accountant after training is the reported figure.

## 5. Memorisation metrics

| Point | Seed | Train AUC | Test AUC | AUC gap | Train BCE | Test BCE | Loss gap | Acc gap | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| non-private | 42 | 0.9999 | 0.9407 | 0.0592 | 0.0153 | 0.7611 | 0.7458 | 0.0935 | PASS |
| non-private | 43 | 0.9999 | 0.9232 | 0.0767 | 0.0105 | 1.0720 | 1.0615 | 0.1208 | PASS |
| non-private | 44 | 0.9999 | 0.9387 | 0.0612 | 0.0267 | 0.7887 | 0.7620 | 0.0804 | PASS |
| eps32 | 42 | 0.9632 | 0.9626 | 0.0006 | 0.2414 | 0.1984 | -0.0430 | -0.0076 | FAIL |
| eps32 | 43 | 0.9667 | 0.9211 | 0.0456 | 0.2232 | 0.4331 | 0.2098 | 0.0444 | PASS |
| eps32 | 44 | 0.9606 | 0.9402 | 0.0204 | 0.2468 | 0.2941 | 0.0472 | 0.0123 | PASS |
| eps16 | 42 | 0.9576 | 0.9627 | -0.0051 | 0.2542 | 0.2074 | -0.0468 | -0.0073 | FAIL |
| eps16 | 43 | 0.9621 | 0.9188 | 0.0433 | 0.2323 | 0.4432 | 0.2109 | 0.0470 | PASS |
| eps16 | 44 | 0.9482 | 0.9513 | -0.0031 | 0.2614 | 0.2754 | 0.0139 | 0.0050 | FAIL |
| eps8 | 42 | 0.9517 | 0.9576 | -0.0059 | 0.2766 | 0.2245 | -0.0521 | -0.0130 | FAIL |
| eps8 | 43 | 0.9580 | 0.9192 | 0.0387 | 0.2509 | 0.4603 | 0.2094 | 0.0423 | PASS |
| eps8 | 44 | 0.9424 | 0.9525 | -0.0101 | 0.2917 | 0.2717 | -0.0200 | -0.0046 | FAIL |
| eps4 | 42 | 0.9489 | 0.9559 | -0.0070 | 0.3055 | 0.2520 | -0.0535 | -0.0127 | FAIL |
| eps4 | 43 | 0.9552 | 0.9182 | 0.0370 | 0.2736 | 0.4730 | 0.1995 | 0.0367 | PASS |
| eps4 | 44 | 0.9406 | 0.9535 | -0.0129 | 0.3221 | 0.2770 | -0.0451 | -0.0043 | FAIL |
| eps2 | 42 | 0.9454 | 0.9541 | -0.0087 | 0.3095 | 0.2618 | -0.0477 | -0.0144 | FAIL |
| eps2 | 43 | 0.9520 | 0.9181 | 0.0339 | 0.2871 | 0.4515 | 0.1644 | 0.0318 | PASS |
| eps2 | 44 | 0.9373 | 0.9532 | -0.0159 | 0.3392 | 0.2863 | -0.0529 | 0.0096 | FAIL |

## 6. Aggregate LiRA results

| Point | Seed | Attack | LiRA AUC | TPR@0.1% | TPR@1% | TPR@10% | Detected @1% | FP @1% | OUT | IN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| non-private | 42 | lira-offline | 0.5859 | 0.0469 | 0.0547 | 0.2070 | 14/256 | 2/256 | 11 | 20 |
| non-private | 42 | lira-online | 0.6420 | 0.0078 | 0.0078 | 0.2031 | 2/256 | 0/256 | 11 | 20 |
| non-private | 43 | lira-offline | 0.5794 | 0.0310 | 0.0581 | 0.1512 | 15/258 | 2/258 | 11 | 20 |
| non-private | 43 | lira-online | 0.6255 | 0.0116 | 0.0116 | 0.1318 | 3/258 | 0/258 | 11 | 20 |
| non-private | 44 | lira-offline | 0.5582 | 0.0344 | 0.0725 | 0.1450 | 19/262 | 2/262 | 11 | 20 |
| non-private | 44 | lira-online | 0.5602 | 0.0038 | 0.0076 | 0.1183 | 2/262 | 2/262 | 11 | 20 |
| eps32 | 42 | lira-offline | 0.5917 | 0.0000 | 0.0195 | 0.1914 | 5/256 | 2/256 | 11 | 20 |
| eps32 | 42 | lira-online | 0.5356 | 0.0039 | 0.0078 | 0.1562 | 2/256 | 2/256 | 11 | 20 |
| eps32 | 43 | lira-offline | 0.5258 | 0.0078 | 0.0078 | 0.1163 | 2/258 | 0/258 | 11 | 20 |
| eps32 | 43 | lira-online | 0.5557 | 0.0116 | 0.0194 | 0.1047 | 5/258 | 2/258 | 11 | 20 |
| eps32 | 44 | lira-offline | 0.4994 | 0.0000 | 0.0115 | 0.0954 | 3/262 | 2/262 | 11 | 20 |
| eps32 | 44 | lira-online | 0.5107 | 0.0000 | 0.0038 | 0.0763 | 1/262 | 1/262 | 11 | 20 |
| eps16 | 42 | lira-offline | 0.5876 | 0.0000 | 0.0078 | 0.1914 | 2/256 | 1/256 | 11 | 20 |
| eps16 | 42 | lira-online | 0.5117 | 0.0078 | 0.0117 | 0.0898 | 3/256 | 2/256 | 11 | 20 |
| eps16 | 43 | lira-offline | 0.5340 | 0.0000 | 0.0039 | 0.1473 | 1/258 | 1/258 | 11 | 20 |
| eps16 | 43 | lira-online | 0.5423 | 0.0039 | 0.0155 | 0.1550 | 4/258 | 1/258 | 11 | 20 |
| eps16 | 44 | lira-offline | 0.5101 | 0.0000 | 0.0038 | 0.1221 | 1/262 | 2/262 | 11 | 20 |
| eps16 | 44 | lira-online | 0.5079 | 0.0038 | 0.0115 | 0.0687 | 3/262 | 2/262 | 11 | 20 |
| eps8 | 42 | lira-offline | 0.5792 | 0.0000 | 0.0078 | 0.1797 | 2/256 | 2/256 | 11 | 20 |
| eps8 | 42 | lira-online | 0.5097 | 0.0039 | 0.0039 | 0.1016 | 1/256 | 0/256 | 11 | 20 |
| eps8 | 43 | lira-offline | 0.5479 | 0.0078 | 0.0116 | 0.1202 | 3/258 | 1/258 | 11 | 20 |
| eps8 | 43 | lira-online | 0.5211 | 0.0000 | 0.0271 | 0.1202 | 7/258 | 2/258 | 11 | 20 |
| eps8 | 44 | lira-offline | 0.5238 | 0.0000 | 0.0038 | 0.1374 | 1/262 | 1/262 | 11 | 20 |
| eps8 | 44 | lira-online | 0.4881 | 0.0000 | 0.0115 | 0.0573 | 3/262 | 2/262 | 11 | 20 |
| eps4 | 42 | lira-offline | 0.5715 | 0.0000 | 0.0312 | 0.1719 | 8/256 | 2/256 | 11 | 20 |
| eps4 | 42 | lira-online | 0.4964 | 0.0039 | 0.0039 | 0.0625 | 1/256 | 0/256 | 11 | 20 |
| eps4 | 43 | lira-offline | 0.5289 | 0.0000 | 0.0155 | 0.1318 | 4/258 | 2/258 | 11 | 20 |
| eps4 | 43 | lira-online | 0.5004 | 0.0000 | 0.0116 | 0.0659 | 3/258 | 2/258 | 11 | 20 |
| eps4 | 44 | lira-offline | 0.5288 | 0.0000 | 0.0076 | 0.1412 | 2/262 | 1/262 | 11 | 20 |
| eps4 | 44 | lira-online | 0.4845 | 0.0000 | 0.0076 | 0.0763 | 2/262 | 1/262 | 11 | 20 |
| eps2 | 42 | lira-offline | 0.5472 | 0.0000 | 0.0156 | 0.1523 | 4/256 | 2/256 | 11 | 20 |
| eps2 | 42 | lira-online | 0.4868 | 0.0039 | 0.0039 | 0.0547 | 1/256 | 0/256 | 11 | 20 |
| eps2 | 43 | lira-offline | 0.5198 | 0.0000 | 0.0078 | 0.1240 | 2/258 | 2/258 | 11 | 20 |
| eps2 | 43 | lira-online | 0.4908 | 0.0000 | 0.0155 | 0.0930 | 4/258 | 2/258 | 11 | 20 |
| eps2 | 44 | lira-offline | 0.5296 | 0.0000 | 0.0000 | 0.1031 | 0/262 | 0/262 | 11 | 20 |
| eps2 | 44 | lira-online | 0.4986 | 0.0000 | 0.0076 | 0.0878 | 2/262 | 1/262 | 11 | 20 |

## 7. Permutation and bootstrap results

Permutation nulls reshuffle membership within each sex, preserving exact cell counts with scores fixed. Holm adjustment is across ladder points within each seed.

| Point | Seed | Attack | AUC | Null AUC 95% | p | Holm p | TPR@1% | p | Holm p | Bootstrap TPR 95% |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| non-private | 42 | lira-offline | 0.5859 | [0.4515, 0.5537] | 0.0010 | 0.0060 | 0.0547 | 0.0040 | 0.0240 | [0.0273, 0.0820] |
| non-private | 42 | lira-online | 0.6420 | [0.4453, 0.5506] | 0.0010 | 0.0060 | 0.0078 | 0.7113 | 1.0000 | [0.0000, 0.0273] |
| non-private | 43 | lira-offline | 0.5794 | [0.4543, 0.5490] | 0.0020 | 0.0120 | 0.0581 | 0.0020 | 0.0120 | [0.0233, 0.0969] |
| non-private | 43 | lira-online | 0.6255 | [0.4507, 0.5500] | 0.0010 | 0.0060 | 0.0116 | 0.4945 | 1.0000 | [0.0039, 0.0349] |
| non-private | 44 | lira-offline | 0.5582 | [0.4533, 0.5509] | 0.0130 | 0.0779 | 0.0725 | 0.0010 | 0.0060 | [0.0191, 0.1069] |
| non-private | 44 | lira-online | 0.5602 | [0.4509, 0.5489] | 0.0080 | 0.0480 | 0.0076 | 0.6633 | 1.0000 | [0.0000, 0.0458] |
| eps32 | 42 | lira-offline | 0.5917 | [0.4478, 0.5515] | 0.0010 | 0.0060 | 0.0195 | 0.2318 | 0.9271 | [0.0000, 0.0742] |
| eps32 | 42 | lira-online | 0.5356 | [0.4458, 0.5522] | 0.0749 | 0.3746 | 0.0078 | 0.6803 | 1.0000 | [0.0000, 0.0234] |
| eps32 | 43 | lira-offline | 0.5258 | [0.4554, 0.5511] | 0.1688 | 0.3556 | 0.0078 | 0.6903 | 1.0000 | [0.0000, 0.0466] |
| eps32 | 43 | lira-online | 0.5557 | [0.4512, 0.5453] | 0.0130 | 0.0649 | 0.0194 | 0.2098 | 1.0000 | [0.0039, 0.0465] |
| eps32 | 44 | lira-offline | 0.4994 | [0.4472, 0.5488] | 0.5115 | 0.7073 | 0.0115 | 0.5305 | 1.0000 | [0.0000, 0.0382] |
| eps32 | 44 | lira-online | 0.5107 | [0.4490, 0.5495] | 0.3457 | 1.0000 | 0.0038 | 0.8811 | 1.0000 | [0.0000, 0.0153] |
| eps16 | 42 | lira-offline | 0.5876 | [0.4509, 0.5507] | 0.0010 | 0.0060 | 0.0078 | 0.6823 | 1.0000 | [0.0000, 0.0742] |
| eps16 | 42 | lira-online | 0.5117 | [0.4484, 0.5479] | 0.3047 | 1.0000 | 0.0117 | 0.5045 | 1.0000 | [0.0000, 0.0235] |
| eps16 | 43 | lira-offline | 0.5340 | [0.4497, 0.5526] | 0.0889 | 0.3556 | 0.0039 | 0.8751 | 1.0000 | [0.0000, 0.0426] |
| eps16 | 43 | lira-online | 0.5423 | [0.4493, 0.5484] | 0.0480 | 0.1918 | 0.0155 | 0.3257 | 1.0000 | [0.0000, 0.0388] |
| eps16 | 44 | lira-offline | 0.5101 | [0.4465, 0.5491] | 0.3536 | 0.7073 | 0.0038 | 0.8751 | 1.0000 | [0.0000, 0.0192] |
| eps16 | 44 | lira-online | 0.5079 | [0.4536, 0.5483] | 0.4036 | 1.0000 | 0.0115 | 0.4855 | 1.0000 | [0.0000, 0.0267] |
| eps8 | 42 | lira-offline | 0.5792 | [0.4504, 0.5514] | 0.0020 | 0.0060 | 0.0078 | 0.7043 | 1.0000 | [0.0000, 0.0859] |
| eps8 | 42 | lira-online | 0.5097 | [0.4472, 0.5460] | 0.3367 | 1.0000 | 0.0039 | 0.8871 | 1.0000 | [0.0000, 0.0117] |
| eps8 | 43 | lira-offline | 0.5479 | [0.4493, 0.5485] | 0.0290 | 0.1449 | 0.0116 | 0.5045 | 1.0000 | [0.0000, 0.0426] |
| eps8 | 43 | lira-online | 0.5211 | [0.4515, 0.5478] | 0.1958 | 0.5874 | 0.0271 | 0.0879 | 0.5275 | [0.0000, 0.0465] |
| eps8 | 44 | lira-offline | 0.5238 | [0.4490, 0.5505] | 0.1898 | 0.5744 | 0.0038 | 0.8911 | 1.0000 | [0.0000, 0.0305] |
| eps8 | 44 | lira-online | 0.4881 | [0.4527, 0.5503] | 0.6883 | 1.0000 | 0.0115 | 0.4825 | 1.0000 | [0.0000, 0.0305] |
| eps4 | 42 | lira-offline | 0.5715 | [0.4513, 0.5514] | 0.0070 | 0.0140 | 0.0312 | 0.0420 | 0.2098 | [0.0000, 0.0703] |
| eps4 | 42 | lira-online | 0.4964 | [0.4474, 0.5483] | 0.5475 | 1.0000 | 0.0039 | 0.8871 | 1.0000 | [0.0000, 0.0117] |
| eps4 | 43 | lira-offline | 0.5289 | [0.4481, 0.5465] | 0.0969 | 0.3556 | 0.0155 | 0.3327 | 1.0000 | [0.0000, 0.0466] |
| eps4 | 43 | lira-online | 0.5004 | [0.4532, 0.5497] | 0.4925 | 0.9850 | 0.0116 | 0.4865 | 1.0000 | [0.0000, 0.0310] |
| eps4 | 44 | lira-offline | 0.5288 | [0.4523, 0.5506] | 0.1279 | 0.5744 | 0.0076 | 0.7203 | 1.0000 | [0.0000, 0.0344] |
| eps4 | 44 | lira-online | 0.4845 | [0.4511, 0.5498] | 0.7283 | 1.0000 | 0.0076 | 0.6853 | 1.0000 | [0.0000, 0.0229] |
| eps2 | 42 | lira-offline | 0.5472 | [0.4506, 0.5499] | 0.0330 | 0.0330 | 0.0156 | 0.3487 | 1.0000 | [0.0000, 0.0977] |
| eps2 | 42 | lira-online | 0.4868 | [0.4483, 0.5479] | 0.7033 | 1.0000 | 0.0039 | 0.8741 | 1.0000 | [0.0000, 0.0156] |
| eps2 | 43 | lira-offline | 0.5198 | [0.4491, 0.5448] | 0.1958 | 0.3556 | 0.0078 | 0.6843 | 1.0000 | [0.0000, 0.0349] |
| eps2 | 43 | lira-online | 0.4908 | [0.4506, 0.5512] | 0.6474 | 0.9850 | 0.0155 | 0.3516 | 1.0000 | [0.0000, 0.0349] |
| eps2 | 44 | lira-offline | 0.5296 | [0.4502, 0.5519] | 0.1149 | 0.5744 | 0.0000 | 1.0000 | 1.0000 | [0.0000, 0.0191] |
| eps2 | 44 | lira-online | 0.4986 | [0.4503, 0.5497] | 0.5604 | 1.0000 | 0.0076 | 0.7023 | 1.0000 | [0.0000, 0.0229] |

## 8. Detectability decision per ladder point

| Point | Achieved epsilon | Noise | Gate passed | Mean offline AUC | Mean TPR@1% | Min Holm p | Decision | Basis |
|---|---:|---:|---:|---:|---:|---:|---|---|
| non-private | - | - | 3/3 | 0.5745 | 0.0618 | 0.0060 | **DETECTABLE WITH MEMORISATION** | Detected (mean offline-LiRA AUC 0.5745 (>= 0.55), above 0.5 on every seed, Holm-adjusted permutation null rejected on 2/3 seeds (>= 2 required)); target memorised on every seed. |
| eps32 | 31.0307 | 1.3486 | 2/3 | 0.5390 | 0.0129 | 0.0060 | **UNDETECTABLE WITH LOW MEMORISATION** | Not detected (mean AUC 0.5390 < 0.55; at least one seed at or below chance; Holm-adjusted rejection on only 1/3 seeds (2 required)) and measured memorisation is below the 0.03 criterion on at least one seed. |
| eps16 | 15.5427 | 2.1112 | 1/3 | 0.5439 | 0.0052 | 0.0060 | **UNDETECTABLE WITH LOW MEMORISATION** | Not detected (mean AUC 0.5439 < 0.55; Holm-adjusted rejection on only 1/3 seeds (2 required)) and measured memorisation is below the 0.03 criterion on at least one seed. |
| eps8 | 7.7749 | 3.5791 | 1/3 | 0.5503 | 0.0078 | 0.0060 | **UNDETECTABLE WITH LOW MEMORISATION** | Not detected (Holm-adjusted rejection on only 1/3 seeds (2 required)) and measured memorisation is below the 0.03 criterion on at least one seed. |
| eps4 | 3.8939 | 6.3965 | 1/3 | 0.5431 | 0.0181 | 0.0140 | **UNDETECTABLE WITH LOW MEMORISATION** | Not detected (mean AUC 0.5431 < 0.55; Holm-adjusted rejection on only 1/3 seeds (2 required)) and measured memorisation is below the 0.03 criterion on at least one seed. |
| eps2 | 1.9432 | 11.8359 | 1/3 | 0.5322 | 0.0078 | 0.0330 | **UNDETECTABLE WITH LOW MEMORISATION** | Not detected (mean AUC 0.5322 < 0.55; Holm-adjusted rejection on only 1/3 seeds (2 required)) and measured memorisation is below the 0.03 criterion on at least one seed. |

## 9. Detectability-frontier bracket

detectable at epsilon = None (non-private); not detectable at epsilon = 31.0307 (eps32); therefore the empirical transition lies between them. Six points do not support interpolating a precise threshold.

```text
detectable at epsilon = - (non-private)
not detectable at epsilon = 31.0307 (eps32)
therefore the empirical transition lies between them
```

No interpolation is performed across six points, and no point is described as private on the basis of non-detection.

## 10. Secondary subgroup analysis

Primary statistic: signed contrast `male TPR@1% - female TPR@1%`. The absolute max-minus-min gap is descriptive only -- it is positively biased under the null. `resolution = 1 / subgroup_member_count` is the smallest movement a single member can produce.

| Point | Seed | Female TPR@1% | Male TPR@1% | Signed gap | Absolute gap (descriptive) | Resolution | Signed Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| non-private | 42 | 0.0234 | 0.0859 | 0.0625 | 0.0625 | 0.0078 | 0.0959 |
| non-private | 43 | 0.0620 | 0.0620 | 0.0000 | 0.0000 | 0.0078 | 1.0000 |
| non-private | 44 | 0.0763 | 0.0687 | -0.0076 | 0.0076 | 0.0076 | 1.0000 |
| eps32 | 42 | 0.0078 | 0.0312 | 0.0234 | 0.0234 | 0.0078 | 0.9500 |
| eps32 | 43 | 0.0310 | 0.0078 | -0.0233 | 0.0233 | 0.0078 | 1.0000 |
| eps32 | 44 | 0.0229 | 0.0534 | 0.0305 | 0.0305 | 0.0076 | 0.7632 |
| eps16 | 42 | 0.0000 | 0.0156 | 0.0156 | 0.0156 | 0.0078 | 0.9850 |
| eps16 | 43 | 0.0078 | 0.0000 | -0.0078 | 0.0078 | 0.0078 | 1.0000 |
| eps16 | 44 | 0.0076 | 0.0611 | 0.0534 | 0.0534 | 0.0076 | 0.2158 |
| eps8 | 42 | 0.0000 | 0.0156 | 0.0156 | 0.0156 | 0.0078 | 0.9850 |
| eps8 | 43 | 0.0155 | 0.0078 | -0.0078 | 0.0078 | 0.0078 | 1.0000 |
| eps8 | 44 | 0.0000 | 0.0382 | 0.0382 | 0.0382 | 0.0076 | 0.5045 |
| eps4 | 42 | 0.0625 | 0.0156 | -0.0469 | 0.0469 | 0.0078 | 0.2517 |
| eps4 | 43 | 0.0078 | 0.0233 | 0.0155 | 0.0155 | 0.0078 | 1.0000 |
| eps4 | 44 | 0.0076 | 0.0305 | 0.0229 | 0.0229 | 0.0076 | 0.9351 |
| eps2 | 42 | 0.0859 | 0.0234 | -0.0625 | 0.0625 | 0.0078 | 0.1099 |
| eps2 | 43 | 0.0000 | 0.0388 | 0.0388 | 0.0388 | 0.0078 | 0.6653 |
| eps2 | 44 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0076 | 1.0000 |

| Point | Subgroup verdict | Basis |
|---|---|---|
| non-private | **SUBGROUP DISPARITY UNSUPPORTED** | The signed direction is not stable across non-zero seeds; the adjusted permutation null is rejected on only 0/3 seeds (2 required); at least one contrast is within the 0.0078 one-person resolution. |
| eps32 | **SUBGROUP DISPARITY UNSUPPORTED** | The signed direction is not stable across non-zero seeds; the adjusted permutation null is rejected on only 0/3 seeds (2 required). |
| eps16 | **SUBGROUP DISPARITY UNSUPPORTED** | The signed direction is not stable across non-zero seeds; the adjusted permutation null is rejected on only 0/3 seeds (2 required); at least one contrast is within the 0.0078 one-person resolution. |
| eps8 | **SUBGROUP DISPARITY UNSUPPORTED** | The signed direction is not stable across non-zero seeds; the adjusted permutation null is rejected on only 0/3 seeds (2 required); at least one contrast is within the 0.0078 one-person resolution. |
| eps4 | **SUBGROUP DISPARITY UNSUPPORTED** | The signed direction is not stable across non-zero seeds; the adjusted permutation null is rejected on only 0/3 seeds (2 required). |
| eps2 | **SUBGROUP DISPARITY UNSUPPORTED** | The signed direction is not stable across non-zero seeds; the adjusted permutation null is rejected on only 0/3 seeds (2 required); at least one contrast is within the 0.0078 one-person resolution. |

## 10b. Generalisation gap and measured leakage

**These analyses measure association only. DP noise changes both generalisation and leakage, so this experiment does not identify an effect of DP on membership leakage independently of its effect on memorisation.**

Every seed-level observation is retained (18 in total), including those below the memorisation criterion. Spearman rank correlations, with seed-stratified bootstrap intervals:

| Generalisation gap | Leakage metric | n | Spearman rho | 95% CI |
|---|---|---:|---:|---|
| bce_gap | lira_auc | 18 | -0.0237 | [-0.4216, 0.3919] |
| bce_gap | tpr_at_1pct | 18 | 0.3318 | [-0.2435, 0.7389] |
| auc_gap | lira_auc | 18 | 0.1950 | [-0.1993, 0.5588] |
| auc_gap | tpr_at_1pct | 18 | 0.5323 | [0.1029, 0.8296] |
| accuracy_gap | lira_auc | 18 | -0.0568 | [-0.4870, 0.4190] |
| accuracy_gap | tpr_at_1pct | 18 | 0.2140 | [-0.3160, 0.6320] |

Figures `lira_auc_vs_bce_gap.png`, `lira_auc_vs_auc_gap.png`, `lira_auc_vs_accuracy_gap.png` and `tpr_vs_bce_gap.png` plot every observation, distinguishing non-private from DP-SGD and threshold-passing from low-memorisation points.

No causal, mediation or "beyond generalisation" claim is made or implied.

## 11. Conclusion

### 1. Does the attack detect aggregate leakage without DP?

Yes. The non-private target reaches mean offline-LiRA AUC 0.5745 with the Holm-adjusted permutation null rejected on the required number of seeds. The instrument works.

### 2. At which DP noise levels is aggregate leakage detectable?

At none of the finite privacy points tested. Detection survives only in the non-private control, if at all.

### 3. Where does detection disappear?

detectable at epsilon = None (non-private); not detectable at epsilon = 31.0307 (eps32); therefore the empirical transition lies between them. Six points do not support interpolating a precise threshold.

### 4. Are subgroup differences supported?

**No.** No ladder point satisfies the pre-registered signed-contrast rule. Aggregate detection, where it occurs, is reported as aggregate leakage only and is never read as subgroup disparity.

### 5. Is the insurance dataset adequate for aggregate auditing?

Yes, at non-private capacity, and only there -- every finite point tested falls below detectability.

### 6. Is it adequate for small subgroup-leakage comparisons?

**No.** Subgroup TPR@1% FPR moves in steps of 0.0078 (one member out of 128), and at a 1% FPR the operating point admits roughly one false positive per group. The measurement grid is coarser than the effects a subgroup claim would need to resolve, independently of how much noise is applied.

---

Non-detection is reported as *not detected at this power*, never as privacy, safety or verification of a DP guarantee. No subgroup claim is made from unstable or single-seed results.
