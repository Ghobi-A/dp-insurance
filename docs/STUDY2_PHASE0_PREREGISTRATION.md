# Study 2, Phase 0 — natural-leakage feasibility gate (preregistration)

Frozen before any Phase 0 result exists. Implemented by
`research/study2_acs_slice.py`, `research/study2_phase0_natural_leakage.py` and
`.github/workflows/study2-phase0-natural-leakage.yml`.

## Question

Does a naturally (non-privately) trained model on a real ACS slice leak
membership strongly enough that a DP epsilon ladder built on top of it could
measure anything? If it does not, every downstream privacy-utility comparison
would be a comparison of noise with noise, and the ladder should not be built.

Phase 0 is a **gate**, not a finding. It is not intended to be interesting.

## Frozen design

| Item | Value |
| --- | --- |
| Task | Folktables `ACSIncome` (Ding et al. 2021) |
| Survey year / horizon | 2018 / 1-Year |
| State | CA |
| Sensitive attribute | `SEX` |
| Slice | 50,000 eligible rows, sampling seed `20260822` |
| Target seeds | 42, 43, 44 |
| Architecture | `Linear(d,128)->ReLU->Linear(128,128)->ReLU->Linear(128,64)->ReLU->Linear(64,1)` |
| Optimiser | Adam, lr = 1e-3 |
| Batch size | 512 |
| Epochs | 60 |
| Regularisation | none — no dropout, no weight decay, no early stopping |
| Shadows | 64, matched to the target in size, architecture and recipe |
| Primary attack | **online LiRA** |
| Secondary attack | offline LiRA (descriptive only; never decides the gate) |
| Permutation replicates | 1000, membership permuted within sensitive group |
| Target train size | 20,000 rows of the slice; the rest are non-members and the shadow pool |
| Attack cohort | `SEX x membership` cells equalised, capped at 2,500 per cell |

Feature standardisation is fitted on the whole frozen slice, so it is a property
of the data and is identical for the target and all 64 shadows; it cannot
itself distinguish them.

## Continuation gate

All three criteria must hold:

1. mean online-LiRA ROC-AUC across the three seeds **>= 0.60**;
2. online-LiRA ROC-AUC **> 0.5 on every seed**;
3. the membership-permutation null **rejected at alpha = 0.05 on all three
   seeds**.

The gate is **strict**. There is **no reconsideration band**. A result such as
mean AUC = 0.58 with 3/3 seeds significant is a **FAIL for continuation**; it is
reported descriptively as measurable leakage below the predeclared continuation
margin, and is not a basis for continuing.

The gate is evaluated **once**, on this frozen slice, and only after all three
seeds complete. A missing seed yields `INCOMPLETE`, never a verdict.

## On failure

A FAIL is **terminal** for the natural-leakage Folktables/ACS branch of Study 2:

- stop;
- do **not** launch the DP epsilon ladder;
- do **not** tune the target for more leakage;
- do **not** retry another state, year, task, architecture, epoch budget or
  model after seeing the result;
- report that the natural-leakage branch is closed.

The preregistered fallback direction is **canary-based auditing**. It is
*identified* here only. It is not implemented and not run until its own estimand
and analysis plan are frozen separately.

## On success

A PASS licenses the DP epsilon ladder on this slice. Nothing downstream is
triggered automatically: the workflow uploads artifacts, commits nothing, and
never dispatches a ladder workflow on either verdict.

## Running it

Manual dispatch only (`workflow_dispatch`, confirmation input `RUN-PHASE-0`).
Pull requests run lint, unit tests and a synthetic smoke run that never touches
the network; smoke and any budget-deviating run is stamped "Not a Phase 0
result" in its report and can never be read as the gate.
