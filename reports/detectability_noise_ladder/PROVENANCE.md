# Archived ladder output — provenance

These files are copied unchanged from the GitHub Actions artifact
`detectability-noise-ladder` (artifact `9433303328`) of
[Actions run 32441370755](https://github.com/Ghobi-A/dp-insurance/actions/runs/32441370755),
commit `d8550bf89465ac13f23776aec01f5e345289f515`, completed 2026-08-21.
They were downloaded on 2026-10-20 during the independent statistical review,
because Actions artifacts expire and the repository previously held only the
point means quoted in `docs/PUBLICATION_STATUS.md`.

Downloaded zip SHA-256: `920bfd414e08a3a4d060c82f00aa50e12a4e88d01c21a38268241c61fb41a92f`.
The per-point JSON files (about 0.75 MB each) and figures other than
`detectability_auc.png` are not copied; `noise_ladder.json` carries every
point's per-seed results and `per_example.csv` carries the per-example scores.

| File | SHA-256 |
|---|---|
| `noise_ladder.json` | `8cb6e92392409ac2a1642aa202f7769cc9add877ffee3c76d09a2205a6012436` |
| `noise_ladder.md` | `494d885e1f38a63c28f350c12b756515edf549181003b43debc9e3688ce74039` |
| `noise_ladder.csv` | `34570fdbe06e11f9a32235155218bd44e5d5d399f57abe7779ba8aef795c04ee` |
| `observations.csv` | `5e849dfd69a20dd683b692f2d58ae5d2971634db220c19da55fc8ad2e56fa59d` |
| `per_example.csv` | `002fc9b991041984f944b6d0e29c7c84db64460d95e98b5a5db3042256ea8c47` |
| `detectability_auc.png` | `1164cb63ed2951f4f1cc65bca393b45e7355fed2f05b33a650b0937a97dcb7f5` |

Re-aggregating the six downloaded point files with
`python research/detectability_noise_ladder.py aggregate --input-dir <points> --output-dir <fresh dir>`
reproduced every point decision and the frontier summary exactly.

## Runtime and retraining reproduction

The artifact does not record library versions. The point-job logs show the CI
runtime: CPython 3.11.16, `torch 2.13.0+cpu`, `opacus 1.6.0`, `numpy 2.4.6`,
`scikit-learn 1.9.0` (torch installed unpinned from the CPU index). Ladder
point artifacts now record a `runtime` block.

Retraining the non-private point for seed 42 in the review environment
(CPython 3.13, `torch 2.14.1+cpu`, `opacus 1.6.0`, `numpy 2.5.4`) reproduced
the split, cohort and shadow-schedule hashes exactly but gave offline AUC
0.6078 versus the recorded 0.5859 (test AUC 0.9336 versus 0.9407). Seeds fix
the experimental design, not the floating-point trajectory across torch builds.
The difference is about 0.9 null standard deviations of a seed AUC, so a binary
rule placed close to its threshold (epsilon 8 mean 0.5503 against 0.55) is not
robust to the numerical environment.

Read `noise_ladder.md` with the review caveats in
[`docs/STATISTICAL_REVIEW.md`](../../docs/STATISTICAL_REVIEW.md): the frozen
rule's failures at finite epsilon are low-power non-detections, not evidence of
absent leakage, and the non-private endpoint differs from the finite points in
clipping and sampling as well as noise.
