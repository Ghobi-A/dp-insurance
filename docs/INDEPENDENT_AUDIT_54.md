# Independent correctness review of PR #54 — 2026-10-17

Reviewed original head `966b911b196300f924a55f57ac5e0decc2745102` against
`fa7d15164b870966c3ef10506d8e92263631dc8f`, followed by the fixes in this branch.

## Findings and disposition

- **One-run mathematics:** compared the implementation directly with Steinke,
  Nasr and Jagielski, arXiv:2305.08846, Corollary 5.4 and Appendix D. The
  accumulated binomial mass, maximisation over offsets, factor `2*m*delta`,
  and use of the candidate count before abstention agree. The two published
  epsilon examples remain regression tests. Confidence is conditional on fair
  independent inclusion bits and guesses derived only from audited output;
  tuning attack rules against revealed inclusion bits requires a separate
  multiplicity/selection analysis. A lower bound is not a privacy certificate.
- **Subgroup inference:** withholding disparity verdicts is correct. Permuting
  membership labels within groups simulates no association, not equal leakage
  under nonzero signal. No confirmatory equal-leakage test has been supplied.
  Historical supported/unsupported labels remain superseded, not revalidated.
- **Provenance bug fixed:** the seed CLI previously used only `args.smoke`,
  even when `_prepare_arrays` identified the input slice as smoke. A synthetic
  slice with frozen-sized settings could therefore receive a preregistered
  stamp. The executed configuration now inherits the slice's smoke flag;
  aggregation rejects smoke slices presented as non-smoke. Tests exercise
  both paths without launching research training.
- **Audit input bugs fixed:** nonfinite membership scores and invalid delta
  values in the multi-run API now fail explicitly. Fractional multi-run counts
  are rejected. Documentation no longer describes a statistical rejection as
  a proof without false-positive risk.
- **Privacy unit:** duplicated canaries toggle a k-record group. The result now
  records that unit explicitly. The API documents both k*epsilon and the
  approximate-DP group-delta sum; it does not silently compare a group bound
  with a single-record claim.

## Accounting and provenance limits

The insurance DP-SGD budget is conditional on prepared records; fitted
preprocessing and target construction are outside that budget. Preserving
categorical fields is not full-record local DP. Repeated fits, data-dependent
selection, raw evaluation statistics and research releases are not composed.
Predictable experimental seeds/non-secure RNG are reproducibility devices,
not a secure deployment configuration. These corrections do not establish
raw-record, household, or complete search/release privacy.

Config equality and fingerprints detect inconsistent supplied artifacts;
they are not independent attestation of their origin. Phase 0's stored
fingerprint is checked against supplied metadata, not a trusted external
Census digest. A clean real-data reproduction remains a research gate.
The failed finite insurance ladder must not be followed by a retuned
confirmatory iso-epsilon study. No real ACS Phase 0 dispatch was performed.

## Validation

Original-head GitHub Actions run 36678086517 passed lint, tests on Python
3.10/3.11/3.12, benchmark smoke and notebook execution. Original Phase 0,
ladder, spike and power-control workflows also completed successfully.
Passing workflow status is software evidence, not scientific confirmation.
Local validation of the follow-up changes is recorded in the PR review.
