# TMLR-style review draft

The PDF is a research review copy, **not submitted or under review**. The
`tmlr.sty` was retrieved from the official
[JmlrOrg/tmlr-style-file](https://github.com/JmlrOrg/tmlr-style-file/blob/main/tmlr.sty),
blob `c0c62d03c6e2383c688b20965de03800dc15c79c` on 1 October 2026; only trailing
whitespace is normalized in the local copy. The header
overrides the review status; the preprint option is a formatting choice.

Build from the maintained Markdown source, with Pandoc, pdfLaTeX and the usual
TeX packages installed:

```bash
python research/configuration_selection/build_review_pdf.py
```

Editable `.tex` and rendered `.pdf` outputs are generated from the same
Markdown. Unicode punctuation is normalized for pdfLaTeX; evidence and
reference numbers are retained. This is a typesetting conversion, not a new
analysis. The references are numbered explicitly in the maintained source.

Before a real double-blind submission, review all content, finalize actual
author/disclosure information and prepare an anonymous supplement. Public
repository provenance is not anonymous. See `docs/SUBMISSION_REVIEW.md`.
