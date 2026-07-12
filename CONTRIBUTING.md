# Contributing

## Running tests locally

```bash
pip install -e ".[dev]"
pytest
```

Coverage reports are written to `htmlcov/` automatically via `[tool.pytest.ini_options]` in `pyproject.toml`.

## Code style

- **Type hints** on all public functions (Python 3.10+ union syntax: `int | None`).
- **Docstrings** follow the Google style used in `src/dp/mechanisms.py`: one-line summary, then named sections (`Args:`, `Returns:`, `Raises:`, `Examples:`) where applicable.
- No docstring needed for private helpers (`_name`); a one-line comment is enough.
- Formatters: no hard requirement, but keep lines under 100 characters.

## Adding a new DP mechanism

1. **Implement** the function in `src/dp/mechanisms.py`. Follow the signature pattern: data input, `epsilon`, optional `sensitivity` / `delta`, `random_state`.
2. **Document** it with the four-section docstring pattern: what it does, when to use it, sensitivity calculation example, production considerations.
3. **Export** it by adding the name to `src/dp/__init__.py`.
4. **Test** it in `tests/test_mechanisms.py`. Cover: correct output shape, `ValueError` on bad inputs, and at least one statistical sanity check (e.g. mean of large noise sample ≈ 0).
5. **Reference** the paper that proves the DP guarantee in the docstring.

## Reporting issues

Open a GitHub issue with:
- A minimal reproducible example.
- The expected vs. actual behaviour.
- Python version and relevant package versions (`pip show dp-insurance`).

For security-relevant issues (e.g. a mechanism that does not satisfy its stated DP guarantee), please open a private advisory rather than a public issue.

## Relevant papers

- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.* — canonical reference for Laplace, Gaussian, and exponential mechanisms.
- Abadi et al. (2016). *Deep Learning with Differential Privacy.* — DP-SGD and the moments accountant.
- Balle & Wang (2018). *Improving the Gaussian Mechanism for Differential Privacy.* — tighter σ bounds than the classical formula.
- Erlingsson et al. (2014). *RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response.* — local DP and randomized response at scale.
- Steinke, Nasr & Jagielski (2023). *Privacy Auditing with One (1) Training Run.* — the one-run auditing bound in `src/dp/audit.py`.
- Jagielski, Ullman & Oprea (2020). *Auditing Differentially Private Machine Learning.* — Clopper–Pearson multi-run auditing.
- Carlini et al. (2022). *Membership Inference Attacks From First Principles.* — offline/online LiRA and the TPR@low-FPR metric in `src/dp/attacks.py`.
- Nasr et al. (2021). *Adversary Instantiation: Lower Bounds for Differentially Private Machine Learning.* — worst-case canary design behind `src/dp/canaries.py`.
- Bagdasaryan, Poursaeed & Shmatikov (2019). *Differential Privacy Has Disparate Impact on Model Accuracy.* — the privacy–fairness interaction.
