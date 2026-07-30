# Streamlit results explorer

The dashboard is a read-only interface over the benchmark outputs in
`reports/generated/summary.csv`. It does not train models, accept uploaded data,
or provide insurance pricing.

## Run locally

```bash
pip install -e ".[app]"
streamlit run streamlit_app.py
```

To regenerate every displayed result first:

```bash
pip install -e ".[experimental,dev,app]"
python -m dp.benchmark --task all --seeds 0 1 2 3 4 --output-dir reports/generated
python -m dp.reporting --results reports/generated --reports-dir reports
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Connect the GitHub repository.
2. Select `streamlit_app.py` as the entrypoint.
3. Use Python 3.10 or newer.
4. The root `requirements.txt` installs the package and dashboard dependencies.

The app intentionally uses committed, versioned outputs so deployment remains
fast, deterministic and inexpensive.
