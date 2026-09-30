"""Separate ACSIncome utility-study preparation, with fixed public representation."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("CA", "OR", "WA", "NV", "AZ")
COLUMNS = ("SERIALNO", "SPORDER", "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP",
           "RELP", "WKHP", "SEX", "RAC1P", "PINCP", "PWGTP")
NUMERIC_BOUNDS = {"AGEP": (0, 100), "SCHL": (0, 24), "WKHP": (0, 100)}
CATEGORIES = {"COW": tuple(range(1, 10)), "MAR": tuple(range(1, 6)),
              "RELP": tuple(range(18)), "SEX": (1, 2), "RAC1P": tuple(range(1, 10))}
HASH_BUCKETS = {"OCCP": 32, "POBP": 16}


def download(state: str, cache: Path) -> Path:
    if state not in STATES:
        raise ValueError("state outside frozen study")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"acs2018_{state}.zip"
    if path.exists():
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("corrupt cached archive")
        return path
    url = f"https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_p{state.lower()}.zip"
    partial = path.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError("corrupt download")
    partial.replace(path)
    return path


def prepare_frame(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Eligibility matches ACSIncome; public hashing is an explicit representation variant."""
    eligible = raw[(raw.AGEP > 16) & (raw.PINCP > 100) & (raw.WKHP > 0) & (raw.PWGTP >= 1)]
    eligible = eligible.dropna(subset=list(COLUMNS)).sort_values(["SERIALNO", "SPORDER"])
    if eligible.duplicated(["SERIALNO", "SPORDER"]).any():
        raise ValueError("duplicate person identifiers")
    feature_columns = list(NUMERIC_BOUNDS) + list(CATEGORIES) + list(HASH_BUCKETS)
    if not np.isfinite(eligible[feature_columns + ["PINCP"]].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite ACS values")
    columns, names = [], []
    for name, (low, high) in NUMERIC_BOUNDS.items():
        columns.append(2 * np.clip((eligible[name].to_numpy() - low) / (high - low), 0, 1) - 1)
        names.append(name)
    for name, domain in CATEGORIES.items():
        values = eligible[name].to_numpy()
        for value in domain:
            columns.append((values == value).astype(float))
            names.append(f"{name}_{value}")
        columns.append((~np.isin(values, domain)).astype(float))
        names.append(f"{name}_other")
    for name, buckets in HASH_BUCKETS.items():
        values = eligible[name].to_numpy()
        if not (values == np.floor(values)).all():
            raise ValueError("categorical ACS codes must be integers")
        index = values.astype(np.int64) % buckets
        for bucket in range(buckets):
            columns.append((index == bucket).astype(float))
            names.append(f"{name}_bucket_{bucket}")
    X = np.column_stack(columns).astype(np.float32)
    y = (eligible.PINCP.to_numpy() > 50000).astype(np.float32)
    groups = eligible.SERIALNO.astype(str).to_numpy(dtype=str)
    metadata = {"task": "ACSIncome", "year": 2018, "threshold": 50000,
                "representation": "public bounded numeric + onehot + fixed modulo categorical hashing",
                "feature_names": names, "occupation_columns": [i for i, n in enumerate(names)
                                                                 if n.startswith("OCCP_")],
                "eligible_rows": len(y), "households": len(np.unique(groups)),
                "weighted_population_estimate": False}
    return X, y, groups, metadata


def prepare(state: str, cache: Path) -> Path:
    output = cache / f"acs2018_{state}.npz"
    if output.exists():
        return output
    archive_path = download(state, cache)
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(n for n in archive.namelist() if n.lower().endswith(".csv"))
        frames = []
        for member in members:
            with archive.open(member) as handle:
                frames.append(pd.read_csv(handle, usecols=list(COLUMNS), dtype={"SERIALNO": str}))
    X, y, groups, metadata = prepare_frame(pd.concat(frames, ignore_index=True))
    metadata.update(state=state, archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                    source_url=f"https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_p{state.lower()}.zip")
    np.savez_compressed(output, X=X, y=y, groups=groups)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def sample_disjoint_households(X, y, groups, sizes: tuple[int, ...], seed: int):
    """Randomly assign whole households; truncate rows only within assigned blocks."""
    rng = np.random.default_rng(seed)
    households, codes = np.unique(groups, return_inverse=True)
    order = rng.permutation(len(households))
    sorted_rows = np.argsort(codes, kind="stable")
    boundaries = np.concatenate([[0], np.cumsum(np.bincount(codes))])
    cohorts, cursor = [], 0
    for size in sizes:
        chunks, count = [], 0
        while count < size:
            if cursor == len(order):
                raise ValueError("not enough disjoint household rows")
            group = order[cursor]
            chunk = sorted_rows[boundaries[group]:boundaries[group + 1]]
            chunks.append(chunk)
            count += len(chunk)
            cursor += 1
        indices = np.concatenate(chunks)[:size]
        cohorts.append((X[indices], y[indices], groups[indices]))
    return cohorts
