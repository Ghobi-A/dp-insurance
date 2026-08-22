"""Frozen ACS/Folktables data slice for Study 2, Phase 0.

Phase 0 asks one question only: does a *naturally* trained (non-private) model
on real ACS data leak membership strongly enough to be worth building a DP
epsilon ladder on top of? A question of that kind is only answerable once, and
only if the data it is asked about cannot drift. Everything the slice depends
on is therefore frozen here as a module constant, and the runner refuses to
proceed unless the materialised slice reproduces the frozen fingerprint:

* task ``ACSIncome`` as defined by Ding et al. (2021) ("Retiring Adult"),
* ACS PUMS survey year 2018, 1-Year horizon, state CA,
* the sensitive attribute is ``SEX``,
* exactly 50,000 eligible rows, drawn with the fixed sampling seed 20260822.

The eligibility filter and the feature/target definitions below are the
Folktables ``ACSIncome`` definitions restated in full rather than imported, so
that the frozen slice does not silently change if an upstream package changes
its defaults. 2018 is a ``RELP`` year -- ``RELSHIPP`` only replaces it from 2019
-- which is one more reason the year is frozen rather than parameterised.

Nothing here trains anything. It downloads, caches, filters, samples, verifies
and reports.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Frozen slice definition
# --------------------------------------------------------------------------- #

TASK = "ACSIncome"
SURVEY_YEAR = 2018
HORIZON = "1-Year"
STATE = "CA"
SENSITIVE_ATTRIBUTE = "SEX"
SAMPLE_ROWS = 50_000
SAMPLING_SEED = 20260822

#: ACSIncome features, in the frozen column order. ``SEX`` is both a model
#: feature (as in the published task) and the sensitive attribute the attack is
#: analysed by; it is not dropped, because dropping it would change the task.
FEATURES = (
    "AGEP",
    "COW",
    "SCHL",
    "MAR",
    "OCCP",
    "POBP",
    "RELP",
    "WKHP",
    "SEX",
    "RAC1P",
)
TARGET_COLUMN = "PINCP"
TARGET_THRESHOLD = 50_000

#: Census PUMS person-file location. State FIPS 06 = California.
PUMS_URL = (
    "https://www2.census.gov/programs-surveys/acs/data/pums/"
    f"{SURVEY_YEAR}/{HORIZON}/csv_p{STATE.lower()}.zip"
)
PUMS_MEMBER = "psam_p06.csv"

DEFAULT_CACHE_DIR = Path(
    os.environ.get("STUDY2_ACS_CACHE", Path.home() / ".cache" / "dp-insurance" / "acs")
)

REQUIRED_COLUMNS = tuple(FEATURES) + (TARGET_COLUMN, "PWGTP")


def eligible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The ACSIncome eligibility filter (Ding et al. 2021).

    Employed-age adults with a positive reported income, positive usual hours
    worked and a valid person weight.
    """
    mask = (
        (frame["AGEP"] > 16)
        & (frame[TARGET_COLUMN] > 100)
        & (frame["WKHP"] > 0)
        & (frame["PWGTP"] >= 1)
    )
    return frame.loc[mask]


def build_slice(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter, drop incomplete rows, sample deterministically, freeze order.

    The sample is drawn with :data:`SAMPLING_SEED` from the eligible rows sorted
    by ``SERIALNO``/index, so the selection depends on the data alone and not on
    the order the CSV happened to arrive in.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"PUMS frame is missing required columns: {', '.join(missing)}")

    frame = eligible_rows(raw)
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS))
    frame = frame.sort_index(kind="stable")
    if len(frame) < SAMPLE_ROWS:
        raise ValueError(
            f"only {len(frame)} eligible rows; the frozen slice needs {SAMPLE_ROWS}"
        )

    rng = np.random.default_rng(SAMPLING_SEED)
    positions = np.sort(rng.choice(len(frame), size=SAMPLE_ROWS, replace=False))
    sampled = frame.iloc[positions]

    out = pd.DataFrame(index=range(SAMPLE_ROWS))
    for column in FEATURES:
        out[column] = sampled[column].to_numpy(dtype=float)
    out["y"] = (sampled[TARGET_COLUMN].to_numpy(dtype=float) > TARGET_THRESHOLD).astype(int)
    out["group"] = np.where(out["SEX"].to_numpy() == 1, "male", "female")
    return out


def slice_fingerprint(frame: pd.DataFrame) -> str:
    """SHA-256 over the slice's contents in its exact row and column order.

    Row order is part of the design: shadow schedules and per-example records
    are keyed by cohort position, so a reordered slice is a different slice even
    when it holds the same rows.
    """
    digest = hashlib.sha256()
    digest.update(",".join(frame.columns).encode())
    values = frame[list(FEATURES) + ["y"]].to_numpy(dtype=float)
    digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(",".join(frame["group"].astype(str)).encode())
    return digest.hexdigest()


def slice_metadata(frame: pd.DataFrame) -> dict[str, object]:
    """Machine-readable description of a materialised slice."""
    groups = frame["group"].astype(str)
    return {
        "task": TASK,
        "survey_year": SURVEY_YEAR,
        "horizon": HORIZON,
        "state": STATE,
        "sensitive_attribute": SENSITIVE_ATTRIBUTE,
        "sampling_seed": SAMPLING_SEED,
        "expected_rows": SAMPLE_ROWS,
        "rows": int(len(frame)),
        "features": list(FEATURES),
        "target": f"{TARGET_COLUMN} > {TARGET_THRESHOLD}",
        "positive_rate": float(frame["y"].mean()),
        "group_counts": {
            group: int((groups == group).sum()) for group in sorted(groups.unique())
        },
        "fingerprint": slice_fingerprint(frame),
        "source_url": PUMS_URL,
    }


def verify_slice(frame: pd.DataFrame, metadata: dict[str, object] | None = None) -> dict[str, object]:
    """Raise unless ``frame`` is the frozen slice.

    Checks the row count, the frozen column set, both sensitive groups being
    present with both label classes, and -- when ``metadata`` from an earlier
    materialisation is supplied -- that the content fingerprint is unchanged.
    """
    if len(frame) != SAMPLE_ROWS:
        raise ValueError(f"slice has {len(frame)} rows; frozen slice is {SAMPLE_ROWS}")
    expected_columns = list(FEATURES) + ["y", "group"]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            "slice columns differ from the frozen definition: "
            f"{list(frame.columns)} != {expected_columns}"
        )
    if frame.isna().any().any():
        raise ValueError("slice contains missing values")
    groups = frame["group"].astype(str)
    if set(groups.unique()) != {"male", "female"}:
        raise ValueError(f"slice sensitive groups are {sorted(groups.unique())}")
    for group in ("male", "female"):
        labels = frame.loc[groups == group, "y"]
        if labels.nunique() != 2:
            raise ValueError(f"group {group!r} does not carry both label classes")

    current = slice_metadata(frame)
    if metadata is not None and str(metadata.get("fingerprint")) != current["fingerprint"]:
        raise ValueError(
            "slice fingerprint does not match the recorded metadata; the frozen "
            "slice has changed and Phase 0 must not be run against it"
        )
    return current


# --------------------------------------------------------------------------- #
# Download / cache / smoke
# --------------------------------------------------------------------------- #


def download_pums(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Fetch (or reuse) the cached CA 2018 1-Year PUMS person archive."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"csv_p{STATE.lower()}_{SURVEY_YEAR}_{HORIZON}.zip"
    if archive.exists() and archive.stat().st_size > 0:
        return archive
    tmp = archive.with_suffix(".zip.partial")
    with urllib.request.urlopen(PUMS_URL) as response, tmp.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    tmp.replace(archive)
    return archive


def read_pums_archive(archive: Path) -> pd.DataFrame:
    """Read the person CSV out of a PUMS zip archive."""
    with zipfile.ZipFile(archive) as bundle:
        names = [name for name in bundle.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"no CSV member in {archive}")
        member = PUMS_MEMBER if PUMS_MEMBER in names else names[0]
        with bundle.open(member) as handle:
            return pd.read_csv(
                io.BytesIO(handle.read()),
                usecols=list(REQUIRED_COLUMNS),
                low_memory=False,
            )


def smoke_slice(rows: int = 4_000, seed: int = SAMPLING_SEED) -> pd.DataFrame:
    """A synthetic ACS-shaped frame for CI.

    Smoke mode never touches the network and is never a Phase 0 result: it
    exists so the workflow can exercise download-free plumbing end to end. Every
    artifact produced from it is stamped ``smoke: true``.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(index=range(rows))
    frame["AGEP"] = rng.integers(17, 90, rows).astype(float)
    frame["COW"] = rng.integers(1, 9, rows).astype(float)
    frame["SCHL"] = rng.integers(1, 25, rows).astype(float)
    frame["MAR"] = rng.integers(1, 6, rows).astype(float)
    frame["OCCP"] = rng.integers(10, 9800, rows).astype(float)
    frame["POBP"] = rng.integers(1, 60, rows).astype(float)
    frame["RELP"] = rng.integers(0, 18, rows).astype(float)
    frame["WKHP"] = rng.integers(1, 80, rows).astype(float)
    frame["SEX"] = rng.integers(1, 3, rows).astype(float)
    frame["RAC1P"] = rng.integers(1, 10, rows).astype(float)
    logits = (
        0.04 * (frame["AGEP"] - 40)
        + 0.10 * (frame["SCHL"] - 12)
        + 0.03 * (frame["WKHP"] - 40)
        + rng.normal(0, 1.0, rows)
    )
    frame["y"] = (logits > 0).astype(int)
    frame["group"] = np.where(frame["SEX"].to_numpy() == 1, "male", "female")
    return frame[list(FEATURES) + ["y", "group"]]


def materialise(
    output: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    smoke: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Produce the slice CSV plus its metadata JSON next to it."""
    if smoke:
        frame = smoke_slice()
        metadata = slice_metadata(frame)
        metadata.update({"smoke": True, "expected_rows": int(len(frame))})
    else:
        raw = read_pums_archive(download_pums(cache_dir))
        frame = build_slice(raw)
        metadata = verify_slice(frame)
        metadata["smoke"] = False

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return frame, metadata


def load_slice(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a materialised slice and its metadata."""
    frame = pd.read_csv(path)
    meta_path = path.with_suffix(".metadata.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return frame, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="download, build and write the frozen slice")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    prepare.add_argument("--smoke", action="store_true")

    verify = sub.add_parser("verify", help="verify a materialised slice against its metadata")
    verify.add_argument("--slice", dest="slice_path", type=Path, required=True)
    verify.add_argument(
        "--smoke",
        action="store_true",
        help="allow a synthetic smoke slice (row count check relaxed to its metadata)",
    )

    args = parser.parse_args(argv)

    if args.command == "prepare":
        _, metadata = materialise(args.output, args.cache_dir, smoke=args.smoke)
        print(json.dumps(metadata, indent=2))
        return 0

    frame, metadata = load_slice(args.slice_path)
    if args.smoke or metadata.get("smoke"):
        expected = int(metadata.get("expected_rows", len(frame)))
        if len(frame) != expected:
            raise SystemExit(f"smoke slice has {len(frame)} rows; expected {expected}")
        print(json.dumps({**metadata, "verified": "smoke"}, indent=2))
        return 0
    verified = verify_slice(frame, metadata or None)
    print(json.dumps({**verified, "smoke": False, "verified": "frozen"}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
