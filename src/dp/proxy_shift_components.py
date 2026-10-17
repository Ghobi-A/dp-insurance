"""Separate clipping and added-noise interactions without changing raw pilot evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def decompose(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "strength", "partition", "variant", "environment"]
    clipped = rows[rows.condition == "clipped"][keys + ["auc"]].rename(
        columns={"auc": "clipped_auc"})
    if clipped.duplicated(keys).any():
        raise ValueError("duplicate clipping control")
    result = rows[rows.condition == "dpsgd"].merge(
        clipped, on=keys, validate="many_to_one", how="left")
    if result.clipped_auc.isna().any():
        raise ValueError("missing clipping control")
    result["clipping_gap"] = result.reference_auc - result.clipped_auc
    result["noise_gap"] = result.clipped_auc - result.auc
    source_keys = ["seed", "strength", "partition", "variant", "requested_epsilon"]
    source = result[result.environment == "source"][
        source_keys + ["clipping_gap", "noise_gap"]].rename(columns={
            "clipping_gap": "source_clipping_gap", "noise_gap": "source_noise_gap"})
    result = result.merge(source, on=source_keys, how="left", validate="many_to_one")
    if result.source_noise_gap.isna().any():
        raise ValueError("missing source environment")
    result["clipping_interaction"] = result.clipping_gap - result.source_clipping_gap
    result["noise_interaction"] = result.noise_gap - result.source_noise_gap
    return result


def analyse(input_dir: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("component output directory must be new or empty")
    input_path = input_dir / "metrics.csv"
    rows = decompose(pd.read_csv(input_path))
    part = rows[(rows.partition == "test") & (rows.variant == "proxy")
                & (rows.environment != "source")]
    summary = part.groupby("environment")[["gap_interaction", "clipping_interaction",
                                           "noise_interaction"]].mean()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "components.csv", index=False)
    summary.to_csv(output_dir / "grid_means.csv")
    manifest = {"input_metrics_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "status": "exploratory_component_analysis",
                "warning": "pooled grid means are descriptive; no significance or equivalence claim"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Clipping and noise decomposition", "",
             "These are pooled exploratory grid means, not inferential effects.", "",
             "| Environment | Total DP gap change | Clipping component | Added-noise component |",
             "|---|---:|---:|---:|"]
    for environment, row in summary.iterrows():
        lines.append(f"| {environment} | {row.gap_interaction:.6f} | "
                     f"{row.clipping_interaction:.6f} | {row.noise_interaction:.6f} |")
    lines += ["", "Total = clipping + added noise under matched sampling/optimisation.",
              "Inspect seed/budget/strength cells before drawing conclusions. A small pooled",
              "noise interaction does not establish equivalence or no effect in every cell.",
              "An interaction dominated by clipping cannot be attributed to Gaussian noise.", ""]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyse(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
