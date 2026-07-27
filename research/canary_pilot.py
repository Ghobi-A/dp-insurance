"""Conditional canary pilot for the worst-case neural auditability ladder.

The ladder measures *natural* memorisation of real records. Canary auditing asks
a different, worst-case question: insert crafted records whose inclusion you
control, and invert the attacker's success into an epsilon lower bound
(Steinke, Nasr & Jagielski, NeurIPS 2023). That bound does not depend on natural
memorisation existing in the data at all, which makes it a useful second
instrument -- *if* it has power against this particular target.

This script establishes whether it does, and refuses to go further if it does
not. It runs a **non-private pilot only**. Extending to DP sentinel points is
gated on the pilot clearing:

    attacker accuracy >= 0.70  and  epsilon lower bound > 0

If no construction clears that bar, the pilot reports

    CANARY AUDITOR UNDERPOWERED FOR THIS NEURAL TARGET

and stops. There is no automatic six-point canary ladder: a weak auditor run
across a privacy sweep would produce six uninformative bounds, not evidence.

The target matches the ladder exactly: 128->128->64->1, 400 epochs, Adam at
1e-3, batch 64, no regularisation, same leakage-safe ``high_cost`` preparation.

Run::

    python research/canary_pilot.py --epochs 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectability_noise_ladder import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_UNITS,
    LEARNING_RATE,
    TASK_NAME,
    prepare_seed_arrays,
)

#: Pilot must clear both to justify any DP sentinel run.
MIN_ATTACKER_ACCURACY = 0.70
MIN_EPSILON_LOWER_BOUND = 0.0

UNDERPOWERED = "CANARY AUDITOR UNDERPOWERED FOR THIS NEURAL TARGET"
PILOT_PASSED = "CANARY AUDITOR HAS POWER AGAINST THIS NEURAL TARGET"


def build_ladder_scorer(canary_labels: np.ndarray, epochs: int, seed: int):
    """`train_and_score` callback training the exact ladder architecture.

    Non-private: no Opacus, no clipping, no noise. Scores each canary by its
    negative loss, so a lower loss reads as more member-like.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from dp.attacks import logit_confidence_from_bce

    def per_example_bce(y_true: np.ndarray, prob: np.ndarray) -> np.ndarray:
        prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
        y_true = np.asarray(y_true, dtype=float)
        return -(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob))

    def train_and_score(x_aug, y_aug, canary_features):
        torch.manual_seed(seed)
        x = torch.tensor(np.asarray(x_aug, dtype=np.float32))
        y = torch.tensor(np.asarray(y_aug, dtype=np.float32))
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=min(BATCH_SIZE, len(y)),
            shuffle=True,
            generator=generator,
        )

        layers: list[nn.Module] = []
        in_dim = x.shape[1]
        for units in HIDDEN_UNITS:
            layers += [nn.Linear(in_dim, units), nn.ReLU()]
            in_dim = units
        layers.append(nn.Linear(in_dim, 1))
        net = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.BCEWithLogitsLoss()
        net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(net(xb).squeeze(-1), yb).backward()
                optimizer.step()

        net.eval()
        with torch.no_grad():
            logits = net(
                torch.tensor(np.asarray(canary_features, dtype=np.float32))
            ).squeeze(-1)
            prob = torch.sigmoid(logits).cpu().numpy()
        losses = per_example_bce(canary_labels, prob)
        # Logit confidence is monotone in -loss and numerically better behaved.
        return logit_confidence_from_bce(losses)

    return train_and_score


def run_pilot(
    epochs: int,
    seed: int,
    num_canaries: int,
    duplication: int,
    log,
) -> dict[str, object]:
    """Non-private pilot over the two requested canary constructions."""
    from dp.audit import one_run_model_audit
    from dp.canaries import make_duplicated_canaries, make_label_flip_canaries

    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "insurance.csv")
    arrays = prepare_seed_arrays(df, seed)
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]
    n_features = X_train.shape[1]

    constructions = {
        "label-flip": {
            "canaries": make_label_flip_canaries(
                X_test, y_test, min(num_canaries, len(y_test)), random_state=1
            ),
            "probe": "record-level membership",
        },
        f"duplicated-x{duplication}": {
            "canaries": make_duplicated_canaries(
                num_canaries, n_features, duplication=duplication, random_state=1
            ),
            # Duplication amplifies influence k-fold, so the bound must be read
            # against a group-privacy budget of roughly k*epsilon, not epsilon.
            "probe": f"GROUP-privacy probe (group size {duplication})",
        },
    }

    results = []
    for name, spec in constructions.items():
        canaries = spec["canaries"]
        scorer = build_ladder_scorer(canaries.labels, epochs, seed)
        audit = one_run_model_audit(
            canaries,
            X_train,
            y_train,
            scorer,
            delta=0.0,
            confidence=0.95,
            random_state=seed,
        )
        accuracy = (
            float(audit.num_correct / audit.num_guesses) if audit.num_guesses else 0.0
        )
        passed = (
            accuracy >= MIN_ATTACKER_ACCURACY
            and float(audit.epsilon_lower_bound) > MIN_EPSILON_LOWER_BOUND
        )
        entry = {
            "construction": name,
            "probe": spec["probe"],
            "num_canaries": int(len(canaries.labels)),
            "duplication": int(getattr(canaries, "duplication", 1)),
            "num_guesses": int(audit.num_guesses),
            "num_correct": int(audit.num_correct),
            "attacker_accuracy": accuracy,
            "epsilon_lower_bound": float(audit.epsilon_lower_bound),
            "confidence": float(audit.confidence),
            "passes_pilot": bool(passed),
        }
        results.append(entry)
        log(
            f"[canary {name}] guesses={entry['num_guesses']} "
            f"correct={entry['num_correct']} accuracy={accuracy:.4f} "
            f"eps_lb={entry['epsilon_lower_bound']:.4f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    any_passed = any(r["passes_pilot"] for r in results)
    verdict = PILOT_PASSED if any_passed else UNDERPOWERED
    return {
        "experiment": "canary_pilot_non_private",
        "task": TASK_NAME,
        "private": False,
        "epochs": epochs,
        "architecture": f"Linear(d,{'->'.join(map(str, HIDDEN_UNITS))}->1) + ReLU",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": seed,
        "min_attacker_accuracy": MIN_ATTACKER_ACCURACY,
        "min_epsilon_lower_bound": MIN_EPSILON_LOWER_BOUND,
        "results": results,
        "verdict": verdict,
        "extend_to_dp_sentinels": bool(any_passed),
    }


def render_markdown(payload: dict[str, object]) -> str:
    """Short report; its only job is the go/no-go on canary work."""
    lines = [
        "# Canary auditor pilot (non-private)",
        "",
        "Conditional pilot for the worst-case neural auditability ladder. The target "
        "matches the ladder exactly: "
        f"`{payload['architecture']}`, {payload['epochs']} epochs, Adam at "
        f"{payload['learning_rate']}, batch {payload['batch_size']}, no regularisation, "
        f"same leakage-safe `{payload['task']}` preparation. **Non-private only** -- no "
        "canary ladder is run across privacy points.",
        "",
        "Extension to DP sentinel points requires a construction reaching "
        f"attacker accuracy >= {payload['min_attacker_accuracy']} **and** an epsilon "
        f"lower bound > {payload['min_epsilon_lower_bound']}.",
        "",
        "| Construction | Probe | Canaries | Guesses | Correct | Accuracy | eps lower "
        "bound | Pilot |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for entry in payload["results"]:
        lines.append(
            f"| {entry['construction']} | {entry['probe']} | {entry['num_canaries']} | "
            f"{entry['num_guesses']} | {entry['num_correct']} | "
            f"{entry['attacker_accuracy']:.4f} | {entry['epsilon_lower_bound']:.4f} | "
            f"{'PASS' if entry['passes_pilot'] else 'FAIL'} |"
        )

    lines += ["", f"## Verdict: {payload['verdict']}", ""]
    if payload["extend_to_dp_sentinels"]:
        lines.append(
            "At least one construction has power against this target, so extending the "
            "canary auditor to sentinel DP points is justified. That extension is a "
            "separate, explicitly requested run -- it does not start automatically."
        )
    else:
        lines += [
            f"**{UNDERPOWERED}**",
            "",
            "No construction reached the required attacker accuracy with a positive "
            "epsilon lower bound. The canary experiment stops here: running this "
            "auditor across DP privacy points would produce uninformative bounds at "
            "every point, which is not evidence about privacy.",
            "",
            "A loose bound means *not detected at this power*, never *proven private*.",
        ]
    lines += [
        "",
        "The duplicated construction is a **group-privacy** probe: duplication "
        "amplifies a record's influence k-fold, so its bound must be read against a "
        "budget of roughly k*epsilon, not the single-record epsilon.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--canaries", type=int, default=200)
    parser.add_argument("--duplication", type=int, default=8)
    parser.add_argument("--output-dir", default="reports/canary_pilot")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = (output_dir / "canary_pilot.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        handle.write(message + "\n")
        handle.flush()

    payload = run_pilot(
        epochs=args.epochs,
        seed=args.seed,
        num_canaries=args.canaries,
        duplication=args.duplication,
        log=log,
    )
    (output_dir / "canary_pilot.json").write_text(json.dumps(payload, indent=2, default=float))
    (output_dir / "canary_pilot.md").write_text(render_markdown(payload))
    log(f"\nVerdict: {payload['verdict']}")
    log(f"Extend to DP sentinels: {payload['extend_to_dp_sentinels']}")
    handle.close()


if __name__ == "__main__":
    sys.exit(main())
