"""Recreate and archive the fixed-seed random portfolio comparison.

This audit does not overwrite the frozen portfolio results. It replays the
same 5,000 random selections per budget and checks the saved percentile values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make `src.*` imports stable when invoked as `python src/script.py`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate_experiment_portfolio import objective, prepare_candidates


RESULTS = ROOT / "results" / "experiment_portfolio"
OUT = ROOT / "results" / "portfolio_random_audit_20260927"
SEED = 20260921
REPEATS = 5000
BUDGETS = [5, 8, 12]


def sha256(path: Path) -> str:
    """Return a stable digest for each input used by this replay."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, uncertainty, distance = prepare_candidates()
    frozen = pd.read_csv(RESULTS / "portfolio_benchmark.csv")
    rng = np.random.default_rng(SEED)
    draw_rows: list[dict] = []
    summary_rows: list[dict] = []

    for budget in BUDGETS:
        # Keep the original experiment's single RNG stream and budget order.
        draws = []
        for draw_id in range(REPEATS):
            selected = rng.choice(len(candidates), size=budget, replace=False).tolist()
            metrics = objective(selected, uncertainty, distance, weight=0.55)
            draws.append(metrics["objective"])
            draw_rows.append(
                {
                    "budget": budget,
                    "draw_id": draw_id,
                    "selected_indices_json": json.dumps(selected),
                    **metrics,
                }
            )

        values = np.asarray(draws, dtype=float)
        greedy_score = float(
            frozen.loc[
                frozen["budget"].eq(budget) & frozen["method"].eq("greedy_hybrid"),
                "objective",
            ].iloc[0]
        )
        saved_percentile = float(
            frozen.loc[
                frozen["budget"].eq(budget) & frozen["method"].eq("greedy_hybrid"),
                "random_percentile",
            ].iloc[0]
        )
        replay_percentile = float(np.mean(values <= greedy_score))
        if not np.isclose(replay_percentile, saved_percentile, atol=1e-12, rtol=0):
            raise AssertionError(
                f"budget={budget}: replay percentile {replay_percentile} "
                f"does not match frozen value {saved_percentile}"
            )

        summary_rows.append(
            {
                "budget": budget,
                "random_repeats": REPEATS,
                "random_mean": float(values.mean()),
                "random_sd_population": float(values.std(ddof=0)),
                "random_best_observed": float(values.max()),
                "greedy_objective": greedy_score,
                "greedy_minus_random_best_observed": float(greedy_score - values.max()),
                "greedy_random_percentile_leq": replay_percentile,
                "greedy_at_or_above_all_random_draws": bool(greedy_score + 1e-12 >= values.max()),
            }
        )

    pd.DataFrame(draw_rows).to_csv(OUT / "random_portfolio_draws.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "random_comparison_summary.csv", index=False)

    inputs = [
        ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv",
        ROOT / "results" / "dishbrain_fingerprint_pilot" / "grouped_oof_predictions.csv",
        RESULTS / "portfolio_benchmark.csv",
        ROOT / "src" / "evaluate_experiment_portfolio.py",
    ]
    audit = {
        "purpose": "deterministic replay of the 5,000-draw random baseline; not a global optimality proof",
        "seed": SEED,
        "random_repeats_per_budget": REPEATS,
        "budget_order": BUDGETS,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "replay_percentiles_match_frozen_results": True,
        "summary": summary_rows,
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Saved raw draws and audit manifest under: {OUT}")


if __name__ == "__main__":
    main()
