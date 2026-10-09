"""Exhaustively audit the one-exchange neighborhoods of frozen portfolios.

This is a local-optimality check for the existing uncertainty-diversity
objective. It neither changes the production greedy selector nor proves a
global optimum, prospective wet-lab utility, or quantum advantage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

try:
    # Supports `python -m src.audit_portfolio_one_swap` from the repository root.
    from src.evaluate_experiment_portfolio import greedy, objective, prepare_candidates
except ModuleNotFoundError as error:
    if error.name != "src":
        raise
    # Also supports the direct-script command documented for Windows users.
    from evaluate_experiment_portfolio import greedy, objective, prepare_candidates


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "portfolio_one_swap_audit_20260930"
WEIGHT = 0.55
BUDGETS = (5, 8, 12)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_selection(
    selected: list[int], uncertainty, distance, weight: float
) -> dict:
    """Return exhaustive one-for-one neighborhood metrics for one bank."""
    budget = len(selected)
    selected_set = set(selected)
    base = objective(selected, uncertainty, distance, weight)["objective"]
    neighbor_scores: list[tuple[float, int, int]] = []
    for removed in selected:
        for added in range(len(uncertainty)):
            if added in selected_set:
                continue
            trial = [added if index == removed else index for index in selected]
            score = objective(trial, uncertainty, distance, weight)["objective"]
            neighbor_scores.append((score, removed, added))

    best_score, removed, added = max(neighbor_scores, key=lambda item: item[0])
    improving = [score for score, _, _ in neighbor_scores if score > base + 1e-12]
    return {
        "budget": budget,
        "candidate_count": len(uncertainty),
        "neighbor_count_expected": budget * (len(uncertainty) - budget),
        "neighbor_count_evaluated": len(neighbor_scores),
        "greedy_objective": base,
        "best_neighbor_objective": best_score,
        "best_neighbor_minus_greedy": best_score - base,
        "strictly_improving_neighbors_at_1e-12": len(improving),
        "is_one_swap_local_optimum": not improving,
        "best_neighbor_remove_index": removed,
        "best_neighbor_add_index": added,
    }


def main() -> None:
    candidate, uncertainty, distance = prepare_candidates()
    n = len(uncertainty)
    rows: list[dict] = []
    audit: dict[str, dict] = {}

    for budget in BUDGETS:
        selected = greedy(budget, uncertainty, distance, WEIGHT)
        result = audit_selection(selected, uncertainty, distance, WEIGHT)
        rows.append(result)
        audit[str(budget)] = {
            "is_one_swap_local_optimum": result["is_one_swap_local_optimum"],
            "neighbors_evaluated": result["neighbor_count_evaluated"],
            "best_neighbor_minus_greedy": result["best_neighbor_minus_greedy"],
            "strictly_improving_neighbors": result[
                "strictly_improving_neighbors_at_1e-12"
            ],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "one_swap_neighborhood.csv", index=False)
    input_paths = [
        ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv",
        ROOT / "results" / "dishbrain_fingerprint_pilot" / "grouped_oof_predictions.csv",
    ]
    manifest = {
        "purpose": "exhaustive one-exchange neighborhood audit of frozen greedy selections",
        "candidate_count": n,
        "uncertainty_weight": WEIGHT,
        "diversity_weight": 1 - WEIGHT,
        "budgets": list(BUDGETS),
        "improvement_tolerance": 1e-12,
        "result_by_budget": audit,
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in input_paths},
        "claim_boundary": (
            "one-swap local optimality only; not a global-optimum proof, prospective "
            "wet-lab utility result, or quantum-advantage claim"
        ),
    }
    (OUT / "one_swap_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
