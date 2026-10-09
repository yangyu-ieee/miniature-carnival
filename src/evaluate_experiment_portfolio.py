"""Benchmark budget-constrained neural-MPS follow-up experiment portfolios.

Candidates have already completed a low-cost rest recording.  The planner
selects exactly k candidates for a more expensive closed-loop functional assay,
balancing model uncertainty and phenotype diversity.  The same objective is
exported as a QUBO, but quantum hardware is not required for this benchmark.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv"
PRED = ROOT / "results" / "dishbrain_fingerprint_pilot" / "grouped_oof_predictions.csv"
OUT = ROOT / "results" / "experiment_portfolio"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["Carbamazepine", "Perampanel", "Phenytoin"]
FEATURES = [
    "rest_culture_mean_firing_rates", "rest_culture_var_firing_rates",
    "rest_culture_max_firing_rates", "rest_culture_ISI_mean", "rest_amplitude",
    "rest_DCC", "rest_BR", "rest_small_bursts", "rest_med_bursts",
    "rest_large_bursts", "rest_dist_type_numeric", "rest_burst_rates_numeric",
]


def prepare_candidates() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = pd.read_csv(DATA)
    data = data.loc[data["drug"].isin(CLASSES)].reset_index(drop=True)
    pred = pd.read_csv(PRED)
    pred = pred.loc[
        pred["feature_set"].eq("full_rest_electrophysiology")
        & pred["model"].eq("extra_trees")
    ].sort_values("row_index")
    if not np.array_equal(pred["row_index"].to_numpy(), np.arange(len(data))):
        raise RuntimeError("OOF predictions are not aligned with candidate rows")
    probability = pred[[f"p_{label}" for label in CLASSES]].to_numpy()
    uncertainty = -(probability * np.log(probability + 1e-12)).sum(axis=1) / np.log(len(CLASSES))
    numeric = SimpleImputer(strategy="median").fit_transform(data[FEATURES])
    numeric = RobustScaler().fit_transform(numeric)
    pca = PCA(n_components=5, random_state=20260921).fit_transform(numeric)
    embedding = np.column_stack([probability, pca])
    distance = squareform(pdist(embedding, metric="euclidean"))
    maximum = distance.max()
    if maximum > 0:
        distance /= maximum
    candidate = data[["chip_id", "date", "drug", "dose", "cell_type"]].copy()
    candidate["uncertainty"] = uncertainty
    return candidate, uncertainty, distance


def objective(selected: list[int], uncertainty: np.ndarray, distance: np.ndarray, weight: float) -> dict:
    selected = list(selected)
    k = len(selected)
    uncertainty_term = float(np.mean(uncertainty[selected]))
    if k > 1:
        pair = distance[np.ix_(selected, selected)]
        diversity_term = float(pair[np.triu_indices(k, 1)].mean())
    else:
        diversity_term = 0.0
    return {
        "objective": weight * uncertainty_term + (1 - weight) * diversity_term,
        "mean_uncertainty": uncertainty_term,
        "mean_pairwise_diversity": diversity_term,
    }


def greedy(k: int, uncertainty: np.ndarray, distance: np.ndarray, weight: float) -> list[int]:
    selected: list[int] = []
    remaining = set(range(len(uncertainty)))
    while len(selected) < k:
        best = max(
            remaining,
            key=lambda index: objective(selected + [index], uncertainty, distance, weight)["objective"],
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def kmeans_medoids(k: int, uncertainty: np.ndarray, distance: np.ndarray) -> list[int]:
    # Classical representative-selection baseline based on the same distance
    # geometry, intentionally ignoring uncertainty.
    coordinates = PCA(n_components=min(8, len(uncertainty) - 1), random_state=20260921).fit_transform(distance)
    labels = KMeans(n_clusters=k, n_init=50, random_state=20260921).fit_predict(coordinates)
    selected = []
    for label in range(k):
        members = np.flatnonzero(labels == label)
        within = distance[np.ix_(members, members)].mean(axis=1)
        selected.append(int(members[np.argmin(within)]))
    return selected


def exact_milp(k: int, uncertainty: np.ndarray, distance: np.ndarray, weight: float) -> tuple[list[int], dict]:
    n = len(uncertainty)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(pairs)
    c = np.zeros(n + m)
    c[:n] = -weight * uncertainty / k
    pair_scale = (1 - weight) * 2 / (k * (k - 1))
    c[n:] = [-pair_scale * distance[i, j] for i, j in pairs]

    # One cardinality equality plus three standard product-linearization
    # inequalities for every y_ij = x_i x_j.
    matrix = lil_matrix((1 + 3 * m, n + m), dtype=float)
    lower = np.full(1 + 3 * m, -np.inf)
    upper = np.zeros(1 + 3 * m)
    matrix[0, :n] = 1
    lower[0] = upper[0] = k
    row = 1
    for pair_index, (i, j) in enumerate(pairs):
        y = n + pair_index
        matrix[row, y], matrix[row, i] = 1, -1  # y <= x_i
        row += 1
        matrix[row, y], matrix[row, j] = 1, -1  # y <= x_j
        row += 1
        matrix[row, i], matrix[row, j], matrix[row, y] = 1, 1, -1  # x_i+x_j-y <= 1
        upper[row] = 1
        row += 1
    start = time.perf_counter()
    result = milp(
        c,
        integrality=np.ones(n + m),
        bounds=Bounds(np.zeros(n + m), np.ones(n + m)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 10, "mip_rel_gap": 1e-9},
    )
    selected = np.flatnonzero(result.x[:n] > 0.5).tolist() if result.x is not None else []
    metadata = {
        "success": bool(result.success), "status": int(result.status),
        "message": str(result.message), "seconds": time.perf_counter() - start,
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
    }
    return selected, metadata


def qubo_matrix(k: int, uncertainty: np.ndarray, distance: np.ndarray, weight: float, penalty: float = 2.0) -> np.ndarray:
    """Return upper-triangular Q where x'Qx equals the objective energy (minus a constant)."""
    n = len(uncertainty)
    q = np.zeros((n, n), dtype=float)
    q[np.diag_indices(n)] = -weight * uncertainty / k + penalty * (1 - 2 * k)
    pair_scale = (1 - weight) * 2 / (k * (k - 1))
    for i in range(n):
        for j in range(i + 1, n):
            q[i, j] = 2 * penalty - pair_scale * distance[i, j]
    return q


def main() -> None:
    candidate, uncertainty, distance = prepare_candidates()
    weight = 0.55
    rng = np.random.default_rng(20260921)
    rows, selections, exact_meta = [], {}, {}
    for k in [5, 8, 12]:
        methods = {
            "uncertainty_only": np.argsort(-uncertainty)[:k].tolist(),
            "kmeans_medoids": kmeans_medoids(k, uncertainty, distance),
            "greedy_hybrid": greedy(k, uncertainty, distance, weight),
        }
        exact, metadata = exact_milp(k, uncertainty, distance, weight)
        exact_meta[str(k)] = metadata
        # Never label a time-limited incumbent as an exact optimum.  The raw
        # incumbent and MIP gap remain archived in the solver metadata.
        if metadata["success"] and len(exact) == k:
            methods["exact_milp"] = exact
        random_values = []
        for _ in range(5000):
            selected = rng.choice(len(candidate), size=k, replace=False).tolist()
            random_values.append(objective(selected, uncertainty, distance, weight)["objective"])
        random_values = np.asarray(random_values)
        for method, selected in methods.items():
            metrics = objective(selected, uncertainty, distance, weight)
            rows.append(
                {
                    "budget": k, "method": method, **metrics,
                    "random_percentile": float((random_values <= metrics["objective"]).mean()),
                }
            )
            selections[f"k{k}_{method}"] = selected
        q = qubo_matrix(k, uncertainty, distance, weight)
        np.savez_compressed(OUT / f"portfolio_qubo_k{k}.npz", Q=q, penalty=2.0, budget=k, weight=weight)

    summary = pd.DataFrame(rows).sort_values(["budget", "objective"], ascending=[True, False])
    summary.to_csv(OUT / "portfolio_benchmark.csv", index=False)
    selection_rows = []
    for name, indices in selections.items():
        for rank, index in enumerate(indices, start=1):
            selection_rows.append({"selection": name, "rank": rank, "candidate_index": index, **candidate.loc[index].to_dict()})
    pd.DataFrame(selection_rows).to_csv(OUT / "selected_candidates.csv", index=False)
    report = {
        "candidate_count": int(len(candidate)), "weight_uncertainty": weight,
        "weight_diversity": 1 - weight, "random_repeats": 5000,
        "exact_milp": exact_meta,
        "qubo_convention": "upper-triangular polynomial coefficients; energy = x.T @ Q @ x",
        "claim_boundary": "retrospective workflow benchmark, not prospective wet-lab cost reduction",
    }
    (OUT / "portfolio_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(exact_meta, indent=2))


if __name__ == "__main__":
    main()
