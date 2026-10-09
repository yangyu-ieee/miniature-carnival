"""Pilot evaluation for budget-constrained organoid experiment selection.

This is a claim-screening experiment, not a final competition benchmark.
It compares deterministic selection policies on the authors' 16-dimensional
organoid phenotype table. The held-out reconstruction metric asks how well a
small selected cohort represents the remaining phenotype space.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "organoid_feature_table.csv"
OUT = ROOT / "results" / "selection_pilot"
RNG_SEED = 20260921


def prepare_embedding(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    author_features = [
        "FR", "STTC", "density", "clustering", "sigma", "burst_rate",
        "Gini", "BR", "m_MR_4ms", "tau_size", "alpha_duration",
        "mean_TE", "TE_asym", "NMI", "SampEn", "PE",
    ]
    values = frame[author_features].replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True))
    scaled = RobustScaler().fit_transform(values)
    n_components = min(8, len(author_features), len(frame) - 1)
    embedding = PCA(n_components=n_components, random_state=RNG_SEED).fit_transform(scaled)
    return embedding, author_features


def nearest_to_centers(points: np.ndarray, centers: np.ndarray) -> list[int]:
    distance = pairwise_distances(centers, points)
    selected: list[int] = []
    for row in distance:
        for index in np.argsort(row):
            if int(index) not in selected:
                selected.append(int(index))
                break
    return selected


def kmeans_select(points: np.ndarray, budget: int) -> list[int]:
    model = KMeans(n_clusters=budget, random_state=RNG_SEED, n_init=50).fit(points)
    return nearest_to_centers(points, model.cluster_centers_)


def farthest_first(points: np.ndarray, budget: int) -> list[int]:
    distances = pairwise_distances(points)
    selected = [int(np.argmin(distances.mean(axis=1)))]
    while len(selected) < budget:
        nearest = distances[:, selected].min(axis=1)
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return selected


def objective(points: np.ndarray, selected: list[int]) -> tuple[float, dict[str, float]]:
    distances = pairwise_distances(points, points[selected])
    nearest = distances.min(axis=1)
    coverage = float(nearest.mean())
    worst_case = float(nearest.max())

    full_mean = points.mean(axis=0)
    selected_mean = points[selected].mean(axis=0)
    mean_error = float(np.linalg.norm(selected_mean - full_mean) / np.sqrt(points.shape[1]))

    if len(selected) > 1:
        full_cov = np.cov(points, rowvar=False)
        selected_cov = np.cov(points[selected], rowvar=False)
        covariance_error = float(
            np.linalg.norm(selected_cov - full_cov, ord="fro")
            / max(np.linalg.norm(full_cov, ord="fro"), 1e-12)
        )
    else:
        covariance_error = 1.0

    score = coverage + 0.35 * worst_case + 0.50 * mean_error + 0.35 * covariance_error
    return score, {
        "coverage_mean": coverage,
        "coverage_max": worst_case,
        "mean_error": mean_error,
        "covariance_error": covariance_error,
    }


def phenotype_balanced_select(points: np.ndarray, budget: int) -> list[int]:
    distances = pairwise_distances(points)
    selected = [int(np.argmin(distances.mean(axis=1)))]
    while len(selected) < budget:
        best_candidate = None
        best_score = np.inf
        for candidate in range(len(points)):
            if candidate in selected:
                continue
            score, _ = objective(points, selected + [candidate])
            if score < best_score:
                best_candidate = candidate
                best_score = score
        assert best_candidate is not None
        selected.append(int(best_candidate))
    return selected


def evaluate(points: np.ndarray, selected: list[int]) -> dict[str, float]:
    score, metrics = objective(points, selected)
    selected_set = set(selected)
    held_out = [index for index in range(len(points)) if index not in selected_set]
    distances = pairwise_distances(points[held_out], points[selected])
    nearest_index = np.argmin(distances, axis=1)
    reconstructed = points[np.asarray(selected)[nearest_index]]
    rmse = float(np.sqrt(np.mean((points[held_out] - reconstructed) ** 2)))
    return {"objective": score, "heldout_nearest_rmse": rmse, **metrics}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    points, feature_names = prepare_embedding(frame)
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, object]] = []
    selections: dict[str, list[str]] = {}

    for budget in (5, 8, 10, 15):
        deterministic = {
            "phenotype_balanced": phenotype_balanced_select(points, budget),
            "kmeans_medoid": kmeans_select(points, budget),
            "farthest_first": farthest_first(points, budget),
        }
        for method, selected in deterministic.items():
            rows.append({"budget": budget, "method": method, "repeat": 0, **evaluate(points, selected)})
            selections[f"{method}_k{budget}"] = frame.iloc[selected]["organoid"].tolist()

        for repeat in range(1000):
            selected = rng.choice(len(points), size=budget, replace=False).tolist()
            rows.append({"budget": budget, "method": "random", "repeat": repeat, **evaluate(points, selected)})

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "selection_results.csv", index=False)
    random_summary = (
        results[results["method"] == "random"]
        .groupby("budget")
        .agg({column: ["mean", "std", "median"] for column in ["objective", "heldout_nearest_rmse", "coverage_mean", "coverage_max", "mean_error", "covariance_error"]})
    )
    random_summary.to_csv(OUT / "random_summary.csv")
    (OUT / "selected_organoids.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    compact = results[results["method"] != "random"].copy()
    random_means = results[results["method"] == "random"].groupby("budget").mean(numeric_only=True)
    compact["objective_vs_random_pct"] = compact.apply(
        lambda row: 100 * (random_means.loc[row["budget"], "objective"] - row["objective"])
        / random_means.loc[row["budget"], "objective"], axis=1
    )
    compact["rmse_vs_random_pct"] = compact.apply(
        lambda row: 100 * (random_means.loc[row["budget"], "heldout_nearest_rmse"] - row["heldout_nearest_rmse"])
        / random_means.loc[row["budget"], "heldout_nearest_rmse"], axis=1
    )
    compact.to_csv(OUT / "deterministic_summary.csv", index=False)
    print(compact[["budget", "method", "objective", "heldout_nearest_rmse", "objective_vs_random_pct", "rmse_vs_random_pct"]].to_string(index=False))
    print(f"PCA input features: {feature_names}")


if __name__ == "__main__":
    main()

