"""Leakage-safe pilot for early neural-MPS functional response prediction.

This is a falsification experiment, not a leaderboard search.  Every outer
fold holds out complete chips.  Hyperparameters are selected only within the
training chips.  The script compares condition-only, electrophysiology-only,
early-rest, and combined inputs against a training-mean baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv"
OUT = ROOT / "results" / "dishbrain_response_pilot"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "active_mean_hit_rate",
    "active_mean_mean_rally_hits",
    "delta_hit_rate",
    "delta_mean_rally_hits",
]
CONDITION = ["drug", "dose", "cell_type"]
ELECTROPHYS = [
    "rest_culture_mean_firing_rates",
    "rest_culture_var_firing_rates",
    "rest_culture_max_firing_rates",
    "rest_culture_ISI_mean",
    "rest_amplitude",
    "rest_DCC",
    "rest_BR",
    "rest_small_bursts",
    "rest_med_bursts",
    "rest_large_bursts",
    "rest_dist_type_numeric",
    "rest_burst_rates_numeric",
    "rest_super_bursts",
]
REST_GAME = [
    "rest_game_hit_rate",
    "rest_game_mean_rally_hits",
    "rest_game_max_rally_hits",
    "rest_game_hit_miss_ratio",
    "rest_game_mean_rally_hits_including_aces",
    "rest_game_aces_to_game_ratio",
]
FEATURE_SETS = {
    "condition_only": CONDITION,
    "electrophysiology_only": ELECTROPHYS,
    "early_rest_only": ELECTROPHYS + REST_GAME,
    "combined": CONDITION + ELECTROPHYS + REST_GAME,
}


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [
        column
        for column in columns
        if column in {"drug", "cell_type", "rest_super_bursts"}
    ]
    numeric = [column for column in columns if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", RobustScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def metric_row(target: str, feature_set: str, model: str, y: np.ndarray, pred: np.ndarray) -> dict:
    rho = spearmanr(y, pred).statistic
    return {
        "target": target,
        "feature_set": feature_set,
        "model": model,
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "spearman": float(rho) if np.isfinite(rho) else None,
        "normalized_mae": float(mean_absolute_error(y, pred) / np.std(y, ddof=1)),
    }


def main() -> None:
    data = pd.read_csv(DATA)
    # Ethanol is a three-experiment, single-chip side condition rather than one
    # of the three anti-seizure medicines in the paper's main question.
    data = data.loc[data["drug"].ne("Ethanol")].reset_index(drop=True)
    groups = data["chip_id"].astype(str).to_numpy()
    outer = GroupKFold(n_splits=5)

    prediction_rows: list[dict] = []
    selected_rows: list[dict] = []

    for target in TARGETS:
        y = data[target].to_numpy(dtype=float)
        for fold, (train_idx, test_idx) in enumerate(outer.split(data, y, groups), start=1):
            y_train, y_test = y[train_idx], y[test_idx]
            baseline = np.repeat(np.mean(y_train), len(test_idx))
            for row_index, truth, pred in zip(test_idx, y_test, baseline):
                prediction_rows.append(
                    {
                        "row_index": int(row_index),
                        "chip_id": str(data.loc[row_index, "chip_id"]),
                        "target": target,
                        "feature_set": "none",
                        "model": "training_mean",
                        "fold": fold,
                        "truth": float(truth),
                        "prediction": float(pred),
                    }
                )

            train_groups = groups[train_idx]
            inner_splits = min(4, len(np.unique(train_groups)))
            inner = GroupKFold(n_splits=inner_splits)

            for feature_name, columns in FEATURE_SETS.items():
                x_train = data.loc[train_idx, columns]
                x_test = data.loc[test_idx, columns]
                preprocessor = build_preprocessor(columns)

                ridge = Pipeline([("prep", preprocessor), ("model", Ridge())])
                ridge_search = GridSearchCV(
                    ridge,
                    {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
                    scoring="neg_mean_absolute_error",
                    cv=inner,
                    n_jobs=1,
                )
                ridge_search.fit(x_train, y_train, groups=train_groups)
                ridge_pred = ridge_search.predict(x_test)
                selected_rows.append(
                    {
                        "target": target,
                        "feature_set": feature_name,
                        "model": "ridge",
                        "fold": fold,
                        "best_params": json.dumps(ridge_search.best_params_, sort_keys=True),
                    }
                )
                for row_index, truth, pred in zip(test_idx, y_test, ridge_pred):
                    prediction_rows.append(
                        {
                            "row_index": int(row_index),
                            "chip_id": str(data.loc[row_index, "chip_id"]),
                            "target": target,
                            "feature_set": feature_name,
                            "model": "ridge",
                            "fold": fold,
                            "truth": float(truth),
                            "prediction": float(pred),
                        }
                    )

                forest = Pipeline(
                    [
                        ("prep", preprocessor),
                        (
                            "model",
                            ExtraTreesRegressor(
                                n_estimators=500,
                                min_samples_leaf=3,
                                max_features=0.8,
                                random_state=20260921,
                                n_jobs=1,
                            ),
                        ),
                    ]
                )
                forest.fit(x_train, y_train)
                forest_pred = forest.predict(x_test)
                for row_index, truth, pred in zip(test_idx, y_test, forest_pred):
                    prediction_rows.append(
                        {
                            "row_index": int(row_index),
                            "chip_id": str(data.loc[row_index, "chip_id"]),
                            "target": target,
                            "feature_set": feature_name,
                            "model": "extra_trees",
                            "fold": fold,
                            "truth": float(truth),
                            "prediction": float(pred),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(OUT / "grouped_oof_predictions.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / "ridge_selected_hyperparameters.csv", index=False)

    metrics = []
    for (target, feature_set, model), group in predictions.groupby(
        ["target", "feature_set", "model"], sort=True
    ):
        metrics.append(
            metric_row(
                target,
                feature_set,
                model,
                group["truth"].to_numpy(),
                group["prediction"].to_numpy(),
            )
        )
    metrics_frame = pd.DataFrame(metrics).sort_values(["target", "mae"])
    metrics_frame.to_csv(OUT / "grouped_cv_metrics.csv", index=False)

    best = (
        metrics_frame.loc[metrics_frame["model"].ne("training_mean")]
        .sort_values(["target", "mae"])
        .groupby("target", as_index=False)
        .first()
    )
    baseline = metrics_frame.loc[metrics_frame["model"].eq("training_mean"), ["target", "mae"]]
    baseline = baseline.rename(columns={"mae": "baseline_mae"})
    verdict = best.merge(baseline, on="target", how="left")
    verdict["mae_improvement_pct"] = 100 * (verdict["baseline_mae"] - verdict["mae"]) / verdict[
        "baseline_mae"
    ]
    verdict["pilot_pass"] = (verdict["mae_improvement_pct"] >= 10) & (verdict["spearman"] >= 0.30)
    verdict.to_csv(OUT / "pilot_verdict.csv", index=False)

    summary = {
        "design": "5-fold outer GroupKFold by chip_id; nested grouped tuning for ridge",
        "rows": int(len(data)),
        "chips": int(data["chip_id"].nunique()),
        "excluded": "three Ethanol experiments from one chip",
        "pass_rule": "at least 10% MAE improvement over training-mean and Spearman >= 0.30",
        "verdicts": verdict.to_dict(orient="records"),
        "warning": "Exploratory model/target selection; winning endpoint requires a locked confirmation split or second dataset.",
    }
    (OUT / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(verdict.to_string(index=False))


if __name__ == "__main__":
    main()
