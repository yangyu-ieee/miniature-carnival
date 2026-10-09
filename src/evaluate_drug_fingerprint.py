"""Cross-chip classification of anti-seizure drug response fingerprints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, label_binarize


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv"
OUT = ROOT / "results" / "dishbrain_fingerprint_pilot"
OUT.mkdir(parents=True, exist_ok=True)

BASIC = [
    "rest_culture_mean_firing_rates",
    "rest_culture_var_firing_rates",
    "rest_culture_max_firing_rates",
    "rest_culture_ISI_mean",
    "rest_amplitude",
]
NETWORK = [
    "rest_DCC",
    "rest_BR",
    "rest_small_bursts",
    "rest_med_bursts",
    "rest_large_bursts",
    "rest_dist_type_numeric",
    "rest_burst_rates_numeric",
    "rest_super_bursts",
]
FEATURE_SETS = {"basic_firing": BASIC, "full_rest_electrophysiology": BASIC + NETWORK}


def preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in columns if column == "rest_super_bursts"]
    numeric = [column for column in columns if column not in categorical]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", RobustScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def main() -> None:
    data = pd.read_csv(DATA)
    data = data.loc[data["drug"].isin(["Carbamazepine", "Perampanel", "Phenytoin"])].reset_index(drop=True)
    y = data["drug"].to_numpy()
    groups = data["chip_id"].astype(str).to_numpy()
    classes = np.unique(y)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260921)
    records: list[dict] = []
    params: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(data, y, groups), start=1):
        y_train, y_test = y[train_idx], y[test_idx]
        majority = pd.Series(y_train).mode().iloc[0]
        majority_pred = np.repeat(majority, len(test_idx))
        for row_index, truth, pred in zip(test_idx, y_test, majority_pred):
            records.append(
                {
                    "row_index": int(row_index), "chip_id": str(data.loc[row_index, "chip_id"]),
                    "fold": fold, "feature_set": "none", "model": "training_majority",
                    "truth": truth, "prediction": pred,
                    **{f"p_{label}": float(label == pred) for label in classes},
                }
            )

        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260921 + fold)
        for feature_name, columns in FEATURE_SETS.items():
            x_train, x_test = data.loc[train_idx, columns], data.loc[test_idx, columns]
            prep = preprocessor(columns)
            logistic = Pipeline(
                [
                    ("prep", prep),
                    ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
                ]
            )
            search = GridSearchCV(
                logistic,
                {"model__C": [0.01, 0.1, 1.0, 10.0]},
                scoring="balanced_accuracy",
                cv=inner,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(x_train, y_train, groups=groups[train_idx])
            pred = search.predict(x_test)
            prob = search.predict_proba(x_test)
            params.append(
                {"fold": fold, "feature_set": feature_name, "model": "logistic", "best_C": search.best_params_["model__C"]}
            )
            for local, row_index in enumerate(test_idx):
                row = {
                    "row_index": int(row_index), "chip_id": str(data.loc[row_index, "chip_id"]),
                    "fold": fold, "feature_set": feature_name, "model": "logistic",
                    "truth": y_test[local], "prediction": pred[local],
                }
                row.update({f"p_{label}": float(prob[local, list(search.classes_).index(label)]) for label in classes})
                records.append(row)

            forest = Pipeline(
                [
                    ("prep", prep),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=1000, min_samples_leaf=3, max_features=0.8,
                            class_weight="balanced", random_state=20260921, n_jobs=1,
                        ),
                    ),
                ]
            )
            forest.fit(x_train, y_train)
            pred = forest.predict(x_test)
            prob = forest.predict_proba(x_test)
            for local, row_index in enumerate(test_idx):
                row = {
                    "row_index": int(row_index), "chip_id": str(data.loc[row_index, "chip_id"]),
                    "fold": fold, "feature_set": feature_name, "model": "extra_trees",
                    "truth": y_test[local], "prediction": pred[local],
                }
                row.update({f"p_{label}": float(prob[local, list(forest.classes_).index(label)]) for label in classes})
                records.append(row)

    pred_frame = pd.DataFrame(records)
    pred_frame.to_csv(OUT / "grouped_oof_predictions.csv", index=False)
    pd.DataFrame(params).to_csv(OUT / "selected_hyperparameters.csv", index=False)

    rows = []
    for (feature_set, model), group in pred_frame.groupby(["feature_set", "model"]):
        truth, pred = group["truth"].to_numpy(), group["prediction"].to_numpy()
        probs = group[[f"p_{label}" for label in classes]].to_numpy()
        try:
            auc = roc_auc_score(label_binarize(truth, classes=classes), probs, average="macro", multi_class="ovr")
        except ValueError:
            auc = np.nan
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "n": int(len(group)),
                "accuracy": float(accuracy_score(truth, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
                "macro_f1": float(f1_score(truth, pred, average="macro")),
                "macro_ovr_auc": float(auc) if np.isfinite(auc) else None,
            }
        )
        pd.DataFrame(confusion_matrix(truth, pred, labels=classes), index=classes, columns=classes).to_csv(
            OUT / f"confusion_{feature_set}_{model}.csv"
        )
    metrics = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    metrics.to_csv(OUT / "grouped_cv_metrics.csv", index=False)
    best = metrics.iloc[0].to_dict()
    summary = {
        "task": "identify three anti-seizure drugs from earliest post-treatment rest electrophysiology",
        "design": "5-fold StratifiedGroupKFold by chip_id; no row-random split",
        "rows": int(len(data)), "chips": int(data["chip_id"].nunique()),
        "class_counts": data["drug"].value_counts().to_dict(),
        "best": best,
        "pilot_pass_rule": "balanced accuracy >= 0.55 and macro-F1 >= 0.50",
        "pilot_pass": bool(best["balanced_accuracy"] >= 0.55 and best["macro_f1"] >= 0.50),
        "warning": "Exploratory endpoint/model selection; requires locked confirmation before headline use.",
    }
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
