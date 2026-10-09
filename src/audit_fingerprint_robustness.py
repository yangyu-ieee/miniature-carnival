"""Robustness and confounding audit for the frozen fingerprint pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv"
PRED = ROOT / "results" / "dishbrain_fingerprint_pilot" / "grouped_oof_predictions.csv"
OUT = ROOT / "results" / "dishbrain_fingerprint_robustness"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "rest_culture_mean_firing_rates", "rest_culture_var_firing_rates",
    "rest_culture_max_firing_rates", "rest_culture_ISI_mean", "rest_amplitude",
    "rest_DCC", "rest_BR", "rest_small_bursts", "rest_med_bursts",
    "rest_large_bursts", "rest_dist_type_numeric", "rest_burst_rates_numeric",
    "rest_super_bursts",
]


def model() -> Pipeline:
    categorical = ["rest_super_bursts"]
    numeric = [column for column in FEATURES if column not in categorical]
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
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
    return Pipeline(
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


def clustered_bootstrap(frame: pd.DataFrame, repeats: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(20260921)
    chips = frame["chip_id"].astype(str).unique()
    records = []
    for repeat in range(repeats):
        sampled = rng.choice(chips, size=len(chips), replace=True)
        parts = [frame.loc[frame["chip_id"].astype(str).eq(chip)] for chip in sampled]
        boot = pd.concat(parts, ignore_index=True)
        records.append(
            {
                "repeat": repeat,
                "balanced_accuracy": balanced_accuracy_score(boot["truth"], boot["prediction"]),
                "macro_f1": f1_score(boot["truth"], boot["prediction"], average="macro"),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    data = pd.read_csv(DATA)
    data = data.loc[data["drug"].isin(["Carbamazepine", "Perampanel", "Phenytoin"])].reset_index(drop=True)
    predictions = pd.read_csv(PRED, dtype={"chip_id": str})
    best = predictions.loc[
        predictions["feature_set"].eq("full_rest_electrophysiology")
        & predictions["model"].eq("extra_trees")
    ].copy()
    boot = clustered_bootstrap(best)
    boot.to_csv(OUT / "cluster_bootstrap.csv", index=False)

    drug_count = data.groupby("chip_id")["drug"].nunique()
    shared_chips = drug_count.loc[drug_count.eq(3)].index
    shared = data.loc[data["chip_id"].isin(shared_chips)].copy().reset_index(drop=True)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260921)
    shared_pred = cross_val_predict(
        model(), shared[FEATURES], shared["drug"], groups=shared["chip_id"], cv=cv, n_jobs=1
    )
    shared_out = shared[["chip_id", "date", "drug", "dose"]].copy()
    shared_out["prediction"] = shared_pred
    shared_out.to_csv(OUT / "shared_chip_predictions.csv", index=False)

    # Risk diagnostic for causal interpretation: on chips receiving all three
    # compounds, protocol order alone predicts the label well.  This flags a
    # serious identifiability concern; it does not prove a unique causal source.
    shared["date"] = pd.to_datetime(shared["date"])
    shared["protocol_rank"] = shared.groupby("chip_id")["date"].rank(method="dense")
    shared["protocol_rank_normalized"] = shared["protocol_rank"] / shared.groupby("chip_id")[
        "protocol_rank"
    ].transform("max")

    # Describe the observed treatment blocks directly; this is a design audit,
    # not a causal estimate and not an independent validation set.
    order_rows = []
    for chip_id, chip_frame in shared.sort_values(["chip_id", "date"]).groupby("chip_id", sort=True):
        ordered_drugs = chip_frame["drug"].astype(str).tolist()
        blocks: list[tuple[str, int]] = []
        for drug in ordered_drugs:
            if blocks and blocks[-1][0] == drug:
                blocks[-1] = (drug, blocks[-1][1] + 1)
            else:
                blocks.append((drug, 1))
        first_appearance = list(dict.fromkeys(ordered_drugs))
        order_rows.append(
            {
                "chip_id": str(chip_id),
                "n_records": len(ordered_drugs),
                "n_treatment_blocks": len(blocks),
                "first_appearance_order": " > ".join(first_appearance),
                "block_order": " > ".join(drug for drug, _ in blocks),
                "records_per_block": "/".join(str(count) for _, count in blocks),
                "all_three_drugs_present": len(first_appearance) == 3,
                "each_drug_in_one_contiguous_block": len(blocks) == 3
                and len(first_appearance) == 3,
            }
        )
    order_audit = pd.DataFrame(order_rows)
    order_audit.to_csv(OUT / "shared_chip_protocol_block_order.csv", index=False)
    first_order_counts = order_audit["first_appearance_order"].value_counts()
    modal_order = str(first_order_counts.index[0])
    modal_order_n = int(first_order_counts.iloc[0])
    chronology_model = DecisionTreeClassifier(max_depth=3, random_state=20260921)
    chronology_pred = cross_val_predict(
        chronology_model,
        shared[["protocol_rank_normalized"]],
        shared["drug"],
        groups=shared["chip_id"],
        cv=cv,
    )
    pd.DataFrame(
        {
            "chip_id": shared["chip_id"], "truth": shared["drug"],
            "protocol_rank_normalized": shared["protocol_rank_normalized"],
            "prediction": chronology_pred,
        }
    ).to_csv(OUT / "chronology_negative_control.csv", index=False)

    ci = {
        metric: {
            "median": float(boot[metric].median()),
            "ci95_low": float(boot[metric].quantile(0.025)),
            "ci95_high": float(boot[metric].quantile(0.975)),
        }
        for metric in ["balanced_accuracy", "macro_f1"]
    }
    report = {
        "cluster_bootstrap": ci,
        "shared_chip_subset": {
            "rows": int(len(shared)), "chips": int(shared["chip_id"].nunique()),
            "balanced_accuracy": float(balanced_accuracy_score(shared["drug"], shared_pred)),
            "macro_f1": float(f1_score(shared["drug"], shared_pred, average="macro")),
        },
        "chronology_negative_control": {
            "balanced_accuracy": float(balanced_accuracy_score(shared["drug"], chronology_pred)),
            "macro_f1": float(f1_score(shared["drug"], chronology_pred, average="macro")),
            "interpretation": "Protocol order predicts drug identity, so pharmacological causality is not identifiable.",
        },
        "shared_chip_protocol_block_order": {
            "n_shared_chips": int(len(order_audit)),
            "modal_first_appearance_order": modal_order,
            "chips_following_modal_first_appearance_order": modal_order_n,
            "fraction_following_modal_first_appearance_order": float(modal_order_n / len(order_audit)),
            "chips_with_three_contiguous_treatment_blocks": int(
                order_audit["each_drug_in_one_contiguous_block"].sum()
            ),
            "interpretation": (
                "Describes the observed chronological design in the shared-chip subset; "
                "does not identify the unique causal source of label predictability."
            ),
        },
        "frozen_claim": "Cross-chip treatment-condition electrophysiological discrimination.",
        "prohibited_claim": "Causal drug-mechanism identification or clinical efficacy prediction.",
    }
    (OUT / "robustness_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
