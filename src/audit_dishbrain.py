"""Audit and assemble the public DishBrain drug-response dataset.

The output unit is one independent chip-date-treatment experiment.  Only the
first post-treatment rest session is used as a prospective feature snapshot;
subsequent active-game sessions provide functional outcomes.  This prevents
future-session information from leaking into the predictor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dishbrain"
OUT = ROOT / "data" / "processed" / "dishbrain"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENT_KEYS = ["chip_id", "date", "drug", "dose"]
SESSION_KEYS = EXPERIMENT_KEYS + ["session_num", "status"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pickle(name: str) -> pd.DataFrame:
    return pd.read_pickle(RAW / name)


def normalize(df: pd.DataFrame, session_col: str) -> pd.DataFrame:
    df = df.copy()
    df["chip_id"] = df["chip_id"].astype(str).str.zfill(5)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["dose"] = pd.to_numeric(df["dose"], errors="coerce")
    df["session_num"] = pd.to_numeric(df[session_col], errors="coerce")
    df["status"] = df["status"].astype(str)
    return df


def prefixed_features(df: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    keep = SESSION_KEYS + columns
    result = df[keep].copy()
    return result.rename(columns={column: f"{prefix}{column}" for column in columns})


def select_earliest_rest_rows(sessions: pd.DataFrame) -> pd.DataFrame:
    """Select one actual earliest rest-session row per experiment key.

    This deliberately preserves nulls from that row; pandas ``GroupBy.first``
    skips nulls column-wise and can silently borrow later-session values.
    """
    rest = sessions.loc[sessions["status"].eq("rest")].copy()
    rest["session_num"] = pd.to_numeric(rest["session_num"], errors="coerce")
    return rest.sort_values("session_num", kind="stable").drop_duplicates(
        EXPERIMENT_KEYS, keep="first"
    ).copy()


def main() -> None:
    game_path = RAW / "drug_game_stats"
    firing_path = RAW / "drug_testing_firing_stats.feather"
    critical_path = RAW / "criticality_metrics_drug"
    burst_path = RAW / "burst_pattern_class_drug"

    game = normalize(read_pickle("drug_game_stats"), "session_num")
    # The public pickle contains four placeholder rows with no gameplay
    # outcome, duplicating an existing session key.  They are not observations.
    game = game.loc[game["hit_rate"].notna()].copy()
    firing = normalize(pd.read_feather(firing_path), "session")
    critical = normalize(read_pickle("criticality_metrics_drug"), "session_num")
    burst = normalize(read_pickle("burst_pattern_class_drug"), "session_num")

    firing_columns = [
        "culture_mean_firing_rates",
        "culture_var_firing_rates",
        "culture_max_firing_rates",
        "culture_ISI_mean",
        "amplitude",
    ]
    critical_columns = ["DCC", "BR", "Shape Collapse beta", "Shape Collapse error"]
    burst_columns = [
        "small_bursts",
        "med_bursts",
        "large_bursts",
        "dist_type_numeric",
        "burst_rates_numeric",
        "super_bursts",
    ]

    sessions = game.copy()
    sessions = sessions.merge(
        prefixed_features(firing, firing_columns, "rest_"), on=SESSION_KEYS, how="left", validate="one_to_one"
    )
    sessions = sessions.merge(
        prefixed_features(critical, critical_columns, "rest_"), on=SESSION_KEYS, how="left", validate="one_to_one"
    )
    sessions = sessions.merge(
        prefixed_features(burst, burst_columns, "rest_"), on=SESSION_KEYS, how="left", validate="one_to_one"
    )

    # Freeze the earliest available post-treatment rest session before looking
    # at any active-game outcome.
    rest = sessions.loc[sessions["status"].eq("rest")].sort_values(
        "session_num", kind="stable"
    )
    # Select a real row, not column-wise non-null values. This prevents later
    # post-outcome measurements from filling the prospective feature snapshot.
    first_rest = select_earliest_rest_rows(sessions)

    active = game.loc[game["status"].eq("full game")].copy()
    outcome_columns = [
        "hit_rate",
        "mean_rally_hits",
        "max_rally_hits",
        "hit_miss_ratio",
        "mean_rally_hits_including_aces",
        "aces_to_game_ratio",
    ]
    # These values are observable in the initial no-feedback rest block and
    # can be evaluated separately from pure electrophysiology features.
    for column in outcome_columns:
        first_rest[f"rest_game_{column}"] = first_rest[column]
    outcomes = active.groupby(EXPERIMENT_KEYS, as_index=False)[outcome_columns].mean()
    outcomes = outcomes.rename(columns={column: f"active_mean_{column}" for column in outcome_columns})

    # Within-experiment active-minus-rest contrasts are descriptive endpoints,
    # not predictor inputs.
    rest_game = rest.groupby(EXPERIMENT_KEYS, as_index=False)[outcome_columns].mean()
    rest_game = rest_game.rename(columns={column: f"rest_mean_{column}" for column in outcome_columns})
    outcomes = outcomes.merge(rest_game, on=EXPERIMENT_KEYS, how="left", validate="one_to_one")
    for column in outcome_columns:
        outcomes[f"delta_{column}"] = (
            outcomes[f"active_mean_{column}"] - outcomes[f"rest_mean_{column}"]
        )

    feature_columns = [column for column in first_rest if column.startswith("rest_")]
    metadata_columns = [
        "chip_id",
        "date",
        "drug",
        "dose",
        "tag",
        "cell_type",
        "start_condition",
        "session_num",
    ]
    table = first_rest[metadata_columns + feature_columns].merge(
        outcomes, on=EXPERIMENT_KEYS, how="inner", validate="one_to_one"
    )
    table = table.sort_values(EXPERIMENT_KEYS).reset_index(drop=True)
    table.to_csv(OUT / "drug_experiment_table.csv", index=False)
    sessions.to_csv(OUT / "drug_session_join_audit.csv", index=False)

    experiment_counts = (
        table.groupby(["drug", "dose"], dropna=False)
        .agg(experiments=("chip_id", "size"), chips=("chip_id", "nunique"))
        .reset_index()
    )
    experiment_counts.to_csv(OUT / "drug_dose_counts.csv", index=False)

    missing = table.isna().mean().sort_values(ascending=False)
    audit = {
        "source": {
            "paper_doi": "10.1038/s42003-025-08194-6",
            "osf_resource": "https://osf.io/xwf7v/",
            "license_note": "Paper is CC BY-NC-ND 4.0; OSF file metadata/license must be preserved.",
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in [game_path, firing_path, critical_path, burst_path]
            },
        },
        "unit_of_analysis": "one chip-date-drug-dose experiment",
        "prospective_feature_rule": "earliest post-treatment rest session only",
        "outcome_rule": "mean across later full-game sessions",
        "rows": int(len(table)),
        "unique_chips": int(table["chip_id"].nunique()),
        "unique_chip_dates": int(table[["chip_id", "date"]].drop_duplicates().shape[0]),
        "drug_dose_conditions": int(table[["drug", "dose"]].drop_duplicates().shape[0]),
        "all_experiments_have_active_outcome": bool(len(table) == game[EXPERIMENT_KEYS].drop_duplicates().shape[0]),
        "first_rest_precedes_active": bool(
            all(
                group.loc[group["status"].eq("rest"), "session_num"].min()
                < group.loc[group["status"].eq("full game"), "session_num"].min()
                for _, group in game.groupby(EXPERIMENT_KEYS)
            )
        ),
        "feature_columns": feature_columns,
        "target_candidates": [
            "active_mean_hit_rate",
            "active_mean_mean_rally_hits",
            "delta_hit_rate",
            "delta_mean_rally_hits",
        ],
        "high_missingness_columns": {
            column: float(rate) for column, rate in missing.items() if rate > 0.10
        },
        "leakage_guard": "All validation must group by chip_id; row-random splitting is prohibited.",
    }
    (OUT / "dishbrain_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
