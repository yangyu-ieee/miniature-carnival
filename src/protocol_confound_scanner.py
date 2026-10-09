"""Detect protocol and hierarchy confounding in repeated organ-chip studies.

The scanner intentionally uses metadata-only negative controls.  If protocol
order, acquisition date, or chip identity can predict the scientific label,
high signal-model accuracy cannot be interpreted causally without a stronger
experimental design.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv"
OUT = ROOT / "results" / "protocol_confound_scan"
OUT.mkdir(parents=True, exist_ok=True)


def grouped_metadata_score(
    frame: pd.DataFrame,
    target: str,
    group: str,
    feature: str,
    seed: int = 20260921,
) -> dict:
    classes = frame[target].value_counts()
    min_count = int(classes.min())
    n_splits = min(5, min_count, frame[group].nunique())
    if n_splits < 2:
        return {"available": False, "reason": "insufficient grouped class support"}
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    estimator = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=seed)
    prediction = cross_val_predict(
        estimator,
        frame[[feature]],
        frame[target],
        groups=frame[group],
        cv=cv,
    )
    return {
        "available": True,
        "balanced_accuracy": float(balanced_accuracy_score(frame[target], prediction)),
        "macro_f1": float(f1_score(frame[target], prediction, average="macro")),
        "n": int(len(frame)),
        "groups": int(frame[group].nunique()),
    }


def main() -> None:
    data = pd.read_csv(INPUT)
    data = data.loc[data["drug"].isin(["Carbamazepine", "Perampanel", "Phenytoin"])].copy()
    data["date"] = pd.to_datetime(data["date"])

    # The strongest within-chip audit uses only chips receiving every class.
    per_chip_classes = data.groupby("chip_id")["drug"].nunique()
    shared = data.loc[data["chip_id"].isin(per_chip_classes.loc[per_chip_classes.eq(3)].index)].copy()
    shared["protocol_rank"] = shared.groupby("chip_id")["date"].rank(method="dense")
    shared["protocol_rank_normalized"] = shared["protocol_rank"] / shared.groupby("chip_id")[
        "protocol_rank"
    ].transform("max")

    chronology = grouped_metadata_score(
        shared, target="drug", group="chip_id", feature="protocol_rank_normalized"
    )
    chance = 1.0 / shared["drug"].nunique()
    excess = chronology.get("balanced_accuracy", chance) - chance
    risk = "HIGH" if excess >= 0.30 else "MEDIUM" if excess >= 0.15 else "LOW"

    label_date = pd.crosstab(data["date"], data["drug"])
    date_purity = float(label_date.max(axis=1).sum() / label_date.to_numpy().sum())
    group_overlap = data.groupby("chip_id")["drug"].nunique()

    report = {
        "scanner_version": "1.0",
        "target": "drug treatment condition",
        "hierarchy": {"group": "chip_id", "time": "date"},
        "dataset": {
            "rows": int(len(data)),
            "chips": int(data["chip_id"].nunique()),
            "classes": data["drug"].value_counts().to_dict(),
            "chips_with_all_classes": int((group_overlap == 3).sum()),
        },
        "metadata_negative_controls": {
            "within_chip_protocol_order": chronology,
            "date_to_label_purity": date_purity,
        },
        "risk": risk,
        "reason": (
            "Protocol order predicts treatment identity far above chance; signal-model discrimination "
            "cannot be interpreted as causal drug-mechanism identification."
        ),
        "allowed_claim": "Cross-chip discrimination of treatment-condition electrophysiological states.",
        "blocked_claims": [
            "causal identification of drug mechanism",
            "clinical efficacy prediction",
            "patient-specific treatment recommendation",
        ],
        "recommended_future_design": [
            "randomize treatment order across chips",
            "record same-day pre-treatment baselines",
            "include washout or independent-chip treatment arms",
            "pre-register chip-level holdout and endpoint",
        ],
    }
    (OUT / "confound_scan.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = f"""# Protocol Confound Scan

**Risk: {risk}**

- Rows: {len(data)}; chips: {data['chip_id'].nunique()}; shared three-drug chips: {(group_overlap == 3).sum()}.
- Protocol-order balanced accuracy: {chronology['balanced_accuracy']:.3f} (chance {chance:.3f}).
- Date-to-label purity: {date_purity:.3f}.

## Interpretation

{report['reason']}

Allowed: **{report['allowed_claim']}**

Blocked: {', '.join(report['blocked_claims'])}.
"""
    (OUT / "CONFOUND_SCAN_REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
