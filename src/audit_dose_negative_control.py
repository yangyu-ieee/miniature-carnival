"""剂量元数据负对照：检查 dose 单变量能否预测三药标签。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "dishbrain" / "drug_experiment_table.csv"
OUT = ROOT / "results" / "dishbrain_dose_negative_control"
CLASSES = ["Carbamazepine", "Perampanel", "Phenytoin"]
SEED = 20260921


def sha256(path: Path) -> str:
    """返回文件 SHA-256，记录本次审计所用输入和代码版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(frame: pd.DataFrame, name: str) -> tuple[dict, pd.DataFrame]:
    """只用一列剂量元数据，按物理芯片分组生成 OOF 预测。"""
    frame = frame.reset_index(drop=True)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    prediction = cross_val_predict(
        model,
        frame[["dose"]],
        frame["drug"],
        groups=frame["chip_id"].astype(str),
        cv=cv,
        n_jobs=1,
    )
    output = frame[["chip_id", "date", "drug", "dose"]].copy()
    output["prediction"] = prediction
    summary = {
        "scope": name,
        "records": int(len(frame)),
        "chips": int(frame["chip_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in frame["drug"].value_counts().items()},
        "model": "DecisionTreeClassifier(max_depth=3, random_state=20260921)",
        "feature": "dose only; no electrophysiology, date, or protocol-rank inputs",
        "validation": "5-fold StratifiedGroupKFold by chip_id, shuffle=True, random_state=20260921",
        "balanced_accuracy": float(balanced_accuracy_score(frame["drug"], prediction)),
        "macro_f1": float(f1_score(frame["drug"], prediction, average="macro")),
        "confusion_matrix_labels_order": CLASSES,
        "confusion_matrix": confusion_matrix(frame["drug"], prediction, labels=CLASSES).tolist(),
        "dose_values_by_class": {
            str(label): sorted(float(x) for x in frame.loc[frame["drug"].eq(label), "dose"].unique())
            for label in CLASSES
        },
    }
    output.insert(0, "scope", name)
    return summary, output


def main() -> None:
    """复现全队列及三药共享芯片子集的单变量剂量风险审计。"""
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    data = data.loc[data["drug"].isin(CLASSES)].reset_index(drop=True)

    all_summary, all_predictions = evaluate(data, "all_three_drug_records")
    per_chip_count = data.groupby("chip_id")["drug"].nunique()
    shared_chips = per_chip_count.loc[per_chip_count.eq(3)].index
    shared = data.loc[data["chip_id"].isin(shared_chips)].copy()
    shared_summary, shared_predictions = evaluate(shared, "three_drug_shared_chip_subset")

    predictions = pd.concat([all_predictions, shared_predictions], ignore_index=True)
    predictions.to_csv(OUT / "dose_negative_control_predictions.csv", index=False)
    summary = {
        "purpose": "post-hoc metadata-confounding diagnostic; not a confirmatory or causal analysis",
        "input_sha256": sha256(DATA),
        "script_sha256": sha256(Path(__file__)),
        "results": [all_summary, shared_summary],
        "interpretation": (
            "Dose metadata alone predicts treatment labels under chip-grouped CV. "
            "This shows that dose protocol is a strong label-associated signal; it does not "
            "identify a unique causal source or prove the electrophysiology model uses dose."
        ),
        "claim_boundary": (
            "Do not describe electrophysiology classification as compound-specific pharmacology; "
            "report as exploratory treatment-condition discrimination with dose/order confounding."
        ),
    }
    (OUT / "dose_negative_control_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
