"""Leakage-safe pilot for detecting common MEA recording-quality failures.

Synthetic corruptions are applied to real organoid spike trains. They emulate
failure signatures described in public NeuroPlatform best practices: inactive
channels, excessive/noisy firing, suspiciously duplicated cross-channel events,
and coarse timestamp artefacts. Results establish software sensitivity only;
they are not prevalence estimates for real experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Indiv_organoids"
OUT = ROOT / "results" / "qc_pilot"
SEED = 20260921
WINDOW_S = 60.0
BIN_S = 0.1
CLASSES = ("clean", "dropout", "noise", "duplicate", "quantized")


def decode_u16(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1)
    return "".join(chr(int(value)) for value in values if value)


def load_recording(path: Path) -> tuple[str, float, list[np.ndarray]]:
    with h5py.File(path, "r") as handle:
        name = decode_u16(handle["organoid_id"])
        duration = float(np.asarray(handle["metadata/recording_duration"]).reshape(-1)[0])
        refs = handle["spike_times"]
        channels = [np.asarray(handle[refs[index, 0]]).reshape(-1).astype(float) for index in range(8)]
    return name, duration, channels


def slice_window(channels: list[np.ndarray], start: float, stop: float) -> list[np.ndarray]:
    return [values[(values >= start) & (values < stop)] - start for values in channels]


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.all(values == 0):
        return 0.0
    sorted_values = np.sort(values)
    n = len(values)
    return float((2 * np.arange(1, n + 1) - n - 1) @ sorted_values / (n * sorted_values.sum()))


def features(channels: list[np.ndarray], duration: float = WINDOW_S) -> dict[str, float]:
    rates = np.asarray([len(values) / duration for values in channels], dtype=float)
    edges = np.arange(0, duration + BIN_S, BIN_S)
    counts = np.vstack([np.histogram(values, edges)[0] for values in channels])
    active_per_bin = (counts > 0).sum(axis=0)

    correlations: list[float] = []
    for left in range(len(channels)):
        for right in range(left + 1, len(channels)):
            if counts[left].std() > 0 and counts[right].std() > 0:
                correlations.append(float(np.corrcoef(counts[left], counts[right])[0, 1]))
    correlations_arr = np.asarray(correlations or [0.0])

    isi_cvs = []
    for values in channels:
        isi = np.diff(values)
        if len(isi) >= 5 and isi.mean() > 0:
            isi_cvs.append(float(isi.std() / isi.mean()))

    ordered = np.sort(rates)
    output = {
        "rate_mean": float(rates.mean()),
        "rate_std": float(rates.std()),
        "rate_cv": float(rates.std() / max(rates.mean(), 1e-12)),
        "rate_min": float(rates.min()),
        "rate_max": float(rates.max()),
        "rate_gini": gini(rates),
        "active_channels": float((rates > 0).sum()),
        "mean_pair_corr": float(correlations_arr.mean()),
        "max_pair_corr": float(correlations_arr.max()),
        "high_sync_bin_fraction": float((active_per_bin >= 4).mean()),
        "all_sync_bin_fraction": float((active_per_bin == 8).mean()),
        "silent_bin_fraction": float((active_per_bin == 0).mean()),
        "mean_isi_cv": float(np.mean(isi_cvs)) if isi_cvs else 0.0,
        "total_spikes": float(sum(len(values) for values in channels)),
    }
    output.update({f"sorted_rate_{index + 1}": float(value) for index, value in enumerate(ordered)})
    return output


def corrupt(channels: list[np.ndarray], label: str, rng: np.random.Generator) -> list[np.ndarray]:
    result = [values.copy() for values in channels]
    if label == "clean":
        return result
    if label == "dropout":
        for index in rng.choice(8, size=int(rng.integers(1, 4)), replace=False):
            result[int(index)] = np.asarray([], dtype=float)
    elif label == "noise":
        for index in rng.choice(8, size=int(rng.integers(1, 4)), replace=False):
            base_rate = max(len(result[int(index)]) / WINDOW_S, 0.5)
            added = rng.uniform(0, WINDOW_S, size=int(rng.poisson(base_rate * WINDOW_S * rng.uniform(1.5, 4.0))))
            result[int(index)] = np.sort(np.concatenate([result[int(index)], added]))
    elif label == "duplicate":
        source = int(np.argmax([len(values) for values in result]))
        source_values = result[source]
        if len(source_values):
            copied = rng.choice(source_values, size=max(1, int(0.5 * len(source_values))), replace=False)
            targets = rng.choice([idx for idx in range(8) if idx != source], size=5, replace=False)
            for index in targets:
                jittered = np.clip(copied + rng.normal(0, 0.0002, size=len(copied)), 0, WINDOW_S - 1e-8)
                result[int(index)] = np.sort(np.concatenate([result[int(index)], jittered]))
    elif label == "quantized":
        quantum = float(rng.choice([0.01, 0.02, 0.05]))
        result = [np.unique(np.round(values / quantum) * quantum) for values in result]
    else:
        raise ValueError(label)
    return result


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for path in sorted(RAW.glob("*.mat")):
        organoid, duration, channels = load_recording(path)
        n_windows = int(duration // WINDOW_S)
        for window_index in range(n_windows):
            window = slice_window(channels, window_index * WINDOW_S, (window_index + 1) * WINDOW_S)
            if sum(len(values) for values in window) < 20:
                continue
            for label in CLASSES:
                variant = corrupt(window, label, rng)
                rows.append({
                    "organoid": organoid,
                    "window": window_index,
                    "label": label,
                    **features(variant),
                })
    return pd.DataFrame(rows)


def evaluate_model(frame: pd.DataFrame, model_name: str):
    feature_columns = [column for column in frame.columns if column not in {"organoid", "window", "label"}]
    x = frame[feature_columns].to_numpy()
    y = frame["label"].to_numpy()
    groups = frame["organoid"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        )
    elif model_name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
        )
    else:
        raise ValueError(model_name)
    predicted = cross_val_predict(model, x, y, groups=groups, cv=splitter, n_jobs=1)
    report = classification_report(y, predicted, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, predicted, labels=CLASSES)
    return report, matrix, predicted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = build_dataset()
    frame.to_csv(OUT / "synthetic_qc_features.csv", index=False)
    summaries = {}
    predictions = frame[["organoid", "window", "label"]].copy()
    for model_name in ("logistic", "random_forest"):
        report, matrix, predicted = evaluate_model(frame, model_name)
        summaries[model_name] = report
        predictions[f"pred_{model_name}"] = predicted
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(OUT / f"confusion_{model_name}.csv")
    predictions.to_csv(OUT / "grouped_cv_predictions.csv", index=False)
    (OUT / "qc_metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"rows={len(frame)}, organoids={frame['organoid'].nunique()}, windows={frame[['organoid','window']].drop_duplicates().shape[0]}")
    for model_name, report in summaries.items():
        print(model_name, "accuracy=", round(report["accuracy"], 4), "macro_f1=", round(report["macro avg"]["f1-score"], 4))


if __name__ == "__main__":
    main()

