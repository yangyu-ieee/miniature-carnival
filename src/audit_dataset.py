"""Audit the public forebrain-organoid MEA dataset used by NeuroChip Copilot.

The script is deliberately read-only with respect to source data. It records
file hashes, extracts MATLAB-v7.3 metadata and spike counts, and joins these
records to the authors' figure/source table. No biological claim is inferred.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import openpyxl
import pandas as pd

try:  # Supports both ``python -m src.audit_dataset`` and direct script runs.
    from .matlab_v73 import read_referenced_cell_vectors
except ImportError:  # pragma: no cover - exercised by the documented CLI entrypoint.
    from matlab_v73 import read_referenced_cell_vectors


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SOURCE_WORKBOOK = RAW / "all_fig_table_data_revision2.xlsx"
SOURCE_DOI = "10.5281/zenodo.22755004"
SOURCE_URL = "https://zenodo.org/records/22755004"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(group: h5py.Group, name: str) -> float:
    return float(np.asarray(group[name]).reshape(-1)[0])


def decode_u16(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1)
    return "".join(chr(int(value)) for value in values if value)


def audit_mat(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        meta = handle["metadata"]
        spikes = read_referenced_cell_vectors(handle, handle["spike_times"])
        amplitudes = read_referenced_cell_vectors(handle, handle["amplitudes"])
        duration = scalar(meta, "recording_duration")
        counts = np.asarray([len(channel) for channel in spikes], dtype=int)
        amp_counts = np.asarray([len(channel) for channel in amplitudes], dtype=int)
        empty_spike_channels = np.flatnonzero(counts == 0) + 1
        empty_amplitude_channels = np.flatnonzero(amp_counts == 0) + 1
        return {
            "organoid": decode_u16(handle["organoid_id"]),
            "file": path.name,
            "sha256": sha256(path),
            "fs_id": int(scalar(meta, "fs_id")),
            "org_id": int(scalar(meta, "org_id")),
            "n_channels": int(scalar(meta, "n_channels")),
            "recording_duration_s": duration,
            "total_spikes": int(counts.sum()),
            "active_channels": int((counts > 0).sum()),
            "n_empty_spike_time_vectors": int(len(empty_spike_channels)),
            "empty_spike_time_channels_1based": ",".join(map(str, empty_spike_channels.tolist())),
            "mean_firing_rate_hz": float(counts.sum() / (duration * len(counts))),
            "min_channel_spikes": int(counts.min()),
            "max_channel_spikes": int(counts.max()),
            "spike_amplitude_count_match": bool(np.array_equal(counts, amp_counts)),
            "n_empty_amplitude_vectors": int(sum(len(channel) == 0 for channel in amplitudes)),
            "empty_amplitude_channels_1based": ",".join(map(str, empty_amplitude_channels.tolist())),
        }


def read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook[sheet]
    rows = list(worksheet.values)
    return pd.DataFrame(rows[1:], columns=rows[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mat_files = sorted((RAW / "Indiv_organoids").glob("*.mat"))
    records = pd.DataFrame(audit_mat(path) for path in mat_files)
    source_features = read_sheet(SOURCE_WORKBOOK, "ST1")
    merged = source_features.merge(records, on="organoid", how="outer", validate="one_to_one")

    prior_workbook = RAW / "fig_table_source_data.xlsx"
    prior_features = read_sheet(prior_workbook, "ST1")
    feature_names = [column for column in source_features.columns if column != "organoid"]
    joined_versions = prior_features.merge(
        source_features, on="organoid", how="inner", suffixes=("_v2", "_v3"), validate="one_to_one"
    )
    version_changes: dict[str, dict[str, float | int]] = {}
    for feature in feature_names:
        prior = pd.to_numeric(joined_versions[f"{feature}_v2"], errors="coerce")
        latest = pd.to_numeric(joined_versions[f"{feature}_v3"], errors="coerce")
        absolute_change = (latest - prior).abs()
        changed = absolute_change > 1e-12
        if bool(changed.any()):
            version_changes[feature] = {
                "n_values_changed": int(changed.sum()),
                "max_absolute_change": float(absolute_change[changed].max()),
            }

    feature_columns = [column for column in source_features.columns if column != "organoid"]
    report = {
        "dataset_title": "Spontaneous activity spike data and RNA-seq gene expression data from human forebrain organoids",
        "dataset_doi": SOURCE_DOI,
        "dataset_version": "v3",
        "dataset_url": SOURCE_URL,
        "license": "CC-BY-4.0",
        "version_comparison": {
            "prior_version": "v2",
            "prior_doi": "10.5281/zenodo.20286251",
            "prior_workbook_sha256": sha256(prior_workbook),
            "current_workbook": SOURCE_WORKBOOK.name,
            "current_workbook_sha256": sha256(SOURCE_WORKBOOK),
            "changed_source_features": version_changes,
        },
        "n_mat_files": len(mat_files),
        "n_source_table_rows": int(len(source_features)),
        "n_joined_rows": int(len(merged)),
        "all_ids_joined": bool(merged["file"].notna().all() and merged[feature_columns].notna().any(axis=1).all()),
        "recording_duration_values_s": sorted(records["recording_duration_s"].unique().tolist()),
        "n_channels_values": sorted(records["n_channels"].unique().tolist()),
        "active_channel_range": [int(records["active_channels"].min()), int(records["active_channels"].max())],
        "n_records_with_all_eight_channels_active": int((records["active_channels"] == 8).sum()),
        "n_records_with_empty_spike_time_vectors": int((records["n_empty_spike_time_vectors"] > 0).sum()),
        "n_empty_spike_time_vectors": int(records["n_empty_spike_time_vectors"].sum()),
        "all_spike_amplitude_counts_match": bool(records["spike_amplitude_count_match"].all()),
        "n_records_with_empty_amplitude_vectors": int((records["n_empty_amplitude_vectors"] > 0).sum()),
        "n_empty_amplitude_vectors": int(records["n_empty_amplitude_vectors"].sum()),
        "feature_columns": feature_columns,
        "source_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(RAW.glob("*.zip")) + [prior_workbook, SOURCE_WORKBOOK]
        },
        "limitations": [
            "MATLAB v7.3 MATLAB_empty cell payloads are interpreted as empty vectors, not as two data observations.",
            "This release contains 45 organoid-level records but does not expose a machine-readable age table in ST1.",
            "The data are spontaneous recordings without treatment labels, so drug-response prediction is not supported.",
            "Samples come from three differentiation batches; batch identifiers must be recovered before claiming cross-batch generalization.",
        ],
    }

    records.to_csv(OUT / "recording_audit.csv", index=False)
    merged.to_csv(OUT / "organoid_feature_table.csv", index=False)
    (OUT / "dataset_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
