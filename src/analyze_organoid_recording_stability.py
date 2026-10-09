"""Audit temporal stability and integrity of public forebrain-organoid MEA files.

This is a descriptive recording-level QC module. It does not assign biological
quality labels or infer drug response. Spike-time channels are the analysis
source; amplitude vectors are checked separately and never silently aligned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import matplotlib

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Supports both ``python -m src.analyze_organoid_recording_stability`` and direct runs.
    from .matlab_v73 import read_referenced_cell_vectors
except ImportError:  # pragma: no cover - exercised by the documented CLI entrypoint.
    from matlab_v73 import read_referenced_cell_vectors


MAT_DIR = ROOT / "data" / "raw" / "Indiv_organoids"
FEATURES = ROOT / "data" / "processed" / "organoid_feature_table.csv"
OUT = ROOT / "results" / "organoid_recording_qc"
WINDOW_SECONDS = 10.0


def audit_recording(path: Path, reference_fr: float | None) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        metadata = handle["metadata"]
        duration = float(np.asarray(metadata["recording_duration"]).reshape(-1)[0])
        spikes = read_referenced_cell_vectors(handle, handle["spike_times"])
        amplitudes = read_referenced_cell_vectors(handle, handle["amplitudes"])

    counts = np.asarray([len(channel) for channel in spikes], dtype=int)
    amplitude_counts = np.asarray([len(channel) for channel in amplitudes], dtype=int)
    empty_spike_channels = np.flatnonzero(counts == 0) + 1
    empty_amplitude_channels = np.flatnonzero(amplitude_counts == 0) + 1
    total_spikes = int(counts.sum())
    computed_fr = total_spikes / (duration * len(spikes))

    # Split the recording at its temporal midpoint. This measures within-file
    # drift, not biological quality and not test-retest reproducibility.
    midpoint = duration / 2.0
    first_counts = np.asarray([np.count_nonzero(x < midpoint) for x in spikes], dtype=float)
    second_counts = np.asarray([np.count_nonzero(x >= midpoint) for x in spikes], dtype=float)
    first_fr = float(first_counts.sum() / (midpoint * len(spikes)))
    second_fr = float(second_counts.sum() / ((duration - midpoint) * len(spikes)))
    relative_half_change = abs(second_fr - first_fr) / max((first_fr + second_fr) / 2.0, 1e-12)

    # Compare the spatial firing-rate profile across halves; zero-variance
    # profiles are reported as undefined rather than assigned an artificial score.
    first_profile = first_counts / midpoint
    second_profile = second_counts / (duration - midpoint)
    if np.std(first_profile) and np.std(second_profile):
        profile_correlation = float(np.corrcoef(first_profile, second_profile)[0, 1])
    else:
        profile_correlation = float("nan")

    n_windows = int(np.floor(duration / WINDOW_SECONDS))
    window_counts = np.zeros(n_windows, dtype=float)
    edges = np.arange(n_windows + 1, dtype=float) * WINDOW_SECONDS
    for channel in spikes:
        window_counts += np.histogram(channel, bins=edges)[0]
    window_fr = window_counts / (WINDOW_SECONDS * len(spikes))
    mean_window_fr = float(window_fr.mean()) if len(window_fr) else 0.0
    window_cv = (
        float(window_fr.std(ddof=1) / mean_window_fr)
        if len(window_fr) > 1 and mean_window_fr > 0
        else float("nan")
    )

    mismatch_channels = np.flatnonzero(counts != amplitude_counts) + 1
    raw_times_valid = all(
        np.isfinite(channel).all()
        and (len(channel) < 2 or np.all(np.diff(channel) >= 0))
        and (len(channel) == 0 or (channel.min() >= 0 and channel.max() <= duration))
        for channel in spikes
    )
    return {
        "organoid": path.stem,
        "duration_s": duration,
        "n_channels": len(spikes),
        "active_channels": int((counts > 0).sum()),
        "n_empty_spike_time_vectors": int(len(empty_spike_channels)),
        "empty_spike_time_channels_1based": ",".join(map(str, empty_spike_channels.tolist())),
        "total_spikes": total_spikes,
        "computed_mean_fr_hz": computed_fr,
        "source_table_fr_hz": reference_fr,
        "source_fr_abs_error_hz": abs(computed_fr - reference_fr) if reference_fr is not None else None,
        "first_half_fr_hz": first_fr,
        "second_half_fr_hz": second_fr,
        "half_rate_relative_change": relative_half_change,
        "half_channel_profile_correlation": profile_correlation,
        "ten_second_window_fr_cv": window_cv,
        "n_ten_second_windows": n_windows,
        "spike_times_valid": bool(raw_times_valid),
        "amplitude_counts_match": bool(np.array_equal(counts, amplitude_counts)),
        "n_empty_amplitude_vectors": int(sum(len(channel) == 0 for channel in amplitudes)),
        "empty_amplitude_channels_1based": ",".join(map(str, empty_amplitude_channels.tolist())),
        "amplitude_mismatch_channels_1based": ",".join(map(str, mismatch_channels.tolist())),
        "amplitude_mismatch_count_delta": int(np.abs(counts - amplitude_counts).sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(FEATURES).set_index("organoid")
    rows = []
    for path in sorted(MAT_DIR.glob("*.mat")):
        reference_fr = float(features.loc[path.stem, "FR"]) if path.stem in features.index else None
        rows.append(audit_recording(path, reference_fr))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "recording_qc.csv", index=False)

    duration_summary = (
        table.groupby("duration_s", as_index=False)
        .agg(
            n_recordings=("organoid", "size"),
            median_half_rate_change=("half_rate_relative_change", "median"),
            median_half_profile_correlation=("half_channel_profile_correlation", "median"),
            median_window_fr_cv=("ten_second_window_fr_cv", "median"),
        )
    )
    duration_summary.to_csv(OUT / "duration_summary.csv", index=False)
    bad_amplitude = table.loc[~table["amplitude_counts_match"]]
    summary = {
        "n_recordings": int(len(table)),
        "duration_values_and_counts": {
            str(float(k)): int(v) for k, v in table["duration_s"].value_counts().sort_index().items()
        },
        "all_spike_time_vectors_valid": bool(table["spike_times_valid"].all()),
        "all_eight_channels_active": bool((table["active_channels"] == 8).all()),
        "n_records_with_all_eight_channels_active": int((table["active_channels"] == 8).sum()),
        "n_records_with_empty_spike_time_vectors": int((table["n_empty_spike_time_vectors"] > 0).sum()),
        "n_empty_spike_time_vectors": int(table["n_empty_spike_time_vectors"].sum()),
        "max_source_table_firing_rate_abs_error_hz": float(table["source_fr_abs_error_hz"].max()),
        "n_source_table_fr_differences_over_1e_3_hz": int((table["source_fr_abs_error_hz"] > 1e-3).sum()),
        "n_amplitude_count_mismatches": int(len(bad_amplitude)),
        "n_records_with_empty_amplitude_vectors": int((table["n_empty_amplitude_vectors"] > 0).sum()),
        "n_empty_amplitude_vectors": int(table["n_empty_amplitude_vectors"].sum()),
        "amplitude_mismatch_records": bad_amplitude[
            ["organoid", "empty_spike_time_channels_1based", "empty_amplitude_channels_1based", "amplitude_mismatch_channels_1based", "amplitude_mismatch_count_delta"]
        ].to_dict(orient="records"),
        "median_half_rate_relative_change": float(table["half_rate_relative_change"].median()),
        "median_half_channel_profile_correlation": float(table["half_channel_profile_correlation"].median()),
        "interpretation_boundary": [
            "Half-recording comparisons are within-recording drift diagnostics, not biological quality labels.",
            "Recording durations are observational and confounded with which organoids were recorded; duration-group contrasts are descriptive, not causal.",
            "Empty or count-mismatched amplitude vectors must not be used for spike-amplitude analyses without source clarification.",
            "Empty spike-time vectors are zero-event channels, not malformed timestamps; channel activity is reported separately from vector validity.",
            "No treatment labels are present; these data do not support drug-response prediction.",
        ],
    }
    (OUT / "recording_qc_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    # The plot is descriptive and uses one point per organoid recording.
    # Duration categories are jittered only for visibility; no threshold or
    # quality classification is implied by the visual encoding.
    durations = sorted(table["duration_s"].unique())
    category = {duration: idx for idx, duration in enumerate(durations)}
    ordered = table.sort_values(["duration_s", "organoid"]).reset_index(drop=True)
    jitter_index = ordered.groupby("duration_s").cumcount()
    group_sizes = ordered.groupby("duration_s")["organoid"].transform("size")
    ordered["x_jitter"] = ordered["duration_s"].map(category) + (
        jitter_index - (group_sizes - 1) / 2
    ) * 0.035

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    mismatch = ~ordered["amplitude_counts_match"]
    for axis, metric, label in [
        (axes[0], "half_rate_relative_change", "Relative change in mean firing rate"),
        (axes[1], "half_channel_profile_correlation", "Correlation of 8-channel rate profiles"),
    ]:
        axis.scatter(
            ordered.loc[~mismatch, "x_jitter"],
            ordered.loc[~mismatch, metric],
            s=42,
            color="#177E89",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
                label="Amplitude vectors complete/counts match",
        )
        if mismatch.any():
            axis.scatter(
                ordered.loc[mismatch, "x_jitter"],
                ordered.loc[mismatch, metric],
                s=74,
                marker="X",
                color="#D1495B",
                edgecolor="white",
                linewidth=0.6,
                label="Empty/mismatched amplitude vector",
                zorder=4,
            )
        axis.set_xticks(range(len(durations)))
        axis.set_xticklabels([f"{int(d)}\n(n={int((table.duration_s == d).sum())})" for d in durations])
        axis.set_xlabel("Recording duration (s)")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_title("Within-recording rate drift")
    axes[1].set_title("Spatial profile consistency")
    axes[1].set_ylim(-0.05, 1.05)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Forebrain-organoid MEA recording audit (n=45)", y=1.15, fontsize=13, fontweight="bold")
    fig.savefig(OUT / "recording_stability.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "recording_stability.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
