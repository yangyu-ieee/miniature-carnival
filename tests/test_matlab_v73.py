"""Regression test for MATLAB v7.3 empty-cell sentinel handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.matlab_v73 import read_referenced_cell_vectors
from src.analyze_organoid_recording_stability import audit_recording


class MatlabV73CellVectorTests(unittest.TestCase):
    def test_empty_array_sentinel_is_not_counted_as_two_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matlab_cells.h5"
            with h5py.File(path, "w") as handle:
                refs = handle.create_group("#refs#")
                empty = refs.create_dataset("empty", data=np.asarray([0, 0], dtype=np.uint64))
                empty.attrs["MATLAB_empty"] = np.uint8(1)
                populated = refs.create_dataset("populated", data=np.asarray([1.25, -2.5]))
                cells = handle.create_dataset("amplitudes", shape=(2, 1), dtype=h5py.ref_dtype)
                cells[0, 0] = empty.ref
                cells[1, 0] = populated.ref

            with h5py.File(path, "r") as handle:
                vectors = read_referenced_cell_vectors(handle, handle["amplitudes"])

            self.assertEqual([len(vector) for vector in vectors], [0, 2])
            np.testing.assert_allclose(vectors[1], [1.25, -2.5])

    def test_empty_amplitude_sentinel_does_not_false_match_two_spikes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matlab_cells.h5"
            with h5py.File(path, "w") as handle:
                refs = handle.create_group("#refs#")
                empty = refs.create_dataset("empty", data=np.asarray([0, 0], dtype=np.uint64))
                empty.attrs["MATLAB_empty"] = np.uint8(1)
                spikes = refs.create_dataset("two_spikes", data=np.asarray([0.5, 1.0]))
                cells = handle.create_dataset("vectors", shape=(2, 1), dtype=h5py.ref_dtype)
                cells[0, 0] = empty.ref
                cells[1, 0] = spikes.ref

            with h5py.File(path, "r") as handle:
                vectors = read_referenced_cell_vectors(handle, handle["vectors"])

            counts = np.asarray([len(vector) for vector in vectors])
            np.testing.assert_array_equal(counts, [0, 2])
            self.assertFalse(np.array_equal(np.asarray([2, 2]), counts))

    def test_frozen_organoid_dataset_empty_vector_accounting(self):
        root = Path(__file__).resolve().parents[1] / "data" / "raw" / "Indiv_organoids"
        files = sorted(root.glob("*.mat"))
        if not files:
            self.skipTest("public raw MAT files are not staged in this source-only environment")
        rows = [audit_recording(path, None) for path in files]
        self.assertEqual(len(rows), 45)
        self.assertTrue(all(row["spike_times_valid"] for row in rows))
        self.assertEqual(sum(int(row["n_empty_spike_time_vectors"]) for row in rows), 4)
        self.assertEqual(sum(int(row["n_empty_amplitude_vectors"]) for row in rows), 12)
        mismatches = [row for row in rows if not row["amplitude_counts_match"]]
        self.assertEqual([row["organoid"] for row in mismatches], ["fs417-org0"])
        self.assertEqual(mismatches[0]["amplitude_mismatch_count_delta"], 33096)
        self.assertEqual(sum(int(row["active_channels"]) == 8 for row in rows), 41)


if __name__ == "__main__":
    unittest.main()
