"""Small helpers for MATLAB v7.3 cell arrays stored as HDF5 references.

MATLAB encodes an empty array in a v7.3 file as a two-element dimensions
dataset carrying the ``MATLAB_empty`` attribute.  Those two dimension values
are metadata, not two observations; callers must see an empty vector.
"""

from __future__ import annotations

import h5py
import numpy as np


def read_referenced_cell_vectors(handle: h5py.File, dataset: h5py.Dataset) -> list[np.ndarray]:
    """Dereference a MATLAB cell-vector dataset without counting empty sentinels."""
    vectors: list[np.ndarray] = []
    for reference in dataset[:, 0]:
        payload = handle[reference]
        if int(payload.attrs.get("MATLAB_empty", 0)) == 1:
            vectors.append(np.empty(0, dtype=float))
        else:
            vectors.append(np.asarray(payload).reshape(-1))
    return vectors
