"""Load the prepared 169-column CSV layout used by the recovered scripts."""

from pathlib import Path

import numpy as np


def load_csv(path, skip_header=False):
    values = np.loadtxt(
        Path(path), delimiter=",", dtype=np.float32, ndmin=2,
        encoding="utf-8-sig", skiprows=int(skip_header),
    )
    if values.shape[0] == 0 or values.shape[1] != 169:
        raise ValueError(
            f"Expected a nonempty 169-column prepared CSV, got {values.shape}. "
            "The 208-column raw database must be preprocessed separately."
        )
    if not np.isfinite(values).all():
        raise ValueError("CSV contains NaN or infinity.")
    return np.ascontiguousarray(values[:, 1:86]), np.ascontiguousarray(values[:, 86:169])
