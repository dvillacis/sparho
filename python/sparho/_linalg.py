"""Small numeric helpers shared by the native BCD solver and hypergradient.

Kept dependency-free (numpy + scipy.sparse only) so both the adapters layer and
the hypergrad layer can import it without creating a package-level cycle.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


def column_lipschitz(design: object, n_samples: int) -> NDArray[np.float64]:
    """Per-column Lipschitz constants ``L_j = ‖X_j‖² / n`` (BLAS, not ported)."""
    if sp.issparse(design):
        sq = design.multiply(design)  # type: ignore[attr-defined]
        col_sq = np.asarray(sq.sum(axis=0)).ravel()
    else:
        Xa = np.asarray(design, dtype=np.float64)
        col_sq = np.einsum("ij,ij->j", Xa, Xa)
    return np.ascontiguousarray(col_sq / n_samples, dtype=np.float64)
