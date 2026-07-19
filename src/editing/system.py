"""Assemble the sparse linear system that defines each Poisson-editing solve.

The solver works by turning one image channel into a discrete system of the form
$A x = b$, where unknowns are the values of the masked pixels. The guidance
field supplies the desired gradient constraints, and the destination image
provides the Dirichlet boundary values for pixels outside the mask.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
Edge = Tuple[int, int]


@dataclass(frozen=True)
class PoissonSystem:
    """Container for the assembled sparse system and the associated indexing data."""
    A: csr_matrix
    b: np.ndarray
    index_map: np.ndarray
    coords: np.ndarray

def build_index_map(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map every masked pixel to a compact index used by the linear system."""
    mask = mask.astype(bool)
    coords = np.argwhere(mask)
    index_map = -np.ones(mask.shape, dtype=np.int32)

    for k, (y, x) in enumerate(coords):
        index_map[y, x] = k

    return index_map, coords

def build_system(
    destination_channel: np.ndarray,
    mask: np.ndarray,
    guidance: Dict[Edge, np.ndarray] | None = None,
) -> PoissonSystem:
    """
    Build the sparse system A x = b for one color channel.

    The guidance dictionary stores per-direction gradient values for each pixel.
    When a neighboring pixel is outside the mask, the corresponding boundary value
    is taken from the destination channel. When it is inside the mask, it becomes
    a coupling term in the matrix A.
    """
    dest = np.asarray(destination_channel, dtype=np.float64)
    mask = mask.astype(bool)

    h, w = mask.shape
    index_map, coords = build_index_map(mask)
    n = len(coords)

    rows = []
    cols = []
    data = []
    b = np.zeros(n, dtype=np.float64)

    if guidance is None:
        guidance = {}

    for k, (y, x) in enumerate(coords):
        diag = 0.0
        rhs = 0.0

        for dy, dx in _NEIGHBORS_4:
            ny, nx = y + dy, x + dx

            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue

            diag += 1.0

            edge_field = guidance.get((dy, dx))
            if edge_field is not None:
                rhs += float(edge_field[y, x])

            if mask[ny, nx]:
                # Interior neighbor: add a -1 entry in the matrix for the coupling.
                rows.append(k)
                cols.append(index_map[ny, nx])
                data.append(-1.0)
            else:
                # Exterior neighbor: contribute its value as a boundary term to b.
                rhs += dest[ny, nx]

        rows.append(k)
        cols.append(k)
        data.append(diag)
        b[k] = rhs

    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return PoissonSystem(A=A, b=b, index_map=index_map, coords=coords)