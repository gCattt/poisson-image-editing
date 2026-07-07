from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
Edge = Tuple[int, int]


@dataclass(frozen=True)
class PoissonSystem:
    A: csr_matrix
    b: np.ndarray
    index_map: np.ndarray
    coords: np.ndarray

def build_index_map(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    Build A x = b for one color channel.

    guidance is a dict:
        (dy, dx) -> array with the same shape of the image
    containing v_pq values for each pixel p and direction (dy, dx).
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
                rows.append(k)
                cols.append(index_map[ny, nx])
                data.append(-1.0)
            else:
                rhs += dest[ny, nx]

        rows.append(k)
        cols.append(k)
        data.append(diag)
        b[k] = rhs

    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return PoissonSystem(A=A, b=b, index_map=index_map, coords=coords)