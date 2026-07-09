from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import cg, spsolve

from .system import build_system


def solve_sparse(
    A,
    b,
    method: str = "spsolve",
    tol: float = 1e-6,
    maxiter: int | None = None,
) -> np.ndarray:
    if method == "spsolve":
        x = spsolve(A, b)
        return np.asarray(x, dtype=np.float64)

    if method == "cg":
        x, info = cg(A, b, rtol=tol, maxiter=maxiter)
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
        return np.asarray(x, dtype=np.float64)

    raise ValueError(f"Unknown method: {method}")

def solve_poisson_channel(
    destination_channel: np.ndarray,
    mask: np.ndarray,
    guidance: dict[tuple[int, int], np.ndarray],
    method: str = "spsolve",
) -> np.ndarray:
    system = build_system(destination_channel, mask, guidance=guidance)
    sol = solve_sparse(system.A, system.b, method=method)

    result = np.asarray(destination_channel, dtype=np.float64).copy()
    result[mask] = sol
    return result