"""Wrappers around the sparse linear solver used by each Poisson effect.

Each effect solves one color channel at a time by building a sparse system and
then applying a sparse solver. This module keeps the solving logic centralized so
that the effect implementations can focus on the guidance field they provide.
"""

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
    """Solve a sparse linear system with either a direct or iterative method."""
    if method == "spsolve":
        x = spsolve(A, b)
        return np.asarray(x, dtype=np.float64)

    if method == "cg":
        # Conjugate gradient is useful when the system is large and iterative solve
        # is preferred over a direct factorization.
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
    """Solve one channel of the Poisson problem and return the reconstructed values."""
    system = build_system(destination_channel, mask, guidance=guidance)
    sol = solve_sparse(system.A, system.b, method=method)

    # Only the masked pixels are unknowns; the rest of the channel is left unchanged.
    result = np.asarray(destination_channel, dtype=np.float64).copy()
    result[mask] = sol
    return result