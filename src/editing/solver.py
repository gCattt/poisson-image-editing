from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import cg, spsolve


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