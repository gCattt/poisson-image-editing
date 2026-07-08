from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from editing.system import build_system
from editing.solver import solve_sparse


def main() -> None:
    # 5x5 image, 3x3 mask in the center
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    # Constant destination channel
    destination = np.full((5, 5), 50.0, dtype=np.float64)

    system = build_system(destination, mask, guidance=None)

    # Basic structural checks
    assert system.A.shape == (mask.sum(), mask.sum()), system.A.shape
    assert system.b.shape == (mask.sum(),), system.b.shape
    assert system.index_map.shape == mask.shape

    # Solve
    x = solve_sparse(system.A, system.b, method="spsolve")

    # Reconstruct the full channel
    reconstructed = destination.copy()
    reconstructed[mask] = x

    # Numerical check: the masked region should stay constant
    max_err = np.max(np.abs(reconstructed[mask] - 50.0))
    print("max error inside mask:", max_err)
    print("solution inside mask:\n", reconstructed)

    assert np.allclose(reconstructed[mask], 50.0, atol=1e-8), reconstructed[mask]

    print("system.py + solver.py test passed.")


if __name__ == "__main__":
    main()