"""Simple regression test for the sparse Poisson system assembly and solver."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from editing.solver import solve_poisson_channel


def test_constant_fill() -> None:
    """Test standard Laplace equation (no guidance) with constant boundaries."""
    print("Running test_constant_fill...")
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    # Constant destination channel means the solution for the masked region should remain constant.
    destination = np.full((5, 5), 50.0, dtype=np.float64)

    # Use the public wrapper directly
    for method in ["spsolve", "cg"]:
        reconstructed = solve_poisson_channel(destination, mask, guidance=None, method=method)
        
        atol = 1e-8 if method == "spsolve" else 1e-5

        # Verify that the reconstructed image matches the constant value outside and inside the mask
        assert np.allclose(reconstructed[~mask], 50.0, atol=atol), f"Failed boundary preservation with {method}"
        assert np.allclose(reconstructed[mask], 50.0, atol=atol), f"Failed constant fill with {method}"
    
    print("  -> Passed!")

def test_gradient_reconstruction() -> None:
    """Test full Poisson equation by reconstructing a known gradient field."""
    print("Running test_gradient_reconstruction...")
    
    # 1. Create a 5x5 image with a known gradient (e.g., values from 0 to 24)
    original = np.arange(25, dtype=np.float64).reshape((5, 5))
    
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    # 2. Extract exact guidance (gradients) from the original image
    guidance = {}
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        v = np.zeros_like(original)
        if dy == -1:    v[1:, :] = original[1:, :] - original[:-1, :]
        elif dy == 1:   v[:-1, :] = original[:-1, :] - original[1:, :]
        elif dx == -1:  v[:, 1:] = original[:, 1:] - original[:, :-1]
        elif dx == 1:   v[:, :-1] = original[:, :-1] - original[:, 1:]
        guidance[(dy, dx)] = v

    # 3. Corrupt the destination image inside the mask
    dest = original.copy()
    rng = np.random.default_rng(0)
    dest[mask] = rng.uniform(-50, 50, size=mask.sum())

    # 4. Solve and verify that the original image is reconstructed within numerical tolerance.
    for method in ["spsolve", "cg"]:
        reconstructed = solve_poisson_channel(dest, mask, guidance=guidance, method=method)
        
        atol = 1e-8 if method == "spsolve" else 1e-5
        
        assert np.allclose(reconstructed[~mask], original[~mask], atol=atol), f"Boundary corrupted with {method}"
        assert np.allclose(reconstructed[mask], original[mask], atol=atol), f"Failed interior reconstruction with {method}"
    
    print("  -> Passed!")

def main() -> None:
    test_constant_fill()
    test_gradient_reconstruction()
    print("\nAll system.py + solver.py tests passed.")


if __name__ == "__main__":
    main()