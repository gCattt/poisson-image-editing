"""Finite-difference gradient and divergence helpers used by the Poisson solver.

These operations are the discrete counterparts of the continuous gradient and
Laplacian operators used in Poisson image editing. They provide the numerical
building blocks for forming guidance fields and solving for image values that
preserve local image structure.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def image_gradient(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute horizontal and vertical finite-difference gradients of an image.

    The returned arrays are the same shape as the input image. Each pixel stores
    the local difference to its right neighbor (for the x-gradient) and the local
    difference to its lower neighbor (for the y-gradient).
    """
    img = np.asarray(img, dtype=np.float64)
    gx = np.zeros_like(img, dtype=np.float64)
    gy = np.zeros_like(img, dtype=np.float64)

    # The last column/row has no neighbor on the right/bottom, so it remains zero.
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return gx, gy

def divergence(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Compute the discrete divergence of a vector field.

    This is the operator that turns a gradient-like field back into a scalar field
    and is used in the Poisson formulation to express the equilibrium condition.
    """
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    div = np.zeros_like(vx, dtype=np.float64)

    # The divergence is assembled by accumulating outgoing fluxes from each edge.
    div[:, :-1] += vx[:, :-1]
    div[:, 1:] -= vx[:, :-1]
    div[:-1, :] += vy[:-1, :]
    div[1:, :] -= vy[:-1, :]
    return div