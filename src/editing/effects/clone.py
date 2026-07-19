"""Seamless cloning using a Poisson-guided image blend.

The implementation solves for values inside the masked region by preserving the
source image gradients while anchoring the result to the destination image on
its boundary. This makes pasted regions appear naturally integrated.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from ..solver import solve_poisson_channel

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
Edge = Tuple[int, int]


def _build_source_guidance(channel: np.ndarray) -> Dict[Edge, np.ndarray]:
    """
    Build the guidance field for seamless cloning:
    v_pq = g_p - g_q for each 4-connected edge.
    """
    channel = np.asarray(channel, dtype=np.float64)
    h, w = channel.shape
    guidance: Dict[Edge, np.ndarray] = {}

    for dy, dx in _NEIGHBORS_4:
        field = np.zeros((h, w), dtype=np.float64)

        y0 = max(0, -dy)
        y1 = h - max(0, dy)
        x0 = max(0, -dx)
        x1 = w - max(0, dx)

        if y1 > y0 and x1 > x0:
            # The gradient is taken from the source patch, which steers the result.
            field[y0:y1, x0:x1] = (
                channel[y0:y1, x0:x1] - channel[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            )

        guidance[(dy, dx)] = field

    return guidance

def _solve_channel(
    source_channel: np.ndarray,
    destination_channel: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Solve one color channel of seamless cloning."""
    guidance = _build_source_guidance(source_channel)
    result = solve_poisson_channel(destination_channel, mask, guidance)
    return result

def seamless_cloning(
    source: np.ndarray,
    destination: np.ndarray,
    mask: np.ndarray,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    Seamless cloning for RGB images.

	source: patch to be cloned, shape (H_s, W_s, 3) 
    destination: full image, shape (H_d, W_d, 3) 
    mask: boolean mask for the source patch, shape (H_s, W_s) 
    offset = (y, x): top-left corner of the source patch in destination. 
    """
    source = np.asarray(source)
    destination = np.asarray(destination)
    mask = np.asarray(mask).astype(bool)

    if source.ndim != 3 or destination.ndim != 3:
        raise ValueError("source and destination must be RGB images")
    if source.shape[2] != 3 or destination.shape[2] != 3:
        raise ValueError("source and destination must have 3 channels")
    if mask.shape != source.shape[:2]:
        raise ValueError("mask must have shape (H_s, W_s) matching source")

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return destination.copy()

    h_s, w_s = source.shape[:2]

    # Crop slightly around the mask to keep the boundary context available to the solver.
    ymin = max(0, ys.min() - 1)
    ymax = min(h_s - 1, ys.max() + 1)
    xmin = max(0, xs.min() - 1)
    xmax = min(w_s - 1, xs.max() + 1)

    src_crop = source[ymin : ymax + 1, xmin : xmax + 1]
    mask_crop = mask[ymin : ymax + 1, xmin : xmax + 1]

    oy, ox = offset
    noy = oy
    nox = ox

    h_crop, w_crop = src_crop.shape[:2]
    h_d, w_d = destination.shape[:2]

    if noy < 0 or nox < 0 or noy + h_crop > h_d or nox + w_crop > w_d:
        raise ValueError("source patch placed at offset exceeds destination bounds")

    result = destination.astype(np.float64).copy()
    dest_roi = result[noy : noy + h_crop, nox : nox + w_crop, :].copy()

    for c in range(3):
        dest_roi[:, :, c] = _solve_channel(src_crop[:, :, c], dest_roi[:, :, c], mask_crop)

    result[noy : noy + h_crop, nox : nox + w_crop, :] = dest_roi
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)