"""Texture flattening that preserves strong edges while smoothing fine texture.

The implementation uses Canny edges to decide which gradient constraints should be
kept. This means the solver smooths the interior of the masked region but leaves
important contours intact.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from skimage.feature import canny

from ..solver import solve_poisson_channel

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
Edge = Tuple[int, int]


def _build_flatten_guidance(
    img_roi_float: np.ndarray,
    edge_mask: np.ndarray,
    c: int
) -> Dict[Edge, np.ndarray]:
    """Keep the original gradient only when an edge is present between two pixels."""
    h, w = img_roi_float.shape[:2]
    guidance: Dict[Edge, np.ndarray] = {}

    for dy, dx in _NEIGHBORS_4:
        field = np.zeros((h, w), dtype=np.float64)

        y0 = max(0, -dy)
        y1 = h - max(0, dy)
        x0 = max(0, -dx)
        x1 = w - max(0, dx)

        if y1 > y0 and x1 > x0:
            # Original gradient for channel c
            diff_c = img_roi_float[y0:y1, x0:x1, c] - img_roi_float[y0 + dy : y1 + dy, x0 + dx : x1 + dx, c]

            # Only preserve gradients that cross a detected edge, which helps keep contours sharp.
            edge_between = edge_mask[y0:y1, x0:x1] | edge_mask[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            
            # Sparse sieve M
            field[y0:y1, x0:x1] = np.where(edge_between, diff_c, 0.0)

        guidance[(dy, dx)] = field

    return guidance


def texture_flattening(
    image: np.ndarray,
    mask: np.ndarray,
    canny_sigma: float = 1.5,
    low_threshold: float | None = None,
    high_threshold: float | None = None
) -> np.ndarray:
    """
    Flatten texture inside a region while preserving strong edges.
    The Canny detector identifies which structures should remain visible. 
    The Poisson solve then smooths the rest of the masked area while respecting those
    edge constraints.
    
    canny_sigma: Standard deviation for the internal Gaussian filter in Canny.
                 A higher value will ignore finer textures and focus on more pronounced edges (like deep wrinkles).
    low_threshold, high_threshold: Optional thresholds for the Canny edge detector. If not provided, they will be automatically determined based on the image.
    """
    image = np.asarray(image)
    mask = np.asarray(mask).astype(bool)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The image must be RGB")
    if mask.shape != image.shape[:2]:
        raise ValueError("The mask must have the same shape (H, W) as the image")

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return image.copy()

    h, w = image.shape[:2]

    ymin = max(0, ys.min() - 1)
    ymax = min(h - 1, ys.max() + 1)
    xmin = max(0, xs.min() - 1)
    xmax = min(w - 1, xs.max() + 1)

    # 1. Extract the region of interest (ROI) from the image and mask
    img_roi_float = image[ymin : ymax + 1, xmin : xmax + 1].astype(np.float64)
    mask_crop = mask[ymin : ymax + 1, xmin : xmax + 1]
    
    # 2. Calculate the luminance of the ROI for edge detection
    # Standard formula for converting RGB to luminance (Y channel). This gives Canny a single-channel signal.
    luminance = (
        0.299 * (img_roi_float[:, :, 0] / 255.0) +
        0.587 * (img_roi_float[:, :, 1] / 255.0) +
        0.114 * (img_roi_float[:, :, 2] / 255.0)
    )

    # 3. Canny Edge Detector
    edge_mask = canny(
        luminance,
        sigma=canny_sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )

    # 4. Run the Poisson solver for each channel with the constructed guidance field
    result = image.astype(np.float64).copy()
    dest_roi = result[ymin : ymax + 1, xmin : xmax + 1].copy()

    for c in range(3):
        guidance = _build_flatten_guidance(img_roi_float, edge_mask, c)
        dest_roi[:, :, c] = solve_poisson_channel(img_roi_float[:, :, c], mask_crop, guidance=guidance)

    result[ymin : ymax + 1, xmin : xmax + 1] = dest_roi
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)