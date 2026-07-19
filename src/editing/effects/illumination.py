"""Local illumination change using logarithmic-domain gradient compression.

The effect brightens shadows and suppresses highlights by compressing the image's
log-domain gradients according to a luminance-based scale map. The Poisson solve
then rebuilds the region so the resulting illumination change feels natural.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from scipy.ndimage import sobel

from ..solver import solve_poisson_channel

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
Edge = Tuple[int, int]


def _build_illumination_guidance(
    img_log_c: np.ndarray,
    scale_map: np.ndarray
) -> Dict[Edge, np.ndarray]:
    """Construct the guidance field by scaling log-domain gradients with a luminance map."""
    h, w = img_log_c.shape
    guidance: Dict[Edge, np.ndarray] = {}

    for dy, dx in _NEIGHBORS_4:
        field = np.zeros((h, w), dtype=np.float64)

        y0 = max(0, -dy)
        y1 = h - max(0, dy)
        x0 = max(0, -dx)
        x1 = w - max(0, dx)

        if y1 > y0 and x1 > x0:
            # Original gradient in the logarithmic domain for channel c
            diff_log_c = img_log_c[y0:y1, x0:x1] - img_log_c[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            
            # Interpolate the scale map to the midpoint between p and q
            edge_scale = (scale_map[y0:y1, x0:x1] + scale_map[y0 + dy : y1 + dy, x0 + dx : x1 + dx]) / 2.0
            
            # Compress the gradient by multiplying with the scale map
            field[y0:y1, x0:x1] = diff_log_c * edge_scale

        guidance[(dy, dx)] = field

    return guidance


def local_illumination_change(
    image: np.ndarray,
    mask: np.ndarray,
    alpha_factor: float = 0.2,
    beta: float = 0.2
) -> np.ndarray:
    """
    Modify local illumination of a region by compressing its dynamic range in the logarithmic domain.
    Useful for brightening shadows or reducing strong highlights while preserving texture.

    alpha_factor: Fraction (e.g., 0.2) of the average gradient. Controls the compression threshold.
    beta: Compression exponent (0.0 - 1.0). 
          beta = 0.0 does nothing. 
          beta > 0.0 compresses strong contrasts and amplifies weak textures.
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

    # 1. Extract the region of interest (ROI) and normalize to [0.0, 1.0]
    img_roi_float = image[ymin : ymax + 1, xmin : xmax + 1].astype(np.float64) / 255.0
    mask_crop = mask[ymin : ymax + 1, xmin : xmax + 1]
    
    # Constant epsilon to avoid instability with the logarithm of zero or divisions by zero
    eps = 1e-6

    # 2. Transform the entire ROI to the logarithmic domain
    img_log = np.log(np.clip(img_roi_float, eps, 1.0))

    # 3. Luminance is used to compute a single scale map that applies to all channels.
    lum = (0.299 * img_roi_float[:, :, 0] +
           0.587 * img_roi_float[:, :, 1] +
           0.114 * img_roi_float[:, :, 2])
    log_lum = np.log(np.clip(lum, eps, 1.0))
    
    # 4. Calculate the magnitude of the logarithmic gradient (via Sobel)
    gx = sobel(log_lum, axis=1)
    gy = sobel(log_lum, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # 5. Use the average gradient magnitude in the masked region to determine the compression threshold.
    mean_mag = np.mean(magnitude[mask_crop]) + eps
    alpha_val = alpha_factor * mean_mag
    
    # 6. Scale map of the gradients (the same map will be used for the 3 RGB channels)
    scale_map = (alpha_val ** beta) * (magnitude + eps) ** (-beta)

    # 7. Solving the Poisson system in the LOGARITHMIC DOMAIN
    result = image.astype(np.float64).copy()
    dest_roi_log = img_log.copy()

    for c in range(3):
        guidance = _build_illumination_guidance(img_log[:, :, c], scale_map)
        
        # The boundary conditions are given by img_log (which represents the original image in log)
        dest_roi_log[:, :, c] = solve_poisson_channel(
            img_log[:, :, c],
            mask_crop,
            guidance=guidance
        )
        
    # 8. Conversion from the logarithmic domain to the linear domain (exponential)
    dest_roi_exp = np.exp(dest_roi_log)
    
    # 9. Denormalization to [0, 255] and recomposition
    result[ymin : ymax + 1, xmin : xmax + 1] = np.clip(dest_roi_exp * 255.0, 0, 255)
    return np.rint(result).astype(np.uint8)