"""Local color change by scaling the gradient of each color channel.

The effect works by solving a Poisson problem whose guidance field is derived
from the original image gradients multiplied by per-channel scale factors. This
changes the hue of the selected region while keeping its structure consistent.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from ..solver import solve_poisson_channel

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
Edge = Tuple[int, int]


def _build_color_guidance(
    img_roi_float: np.ndarray,
    c: int,
    scale: float
) -> Dict[Edge, np.ndarray]:
    """
    Construct the guidance field for Local Color Change.
    Equivalent to scaling the source image g by a constant factor and taking its gradient.
    v = nabla(scale * f_original) = scale * nabla(f_original)
    """
    h, w = img_roi_float.shape[:2]
    guidance: Dict[Edge, np.ndarray] = {}

    for dy, dx in _NEIGHBORS_4:
        field = np.zeros((h, w), dtype=np.float64)

        y0 = max(0, -dy)
        y1 = h - max(0, dy)
        x0 = max(0, -dx)
        x1 = w - max(0, dx)

        if y1 > y0 and x1 > x0:
            # Each channel is solved independently, but its gradient is scaled.
            diff_c = img_roi_float[y0:y1, x0:x1, c] - img_roi_float[y0 + dy : y1 + dy, x0 + dx : x1 + dx, c]
            
            # Multiply the gradient by the color scaling factor
            field[y0:y1, x0:x1] = diff_c * scale

        guidance[(dy, dx)] = field

    return guidance


def local_color_change(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str = 'subject',
    red_scale: float = 1.5,
    green_scale: float = 0.5,
    blue_scale: float = 0.5
) -> np.ndarray:
    """
    Modify the color of a selected object in a coarse manner.
    Scales the gradients of individual RGB channels to change the hue while preserving texture and illumination.

    mode: In 'subject' mode, the masked object is recolored by scaling the gradient in each channel. 
    In 'background' mode, the image background is converted to gray and the object is solved against that grayscale boundary.
    red_scale, green_scale, blue_scale: Scaling factors for the R, G, B channels respectively.
    For example, (1.5, 0.5, 0.5) would increase the red channel and decrease the green and blue channels, warming up the image).
    """
    image = np.asarray(image)
    mask = np.asarray(mask).astype(bool)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The image must be RGB")
    if mask.shape != image.shape[:2]:
        raise ValueError("The mask must have the same shape (H, W) as the image")
    
    # 1. Initialize result early to avoid UnboundLocalError
    result = image.astype(np.float64).copy()

    ys, xs = np.where(mask)
    if len(ys) == 0:
        if mode == 'subject':
            return image.copy()
        else:
            # If no mask and mode is background, return fully grayscale image
            lum = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
            return np.stack([lum, lum, lum], axis=-1).astype(np.uint8)

    # 2. Restrict the solve to a tight bounding box around the masked object for speed
    h, w = image.shape[:2]
    ymin = max(0, ys.min() - 1)
    ymax = min(h - 1, ys.max() + 1)
    xmin = max(0, xs.min() - 1)
    xmax = min(w - 1, xs.max() + 1)

    img_roi_float = result[ymin : ymax + 1, xmin : xmax + 1]
    mask_crop = mask[ymin : ymax + 1, xmin : xmax + 1]

    if mode == 'subject':
        # Array of scaling factors corresponding to the channels (R=0, G=1, B=2)
        scales = [red_scale, green_scale, blue_scale]
        dest_roi = img_roi_float.copy()

        # Solve the Poisson system independently for the 3 RGB channels
        for c in range(3):
            guidance = _build_color_guidance(img_roi_float, c, scales[c])
            dest_roi[:, :, c] = solve_poisson_channel(img_roi_float[:, :, c], mask_crop, guidance=guidance)

        result[ymin : ymax + 1, xmin : xmax + 1] = dest_roi

    elif mode == 'background':
        # Convert the full image to grayscale so the object is solved against a flat background.
        lum = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        dest_gray = np.stack([lum, lum, lum], axis=-1)
        
        # Extract the ROI of the grayscale background
        dest_roi = dest_gray[ymin : ymax + 1, xmin : xmax + 1].copy()
        
        # Solve Poisson INSIDE the object. The boundary is the grayscale background.
        for c in range(3):
            # Scale is 1.0 (keep the original colors of the object)
            guidance = _build_color_guidance(img_roi_float, c, 1.0)
            dest_roi[:, :, c] = solve_poisson_channel(dest_roi[:, :, c], mask_crop, guidance=guidance)
            
        # The final image is the fully grayscale background, with the solved colored object pasted in
        result = dest_gray
        result[ymin : ymax + 1, xmin : xmax + 1] = dest_roi

    else:
        raise ValueError("mode must be 'subject' or 'background'")

    # Final clipping to bring the altered values back to the visible range 0-255
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)