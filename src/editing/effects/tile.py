"""Create tileable images by solving a Poisson problem on the image borders.

The method blends the image edges so that a tiled composition is visually smooth.
It uses a boundary image with averaged edge values and then solves for the inner
pixels while preserving the original image gradients.
"""

from __future__ import annotations
import numpy as np

from ..solver import solve_poisson_channel


def _compute_guidance(img_channel: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Compute four directional gradient fields for the image channel."""
    guidance = {}

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        v = np.zeros_like(img_channel, dtype=np.float64)
        
        # The sign convention is chosen to match the Poisson system's neighbor handling.
        if dy == -1:    # Nord (q is above p: y-1)
            v[1:, :] = img_channel[1:, :] - img_channel[:-1, :]
        elif dy == 1:   # Sud (q is below p: y+1)
            v[:-1, :] = img_channel[:-1, :] - img_channel[1:, :]
        elif dx == -1:  # Ovest (q is to the left of p: x-1)
            v[:, 1:] = img_channel[:, 1:] - img_channel[:, :-1]
        elif dx == 1:   # Est (q is to the right of p: x+1)
            v[:, :-1] = img_channel[:, :-1] - img_channel[:, 1:]

        guidance[(dy, dx)] = v

    return guidance

def _create_boundary_image(img_channel: np.ndarray) -> np.ndarray:
    """Construct a boundary image whose edge values are averaged to remove seams."""
    dest = img_channel.copy().astype(np.float64)
    
    # Calculate the averages using the original values to avoid sequential overwriting
    ns_avg = 0.5 * (img_channel[0, :] + img_channel[-1, :])
    ew_avg = 0.5 * (img_channel[:, 0] + img_channel[:, -1])
    
    # For the corners, take the average of all four corners to avoid asymmetry
    corner_avg = 0.25 * (
        img_channel[0, 0] + img_channel[0, -1] +
        img_channel[-1, 0] + img_channel[-1, -1]
    )

    # The border is forced to the average of its opposite neighbors to make the tiling seamless.
    dest[0, :] = ns_avg
    dest[-1, :] = ns_avg
    dest[:, 0] = ew_avg
    dest[:, -1] = ew_avg
    dest[0, 0] = corner_avg
    dest[0, -1] = corner_avg
    dest[-1, 0] = corner_avg
    dest[-1, -1] = corner_avg

    return dest

def seamless_tiling(image: np.ndarray, method: str = "spsolve") -> np.ndarray:
    """Generate a tileable image by solving a Poisson system on the inner region."""
    is_rgb = image.ndim == 3
    if not is_rgb:
        image = image[..., np.newaxis]

    h, w, c = image.shape
    result = np.zeros_like(image, dtype=np.float64)

    # The mask excludes the border pixels, so the solver only reconstructs the interior.
    mask = np.ones((h, w), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False

    for ch in range(c):
        channel = image[:, :, ch].astype(np.float64)
        
        # 1. Prepare the destination image with "fused" edges (Dirichlet boundary)
        dest = _create_boundary_image(channel)
        
        # 2. Extract the original features of the image to maintain the internal texture
        guidance = _compute_guidance(channel)
        
        # 3. Solve the system using the existing solver
        result[:, :, ch] = solve_poisson_channel(
            destination_channel=dest,
            mask=mask,
            guidance=guidance,
            method=method
        )

    # Ensure the output stays within valid limits
    result = np.clip(result, 0, 255)

    if not is_rgb:
        return result[:, :, 0]

    return result