from __future__ import annotations
import numpy as np

from ..solver import solve_poisson_channel


def _compute_guidance(img_channel: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """
    Calculate the vector field (g_q - g_p) for the 4 directions.
    This logic aligns perfectly with _NEIGHBORS_4 in system.py.
    """
    guidance = {}
    
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        v = np.zeros_like(img_channel, dtype=np.float64)
        
        # Calculate finite differences: Current Pixel (p) - Neighbor Pixel (q)
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
    """
    Apply the boundary condition by calculating the average of opposite edges.
    """
    dest = img_channel.copy().astype(np.float64)
    
    # Calculate the averages using the original values to avoid sequential overwriting
    ns_avg = 0.5 * (img_channel[0, :] + img_channel[-1, :])
    ew_avg = 0.5 * (img_channel[:, 0] + img_channel[:, -1])
    
    # For the corners, we take the average of all four corners to avoid asymmetry
    corner_avg = 0.25 * (
        img_channel[0, 0] + img_channel[0, -1] + 
        img_channel[-1, 0] + img_channel[-1, -1]
    )
    
    # Assign the averaged values to the edges
    dest[0, :] = ns_avg
    dest[-1, :] = ns_avg
    dest[:, 0] = ew_avg
    dest[:, -1] = ew_avg
    
    # Force the exact angles
    dest[0, 0] = corner_avg
    dest[0, -1] = corner_avg
    dest[-1, 0] = corner_avg
    dest[-1, -1] = corner_avg
    
    return dest

def seamless_tiling(image: np.ndarray, method: str = "spsolve") -> np.ndarray:
    """
    Generate a seamless (tileable) image using Poisson editing.
    Supports grayscale (2D) and color (3D) images.
    """
    is_rgb = image.ndim == 3
    if not is_rgb:
        image = image[..., np.newaxis]

    h, w, c = image.shape
    result = np.zeros_like(image, dtype=np.float64)

    # The mask defines the internal region where the Poisson equation is solved.
    # The edges of the image (1 pixel) will remain False (these are our boundary conditions).
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