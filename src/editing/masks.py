from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon

PathLike = Union[str, Path]


def load_mask(path: PathLike, threshold: int = 128) -> np.ndarray:
    mask = Image.open(path).convert("L")
    arr = np.asarray(mask, dtype=np.uint8)
    return arr > threshold

def polygon_to_mask(points: Iterable[Tuple[float, float]], shape: Tuple[int, int]) -> np.ndarray:
    pts = np.asarray(list(points), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError("points must contain at least 3 (x, y) pairs")
    rr, cc = sk_polygon(pts[:, 1], pts[:, 0], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask

def mask_to_indices(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask.astype(bool))

def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    h, w = mask.shape
    boundary = []
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        is_boundary = False
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or not mask[ny, nx]:
                is_boundary = True
                break
        if is_boundary:
            boundary.append((y, x))
    return np.asarray(boundary, dtype=int)