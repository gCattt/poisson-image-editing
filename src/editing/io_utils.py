"""Convenience helpers for image loading, saving, and datatype conversion.

These small utilities keep the effect modules focused on the editing logic rather
than repeated boilerplate around file I/O and numeric conversion.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def load_image(path: PathLike) -> np.ndarray:
    """Load an image from disk and return it as a uint8 NumPy array."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

def save_image(arr: np.ndarray, path: PathLike) -> None:
    """Persist an image array to disk, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)

    # Clamp the values to the displayable range before writing to disk.
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def to_float(arr: np.ndarray) -> np.ndarray:
    """Convert an image-style array to float64 for numerical processing."""
    return np.asarray(arr, dtype=np.float64)

def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Round and convert an array to the uint8 image format used by PIL."""
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)