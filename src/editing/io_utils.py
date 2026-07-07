from __future__ import annotations
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def load_image(path: PathLike) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

def save_image(arr: np.ndarray, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def to_float(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)

def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)