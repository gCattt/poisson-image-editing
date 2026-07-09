from __future__ import annotations
from typing import Tuple
import numpy as np


def image_gradient(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = np.asarray(img, dtype=np.float64)
    gx = np.zeros_like(img, dtype=np.float64)
    gy = np.zeros_like(img, dtype=np.float64)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    return gx, gy

def divergence(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    div = np.zeros_like(vx, dtype=np.float64)
    div[:, :-1] += vx[:, :-1]
    div[:, 1:] -= vx[:, :-1]
    div[:-1, :] += vy[:-1, :]
    div[1:, :] -= vy[:-1, :]
    return div