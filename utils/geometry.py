"""
Geometry helpers for shape analysis.
"""

from __future__ import annotations

import cv2
import numpy as np


def count_vertices(contour: np.ndarray, epsilon_ratio: float = 0.02) -> int:
    """Approximate contour to polygon and return vertex count."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    return len(approx)


def circularity(contour: np.ndarray) -> float:
    """Compute circularity = 4π·area / perimeter²."""
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    return (4 * np.pi * area) / (peri ** 2)


def aspect_ratio(bbox: list[int]) -> float:
    """Width/height ratio from [x, y, w, h]."""
    _, _, w, h = bbox
    return w / h if h > 0 else 0.0


def bbox_center(bbox: list[int]) -> tuple[int, int]:
    """Return centre point of [x, y, w, h]."""
    x, y, w, h = bbox
    return x + w // 2, y + h // 2


def bbox_distance(b1: list[int], b2: list[int]) -> float:
    """Euclidean distance between two bbox centres."""
    c1, c2 = bbox_center(b1), bbox_center(b2)
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
