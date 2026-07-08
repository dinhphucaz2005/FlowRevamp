"""
RANSAC Shape Fitting
====================
Custom RANSAC implementations for fitting geometric primitives
(rectangle, rhombus, circle/ellipse) to 2D contour points.

Each model follows the protocol:
  - estimate(points)   → fit from a minimal sample
  - residuals(points)  → distance from each point to the fitted shape
  - is_valid()         → sanity-check on fitted parameters
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Utility ──────────────────────────────────────────────────

def _point_to_segment_dist(pt: np.ndarray,
                           a: np.ndarray, b: np.ndarray) -> float:
    """Shortest distance from *pt* to line segment [a, b]."""
    ab = b - a
    t = np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2-D vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


# ── Data classes for fitted shapes ───────────────────────────

@dataclass
class FittedRectangle:
    center: tuple[float, float]
    width: float
    height: float
    angle: float          # degrees
    corners: np.ndarray   # (4, 2)
    inlier_ratio: float = 0.0

    @property
    def node_type(self) -> str:
        return "Process"

    @property
    def bbox(self) -> list[int]:
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.int32))
        return [int(x), int(y), int(w), int(h)]


@dataclass
class FittedRhombus:
    center: tuple[float, float]
    half_diag_h: float    # horizontal half-diagonal
    half_diag_v: float    # vertical half-diagonal
    angle: float
    corners: np.ndarray   # (4, 2)
    inlier_ratio: float = 0.0

    @property
    def node_type(self) -> str:
        return "Decision"

    @property
    def bbox(self) -> list[int]:
        x, y, w, h = cv2.boundingRect(self.corners.astype(np.int32))
        return [int(x), int(y), int(w), int(h)]


@dataclass
class FittedCircle:
    center: tuple[float, float]
    radius: float
    inlier_ratio: float = 0.0

    @property
    def node_type(self) -> str:
        return "Connector"

    @property
    def bbox(self) -> list[int]:
        cx, cy = self.center
        r = self.radius
        return [int(cx - r), int(cy - r), int(2 * r), int(2 * r)]


@dataclass
class FittedEllipse:
    center: tuple[float, float]
    axes: tuple[float, float]   # (major, minor) half-lengths
    angle: float
    inlier_ratio: float = 0.0

    @property
    def node_type(self) -> str:
        ratio = min(self.axes) / (max(self.axes) + 1e-9)
        return "Connector" if ratio > 0.85 else "Terminal"

    @property
    def bbox(self) -> list[int]:
        cx, cy = self.center
        a, b = self.axes
        r = max(a, b)
        return [int(cx - r), int(cy - r), int(2 * r), int(2 * r)]


# ── RANSAC: Rectangle ────────────────────────────────────────

def ransac_fit_rectangle(
    points: np.ndarray,
    n_iter: int = 200,
    inlier_thresh: float = 3.0,
    min_inlier_ratio: float = 0.50,
) -> FittedRectangle | None:
    """Fit a rectangle to *points* (N×2).
    Since child contours are usually clean, we just fit minAreaRect on all points
    and verify if it's a good fit, acting as a deterministic model evaluation.
    """
    if len(points) < 5:
        return None

    pts = points.reshape(-1, 2).astype(np.float32)
    
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    if w < 10 or h < 10:
        return None

    # Process blocks are axis-aligned rectangles, so their axis-aligned bounding box
    # should be tightly filled. Diamonds have a fill ratio around 0.5.
    bx, by, bw, bh = cv2.boundingRect(pts)
    area = cv2.contourArea(cv2.convexHull(pts))
    if area / (bw * bh + 1e-9) < 0.75:
        return None

    corners = cv2.boxPoints(rect)  # (4, 2)

    # Compute residuals for all points
    dists = _polygon_residuals(pts, corners)
    inliers = np.sum(dists < inlier_thresh)
    ratio = inliers / len(pts)

    if ratio >= min_inlier_ratio:
        # Check rectangularity: angles should be ~90°
        c = corners
        angles = []
        for i in range(4):
            v1 = c[(i + 1) % 4] - c[i]
            v2 = c[(i - 1) % 4] - c[i]
            angles.append(_angle_between(v1, v2))
        angle_dev = np.mean([abs(a - 90) for a in angles])
        if angle_dev > 20:
            return None
            
        return FittedRectangle(
            center=(cx, cy), width=w, height=h, angle=angle,
            corners=corners, inlier_ratio=ratio,
        )
    return None


# ── RANSAC: Rhombus (Diamond) ────────────────────────────────

def ransac_fit_rhombus(
    points: np.ndarray,
    n_iter: int = 200,
    inlier_thresh: float = 3.0,
    min_inlier_ratio: float = 0.50,
) -> FittedRhombus | None:
    """Fit a rhombus (diamond) to *points*.
    Flowchart decision nodes are typically axis-aligned diamonds.
    We fit an axis-aligned diamond using the bounding box midpoints.
    """
    if len(points) < 5:
        return None

    pts = points.reshape(-1, 2).astype(np.float32)
    
    x, y, w, h = cv2.boundingRect(pts)
    if w < 10 or h < 10:
        return None

    # Decision blocks (diamonds) have an area roughly half of their bounding box
    area = cv2.contourArea(cv2.convexHull(pts))
    if area / (w * h + 1e-9) > 0.75:
        return None

    cx, cy = x + w / 2, y + h / 2
    half_w, half_h = w / 2, h / 2
    
    corners = np.array([
        [cx, cy - half_h],
        [cx + half_w, cy],
        [cx, cy + half_h],
        [cx - half_w, cy],
    ], dtype=np.float32)

    dists = _polygon_residuals(pts, corners)
    inliers = np.sum(dists < inlier_thresh)
    ratio = inliers / len(pts)

    if ratio >= min_inlier_ratio:
        return FittedRhombus(
            center=(cx, cy), half_diag_h=half_w, half_diag_v=half_h,
            angle=0.0, corners=corners, inlier_ratio=ratio,
        )
    return None


# ── RANSAC: Circle ───────────────────────────────────────────

def ransac_fit_circle(
    points: np.ndarray,
    n_iter: int = 200,
    inlier_thresh: float = 3.0,
    min_inlier_ratio: float = 0.60,
) -> FittedCircle | None:
    """Fit a circle to *points* deterministically using minEnclosingCircle."""
    if len(points) < 4:
        return None

    pts = points.reshape(-1, 2).astype(np.float64)
    
    (cx, cy), radius = cv2.minEnclosingCircle(pts.astype(np.float32))
    
    if radius < 5 or radius > min(pts.max(axis=0) - pts.min(axis=0)) * 0.8:
        return None

    dists = np.abs(np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2) - radius)
    inliers = np.sum(dists < inlier_thresh)
    ratio = inliers / len(pts)

    if ratio >= min_inlier_ratio:
        return FittedCircle(
            center=(cx, cy), radius=radius, inlier_ratio=ratio,
        )
    return None


# ── Helpers ──────────────────────────────────────────────────

def _polygon_residuals(pts: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest edge of a polygon."""
    n_edges = len(corners)
    dists = np.full(len(pts), np.inf)
    for i in range(n_edges):
        a = corners[i]
        b = corners[(i + 1) % n_edges]
        for j, pt in enumerate(pts):
            d = _point_to_segment_dist(pt, a, b)
            if d < dists[j]:
                dists[j] = d
    return dists


def best_fit(
    contour_points: np.ndarray,
    n_iter: int = 200,
    inlier_thresh: float = 3.0,
) -> FittedRectangle | FittedRhombus | FittedCircle | None:
    """Try all shape models on *contour_points*, return the best fit."""
    pts = contour_points.reshape(-1, 2).astype(np.float32)

    candidates: list[Any] = []

    circ = ransac_fit_circle(pts, n_iter, inlier_thresh)
    if circ:
        candidates.append(circ)

    rect = ransac_fit_rectangle(pts, n_iter, inlier_thresh)
    if rect:
        candidates.append(rect)

    rhomb = ransac_fit_rhombus(pts, n_iter, inlier_thresh)
    if rhomb:
        candidates.append(rhomb)

    if not candidates:
        return None

    # Pick the one with the highest inlier ratio
    return max(candidates, key=lambda c: c.inlier_ratio)
