"""
Step 2 – Node Detection  (Iterative Hough + RANSAC)
====================================================
Detects individual flowchart nodes using contour hierarchy analysis
and RANSAC shape fitting.

Key insight: In a flowchart binary image, all shapes and connecting lines
form one big connected component. The INDIVIDUAL shapes (rectangles,
diamonds) appear as **child contours** inside the outermost boundary.
We use RETR_TREE to get the hierarchy, then extract child contours as
candidate nodes.

Algorithm:
  1. Binary threshold → find contour hierarchy (RETR_TREE)
  2. Identify the outermost contour (largest area)
  3. Extract its direct children → these are individual shapes
  4. RANSAC-fit each child to rectangle / rhombus / circle
  5. Mask out detected shapes and repeat (max RANSAC_MAX_ITERATIONS)

No ML model required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from utils.ransac_shapes import best_fit

logger = logging.getLogger(__name__)


# ── Shape classification ─────────────────────────────────────

def _classify_contour(contour: np.ndarray) -> str:
    """Classify a contour as Process, Decision, Terminal, or Connector."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "Process"

    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    verts = len(approx)
    area = cv2.contourArea(contour)
    circ = (4 * np.pi * area) / (peri ** 2 + 1e-9)

    x, y, w, h = cv2.boundingRect(contour)
    fill_ratio = area / (w * h + 1e-9)

    # Circle / Connector: high circularity
    if circ > config.CIRCULARITY_THRESHOLD:
        return "Connector"

    # Diamond / Decision: low fill ratio (a diamond fills ~50% of its bounding rect)
    if fill_ratio < 0.70:
        return "Decision"

    # Rounded rectangle / Terminal: many vertices or high circularity with moderate fill
    if (verts >= 5 and circ > 0.6) or (circ > 0.72 and fill_ratio < 0.85):
        return "Terminal"

    # Parallelogram / Data: moderate fill ratio (0.70 - 0.85)
    if 0.70 <= fill_ratio < 0.85:
        return "Data"

    # Rectangle / Process: 4 vertices AND high fill ratio
    if verts == 4 and fill_ratio >= 0.75:
        return "Process"

    # Default
    return "Process"


def _mask_out_contour_dynamic(binary: np.ndarray, contour: np.ndarray,
                              dist: np.ndarray) -> np.ndarray:
    """Zero out pixels inside contour + dynamic margin based on Distance Transform."""
    # Create a thin mask of the contour
    c_mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(c_mask, [contour], -1, 255, thickness=3)
    
    # Find the maximum distance transform value along this contour
    # This gives us roughly half the local line thickness
    line_distances = dist[c_mask == 255]
    if len(line_distances) > 0:
        max_dist = np.max(line_distances)
        margin = int(np.ceil(max_dist * 2.0)) + 1
    else:
        margin = config.RANSAC_MASK_MARGIN

    out = binary.copy()
    cv2.drawContours(out, [contour], -1, 0, thickness=cv2.FILLED)
    cv2.drawContours(out, [contour], -1, 0, thickness=margin)
    return out


def _is_duplicate(bbox: list[int], existing_nodes: list[dict],
                  iou_thresh: float = 0.5) -> bool:
    """Check if bbox overlaps significantly with any existing node."""
    x, y, w, h = bbox
    for node in existing_nodes:
        ex, ey, ew, eh = node["bbox"]
        ix1, iy1 = max(x, ex), max(y, ey)
        ix2, iy2 = min(x + w, ex + ew), min(y + h, ey + eh)
        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            area1 = w * h
            area2 = ew * eh
            union = area1 + area2 - inter
            iou = inter / (union + 1e-9)
            containment = max(inter / (area1 + 1e-9), inter / (area2 + 1e-9))
            if iou > iou_thresh or containment > 0.8:
                return True
    return False

def _flag_bad_nodes(nodes: list[dict[str, Any]]) -> None:
    """Flag artifact nodes (e.g. holes formed by closed arrow circuits).

    Greedy: repeatedly flag the node with the most overlaps (>= 2) and
    exclude it from further counting.  This way real nodes that merely
    overlap ONE artifact are not dragged down with it.
    """
    def _overlaps(a: dict, b: dict) -> bool:
        x1, y1, w1, h1 = a["bbox"]
        x2, y2, w2, h2 = b["bbox"]
        ix1, iy1 = max(x1, x2), max(y1, y2)
        ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter > 0.1 * min(w1 * h1, w2 * h2)

    for node in nodes:
        node["is_bad"] = False

    active = list(nodes)
    while True:
        counts = [sum(_overlaps(a, b) for b in active if b is not a)
                  for a in active]
        if not counts or max(counts) < 2:
            break
        # A bad node is typically a giant box overlapping several real nodes;
        # among ties, flag the largest one
        worst = max(range(len(active)),
                    key=lambda i: (counts[i],
                                   active[i]["bbox"][2] * active[i]["bbox"][3]))
        active[worst]["is_bad"] = True
        active[worst]["type"] = "Bad"
        active.pop(worst)


def _remove_closed_arrowheads(binary: np.ndarray, dist: np.ndarray) -> None:
    """Find and mask out arrowhead (triangular/chevron) closed regions.
    This prevents the node detection algorithm from seeing them as flowchart nodes.
    """
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None or len(contours) == 0:
        return

    h = hierarchy[0]
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 1500 or area < 20:  # Arrowheads are small-medium regions
            continue
            
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
            
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        verts = len(approx)
        
        is_arrow = False
        if 3 <= verts <= 5:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
                
            solidity = area / hull_area
            
            # If it's a triangle, it is convex (solidity ~ 1.0)
            if verts == 3 and solidity > 0.8:
                is_arrow = True
            # If it's a chevron (4 vertices, concave), solidity is low
            elif verts == 4 and solidity < 0.85:
                is_arrow = True

        if is_arrow:
            binary[:] = _mask_out_contour_dynamic(binary, cnt, dist)
            logger.debug("Removed closed arrowhead region with area=%d", area)


def _extract_nodes_from_hierarchy(
    binary: np.ndarray,
    dist: np.ndarray,
    nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Find contour hierarchy and extract child contours as nodes.

    Returns (updated_nodes, found_any). Modifies `binary` in-place.
    """
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or len(contours) == 0:
        return nodes, False

    # Consider ALL contours in the hierarchy as candidates.  Shapes can sit
    # at any depth (e.g. inside a page border / frame), so restricting to
    # direct children of the largest root misses them.  The size, area and
    # duplicate filters below reject the page frame, text and stroke/hole
    # doubles.  Sort by area descending so the outer stroke contour of a
    # shape wins over its inner hole (near-identical bbox → duplicate).
    candidate_indices = sorted(
        range(len(contours)),
        key=lambda i: cv2.contourArea(contours[i]),
        reverse=True,
    )

    logger.debug("  %d contour candidates", len(candidate_indices))

    # Effective size limits: absolute config values if set, otherwise a
    # fraction of the image size (resolution-independent)
    img_h, img_w = binary.shape[:2]
    if config.MAX_NODE_WIDTH is not None:
        max_w = config.MAX_NODE_WIDTH * config.NODE_SIZE_TOLERANCE
    else:
        max_w = img_w * config.MAX_NODE_SIZE_RATIO
    if config.MAX_NODE_HEIGHT is not None:
        max_h = config.MAX_NODE_HEIGHT * config.NODE_SIZE_TOLERANCE
    else:
        max_h = img_h * config.MAX_NODE_SIZE_RATIO

    found_any = False
    for idx in candidate_indices:
        cnt = contours[idx]
        area = cv2.contourArea(cnt)
        if area < config.MIN_NODE_AREA:
            continue

        x, y, w, h_val = cv2.boundingRect(cnt)
        if w < 15 or h_val < 15:
            continue

        if w > max_w or h_val > max_h:
            continue

        # Real flowchart shapes (rect, diamond, ellipse, parallelogram) are
        # convex; concave background holes that wrap around other shapes
        # have low solidity
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area > 0 and area / hull_area < config.MIN_NODE_SOLIDITY:
            continue

        bbox = [int(x), int(y), int(w), int(h_val)]
        if _is_duplicate(bbox, nodes):
            continue

        # RANSAC fit
        pts = cnt.reshape(-1, 2).astype(np.float32)
        shape = best_fit(
            pts,
            n_iter=config.RANSAC_N_ITER,
            inlier_thresh=config.RANSAC_INLIER_THRESH,
        )

        contour_type = _classify_contour(cnt)
        
        if shape is not None:
            node_type = shape.node_type
            fit_score = shape.inlier_ratio
            
            if contour_type == "Decision" and node_type == "Process":
                node_type = "Decision"
        else:
            node_type = contour_type
            fit_score = 0.0

        node_id = f"n{len(nodes) + 1}"
        nodes.append({
            "id": node_id,
            "type": node_type,
            "bbox": bbox,
            "fit_score": round(fit_score, 3),
        })
        logger.debug("  Found %s %s  bbox=[%d,%d,%d,%d]  area=%d  score=%.2f",
                     node_type, node_id, x, y, w, h_val, area, fit_score)
        
        # Mask out the exact contour using Distance Transform margin
        binary[:] = _mask_out_contour_dynamic(binary, cnt, dist)
        
        found_any = True

    return nodes, found_any


# ── Main public API ──────────────────────────────────────────

def detect_nodes(
    image_path: Path,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Iteratively detect flowchart nodes using hierarchy + RANSAC.

    Parameters
    ----------
    image_path : Path
        Path to the original flowchart image.
    output_dir : Path, optional
        Where to save results.  Defaults to ``config.STEP3_DIR``.

    Returns
    -------
    list[dict]
        Each dict: ``{"id", "type", "bbox": [x, y, w, h], "fit_score"}``.
    """
    output_dir = output_dir or config.STEP2_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Try to load the preprocessed (skeletonized/preprocessed) image from Step 1
    binary_path = config.STEP1_DIR / f"{image_path.stem}_preprocessed.png"
    binary = cv2.imread(str(binary_path), cv2.IMREAD_GRAYSCALE)
    
    if binary is None:
        raise FileNotFoundError(
            f"Preprocessed image from Step 1 not found: {binary_path}. "
            "Please run preprocessing (Step 1) first."
        )

    nodes: list[dict[str, Any]] = []
    remaining = binary.copy()
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    logger.info("Preprocessing to remove closed arrowheads")
    _remove_closed_arrowheads(remaining, dist)

    logger.info("Starting iterative detection (max %d iterations)",
                config.RANSAC_MAX_ITERATIONS)

    for iteration in range(1, config.RANSAC_MAX_ITERATIONS + 1):
        logger.debug("── Iteration %d / %d ──", iteration,
                     config.RANSAC_MAX_ITERATIONS)


        nodes, found_any = _extract_nodes_from_hierarchy(remaining, dist, nodes)

        if not found_any:
            logger.debug("  No new shapes found – stopping")
            break

    # ── Sort nodes top-to-bottom, left-to-right ──────────────
    nodes.sort(key=lambda n: (n["bbox"][1], n["bbox"][0]))
    # Re-assign IDs in sorted order
    for i, node in enumerate(nodes):
        node["id"] = f"n{i + 1}"

    logger.info("Detected %d raw nodes in %d iterations", len(nodes), iteration)

    return nodes


def finetune_nodes(
    image_path: Path,
    nodes: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Fine-tune detected nodes by flagging overlapping/giant artifacts."""
    # Flag overlapping giant nodes as bad
    _flag_bad_nodes(nodes)
    return nodes
