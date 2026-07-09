"""
Step 3 – Edge Detection
=======================
Detects connecting lines and arrowheads between flowchart nodes using
Probabilistic Hough Transform and arrowhead-anchored tracing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def _node_center(node: dict) -> tuple[int, int]:
    x, y, w, h = node["bbox"]
    return x + w // 2, y + h // 2


def _point_in_bbox(px, py, bbox, margin=0):
    x, y, w, h = bbox
    return (x - margin <= px <= x + w + margin and
            y - margin <= py <= y + h + margin)


def _nearest_node(px, py, nodes, max_dist):
    """Find the nearest node to point (px, py) within max_dist (to the bounding box)."""
    best, best_d = None, max_dist
    for node in nodes:
        x, y, w, h = node["bbox"]
        # Distance from point to rectangle
        dx = max(x - px, 0, px - (x + w))
        dy = max(y - py, 0, py - (y + h))
        d = np.hypot(dx, dy)
        
        if d < best_d:
            best, best_d = node, d
    return best


def _detect_arrowheads(binary: np.ndarray, nodes: list[dict]) -> list[dict]:
    """Find arrowhead-like triangular contours not inside any node bbox."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrows = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > 800:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if 3 <= len(approx) <= 5:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Skip if inside a node
            inside = any(_point_in_bbox(cx, cy, n["bbox"], 5) for n in nodes)
            if not inside:
                arrows.append({"cx": cx, "cy": cy, "contour": cnt})
    return arrows


def detect_edges(
    image_path: Path,
    nodes: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Detect connecting edges using Pairwise ROI Pathfinding."""
    
    # Exclude bad (overlapping/giant) nodes from edge detection entirely
    valid_nodes = [n for n in nodes if not n.get("is_bad", False)]

    output_dir = output_dir or config.STEP3_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Try to load the low-C preprocessed skeleton image from Step 1
    binary_path = config.STEP1_DIR / f"{image_path.stem}_preprocessed_low_c.png"
    binary = cv2.imread(str(binary_path), cv2.IMREAD_GRAYSCALE)
    
    if binary is None:
        raise FileNotFoundError(
            f"Preprocessed low-C skeleton from Step 1 not found: {binary_path}. "
            "Please run preprocessing (Step 1) first."
        )
    
    # Mask out ALL valid node interiors to isolate lines
    mask = binary.copy()
    for node in valid_nodes:
        x, y, w, h = node["bbox"]
        pad = 3
        mask[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad] = 0

    # Thick (non-skeleton) high-C binary: used to measure ink density at the
    # two ends of a connector — the arrowhead end is much denser
    thick_path = config.STEP1_DIR / f"{image_path.stem}_preprocessed.png"
    thick = cv2.imread(str(thick_path), cv2.IMREAD_GRAYSCALE)
    if thick is None:
        thick = binary
    thick_mask = thick.copy()
    for node in valid_nodes:
        x, y, w, h = node["bbox"]
        pad = 3
        thick_mask[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad] = 0

    arrowheads = _detect_arrowheads(binary, valid_nodes)
    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    debug_img = img.copy()
    used_contours = []
    snap = 15

    def _touch_point(cnt: np.ndarray, node: dict) -> np.ndarray:
        """Contour point closest to the node's bbox."""
        pts = cnt.reshape(-1, 2)
        x, y, w, h = node["bbox"]
        dx = np.maximum(x - pts[:, 0], 0) + np.maximum(pts[:, 0] - (x + w), 0)
        dy = np.maximum(y - pts[:, 1], 0) + np.maximum(pts[:, 1] - (y + h), 0)
        return pts[int(np.argmin(np.hypot(dx, dy)))]

    def _end_ink(pt: np.ndarray, r: int = 14) -> int:
        """Ink pixels in the thick binary around a connector end."""
        x, y = int(pt[0]), int(pt[1])
        roi = thick_mask[max(0, y - r):y + r, max(0, x - r):x + r]
        return int(np.count_nonzero(roi))

    def _add_edge(n_a: dict, n_b: dict, cnt: np.ndarray) -> None:
        """Record an edge; direction from end ink density (arrowhead end)."""
        src_node, tgt_node = n_a, n_b
        ink_a = _end_ink(_touch_point(cnt, n_a))
        ink_b = _end_ink(_touch_point(cnt, n_b))
        if ink_a > 1.3 * ink_b:
            src_node, tgt_node = n_b, n_a   # arrowhead at n_a → points to n_a
        elif ink_b > 1.3 * ink_a:
            src_node, tgt_node = n_a, n_b

        pair = (src_node["id"], tgt_node["id"])
        rpair = (tgt_node["id"], src_node["id"])
        if pair in seen_pairs or rpair in seen_pairs:
            return
        seen_pairs.add(pair)
        cv2.drawContours(debug_img, [cnt], -1, (0, 165, 255), 3)
        used_contours.append(cnt)
        approx = cv2.approxPolyDP(cnt, 2.0, closed=True)
        edges.append({
            "source": src_node["id"],
            "target": tgt_node["id"],
            "label": "",
            "path": approx.reshape(-1, 2).tolist(),
        })

    def _touched_nodes(cnt: np.ndarray) -> list[dict]:
        pts = cnt.reshape(-1, 2)
        touched = []
        for n in valid_nodes:
            x, y, w, h = n["bbox"]
            if np.any((pts[:, 0] >= x - snap) & (pts[:, 0] <= x + w + snap) &
                      (pts[:, 1] >= y - snap) & (pts[:, 1] <= y + h + snap)):
                touched.append(n)
        return touched

    # ── Pass 1: global connected components ─────────────────────────────
    # A component of the node-masked skeleton that touches EXACTLY two
    # nodes is an edge between them — no ROI clipping, so curved arcs that
    # bulge far outside the nodes' union bbox are handled too.
    # RETR_LIST, not RETR_EXTERNAL: components inside another component's
    # hole (e.g. everything inside a page border) are nested in the contour
    # hierarchy and RETR_EXTERNAL would hide them.
    global_contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)
    for cnt in global_contours:
        _, _, cw, ch = cv2.boundingRect(cnt)
        if cw < 10 and ch < 10:
            continue  # speck
        touched = _touched_nodes(cnt)
        if len(touched) == 2:
            _add_edge(touched[0], touched[1], cnt)
    logger.info("Pass 1 (global components): %d edges", len(edges))

    # ── Pass 2: pairwise ROI pathfinding ─────────────────────────────────
    # Fallback for components touching > 2 nodes (crossing / merged lines):
    # clip to the pair's ROI so the crossing splits into separate contours.

    # Max node-pair gap scales with image size (long arcs on hi-res scans)
    max_pair_dist = max(600, 0.4 * (img.shape[0] + img.shape[1]))

    for i in range(len(valid_nodes)):
        for j in range(i + 1, len(valid_nodes)):
            n1 = valid_nodes[i]
            n2 = valid_nodes[j]

            if (n1["id"], n2["id"]) in seen_pairs or (n2["id"], n1["id"]) in seen_pairs:
                continue

            x1, y1, w1, h1 = n1["bbox"]
            x2, y2, w2, h2 = n2["bbox"]

            # 1. Check if they are close enough (có gần nhau không)
            dx = max(0, max(x1 - (x2 + w2), x2 - (x1 + w1)))
            dy = max(0, max(y1 - (y2 + h2), y2 - (y1 + h1)))
            dist = np.hypot(dx, dy)
            if dist > max_pair_dist:  # Ignore nodes that are extremely far apart
                continue

            # 2. Extract the ROI (ảnh nhỏ giữa 2 node)
            pad = 5
            rx1 = max(0, min(x1, x2) - pad)
            ry1 = max(0, min(y1, y2) - pad)
            rx2 = min(img.shape[1], max(x1 + w1, x2 + w2) + pad)
            ry2 = min(img.shape[0], max(y1 + h1, y2 + h2) + pad)

            roi = mask[ry1:ry2, rx1:rx2]

            # 3. Find line contours in this ROI
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            path_found = False
            for cnt in contours:
                # Shift contour coordinates back to global image space
                global_cnt = cnt + np.array([rx1, ry1])
                
                # Check if this contour touches BOTH n1 and n2
                # We do this by checking the bounding box of the contour
                # Or simply by drawing the contour and checking intersection
                
                # Fast check: does the contour bounding rect intersect both node's expanded bounds?
                cx, cy, cw, ch = cv2.boundingRect(global_cnt)
                
                # Expanded bounds for n1 and n2 (snap distance)
                snap = 15
                n1_rect = (x1 - snap, y1 - snap, w1 + 2*snap, h1 + 2*snap)
                n2_rect = (x2 - snap, y2 - snap, w2 + 2*snap, h2 + 2*snap)
                
                def rect_intersect(r1, r2):
                    return not (r1[0] > r2[0] + r2[2] or r1[0] + r1[2] < r2[0] or
                                r1[1] > r2[1] + r2[3] or r1[1] + r1[3] < r2[1])

                if rect_intersect((cx, cy, cw, ch), n1_rect) and rect_intersect((cx, cy, cw, ch), n2_rect):
                    # To be perfectly accurate, check if the contour points actually fall inside the snap areas
                    touches_n1 = any(n1_rect[0] <= pt[0][0] <= n1_rect[0]+n1_rect[2] and n1_rect[1] <= pt[0][1] <= n1_rect[1]+n1_rect[3] for pt in global_cnt)
                    touches_n2 = any(n2_rect[0] <= pt[0][0] <= n2_rect[0]+n2_rect[2] and n2_rect[1] <= pt[0][1] <= n2_rect[1]+n2_rect[3] for pt in global_cnt)
                    
                    if touches_n1 and touches_n2:
                        path_found = True
                        break

            if path_found:
                _add_edge(n1, n2, global_cnt)

    # Save intermediate debug images
    # 1. Save the masked skeleton (which isolates the edges/lines)
    mask_path = output_dir / f"{image_path.stem}_edge_mask.png"
    cv2.imwrite(str(mask_path), mask)
    logger.info("Saved edge mask skeleton → %s", mask_path)

    # 2. Save the debug visualization of traced edges
    edges_vis_path = output_dir / f"{image_path.stem}_edges_traced.png"
    cv2.imwrite(str(edges_vis_path), debug_img)
    logger.info("Saved traced edges visualisation → %s", edges_vis_path)

    logger.info("Detected %d raw edges", len(edges))
    return edges, used_contours


def finetune_edges(
    image_path: Path,
    nodes: list[dict[str, Any]],
    used_contours: list[np.ndarray],
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Fine-tune edge detection by finding leftover unconnected lines as Bad Nodes."""
    # Try to load the low-C preprocessed skeleton image from Step 1
    binary_path = config.STEP1_DIR / f"{image_path.stem}_preprocessed_low_c.png"
    binary = cv2.imread(str(binary_path), cv2.IMREAD_GRAYSCALE)
    
    if binary is None:
        raise FileNotFoundError(
            f"Preprocessed low-C skeleton from Step 1 not found: {binary_path}. "
            "Please run preprocessing (Step 1) first."
        )
    
    # Mask out ALL valid node interiors
    valid_nodes = [n for n in nodes if not n.get("is_bad", False)]
    mask = binary.copy()
    for node in valid_nodes:
        x, y, w, h = node["bbox"]
        pad = 3
        mask[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad] = 0

    # Fine-tune: Find leftover unconnected lines and flag them as Bad Nodes
    for cnt in used_contours:
        cv2.drawContours(mask, [cnt], -1, 0, thickness=5)
        
    rem_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bad_count = 0
    for cnt in rem_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 40 or h > 40:  # Large enough to be considered a leftover artifact/line
            new_id = f"n{len(nodes) + 1}"
            nodes.append({
                "id": new_id,
                "type": "Bad",
                "text": "",
                "bbox": [x, y, w, h],
                "is_bad": True
            })
            bad_count += 1
            
    if bad_count > 0:
        logger.info("Fine-tune: Flagged %d unconnected leftover lines as Bad nodes", bad_count)
    
    return nodes
