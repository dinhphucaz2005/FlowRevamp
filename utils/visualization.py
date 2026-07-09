"""
Visualisation – draw detection results on the original image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def append_legend(img: "np.ndarray",
                  items: list[tuple[str, str, tuple[int, int, int]]],
                  ) -> "np.ndarray":
    """Append a white legend strip below the image explaining the overlay.

    Each item is ``(glyph, label, color)`` where glyph is one of
    ``"rect"``, ``"line"``, ``"thickline"``, ``"dot"``, ``"target"``.
    Items wrap onto extra rows when the image is narrow.
    """
    h, w = img.shape[:2]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    glyph_w, gap, margin, row_h = 30, 20, 10, 30

    # Lay out items into rows first to know the strip height
    rows: list[list[tuple[int, tuple[str, str, tuple[int, int, int]]]]] = [[]]
    x = margin
    for item in items:
        (tw, _), _ = cv2.getTextSize(item[1], font, scale, thick)
        item_w = glyph_w + tw + gap
        if rows[-1] and x + item_w > w - margin:
            rows.append([])
            x = margin
        rows[-1].append((x, item))
        x += item_w

    strip_h = row_h * len(rows) + 10
    canvas = np.full((h + strip_h, w, 3), 255, dtype=np.uint8)
    canvas[:h] = img
    cv2.line(canvas, (0, h), (w, h), (180, 180, 180), 1)

    for row_idx, row in enumerate(rows):
        y = h + 5 + row_h * row_idx + row_h // 2
        for x, (glyph, label, color) in row:
            if glyph == "rect":
                cv2.rectangle(canvas, (x, y - 8), (x + 22, y + 8), color, 2)
            elif glyph == "diamond":
                pts = np.array([
                    [x + 11, y - 9],
                    [x + 22, y],
                    [x + 11, y + 9],
                    [x, y]
                ], dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
            elif glyph == "circle":
                cv2.circle(canvas, (x + 11, y), 8, color, 2)
            elif glyph == "ellipse":
                cv2.ellipse(canvas, (x + 11, y), (11, 7), 0, 0, 360, color, 2)
            elif glyph == "parallelogram":
                skew = 5
                pts = np.array([
                    [x + skew, y - 8],
                    [x + 22, y - 8],
                    [x + 22 - skew, y + 8],
                    [x, y + 8]
                ], dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
            elif glyph == "line":
                cv2.line(canvas, (x, y), (x + 22, y), color, 2)
            elif glyph == "thickline":
                cv2.line(canvas, (x, y), (x + 22, y), color, 4)
            elif glyph == "dot":
                cv2.circle(canvas, (x + 11, y), 6, color, -1)
            elif glyph == "target":
                cv2.circle(canvas, (x + 11, y), 6, (0, 0, 255), -1)
                cv2.circle(canvas, (x + 11, y), 8, color, 2)
            cv2.putText(canvas, label, (x + glyph_w, y + 5), font, scale,
                        (40, 40, 40), thick, cv2.LINE_AA)

    return canvas


def draw_results(image_path: Path, nodes: list[dict[str, Any]],
                 edges: list[dict[str, Any]],
                 output_dir: Path | None = None) -> Path:
    """Overlay nodes and edges on the original image, and also draw on a blank white canvas."""
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    # Read color palette definitions dynamically from config
    NODE_COLOR = getattr(config, "VIS_NODE_COLOR", (39, 227, 145))
    EDGE_COLOR = getattr(config, "VIS_EDGE_COLOR", (255, 0, 255))
    BAD_NODE_COLOR = getattr(config, "VIS_BAD_NODE_COLOR", (0, 0, 255))
    TEXT_COLOR = getattr(config, "VIS_TEXT_COLOR", (255, 255, 255))
    TEXT_BG_COLOR = getattr(config, "VIS_TEXT_BG_COLOR", (30, 30, 30))
    font_scale = getattr(config, "VIS_FONT_SCALE", 0.55)

    node_map = {n["id"]: n for n in nodes}

    # Create a blank white canvas of the same shape
    img_blank = np.full_like(img, 255)

    # Overlay layers for semi-transparent label boxes
    overlay = img.copy()
    overlay_blank = img_blank.copy()

    # Helper function to draw node shapes consistently on any canvas
    def draw_node_shape(canvas, node, color, thickness):
        x, y, w, h = node["bbox"]
        node_type = node.get("type", "Process")
        is_bad = node.get("is_bad", False)
        
        if is_bad:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness - 1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x, y), (x + w, y + h), color, 1)
            cv2.line(canvas, (x + w, y), (x, y + h), color, 1)
        elif node_type == "Decision":
            pts = np.array([
                [x + w // 2, y],
                [x + w, y + h // 2],
                [x + w // 2, y + h],
                [x, y + h // 2]
            ], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        elif node_type == "Connector":
            cx, cy = x + w // 2, y + h // 2
            r = min(w, h) // 2
            cv2.circle(canvas, (cx, cy), r, color, thickness=thickness, lineType=cv2.LINE_AA)
        elif node_type == "Terminal":
            cx, cy = x + w // 2, y + h // 2
            r_w = max(w // 2, 1)
            r_h = max(h // 2, 1)
            cv2.ellipse(canvas, (cx, cy), (r_w, r_h), 0, 0, 360, color, thickness=thickness, lineType=cv2.LINE_AA)
        elif node_type == "Data":
            skew = int(w * 0.18)
            pts = np.array([
                [x + skew, y],
                [x + w, y],
                [x + w - skew, y + h],
                [x, y + h]
            ], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        else:  # Process / Default
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)

    # Draw nodes on both canvases
    thick = config.VIS_LINE_THICKNESS + 3 if hasattr(config, "VIS_LINE_THICKNESS") else 5
    for node in nodes:
        x, y, w, h = node["bbox"]
        is_bad = node.get("is_bad", False)
        color = BAD_NODE_COLOR if is_bad else NODE_COLOR

        draw_node_shape(img, node, color, thick)
        draw_node_shape(img_blank, node, color, thick)

        # Draw elegant semi-transparent label boxes
        label = f'{node["id"]}:{node.get("type", "Process")}'
        text = node.get("text", "")
        if text:
            label += f' "{text[:15]}"'
            
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        
        # Coordinates for text background box
        bx1, by1 = x, max(y - th - 12, 0)
        bx2, by2 = x + tw + 8, max(y, th + 12)
        
        # Draw background onto overlays
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), TEXT_BG_COLOR, -1)
        cv2.rectangle(overlay_blank, (bx1, by1), (bx2, by2), TEXT_BG_COLOR, -1)
        
        # Put text on actual images
        cv2.putText(img, label, (x + 4, by2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img_blank, label, (x + 4, by2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)

    # Blend overlay for translucent background boxes
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.addWeighted(overlay_blank, 0.7, img_blank, 0.3, 0, img_blank)

    # Helper function to draw arrowheads
    def draw_arrowhead(canvas, pt_to, pt_from, color, size=18):
        v = np.array(pt_to) - np.array(pt_from)
        norm = np.linalg.norm(v)
        if norm == 0:
            return
        v = v / norm
        
        # 30-degree rotation matrices
        angle = np.pi / 6
        c, s = np.cos(angle), np.sin(angle)
        rot1 = np.array([[c, -s], [s, c]])
        rot2 = np.array([[c, s], [-s, c]])
        
        p1 = np.array(pt_to) - size * (rot1 @ v)
        p2 = np.array(pt_to) - size * (rot2 @ v)
        
        # Draw filled triangle for arrowhead
        arrow_pts = np.array([pt_to, p1, p2], dtype=np.int32)
        cv2.fillPoly(canvas, [arrow_pts], color)

    # Draw edges on both canvases
    for edge in edges:
        src = node_map.get(edge["source"])
        tgt = node_map.get(edge["target"])
        if not src or not tgt:
            continue
            
        scx, scy = src["bbox"][0] + src["bbox"][2] // 2, src["bbox"][1] + src["bbox"][3] // 2
        tcx, tcy = tgt["bbox"][0] + tgt["bbox"][2] // 2, tgt["bbox"][1] + tgt["bbox"][3] // 2

        # Draw the exact ink path if available
        if "path" in edge and edge["path"]:
            path_pts = np.array(edge["path"], dtype=np.int32)
            cv2.fillPoly(img, [path_pts], EDGE_COLOR)
            cv2.fillPoly(img_blank, [path_pts], EDGE_COLOR)
            
            # Find target node border contact point
            distances = np.hypot(path_pts[:, 0] - tcx, path_pts[:, 1] - tcy)
            idx_to = np.argmin(distances)
            pt_to = path_pts[idx_to]
            
            # Find point slightly upstream for vector
            pt_from = None
            for step in [5, 10, 15, 20]:
                upstream_idx = idx_to - step if idx_to - step >= 0 else idx_to + step
                if upstream_idx < len(path_pts):
                    test_pt = path_pts[upstream_idx]
                    if np.hypot(test_pt[0] - pt_to[0], test_pt[1] - pt_to[1]) > 8:
                        pt_from = test_pt
                        break
            if pt_from is None:
                pt_from = np.array([scx, scy])

            # Draw arrowhead and source dot on both canvases
            draw_arrowhead(img, pt_to, pt_from, EDGE_COLOR)
            draw_arrowhead(img_blank, pt_to, pt_from, EDGE_COLOR)
            
            distances_src = np.hypot(path_pts[:, 0] - scx, path_pts[:, 1] - scy)
            pt_src = path_pts[np.argmin(distances_src)]
            cv2.circle(img, tuple(pt_src), 6, EDGE_COLOR, -1, lineType=cv2.LINE_AA)
            cv2.circle(img_blank, tuple(pt_src), 6, EDGE_COLOR, -1, lineType=cv2.LINE_AA)
            
        else:
            # Fallback direct straight line drawing
            v = np.array([tcx, tcy]) - np.array([scx, scy])
            norm = np.linalg.norm(v)
            if norm > 0:
                r_tgt = min(tgt["bbox"][2], tgt["bbox"][3]) // 2
                pt_to = np.array([tcx, tcy]) - r_tgt * v / norm
                pt_from = pt_to - 15 * v / norm
                
                cv2.line(img, (scx, scy), tuple(pt_to.astype(int)), EDGE_COLOR, thick, cv2.LINE_AA)
                cv2.line(img_blank, (scx, scy), tuple(pt_to.astype(int)), EDGE_COLOR, thick, cv2.LINE_AA)
                
                draw_arrowhead(img, pt_to.astype(int), pt_from.astype(int), EDGE_COLOR)
                draw_arrowhead(img_blank, pt_to.astype(int), pt_from.astype(int), EDGE_COLOR)

    # Append legend to both images
    img = append_legend(img, [
        ("rect", "Process", NODE_COLOR),
        ("diamond", "Decision", NODE_COLOR),
        ("ellipse", "Terminal", NODE_COLOR),
        ("circle", "Connector", NODE_COLOR),
        ("parallelogram", "Data (I/O)", NODE_COLOR),
        ("rect", "Bad node (artifact)", BAD_NODE_COLOR),
        ("thickline", "Edge (traced ink path)", EDGE_COLOR),
        ("target", "Edge target (arrowhead)", EDGE_COLOR),
    ])
    img_blank = append_legend(img_blank, [
        ("rect", "Process", NODE_COLOR),
        ("diamond", "Decision", NODE_COLOR),
        ("ellipse", "Terminal", NODE_COLOR),
        ("circle", "Connector", NODE_COLOR),
        ("parallelogram", "Data (I/O)", NODE_COLOR),
        ("rect", "Bad node (artifact)", BAD_NODE_COLOR),
        ("thickline", "Edge (traced ink path)", EDGE_COLOR),
        ("target", "Edge target (arrowhead)", EDGE_COLOR),
    ])

    out_path = output_dir / f"{image_path.stem}_result.png"
    out_path_blank = output_dir / f"{image_path.stem}_reconstructed.png"
    
    cv2.imwrite(str(out_path), img)
    cv2.imwrite(str(out_path_blank), img_blank)
    
    logger.info("Visualisation saved (Overlay) → %s", out_path)
    logger.info("Visualisation saved (Reconstructed Blank) → %s", out_path_blank)
    
    return out_path
