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
    """Overlay nodes and edges on the original image. Returns saved path."""
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    node_map = {n["id"]: n for n in nodes}

    # Draw nodes
    for node in nodes:
        x, y, w, h = node["bbox"]
        
        is_bad = node.get("is_bad", False)
        color = (0, 0, 255) if is_bad else config.VIS_NODE_COLOR
        thick = config.VIS_LINE_THICKNESS
        
        node_type = node.get("type", "Process")
        if is_bad:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        elif node_type == "Decision":
            pts = np.array([
                [x + w // 2, y],
                [x + w, y + h // 2],
                [x + w // 2, y + h],
                [x, y + h // 2]
            ], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thick)
        elif node_type == "Connector":
            cx, cy = x + w // 2, y + h // 2
            r = min(w, h) // 2
            cv2.circle(img, (cx, cy), r, color, thickness=thick)
        elif node_type == "Terminal":
            cx, cy = x + w // 2, y + h // 2
            cv2.ellipse(img, (cx, cy), (w // 2, h // 2), 0, 0, 360, color, thickness=thick)
        elif node_type == "Data":
            skew = int(w * 0.18)
            pts = np.array([
                [x + skew, y],
                [x + w, y],
                [x + w - skew, y + h],
                [x, y + h]
            ], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thick)
        else:  # Process / Default
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)

        label = f'{node["id"]}:{node_type}'
        text = node.get("text", "")
        if text:
            label += f' "{text[:20]}"'
        cv2.putText(img, label, (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, config.VIS_FONT_SCALE,
                    config.VIS_TEXT_COLOR, 1, cv2.LINE_AA)

    # ── Draw exact edge paths ───────────────────────────────────
    for edge in edges:
        src = node_map.get(edge["source"])
        tgt = node_map.get(edge["target"])
        if not src or not tgt:
            continue
            
        # Draw the exact ink path if available
        if "path" in edge and edge["path"]:
            pts = np.array(edge["path"], dtype=np.int32)
            # Fill the exact line contour for a beautiful "neon wire" effect
            cv2.fillPoly(img, [pts], config.VIS_EDGE_COLOR)
            
            # Draw a circle indicator at the Target node to show direction
            tx = tgt["bbox"][0] + tgt["bbox"][2] // 2
            ty = tgt["bbox"][1] + tgt["bbox"][3] // 2
            cv2.circle(img, (tx, ty), 6, (0, 0, 255), -1) # Red dot at destination
            cv2.circle(img, (tx, ty), 8, config.VIS_EDGE_COLOR, 2)
            
        else:
            # Fallback for older JSONs without path data
            sx = src["bbox"][0] + src["bbox"][2] // 2
            sy = src["bbox"][1] + src["bbox"][3] // 2
            tx = tgt["bbox"][0] + tgt["bbox"][2] // 2
            ty = tgt["bbox"][1] + tgt["bbox"][3] // 2
            cv2.arrowedLine(img, (sx, sy), (tx, ty),
                             config.VIS_EDGE_COLOR,
                             config.VIS_LINE_THICKNESS, tipLength=0.03)

    img = append_legend(img, [
        ("rect", "Process", config.VIS_NODE_COLOR),
        ("diamond", "Decision", config.VIS_NODE_COLOR),
        ("ellipse", "Terminal", config.VIS_NODE_COLOR),
        ("circle", "Connector", config.VIS_NODE_COLOR),
        ("parallelogram", "Data (I/O)", config.VIS_NODE_COLOR),
        ("rect", "Bad node (artifact)", (0, 0, 255)),
        ("thickline", "Edge (traced ink path)", config.VIS_EDGE_COLOR),
        ("target", "Edge target (direction)", config.VIS_EDGE_COLOR),
    ])

    out_path = output_dir / f"{image_path.stem}_result.png"
    cv2.imwrite(str(out_path), img)
    logger.info("Visualisation saved → %s", out_path)
    return out_path
