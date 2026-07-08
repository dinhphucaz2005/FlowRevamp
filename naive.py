#!/usr/bin/env python3
"""
naive.py – Pipeline Baseline "ngây thơ" (không có các cải tiến tối ưu)
======================================================================
Giữ nguyên cấu trúc 4 bước của pipeline chính (main.py) nhưng loại bỏ
toàn bộ các ý tưởng cải tiến, dùng làm mốc so sánh (baseline):

  Step 1 – Preprocessing:
      Otsu Threshold TOÀN CỤC duy nhất — KHÔNG dùng Adaptive Thresholding,
      KHÔNG có 2 ngưỡng High-C/Low-C riêng cho node và line,
      KHÔNG skeletonize.

  Step 2 – Node detection:
      findContours MỘT LƯỢT trực tiếp trên ảnh Otsu + đếm đỉnh
      approxPolyDP — KHÔNG Canny, KHÔNG phân tích phả hệ contour
      (RETR_TREE), KHÔNG RANSAC/khớp hình học, KHÔNG Iterative Masking,
      KHÔNG khử arrowhead, KHÔNG giới hạn kích thước node
      (MAX_NODE_WIDTH/HEIGHT), KHÔNG lọc Bad Node (chồng lấp/khổng lồ).

  Step 3 – Edge detection:
      HoughLinesP TOÀN CỤC + nối tâm gần nhất (Center Snapping) —
      KHÔNG Pairwise ROI Pathfinding, KHÔNG xác định hướng bằng
      arrowhead, KHÔNG hậu xử lý nét vẽ mồ côi (Leftover).

  Step 4 – OCR: Bỏ qua.

  Trực quan hóa: nối đường thẳng đâm xuyên tâm 2 node —
      KHÔNG bám quỹ đạo nét vẽ thực tế (fillPoly).

Usage:
    python naive.py path/to/image.png [-v]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("NaivePipeline")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Step 1: Preprocessing (Otsu toàn cục, một ngưỡng duy nhất) ──

def naive_preprocess(gray: np.ndarray, image_stem: str) -> np.ndarray:
    """Nhị phân hóa bằng Otsu toàn cục. Một ảnh duy nhất dùng chung
    cho cả detect node lẫn detect line (không Dual-C, không skeleton)."""
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    step1_dir = Path("data/step1_preprocessed")
    step1_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(step1_dir / f"{image_stem}_naive_preprocessed.png"), binary)
    return binary


# ── Step 2: Node detection (một lượt contour, không lọc gì) ─────

def naive_detect_nodes(binary: np.ndarray) -> list[dict]:
    """Quét contour MỘT lượt duy nhất trên ảnh nhị phân Otsu, phân loại
    thô bằng số đỉnh. Dùng RETR_LIST để lấy mọi contour (kể cả lỗ bên
    trong các khối) nhưng KHÔNG phân tích quan hệ cha–con của phả hệ,
    không giới hạn kích thước, không lọc chồng lấp, không mask-and-repeat.
    Hệ quả: khối liên thông bao ngoài cũng bị nhận nhầm thành một node
    khổng lồ và không có bộ lọc nào loại nó ra."""
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    nodes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # chỉ lọc nhiễu li ti, không có giới hạn trên
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        verts = len(approx)

        # Phân loại ngây thơ: chỉ dựa vào số đỉnh
        if verts == 4:
            node_type = "Process"
        elif verts == 3:
            node_type = "Decision"
        elif verts > 4:
            node_type = "Connector" if 0.8 < w / h < 1.2 else "Terminal"
        else:
            node_type = "Process"

        nodes.append({
            "id": f"n{len(nodes) + 1}",
            "type": node_type,
            "bbox": [int(x), int(y), int(w), int(h)],
            "center": (int(x + w // 2), int(y + h // 2)),
        })

    logger.info("Detected %d nodes (single-pass, no filtering)", len(nodes))
    return nodes


# ── Step 3: Edge detection (Hough toàn cục + Center Snapping) ───

def naive_detect_edges(binary: np.ndarray, nodes: list[dict]) -> list[dict]:
    """Chạy HoughLinesP trên ảnh nhị phân Otsu TOÀN cục, snap 2 đầu mút
    của mỗi đoạn thẳng về tâm node gần nhất. Không ROI theo cặp,
    không xác định hướng."""
    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=50, maxLineGap=15,
    )

    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()

    if lines is None or len(nodes) < 2:
        logger.info("Detected 0 edges")
        return edges

    SNAP_DIST = 120.0  # ngưỡng snap lớn, chấp nhận sai
    for line in lines:
        x1, y1, x2, y2 = line[0]

        n1_best, n1_dist = None, SNAP_DIST
        n2_best, n2_dist = None, SNAP_DIST
        for node in nodes:
            cx, cy = node["center"]
            d1 = np.hypot(x1 - cx, y1 - cy)
            d2 = np.hypot(x2 - cx, y2 - cy)
            if d1 < n1_dist:
                n1_best, n1_dist = node["id"], d1
            if d2 < n2_dist:
                n2_best, n2_dist = node["id"], d2

        if n1_best and n2_best and n1_best != n2_best:
            pair = tuple(sorted([n1_best, n2_best]))
            if pair not in seen:
                seen.add(pair)
                # Không có arrowhead detection → hướng chỉ là thứ tự đầu mút
                edges.append({"source": n1_best, "target": n2_best, "label": ""})

    logger.info("Detected %d edges (global Hough + center snapping)", len(edges))
    return edges


# ── Visualization (nối thẳng tâm, không bám nét vẽ) ─────────────

NODE_COLORS = {
    "Process": (0, 255, 0),
    "Decision": (0, 165, 255),
    "Connector": (255, 0, 0),
    "Terminal": (255, 255, 0),
}


def naive_visualize(img: np.ndarray, nodes: list[dict], edges: list[dict],
                    out_path: Path) -> None:
    from utils.visualization import append_legend

    vis = img.copy()
    colors = NODE_COLORS

    for node in nodes:
        x, y, w, h = node["bbox"]
        color = colors.get(node["type"], (0, 255, 0))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, f'{node["id"]}:{node["type"]}', (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    node_map = {n["id"]: n for n in nodes}
    for edge in edges:
        n1 = node_map.get(edge["source"])
        n2 = node_map.get(edge["target"])
        if n1 and n2:
            # Đường thẳng ảo đâm xuyên tâm — không theo quỹ đạo nét vẽ thật
            cv2.line(vis, n1["center"], n2["center"], (0, 0, 255), 2)
            cv2.circle(vis, n1["center"], 6, (255, 0, 0), -1)
            cv2.circle(vis, n2["center"], 6, (255, 0, 0), -1)

    vis = append_legend(vis, [
        *[("rect", name, color) for name, color in NODE_COLORS.items()],
        ("line", "Edge (straight center line)", (0, 0, 255)),
        ("dot", "Snapped endpoint", (255, 0, 0)),
    ])
    cv2.imwrite(str(out_path), vis)


# ── Pipeline ────────────────────────────────────────────────────

def run_naive_pipeline(image_path: Path) -> dict:
    logger.info("=" * 60)
    logger.info("NAIVE PIPELINE START: %s", image_path.name)
    logger.info("=" * 60)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1 – một ảnh nhị phân duy nhất dùng chung cho mọi bước sau
    logger.info("▶ Step 1: Preprocessing (Global Otsu, single threshold)")
    binary = naive_preprocess(gray, image_path.stem)

    # Step 2 – node detection một lượt, không tinh chỉnh
    logger.info("▶ Step 2: Node detection (single-pass contours)")
    nodes = naive_detect_nodes(binary)
    logger.info("▶ Step 2b: Fine-tuning — SKIPPED (no size limit, no bad-node filter)")

    # Step 3 – Hough toàn cục
    logger.info("▶ Step 3: Edge detection (global HoughLinesP)")
    edges = naive_detect_edges(binary, nodes)
    logger.info("▶ Step 3b: Fine-tuning — SKIPPED (no leftover detection)")

    # Step 4 – bỏ qua
    logger.info("▶ Step 4: OCR — SKIPPED")

    graph = {
        "source_image": image_path.name,
        "nodes": [
            {"id": n["id"], "type": n["type"], "text": "",
             "bbox": n["bbox"], "is_bad": False}
            for n in nodes
        ],
        "edges": edges,
    }

    json_path = output_dir / f"{image_path.stem}_naive_graph.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    logger.info("✓ Graph JSON → %s", json_path)

    vis_path = output_dir / f"{image_path.stem}_naive_result.png"
    naive_visualize(img, nodes, edges, vis_path)
    logger.info("✓ Visualisation → %s", vis_path)

    logger.info("NAIVE PIPELINE COMPLETE: %s  (%d nodes, %d edges)",
                image_path.name, len(graph["nodes"]), len(graph["edges"]))
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Naive baseline flowchart digitisation (no optimisations)."
    )
    parser.add_argument(
        "image", nargs="?", type=Path, default=None,
        help="Path to a flowchart image. "
             "Defaults to data/input/test_flowchart.png.",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")
    args = parser.parse_args()

    setup_logging(args.verbose)

    img_path = args.image or Path("data/input/test_flowchart.png")
    if not img_path.exists():
        logger.error("Không tìm thấy file ảnh: %s", img_path)
        sys.exit(1)

    run_naive_pipeline(img_path)


if __name__ == "__main__":
    main()
