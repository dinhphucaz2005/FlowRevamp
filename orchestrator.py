"""
Orchestrator – Coordinates the 4-step flowchart digitisation pipeline.

Reads images from data/input/, runs Steps 1–4 sequentially, merges all
results into a final JSON graph, and generates a visualisation overlay.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import config
from utils.io_helpers import ensure_dirs, save_json, list_images
from utils.visualization import draw_results

from steps.step1_preprocess import preprocess
from steps.step2_detect_nodes import detect_nodes, finetune_nodes
from steps.step3_detect_edges import detect_edges, finetune_edges
from steps.step4_ocr import extract_text

logger = logging.getLogger(__name__)


def run_pipeline(image_path: Path) -> dict[str, Any]:
    """Execute the full pipeline on a single flowchart image.

    Returns the final graph dict: ``{"nodes": [...], "edges": [...]}``.
    """
    logger.info("=" * 60)
    logger.info("PIPELINE START: %s", image_path.name)
    logger.info("=" * 60)

    # Step 1 – Preprocessing
    logger.info("▶ Step 1: Preprocessing")
    skeleton_path = preprocess(image_path)

    # Step 2a – Node detection (Raw)
    logger.info("▶ Step 2a: Node detection (Raw)")
    nodes_raw = detect_nodes(image_path)
    
    # Step 2b – Node fine-tuning
    logger.info("▶ Step 2b: Node fine-tuning (Overlap filter)")
    nodes = finetune_nodes(image_path, nodes_raw)

    # Step 3a – Edge detection (Raw)
    logger.info("▶ Step 3a: Edge detection (Raw ROI Pathfinding)")
    edges, used_contours = detect_edges(image_path, nodes)
    
    # Step 3b – Edge fine-tuning
    logger.info("▶ Step 3b: Edge fine-tuning (Leftover artifacts)")
    nodes_final = finetune_edges(image_path, nodes, used_contours)

    # Step 4 – OCR (Last step)
    if getattr(config, "ENABLE_OCR", True):
        logger.info("▶ Step 4: OCR text extraction")
        nodes_final_with_text = extract_text(image_path, nodes_final)
    else:
        logger.info("▶ Step 4: OCR disabled in config (Skipping)")
        nodes_final_with_text = nodes_final

    # ── Merge into final graph ────────────────────────────────
    graph: dict[str, Any] = {
        "source_image": image_path.name,
        "nodes": [
            {
                "id": n["id"],
                "type": n.get("type", "Unknown"),
                "text": n.get("text", ""),
                "bbox": n["bbox"],
                "is_bad": n.get("is_bad", False),
            }
            for n in nodes_final_with_text
        ],
        "edges": [
            {
                "source": e["source"],
                "target": e["target"],
                "label": e.get("label", ""),
            }
            for e in edges
        ],
    }

    # Save final JSON
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.OUTPUT_DIR / f"{image_path.stem}_graph.json"
    save_json(graph, json_path)
    logger.info("✓ Graph JSON → %s", json_path)

    # Generate visualisation
    vis_path = draw_results(image_path, nodes_final_with_text, edges)
    logger.info("✓ Visualisation → %s", vis_path)

    logger.info("PIPELINE COMPLETE: %s  (%d nodes, %d edges)",
                image_path.name, len(graph["nodes"]), len(graph["edges"]))
    return graph


def run_all() -> list[dict[str, Any]]:
    """Process every image in data/input/."""
    ensure_dirs()
    images = list_images(config.INPUT_DIR)
    if not images:
        logger.warning("No images found in %s", config.INPUT_DIR)
        return []

    results = []
    for img_path in images:
        try:
            graph = run_pipeline(img_path)
            results.append(graph)
        except Exception:
            logger.exception("Failed to process %s", img_path.name)
    return results
