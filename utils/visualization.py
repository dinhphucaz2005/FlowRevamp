"""
Visualisation – save flowchart results to a JS data file for the static HTML visualizer.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)


def draw_results(image_path: Path, nodes: list[dict[str, Any]],
                 edges: list[dict[str, Any]],
                 output_dir: Path | None = None) -> Path:
    """Save nodes and edges into a JS data file, update the list, and setup visualizer.html."""
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare graph payload (preserving exact edge paths)
    graph = {
        "source_image": image_path.name,
        "nodes": [
            {
                "id": n["id"],
                "type": n.get("type", "Unknown"),
                "text": n.get("text", ""),
                "bbox": n["bbox"],
                "is_bad": n.get("is_bad", False),
            }
            for n in nodes
        ],
        "edges": [
            {
                "source": e["source"],
                "target": e["target"],
                "label": e.get("label", ""),
                "path": e.get("path", []),
            }
            for e in edges
        ],
    }

    # 2. Save the [image_stem]_data.js file
    js_path = output_dir / f"{image_path.stem}_data.js"
    js_content = (
        f"window.flowchartResults = window.flowchartResults || {{}};\n"
        f"window.flowchartResults[{json.dumps(image_path.stem)}] = "
        f"{json.dumps(graph, indent=2)};\n"
    )
    js_path.write_text(js_content, encoding="utf-8")
    logger.info("Saved flowchart data to JS → %s", js_path)

    # 3. Update the list of flowcharts in flowcharts_list.js
    update_flowcharts_list(image_path.stem, output_dir)
    
    # 4. Make sure visualizer.html exists in output directory
    ensure_visualizer_html(output_dir)

    return js_path


def update_flowcharts_list(new_stem: str, output_dir: Path) -> None:
    """Append the new flowchart stem to the list of available flowcharts."""
    list_path = output_dir / "flowcharts_list.js"
    stems = []
    if list_path.exists():
        try:
            content = list_path.read_text(encoding="utf-8")
            import re
            match = re.search(r"window\.flowchartsList\s*=\s*(\[.*?\]);", content, re.DOTALL)
            if match:
                stems = json.loads(match.group(1))
        except Exception as e:
            logger.warning("Failed to parse existing flowcharts_list.js: %s", e)
            
    if new_stem not in stems:
        stems.append(new_stem)
        stems.sort()
        
    content = f"window.flowchartsList = {json.dumps(stems, indent=2)};\n"
    list_path.write_text(content, encoding="utf-8")
    logger.info("Updated flowcharts list → %s", list_path)


def ensure_visualizer_html(output_dir: Path) -> None:
    """Copy the master visualizer.html from the ui directory to the output directory."""
    src_path = config.PROJECT_ROOT / "ui" / "visualizer.html"
    dest_path = output_dir / "visualizer.html"
    
    if src_path.exists():
        try:
            shutil.copy2(src_path, dest_path)
            logger.info("Copied visualizer.html to output → %s", dest_path)
        except Exception as e:
            logger.error("Failed to copy visualizer.html: %s", e)
    else:
        logger.warning("Master visualizer.html not found in %s", src_path)
