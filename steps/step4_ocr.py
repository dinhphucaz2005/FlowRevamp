"""
Step 4 – OCR (Text Extraction)
==============================
Extracts text from each detected node using EasyOCR.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import config

logger = logging.getLogger(__name__)
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=config.OCR_GPU)
    return _reader


def _pad_bbox(bbox, img_shape, pad_ratio):
    x, y, w, h = bbox
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    return max(x - px, 0), max(y - py, 0), min(x + w + px, img_shape[1]), min(y + h + py, img_shape[0])


def extract_text(image_path: Path, nodes: list[dict[str, Any]],
                 output_dir: Path | None = None) -> list[dict[str, Any]]:
    """Run OCR on each node crop and return nodes with 'text' field."""
    output_dir = output_dir or config.STEP4_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    reader = _get_reader()
    results = []
    debug_img = img.copy()
    
    for node in nodes:
        x1, y1, x2, y2 = _pad_bbox(node["bbox"], img.shape, config.OCR_ROI_PADDING_RATIO)
        crop = img[y1:y2, x1:x2]
        detections = reader.readtext(crop, detail=0, paragraph=True)
        text = " ".join(detections).strip() if detections else ""
        results.append({**node, "text": text})
        
        # Draw text on debug image
        if text:
            # Draw a dark background rectangle for text visibility
            (tw, th), _ = cv2.getTextSize(text[:20], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(debug_img, (x1, y1 - 25), (x1 + tw, y1), (0, 0, 0), -1)
            cv2.putText(debug_img, text[:20], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Save Edge JSON
    out_path = output_dir / f"{image_path.stem}_ocr.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # Save Debug Image
    debug_path = output_dir / f"{image_path.stem}_ocr_debug.png"
    cv2.imwrite(str(debug_path), debug_img)
    
    logger.info("OCR: %d nodes → %s", len(results), out_path)
    return results
