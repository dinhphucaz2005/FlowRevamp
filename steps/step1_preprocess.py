"""
Step 1 – Preprocessing
======================
Converts flowchart images to clean binary form suitable for downstream
detection and segmentation.

Pipeline (two branches):
  High-C: Adaptive threshold → Morph Close → thick binary
          (for Node Detection via contour hierarchy — NOT skeletonised)
  Low-C:  Adaptive threshold → Morph Close → Skeletonize
          (for Edge Detection / Pairwise ROI pathfinding)
Both results are saved to data/step1_preprocessed/
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

import config

logger = logging.getLogger(__name__)

def remove_small_components(binary: np.ndarray,
                            min_area: int = 150,
                            max_area: int = 3000,
                            max_dim: int = 80) -> np.ndarray:
    """
    Remove connected components likely to be text.

    Parameters
    ----------
    binary : uint8 image (0 / 255)

    Returns
    -------
    uint8 image
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    output = np.zeros_like(binary)

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Remove components that look like text
        if area < min_area:
            continue

        if area < max_area and w < max_dim and h < max_dim:
            continue

        output[labels == i] = 255

    return output

def preprocess(image_path: Path, output_dir: Path | None = None) -> Path:
    """Run the full preprocessing pipeline on a single image.

    Parameters
    ----------
    image_path : Path
        Absolute or relative path to the source flowchart image.
    output_dir : Path, optional
        Directory to save the result.  Defaults to ``config.STEP1_DIR``.

    Returns
    -------
    Path
        Path to the saved preprocessed (binary/skeleton) image.
    """
    output_dir = output_dir or config.STEP1_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    logger.info("Loaded image %s  (%s)", image_path.name, img.shape)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary_high = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=config.ADAPTIVE_THRESH_BLOCK_SIZE,
        C=config.ADAPTIVE_THRESH_C_HIGH,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL_SIZE)
    cleaned_high = cv2.morphologyEx(binary_high, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4. Process Low C (Preserves faint lines for Edge Detection / Pathfinding)
    binary_low = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=config.ADAPTIVE_THRESH_BLOCK_SIZE,
        C=config.ADAPTIVE_THRESH_C_LOW,
    )
    cleaned_low = cv2.morphologyEx(binary_low, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Remove small noise components (likely text)
    cleaned_low = remove_small_components(
        cleaned_low,
        min_area=150,
        max_area=3000,
        max_dim=80
    )
    
    skeleton_bool_low = skeletonize(cleaned_low > 0)
    skeleton_low = (skeleton_bool_low.astype(np.uint8)) * 255

    # 5. Save High C outputs (thick binary, NOT skeletonised)
    out_name = f"{image_path.stem}_preprocessed.png"
    out_path = output_dir / out_name
    cv2.imwrite(str(out_path), cleaned_high)
    logger.info("Saved high-C preprocessed binary → %s", out_path)

    # 6. Save Low C outputs
    out_path_low = output_dir / f"{image_path.stem}_preprocessed_low_c.png"
    cv2.imwrite(str(out_path_low), skeleton_low)
    logger.info("Saved low-C preprocessed skeleton → %s", out_path_low)

    return out_path
