"""
I/O helpers – file reading, JSON serialisation, directory setup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)


def ensure_dirs() -> None:
    """Create all data directories defined in config."""
    for d in config.ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    logger.debug("Data directories ensured")


def save_json(data: Any, path: Path) -> None:
    """Write *data* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug("Saved JSON → %s", path)


def load_json(path: Path) -> Any:
    """Read and return parsed JSON from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_images(directory: Path, extensions=(".png", ".jpg", ".jpeg", ".bmp")) -> list[Path]:
    """Return sorted list of image files in *directory*."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
