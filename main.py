#!/usr/bin/env python3
"""
FlowRevamp – Flowchart Digitisation CLI
========================================

Usage:
    python main.py                      # Process all images in data/input/
    python main.py path/to/image.png    # Process a single image
    python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
from orchestrator import run_pipeline, run_all
from utils.io_helpers import ensure_dirs


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Digitise flowchart images into structured JSON graphs."
    )
    parser.add_argument(
        "image", nargs="?", type=Path, default=None,
        help="Path to a single flowchart image.  "
             "If omitted, processes all images in data/input/.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--max-node-size", type=str, default=None,
        help="Maximum node size in the format WIDTH,HEIGHT (e.g. 200,200). "
             "Nodes larger than this limit will be ignored.",
    )
    args = parser.parse_args()

    if args.max_node_size:
        try:
            w, h = map(int, args.max_node_size.split(","))
            config.MAX_NODE_WIDTH = w
            config.MAX_NODE_HEIGHT = h
        except ValueError:
            print("Error: --max-node-size must be in the format WIDTH,HEIGHT (e.g. 200,200)", file=sys.stderr)
            sys.exit(1)

    _setup_logging(args.verbose)
    ensure_dirs()

    if args.image:
        if not args.image.exists():
            print(f"Error: file not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        graph = run_pipeline(args.image)
        print(f"\n✓ Done – {len(graph['nodes'])} nodes, "
              f"{len(graph['edges'])} edges")
        print(f"  JSON:  {config.OUTPUT_DIR / (args.image.stem + '_graph.json')}")
        print(f"  Image: {config.OUTPUT_DIR / (args.image.stem + '_result.png')}")
    else:
        results = run_all()
        print(f"\n✓ Processed {len(results)} image(s)")


if __name__ == "__main__":
    main()
