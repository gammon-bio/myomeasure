#!/usr/bin/env python3
"""CLI entry point for myotube diameter measurement."""

import logging
import sys

from myotube.config import load_config
from myotube.pipeline import process_batch


def main():
    config = load_config()

    # Set up logging
    level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not config.input_path:
        print("Error: input_path is required. Usage:")
        print("  python measure_myotubes.py path/to/images/")
        print("  python measure_myotubes.py single_image.tif")
        print("  python measure_myotubes.py --help")
        sys.exit(1)

    process_batch(config.input_path, config)


if __name__ == "__main__":
    main()
