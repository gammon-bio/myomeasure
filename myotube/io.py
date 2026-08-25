"""Image I/O, batch discovery, and scale calibration."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def discover_images(directory: str) -> List[Path]:
    """Find image files in directory, sorted naturally."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    images = [
        p for p in dir_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    ]
    images.sort(key=lambda p: _natural_sort_key(p.name))
    logger.info(f"Found {len(images)} images in {directory}")
    return images


def _natural_sort_key(text: str):
    """Sort strings with embedded numbers naturally."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]


def load_image(path: str) -> np.ndarray:
    """Load image as float64 [0, 1].

    Uses tifffile for TIFF, OpenCV for other formats.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix in (".tif", ".tiff"):
        import tifffile
        img = tifffile.imread(path)
    else:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to load image: {path}")
        # OpenCV loads BGR; convert to RGB if color
        if img.ndim == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float64)

    # Normalize to [0, 1]
    if img.max() > 1.0:
        if img.max() <= 255:
            img /= 255.0
        elif img.max() <= 4095:
            img /= 4095.0
        elif img.max() <= 65535:
            img /= 65535.0
        else:
            img /= img.max()

    return img


def extract_green_channel(img: np.ndarray) -> np.ndarray:
    """Reduce a multi-channel image to a single 2-D intensity channel.

    The calibrated TIFFs written by convert_vsi_to_tiff are single-channel, so
    grayscale input is returned unchanged (the usual path). For RGB or other
    multi-channel inputs, one channel is returned as a fallback.
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] >= 3:
            # RGB: green is channel 1
            return img[:, :, 1]
        elif img.shape[2] == 2:
            return img[:, :, 0]
        elif img.shape[0] in (2, 3, 4) and img.shape[0] < img.shape[1]:
            # Channel-first format (C, H, W)
            if img.shape[0] >= 2:
                return img[1, :, :]
            return img[0, :, :]
    # Fallback: squeeze or take first 2D slice
    if img.ndim > 2:
        return img.reshape(img.shape[-2], img.shape[-1])
    return img


def parse_pixel_size(path: str, config_pixel_size: Optional[float] = None) -> Tuple[float, str]:
    """Determine pixel size in microns.

    Tries: TIFF metadata -> config value -> default with warning.
    Returns (pixel_size, unit) where unit is 'um' or 'pixels'.
    """
    # Try TIFF metadata
    suffix = Path(path).suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tif:
                # Try ImageJ metadata
                if tif.imagej_metadata:
                    meta = tif.imagej_metadata
                    if "unit" in meta and meta["unit"] in ("um", "µm", "micron"):
                        # Resolution is stored in TIFF tags
                        page = tif.pages[0]
                        tags = page.tags
                        if "XResolution" in tags:
                            xres = tags["XResolution"].value
                            if isinstance(xres, tuple) and len(xres) == 2 and xres[0] > 0:
                                pixel_size = xres[1] / xres[0]
                                logger.info(f"Pixel size from TIFF metadata: {pixel_size:.4f} um/px")
                                return pixel_size, "um"

                # Try OME-XML
                if tif.ome_metadata:
                    ome = tif.ome_metadata
                    if "PhysicalSizeX" in str(ome):
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(ome)
                        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                        for pixels in root.iter():
                            if "PhysicalSizeX" in pixels.attrib:
                                px_size = float(pixels.attrib["PhysicalSizeX"])
                                logger.info(f"Pixel size from OME metadata: {px_size:.4f} um/px")
                                return px_size, "um"
        except Exception as e:
            logger.debug(f"Could not parse TIFF metadata: {e}")

    # Use config value
    if config_pixel_size is not None:
        logger.info(f"Using config pixel size: {config_pixel_size} um/px")
        return config_pixel_size, "um"

    # No calibration available
    logger.warning(
        "No pixel size calibration found. Measurements will be in pixels. "
        "Set --pixel-size or add calibration to TIFF metadata."
    )
    return 1.0, "pixels"


def save_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved results to {path}")
