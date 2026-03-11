"""Two-tier segmentation: Cellpose (preferred) + classical adaptive threshold/watershed."""

import logging

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_objects
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)


def segment(image: np.ndarray, config) -> np.ndarray:
    """Segment myotubes using configured method.

    Args:
        image: Preprocessed 2D float64 image in [0, 1].
        config: Config object.

    Returns:
        Integer-labeled instance mask (0 = background).
    """
    method = config.method

    if method == "auto":
        try:
            labels = segment_cellpose(image, config)
            logger.info(f"Cellpose segmentation: {labels.max()} objects")
            return labels
        except ImportError:
            logger.info("Cellpose not available, falling back to classical segmentation")
            method = "classical"
        except Exception as e:
            logger.warning(f"Cellpose failed ({e}), falling back to classical")
            method = "classical"

    if method == "cellpose":
        labels = segment_cellpose(image, config)
        logger.info(f"Cellpose segmentation: {labels.max()} objects")
        return labels

    labels = segment_classical(image, config)
    logger.info(f"Classical segmentation: {labels.max()} objects")
    return labels


def segment_cellpose(image: np.ndarray, config) -> np.ndarray:
    """Segment using Cellpose cyto2 model."""
    from cellpose import models

    model = models.Cellpose(model_type=config.cellpose_model, gpu=True)

    # Cellpose expects uint8 or uint16
    img_uint8 = (image * 255).astype(np.uint8)

    masks, flows, styles, diams = model.eval(
        img_uint8,
        diameter=config.cellpose_diameter,
        flow_threshold=config.cellpose_flow_threshold,
        channels=[0, 0],  # grayscale
    )

    return masks.astype(np.int32)


def segment_classical(image: np.ndarray, config) -> np.ndarray:
    """Classical segmentation: adaptive threshold + morphology + watershed.

    Steps:
    1. Adaptive Gaussian threshold -> binary mask
    2. Morphological opening -> remove small noise
    3. Morphological closing -> fill gaps from sarcomeric MF-20 pattern
    4. Remove small objects
    5. Distance transform + local maxima -> watershed seeds
    6. Marker-controlled watershed -> instance labels
    """
    # Convert to uint8 for OpenCV adaptive threshold
    img_uint8 = (image * 255).astype(np.uint8)

    # Adaptive threshold
    block_size = config.adaptive_block_size
    c_val = config.adaptive_c
    binary = cv2.adaptiveThreshold(
        img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c_val
    )
    binary = binary.astype(bool)

    # Morphological cleanup
    binary = binary_opening(binary, disk(config.morph_open_size))
    binary = binary_closing(binary, disk(config.morph_close_size))

    # Remove small objects
    binary = remove_small_objects(binary, min_size=500)

    if not binary.any():
        logger.warning("No objects found after thresholding")
        return np.zeros(image.shape, dtype=np.int32)

    # Distance transform for watershed seeds
    distance = ndimage.distance_transform_edt(binary)

    # Find local maxima as watershed markers
    # min_distance prevents over-segmentation
    min_dist = max(10, config.morph_close_size)
    coords = peak_local_max(distance, min_distance=min_dist, labels=binary)

    if len(coords) == 0:
        # No peaks found; label connected components directly
        labels, _ = ndimage.label(binary)
        return labels.astype(np.int32)

    # Create marker image
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # Watershed
    labels = watershed(-distance, markers, mask=binary)

    return labels.astype(np.int32)
