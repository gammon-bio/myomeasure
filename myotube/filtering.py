"""Filter segmented objects to retain only myotubes."""

import logging

import numpy as np
from skimage.measure import regionprops

logger = logging.getLogger(__name__)


def filter_myotubes(labels: np.ndarray, config, pixel_size: float = 1.0) -> np.ndarray:
    """Remove non-myotube objects based on morphological criteria.

    Filters applied:
    1. Area >= min_object_area
    2. Aspect ratio (major/minor axis) >= min_aspect_ratio
    3. Length (major axis * pixel_size) >= min_length_um
    4. Circularity (4*pi*area/perimeter^2) <= max_circularity
    5. Solidity (area/convex_area) >= min_solidity

    Args:
        labels: Integer-labeled instance mask.
        config: Config object with filter parameters.
        pixel_size: Microns per pixel for length calculation.

    Returns:
        Filtered and relabeled instance mask.
    """
    if labels.max() == 0:
        logger.warning("No objects to filter")
        return labels

    props = regionprops(labels)
    keep_labels = set()
    reasons_removed = {}

    for prop in props:
        label_id = prop.label
        area = prop.area
        major = prop.axis_major_length
        minor = prop.axis_minor_length
        perimeter = prop.perimeter

        # Area filter
        if area < config.min_object_area:
            reasons_removed[label_id] = f"area={area} < {config.min_object_area}"
            continue

        # Aspect ratio filter
        if minor > 0:
            aspect_ratio = major / minor
        else:
            aspect_ratio = float("inf")

        if aspect_ratio < config.min_aspect_ratio:
            reasons_removed[label_id] = f"AR={aspect_ratio:.1f} < {config.min_aspect_ratio}"
            continue

        # Length filter (convert to microns)
        length_um = major * pixel_size
        if length_um < config.min_length_um:
            reasons_removed[label_id] = f"length={length_um:.1f}um < {config.min_length_um}um"
            continue

        # Circularity filter
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        if circularity > config.max_circularity:
            reasons_removed[label_id] = f"circ={circularity:.2f} > {config.max_circularity}"
            continue

        # Solidity filter
        solidity = prop.solidity
        if solidity < config.min_solidity:
            reasons_removed[label_id] = f"solidity={solidity:.2f} < {config.min_solidity}"
            continue

        keep_labels.add(label_id)

    n_removed = len(props) - len(keep_labels)
    logger.info(f"Filtering: kept {len(keep_labels)}/{len(props)} objects "
                f"(removed {n_removed})")

    if logger.isEnabledFor(logging.DEBUG):
        for label_id, reason in reasons_removed.items():
            logger.debug(f"  Removed label {label_id}: {reason}")

    if not keep_labels:
        logger.warning(
            "All objects were filtered out. Consider relaxing filter parameters "
            "(--min-area, --min-aspect-ratio, etc.)"
        )
        return np.zeros_like(labels)

    # Relabel sequentially
    filtered = np.zeros_like(labels)
    for new_id, old_id in enumerate(sorted(keep_labels), start=1):
        filtered[labels == old_id] = new_id

    return filtered
