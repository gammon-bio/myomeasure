"""Image preprocessing: normalization, denoising, contrast enhancement."""

import logging

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist

logger = logging.getLogger(__name__)


def preprocess(image: np.ndarray, config) -> np.ndarray:
    """Full preprocessing pipeline: normalize -> denoise -> CLAHE.

    Args:
        image: 2D float64 image in [0, 1].
        config: Config object with preprocessing parameters.

    Returns:
        Preprocessed 2D float64 image in [0, 1].
    """
    img = normalize_intensity(image)
    img = denoise(img, sigma=config.gaussian_sigma)
    img = enhance_contrast(img, clip_limit=config.clahe_clip_limit,
                           tile_size=config.clahe_tile_size)
    return img


def normalize_intensity(image: np.ndarray, low_pct: float = 0.5,
                        high_pct: float = 99.5) -> np.ndarray:
    """Percentile-based intensity normalization.

    Robust to hot pixels and outliers.
    """
    p_low = np.percentile(image, low_pct)
    p_high = np.percentile(image, high_pct)

    if p_high - p_low < 1e-10:
        logger.warning("Image has near-zero dynamic range")
        return image

    img = (image - p_low) / (p_high - p_low)
    return np.clip(img, 0, 1)


def denoise(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian blur to suppress shot noise."""
    if sigma <= 0:
        return image
    return gaussian_filter(image, sigma=sigma)


def enhance_contrast(image: np.ndarray, clip_limit: float = 3.0,
                     tile_size: int = 64) -> np.ndarray:
    """CLAHE to normalize uneven staining intensity.

    The kernel_size parameter in equalize_adapthist specifies tile dimensions.
    """
    h, w = image.shape
    # kernel_size must not exceed image dimensions
    ks = min(tile_size, h, w)

    result = equalize_adapthist(image, kernel_size=ks,
                                clip_limit=clip_limit / 100.0)
    return result.astype(np.float64)
