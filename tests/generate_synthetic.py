"""Generate synthetic test images with known ground-truth diameters."""

import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk, line, polygon


OUTPUT_DIR = Path(__file__).parent / "fixtures"


def generate_all():
    """Generate all synthetic test images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    straight_tube(width=30, length=300, filename="straight_tube_w30.tif")
    curved_tube(width=25, filename="curved_tube_w25.tif")
    two_parallel_tubes(width=20, gap=2, filename="two_parallel_w20.tif")
    branching_tube(width=20, filename="branching_tube_w20.tif")
    mixed_scene(filename="mixed_scene.tif")
    variable_intensity_tube(width=30, filename="variable_intensity_w30.tif")

    print(f"Generated synthetic images in {OUTPUT_DIR}")


def straight_tube(width=30, length=300, filename="straight_tube.tif"):
    """Single straight horizontal tube with known width."""
    h, w_img = 200, 400
    img = np.zeros((h, w_img), dtype=np.float64)

    cy = h // 2
    x_start = (w_img - length) // 2
    x_end = x_start + length
    y_start = cy - width // 2
    y_end = cy + width // 2

    img[y_start:y_end, x_start:x_end] = 1.0

    # Slight blur to simulate real edges
    img = gaussian_filter(img, sigma=1.0)

    _save(img, filename)
    return img


def curved_tube(width=25, filename="curved_tube.tif"):
    """Single curved tube (arc) with constant width."""
    h, w_img = 300, 300
    img = np.zeros((h, w_img), dtype=np.float64)

    # Create arc by sweeping a disk along a curve
    t = np.linspace(0.3 * np.pi, 0.7 * np.pi, 500)
    radius = 100
    cx, cy = w_img // 2, h // 2 + 50

    half_w = width / 2.0

    for ti in t:
        r = int(cy - radius * np.sin(ti))
        c = int(cx + radius * np.cos(ti))
        rr, cc = disk((r, c), half_w, shape=img.shape)
        img[rr, cc] = 1.0

    img = gaussian_filter(img, sigma=1.0)
    _save(img, filename)
    return img


def two_parallel_tubes(width=20, gap=2, filename="two_parallel.tif"):
    """Two parallel touching tubes for testing watershed separation."""
    h, w_img = 200, 400
    img = np.zeros((h, w_img), dtype=np.float64)

    cy = h // 2
    x_start, x_end = 50, 350

    # Tube 1
    y1_start = cy - gap // 2 - width
    y1_end = cy - gap // 2
    img[y1_start:y1_end, x_start:x_end] = 1.0

    # Tube 2
    y2_start = cy + gap // 2
    y2_end = cy + gap // 2 + width
    img[y2_start:y2_end, x_start:x_end] = 1.0

    img = gaussian_filter(img, sigma=1.0)
    _save(img, filename)
    return img


def branching_tube(width=20, filename="branching_tube.tif"):
    """Y-shaped branching tube for testing branch pruning."""
    h, w_img = 300, 300
    img = np.zeros((h, w_img), dtype=np.float64)

    half_w = width // 2

    # Main horizontal segment
    cy = h // 2
    img[cy - half_w:cy + half_w, 30:200] = 1.0

    # Upper branch (short, should be pruned if < prune_length)
    for x in range(200, 240):
        y = cy - (x - 200) // 2
        img[y - half_w // 2:y + half_w // 2, x] = 1.0

    # Continuation (long, should be kept)
    img[cy - half_w:cy + half_w, 200:270] = 1.0

    img = gaussian_filter(img, sigma=1.0)
    _save(img, filename)
    return img


def mixed_scene(filename="mixed_scene.tif"):
    """Scene with tubes + round blobs + dots for testing filtering."""
    h, w_img = 400, 400
    img = np.zeros((h, w_img), dtype=np.float64)

    # Myotube 1: horizontal
    img[80:110, 30:350] = 1.0

    # Myotube 2: slightly angled
    for x in range(30, 350):
        y = 200 + int((x - 30) * 0.1)
        img[y:y + 25, x] = 1.0

    # Round blob (should be filtered)
    rr, cc = disk((300, 100), 30, shape=img.shape)
    img[rr, cc] = 1.0

    # Small dots (should be filtered)
    for cx, cy in [(300, 250), (300, 300), (350, 280)]:
        rr, cc = disk((cy, cx), 5, shape=img.shape)
        img[rr, cc] = 1.0

    img = gaussian_filter(img, sigma=1.0)
    _save(img, filename)
    return img


def variable_intensity_tube(width=30, filename="variable_intensity.tif"):
    """Tube with intensity gradient along its length."""
    h, w_img = 200, 400
    img = np.zeros((h, w_img), dtype=np.float64)

    cy = h // 2
    x_start, x_end = 50, 350
    y_start = cy - width // 2
    y_end = cy + width // 2

    # Gradient along x
    for x in range(x_start, x_end):
        intensity = 0.3 + 0.7 * (x - x_start) / (x_end - x_start)
        img[y_start:y_end, x] = intensity

    img = gaussian_filter(img, sigma=1.0)
    _save(img, filename)
    return img


def _save(img: np.ndarray, filename: str):
    """Save as 16-bit TIFF."""
    import tifffile
    path = OUTPUT_DIR / filename
    # Convert to 16-bit
    img_16 = (img * 65535).astype(np.uint16)
    tifffile.imwrite(str(path), img_16)


if __name__ == "__main__":
    generate_all()
