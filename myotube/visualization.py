"""QC overlay generation and summary figures."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from skimage.measure import find_contours

logger = logging.getLogger(__name__)


def create_overlay(image: np.ndarray, labels: np.ndarray,
                   measurements: List[Dict], output_path: str,
                   pixel_size: float = 1.0):
    """Generate QC overlay image.

    Shows:
    - Original image as green-tinted background
    - Colored contours around each myotube
    - Skeleton in red
    - Perpendicular measurement lines in yellow
    - Mean diameter labels
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Green-tinted background
    h, w = image.shape[:2] if image.ndim >= 2 else image.shape
    if image.ndim == 2:
        rgb_bg = np.zeros((h, w, 3))
        rgb_bg[:, :, 1] = image  # Green channel
    else:
        rgb_bg = image[:, :, :3] if image.shape[2] >= 3 else np.stack([image[:,:,0]]*3, axis=-1)

    ax.imshow(rgb_bg, origin="upper")

    # Generate distinct colors for each myotube
    n_labels = labels.max()
    colors = _generate_colors(n_labels)

    # Draw contours
    for label_id in range(1, n_labels + 1):
        mask = (labels == label_id)
        contours = find_contours(mask.astype(float), 0.5)
        color = colors[label_id - 1]
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=color)

    # Draw skeletons, measurement lines, and labels
    for m in measurements:
        label_id = m["label_id"]
        if label_id < 1 or label_id > n_labels:
            continue
        color = colors[label_id - 1]

        # Skeleton path in red
        path = m.get("_skeleton_path", [])
        if len(path) > 1:
            path_arr = np.array(path)
            ax.plot(path_arr[:, 1], path_arr[:, 0], color="red",
                    linewidth=0.8, alpha=0.8)

        # Measurement lines in yellow
        lines = m.get("_measurement_lines", [])
        for (r1, c1), (r2, c2) in lines:
            ax.plot([c1, c2], [r1, r2], color="yellow",
                    linewidth=1.0, alpha=0.7)

        # Sample points as dots
        sample_pts = m.get("_sample_points", [])
        for r, c in sample_pts:
            ax.plot(c, r, "o", color="yellow", markersize=2)

        # Mean diameter label
        if path:
            mid_idx = len(path) // 2
            mr, mc = path[mid_idx]
            mean_d = m.get("mean_diameter", 0)
            unit = m.get("unit", "px")
            unit_short = "um" if unit == "um" else "px"
            ax.text(mc + 5, mr - 5, f"{mean_d:.1f}{unit_short}",
                    color=color, fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                              alpha=0.5, edgecolor="none"))

    ax.set_axis_off()
    ax.set_title(Path(output_path).stem, fontsize=10)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved overlay to {output_path}")


def create_summary_figure(all_measurements: List[Dict], output_path: str):
    """Create summary figure with histogram and box plot of diameters."""
    diameters = [m["mean_diameter"] for m in all_measurements if m["mean_diameter"] > 0]

    if not diameters:
        logger.warning("No valid measurements for summary figure")
        return

    unit = all_measurements[0].get("unit", "pixels")
    unit_label = "Diameter (um)" if unit == "um" else "Diameter (pixels)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(diameters, bins="auto", color="#4CAF50", edgecolor="black", alpha=0.8)
    ax1.set_xlabel(unit_label)
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Mean Myotube Diameters")
    ax1.axvline(np.mean(diameters), color="red", linestyle="--",
                label=f"Mean: {np.mean(diameters):.1f}")
    ax1.axvline(np.median(diameters), color="blue", linestyle="--",
                label=f"Median: {np.median(diameters):.1f}")
    ax1.legend()

    # Box plot
    ax2 = axes[1]
    ax2.boxplot(diameters, vert=True)
    ax2.set_ylabel(unit_label)
    ax2.set_title("Myotube Diameter Distribution")

    # Summary stats as text
    stats_text = (
        f"n = {len(diameters)}\n"
        f"Mean = {np.mean(diameters):.2f}\n"
        f"Median = {np.median(diameters):.2f}\n"
        f"SD = {np.std(diameters):.2f}\n"
        f"Min = {np.min(diameters):.2f}\n"
        f"Max = {np.max(diameters):.2f}"
    )
    ax2.text(1.3, 0.5, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary figure to {output_path}")


def _generate_colors(n: int) -> List:
    """Generate n visually distinct colors."""
    if n == 0:
        return []
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [hsv_to_rgb([h, 0.9, 0.95]) for h in hues]
    return colors
