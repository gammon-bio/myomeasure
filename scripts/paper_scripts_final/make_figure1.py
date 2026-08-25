"""Figure 1 panels B and C -- the measurement pipeline on one field.

Panel A (schematic) is built separately in BioRender. This script renders:

  B. Raw fluorescence image (MF-20 / AF647), false-coloured red.
  C. Pipeline output for the SAME field: filtered masks + 9-point ray-cast
     diameter measurements, drawn over a red-tissue background. The
     skeleton is recoloured white (the pipeline default red would vanish
     on the red tissue); mask contours and measurement rays are kept.

Both panels are rendered at the native 2304x2304 frame so they are the
same size. Panel titles and the per-image filename are intentionally
omitted (added in PowerPoint during assembly).

Source images (C26 CM validation experiment, plate_016 well B1):
  raw   - data/real/c26_cm_exp2/Well plate_016/B1/tiff/Con_Veh_Snapshot_20260402_192.tif
  masks - data/real/c26_cm_exp2/Well plate_016/B1/tiff/Con_Veh_Snapshot_20260402_192_cp_masks.tif

Outputs (Cell-Press quality):
  - reports/figures_final/figure1.png  (600 DPI)
  - reports/figures_final/figure1.pdf

Run in the cellpose conda env:
    python scripts/paper_scripts_final/make_figure1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.measure import find_contours

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from myotube.config import load_config
from myotube.filtering import filter_myotubes
from myotube.io import parse_pixel_size
from myotube.measurement import measure_all_myotubes
from myotube.visualization import _generate_colors

RAW_TIF = (ROOT / "data" / "real" / "c26_cm_exp2" / "Well plate_016" / "B1"
           / "tiff" / "Con_Veh_Snapshot_20260402_192.tif")
MASKS_TIF = RAW_TIF.with_name(RAW_TIF.stem + "_cp_masks.tif")
OUT_PNG = ROOT / "reports" / "figures_final" / "figure1.png"
OUT_PDF = ROOT / "reports" / "figures_final" / "figure1.pdf"

UM_PER_PX = 0.65
SCALEBAR_UM = 100.0
STRETCH_LO_PCT = 1.0
STRETCH_HI_PCT = 99.5
SKELETON_COLOR = "white"   # pipeline default is red; invisible on red tissue

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def red_false_color(path: Path) -> np.ndarray:
    """Single-channel TIFF -> RGB with intensity in the red channel."""
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img.reshape(img.shape[-2], img.shape[-1]) if img.shape[0] == 1 \
            else img.max(axis=0)
    lo = np.percentile(img, STRETCH_LO_PCT)
    hi = np.percentile(img, STRETCH_HI_PCT)
    norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = np.zeros((*norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = norm
    return rgb


def add_scalebar(ax, img_w_px: int) -> None:
    bar_px = SCALEBAR_UM / UM_PER_PX
    margin = img_w_px * 0.04
    x1 = img_w_px - margin
    x0 = x1 - bar_px
    y = img_w_px - margin
    h = img_w_px * 0.012
    ax.add_patch(Rectangle((x0, y - h), bar_px, h,
                           facecolor="white", edgecolor="none", zorder=6))
    ax.text((x0 + x1) / 2, y - h - img_w_px * 0.018,
            f"{int(SCALEBAR_UM)} µm", color="white", ha="center", va="bottom",
            fontsize=8, fontweight="bold", zorder=6)


def clean_micrograph_ax(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_overlay(ax, rgb_bg: np.ndarray, labels: np.ndarray,
                 measurements: list) -> None:
    """Reproduce the pipeline measurement overlay on a red-tissue panel:
    per-myotube mask contours, white skeleton, yellow 9-point rays + sample
    points, and per-myotube diameter labels. Line art is rasterised to keep
    the vector PDF small."""
    ax.imshow(rgb_bg, origin="upper", interpolation="nearest")
    n_labels = int(labels.max())
    colors = _generate_colors(n_labels)

    for label_id in range(1, n_labels + 1):
        for contour in find_contours((labels == label_id).astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.0,
                    color=colors[label_id - 1], rasterized=True)

    for m in measurements:
        label_id = m["label_id"]
        if label_id < 1 or label_id > n_labels:
            continue
        color = colors[label_id - 1]
        path = m.get("_skeleton_path", [])
        if len(path) > 1:
            p = np.array(path)
            ax.plot(p[:, 1], p[:, 0], color=SKELETON_COLOR, linewidth=0.8,
                    alpha=0.9, rasterized=True)
        for (r1, c1), (r2, c2) in m.get("_measurement_lines", []):
            ax.plot([c1, c2], [r1, r2], color="yellow", linewidth=1.0,
                    alpha=0.7, rasterized=True)
        for r, c in m.get("_sample_points", []):
            ax.plot(c, r, "o", color="yellow", markersize=1.6, rasterized=True)
        if path:
            mr, mc = path[len(path) // 2]
            ax.text(mc + 5, mr - 5, f"{m.get('mean_diameter', 0):.1f}um",
                    color=color, fontsize=4.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                              alpha=0.5, edgecolor="none"))
    ax.set_xlim(0, rgb_bg.shape[1]); ax.set_ylim(rgb_bg.shape[0], 0)


def main() -> None:
    config = load_config([])
    pixel_size, unit = parse_pixel_size(str(RAW_TIF), config.pixel_size)

    rgb = red_false_color(RAW_TIF)
    masks = tifffile.imread(MASKS_TIF).astype(np.int32)
    filtered = filter_myotubes(masks, config, pixel_size)
    measurements = measure_all_myotubes(filtered, config, pixel_size, unit)
    print(f"Figure 1: {int(filtered.max())} myotubes measured on "
          f"{RAW_TIF.name} ({rgb.shape[0]}x{rgb.shape[1]} px)")

    fig = plt.figure(figsize=(10.0, 5.0))
    gs = fig.add_gridspec(1, 2, wspace=0.04,
                          left=0.01, right=0.99, top=0.99, bottom=0.01)
    axB = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])

    axB.imshow(rgb, origin="upper", interpolation="nearest")
    add_scalebar(axB, rgb.shape[1])
    clean_micrograph_ax(axB)

    draw_overlay(axC, rgb, filtered, measurements)
    clean_micrograph_ax(axC)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
