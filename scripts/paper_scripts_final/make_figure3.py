"""Figure 3 -- automated detection of C26 conditioned-media induced atrophy.

Panel A -- Representative AF647 (Cy5) micrograph of a Control (Con_Veh)
          well, false-coloured red, 200 µm scale bar.

Panel B -- Representative AF647 (Cy5) micrograph of a C26 conditioned-media
          (C26_CM) well, false-coloured red, 200 µm scale bar.

Panel C -- Well-level mean myotube diameter for Control (Con_Veh) and C26
          conditioned media (C26_CM); bar = mean of well-means with SD
          error bars; jittered dots = individual wells; Welch's t-test
          and Cohen's d annotation in the upper right.

Panel D -- Full per-myotube diameter distribution for the same arms,
          shown as overlapping violin + strip + box plot. Median labels
          are placed *outside* each violin (Control to the left, C26 to
          the right) so the Control label no longer overlaps the C26
          dots.

Inputs:
  - results/c26_cm_exp2_inference/measurements.csv
  - data/real/c26_cm_exp2/Well plate_016/C1/tiff/Con_Veh_Snapshot_20260402_199.tif
  - data/real/c26_cm_exp2/Well plate_016/C4/tiff/C26_CM_Snapshot_20260402_225.tif

Outputs (Cell-Press quality):
  - reports/figures_final/figure3.png  (600 DPI)
  - reports/figures_final/figure3.pdf  (vector)

Run in the cellpose conda env:
    python scripts/paper_scripts_final/make_figure3.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from matplotlib.patches import Rectangle
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "c26_cm_exp2_inference" / "measurements.csv"
OUT_PNG = ROOT / "reports" / "figures_final" / "figure3.png"
OUT_PDF = ROOT / "reports" / "figures_final" / "figure3.pdf"

# Representative micrographs (user-selected; verified by cross-correlation).
TIFF_ROOT = ROOT / "data" / "real" / "c26_cm_exp2" / "Well plate_016"
REP_CONTROL = TIFF_ROOT / "C1" / "tiff" / "Con_Veh_Snapshot_20260402_199.tif"
REP_C26 = TIFF_ROOT / "C4" / "tiff" / "C26_CM_Snapshot_20260402_225.tif"

# Acquisition calibration + display stretch (shared with figure 2).
UM_PER_PX = 0.65
SCALEBAR_UM = 200.0
STRETCH_LO_PCT = 1.0
STRETCH_HI_PCT = 99.5

# ── Shared Cell Press style (unified width + type scale, myomeasure.figstyle) ──
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from myomeasure import figstyle
plt.rcParams.update(figstyle.RC)

C_CON = "#4CAF50"
C_C26 = "#E53935"

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", ha="left", va="top")


# ── Representative micrograph helpers (panels A, B) ──────────────────

def red_false_color(path: Path) -> np.ndarray:
    """Single-channel TIFF -> RGB with intensity in the red channel
    (AF647 / Cy5 convention), robust per-image contrast stretch."""
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
    ax.text((x0 + x1) / 2, y - h - img_w_px * 0.015,
            f"{int(SCALEBAR_UM)} µm", color="white", ha="center", va="bottom",
            fontsize=7, fontweight="bold", zorder=6)


def draw_micrograph(ax, path: Path, label: str, color: str) -> None:
    rgb = red_false_color(path)
    ax.imshow(rgb, origin="upper", interpolation="nearest")
    add_scalebar(ax, rgb.shape[1])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(2.2)
    ax.set_title(label, color=color, fontweight="bold", pad=3)


def sig_text(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(CSV)
    df["condition"] = df["image"].str.extract(r"^(.+?)_Snapshot")[0]
    df["well"]      = df["group"].str.extract(r"/([A-Z]\d+)$")[0]
    df["plate"]     = df["group"].str.extract(r"plate_(\d+)")[0]
    df["plate_well"] = df["plate"] + "_" + df["well"]
    df = df[df["mean_diameter"] > 0]

    sub = df[df["condition"].isin(["Con_Veh", "C26_CM"])].copy()
    sub["condition"] = pd.Categorical(sub["condition"],
                                      categories=["Con_Veh", "C26_CM"], ordered=True)
    well_means = (sub.groupby(["condition", "plate_well"], observed=True)
                     ["mean_diameter"].mean().reset_index())
    return sub, well_means


# ── Panel C: well-level means with Welch + Cohen's d ─────────────────

def panel_c_well_means(ax, sub: pd.DataFrame, well_means: pd.DataFrame) -> None:
    groups = ["Con_Veh", "C26_CM"]
    labels = ["Control", "C26 CM"]
    colors = [C_CON, C_C26]

    group_mean = well_means.groupby("condition", observed=True)["mean_diameter"].mean()
    group_sd   = well_means.groupby("condition", observed=True)["mean_diameter"].std()
    x_pos = np.arange(len(groups))

    bar_vals = [group_mean[g] for g in groups]
    ax.bar(x_pos, bar_vals, color=colors, alpha=0.25, edgecolor=colors,
           linewidth=1.2, width=0.55)

    rng = np.random.default_rng(42)
    for i, g in enumerate(groups):
        wells = well_means[well_means["condition"] == g]
        jitter = rng.uniform(-0.07, 0.07, size=len(wells))
        ax.scatter(x_pos[i] + jitter, wells["mean_diameter"],
                   color=colors[i], edgecolor="black", s=40, zorder=5, linewidth=0.5)

    for i, g in enumerate(groups):
        gm = group_mean[g]
        ax.hlines(gm, x_pos[i] - 0.18, x_pos[i] + 0.18,
                  color="black", linewidth=1.4, zorder=6)
        ax.errorbar(x_pos[i], gm, yerr=group_sd[g], fmt="none",
                    color="black", capsize=4, capthick=0.9, linewidth=0.9, zorder=4)

    # Welch's t-test and Cohen's d
    con_w = well_means[well_means["condition"] == "Con_Veh"]["mean_diameter"].values
    c26_w = well_means[well_means["condition"] == "C26_CM"]["mean_diameter"].values
    t_stat, p_welch = stats.ttest_ind(con_w, c26_w, equal_var=False)
    delta_abs = c26_w.mean() - con_w.mean()
    delta_pct = 100.0 * delta_abs / con_w.mean()

    # Well-level Cohen's d -- matches the inferential unit of the Welch
    # test plotted on this panel.
    pooled_sd_well = np.sqrt(((len(con_w) - 1) * con_w.var(ddof=1) +
                              (len(c26_w) - 1) * c26_w.var(ddof=1)) /
                             (len(con_w) + len(c26_w) - 2))
    cohen_d_well = (c26_w.mean() - con_w.mean()) / pooled_sd_well

    y_top = max(bar_vals) + max(group_sd.values) + 0.7
    ax.plot([0, 0, 1, 1], [y_top, y_top + 0.20, y_top + 0.20, y_top],
            color="black", lw=0.8)
    ax.text(0.5, y_top + 0.30, sig_text(p_welch),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.97, 0.55,
            f"Welch's t-test (well-level)\n"
            f"t = {t_stat:.2f}, p = {p_welch:.4f}\n"
            f"Δ = {delta_abs:.2f} µm ({delta_pct:.1f}%)\n"
            f"Cohen's d = {cohen_d_well:.2f} (well-level)",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FFF7E0", edgecolor="#888",
                      linewidth=0.5, alpha=0.95))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean myotube diameter (µm)")
    ax.set_title("Well-level means",
                 fontweight="bold", loc="center")
    ax.set_ylim(0, y_top + 1.2)


# ── Panel D: per-myotube distribution (violin + box + strip) ─────────

def panel_d_distribution(ax, sub: pd.DataFrame) -> None:
    groups = ["Con_Veh", "C26_CM"]
    labels = ["Control", "C26 CM"]
    colors = [C_CON, C_C26]
    x_pos = np.arange(len(groups))

    violin_data = [sub[sub["condition"] == g]["mean_diameter"].values for g in groups]
    parts = ax.violinplot(violin_data, positions=x_pos, showmeans=False,
                          showmedians=False, showextrema=False, widths=0.7)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.35)
        body.set_edgecolor(colors[i])
        body.set_linewidth(1.0)

    rng = np.random.default_rng(7)
    for i, g in enumerate(groups):
        vals = violin_data[i]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x_pos[i] + jitter, vals, color=colors[i], alpha=0.07,
                   s=2.0, zorder=3, rasterized=True)

    bp = ax.boxplot(violin_data, positions=x_pos, widths=0.13,
                    patch_artist=True, showfliers=False, zorder=4)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor("white")
        box.set_edgecolor(colors[i])
        box.set_linewidth(1.0)
        box.set_alpha(0.95)
    for el in ["whiskers", "caps"]:
        for line in bp[el]:
            line.set_color("gray")
            line.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(1.4)

    all_vals = np.concatenate(violin_data)
    y_top = np.percentile(all_vals, 99.5) + 5

    # Median annotations: labels sit well above each violin with a
    # steep (>>45°) arrow pointing back to the actual boxplot median
    # line. Control label is shifted ~¾ label-width LEFT of its violin
    # centre; C26 label is shifted ~½ label-width RIGHT of its violin
    # centre. Arrow tips land on the outer end of each median line
    # (boxplot half-width ≈ 0.065 axis units).
    label_dy = 13.0
    label_dx_control = 0.375   # ≈ ¾ label-width to the LEFT of the Control violin
    label_dx_c26     = 0.25    # ≈ ½ label-width to the RIGHT of the C26 violin
    median_half_w    = 0.065   # boxplot half-width (matches widths=0.13 above)
    placements = [
        (0, x_pos[0] - label_dx_control, "center", x_pos[0] - median_half_w),  # Control
        (1, x_pos[1] + label_dx_c26,     "center", x_pos[1] + median_half_w),  # C26
    ]
    for i, x_label, halign, x_arrow_tip in placements:
        med = np.median(violin_data[i])
        ax.annotate(f"median {med:.1f} µm",
                    xy=(x_arrow_tip, med),
                    xytext=(x_label, med + label_dy),
                    ha=halign, va="center",
                    fontsize=7, fontweight="bold", color=colors[i],
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor=colors[i], linewidth=0.6, alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=0.6,
                                    connectionstyle="arc3,rad=0"))
        ax.text(x_pos[i], y_top, f"n = {len(violin_data[i])}", ha="center",
                fontsize=7, color="gray")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.85, len(groups) - 1 + 0.95)
    ax.set_ylim(0, y_top + 3)
    ax.set_ylabel("Myotube diameter (µm)")
    ax.set_title("All measurements", fontweight="bold", loc="center")


# ── Driver ────────────────────────────────────────────────────────────

def main() -> None:
    sub, well_means = load_data()
    print(f"Con_Veh: n_wells={(well_means.condition=='Con_Veh').sum()}, "
          f"myotubes={int((sub.condition=='Con_Veh').sum())}")
    print(f"C26_CM:  n_wells={(well_means.condition=='C26_CM').sum()}, "
          f"myotubes={int((sub.condition=='C26_CM').sum())}")

    # Two rows: representative micrographs (A, B) over the quantitative
    # panels (C, D), at Cell Press double-column width.
    fig = plt.figure(figsize=(figstyle.FIG_W, 6.7))
    # One 2x2 grid with equal columns so the micrographs (A, B) sit directly
    # above the quantitative panels (C, D) with aligned left/right edges. A
    # single wspace is a compromise between the tight micrograph pair and the
    # room panel D's y-axis label needs.
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92],
                          hspace=0.20, wspace=0.24,
                          left=0.085, right=0.985, top=0.955, bottom=0.075)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # C_CON / C_C26 are read at call time so make_all_figures.py's palette
    # override (Okabe-Ito) propagates to the micrograph frames too.
    draw_micrograph(axA, REP_CONTROL, "Control", C_CON)
    draw_micrograph(axB, REP_C26, "C26 CM", C_C26)
    panel_c_well_means(axC, sub, well_means)
    panel_d_distribution(axD, sub)

    for ax, letter in zip([axA, axB], "AB"):
        ax.text(-0.02, 1.13, letter, transform=ax.transAxes, **PANEL_LABEL_KW)
    for ax, letter in zip([axC, axD], "CD"):
        ax.text(-0.14, 1.07, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF,             bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
