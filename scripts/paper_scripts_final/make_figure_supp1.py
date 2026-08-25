"""Supplementary Figure 1 -- C26 conditioned media remodels the myotube
diameter distribution at the well level.

Panels (each = one well-level metric, 3 wells per arm):
  A. Median diameter per well (µm)
  B. IQR per well (µm)
  C. Fraction of myotubes < 15 µm per well (%)
  D. Fraction of myotubes > 20 µm per well (%)

Bar = mean across the 3 wells, error bar = SD across wells, dots = the
three individual wells (jittered). Welch's t-test (3 vs 3) annotated
above each pair, with Benjamini-Hochberg FDR-adjusted *q*-values
reported across the 4 panels to control for multiple testing. No
object-level p-values are computed or shown.

Reads:
  - results/c26_cm_exp2_inference/measurements.csv

Outputs (Cell-Press quality):
  - reports/figures_final/figure_supp1.png  (600 DPI)
  - reports/figures_final/figure_supp1.pdf  (vector)

Run in the cellpose conda env:
    python scripts/paper_scripts_final/make_figure_supp1.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "c26_cm_exp2_inference" / "measurements.csv"
OUT_PNG = ROOT / "reports" / "figures_final" / "figure_supp1.png"
OUT_PDF = ROOT / "reports" / "figures_final" / "figure_supp1.pdf"

# ── Shared Cell Press style (unified width + type scale, myomeasure.figstyle) ──
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from myomeasure import figstyle
plt.rcParams.update(figstyle.RC)

C_CON = "#4CAF50"
C_C26 = "#E53935"
ARM_LABELS = ["Control", "C26 CM"]
ARM_KEYS   = ["Con_Veh", "C26_CM"]
ARM_COLORS = [C_CON, C_C26]

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", ha="left", va="top")


def sig_text(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_per_well() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["condition"]  = df["image"].str.extract(r"^(.+?)_Snapshot")[0]
    df["well"]       = df["group"].str.extract(r"/([A-Z]\d+)$")[0]
    df["plate"]      = df["group"].str.extract(r"plate_(\d+)")[0]
    df["plate_well"] = df["plate"] + "_" + df["well"]
    df = df[(df["mean_diameter"] > 0)
            & (df["plate"] == "016")
            & (df["condition"].isin(ARM_KEYS))]

    rows = []
    for (cond, pw), g in df.groupby(["condition", "plate_well"], observed=True):
        v = g["mean_diameter"].values
        q1, q3 = np.percentile(v, [25, 75])
        rows.append({
            "condition":   cond,
            "plate_well":  pw,
            "median_um":   float(np.median(v)),
            "iqr_um":      float(q3 - q1),
            "pct_lt_15um": float(100.0 * (v < 15).mean()),
            "pct_gt_20um": float(100.0 * (v > 20).mean()),
        })
    return pd.DataFrame(rows)


def bh_q_values(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    ranked = pvals[order]
    bh = ranked * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    out = np.empty_like(bh)
    out[order] = bh
    return out


def draw_panel(ax, well_df: pd.DataFrame, col: str, ylabel: str, title: str,
               q_value: float, y_pad_frac: float = 0.18) -> None:
    x_pos = np.arange(2)
    means, sds, sample = [], [], []
    for arm in ARM_KEYS:
        v = well_df.loc[well_df.condition == arm, col].values
        sample.append(v)
        means.append(v.mean())
        sds.append(v.std(ddof=1))

    ax.bar(x_pos, means, color=ARM_COLORS, alpha=0.25,
           edgecolor=ARM_COLORS, linewidth=1.2, width=0.55)
    ax.errorbar(x_pos, means, yerr=sds, fmt="none",
                color="black", capsize=4, capthick=0.9, linewidth=0.9, zorder=4)
    for i, arm in enumerate(ARM_KEYS):
        ax.hlines(means[i], x_pos[i] - 0.18, x_pos[i] + 0.18,
                  color="black", linewidth=1.4, zorder=6)

    rng = np.random.default_rng(7)
    for i, v in enumerate(sample):
        jitter = rng.uniform(-0.07, 0.07, size=len(v))
        ax.scatter(x_pos[i] + jitter, v,
                   color=ARM_COLORS[i], edgecolor="black",
                   s=42, linewidth=0.5, zorder=5)

    # Welch's t-test (3 vs 3 wells), well-level only.
    t, p = stats.ttest_ind(sample[0], sample[1], equal_var=False)

    # Significance bracket above the data.
    all_vals = np.concatenate(sample)
    y_data_top = all_vals.max() + max(sds) * 0.6
    y_pad = (y_data_top - all_vals.min()) * y_pad_frac
    y_brk = y_data_top + y_pad * 0.4
    ax.plot([0, 0, 1, 1],
            [y_brk, y_brk + y_pad * 0.18, y_brk + y_pad * 0.18, y_brk],
            color="black", lw=0.8)
    p_str = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.4f}"
    q_str = f"q = {q_value:.1e}" if q_value < 0.001 else f"q = {q_value:.4f}"
    ax.text(0.5, y_brk + y_pad * 0.30,
            f"{sig_text(q_value)}  ({p_str}; {q_str} BH)",
            ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ARM_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", loc="center")
    ax.set_ylim(0, y_brk + y_pad * 1.1)


def main() -> None:
    well_df = load_per_well()
    print(f"Wells per arm: "
          f"Control = {(well_df.condition=='Con_Veh').sum()}, "
          f"C26 CM = {(well_df.condition=='C26_CM').sum()}")

    fig = plt.figure(figsize=(figstyle.FIG_W, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.32,
                          left=0.085, right=0.985, top=0.91, bottom=0.075)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    panel_cols = ["median_um", "iqr_um", "pct_lt_15um", "pct_gt_20um"]
    pvals = []
    for col in panel_cols:
        con = well_df.loc[well_df.condition == "Con_Veh", col].values
        c26 = well_df.loc[well_df.condition == "C26_CM",  col].values
        _, p = stats.ttest_ind(con, c26, equal_var=False)
        pvals.append(p)
    qvals = bh_q_values(np.array(pvals))

    draw_panel(axA, well_df, panel_cols[0],
               "Median diameter (µm)",        "Median diameter per well", qvals[0])
    draw_panel(axB, well_df, panel_cols[1],
               "IQR (µm)",                    "IQR per well", qvals[1])
    draw_panel(axC, well_df, panel_cols[2],
               "Myotubes < 15 µm (%)",        "Fraction < 15 µm per well", qvals[2])
    draw_panel(axD, well_df, panel_cols[3],
               "Myotubes > 20 µm (%)",        "Fraction > 20 µm per well", qvals[3])

    for ax, letter in zip([axA, axB, axC, axD], "ABCD"):
        ax.text(-0.16, 1.07, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

    fig.suptitle("C26 CM remodels the myotube diameter distribution",
                 fontsize=10, fontweight="bold", y=0.97)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF,             bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
