"""Figure 4 -- three blinded reviewers + the pipeline on the same 18
C26 conditioned-media subset images.

Five panels (no panel D from the v3.1 draft; image-1 sensitivity removed):
  A. Rater 1 vs pipeline image-mean scatter.
  B. Rater 2 vs pipeline image-mean scatter.
  C. Rater 3 vs pipeline image-mean scatter.
  D. Well-level C26 atrophy across reviewers 1-3 + pipeline (n = 3 wells / arm)
     with per-rater Welch t / p annotations above the data.
  E. Strictness spectrum: % of measurements <10 µm per arm, four bars
     per arm (R1, R2, R3, pipeline). The unblinded operator (Exp 1)
     reference has been removed.

Reads:
  - data/subset_blinded_validation/_paired4_per_image.csv
  - data/subset_blinded_validation/_reviewer{,2_NJ,3_PL}_long.csv
  - data/subset_blinded_validation/_automated_subset.csv

Outputs (Cell-Press quality):
  - reports/figures_final/figure4.png  (600 DPI)
  - reports/figures_final/figure4.pdf  (vector)

Run in the cellpose conda env:
    python scripts/paper_scripts_final/make_figure4.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "data" / "subset_blinded_validation"
OUT_PNG = ROOT / "reports" / "figures_final" / "figure4.png"
OUT_PDF = ROOT / "reports" / "figures_final" / "figure4.pdf"

# ── Shared Cell Press style (myomeasure.figstyle). This figure renders WIDER
# than the 6.85" column so its three aspect-equal scatter panels (A-C) are not
# squished; figstyle.rc() width-compensates the fonts so the type still renders
# at the shared ~11 pt once the journal scales the figure down to 174 mm. ──
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from myomeasure import figstyle
FIG_WIDTH = 7.5
# bbox_inches="tight" expands the SAVED width to ~7.72" (the y-labels/titles add
# ~0.22"), so compensate the font scaling against that rendered width, not the
# nominal figsize, to keep tick labels at the shared ~11 pt once scaled to 174 mm.
plt.rcParams.update(figstyle.rc(FIG_WIDTH * 1.03))

C_CTRL = "#4CAF50"
C_C26  = "#E53935"
C_R1   = "#1565C0"
C_R2   = "#D81B60"
C_R3   = "#00796B"
C_PIPE = "#F57C00"

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", ha="left", va="top")


# ── Scatter panels (A, B, C) ─────────────────────────────────────────

def scatter_panel(ax, paired: pd.DataFrame, x_col: str, y_col: str,
                  x_label: str, y_label: str, title: str) -> None:
    for cond, color, label in [("Control", C_CTRL, "Control"),
                                ("C26_CM",  C_C26,  "C26 CM")]:
        sub = paired[paired.condition == cond].dropna(subset=[x_col, y_col])
        ax.scatter(sub[x_col], sub[y_col], color=color, edgecolor="black",
                   s=36, linewidth=0.4, label=label, zorder=5)

    lo, hi = 6, 30
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.6, alpha=0.5)

    valid = paired[[x_col, y_col]].dropna()
    r_all, p_all = stats.pearsonr(valid[x_col], valid[y_col])
    ax.text(0.03, 0.97,
            f"All images: r = {r_all:.3f}\n(n = {len(valid)}, p = {p_all:.1e})",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF7E0",
                      edgecolor="#888", linewidth=0.5, alpha=0.95))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight="bold", loc="center")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.2)


# ── Panel D: well-level C26 atrophy ──────────────────────────────────

def well_level_panel(ax, r1: pd.DataFrame, r2: pd.DataFrame,
                     r3: pd.DataFrame, auto: pd.DataFrame) -> None:
    rng = np.random.default_rng(13)
    configs = [
        ("Rater 1", r1,   "diameter_um",   0.0),
        ("Rater 2", r2,   "diameter_um",   1.8),
        ("Rater 3", r3,   "diameter_um",   3.6),
        ("Pipeline",   auto, "mean_diameter", 5.4),
    ]
    for name, df, col, xc in configs:
        well = df.groupby(["well", "condition"])[col].mean().reset_index()
        ctrl = well[well.condition == "Control"][col].values
        c26  = well[well.condition == "C26_CM"][col].values
        for x_off, vals, color in [(-0.34, ctrl, C_CTRL), (+0.34, c26, C_C26)]:
            j = rng.uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(xc + x_off + j, vals, color=color, edgecolor="black",
                       s=36, linewidth=0.4, zorder=5)
            ax.hlines(vals.mean(), xc + x_off - 0.20, xc + x_off + 0.20,
                      color="black", lw=1.4, zorder=6)
        t, pv = stats.ttest_ind(ctrl, c26, equal_var=False)
        d = c26.mean() - ctrl.mean()
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        ax.text(xc, 24.0,
                f"{sig}\nΔ {d:+.2f} µm\np = {pv:.3f}",
                ha="center", va="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.22",
                          facecolor="#FFF7E0", edgecolor="#bdbdbd",
                          linewidth=0.5, alpha=0.95))

    ax.set_xticks([cfg[3] for cfg in configs])
    ax.set_xticklabels([cfg[0] for cfg in configs], rotation=45, ha="right")
    ax.set_ylabel("Well-mean diameter (µm)")
    ax.set_title("Well-level C26 atrophy",
                 fontweight="bold", loc="center")
    ax.set_ylim(8, 27)
    ax.set_xlim(-0.9, 6.4)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="None", color=C_CTRL,
               markersize=5, markeredgecolor="black", markeredgewidth=0.4,
               label="Control"),
        Line2D([0], [0], marker="o", linestyle="None", color=C_C26,
               markersize=5, markeredgecolor="black", markeredgewidth=0.4,
               label="C26 CM"),
    ], loc="lower right", framealpha=0.95, handletextpad=0.4,
       borderpad=0.3, labelspacing=0.2)


# ── Panel E: strictness spectrum ─────────────────────────────────────

def lower_tail_panel(ax, r1: pd.DataFrame, r2: pd.DataFrame,
                     r3: pd.DataFrame, auto: pd.DataFrame) -> None:
    arms = ["Control", "C26 / atrophic"]

    def lt_pct(df: pd.DataFrame, col: str, arm_raw: str) -> float:
        return 100.0 * (df[df.condition == arm_raw][col] < 10).mean()

    series = [
        ({"Control":        lt_pct(r1,   "diameter_um",   "Control"),
          "C26 / atrophic": lt_pct(r1,   "diameter_um",   "C26_CM")},
         C_R1, "Rater 1"),
        ({"Control":        lt_pct(r2,   "diameter_um",   "Control"),
          "C26 / atrophic": lt_pct(r2,   "diameter_um",   "C26_CM")},
         C_R2, "Rater 2"),
        ({"Control":        lt_pct(r3,   "diameter_um",   "Control"),
          "C26 / atrophic": lt_pct(r3,   "diameter_um",   "C26_CM")},
         C_R3, "Rater 3"),
        ({"Control":        lt_pct(auto, "mean_diameter", "Control"),
          "C26 / atrophic": lt_pct(auto, "mean_diameter", "C26_CM")},
         C_PIPE, "Pipeline"),
    ]
    x = np.arange(len(arms))
    w = 0.18
    n = len(series)
    offsets = (np.arange(n) - (n - 1) / 2) * w
    for (vals, color, label), off in zip(series, offsets):
        bars = ax.bar(x + off, [vals[a] for a in arms], w,
                      color=color, edgecolor="black", linewidth=0.5,
                      label=label)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.7,
                    f"{h:.1f}%", ha="center", fontsize=6.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylabel("% of measurements < 10 µm")
    ax.set_title("Strictness spectrum",
                 fontweight="bold", loc="center")
    ax.set_ylim(0, 55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
              fontsize=6.5, framealpha=0.95, ncols=4,
              handletextpad=0.3, columnspacing=0.7,
              borderpad=0.3, labelspacing=0.2)


# ── Driver ───────────────────────────────────────────────────────────

def main() -> None:
    r1   = pd.read_csv(SUBSET / "reviewer1_long.csv")
    r2   = pd.read_csv(SUBSET / "reviewer2_long.csv")
    r3   = pd.read_csv(SUBSET / "reviewer3_long.csv")
    auto = pd.read_csv(SUBSET / "_automated_subset.csv")
    paired = pd.read_csv(SUBSET / "_paired4_per_image.csv")

    print(f"Rater 1: {len(r1)} myotubes")
    print(f"Rater 2: {len(r2)} myotubes")
    print(f"Rater 3: {len(r3)} myotubes")
    print(f"Pipeline:   {len(auto)} myotubes")
    print(f"Paired:     {len(paired)} images")

    # Layout: top row = three scatters (A, B, C); bottom row two
    # wider panels (D, E) at Cell-Press double-column width (~7").
    fig = plt.figure(figsize=(FIG_WIDTH, 6.3))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.12],
                          hspace=0.40, wspace=1.18,
                          left=0.075, right=0.985, top=0.95, bottom=0.075)
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2:4])
    axC = fig.add_subplot(gs[0, 4:6])
    axD = fig.add_subplot(gs[1, 0:3])
    axE = fig.add_subplot(gs[1, 3:6])

    scatter_panel(axA, paired, "r1_mean", "auto_mean",
                  "Rater 1 image-mean (µm)",
                  "Pipeline image-mean (µm)",
                  "Rater 1 vs pipeline")
    scatter_panel(axB, paired, "r2_mean", "auto_mean",
                  "Rater 2 image-mean (µm)",
                  "Pipeline image-mean (µm)",
                  "Rater 2 vs pipeline")
    scatter_panel(axC, paired, "r3_mean", "auto_mean",
                  "Rater 3 image-mean (µm)",
                  "Pipeline image-mean (µm)",
                  "Rater 3 vs pipeline")
    well_level_panel(axD, r1, r2, r3, auto)
    lower_tail_panel(axE, r1, r2, r3, auto)

    # A-D sit far enough left to clear their own y-axis; E is the right-hand
    # bottom panel, so a large negative offset would land on panel D's (long)
    # title -- give it a small offset that stays over its own y-axis.
    # A-C are aspect-equal (square) scatters: their long y-label overflows the
    # top-left corner, so anchor the letter at a FIXED point offset to the left
    # of the axes box (robust to the aspect shrink) rather than an axis-fraction
    # position that would land on the y-label. D and E fill their cells, so a
    # normal axis-fraction offset is fine (E small, to clear panel D's title).
    for ax, letter in zip([axA, axB, axC], "ABC"):
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-50, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=11, fontweight="bold")
    axD.text(-0.15, 1.10, "D", transform=axD.transAxes, **PANEL_LABEL_KW)
    axE.text(-0.05, 1.10, "E", transform=axE.transAxes, **PANEL_LABEL_KW)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF,             bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
