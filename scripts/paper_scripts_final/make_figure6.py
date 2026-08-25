"""Figure 6 - distribution shape across raters.

Four panels (2 x 2):
  A. Per-rater diameter density curves (KDE) on the Control arm, with the
     pipeline lognormal expectation overlaid.
  B. Same KDE for the C26 CM arm.
  C. Per-image Shapiro-Wilk pass-rate clustered bar chart: fraction of 18
     images normal at p > 0.05 vs lognormal at p > 0.05, four raters.
  D. Q-Q grid: one representative C26 image per rater plotted against the
     log-normal reference line on log-axis.

Reads:
  - data/subset_blinded_validation/_reviewer{,2_NJ,3_PL}_long.csv
  - data/subset_blinded_validation/_automated_subset.csv
  - reports/stats_final/normality_per_image.csv

Outputs (Cell Press quality):
  - reports/figures_final/figure6.png  (600 DPI)
  - reports/figures_final/figure6.pdf  (vector)

Run in the cellpose conda env:
    python scripts/paper_scripts_final/make_figure6.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "data" / "subset_blinded_validation"
NORM_CSV = ROOT / "reports" / "stats_final" / "normality_per_image.csv"
OUT_PNG = ROOT / "reports" / "figures_final" / "figure6.png"
OUT_PDF = ROOT / "reports" / "figures_final" / "figure6.pdf"

# ── Shared Cell Press style (unified width + type scale, myomeasure.figstyle) ──
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from myomeasure import figstyle
plt.rcParams.update(figstyle.RC)

C_R1 = "#1565C0"
C_R2 = "#D81B60"
C_R3 = "#00796B"
C_PIPE = "#F57C00"

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", ha="left", va="top")


def kde(ax, samples, color, label, xs, lw=1.4):
    if len(samples) < 5:
        return
    kde_obj = stats.gaussian_kde(samples)
    ax.plot(xs, kde_obj(xs), color=color, lw=lw, label=label)


def kde_panel(ax, r1, r2, r3, auto, arm: str, title: str,
              show_legend: bool):
    xs = np.linspace(0, 50, 400)
    for tag, df, col, color in [
        ("Rater 1", r1, "diameter_um", C_R1),
        ("Rater 2", r2, "diameter_um", C_R2),
        ("Rater 3", r3, "diameter_um", C_R3),
        ("Pipeline",   auto, "mean_diameter", C_PIPE),
    ]:
        v = df[df.condition == arm][col].values
        v = v[v > 0]
        kde(ax, v, color, tag, xs)
    v_pipe = auto[auto.condition == arm]["mean_diameter"].values
    v_pipe = v_pipe[v_pipe > 0]
    if len(v_pipe):
        shape, loc, scale = stats.lognorm.fit(v_pipe, floc=0)
        ax.plot(xs, stats.lognorm.pdf(xs, shape, loc, scale),
                color="black", lw=0.9, ls="--",
                label="Lognormal (pipeline)")
    ax.set_xlim(0, 45)
    # Headroom above the tallest density so the upper-right legend sits over the
    # low tail instead of on top of the curves.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.30)
    ax.set_xlabel("Myotube diameter (µm)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold", loc="center")
    if show_legend:
        ax.legend(loc="upper right", framealpha=0.95,
                  handlelength=1.4, handletextpad=0.4,
                  borderpad=0.3, labelspacing=0.22)


def passrate_panel(ax, pi):
    """Stacked bars per rater: count of images whose per-image diameter
    distribution is compatible with normality / lognormality (Shapiro-Wilk
    fails to reject at p > 0.05). Failing to reject is not a confirmation of
    (log)normality -- it only indicates no significant departure was detected.

    Counts use the power-matched per-image samples (each image downsampled to
    the minimum per-image n across raters; seeded). The raw / rater-specific
    counts are in `normality_per_image.csv` but are not plotted on the
    headline figure because they are power-confounded: pipeline images carry
    ~123-356 myotubes vs ~38-117 for reviewers, so the raw count is not a fair
    distributional comparison across raters.
    """
    raters = ["reviewer 1", "reviewer 2", "reviewer 3", "pipeline"]
    rater_label = {"reviewer 1": "R1", "reviewer 2": "R2",
                   "reviewer 3": "R3", "pipeline": "Pipeline"}
    norm_counts = []
    log_counts = []
    for r in raters:
        sub = pi[pi.rater == r]
        norm_counts.append(int((sub.shapiro_p_pm > 0.05).sum()))
        log_counts.append(int((sub.shapiro_log_p_pm > 0.05).sum()))
    x = np.arange(len(raters))
    w = 0.36
    ax.bar(x - w/2, norm_counts, w, color="#90CAF9", edgecolor="black",
           linewidth=0.5, label="Compatible w/ normality")
    ax.bar(x + w/2, log_counts, w, color="#FFB74D", edgecolor="black",
           linewidth=0.5, label="Compatible w/ lognormality")
    for i, (n, l) in enumerate(zip(norm_counts, log_counts)):
        ax.text(i - w/2, n + 0.4, f"{n}/18", ha="center",
                fontsize=6.5, fontweight="bold")
        ax.text(i + w/2, l + 0.4, f"{l}/18", ha="center",
                fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([rater_label[r] for r in raters])
    ax.set_ylabel("Images compatible (p > 0.05)")
    ax.set_title("Per-image Shapiro-Wilk compatibility",
                 fontweight="bold", loc="center")
    ax.set_ylim(0, 21)
    ax.legend(loc="upper left", framealpha=0.95,
              handletextpad=0.4, borderpad=0.3, labelspacing=0.25)


def qq_panel(ax, r1, r2, r3, auto):
    target_bid = 7
    series = [
        ("Rater 1", r1, "diameter_um", C_R1),
        ("Rater 2", r2, "diameter_um", C_R2),
        ("Rater 3", r3, "diameter_um", C_R3),
        ("Pipeline",   auto, "mean_diameter", C_PIPE),
    ]
    qs = np.linspace(0.01, 0.99, 50)
    for tag, df, col, color in series:
        v = df[df.blinded_id == target_bid][col].values
        v = v[v > 0]
        if len(v) < 5:
            continue
        log_v = np.log(v)
        emp = np.quantile(log_v, qs)
        thr = stats.norm.ppf(qs, loc=log_v.mean(), scale=log_v.std(ddof=1))
        ax.plot(thr, emp, marker="o", linestyle="-", color=color, lw=0.9,
                ms=3, mec="black", mew=0.3, label=tag)
    rng = (1.5, 4.0)
    ax.plot(rng, rng, "k--", lw=0.6, alpha=0.6)
    ax.set_xlim(rng); ax.set_ylim(rng)
    ax.set_aspect("equal")
    ax.set_xlabel("Theoretical lognormal quantile (log µm)")
    ax.set_ylabel("Sample quantile (log µm)")
    ax.set_title(f"Q-Q vs lognormal (image {target_bid}, C26 CM)",
                 fontweight="bold", loc="center")
    ax.legend(loc="lower right", framealpha=0.95,
              handletextpad=0.4, borderpad=0.3, labelspacing=0.25)


def main() -> None:
    r1   = pd.read_csv(SUBSET / "reviewer1_long.csv")
    r2   = pd.read_csv(SUBSET / "reviewer2_long.csv")
    r3   = pd.read_csv(SUBSET / "reviewer3_long.csv")
    auto = pd.read_csv(SUBSET / "_automated_subset.csv")
    pi   = pd.read_csv(NORM_CSV)

    print(f"Rater 1: {len(r1)} myotubes")
    print(f"Rater 2: {len(r2)} myotubes")
    print(f"Rater 3: {len(r3)} myotubes")
    print(f"Pipeline:   {len(auto)} myotubes")

    fig = plt.figure(figsize=(figstyle.FIG_W, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.32,
                          left=0.085, right=0.985, top=0.965, bottom=0.075)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    kde_panel(axA, r1, r2, r3, auto, arm="Control",
              title="Control: per-rater density", show_legend=True)
    kde_panel(axB, r1, r2, r3, auto, arm="C26_CM",
              title="C26 CM: per-rater density", show_legend=False)
    passrate_panel(axC, pi)
    qq_panel(axD, r1, r2, r3, auto)

    # A-C fill their cells, so an axis-fraction offset is fine. D is aspect-equal
    # (square Q-Q), so its y-label reaches the top-left corner -- anchor its
    # letter at a fixed point offset left of the axes box instead.
    for ax, letter in zip([axA, axB, axC], "ABC"):
        ax.text(-0.16, 1.10, letter, transform=ax.transAxes,
                **PANEL_LABEL_KW)
    axD.annotate("D", xy=(0, 1), xycoords="axes fraction",
                 xytext=(-50, 2), textcoords="offset points",
                 ha="left", va="bottom", fontsize=11, fontweight="bold")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF,             bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
