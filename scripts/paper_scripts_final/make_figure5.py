"""Figure 5 -- first-image effect and contextual anchoring in Rater 1.

Five panels:
  A. Cross-rater comparison of images 1, 8, and 9 (all C26 CM) for
     reviewers 1, 2, and 3. Rater 1 reports Control-range diameters
     for all three images while reviewers 2 and 3 do not.
  B. Rater 1 pass-1 timeline (image-mean diameter in blinded session
     order; image 1 highlighted as the first measurement of the session).
  C. Rater 2 pass-1 timeline on the same 18 images, same blinded order.
  D. AB anchor-status strip -- pass-1 C26 images measured AFTER at least
     one Control well (n = 6: images 3, 6, 7, 12, 16, 17) vs C26 images
     measured WITHOUT a Control anchor in the session (n = 3: image 1
     first in session + images 8 and 9 re-measured outside the original
     pass).
  E. Image 8 (C26 CM, well B4) cross-rater contrast: reviewer 1
     re-measure (no anchor) vs reviewers 2 and 3 (blinded session,
     anchored). The original reviewer-1 pass-1 image-8 measurement is
     not available -- the reviewer-1 with-anchor reference is the mean
     of the six other in-session reviewer-1 C26 image-means
     (panel D, blue group).

Inputs:
  data/subset_blinded_validation/reviewer_measurements/reviewer1_measurements.xlsx
  data/subset_blinded_validation/reviewer_measurements/reviewer2_measurements.xlsx
  data/subset_blinded_validation/reviewer_measurements/reviewer3_measurements.xlsx
  data/subset_blinded_validation/code_sheet.xlsx

Outputs (Cell-Press quality):
  reports/figures_final/figure5.png  (600 DPI)
  reports/figures_final/figure5.pdf  (vector)

Run in the cellpose conda env:
    python scripts/paper_scripts_final/make_figure5.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "data" / "subset_blinded_validation"
RM = SUBSET / "reviewer_measurements"
OUT_PNG = ROOT / "reports" / "figures_final" / "figure5.png"
OUT_PDF = ROOT / "reports" / "figures_final" / "figure5.pdf"

# ── Shared Cell Press style (unified width + type scale, myomeasure.figstyle) ──
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from myomeasure import figstyle
plt.rcParams.update(figstyle.RC)

# Colors
CTRL_COLOR = "#4A7C8E"   # blue/teal -- Control wells / "with anchor"
C26_COLOR  = "#C44E3C"   # red       -- C26 wells / "no anchor"
DUP_COLOR  = "#9CA3AF"   # grey      -- re-measured/duplicate marker
R1_COLOR   = "#1565C0"
R2_COLOR   = "#D81B60"
R3_COLOR   = "#00796B"
PIPE_COLOR = "#F57C00"

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", ha="left", va="top")


# ── Loaders ──────────────────────────────────────────────────────────

def load_r1_current() -> pd.DataFrame:
    """Rater 1 workbook *after* the post-session re-measurements of
    images 8 and 9 (and image 1 update). Images 8 and 9 here are the
    no-anchor (isolation) re-measures."""
    return pd.read_excel(RM / "reviewer1_measurements.xlsx", sheet_name="Sheet1")


def load_r1_pass1() -> pd.DataFrame:
    """Rater 1 *original* pass-1 workbook (in-session, with anchor
    where applicable). Used for the timeline (panel B), the in-session
    image-1 first-image-effect outlier, and the with-anchor reference
    for image 8 (panel E)."""
    return pd.read_excel(RM / "reviewer1_measurements_pass1.xlsx", sheet_name="Sheet1")


def load_r2() -> pd.DataFrame:
    r2 = pd.read_excel(RM / "reviewer2_measurements.xlsx", sheet_name="Sheet1",
                       skiprows=1, header=None)
    r2.columns = list(range(1, 19))
    return r2


def load_r3() -> pd.DataFrame:
    """Wide layout: each image i occupies cols [3*(i-1)+0..2]; the
    measurement column is `3*(i-1)+1`, data starts at row 2."""
    return pd.read_excel(RM / "reviewer3_measurements.xlsx", sheet_name="Sheet1",
                         header=None)


def load_codes() -> pd.DataFrame:
    codes = pd.read_excel(SUBSET / "code_sheet.xlsx")
    codes = codes.rename(columns={"Well": "well"})
    return codes.sort_values("blinded_id").reset_index(drop=True)


def im_stats(values) -> tuple[float, float, int]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    v = v[v > 0]
    n = int(len(v))
    if n == 0:
        return float("nan"), float("nan"), 0
    sem = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(v.mean()), sem, n


def per_img(df: pd.DataFrame, codes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in range(1, 19):
        m, s, n = im_stats(df[c])
        info = codes[codes.blinded_id == c].iloc[0]
        rows.append({"order": c, "condition": info.condition,
                     "well": info.well, "mean": m, "sem": s, "n": n})
    return pd.DataFrame(rows)


def r3_image(r3: pd.DataFrame, image_id: int):
    return r3.iloc[2:, 3 * (image_id - 1) + 1]


# ── Panel A: image 1, 8, 9 across R1, R2, R3 ────────────────────────

def panel_a_cross_rater(ax, r1: pd.DataFrame, r2: pd.DataFrame,
                        r3: pd.DataFrame) -> None:
    images = [1, 8, 9]
    raters = [
        ("Rater 1", R1_COLOR, lambda i: im_stats(r1[i])),
        ("Rater 2", R2_COLOR, lambda i: im_stats(r2[i])),
        ("Rater 3", R3_COLOR, lambda i: im_stats(r3_image(r3, i))),
    ]
    n_groups = len(images)
    n_bars   = len(raters)
    width    = 0.26
    x_base   = np.arange(n_groups)

    for j, (rname, color, getter) in enumerate(raters):
        means, sems = [], []
        for img in images:
            m, s, _ = getter(img)
            means.append(m)
            sems.append(s)
        offsets = (j - (n_bars - 1) / 2) * width
        bars = ax.bar(x_base + offsets, means, width,
                      yerr=sems, color=color, edgecolor="black",
                      linewidth=0.5, error_kw=dict(lw=0.7, capsize=2.5),
                      label=rname)
        for b, m, s in zip(bars, means, sems):
            ax.text(b.get_x() + b.get_width() / 2, m + s + 0.7,
                    f"{m:.1f}", ha="center", fontsize=6.5, fontweight="bold")

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"Image {i}\n(C26 CM)" for i in images])
    ax.set_ylabel("Image-mean diameter (µm)")
    ax.set_title("Cross-rater comparison of the three contested C26 images "
                 "(1, 8, 9)", fontweight="bold", loc="center")
    ax.set_ylim(0, 28)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95,
              borderpad=0.3, labelspacing=0.2, handletextpad=0.4)


# ── Panels B and C: pass-1 timelines ─────────────────────────────────

def timeline_panel(ax, dfp: pd.DataFrame, title: str,
                   highlight_image1_text: str | None = None) -> None:
    for _, row in dfp.iterrows():
        c = CTRL_COLOR if row.condition == "Control" else C26_COLOR
        marker = "o" if row.condition == "Control" else "s"
        ax.errorbar(row.order, row["mean"], yerr=row["sem"],
                    fmt=marker, color=c, mec="k", mew=0.6, ms=6,
                    capsize=2.5, ecolor="gray", zorder=3)

    c26_pts = dfp[dfp.condition == "C26_CM"]
    ax.plot(c26_pts.order, c26_pts["mean"], "--",
            color=C26_COLOR, lw=0.9, alpha=0.5, zorder=2)

    # Reference band: range of C26 image-means with image 1 excluded
    c26_other = c26_pts[c26_pts.order != 1]
    if len(c26_other):
        ax.axhspan(c26_other["mean"].min(), c26_other["mean"].max(),
                   color=C26_COLOR, alpha=0.10, zorder=0)

    if highlight_image1_text:
        img1 = dfp.iloc[0]
        ax.annotate(highlight_image1_text,
                    xy=(1, img1["mean"]),
                    xytext=(2.6, max(img1["mean"] + 4, 26)),
                    fontsize=6.5, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.7))

    ax.set_ylabel("Image-mean (µm)")
    ax.set_title(title, fontweight="bold", loc="center")
    ax.set_ylim(0, 32)
    ax.set_xticks(range(1, 19))
    ax.set_xlim(0.4, 18.6)
    ax.grid(True, alpha=0.25, linestyle=":", zorder=0)


# ── Panel D: AB anchor-status strip ─────────────────────────────────

def panel_d_anchor_strip(ax, r1_pass1: pd.DataFrame,
                         r1_current: pd.DataFrame,
                         codes: pd.DataFrame) -> None:
    # WITH anchor: pass-1 C26 images measured AFTER ≥1 Control had been
    # done in the session. Pulled from the *original* workbook.
    # WITHOUT anchor: image 1 (first in session, pass-1) + images 8 and
    # 9 (post-session re-measures from the current workbook).
    with_anchor_imgs = [3, 6, 7, 12, 16, 17]
    in_pts = [im_stats(r1_pass1[i])[:2] for i in with_anchor_imgs]
    out_pts = [
        im_stats(r1_pass1[1])[:2],   # image 1 -- pass 1, first in session
        im_stats(r1_current[8])[:2],    # image 8 -- post-session re-measure
        im_stats(r1_current[9])[:2],    # image 9 -- post-session re-measure
    ]
    no_anchor_imgs = [1, 8, 9]

    # No inferential test is reported on this panel. The anchor partition
    # is a descriptive within-rater contrast: the 9 images come from only
    # 3 C26 wells (well-overlap between groups) and two of the three
    # WITHOUT-anchor images are pass-2 remeasurements that were triggered
    # by the anomaly itself, so an independent-samples Welch p-value
    # would not be valid. The panel reports group means and Δ only;
    # the within-rater pass-1 / pass-2 image-8 shift in panel E is the
    # complementary quantitative measure.
    in_means  = np.array([m for m, _ in in_pts])
    out_means = np.array([m for m, _ in out_pts])

    rng = np.random.default_rng(2)
    for (m, s) in in_pts:
        x = 0 + rng.uniform(-0.10, 0.10)
        ax.errorbar(x, m, yerr=s, fmt="o", color=CTRL_COLOR, mec="k",
                    mew=0.6, ms=8, capsize=2.5, ecolor="gray", zorder=3)
    xs_no = [1 - 0.18, 1.0, 1 + 0.18]
    for x, (m, s), img_id in zip(xs_no, out_pts, no_anchor_imgs):
        ax.errorbar(x, m, yerr=s, fmt="s", color=C26_COLOR, mec="k",
                    mew=0.6, ms=8, capsize=2.5, ecolor="gray", zorder=3)
        ax.text(x, m + 0.9, f"img {img_id}", ha="center", fontsize=6.5,
                fontweight="bold")

    m_anchor = float(in_means.mean())
    m_no     = float(out_means.mean())
    ax.hlines(m_anchor, -0.22, 0.22, colors=CTRL_COLOR, lw=2.2, zorder=4)
    ax.hlines(m_no,      0.78, 1.22, colors=C26_COLOR,  lw=2.2, zorder=4)
    in_top = max(m + s for m, s in in_pts)
    ax.text(0, in_top + 1.5, f"{m_anchor:.1f} µm",
            va="bottom", ha="center", fontsize=7.5,
            color=CTRL_COLOR, fontweight="bold")
    ax.text(1.30, m_no, f"{m_no:.1f} µm",
            va="center", ha="left", fontsize=7.5,
            color=C26_COLOR, fontweight="bold")

    # Δ annotation on the right (descriptive only -- see comment above
    # for why no inferential p-value is reported on this panel).
    ax.annotate("", xy=(1.65, m_no), xytext=(1.65, m_anchor),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    ax.text(1.72, (m_anchor + m_no) / 2,
            f"Δ = {m_no - m_anchor:+.1f} µm\n(descriptive)",
            fontsize=6.8, va="center")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "WITH anchor\n(n = 6 C26 imgs:\n3, 6, 7, 12, 16, 17)",
        "WITHOUT anchor\n(n = 3 C26 imgs:\n1, 8, 9)",
    ], fontsize=7)
    ax.set_xlim(-0.7, 2.2)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Image-mean diameter (µm)")
    ax.set_title("Rater 1 anchor-status strip",
                 fontweight="bold", loc="center")
    ax.grid(True, alpha=0.25, linestyle=":", zorder=0)


# ── Panel E: Image 8 cross-rater contrast ────────────────────────────

def panel_e_image8(ax, r1_pass1: pd.DataFrame,
                   r1_current: pd.DataFrame, r2: pd.DataFrame,
                   r3: pd.DataFrame) -> None:
    """Image 8 (C26 CM, well B4) -- within-rater shift for reviewer 1
    plus reviewers 2 and 3 as cross-rater references."""
    img8_r1_pass1 = im_stats(r1_pass1[8])  # with anchor (pass 1)
    img8_r1_remeas = im_stats(r1_current[8])  # no anchor (re-measured)
    img8_r2 = im_stats(r2[8])
    img8_r3 = im_stats(r3_image(r3, 8))

    xs_lab = [
        "Rater 1\npass 1\n(WITH anchor)",
        "Rater 1\nre-measured\n(NO anchor)",
        "Rater 2\n(blinded\nsession)",
        "Rater 3\n(blinded\nsession)",
    ]
    points  = [img8_r1_pass1[:2], img8_r1_remeas[:2],
               img8_r2[:2],       img8_r3[:2]]
    colors  = [CTRL_COLOR, C26_COLOR, R2_COLOR, R3_COLOR]
    markers = ["o", "s", "^", "D"]

    for i, ((m, s), c, mk) in enumerate(zip(points, colors, markers)):
        ax.errorbar(i, m, yerr=s, fmt=mk, color=c, mec="k", mew=0.6,
                    ms=10, capsize=3, ecolor="gray", zorder=3)
        ax.text(i, m + s + 1.2, f"{m:.1f}", ha="center",
                fontsize=8, fontweight="bold")

    ax.set_xticks(range(4))
    ax.set_xticklabels(xs_lab, fontsize=6.8)
    ax.set_ylabel("Image 8 image-mean (µm)")
    ax.set_title("Image 8 -- within-rater 1 shift",
                 fontweight="bold", loc="center", pad=8)
    ax.set_ylim(0, 28)

    # Rater-1 within-rater shift bracket: pass-1 → re-measured.
    # Place the bracket high enough that the 20.7 value label below it
    # does not collide with the bracket line.
    shift = img8_r1_remeas[0] - img8_r1_pass1[0]
    y_brk = max(img8_r1_pass1[0], img8_r1_remeas[0]) + 3.5
    ax.plot([0, 0, 1, 1], [y_brk, y_brk + 0.35, y_brk + 0.35, y_brk],
            color="black", lw=0.8)
    ax.text(0.5, y_brk + 0.5,
            f"{shift:+.1f} µm",
            ha="center", fontsize=7, fontweight="bold")

    ax.grid(True, alpha=0.25, linestyle=":")


# ── Driver ────────────────────────────────────────────────────────────

def main() -> None:
    r1_current  = load_r1_current()
    r1_pass1 = load_r1_pass1()
    r2 = load_r2()
    r3 = load_r3()
    codes = load_codes()
    # Panel B uses the *original* pass-1 workbook so the timeline matches
    # the first-image-effect example (image 1 sits well above the C26
    # band before any Control reference is established). Panel A uses
    # the current workbook (the contested-images cross-rater contrast is
    # framed with the same data the rest of the figure refers to as
    # post-re-measure for images 8 and 9).
    ab_df = per_img(r1_pass1, codes)
    r2_df = per_img(r2, codes)

    # Layout: 7"×~11.5" (Cell Press double-column).
    # Row 1: A (full width).
    # Rows 2 & 3: B and C (full-width timelines).
    # Row 4: D (left) | E (right).
    fig = plt.figure(figsize=(figstyle.FIG_W, 11.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 0.85, 0.85, 1.15],
                          hspace=0.65, wspace=0.55,
                          left=0.10, right=0.985, top=0.96, bottom=0.045)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, :])
    axC = fig.add_subplot(gs[2, :])
    axD = fig.add_subplot(gs[3, 0])
    axE = fig.add_subplot(gs[3, 1])

    # Panel A -- image 1, 8, 9 cross-rater. For reviewer 1 we use the
    # *current* workbook (post-remeasure values for 8 and 9), because
    # those re-measures are what the contested cross-rater contrast in
    # the manuscript refers to.
    panel_a_cross_rater(axA, r1_current, r2, r3)

    # Panel B (AB)
    img1_ab = ab_df.iloc[0]
    timeline_panel(
        axB, ab_df,
        "Rater 1, pass 1 -- image-mean diameter in blinded order",
        highlight_image1_text=(
            f"Image 1 (C26, well A4)\n"
            f"{img1_ab['mean']:.1f} µm, first image of\n"
            f"session; no Control measured yet"),
    )

    # Panel C (NJ)
    img1_nj = r2_df.iloc[0]
    timeline_panel(
        axC, r2_df,
        "Rater 2, pass 1 -- same images, same blinded order",
        highlight_image1_text=(
            f"Image 1 (same image as panel B)\n"
            f"{img1_nj['mean']:.1f} µm, measured\n"
            f"in C26 range from the start"),
    )
    axC.set_xlabel("Measurement order in blinded session")

    # Panel D
    panel_d_anchor_strip(axD, r1_pass1, r1_current, codes)

    # Panel E
    panel_e_image8(axE, r1_pass1, r1_current, r2, r3)

    # Shared legend for panels B and C, anchored above panel B
    leg_handles = [
        plt.Line2D([0], [0], marker="o", color="w", mfc=CTRL_COLOR, mec="k",
                   ms=6, label="Control well"),
        plt.Line2D([0], [0], marker="s", color="w", mfc=C26_COLOR, mec="k",
                   ms=6, label="C26 CM well"),
        mpatches.Patch(color=C26_COLOR, alpha=0.10,
                       label="Range of remaining C26 image-means"),
    ]
    axB.legend(handles=leg_handles, loc="upper right",
               fontsize=6.5, framealpha=0.95, ncol=3,
               handletextpad=0.4, columnspacing=0.9,
               borderpad=0.3, labelspacing=0.2)

    # Panel labels: anchored above the title so they don't collide with
    # it. Wide rows (A, B, C) sit closer in axis units; the narrower
    # bottom-row panels (D, E) need more clearance to clear the y-label.
    for ax, letter, dx in zip(
        [axA, axB, axC, axD, axE], "ABCDE",
        [-0.06, -0.06, -0.06, -0.18, -0.18],
    ):
        ax.text(dx, 1.18, letter, transform=ax.transAxes, **PANEL_LABEL_KW)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF,             bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
