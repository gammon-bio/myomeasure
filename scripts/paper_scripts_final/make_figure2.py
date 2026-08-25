"""Figure 2 of the manuscript -- dexamethasone dose-response showcase.

Replaces the former image-level validation figure. Six panels:

  A-D. Representative AF647 (Cy5) fluorescence micrographs, one per
       dexamethasone dose condition (Vehicle, 0.1, 1, 10 µM). For each
       condition the image whose image-level mean diameter is closest to
       the condition mean (mean of image means, both dexa plates pooled)
       is shown, false-coloured red. Each panel is framed in its dose
       colour to tie it to panels E-F. 200 µm scale bar.
  E.   Per-well mean diameter -- bar = condition mean of well means, SD
       error bars, plate-coded well dots, Games-Howell-vs-vehicle brackets.
       Drawn natively here (was previously an embedded PNG); the annotated
       statistics are READ from the pipeline's own CSVs in
       results/IGF_dexa_combined (not re-derived) and asserted to reproduce.
  F.   Pooled per-myotube count histogram overlay, drawn natively here.

Inputs:
  - results/IGF_dexa/measurements.csv          (plate 004, filtered)
  - results/IGF_dexa_plate005/measurements.csv (plate 005)
  - data/real/IGF_dexa/<group>/tiff/<image>    (representative TIFFs)
  - results/IGF_dexa_combined/combined_anova_oneway.csv        (panel E global test)
  - results/IGF_dexa_combined/combined_anova_twoway.csv        (plate check only)
  - results/IGF_dexa_combined/combined_variance_homogeneity.csv
  - results/IGF_dexa_combined/combined_effect_sizes.csv

Outputs (Cell-Press quality):
  - reports/figures_final/figure2.png  (600 DPI)
  - reports/figures_final/figure2.pdf

Run in the cellpose conda env (run the dexa analysis first so the CSVs exist):
    python results/IGF_dexa_combined/make_figures.py
    python scripts/paper_scripts_final/make_figure2.py
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

ROOT = Path(__file__).resolve().parents[2]
CSV_004 = ROOT / "results" / "IGF_dexa" / "measurements.csv"
CSV_005 = ROOT / "results" / "IGF_dexa_plate005" / "measurements.csv"
TIFF_ROOT = ROOT / "data" / "real" / "IGF_dexa"
COMBINED = ROOT / "results" / "IGF_dexa_combined"
OUT_PNG = ROOT / "reports" / "figures_final" / "figure2.png"
OUT_PDF = ROOT / "reports" / "figures_final" / "figure2.pdf"

# Acquisition calibration (embedded in the VSI->TIFF conversion). Verified
# against data/real/IGF_dexa/vsi_acquisition_metadata.csv (pixel_size = 0.65 µm,
# 10x UPLXAPO, 2304x2304): 200 µm scale bar = 307.7 px = 13.4% of the field.
UM_PER_PX = 0.65
SCALEBAR_UM = 200.0

CONDITIONS = ["Con_Veh", "0.1uM", "1uM", "10uM"]
DOSES = ["0.1uM", "1uM", "10uM"]
LABELS = ["Con (Veh)", "Dex 0.1 µM", "Dex 1 µM", "Dex 10 µM"]
# Corrected Okabe-Ito palette, shared with results/IGF_dexa_combined and the
# rest of the manuscript: control = sky-blue, Dex doses = vermillion ramp.
COLORS = ["#56B4E9", "#E8A673", "#D55E00", "#803800"]
# Well marker by plate: open-legend circle = plate 004, triangle = plate 005.
PLATE_MARKERS = {"plate_004": "o", "plate_005": "^"}

# Histogram bins copied VERBATIM from the dexa plotting code
# (results/IGF_dexa_combined/make_figures.py, fig4 overlay) -- do not re-derive.
HIST_BINS = np.linspace(5, 45, 61)     # range 5-45 µm, 60 bins
HIST_RANGE = (5.0, 45.0)
# Expected per-arm myotube counts, for the reproduction assertion.
HIST_N_EXPECTED = [5012, 6134, 5983, 5330]

# Explicit representative-image choices (else auto-select the image whose
# image-mean is closest to the condition mean). Both from plate 005.
OVERRIDES = {
    "0.1uM": "0.1uM_Snapshot_20260526_276.tif",
    "10uM":  "10uM_Snapshot_20260526_309.tif",
}

# Robust per-image contrast stretch for the red channel.
STRETCH_LO_PCT = 1.0
STRETCH_HI_PCT = 99.5

# ── Shared Cell Press style (unified width + type scale, myomeasure.figstyle) ──
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from myomeasure import figstyle
plt.rcParams.update(figstyle.RC)

PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", ha="left", va="top")


# ── Data ──────────────────────────────────────────────────────────────

def load_combined() -> pd.DataFrame:
    """Pool the two dexa plates exactly as results/IGF_dexa_combined does
    (plate 004 filtered to its own wells, plate 005 whole), so the per-myotube
    set and per-well means reproduce that analysis."""
    d4 = pd.read_csv(CSV_004)
    d4 = d4[d4["group"].str.startswith("Well plate_004")].copy()
    d4["plate"] = "plate_004"
    d5 = pd.read_csv(CSV_005).copy()
    d5["plate"] = "plate_005"
    df = pd.concat([d4, d5], ignore_index=True)
    df = df[df["mean_diameter"] > 0].copy()
    df["condition"] = df["image"].str.extract(r"^(.+?)_Snapshot")[0]
    df["well"] = df["group"].str.extract(r"/([A-Z]\d+)$")[0]
    df["plate_well"] = df["plate"] + "_" + df["well"]
    return df


def load_dexa_stats() -> dict:
    """Read the pre-computed dexa statistics from the pipeline CSVs (the same
    files that feed Supplementary Tables S1-S3 and S11), so panel E's annotations
    are the analysis's own numbers rather than re-derived here.

    The condition effect annotated on panel E is the ONE-WAY ANOVA on the 24
    well means. The two-way condition x plate model is read as well, but only to
    carry the plate / interaction check through the reproduction assertions --
    it is not shown on the panel."""
    one = pd.read_csv(COMBINED / "combined_anova_oneway.csv")
    orow = one.loc[one["term"] == "condition"].iloc[0]
    oresid = one.loc[one["term"] == "Residual"].iloc[0]
    two = pd.read_csv(COMBINED / "combined_anova_twoway.csv")
    tw_resid_df = int(round(float(
        two.loc[two["term"] == "Residual"].iloc[0]["df"])))
    plate = {}
    for term in ("condition", "plate", "condition:plate"):
        r = two.loc[two["term"] == term].iloc[0]
        plate[term] = {"F": float(r["F"]), "p": float(r["p_value"]),
                       "df1": int(round(float(r["df"]))), "df2": tw_resid_df}
    lev = pd.read_csv(COMBINED / "combined_variance_homogeneity.csv").iloc[0]
    eff = pd.read_csv(COMBINED / "combined_effect_sizes.csv")
    return {
        "cond_F": float(orow["F"]),
        "cond_p": float(orow["p_value"]),
        "df1": int(round(float(orow["df"]))),
        "df2": int(round(float(oresid["df"]))),
        "plate_check": plate,
        "levene_p": float(lev["p_value"]),
        "posthoc": str(lev["posthoc_used"]),
        "dose_p": {str(r["condition"]): float(r["p_adj"]) for _, r in eff.iterrows()},
    }


def sig_stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# ── Red false-colour helper ───────────────────────────────────────────

def red_false_color(path: Path) -> np.ndarray:
    """Load a single-channel TIFF and map it to an RGB image where the
    intensity drives the red channel (AF647 / Cy5 convention)."""
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3:                      # collapse if multi-page/channel
        img = img.reshape(img.shape[-2], img.shape[-1]) if img.shape[0] == 1 \
            else img.max(axis=0)
    lo = np.percentile(img, STRETCH_LO_PCT)
    hi = np.percentile(img, STRETCH_HI_PCT)
    norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = np.zeros((*norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = norm                     # red channel only
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


# ── Representative-image selection + micrograph panels A-D ─────────────

def representative_images(df: pd.DataFrame) -> dict[str, dict]:
    """For each condition, the image whose image-level mean diameter is
    closest to the condition's mean of image means."""
    img = (df.groupby(["condition", "group", "image"])["mean_diameter"]
             .agg(img_mean="mean", n_myo="size").reset_index())
    reps = {}
    for cond in CONDITIONS:
        sub = img[img["condition"] == cond]
        cond_ref = float(sub["img_mean"].mean())
        if cond in OVERRIDES:
            best = sub[sub["image"] == OVERRIDES[cond]]
            if best.empty:
                raise ValueError(
                    f"override image {OVERRIDES[cond]!r} not found for {cond}")
            best = best.iloc[0]
        else:
            sub = sub.assign(dist=(sub["img_mean"] - cond_ref).abs())
            best = sub.sort_values("dist").iloc[0]
        reps[cond] = {
            "group": best["group"],
            "image": best["image"],
            "img_mean": float(best["img_mean"]),
            "cond_ref": cond_ref,
            "n_myo": int(best["n_myo"]),
            "path": TIFF_ROOT / best["group"] / "tiff" / best["image"],
        }
    return reps


def draw_micrograph(ax, rep: dict, label: str, color: str) -> None:
    rgb = red_false_color(rep["path"])
    ax.imshow(rgb, origin="upper", interpolation="nearest")
    add_scalebar(ax, rgb.shape[1])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(2.2)
    ax.set_title(label, color=color, fontweight="bold", pad=3)


# ── Panel E: well-means bar (native redraw of fig1_well_means_bar) ─────

def draw_panel_e_bar(ax, df: pd.DataFrame, st: dict) -> np.ndarray:
    """Bar = condition mean of well means (n = 6 wells/arm), SD error bars,
    plate-coded well dots, and Games-Howell-vs-vehicle brackets whose p-values
    are read from the pipeline CSVs. Returns the per-arm well counts."""
    wm = (df[df["condition"].isin(CONDITIONS)]
          .groupby(["condition", "plate", "well"], observed=True)["mean_diameter"]
          .mean().reset_index(name="well_mean_um"))
    by = {c: wm.loc[wm["condition"] == c, "well_mean_um"].values for c in CONDITIONS}
    n_wells = np.array([by[c].size for c in CONDITIONS])

    x = np.arange(len(CONDITIONS))
    means = np.array([by[c].mean() for c in CONDITIONS])
    sds = np.array([by[c].std() for c in CONDITIONS])        # ddof=0, matches fig1

    ax.bar(x, means, color=COLORS, alpha=0.25, edgecolor=COLORS,
           linewidth=1.2, width=0.6)
    for i in range(len(CONDITIONS)):
        ax.hlines(means[i], x[i] - 0.20, x[i] + 0.20, color="black",
                  linewidth=1.4, zorder=6)
        ax.errorbar(x[i], means[i], yerr=sds[i], fmt="none", color="black",
                    capsize=4, capthick=0.9, linewidth=0.9, zorder=4)

    rng = np.random.default_rng(42)
    for i, cond in enumerate(CONDITIONS):
        w = wm[wm["condition"] == cond]
        for plate, marker in PLATE_MARKERS.items():
            pw = w[w["plate"] == plate]
            if pw.empty:
                continue
            jitter = rng.uniform(-0.10, 0.10, size=len(pw))
            ax.scatter(x[i] + jitter, pw["well_mean_um"], marker=marker,
                       color=COLORS[i], edgecolor="black", s=45, zorder=5,
                       linewidth=0.6)

    # Dose-vs-vehicle brackets, adjusted p from the CSV (Games-Howell).
    y_top = float(means.max() + sds.max() + 1.2)
    bh, pad = 0.30, 0.55
    for k, cond in enumerate(DOSES):
        i = CONDITIONS.index(cond)
        y = y_top + k * (bh + pad + 0.4)
        ax.plot([0, 0, i, i], [y, y + bh, y + bh, y], color="black", lw=0.9)
        p = st["dose_p"][cond]
        ax.text(i / 2.0, y + bh + 0.10, f"{sig_stars(p)}  (p={p:.2g})",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.text(0.03, 0.40,
            f"One-way ANOVA (well-level, n=6/cond)\n"
            f"condition F({st['df1']},{st['df2']}) = {st['cond_F']:.2f}, "
            f"p = {st['cond_p']:.3g}\n"
            f"Levene p = {st['levene_p']:.3g}; post hoc: {st['posthoc']} vs vehicle\n"
            f"Markers: ○ plate 004   ▲ plate 005",
            transform=ax.transAxes, fontsize=6.5, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    top_y = (y_top + (len(DOSES) - 1) * (bh + pad + 0.4) + bh + pad + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_ylabel("Myotube diameter (µm)")
    ax.set_title("Dex dose response", fontweight="bold")
    ax.set_ylim(0, top_y + 0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return n_wells


# ── Panel F: count-histogram overlay (native redraw of fig4 overlay) ───

def draw_panel_f_hist(ax, df: pd.DataFrame) -> tuple[list[int], list[float]]:
    """Overlaid step count-histograms of pooled per-myotube diameters, one per
    arm, with a dashed vertical line at each arm's mean. Returns (n_per_arm,
    mean_per_arm) for the reproduction assertion."""
    sub = df[df["condition"].isin(CONDITIONS)]
    n_arm, mean_arm = [], []
    for cond, label, color in zip(CONDITIONS, LABELS, COLORS):
        vals = sub.loc[sub["condition"] == cond, "mean_diameter"].values
        n_arm.append(int(vals.size))
        mean_arm.append(float(np.mean(vals)))
        ax.hist(vals, bins=HIST_BINS, density=False, histtype="step",
                color=color, linewidth=1.5, label=f"{label}  (n={vals.size})")
        ax.axvline(float(np.mean(vals)), color=color, linestyle="--",
                   linewidth=1.0, alpha=0.7)
    ax.set_xlim(*HIST_RANGE)
    ax.set_xlabel("Myotube diameter (µm)")
    ax.set_ylabel("Count (myotubes)")
    ax.set_title("Dex Dose Measurement Distribution", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, handlelength=1.3,
              handletextpad=0.4, borderpad=0.3, labelspacing=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return n_arm, mean_arm


# ── Reproduction assertions (fail loudly) ─────────────────────────────

def assert_reproduction(st: dict, n_wells: np.ndarray,
                        hist_n: list[int], hist_means: list[float]) -> None:
    """Assert the redrawn panels reproduce the manuscript's headline numbers.
    Prints recomputed-vs-expected for every check and raises on any mismatch."""
    failures = []

    def check(name, got, exp):
        ok = got == exp
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:34s} "
              f"recomputed={got!r:<26} expected={exp!r}")
        if not ok:
            failures.append(name)

    # Primary global test: one-way ANOVA on the 24 well means (annotated on E).
    check("one-way ANOVA condition F", round(st["cond_F"], 2), 25.67)
    check("one-way ANOVA condition df", (st["df1"], st["df2"]), (3, 20))
    check("one-way ANOVA condition p", f"{st['cond_p']:.3g}", "4.62e-07")
    # Plate / interaction check: the two-way model that justifies pooling the
    # two plates. Not drawn on the panel, but must not drift.
    pc = st["plate_check"]
    check("plate check: condition F", round(pc["condition"]["F"], 2), 23.08)
    check("plate check: condition df",
          (pc["condition"]["df1"], pc["condition"]["df2"]), (3, 16))
    check("plate check: condition p", f"{pc['condition']['p']:.3g}", "4.7e-06")
    check("plate check: plate F", round(pc["plate"]["F"], 2), 0.34)
    check("plate check: plate df", (pc["plate"]["df1"], pc["plate"]["df2"]), (1, 16))
    check("plate check: plate p", f"{pc['plate']['p']:.2g}", "0.57")
    check("plate check: interaction F", round(pc["condition:plate"]["F"], 2), 0.55)
    check("plate check: interaction df",
          (pc["condition:plate"]["df1"], pc["condition:plate"]["df2"]), (3, 16))
    check("plate check: interaction p", f"{pc['condition:plate']['p']:.2g}", "0.66")
    check("Levene p", f"{st['levene_p']:.3g}", "0.0314")
    check("post hoc test", st["posthoc"], "Games-Howell")
    check("Games-Howell p 0.1uM vs Veh", f"{st['dose_p']['0.1uM']:.2g}", "1e-05")
    check("Games-Howell p 1uM vs Veh", f"{st['dose_p']['1uM']:.2g}", "1.6e-05")
    check("Games-Howell p 10uM vs Veh", f"{st['dose_p']['10uM']:.2g}", "0.0042")
    check("wells per arm", list(map(int, n_wells)), [6, 6, 6, 6])
    check("histogram n per arm", hist_n, HIST_N_EXPECTED)
    check("histogram bin range", (float(HIST_BINS[0]), float(HIST_BINS[-1])),
          (5.0, 45.0))
    check("histogram bin count", len(HIST_BINS) - 1, 60)
    # Mean lines are derived (no fixed expectation); report for the record.
    print(f"  [rep] dashed mean lines (µm): "
          f"{[round(m, 3) for m in hist_means]}")

    if failures:
        raise AssertionError(
            "Figure 2 reproduction FAILED for: " + ", ".join(failures))
    print("  All Figure 2 reproduction checks PASSED.")


# ── Driver ────────────────────────────────────────────────────────────

def main() -> None:
    df = load_combined()
    st = load_dexa_stats()
    reps = representative_images(df)
    print("Representative images (closest image-mean to condition mean):")
    for cond, r in reps.items():
        print(f"  {cond:8s} img_mean={r['img_mean']:.3f} "
              f"(cond_ref={r['cond_ref']:.3f}, n_myo={r['n_myo']})  "
              f"{r['group']}/{r['image']}")

    fig = plt.figure(figsize=(figstyle.FIG_W, 5.6))
    # Top strip: four square micrographs. Bottom row: the two native
    # quantitative panels. Per the manuscript, F (histogram) stays wider than E
    # (bar chart) -- width_ratios [1, 1.4], NOT equal footprints.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.72],
                             hspace=0.12, left=0.09, right=0.985,
                             top=0.955, bottom=0.11)
    top = outer[0].subgridspec(1, 4, wspace=0.07)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1, 1.4], wspace=0.30)

    micro_axes = [fig.add_subplot(top[0, i]) for i in range(4)]
    for ax, cond, label, color in zip(micro_axes, CONDITIONS, LABELS, COLORS):
        draw_micrograph(ax, reps[cond], label, color)

    axE = fig.add_subplot(bot[0, 0])
    axF = fig.add_subplot(bot[0, 1])
    n_wells = draw_panel_e_bar(axE, df, st)
    hist_n, hist_means = draw_panel_f_hist(axF, df)

    print("Figure 2 reproduction check (recomputed vs manuscript):")
    assert_reproduction(st, n_wells, hist_n, hist_means)

    # Panel labels A-F.
    for ax, letter in zip(micro_axes, "ABCD"):
        ax.text(-0.02, 1.16, letter, transform=ax.transAxes, **PANEL_LABEL_KW)
    axE.text(-0.20, 1.06, "E", transform=axE.transAxes, **PANEL_LABEL_KW)
    axF.text(-0.13, 1.06, "F", transform=axF.transAxes, **PANEL_LABEL_KW)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
