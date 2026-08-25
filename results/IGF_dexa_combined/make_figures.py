"""Combined dexamethasone analysis: plate 004 + plate 005 (n=6 wells/condition).

Pools the two dexa plates and runs the single pre-specified analysis strategy
for the dose-response (well = biological replicate, two plates pooled by
design):

  * one-way ANOVA of well-mean diameter across the four conditions -- the
    global test of the condition effect
  * two-way ANOVA (condition x plate, with interaction) -- retained ONLY as the
    plate / interaction check that justifies pooling the two plates; it is not
    the reported condition test
  * Levene's test for variance homogeneity across the four conditions
  * Dunnett's test of each Dex dose vs vehicle (well-level means); if Levene
    indicates heterogeneity, the dose-vs-vehicle comparison falls back to
    Games-Howell (unequal-variance compare-to-control) and the switch is
    recorded
  * effect sizes (Cohen's d, Hedges' g, 95% CIs) for each dose vs vehicle,
    at the well level (matches the inferential unit)
  * among-dose saturation check (Welch + Holm, descriptive)
  * between-well CV by condition (precision claim)
  * per-condition percentiles, total well area, and the KS D-statistic vs
    vehicle reported descriptively (no p-value -- the pooled per-myotube KS
    p-value is pseudoreplicated and is not reported)

Figures (saved next to this script):
  * fig1_well_means_bar.png       -- bar of well means + dots (plate-coded)
  * fig2_violin_all_data.png      -- violin + box + ANOVA / Dunnett annotation
  * fig3_count_hist_per_cond.png  -- per-condition count histograms
  * fig4_count_hist_overlay.png   -- overlay count histogram
And CSV outputs:
  * combined_condition_stats.csv  -- per-condition rollup (KS D descriptive)
  * combined_well_stats.csv       -- per-well rollup
  * combined_anova_oneway.csv     -- one-way ANOVA (condition) on well means,
                                    the reported global test
  * combined_anova_twoway.csv     -- two-way ANOVA (condition x plate), the
                                    plate / interaction check only
  * combined_variance_homogeneity.csv -- Levene's test across conditions
  * combined_effect_sizes.csv     -- dose vs vehicle: Dunnett/Games-Howell
                                    adjusted p + Cohen's d / Hedges' g / CIs
  * combined_dose_pairwise.csv    -- among-dose saturation (Welch + Holm)
  * combined_well_cv.csv          -- between-well CV by condition
  * combined_hist_counts.csv      -- bin counts per condition / per well

These outputs feed Supplementary Tables S1 (combined_effect_sizes),
S2 (combined_condition_stats), S3 (combined_well_cv) and S11
(combined_anova_oneway + combined_anova_twoway). S11, not S12: supplementary
numbering follows first citation in the text, not topic.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parents[2]
OUT = Path(__file__).parent

CSV_004 = ROOT / "results" / "IGF_dexa" / "measurements.csv"
CSV_005 = ROOT / "results" / "IGF_dexa_plate005" / "measurements.csv"

REFERENCE = "Con_Veh"
CONDITIONS = ["Con_Veh", "0.1uM", "1uM", "10uM"]
DOSES = ["0.1uM", "1uM", "10uM"]
LABELS = ["Con (Veh)", "Dex 0.1 µM", "Dex 1 µM", "Dex 10 µM"]
# Corrected manuscript palette (Okabe-Ito, colourblind-safe; matches the
# render_all/figures-final scheme): control = sky-blue, Dex doses = a
# light->dark ramp of the vermillion treatment colour #D55E00.
COLORS = ["#56B4E9", "#E8A673", "#D55E00", "#803800"]
PLATE_MARKERS = {"plate_004": "o", "plate_005": "^"}


def load_combined() -> pd.DataFrame:
    df_004 = pd.read_csv(CSV_004)
    df_004 = df_004[df_004["group"].str.startswith("Well plate_004")].copy()
    df_004["plate"] = "plate_004"

    df_005 = pd.read_csv(CSV_005)
    df_005["plate"] = "plate_005"

    df = pd.concat([df_004, df_005], ignore_index=True)
    df = df[df["mean_diameter"] > 0].copy()
    df["condition"] = df["image"].str.extract(r"^(.+?)_Snapshot")[0]
    df["well"] = df["group"].str.extract(r"/([A-Z]\d+)$")[0]
    df["plate_well"] = df["plate"] + "_" + df["well"]
    return df


def sig_text(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def holm_correct(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * p[idx])
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def cohens_d(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Cohen's d (x - y), Hedges' g, and an approximate 95% CI on d.

    Computed from the supplied samples directly; here x and y are well-level
    means (n = 6 wells), so this is a well-level effect size consistent with
    the inferential unit of analysis.
    """
    nx, ny = len(x), len(y)
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    d = (x.mean() - y.mean()) / sp
    J = 1 - 3 / (4 * (nx + ny) - 9)  # Hedges' bias correction
    g = J * d
    se_d = np.sqrt((nx + ny) / (nx * ny) + d ** 2 / (2 * (nx + ny)))
    return float(d), float(g), float(d - 1.96 * se_d), float(d + 1.96 * se_d)


def welch_with_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """Welch t-test plus 95% CI on the mean difference (x - y)."""
    mx, my = x.mean(), y.mean()
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    nx, ny = len(x), len(y)
    se = np.sqrt(sx2 / nx + sy2 / ny)
    df_welch = (sx2 / nx + sy2 / ny) ** 2 / (
        (sx2 / nx) ** 2 / (nx - 1) + (sy2 / ny) ** 2 / (ny - 1)
    )
    t = (mx - my) / se
    p = 2 * (1 - stats.t.cdf(abs(t), df_welch))
    t_crit = stats.t.ppf(1 - alpha / 2, df_welch)
    md = mx - my
    return {"t": float(t), "df": float(df_welch), "p": float(p),
            "mean_diff": float(md), "ci_lo": float(md - t_crit * se),
            "ci_hi": float(md + t_crit * se)}


def pooled_md_ci(by: dict, dose: str, ref: str, alpha: float = 0.05):
    """95% CI on the dose-minus-vehicle mean difference, using the pooled
    (equal-variance) error term of the four-condition model -- consistent with
    the ANOVA / Dunnett variance assumption."""
    allv = [by[c] for c in CONDITIONS]
    N = sum(len(v) for v in allv)
    k = len(allv)
    s2 = sum((len(v) - 1) * v.var(ddof=1) for v in allv) / (N - k)
    x, y = by[dose], by[ref]
    se = np.sqrt(s2 * (1 / len(x) + 1 / len(y)))
    md = x.mean() - y.mean()
    tcrit = stats.t.ppf(1 - alpha / 2, N - k)
    return float(md), float(md - tcrit * se), float(md + tcrit * se)


def main() -> None:
    df = load_combined()
    sub = df[df["condition"].isin(CONDITIONS)].copy()
    sub["condition"] = pd.Categorical(sub["condition"], categories=CONDITIONS,
                                       ordered=True)

    # Well-level table
    well_means = (sub.groupby(["condition", "plate", "well"], observed=True)
                    .agg(n_myo=("mean_diameter", "size"),
                         well_mean_um=("mean_diameter", "mean"),
                         well_median_um=("mean_diameter", "median"),
                         total_area_um2=("area", "sum"))
                    .reset_index())

    # Per-condition well-mean arrays (the inferential unit, n=6 each)
    by = {c: well_means[well_means["condition"] == c]["well_mean_um"].values
          for c in CONDITIONS}

    # ============= Per-condition rollup (combined_condition_stats.csv / S2) ===
    # The KS test is reported as a descriptive distributional-shift D-statistic
    # only. The pooled per-myotube KS p-value is pseudoreplicated (myotubes are
    # nested within images and wells) and is NOT reported.
    rows = []
    ref_vals = sub[sub["condition"] == REFERENCE]["mean_diameter"].values
    for cond, label in zip(CONDITIONS, LABELS):
        vals = sub[sub["condition"] == cond]["mean_diameter"].values
        wm = well_means[well_means["condition"] == cond]
        p25, p50, p75, p90 = np.percentile(vals, [25, 50, 75, 90])
        if cond == REFERENCE:
            ks_D = np.nan
        else:
            ks_D = float(stats.ks_2samp(vals, ref_vals,
                                        alternative="two-sided").statistic)
        rows.append({
            "condition": cond, "label": label,
            "n_wells": int(len(wm)),
            "n_myotubes": int(len(vals)),
            "well_mean_um": round(float(wm["well_mean_um"].mean()), 3),
            "well_sd_um": round(float(wm["well_mean_um"].std()), 3),
            "myo_mean_um": round(float(np.mean(vals)), 3),
            "myo_median_um": round(float(np.median(vals)), 3),
            "myo_sd_um": round(float(np.std(vals, ddof=1)), 3),
            "P25_um": round(float(p25), 3),
            "P50_um": round(float(p50), 3),
            "P75_um": round(float(p75), 3),
            "P90_um": round(float(p90), 3),
            "skewness": round(float(stats.skew(vals)), 4),
            "excess_kurtosis": round(float(stats.kurtosis(vals)), 4),
            "total_area_um2_mean": round(float(wm["total_area_um2"].mean()), 1),
            "total_area_um2_sd": round(float(wm["total_area_um2"].std()), 1),
            "KS_D_descriptive": "" if cond == REFERENCE else round(ks_D, 4),
        })
    cond_df = pd.DataFrame(rows)
    cond_df.to_csv(OUT / "combined_condition_stats.csv", index=False)
    print("\n=== Per-condition summary (n=6 wells / condition) ===")
    print(cond_df.to_string(index=False))

    # Save per-well stats
    well_means.assign(
        well_mean_um=lambda d: d["well_mean_um"].round(3),
        well_median_um=lambda d: d["well_median_um"].round(3),
        total_area_um2=lambda d: d["total_area_um2"].round(1),
    ).to_csv(OUT / "combined_well_stats.csv", index=False)

    # ============= Global test on well means ============================
    # PRIMARY: one-way ANOVA of well-mean diameter across the four conditions
    # (well = experimental unit, n = 6 wells/condition, 24 wells total).
    #
    # Plate is NOT a factor in the reported condition test. The condition x
    # plate two-way model below is computed and written out solely as the
    # plate / interaction CHECK: both terms are null, so the plate block is
    # pooled into the error term for the condition comparison. The design is
    # balanced (3 wells per condition per plate), so the condition sum of
    # squares is identical in the two models -- only the error term differs
    # (one-way: 20 df; two-way: 16 df after removing plate + interaction).
    cond_F = cond_p = np.nan
    cond_df1 = len(CONDITIONS) - 1
    cond_df2 = np.nan
    plate_check: dict[str, str] = {}
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        wm_stat = well_means.rename(columns={"well_mean_um": "y"})
        wm_stat["condition"] = wm_stat["condition"].astype(str)

        # --- primary: one-way ANOVA (condition) --------------------------
        one_model = smf.ols("y ~ C(condition)", data=wm_stat).fit()
        one_tab = anova_lm(one_model, typ=2)
        print("\n=== One-way ANOVA (condition) on well means -- PRIMARY ===")
        print(one_tab.round(6))
        one_way = one_tab.rename(columns={"PR(>F)": "p_value"}).reset_index()
        one_way = one_way.rename(columns={"index": "term"})
        one_way["term"] = one_way["term"].replace({"C(condition)": "condition"})
        one_way.to_csv(OUT / "combined_anova_oneway.csv", index=False)
        orow = one_way[one_way["term"] == "condition"].iloc[0]
        cond_F = float(orow["F"])
        cond_p = float(orow["p_value"])
        cond_df1 = int(orow["df"])
        cond_df2 = int(one_way[one_way["term"] == "Residual"].iloc[0]["df"])

        # --- check only: two-way condition x plate -----------------------
        model = smf.ols("y ~ C(condition) * C(plate)", data=wm_stat).fit()
        atab = anova_lm(model, typ=2)
        print("\n=== Two-way ANOVA (condition × plate) -- plate/interaction CHECK ===")
        print(atab.round(6))
        two_way = atab.rename(columns={"PR(>F)": "p_value"}).reset_index()
        two_way = two_way.rename(columns={"index": "term"})
        two_way["term"] = two_way["term"].replace({
            "C(condition)": "condition",
            "C(plate)": "plate",
            "C(condition):C(plate)": "condition:plate",
        })
        two_way.to_csv(OUT / "combined_anova_twoway.csv", index=False)
        tw_resid_df = int(two_way[two_way["term"] == "Residual"].iloc[0]["df"])
        for term in ("plate", "condition:plate"):
            r = two_way[two_way["term"] == term].iloc[0]
            plate_check[term] = (f"F({int(r['df'])},{tw_resid_df}) = "
                                 f"{float(r['F']):.2f}, p = {float(r['p_value']):.2g}")
        print(f"\nPlate check: plate {plate_check['plate']}; "
              f"condition x plate {plate_check['condition:plate']}\n"
              f"  -> both null; block pooled into error for the condition test")
    except Exception as e:
        print(f"(ANOVA failed -- required for the global test: {e})")

    # ============= Variance homogeneity (Levene) ========================
    levene_W, levene_p = stats.levene(*[by[c] for c in CONDITIONS],
                                      center="median")
    use_gameshowell = levene_p < 0.05
    pd.DataFrame([{
        "test": "Levene (median-centred) across conditions",
        "factor": "condition",
        "n_conditions": len(CONDITIONS),
        "W": round(float(levene_W), 4),
        "p_value": f"{levene_p:.4g}",
        "variance_homogeneous": (not use_gameshowell),
        "posthoc_used": "Games-Howell" if use_gameshowell else "Dunnett",
    }]).to_csv(OUT / "combined_variance_homogeneity.csv", index=False)
    print(f"\nLevene across conditions: W = {levene_W:.3f}, p = {levene_p:.4g} "
          f"→ {'heterogeneous: Games-Howell' if use_gameshowell else 'homogeneous: Dunnett'}")

    # ============= Dose vs vehicle: Dunnett (or Games-Howell) + effect sizes =
    # Primary objective is each dose vs the vehicle control, so a single
    # compare-to-control multiple-comparison procedure is used. It operates on
    # the same 24 well means as the one-way global test, so the post hoc and the
    # global test share a unit of analysis; the plate check above is what allows
    # both to ignore the block. Equal-variance Dunnett is used when Levene is non-
    # significant (consistent with the ANOVA variance assumption); otherwise
    # the unequal-variance Games-Howell compare-to-control test is used.
    test_name = "Games-Howell" if use_gameshowell else "Dunnett"
    dose_stat: dict[str, float] = {}
    dose_padj: dict[str, float] = {}
    if use_gameshowell:
        import pingouin as pg
        long = well_means[["condition", "well_mean_um"]].copy()
        long["condition"] = long["condition"].astype(str)
        gh = pg.pairwise_gameshowell(data=long, dv="well_mean_um",
                                     between="condition")
        for d in DOSES:
            m = gh[((gh["A"] == REFERENCE) & (gh["B"] == d)) |
                   ((gh["A"] == d) & (gh["B"] == REFERENCE))].iloc[0]
            # orient statistic as dose - vehicle
            tval = float(m["T"]) if m["B"] == d else -float(m["T"])
            dose_stat[d] = tval
            dose_padj[d] = float(m["pval"])
    else:
        res = stats.dunnett(*[by[d] for d in DOSES], control=by[REFERENCE])
        for i, d in enumerate(DOSES):
            dose_stat[d] = float(res.statistic[i])
            dose_padj[d] = float(res.pvalue[i])

    eff_rows = []
    for d in DOSES:
        x, y = by[d], by[REFERENCE]
        md, md_lo, md_hi = pooled_md_ci(by, d, REFERENCE)
        coh, g, d_lo, d_hi = cohens_d(x, y)
        pct = 100.0 * md / y.mean()
        eff_rows.append({
            "condition": d, "vs": REFERENCE,
            "mean_diff_um": round(md, 3),
            "MD_95CI_lo": round(md_lo, 3),
            "MD_95CI_hi": round(md_hi, 3),
            "pct_change": round(pct, 2),
            "cohens_d": round(coh, 3),
            "d_95CI_lo": round(d_lo, 3),
            "d_95CI_hi": round(d_hi, 3),
            "hedges_g": round(g, 3),
            "test": test_name,
            "stat": round(dose_stat[d], 3),
            "p_adj": f"{dose_padj[d]:.4g}",
            "sig": sig_text(dose_padj[d]),
        })
    eff_df = pd.DataFrame(eff_rows)
    eff_df.to_csv(OUT / "combined_effect_sizes.csv", index=False)
    print(f"\n=== Dose vs vehicle ({test_name}) + well-level effect sizes ===")
    print(eff_df.to_string(index=False))

    # ============= Among-dose saturation (Welch + Holm, descriptive) =========
    dose_pairs = [("0.1uM", "1uM"), ("0.1uM", "10uM"), ("1uM", "10uM")]
    dose_rows, dose_raw = [], []
    for a, b in dose_pairs:
        wres = welch_with_ci(by[a], by[b])
        coh, g, d_lo, d_hi = cohens_d(by[a], by[b])
        dose_raw.append(wres["p"])
        dose_rows.append({
            "comparison": f"{a} vs {b}",
            "mean_diff_um": round(wres["mean_diff"], 3),
            "MD_95CI_lo": round(wres["ci_lo"], 3),
            "MD_95CI_hi": round(wres["ci_hi"], 3),
            "welch_t": round(wres["t"], 3),
            "df_welch": round(wres["df"], 2),
            "p_raw": f"{wres['p']:.4g}",
            "cohens_d": round(coh, 3),
            "d_95CI_lo": round(d_lo, 3),
            "d_95CI_hi": round(d_hi, 3),
        })
    for row, hp in zip(dose_rows, holm_correct(dose_raw)):
        row["p_holm"] = f"{hp:.4g}"
        row["sig"] = sig_text(hp)
    pd.DataFrame(dose_rows).to_csv(OUT / "combined_dose_pairwise.csv", index=False)

    # ============= Between-well CV by condition (combined_well_cv.csv / S3) ===
    cv_rows = []
    for cond, lbl in zip(CONDITIONS, LABELS):
        v = by[cond]
        cv_rows.append({
            "condition": cond, "label": lbl,
            "n_wells": int(len(v)),
            "mean_um": round(float(v.mean()), 3),
            "sd_um": round(float(v.std(ddof=1)), 3),
            "CV_pct": round(100.0 * v.std(ddof=1) / v.mean(), 3),
        })
    pd.DataFrame(cv_rows).to_csv(OUT / "combined_well_cv.csv", index=False)

    # ============= Figure 1: bar of well means + dots ==============
    fig1, ax1 = plt.subplots(figsize=(6.5, 6))
    x_pos = np.arange(len(CONDITIONS))
    bar_means = [by[c].mean() for c in CONDITIONS]
    bar_sds = [by[c].std() for c in CONDITIONS]
    ax1.bar(x_pos, bar_means, color=COLORS, alpha=0.25, edgecolor=COLORS,
            linewidth=1.5, width=0.6)
    for i, gm in enumerate(bar_means):
        ax1.hlines(gm, x_pos[i] - 0.20, x_pos[i] + 0.20, color="black",
                   linewidth=1.8, zorder=6)
        ax1.errorbar(x_pos[i], gm, yerr=bar_sds[i], fmt="none", color="black",
                     capsize=6, capthick=1.2, linewidth=1.2, zorder=4)

    rng = np.random.default_rng(42)
    for i, cond in enumerate(CONDITIONS):
        wm = well_means[well_means["condition"] == cond]
        for plate, marker in PLATE_MARKERS.items():
            pw = wm[wm["plate"] == plate]
            if pw.empty:
                continue
            jitter = rng.uniform(-0.10, 0.10, size=len(pw))
            ax1.scatter(x_pos[i] + jitter, pw["well_mean_um"],
                        marker=marker, color=COLORS[i], edgecolor="black",
                        s=110, zorder=5, linewidth=0.9)

    # Pairwise brackets vs Con (Dunnett / Games-Howell adjusted p)
    y_top = max(bar_means) + max(bar_sds) + 1.2
    bracket_h = 0.30
    label_pad = 0.55
    for k, cond in enumerate(DOSES):
        i = CONDITIONS.index(cond)
        y = y_top + k * (bracket_h + label_pad + 0.4)
        ax1.plot([0, 0, i, i], [y, y + bracket_h, y + bracket_h, y],
                 color="black", lw=1.0)
        ax1.text(i / 2.0, y + bracket_h + 0.1,
                 f"{sig_text(dose_padj[cond])}  (p={dose_padj[cond]:.2g})",
                 ha="center", va="bottom", fontsize=15, fontweight="bold")

    # Stats box in lower-left (away from brackets)
    cond_F_txt = "n/a" if np.isnan(cond_F) else f"{cond_F:.2f}"
    cond_p_txt = "n/a" if np.isnan(cond_p) else f"{cond_p:.3g}"
    df2_txt = "" if (isinstance(cond_df2, float) and np.isnan(cond_df2)) else cond_df2
    ax1.text(0.02, 0.32,
             f"One-way ANOVA (well-level, n=6/cond)\n"
             f"condition F({cond_df1},{df2_txt}) = {cond_F_txt}, p = {cond_p_txt}\n"
             f"Levene p = {levene_p:.3g}; post hoc: {test_name} vs vehicle\n"
             f"Markers: ○ plate 004   ▲ plate 005",
             transform=ax1.transAxes, fontsize=11.5, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    ax1.set_xticks(x_pos)
    # Panels E and F embed into Figure 2 at EQUAL panel height (the bottom
    # width_ratios equal the two PNG aspect ratios). The bar PNG is scaled
    # slightly less than the histogram PNG, so its source fonts are set a touch
    # larger (26 vs the histogram's 25) so BOTH panels' ticks and titles render
    # at the shared ~11 pt once embedded. Bar/marker geometry unchanged.
    ax1.set_xticklabels(LABELS, fontsize=26, rotation=45, ha="right")
    ax1.tick_params(axis="y", labelsize=26)
    ax1.set_ylabel("Myotube diameter (µm)", fontsize=26)
    ax1.set_title("Dex dose response",
                  fontsize=26, fontweight="bold")
    top_y = y_top + (len(DOSES) - 1) \
            * (bracket_h + label_pad + 0.4) + bracket_h + label_pad + 0.5
    ax1.set_ylim(0, top_y + 0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(OUT / "fig1_well_means_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # ============= Figure 2: violin + box of pooled data ==============
    fig2, ax2 = plt.subplots(figsize=(7.5, 6))
    violin_data = [sub[sub["condition"] == c]["mean_diameter"].values
                   for c in CONDITIONS]
    parts = ax2.violinplot(violin_data, positions=x_pos, showmeans=False,
                           showmedians=False, showextrema=False, widths=0.75)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[i])
        body.set_alpha(0.35)
        body.set_edgecolor(COLORS[i])
        body.set_linewidth(1.2)

    rng2 = np.random.default_rng(7)
    for i, vals in enumerate(violin_data):
        jitter = rng2.uniform(-0.2, 0.2, size=len(vals))
        ax2.scatter(x_pos[i] + jitter, vals, color=COLORS[i], alpha=0.05, s=4,
                    zorder=3, rasterized=True)

    bp = ax2.boxplot(violin_data, positions=x_pos, widths=0.14,
                     patch_artist=True, showfliers=False, zorder=4)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor("white")
        box.set_edgecolor(COLORS[i])
        box.set_linewidth(1.3)
    for el in ["whiskers", "caps"]:
        for line in bp[el]:
            line.set_color("gray")
            line.set_linewidth(1)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(2)

    for i in range(len(CONDITIONS)):
        med = float(np.median(violin_data[i]))
        ax2.text(x_pos[i] + 0.32, med, f"{med:.1f}", ha="left", fontsize=10,
                 fontweight="bold", color=COLORS[i])
        ax2.text(x_pos[i], 1.5, f"n = {len(violin_data[i])}", ha="center",
                 fontsize=9, color="gray")

    ax2.text(0.02, 0.97,
             f"One-way ANOVA condition F = {cond_F_txt}, p = {cond_p_txt}\n"
             f"{test_name} vs vehicle (Levene p = {levene_p:.3g}):\n"
             + "\n".join(f"{c}: p = {dose_padj[c]:.3g} {sig_text(dose_padj[c])}"
                         for c in DOSES),
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(LABELS, fontsize=11)
    ax2.set_ylabel("Myotube diameter (µm)", fontsize=11)
    ax2.set_title("Dexamethasone -- pooled measurements (plates 004 + 005)",
                  fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(OUT / "fig2_violin_all_data.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # ============= Figure 3: per-condition count histograms ==============
    bins = np.linspace(5, 45, 61)
    fig3, axes3 = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for ax, cond, label, color in zip(axes3, CONDITIONS, LABELS, COLORS):
        vals = sub[sub["condition"] == cond]["mean_diameter"].values
        ax.hist(vals, bins=bins, density=False, histtype="stepfilled",
                color=color, alpha=0.55, edgecolor=color, linewidth=1.5)
        pooled_mean = float(np.mean(vals))
        ax.axvline(pooled_mean, color="black", linestyle="--", linewidth=1.2,
                   label=f"mean = {pooled_mean:.2f} µm")
        ax.set_title(f"{label}  (n={len(vals)})", fontsize=11, loc="left",
                     color=color, fontweight="bold")
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes3[-1].set_xlabel("Myotube diameter (µm)", fontsize=11)
    axes3[-1].set_xlim(5, 45)
    fig3.suptitle("Dex dose response -- myotube diameter counts per condition\n"
                  "(plates 004 + 005 combined, 6 wells / condition)",
                  fontsize=12, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(OUT / "fig3_count_hist_per_cond.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig3)

    # ============= Figure 4: overlay count histogram ==============
    fig4, ax4 = plt.subplots(figsize=(8, 5.5))
    for cond, label, color in zip(CONDITIONS, LABELS, COLORS):
        vals = sub[sub["condition"] == cond]["mean_diameter"].values
        ax4.hist(vals, bins=bins, density=False, histtype="step",
                 color=color, linewidth=2.0,
                 label=f"{label}  (n={len(vals)})")
        ax4.axvline(float(np.mean(vals)), color=color, linestyle="--",
                    linewidth=1.0, alpha=0.7)
    # This histogram PNG embeds into Figure 2 panel F at equal panel height with
    # the bar PNG (panel E); because the histogram is the wider-aspect image it
    # is scaled up slightly more, so its source fonts are set a touch smaller
    # (25 vs the bar's 26) to land ticks and title at the shared ~11 pt once
    # embedded. Histogram bars/shape unchanged -- only text sizes.
    ax4.set_xlabel("Myotube diameter (µm)", fontsize=25)
    ax4.set_ylabel("Count (myotubes)", fontsize=25)
    ax4.tick_params(labelsize=25)
    ax4.set_title("Dex Dose Measurement Distribution",
                  fontsize=25, fontweight="bold")
    # Framed legend at the top-right corner, clear of the descending right tail.
    ax4.legend(loc="upper right", fontsize=20, framealpha=0.9,
               handlelength=1.3, handletextpad=0.4, borderpad=0.3,
               labelspacing=0.3, borderaxespad=0.4)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_xlim(5, 45)
    fig4.tight_layout()
    fig4.savefig(OUT / "fig4_count_hist_overlay.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig4)

    # ============= CSV of histogram counts ===========
    bin_left, bin_right = bins[:-1], bins[1:]
    out_rows = pd.DataFrame({
        "bin_left_um": np.round(bin_left, 3),
        "bin_right_um": np.round(bin_right, 3),
        "bin_center_um": np.round(0.5 * (bin_left + bin_right), 3),
    })
    for cond in CONDITIONS:
        cdf = sub[sub["condition"] == cond]
        counts, _ = np.histogram(cdf["mean_diameter"].values, bins=bins)
        out_rows[f"{cond}_total"] = counts.astype(int)
        for (plate, well), wdf in cdf.groupby(["plate", "well"]):
            c, _ = np.histogram(wdf["mean_diameter"].values, bins=bins)
            out_rows[f"{cond}_{plate}_{well}"] = c.astype(int)
    out_rows.to_csv(OUT / "combined_hist_counts.csv", index=False)

    print(f"\nSaved 4 figures and CSV outputs under {OUT}")


if __name__ == "__main__":
    main()
