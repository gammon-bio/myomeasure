"""Stats for the four-rater C26 subset analysis (Figure 4 + supporting
supplementary tables). Anonymised — reviewers are reported as 1/2/3.

Reads:
  - data/subset_blinded_validation/reviewer1_long.csv          (rev 1)
  - data/subset_blinded_validation/reviewer2_long.csv      (rev 2)
  - data/subset_blinded_validation/reviewer3_long.csv      (rev 3)
  - data/subset_blinded_validation/_automated_subset.csv       (pipeline)
  - data/subset_blinded_validation/_paired4_per_image.csv

Writes (under reports/stats_final/):
  - table_r6_per_image_agreement.csv  (Pearson r, ICC, Bland-Altman per pair)
  - table_r6_well_level_welch.csv     (well-level Welch per rater)
  - table_r6_asymmetric_jump.csv      (Control→C26 pp jump per rater)
  - table_sd_cv_per_rater.csv         (myotube + well-level SD/CV per rater)

All re-measurements include image 1 (the original outlier was re-traced).
The "without image 1" sensitivity columns are no longer reported. The
unblinded-operator (Exp 1) reference is excluded from these tables — it
did not measure this subset.

Run in the cellpose conda env:
    python scripts/paper_scripts_final/compute_reviewer_subset_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "data" / "subset_blinded_validation"
OUT = ROOT / "reports" / "stats_final"
OUT.mkdir(parents=True, exist_ok=True)

R1_LONG   = SUBSET / "reviewer1_long.csv"
R2_LONG   = SUBSET / "reviewer2_long.csv"
R3_LONG   = SUBSET / "reviewer3_long.csv"
AUTO_LONG = SUBSET / "_automated_subset.csv"
PAIRED4   = SUBSET / "_paired4_per_image.csv"


def icc31_absolute(X: np.ndarray) -> float:
    n, k = X.shape
    grand = X.mean()
    ssb = k * ((X.mean(axis=1) - grand) ** 2).sum()
    ssw = ((X - X.mean(axis=1, keepdims=True)) ** 2).sum()
    ssc = n * ((X.mean(axis=0) - grand) ** 2).sum()
    sse = ssw - ssc
    msb = ssb / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))
    return (msb - mse) / (msb + (k - 1) * mse + k * (msc - mse) / n)


def pair_stats(df: pd.DataFrame, col_a: str, col_b: str, tag: str) -> dict:
    sub = df[[col_a, col_b]].dropna()
    a = sub[col_a].values
    b = sub[col_b].values
    if len(a) < 3:
        return {"comparison": tag, "n_images": len(a),
                "pearson_r": np.nan, "pearson_p": np.nan,
                "icc31_abs": np.nan,
                "mean_bias_um": np.nan, "sd_bias_um": np.nan,
                "loa_lower_um": np.nan, "loa_upper_um": np.nan}
    r, pv = stats.pearsonr(a, b)
    diff = b - a
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    icc = icc31_absolute(np.column_stack([a, b]))
    return {
        "comparison": tag,
        "n_images": len(a),
        "pearson_r": r,
        "pearson_p": pv,
        "icc31_abs": icc,
        "mean_bias_um": bias,
        "sd_bias_um": sd,
        "loa_lower_um": bias - 1.96 * sd,
        "loa_upper_um": bias + 1.96 * sd,
    }


def well_level_welch(long_df: pd.DataFrame, value_col: str, tag: str) -> dict:
    well = long_df.groupby(["well", "condition"])[value_col].mean().reset_index()
    con = well[well.condition == "Control"][value_col].values
    c26 = well[well.condition == "C26_CM"][value_col].values
    t, pv = stats.ttest_ind(con, c26, equal_var=False)
    delta = float(c26.mean() - con.mean())
    n_control_myo = int((long_df["condition"] == "Control").sum())
    n_c26_myo     = int((long_df["condition"] == "C26_CM").sum())
    return {
        "rater": tag,
        "n_control_wells": int(len(con)),
        "n_c26_wells": int(len(c26)),
        "n_control_myotubes": n_control_myo,
        "n_c26_myotubes": n_c26_myo,
        "n_total_myotubes": n_control_myo + n_c26_myo,
        "control_mean_um": float(con.mean()),
        "c26_mean_um": float(c26.mean()),
        "delta_um": delta,
        "pct_change": 100.0 * delta / con.mean(),
        "welch_t": float(t),
        "welch_p": float(pv),
    }


# ── SD / CV helpers ──────────────────────────────────────────────────

def myotube_level(values: np.ndarray) -> dict:
    v = values[values > 0]
    if len(v) == 0:
        return dict(n_myotubes=0, mean_um=np.nan, sd_um=np.nan, cv_pct=np.nan)
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    return dict(n_myotubes=len(v), mean_um=mean, sd_um=sd,
                cv_pct=100.0 * sd / mean if mean > 0 else np.nan)


def well_mean_level(df: pd.DataFrame, value_col: str, arm_raw: str) -> dict:
    sub = df[df.condition == arm_raw]
    well_means = sub.groupby("well")[value_col].mean().values
    if len(well_means) == 0:
        return dict(n_wells=0, well_mean_um=np.nan,
                    well_sd_um=np.nan, well_cv_pct=np.nan)
    mean = float(well_means.mean())
    sd = float(well_means.std(ddof=1)) if len(well_means) > 1 else np.nan
    return dict(n_wells=len(well_means),
                well_mean_um=mean, well_sd_um=sd,
                well_cv_pct=100.0 * sd / mean if (mean > 0 and not np.isnan(sd)) else np.nan)


def main() -> None:
    r1   = pd.read_csv(R1_LONG)
    r2   = pd.read_csv(R2_LONG)
    r3   = pd.read_csv(R3_LONG)
    auto = pd.read_csv(AUTO_LONG)
    paired4 = pd.read_csv(PAIRED4)

    print(f"reviewer 1 myotubes : {len(r1)}")
    print(f"reviewer 2 myotubes : {len(r2)}")
    print(f"reviewer 3 myotubes : {len(r3)}")
    print(f"pipeline myotubes   : {len(auto)}")
    print(f"paired image rows   : {len(paired4)}")
    print()

    # ── Per-image pairwise agreement (all 18 images, image 1 included) ──
    pair_specs = [
        ("r1_mean", "auto_mean", "reviewer 1 vs pipeline"),
        ("r2_mean", "auto_mean", "reviewer 2 vs pipeline"),
        ("r3_mean", "auto_mean", "reviewer 3 vs pipeline"),
        ("r1_mean", "r2_mean",   "reviewer 1 vs reviewer 2"),
        ("r1_mean", "r3_mean",   "reviewer 1 vs reviewer 3"),
        ("r2_mean", "r3_mean",   "reviewer 2 vs reviewer 3"),
    ]
    pair_df = pd.DataFrame([pair_stats(paired4, a, b, tag) for a, b, tag in pair_specs])
    pair_df.to_csv(OUT / "table_r6_per_image_agreement.csv", index=False)
    print("-- table_r6_per_image_agreement.csv --")
    print(pair_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ── Well-level Welch t per rater ──
    welch_df = pd.DataFrame([
        well_level_welch(r1,   "diameter_um",   "reviewer 1"),
        well_level_welch(r2,   "diameter_um",   "reviewer 2"),
        well_level_welch(r3,   "diameter_um",   "reviewer 3"),
        well_level_welch(auto, "mean_diameter", "pipeline"),
    ])
    welch_df.to_csv(OUT / "table_r6_well_level_welch.csv", index=False)
    print("\n-- table_r6_well_level_welch.csv --")
    print(welch_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ── Asymmetric jump (Control → C26 pp difference); NO unblinded op ──
    def lt_pct(df: pd.DataFrame, col: str, arm_raw: str) -> float:
        v = df[df.condition == arm_raw][col]
        return 100.0 * float((v < 10).mean())

    raters_lt = [
        ("reviewer 1", r1,   "diameter_um"),
        ("reviewer 2", r2,   "diameter_um"),
        ("reviewer 3", r3,   "diameter_um"),
        ("pipeline",   auto, "mean_diameter"),
    ]
    jump_rows = []
    for tag, df, col in raters_lt:
        ctrl = lt_pct(df, col, "Control")
        c26  = lt_pct(df, col, "C26_CM")
        jump_rows.append({
            "rater": tag,
            "control_pct_lt_10um": ctrl,
            "c26_pct_lt_10um": c26,
            "jump_pp": c26 - ctrl,
            "ratio": c26 / ctrl if ctrl > 0 else np.inf,
        })
    jump_df = pd.DataFrame(jump_rows)
    jump_df.to_csv(OUT / "table_r6_asymmetric_jump.csv", index=False)
    print("\n-- table_r6_asymmetric_jump.csv --")
    print(jump_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ── SD / CV per rater (myotube and well-mean level) ──
    sdcv_rows = []
    for tag, df, col in raters_lt:
        df = df.rename(columns={col: "diameter_um"})
        for arm_label, arm_raw in [("Control", "Control"), ("C26 CM", "C26_CM")]:
            mt = myotube_level(df.loc[df.condition == arm_raw, "diameter_um"].values)
            wm = well_mean_level(df, "diameter_um", arm_raw)
            row = dict(rater=tag, arm=arm_label)
            row.update(mt); row.update(wm)
            sdcv_rows.append(row)
    sdcv_df = pd.DataFrame(sdcv_rows, columns=[
        "rater", "arm",
        "n_myotubes", "mean_um", "sd_um", "cv_pct",
        "n_wells", "well_mean_um", "well_sd_um", "well_cv_pct",
    ])
    # Reproducibility fold-ratio: each rater's within-arm CV divided by the
    # pipeline's within-arm CV (same arm). The per-well-mean fold
    # (well_cv_fold_vs_pipeline) is the quantity behind the "N-fold more
    # reproducible than blinded raters" statement in Results/Discussion; the
    # per-myotube fold is reported alongside for completeness.
    pipe = sdcv_df[sdcv_df.rater == "pipeline"].set_index("arm")
    sdcv_df["cv_fold_vs_pipeline"] = sdcv_df.apply(
        lambda r: round(r["cv_pct"] / pipe.loc[r["arm"], "cv_pct"], 2), axis=1)
    sdcv_df["well_cv_fold_vs_pipeline"] = sdcv_df.apply(
        lambda r: round(r["well_cv_pct"] / pipe.loc[r["arm"], "well_cv_pct"], 2)
        if pd.notna(r["well_cv_pct"]) else np.nan, axis=1)
    sdcv_df.to_csv(OUT / "table_sd_cv_per_rater.csv", index=False)
    print("\n-- table_sd_cv_per_rater.csv --")
    print(sdcv_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nSaved 4 tables under {OUT}")


if __name__ == "__main__":
    main()
