"""Distribution-shape statistics for Figure 6.

Tests whether each rater's diameter distribution on the 18-image C26 subset
is consistent with normality and / or lognormality, and reports the
Kolmogorov-Smirnov *D* statistic against the pipeline as a descriptive
distributional distance. Reviewer identities are anonymised (1 / 2 / 3).

Note on independence and on power matching
------------------------------------------
Per-myotube measurements within a single image (and within a single well)
are not independent, so KS / Mann-Whitney *p*-values pooled across
thousands of myotubes per arm would be pseudoreplicated. No object-level
*p*-value is therefore reported for the pipeline comparison: only the KS
*D* statistic (ks_D_vs_pipeline) is kept, as a descriptive
distributional-shape distance and the headline distributional distance.
The KS and Mann-Whitney *p*-values are omitted.

Per-image Shapiro pass rates are reported in two flavours:

  - raw, on the actual per-image sample (rater-specific *n*); and
  - power-matched, on each image downsampled to the minimum *n* across
    raters for that image (seeded), so cross-rater pass rates are
    compared at equal sample size and are not artefacts of different
    statistical power.

Reads:
  - data/subset_blinded_validation/reviewer1_long.csv          (rev 1)
  - data/subset_blinded_validation/reviewer2_long.csv      (rev 2)
  - data/subset_blinded_validation/reviewer3_long.csv      (rev 3)
  - data/subset_blinded_validation/_automated_subset.csv       (pipeline)

Writes (under reports/stats_final/):
  - table_distribution_summary.csv  (per-rater per-arm: skew, kurtosis,
                                     Shapiro-Wilk p (raw / log), KS D vs
                                     pipeline (descriptive distance))
  - normality_per_image.csv         (per-image Shapiro on raw and log
                                     diameter for each rater x image)

Run in the cellpose conda env:
    python scripts/paper_scripts_final/compute_normality_stats.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "data" / "subset_blinded_validation"
OUT = ROOT / "reports" / "stats_final"
OUT.mkdir(parents=True, exist_ok=True)

RATERS = [
    ("reviewer 1", SUBSET / "reviewer1_long.csv",        "diameter_um"),
    ("reviewer 2", SUBSET / "reviewer2_long.csv",    "diameter_um"),
    ("reviewer 3", SUBSET / "reviewer3_long.csv",    "diameter_um"),
    ("pipeline",   SUBSET / "_automated_subset.csv",     "mean_diameter"),
]


def shapiro_p(v: np.ndarray, rng: np.random.Generator) -> float:
    if len(v) < 3:
        return float("nan")
    s = v if len(v) <= 5000 else rng.choice(v, 5000, replace=False)
    return float(stats.shapiro(s).pvalue)


def main() -> None:
    rng = np.random.default_rng(0)
    rater_data = {}
    for tag, csv, col in RATERS:
        rater_data[tag] = pd.read_csv(csv).rename(columns={col: "diameter_um"})

    summary_rows = []
    for arm_name, raw in [("Control", "Control"),
                          ("C26 / atrophic", "C26_CM"),
                          ("Pooled", None)]:
        for tag, df in rater_data.items():
            if raw is None:
                v = df["diameter_um"].values
            else:
                v = df[df.condition == raw]["diameter_um"].values
            v = v[v > 0]
            if len(v) == 0:
                continue
            row = {
                "arm": arm_name,
                "rater": tag,
                "n_myotubes": len(v),
                "mean_um": float(v.mean()),
                "sd_um": float(v.std(ddof=1)),
                "skew": float(stats.skew(v)),
                "excess_kurt": float(stats.kurtosis(v)),
                "shapiro_p_raw": shapiro_p(v, rng),
                "shapiro_p_log": shapiro_p(np.log(v), rng),
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    ks_rows = []
    for arm_name, raw in [("Control", "Control"), ("C26 / atrophic", "C26_CM")]:
        pipe_v = rater_data["pipeline"]
        pipe_v = pipe_v[pipe_v.condition == raw]["diameter_um"].values
        pipe_v = pipe_v[pipe_v > 0]
        for tag in ["reviewer 1", "reviewer 2", "reviewer 3"]:
            v = rater_data[tag]
            v = v[v.condition == raw]["diameter_um"].values
            v = v[v > 0]
            ks = stats.ks_2samp(v, pipe_v)
            # Only the KS D statistic is reported, as a descriptive
            # distributional distance vs the pipeline. No object-level
            # p-value (KS or Mann-Whitney) is reported: per-myotube
            # observations are nested within images and wells, so a pooled
            # p-value would be pseudoreplicated.
            ks_rows.append({
                "arm": arm_name,
                "rater": tag,
                "ks_D_vs_pipeline": float(ks.statistic),
            })
    ks_df = pd.DataFrame(ks_rows)
    summary_df = summary_df.merge(ks_df, on=["arm", "rater"], how="left")

    summary_df.to_csv(OUT / "table_distribution_summary.csv", index=False)
    print("-- table_distribution_summary.csv --")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved {OUT / 'table_distribution_summary.csv'}")

    # Pass 1: raw per-image Shapiro (rater-specific n).
    rows = []
    for tag, df in rater_data.items():
        for bid, g in df.groupby("blinded_id"):
            v = g["diameter_um"].values
            v = v[v > 0]
            if len(v) < 3:
                continue
            rows.append({
                "rater": tag,
                "blinded_id": int(bid),
                "condition": g["condition"].iloc[0],
                "n_myotubes": len(v),
                "mean_um": float(v.mean()),
                "skew": float(stats.skew(v)),
                "shapiro_p": float(stats.shapiro(v).pvalue),
                "shapiro_log_p": float(stats.shapiro(np.log(v)).pvalue),
            })
    pi_df = pd.DataFrame(rows)

    # Pass 2: power-matched Shapiro. For each blinded image, downsample
    # every rater's per-image sample to the minimum n across raters for
    # that image (seeded), so cross-rater pass rates are compared at
    # equal sample size.
    pm_rng = np.random.default_rng(42)
    per_image_n = (
        pi_df.groupby("blinded_id")["n_myotubes"]
        .min()
        .to_dict()
    )
    pm_records = []
    for tag, df in rater_data.items():
        for bid, g in df.groupby("blinded_id"):
            v = g["diameter_um"].values
            v = v[v > 0]
            n_target = per_image_n.get(int(bid))
            if n_target is None or n_target < 3 or len(v) < n_target:
                continue
            sub_v = v if len(v) == n_target else pm_rng.choice(v, n_target, replace=False)
            pm_records.append({
                "rater": tag,
                "blinded_id": int(bid),
                "n_power_matched": int(n_target),
                "shapiro_p_pm": float(stats.shapiro(sub_v).pvalue),
                "shapiro_log_p_pm": float(stats.shapiro(np.log(sub_v)).pvalue),
            })
    pm_df = pd.DataFrame(pm_records)
    pi_df = pi_df.merge(pm_df, on=["rater", "blinded_id"], how="left")
    pi_df.to_csv(OUT / "normality_per_image.csv", index=False)

    print("\n-- per-image Shapiro-Wilk: images compatible with normality / "
          "lognormality (fail to reject at alpha = 0.05) --")
    print("    raw (rater-specific n)                 power-matched (min n per image)")
    for tag in ["reviewer 1", "reviewer 2", "reviewer 3", "pipeline"]:
        sub = pi_df[pi_df.rater == tag]
        n = len(sub)
        n_norm = int((sub.shapiro_p > 0.05).sum())
        n_log = int((sub.shapiro_log_p > 0.05).sum())
        n_norm_pm = int((sub.shapiro_p_pm > 0.05).sum())
        n_log_pm = int((sub.shapiro_log_p_pm > 0.05).sum())
        print(
            f"  {tag:11s}: {n_norm}/{n} compatible w/ normality, "
            f"{n_log}/{n} compatible w/ lognormality | "
            f"PM: {n_norm_pm}/{n} normality, {n_log_pm}/{n} lognormality "
            f"(median raw skew {sub['skew'].median():+.2f})"
        )
    print(f"\nSaved {OUT / 'normality_per_image.csv'}")


if __name__ == "__main__":
    main()
