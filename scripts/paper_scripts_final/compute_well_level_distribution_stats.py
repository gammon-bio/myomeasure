"""Per-well distribution statistics for the C26 conditioned-media
biological-validation experiment (plate 016: Con_Veh and C26_CM arms,
n = 3 wells per arm).

For every well we compute:
  - mean diameter (µm)
  - median diameter (µm)
  - SD of diameter (µm)
  - IQR (Q3 − Q1, µm)
  - 75th percentile (µm)
  - 90th percentile (µm)
  - % of myotubes < 12 µm
  - % of myotubes < 15 µm
  - % of myotubes > 20 µm
  - % of myotubes > 25 µm

The arm-level summary is mean ± SD across the 3 wells; arm comparisons
are Welch's t-tests on the 3 vs 3 well-level statistic (no object-level
tests). The 10 metrics in this table are an exploratory distributional
characterisation of how C26 conditioned media remodels the well-level
myotube diameter distribution; the primary endpoint (mean diameter) is
in Figure 3 and is repeated here as the first row. Because 10 hypothesis
tests are run on the same wells, raw Welch *p*-values are accompanied by
Benjamini-Hochberg FDR-adjusted *q*-values across the 10 metrics so the
table can be interpreted without inflating the false-positive rate.

Reads:
  - results/c26_cm_exp2_inference/measurements.csv

Writes:
  - reports/stats_final/table_supp_well_level_distribution.csv
    (paper-ready columns + raw numerics for downstream use)

Run in the cellpose conda env:
    python scripts/paper_scripts_final/compute_well_level_distribution_stats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "c26_cm_exp2_inference" / "measurements.csv"
OUT = ROOT / "reports" / "stats_final" / "table_supp_well_level_distribution.csv"


def load_per_myotube() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["condition"]  = df["image"].str.extract(r"^(.+?)_Snapshot")[0]
    df["well"]       = df["group"].str.extract(r"/([A-Z]\d+)$")[0]
    df["plate"]      = df["group"].str.extract(r"plate_(\d+)")[0]
    df["plate_well"] = df["plate"] + "_" + df["well"]
    df = df[(df["mean_diameter"] > 0)
            & (df["plate"] == "016")
            & (df["condition"].isin(["Con_Veh", "C26_CM"]))]
    return df


def per_well_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (condition, plate_well) with all the metrics."""
    rows = []
    for (cond, pw), g in df.groupby(["condition", "plate_well"], observed=True):
        v = g["mean_diameter"].values
        q1, q3 = np.percentile(v, [25, 75])
        rows.append({
            "condition": cond,
            "plate_well": pw,
            "n_myotubes": int(len(v)),
            "mean_um":         float(v.mean()),
            "median_um":       float(np.median(v)),
            "sd_um":           float(v.std(ddof=1)),
            "iqr_um":          float(q3 - q1),
            "p75_um":          float(np.percentile(v, 75)),
            "p90_um":          float(np.percentile(v, 90)),
            "pct_lt_12um":     float(100.0 * (v < 12).mean()),
            "pct_lt_15um":     float(100.0 * (v < 15).mean()),
            "pct_gt_20um":     float(100.0 * (v > 20).mean()),
            "pct_gt_25um":     float(100.0 * (v > 25).mean()),
        })
    return pd.DataFrame(rows)


# Metrics: (column key, display name, format string, units, "%/µm")
METRICS = [
    ("mean_um",     "Mean diameter",       "{m:.2f} ± {s:.2f}", "µm",  False),
    ("median_um",   "Median diameter",     "{m:.2f} ± {s:.2f}", "µm",  False),
    ("sd_um",       "SD of diameter",      "{m:.2f} ± {s:.2f}", "µm",  False),
    ("iqr_um",      "IQR",                 "{m:.2f} ± {s:.2f}", "µm",  False),
    ("p75_um",      "75th percentile",     "{m:.2f} ± {s:.2f}", "µm",  False),
    ("p90_um",      "90th percentile",     "{m:.2f} ± {s:.2f}", "µm",  False),
    ("pct_lt_12um", "% < 12 µm",           "{m:.1f} ± {s:.1f}", "%",   True),
    ("pct_lt_15um", "% < 15 µm",           "{m:.1f} ± {s:.1f}", "%",   True),
    ("pct_gt_20um", "% > 20 µm",           "{m:.1f} ± {s:.1f}", "%",   True),
    ("pct_gt_25um", "% > 25 µm",           "{m:.1f} ± {s:.1f}", "%",   True),
]


def format_arm(values: np.ndarray, fmt: str, unit: str) -> str:
    return f"{fmt.format(m=values.mean(), s=values.std(ddof=1))} {unit}"


def format_delta(con: np.ndarray, c26: np.ndarray, unit: str, is_pct: bool) -> str:
    delta = c26.mean() - con.mean()
    if is_pct:
        return f"{delta:+.1f} points"
    return f"{delta:+.2f} {unit}"


def main() -> None:
    df = load_per_myotube()
    pw = per_well_stats(df)
    print(f"Loaded {len(df)} myotubes across {pw['plate_well'].nunique()} wells "
          f"(Control = {(pw.condition=='Con_Veh').sum()}, "
          f"C26 CM = {(pw.condition=='C26_CM').sum()})")
    print()
    print(pw.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    raw_results = []
    for col, label, fmt, unit, is_pct in METRICS:
        con = pw.loc[pw.condition == "Con_Veh", col].values
        c26 = pw.loc[pw.condition == "C26_CM",  col].values
        t, p = stats.ttest_ind(con, c26, equal_var=False)
        raw_results.append((col, label, fmt, unit, is_pct, con, c26, t, p))

    # Benjamini-Hochberg FDR adjustment across the 10 metrics tested
    # on the same 3 vs 3 wells.
    pvals = np.array([r[8] for r in raw_results], dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    ranked = pvals[order]
    bh = ranked * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    q_adj = np.empty_like(bh)
    q_adj[order] = bh

    rows = []
    for (col, label, fmt, unit, is_pct, con, c26, t, p), q in zip(raw_results, q_adj):
        rows.append({
            "metric": label,
            "control":          format_arm(con, fmt, unit),
            "c26_cm":           format_arm(c26, fmt, unit),
            "delta_c26_minus_control": format_delta(con, c26, unit, is_pct),
            "welch_p":          f"{p:.4g}",
            "welch_q_BH":       f"{q:.4g}",
            # Raw numerics for downstream re-use:
            "control_mean":     float(con.mean()),
            "control_sd":       float(con.std(ddof=1)),
            "c26_mean":         float(c26.mean()),
            "c26_sd":           float(c26.std(ddof=1)),
            "delta_raw":        float(c26.mean() - con.mean()),
            "welch_t":          float(t),
            "welch_p_raw":      float(p),
            "welch_q_BH_raw":   float(q),
            "n_control_wells":  int(len(con)),
            "n_c26_wells":      int(len(c26)),
        })
    out = pd.DataFrame(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("Per-arm summary (well-level Welch's t-tests, n = 3 vs 3; "
          "BH-FDR adjusted across 10 metrics):")
    print(out[["metric", "control", "c26_cm",
               "delta_c26_minus_control", "welch_p", "welch_q_BH"]]
          .to_string(index=False))
    print()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
