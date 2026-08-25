"""C26 conditioned-media atrophy effect size (well-level), on plate 016 of the
C26 conditioned-media experiment, reported for the 18-image blinded subset and
for the full 30-image plate it was drawn from.

The experimental unit of inference is the well (3 wells per arm), so Cohen's d
is computed from well-mean diameters with a well-level pooled SD (n = wells) —
consistent with the well-level Welch t-test plotted in Figure 3C. A per-image-
mean d is reported alongside as a secondary descriptive aggregate (image mean
is NOT the inferential unit and is provided for context only).

The earlier subset-vs-excluded representativeness hypothesis tests (a KS test
on per-myotube diameters and a Mann-Whitney U test on per-image means) have
been removed: representativeness is a property of the sampling design and is
addressed by the pre-specified, unbiased sampling strategy (3 images per well,
matched wells across arms), not by hypothesis testing.

Reads:
  - results/c26_cm_exp2_inference/measurements.csv
  - data/subset_blinded_validation/code_sheet.xlsx (subset image manifest)

Writes:
  - reports/stats_final/table_subset_representativeness.csv

Run in the cellpose conda env:
    python scripts/paper_scripts_final/compute_subset_representativeness.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FULL_CSV = ROOT / "results" / "c26_cm_exp2_inference" / "measurements.csv"
SUBSET_DIR = ROOT / "data" / "subset_blinded_validation"
CODE_XLSX = SUBSET_DIR / "code_sheet.xlsx"
OUT = ROOT / "reports" / "stats_final" / "table_subset_representativeness.csv"


def cohens_d(control: np.ndarray, treated: np.ndarray) -> float:
    """Cohen's d = (treated - control) / pooled SD, from the supplied
    samples (negative = atrophy)."""
    pooled = np.sqrt(((len(control) - 1) * control.var(ddof=1) +
                      (len(treated) - 1) * treated.var(ddof=1)) /
                     (len(control) + len(treated) - 2))
    return float((treated.mean() - control.mean()) / pooled)


def main() -> None:
    df = pd.read_csv(FULL_CSV)
    df = df[df["mean_diameter"] > 0].copy()
    df["condition"] = df["image"].str.extract(r"^(.+?)_Snapshot")[0]
    df["well"] = df["group"].str.extract(r"/([A-Z]\d+)$")[0]
    df["plate"] = df["group"].str.extract(r"plate_(\d+)")[0]
    plate = df[(df.plate == "016") & (df.condition.isin(["Con_Veh", "C26_CM"]))].copy()
    plate["image_base"] = plate["image"].str.replace(".tif", "", regex=False)

    code = pd.read_excel(CODE_XLSX)
    code.columns = ["real_id", "blinded_id", "well", "condition", "path"]
    code["image_base"] = code["real_id"].str.replace(".vsi", "", regex=False)
    subset_bases = set(code["image_base"])
    in_subset = plate[plate["image_base"].isin(subset_bases)]

    rows = []
    for label, sample in [("subset (18 images)", in_subset),
                          ("full_plate (30 images)", plate)]:
        # Well-level means (the inferential unit) — primary effect size.
        wm = (sample.groupby(["well", "condition"])["mean_diameter"]
              .mean().reset_index())
        con_w = wm[wm.condition == "Con_Veh"]["mean_diameter"].values
        c26_w = wm[wm.condition == "C26_CM"]["mean_diameter"].values
        d_well = cohens_d(con_w, c26_w)

        # Per-image means — secondary descriptive aggregate only.
        im = (sample.groupby(["image_base", "condition"])["mean_diameter"]
              .mean().reset_index())
        con_i = im[im.condition == "Con_Veh"]["mean_diameter"].values
        c26_i = im[im.condition == "C26_CM"]["mean_diameter"].values
        d_img = cohens_d(con_i, c26_i)

        con_mt = sample[sample.condition == "Con_Veh"]["mean_diameter"].values
        c26_mt = sample[sample.condition == "C26_CM"]["mean_diameter"].values
        delta = float(c26_w.mean() - con_w.mean())

        rows.append({
            "sample": label,
            "n_control_wells": int(len(con_w)),
            "n_c26_wells": int(len(c26_w)),
            "n_control_myotubes": int(len(con_mt)),
            "n_c26_myotubes": int(len(c26_mt)),
            "control_well_mean_um": round(float(con_w.mean()), 3),
            "c26_well_mean_um": round(float(c26_w.mean()), 3),
            "delta_um": round(delta, 3),
            "pct_change": round(100.0 * delta / con_w.mean(), 2),
            "cohens_d_well": round(d_well, 3),
            "cohens_d_per_image_mean": round(d_img, 3),
        })

    rep_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rep_df.to_csv(OUT, index=False)
    print(f"Saved {OUT}")
    print(rep_df.to_string(index=False))


if __name__ == "__main__":
    main()
