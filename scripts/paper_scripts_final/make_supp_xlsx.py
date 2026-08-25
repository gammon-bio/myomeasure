"""Generate Supplementary_tables.xlsx deterministically from the committed
per-analysis CSVs.

The supplement was previously hand-assembled in Excel, which let the sheet
numbering drift out of sync with the manuscript's Table S# references (see
audit/results_audit_c39429b.md). This script rebuilds sheets S1-S12 from the
canonical CSVs in a fixed order, so the numbering is reproducible and matches
the accepted-final manuscript:

  S1-S3   dexamethasone dose-response (Figure 2)            -> results/IGF_dexa_combined/
  S4-S10  C26 CM validation + blinded raters (Figs 3-6)     -> reports/stats_final/
  S11     dexamethasone global tests (Figure 2E)            -> results/IGF_dexa_combined/
  S12     per-well distributional remodeling (Figure S1)    -> reports/stats_final/

Sheet order follows Cell Reports Methods' cite-in-order rule, NOT topic. The dex
global-tests table is first cited in the STAR Methods dexamethasone subsection
(which precedes the first citation of the per-well distributional table, itself
first cited in a later STAR Methods subsection), so it takes S11 and the
per-well table moves to S12. That is why the dex ANOVA table sits at S11 rather
than next to its S1-S3 siblings -- do not "tidy" it back.

Run in the cellpose env:
    python scripts/paper_scripts_final/make_supp_xlsx.py

Writes reports/stats_final/Supplementary_tables_generated.xlsx (a new file, so
it does not clobber a copy open in Excel; review, then replace the hand-made
Supplementary_tables.xlsx with it).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SF = ROOT / "reports" / "stats_final"
DEXA = ROOT / "results" / "IGF_dexa_combined"
OUT = SF / "Supplementary_tables_generated.xlsx"

def dexa_global_tests() -> pd.DataFrame:
    """Tidy the two dexamethasone ANOVA tables into one sheet (S11).

    The one-way ANOVA on the 24 well means is the reported global test of the
    condition effect; the two-way condition x plate model is carried alongside
    it purely as the plate / interaction check that justifies pooling the two
    plates. The design is balanced (3 wells per condition per plate), so the
    condition sum of squares is identical in both models -- only the error term
    differs (20 df one-way vs 16 df two-way).
    """
    frames = []
    for csv, model, role in [
        (DEXA / "combined_anova_oneway.csv", "one-way (condition)",
         "primary global test"),
        (DEXA / "combined_anova_twoway.csv", "two-way (condition x plate)",
         "plate / interaction check"),
    ]:
        d = pd.read_csv(csv)
        d.insert(0, "model", model)
        d.insert(1, "role", role)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    return out[["model", "role", "term", "sum_sq", "df", "F", "p_value"]]


# (sheet, source CSV or DataFrame builder, one-line caption) — order defines
# S1..S12
TABLES = [
    ("S1",  DEXA / "combined_effect_sizes.csv",
     "Dexamethasone dose-response: each dose vs vehicle by Dunnett's test (Games-Howell where Levene indicates heterogeneous variance) with well-level effect sizes (Cohen's d, Hedges' g, 95% CIs)."),
    ("S2",  DEXA / "combined_condition_stats.csv",
     "Dexamethasone per-condition diameter distribution summary (well/myotube means, percentiles, skew, kurtosis, and the descriptive KS D-statistic vs vehicle; no p-value)."),
    ("S3",  DEXA / "combined_well_cv.csv",
     "Dexamethasone between-well coefficient of variation by condition."),
    ("S4",  SF / "table_subset_representativeness.csv",
     "C26 conditioned-media atrophy effect size (well-level), for the 18-image blinded subset and the full 30-image plate (well means; Cohen's d at the well level, with per-image-mean d for context)."),
    ("S5",  SF / "table_r6_per_image_agreement.csv",
     "Pairwise per-image agreement on the 18-image C26 CM subset (Pearson r, ICC(3,1) absolute-agreement, Bland-Altman)."),
    ("S6",  SF / "table_sd_cv_per_rater.csv",
     "Per-rater within-arm reproducibility on the 18-image subset (per-myotube and per-well-mean SD and CV, and the CV fold-ratio vs the pipeline within the same arm)."),
    ("S7",  SF / "table_r6_well_level_welch.csv",
     "Well-level Welch t-tests for the Control vs C26 CM contrast, per rater (18-image subset)."),
    ("S8",  SF / "table_r6_asymmetric_jump.csv",
     "Lower-tail capture by rater (fraction of measurements < 10 um): Control, C26 CM, jump (pp), and ratio."),
    ("S9",  SF / "normality_per_image.csv",
     "Per-image normality on the 18-image subset (Shapiro-Wilk on raw and log diameters; raw and power-matched)."),
    ("S10", SF / "table_distribution_summary.csv",
     "Per-arm distribution shape and descriptive distance to the pipeline reference (skew, excess kurtosis, Shapiro-Wilk p on raw and log diameters, and the descriptive KS D-statistic vs pipeline; no object-level p-values)."),
    ("S11", dexa_global_tests,
     "Dexamethasone dose-response global tests on well means (Figure 2E): the one-way ANOVA of condition (reported global test) and the two-way condition x plate ANOVA retained as the plate / interaction check that justifies pooling the two plates."),
    ("S12", SF / "table_supp_well_level_distribution.csv",
     "Per-well distributional remodeling by C26 CM (Figure S1 source data): 10 well-level metrics with Welch t-tests and Benjamini-Hochberg q-values."),
]


def main() -> None:
    missing = [str(src) for _, src, _ in TABLES
               if not callable(src) and not src.exists()]
    if missing:
        raise SystemExit("Missing source CSV(s):\n  " + "\n  ".join(missing))
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        for sheet, src, caption in TABLES:
            df = src() if callable(src) else pd.read_csv(src)
            df.to_excel(xw, sheet_name=sheet, index=False, startrow=1)
            xw.sheets[sheet]["A1"] = f"Table {sheet}. {caption}"
    print(f"Wrote {OUT}")
    print("Sheets:", ", ".join(t[0] for t in TABLES))


if __name__ == "__main__":
    main()
