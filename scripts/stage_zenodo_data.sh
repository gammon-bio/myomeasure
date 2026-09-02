#!/usr/bin/env bash
# ============================================================================
# Stage the paper-relevant raw images (calibrated TIFF + Cellpose masks) for
# the Zenodo data deposit. Copies ONLY the wells/conditions actually used in
# the paper -- not the full 39 GB working tree.
#
#   bash scripts/stage_zenodo_data.sh
#
# Output: ./zenodo_data/ (gitignored) with SHA256SUMS and a README.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="zenodo_data"
rm -rf "$OUT"; mkdir -p "$OUT"

copy_tiff () {  # $1 = source well dir, $2 = dest dir ; copies *.tif (incl _cp_masks.tif)
  if [ -d "$1/tiff" ]; then
    mkdir -p "$2"
    cp "$1"/tiff/*.tif "$2"/ 2>/dev/null || true
  fi
}

echo "[C26]  plate_016: Con_Veh (A1,B1,C1) + C26_CM (A4,B4,C4)"
for w in A1 B1 C1 A4 B4 C4; do
  copy_tiff "data/real/c26_cm_exp2/Well plate_016/$w" "$OUT/c26_cm/Well_plate_016/$w"
done

echo "[Dexa] plate_004 + plate_005: all wells (Vehicle / 0.1 / 1 / 10 uM)"
for plate in "Well plate_004" "Well plate_005"; do
  for wd in "data/real/IGF_Dexa/$plate"/*/; do
    [ -d "$wd" ] || continue
    copy_tiff "${wd%/}" "$OUT/dexamethasone/${plate// /_}/$(basename "$wd")"
  done
done

echo "[Blinded subset] 18 re-coded images + codebook"
mkdir -p "$OUT/blinded_subset"
cp data/subset_blinded_validation/*.tif          "$OUT/blinded_subset/" 2>/dev/null || true
cp data/subset_blinded_validation/code_sheet.xlsx "$OUT/blinded_subset/" 2>/dev/null || true

# --- Per-myotube measurements + anonymised rater worksheets -----------------
# These are no longer tracked in the code repository (it ships code only), so
# the deposit must carry them: they are the inputs the analysis scripts need.
# Destination paths mirror the layout the scripts expect after download.
echo "[Measurements] per-myotube CSVs (analysis inputs)"
for expt in IGF_dexa IGF_dexa_plate005 c26_cm_exp2_inference; do
  if [ -f "results/$expt/measurements.csv" ]; then
    mkdir -p "$OUT/measurements/results/$expt"
    cp "results/$expt/measurements.csv" "$OUT/measurements/results/$expt/"
  fi
done

echo "[Rater data] anonymised blinded-rater worksheets + paired per-image table"
mkdir -p "$OUT/measurements/data/subset_blinded_validation/reviewer_measurements"
cp data/subset_blinded_validation/*.csv \
   "$OUT/measurements/data/subset_blinded_validation/" 2>/dev/null || true
cp data/subset_blinded_validation/code_sheet.xlsx \
   "$OUT/measurements/data/subset_blinded_validation/" 2>/dev/null || true
cp data/subset_blinded_validation/reviewer_measurements/*.xlsx \
   "$OUT/measurements/data/subset_blinded_validation/reviewer_measurements/" 2>/dev/null || true

cat > "$OUT/README.txt" <<'TXT'
MyoMeasure — deposited microscopy images and segmentation masks
================================================================
Paper-relevant subset only.
Each *.tif is a calibrated 16-bit TIFF (0.65 um/pixel embedded); each matching
*_cp_masks.tif is the integer-labelled Cellpose-SAM mask.

  c26_cm/Well_plate_016/{A1,B1,C1}   Control (Con_Veh)  — Figures 1,3,4-6,S1
  c26_cm/Well_plate_016/{A4,B4,C4}   C26 conditioned media (atrophic)
  dexamethasone/Well_plate_004/*     Dexamethasone dose-response, plate 1  — Figure 2
  dexamethasone/Well_plate_005/*     Dexamethasone dose-response, plate 2
  blinded_subset/                    18-image blinded re-measurement set (Figs 4-6)
  measurements/                      per-myotube CSVs + anonymised rater worksheets

Conditions are encoded in each filename prefix (e.g. Con_Veh_*, C26_CM_*, 0.1uM_*).

REPRODUCING THE PUBLISHED FIGURES AND STATISTICS
------------------------------------------------
The code repository ships code only; this deposit carries the data it needs.
Copy the contents of measurements/ over a clone of the repository, preserving
paths, so that you end up with:

  results/IGF_dexa/measurements.csv
  results/IGF_dexa_plate005/measurements.csv
  results/c26_cm_exp2_inference/measurements.csv
  data/subset_blinded_validation/*.csv  (+ reviewer_measurements/*.xlsx)

then run, from the repository root:

  python results/IGF_dexa_combined/make_figures.py      # dexamethasone statistics
  python scripts/paper_scripts_final/compute_reviewer_subset_stats.py
  python scripts/paper_scripts_final/compute_normality_stats.py
  python scripts/paper_scripts_final/compute_well_level_distribution_stats.py
  python scripts/paper_scripts_final/compute_subset_representativeness.py
  python scripts/paper_scripts_final/make_all_figures.py
  python scripts/paper_scripts_final/make_supp_xlsx.py

Analysis code: https://github.com/gammon-bio/myomeasure
TXT

( cd "$OUT" && find . -type f \( -name '*.tif' -o -name '*.xlsx' -o -name '*.csv' \) -print0 \
    | sort -z | xargs -0 shasum -a 256 > SHA256SUMS )

n=$(find "$OUT" -name '*.tif' | wc -l | tr -d ' ')
echo "Done: $n TIFFs staged in $OUT/  ($(du -sh "$OUT" | cut -f1))"
