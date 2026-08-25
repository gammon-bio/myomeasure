"""Measure myotube diameters from pre-computed cellpose masks.

Loads cellpose mask TIFs, applies morphological filtering, measures
diameters using skeleton + perpendicular ray-casting (TRUEFAD method),
and outputs per-image and combined CSV results.

Usage:
    python measure_from_masks.py <dir> [<dir2> ...] [--output-dir results]

Examples:
    python measure_from_masks.py data/B1/tiff data/B2/tiff
    python measure_from_masks.py data/A3/tiff data/B1/tiff data/B2/tiff data/C1/tiff data/C2/tiff
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from myotube.config import Config
from myotube.filtering import filter_myotubes
from myotube.io import parse_pixel_size
from myotube.measurement import measure_all_myotubes
from myotube.visualization import create_overlay, create_summary_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "image", "group", "label_id", "mean_diameter", "median_diameter",
    "min_diameter", "max_diameter", "std_diameter",
    "length", "area", "aspect_ratio", "n_branches", "unit",
]


def load_image_normalized(path: Path) -> np.ndarray:
    """Load a TIFF image and normalize to [0, 1] float64."""
    img = tifffile.imread(str(path)).astype(np.float64)
    if img.max() > 1.0:
        if img.max() <= 255:
            img /= 255.0
        elif img.max() <= 4095:
            img /= 4095.0
        elif img.max() <= 65535:
            img /= 65535.0
        else:
            img /= img.max()
    return img


def process_single(img_path: Path, mask_path: Path, group: str,
                    config: Config, output_dir: Path) -> pd.DataFrame:
    """Process a single image + mask pair."""
    image_name = img_path.name
    logger.info(f"Processing: {image_name}")

    # Load mask
    masks = tifffile.imread(str(mask_path)).astype(np.int32)
    n_raw = masks.max()
    logger.info(f"  Raw masks: {n_raw}")

    if n_raw == 0:
        logger.warning(f"  No masks found in {mask_path.name}")
        return pd.DataFrame(columns=CSV_COLUMNS)

    # Get pixel size from the original image metadata
    pixel_size, unit = parse_pixel_size(str(img_path), config.pixel_size)

    # Filter to keep only myotube-shaped objects
    filtered = filter_myotubes(masks, config, pixel_size)
    n_filtered = filtered.max()
    logger.info(f"  After filtering: {n_filtered} myotubes")

    if n_filtered == 0:
        logger.warning(f"  No myotubes remaining after filtering")
        return pd.DataFrame(columns=CSV_COLUMNS)

    # Measure diameters
    measurements = measure_all_myotubes(filtered, config, pixel_size, unit)

    # Generate QC overlay
    image = load_image_normalized(img_path)
    overlay_path = output_dir / f"{img_path.stem}_measurement_overlay.png"
    try:
        create_overlay(image, filtered, measurements, str(overlay_path), pixel_size)
    except Exception as e:
        logger.warning(f"  Failed to create overlay: {e}")

    # Build DataFrame
    rows = []
    for m in measurements:
        row = {"image": image_name, "group": group}
        for col in CSV_COLUMNS:
            if col not in ("image", "group"):
                row[col] = m.get(col, 0)
        rows.append(row)

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def main():
    parser = argparse.ArgumentParser(
        description="Measure myotube diameters from pre-computed cellpose masks."
    )
    parser.add_argument(
        "dirs", nargs="+",
        help="Directories containing original TIFs and *_cp_masks.tif files",
    )
    parser.add_argument(
        "--output-dir", "-o", default="results",
        help="Output directory for CSVs and overlays (default: results)",
    )
    parser.add_argument(
        "--mask-suffix", default="_cp_masks.tif",
        help="Suffix for mask files (default: _cp_masks.tif)",
    )
    args = parser.parse_args()

    config = Config()
    # Load config.yaml if it exists
    config_yaml = Path(__file__).parent / "config.yaml"
    if config_yaml.exists():
        import yaml
        with open(config_yaml) as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data:
            for key, val in yaml_data.items():
                if hasattr(config, key):
                    setattr(config, key, val)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for dir_path_str in args.dirs:
        dir_path = Path(dir_path_str)
        # Derive group name from parent folder(s)
        # For nested structures like Experiment_2/Control/A1/tiff -> "Control/A1"
        # For flat structures like Experiment_1/A3/tiff -> "A3"
        if dir_path.name == "tiff":
            well = dir_path.parent.name
            condition = dir_path.parent.parent.name
            # Check if the condition folder is an experiment root or a real condition
            if condition.startswith("Experiment") or condition == "test":
                group = well
            else:
                group = f"{condition}/{well}"
        else:
            group = dir_path.name

        mask_files = sorted(dir_path.glob(f"*{args.mask_suffix}"))
        if not mask_files:
            logger.warning(f"No mask files found in {dir_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Group: {group} ({len(mask_files)} images)")
        logger.info(f"{'='*50}")

        for mask_path in mask_files:
            stem = mask_path.name.replace(args.mask_suffix, ".tif")
            img_path = dir_path / stem

            if not img_path.exists():
                logger.warning(f"  Original image not found: {stem}")
                continue

            df = process_single(img_path, mask_path, group, config, output_dir)
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No images processed")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    valid = combined[combined["mean_diameter"] > 0]

    # Save combined CSV
    csv_path = output_dir / "measurements.csv"
    combined.to_csv(csv_path, index=False)
    logger.info(f"\nSaved combined results to {csv_path}")

    # Save per-group summary
    if len(valid) > 0:
        summary = valid.groupby("group")["mean_diameter"].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).round(2)
        summary_path = output_dir / "group_summary.csv"
        summary.to_csv(summary_path)
        logger.info(f"Saved group summary to {summary_path}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("MEASUREMENT SUMMARY")
        logger.info(f"{'='*60}")
        for grp, row in summary.iterrows():
            logger.info(f"  {grp}: n={int(row['count'])}, "
                         f"mean={row['mean']:.2f}, median={row['median']:.2f}, "
                         f"SD={row['std']:.2f} um")
        logger.info(f"{'='*60}")

        # Summary figure
        try:
            all_measurements = valid.to_dict("records")
            summary_fig_path = output_dir / "summary.png"
            create_summary_figure(all_measurements, str(summary_fig_path))
        except Exception as e:
            logger.warning(f"Failed to create summary figure: {e}")


if __name__ == "__main__":
    main()
