"""Run Cellpose inference + diameter measurement — replicates exact GUI workflow.

Loads raw images, runs Cellpose with cpsam model (auto diameter, flow=0.4,
cellprob=0.0), then filters and measures myotubes using the same TRUEFAD
pipeline as measure_from_masks.py.

No preprocessing. No fragment merging. No uint8 conversion.

Usage:
    python run_inference.py <dir1> [<dir2> ...] [-o output_dir]

Examples:
    python run_inference.py data/A1/tiff data/A2/tiff
    python run_inference.py data/A1/tiff -o results/exp1 --verbose
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from myotube.config import Config
from myotube.filtering import filter_myotubes
from myotube.io import parse_pixel_size
from myotube.measurement import measure_all_myotubes
from myotube.visualization import create_overlay, create_summary_figure

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "image", "group", "label_id", "mean_diameter", "median_diameter",
    "min_diameter", "max_diameter", "std_diameter",
    "length", "area", "aspect_ratio", "n_branches", "unit",
]

# Suffixes to skip when discovering images
SKIP_SUFFIXES = ("_cp_masks.tif", "_flows.tif", "_seg.npy", "_overlay.png",
                 "_measurement_overlay.png")


def discover_images(directory: Path) -> list[Path]:
    """Find original TIF images in a directory, skipping masks/flows/overlays."""
    images = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        if any(p.name.endswith(s) for s in SKIP_SUFFIXES):
            continue
        images.append(p)
    return images


def load_image_normalized(path: Path) -> np.ndarray:
    """Load a TIFF image and normalize to [0, 1] float64 (for overlays only)."""
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


def derive_group(dir_path: Path) -> str:
    """Derive group name from directory structure.

    Same logic as measure_from_masks.py:
      .../Experiment_1/A3/tiff -> "A3"
      .../Experiment_2/Control/A1/tiff -> "Control/A1"
    """
    if dir_path.name == "tiff":
        well = dir_path.parent.name
        condition = dir_path.parent.parent.name
        if condition.startswith("Experiment") or condition in ("test", "data"):
            return well
        return f"{condition}/{well}"
    return dir_path.name


def run_cellpose(images: list[Path], model, verbose: bool = False) -> dict[Path, np.ndarray]:
    """Run Cellpose on a list of images, returning {path: masks} dict.

    Exact GUI-equivalent settings:
    - Raw image (original dtype) — Cellpose normalizes internally
    - diameter=None (auto)
    - flow_threshold=0.4
    - cellprob_threshold=0.0
    - Grayscale: pass 2D array directly (channels deprecated in v4.0.1+)
    """
    results = {}
    for img_path in images:
        t0 = time.time()
        raw = tifffile.imread(str(img_path))
        logger.info(f"  {img_path.name}: shape={raw.shape}, dtype={raw.dtype}")

        # Ensure 2D grayscale — Cellpose v4+ auto-detects grayscale from shape
        if raw.ndim == 3:
            raw = raw[:, :, 0] if raw.shape[2] <= 4 else raw[0]

        try:
            masks, flows, styles = model.eval(
                raw,
                diameter=None,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )
        except Exception as e:
            # MPS GPU can hit sporadic index-out-of-bounds errors in flow
            # computation. Retry on CPU for this image.
            logger.warning(f"  GPU failed ({type(e).__name__}), retrying on CPU...")
            from cellpose.models import CellposeModel
            cpu_model = CellposeModel(gpu=False, pretrained_model=model.pretrained_model)
            masks, flows, styles = cpu_model.eval(
                raw,
                diameter=None,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )

        masks = masks.astype(np.int32)
        n_masks = masks.max()
        elapsed = time.time() - t0
        logger.info(f"  -> {n_masks} raw masks ({elapsed:.1f}s)")
        results[img_path] = masks

    return results


def process_single(img_path: Path, masks: np.ndarray, group: str,
                   config: Config, output_dir: Path,
                   save_masks: bool = True,
                   save_overlay: bool = True) -> pd.DataFrame:
    """Filter, measure, and create overlay for a single image."""
    image_name = img_path.name
    n_raw = masks.max()

    if n_raw == 0:
        logger.warning(f"  No masks found for {image_name}")
        return pd.DataFrame(columns=CSV_COLUMNS)

    # Save masks alongside original
    if save_masks:
        mask_path = img_path.parent / f"{img_path.stem}_cp_masks.tif"
        tifffile.imwrite(str(mask_path), masks.astype(np.uint16))
        logger.info(f"  Saved masks to {mask_path.name}")

    # Pixel size
    pixel_size, unit = parse_pixel_size(str(img_path), config.pixel_size)

    # Filter
    filtered = filter_myotubes(masks, config, pixel_size)
    n_filtered = filtered.max()
    logger.info(f"  After filtering: {n_filtered} myotubes (from {n_raw} raw)")

    if n_filtered == 0:
        logger.warning(f"  No myotubes remaining after filtering")
        return pd.DataFrame(columns=CSV_COLUMNS)

    # Measure
    measurements = measure_all_myotubes(filtered, config, pixel_size, unit)

    # Overlay
    if save_overlay:
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
        description="Run Cellpose inference + myotube diameter measurement "
                    "(replicates exact GUI workflow).",
    )
    parser.add_argument(
        "dirs", nargs="+",
        help="Directories containing TIF images to process",
    )
    parser.add_argument(
        "--output-dir", "-o", default="results",
        help="Output directory for CSVs, overlays, and summary (default: results)",
    )
    parser.add_argument(
        "--model", default="cpsam",
        help="Cellpose model name (default: cpsam)",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=None,
        help="Pixel size in um/px (overrides TIFF metadata)",
    )
    parser.add_argument(
        "--no-save-masks", dest="save_masks", action="store_false",
        help="Do not save *_cp_masks.tif alongside originals",
    )
    parser.add_argument(
        "--no-overlay", dest="save_overlay", action="store_false",
        help="Skip overlay generation",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # --- Logging setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(message)s"

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_format, datefmt="%H:%M:%S"))

    # File handler
    file_handler = logging.FileHandler(output_dir / "processing.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt="%H:%M:%S"))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    # --- Config (filtering/measurement params only) ---
    config = Config()
    config_yaml = Path(__file__).parent / "config.yaml"
    if config_yaml.exists():
        import yaml
        with open(config_yaml) as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data:
            # Only apply filtering and measurement params, NOT cellpose params
            filtering_keys = {
                "min_object_area", "min_aspect_ratio", "min_length_um",
                "max_circularity", "min_solidity",
                "num_diameter_samples", "skeleton_prune_length",
            }
            for key, val in yaml_data.items():
                if key in filtering_keys and hasattr(config, key):
                    setattr(config, key, val)

    if args.pixel_size is not None:
        config.pixel_size = args.pixel_size

    # --- Load Cellpose model once ---
    logger.info(f"Loading Cellpose model: {args.model}")
    from cellpose.models import CellposeModel
    model = CellposeModel(gpu=True, pretrained_model=args.model)
    logger.info("Model loaded")

    # --- Process each directory ---
    all_dfs = []

    for dir_str in args.dirs:
        dir_path = Path(dir_str)
        if not dir_path.is_dir():
            logger.error(f"Not a directory: {dir_path}")
            continue

        group = derive_group(dir_path)
        images = discover_images(dir_path)

        if not images:
            logger.warning(f"No TIF images found in {dir_path}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Group: {group} ({len(images)} images)")
        logger.info(f"{'='*50}")

        # Run Cellpose on all images in this directory
        mask_results = run_cellpose(images, model, verbose=args.verbose)

        # Process each image
        for img_path, masks in mask_results.items():
            df = process_single(
                img_path, masks, group, config, output_dir,
                save_masks=args.save_masks,
                save_overlay=args.save_overlay,
            )
            all_dfs.append(df)

    # --- Combine and save results ---
    if not all_dfs:
        logger.error("No images processed")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    valid = combined[combined["mean_diameter"] > 0]

    csv_path = output_dir / "measurements.csv"
    combined.to_csv(csv_path, index=False)
    logger.info(f"\nSaved combined results to {csv_path}")

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
