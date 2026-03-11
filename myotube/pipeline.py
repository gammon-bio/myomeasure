"""Orchestrates the full processing pipeline."""

import logging
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import Config
from .filtering import filter_myotubes
from .io import discover_images, extract_green_channel, load_image, parse_pixel_size, save_csv
from .measurement import measure_all_myotubes
from .preprocessing import preprocess
from .segmentation import segment
from .visualization import create_overlay, create_summary_figure

logger = logging.getLogger(__name__)

# CSV output columns (exclude internal visualization keys)
CSV_COLUMNS = [
    "image", "label_id", "mean_diameter", "median_diameter",
    "min_diameter", "max_diameter", "std_diameter",
    "length", "area", "aspect_ratio", "n_branches", "unit",
]


def process_single_image(path: str, config: Config,
                         output_dir: Optional[str] = None) -> pd.DataFrame:
    """Process a single image through the full pipeline.

    Returns DataFrame with one row per myotube.
    """
    image_name = Path(path).name
    logger.info(f"Processing: {image_name}")

    # Load and extract green channel
    raw = load_image(path)
    green = extract_green_channel(raw)

    # Parse pixel size
    pixel_size, unit = parse_pixel_size(path, config.pixel_size)

    # Preprocess
    enhanced = preprocess(green, config)

    # Segment
    labels = segment(enhanced, config)
    n_raw = labels.max()
    logger.info(f"  Raw segmentation: {n_raw} objects")

    if n_raw == 0:
        logger.warning(f"  No objects detected in {image_name}")
        return _empty_dataframe(image_name)

    # Filter
    labels = filter_myotubes(labels, config, pixel_size)
    n_filtered = labels.max()

    if n_filtered == 0:
        logger.warning(f"  No myotubes remaining after filtering in {image_name}")
        return _empty_dataframe(image_name)

    # Measure
    measurements = measure_all_myotubes(labels, config, pixel_size, unit)

    # Overlay
    if config.save_overlay and output_dir:
        overlay_path = str(Path(output_dir) / f"{Path(path).stem}_overlay.png")
        try:
            create_overlay(green, labels, measurements, overlay_path, pixel_size)
        except Exception as e:
            logger.warning(f"  Failed to create overlay: {e}")

    # Build DataFrame
    rows = []
    for m in measurements:
        row = {"image": image_name}
        for col in CSV_COLUMNS:
            if col != "image":
                row[col] = m.get(col, 0)
        rows.append(row)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    logger.info(f"  Measured {len(df)} myotubes, "
                f"mean diameter: {df['mean_diameter'].mean():.1f} {unit}")
    return df


def process_batch(input_path: str, config: Config) -> pd.DataFrame:
    """Process a batch of images or a single image.

    Args:
        input_path: Path to a single image or a directory.
        config: Config object.

    Returns:
        Combined DataFrame of all measurements.
    """
    input_p = Path(input_path)

    # Determine output directory
    if config.output_dir:
        output_dir = config.output_dir
    elif input_p.is_dir():
        output_dir = str(input_p / "results")
    else:
        output_dir = str(input_p.parent / "results")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set up logging to file
    log_path = Path(output_dir) / "processing.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # Discover images
    if input_p.is_dir():
        images = discover_images(input_path)
    elif input_p.is_file():
        images = [input_p]
    else:
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)

    if not images:
        logger.error(f"No images found in {input_path}")
        sys.exit(1)

    logger.info(f"Processing {len(images)} image(s), output to {output_dir}")

    # Process each image
    all_dfs = []
    all_measurements = []

    for img_path in images:
        try:
            df = process_single_image(str(img_path), config, output_dir)
            all_dfs.append(df)
            # Collect raw measurement dicts for summary
            for _, row in df.iterrows():
                all_measurements.append(row.to_dict())
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            logger.debug(traceback.format_exc())
            all_dfs.append(_empty_dataframe(img_path.name))

    # Combine results
    combined = pd.concat(all_dfs, ignore_index=True)

    # Save CSV
    csv_path = str(Path(output_dir) / "measurements.csv")
    save_csv(combined, csv_path)

    # Summary figure
    if config.save_summary and len(all_measurements) > 0:
        summary_path = str(Path(output_dir) / "summary.png")
        try:
            create_summary_figure(all_measurements, summary_path)
        except Exception as e:
            logger.warning(f"Failed to create summary figure: {e}")

    # Log summary
    valid = combined[combined["mean_diameter"] > 0]
    logger.info(f"\n{'='*50}")
    logger.info(f"BATCH COMPLETE")
    logger.info(f"  Images processed: {len(images)}")
    logger.info(f"  Total myotubes: {len(valid)}")
    if len(valid) > 0:
        logger.info(f"  Mean diameter: {valid['mean_diameter'].mean():.2f}")
        logger.info(f"  Median diameter: {valid['mean_diameter'].median():.2f}")
        logger.info(f"  SD: {valid['mean_diameter'].std():.2f}")
    logger.info(f"  Results: {csv_path}")
    logger.info(f"{'='*50}")

    # Cleanup file handler
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    return combined


def _empty_dataframe(image_name: str) -> pd.DataFrame:
    """Create empty DataFrame for images with no detected myotubes."""
    return pd.DataFrame(columns=CSV_COLUMNS)
