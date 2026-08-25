"""Convert .vsi microscopy images to calibrated TIFF files.

Uses bioio + bioformats_jar to read Olympus .vsi files and export them
as 16-bit TIFF with embedded µm/pixel calibration (ImageJ-compatible).

Usage:
    python convert_vsi_to_tiff.py <input_dir> [--output-dir <output_dir>] [--scene <scene>]

Examples:
    python convert_vsi_to_tiff.py data/A1
    python convert_vsi_to_tiff.py data/A1 --output-dir data/A1_tiff
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile

from bioio_bioformats import Reader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_vsi_files(directory: Path) -> list[Path]:
    """Find all .vsi files in a directory."""
    vsi_files = sorted(directory.glob("*.vsi"))
    if not vsi_files:
        raise FileNotFoundError(f"No .vsi files found in {directory}")
    logger.info(f"Found {len(vsi_files)} .vsi files in {directory}")
    return vsi_files


def convert_single_vsi(
    vsi_path: Path,
    output_dir: Path,
    scene: str | None = None,
) -> Path:
    """Convert a single .vsi file to a calibrated TIFF.

    Parameters
    ----------
    vsi_path : Path
        Path to the .vsi file.
    output_dir : Path
        Directory to write the output TIFF.
    scene : str or None
        Scene name to extract. If None, uses the first non-macro scene.

    Returns
    -------
    Path
        Path to the written TIFF file.
    """
    logger.info(f"Reading {vsi_path.name}...")
    reader = Reader(vsi_path)

    # Select the appropriate scene
    if scene is not None:
        reader.set_scene(scene)
    else:
        # Pick the first scene that isn't the macro/overview image
        for s in reader.scenes:
            if "macro" not in s.lower():
                reader.set_scene(s)
                break
        else:
            reader.set_scene(reader.scenes[0])

    logger.info(f"  Scene: {reader.current_scene}")
    logger.info(f"  Shape: {reader.shape}, Dtype: {reader.dtype}")

    # Extract pixel calibration
    pps = reader.physical_pixel_sizes
    pixel_size_um = None
    if pps.X is not None:
        pixel_size_um = float(pps.X)
        logger.info(f"  Pixel size: {pixel_size_um:.4f} µm/pixel")
    else:
        logger.warning("  No pixel calibration found in metadata!")

    # Read image data — squeeze singleton dims (T, C, Z)
    data = reader.get_image_data("YX")  # returns 2D array (Y, X)
    logger.info(f"  Image data: {data.shape}, dtype={data.dtype}, "
                f"range=[{data.min()}, {data.max()}]")

    # Build output path
    output_path = output_dir / (vsi_path.stem + ".tif")

    # Write TIFF with ImageJ-compatible metadata for pixel calibration
    metadata = {}
    resolution = None
    resolution_unit = None

    if pixel_size_um is not None:
        # TIFF resolution = pixels per unit; we use RESUNIT=MICROMETER (not standard)
        # Instead, use ImageJ convention: resolution in pixels/um, metadata unit=um
        pixels_per_um = 1.0 / pixel_size_um
        resolution = (pixels_per_um, pixels_per_um)
        resolution_unit = None  # will set via ImageJ metadata
        metadata = {"unit": "um"}

    tifffile.imwrite(
        str(output_path),
        data,
        imagej=True,
        resolution=resolution,
        metadata=metadata,
    )

    logger.info(f"  Written: {output_path}")
    return output_path


def convert_batch(
    input_dir: Path,
    output_dir: Path | None = None,
    scene: str | None = None,
) -> list[dict]:
    """Convert all .vsi files in a directory to TIFF.

    Returns a list of dicts with conversion results including pixel calibration.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "tiff"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vsi_files = find_vsi_files(input_dir)
    results = []

    for vsi_path in vsi_files:
        try:
            reader = Reader(vsi_path)

            # Select scene
            if scene is not None:
                reader.set_scene(scene)
            else:
                for s in reader.scenes:
                    if "macro" not in s.lower():
                        reader.set_scene(s)
                        break

            pps = reader.physical_pixel_sizes
            pixel_size_um = float(pps.X) if pps.X is not None else None

            output_path = convert_single_vsi(vsi_path, output_dir, scene)

            results.append({
                "vsi_file": vsi_path.name,
                "tiff_file": output_path.name,
                "scene": reader.current_scene,
                "shape": reader.shape,
                "dtype": str(reader.dtype),
                "pixel_size_um": pixel_size_um,
                "status": "ok",
            })
        except Exception as e:
            logger.error(f"Failed to convert {vsi_path.name}: {e}")
            results.append({
                "vsi_file": vsi_path.name,
                "tiff_file": None,
                "status": f"error: {e}",
            })

    # Print summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    ok = [r for r in results if r["status"] == "ok"]
    print(f"  Converted: {len(ok)}/{len(results)} files")
    if ok:
        sizes = [r["pixel_size_um"] for r in ok if r.get("pixel_size_um")]
        if sizes:
            print(f"  Pixel size: {sizes[0]:.4f} µm/pixel")
        print(f"  Output dir: {output_dir}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert .vsi microscopy images to calibrated TIFF files."
    )
    parser.add_argument("input_dir", help="Directory containing .vsi files")
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for TIFF files (default: <input_dir>/tiff)",
    )
    parser.add_argument(
        "--scene", "-s",
        help="Scene name to extract (default: first non-macro scene)",
    )
    args = parser.parse_args()

    convert_batch(args.input_dir, args.output_dir, args.scene)


if __name__ == "__main__":
    main()
