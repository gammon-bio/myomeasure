"""Configuration management: dataclass defaults, YAML loading, CLI overrides."""

import argparse
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Preprocessing
    gaussian_sigma: float = 2.0
    clahe_clip_limit: float = 3.0
    clahe_tile_size: int = 64

    # Segmentation - Classical
    adaptive_block_size: int = 127
    adaptive_c: float = -5.0
    morph_open_size: int = 5
    morph_close_size: int = 15

    # Segmentation - Cellpose
    cellpose_model: str = "cyto2"
    cellpose_diameter: int = 300
    cellpose_flow_threshold: float = 0.8
    cellpose_cellprob_threshold: float = -2.0

    # Fragment merging
    merge_distance: int = 30
    merge_angle_threshold: float = 30.0

    # Method selection
    method: str = "auto"

    # Object filtering
    min_object_area: int = 2000
    min_aspect_ratio: float = 3.0
    min_length_um: float = 50.0
    max_circularity: float = 0.5
    min_solidity: float = 0.3

    # Measurement
    num_diameter_samples: int = 9
    skeleton_prune_length: int = 20

    # Scale
    pixel_size: Optional[float] = None

    # Output
    save_overlay: bool = True
    save_summary: bool = True
    verbose: bool = False

    # Runtime (set by CLI, not in YAML)
    input_path: Optional[str] = None
    output_dir: Optional[str] = None
    config_path: Optional[str] = None

    def validate(self):
        """Validate parameter ranges."""
        assert self.gaussian_sigma > 0, "gaussian_sigma must be positive"
        assert self.clahe_clip_limit > 0, "clahe_clip_limit must be positive"
        assert self.clahe_tile_size > 0, "clahe_tile_size must be positive"
        assert self.adaptive_block_size > 0 and self.adaptive_block_size % 2 == 1, \
            "adaptive_block_size must be a positive odd integer"
        assert self.morph_open_size > 0, "morph_open_size must be positive"
        assert self.morph_close_size > 0, "morph_close_size must be positive"
        assert self.min_object_area > 0, "min_object_area must be positive"
        assert self.min_aspect_ratio > 0, "min_aspect_ratio must be positive"
        assert 0 < self.max_circularity <= 1, "max_circularity must be in (0, 1]"
        assert 0 < self.min_solidity <= 1, "min_solidity must be in (0, 1]"
        assert self.num_diameter_samples >= 1, "num_diameter_samples must be >= 1"
        assert self.skeleton_prune_length >= 0, "skeleton_prune_length must be non-negative"
        assert self.method in ("auto", "cellpose", "classical"), \
            f"method must be 'auto', 'cellpose', or 'classical', got '{self.method}'"
        if self.pixel_size is not None:
            assert self.pixel_size > 0, "pixel_size must be positive"


def load_config(cli_args=None) -> Config:
    """Load config from YAML defaults, then apply CLI overrides.

    Priority: CLI args > YAML file > dataclass defaults.
    """
    parser = _build_parser()
    args = parser.parse_args(cli_args)

    config = Config()

    # Load YAML if specified or if default config.yaml exists
    yaml_path = args.config_path
    if yaml_path is None:
        default_yaml = Path(__file__).parent.parent / "config.yaml"
        if default_yaml.exists():
            yaml_path = str(default_yaml)

    if yaml_path and os.path.exists(yaml_path):
        logger.info(f"Loading config from {yaml_path}")
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        config_fields = {fld.name for fld in fields(Config)}
        for key, value in yaml_data.items():
            if key in config_fields and value is not None:
                setattr(config, key, value)

    # Apply CLI overrides (only non-None values)
    cli_dict = vars(args)
    for key, value in cli_dict.items():
        if value is not None:
            setattr(config, key, value)

    config.validate()
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure myotube diameters from fluorescence microscopy images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_path", nargs="?", default=None,
                        help="Path to image file or directory of images")
    parser.add_argument("--config", dest="config_path", default=None,
                        help="Path to YAML config file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: input_path/results)")
    parser.add_argument("--method", choices=["auto", "cellpose", "classical"],
                        default=None, help="Segmentation method")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="Pixel size in microns/pixel")
    parser.add_argument("--min-area", dest="min_object_area", type=int, default=None,
                        help="Minimum object area in pixels")
    parser.add_argument("--min-aspect-ratio", type=float, default=None,
                        help="Minimum aspect ratio for myotube filtering")
    parser.add_argument("--no-overlay", dest="save_overlay", action="store_false",
                        default=None, help="Skip overlay generation")
    parser.add_argument("--verbose", action="store_true", default=None,
                        help="Enable verbose/debug logging")
    return parser
