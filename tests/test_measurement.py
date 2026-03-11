"""Unit tests for diameter measurement accuracy on synthetic images."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from myotube.config import Config
from myotube.measurement import (
    find_longest_path,
    measure_single_myotube,
    prune_skeleton,
)
from myotube.preprocessing import preprocess
from myotube.segmentation import segment_classical
from tests.generate_synthetic import straight_tube, curved_tube, two_parallel_tubes


@pytest.fixture
def config():
    return Config(method="classical")


class TestStraightTube:
    """Test diameter measurement on a straight tube with known width."""

    def test_straight_tube_diameter(self, config):
        """Straight tube of width 30 px should measure ~30 px diameter."""
        img = straight_tube(width=30, length=300, filename="_test_straight.tif")

        # Create binary mask by thresholding
        mask = img > 0.3

        result = measure_single_myotube(mask, label_id=1, config=config)

        assert result["mean_diameter"] > 0, "Mean diameter should be positive"
        assert abs(result["mean_diameter"] - 30) < 4, \
            f"Expected ~30 px diameter, got {result['mean_diameter']:.1f}"

    def test_straight_tube_length(self, config):
        """Skeleton length should approximate tube length."""
        img = straight_tube(width=30, length=300, filename="_test_straight_len.tif")
        mask = img > 0.3
        result = measure_single_myotube(mask, label_id=1, config=config)

        # Length should be roughly 300 px (with some tolerance for skeleton)
        assert result["length"] > 250, \
            f"Expected length ~300, got {result['length']:.1f}"
        assert result["length"] < 350, \
            f"Expected length ~300, got {result['length']:.1f}"


class TestCurvedTube:
    """Test diameter measurement on a curved tube."""

    def test_curved_tube_diameter(self, config):
        """Curved tube of width 25 px should measure ~25 px diameter."""
        img = curved_tube(width=25, filename="_test_curved.tif")
        mask = img > 0.3

        result = measure_single_myotube(mask, label_id=1, config=config)

        assert result["mean_diameter"] > 0, "Mean diameter should be positive"
        assert abs(result["mean_diameter"] - 25) < 5, \
            f"Expected ~25 px diameter, got {result['mean_diameter']:.1f}"


class TestWatershed:
    """Test that watershed separates touching tubes."""

    def test_two_tubes_separated(self, config):
        """Two parallel tubes should be segmented as 2 separate objects."""
        img = two_parallel_tubes(width=20, gap=2,
                                 filename="_test_parallel.tif")
        enhanced = preprocess(img, config)
        labels = segment_classical(enhanced, config)

        n_objects = labels.max()
        assert n_objects >= 2, \
            f"Expected >=2 objects from two parallel tubes, got {n_objects}"


class TestSkeletonOperations:
    """Test skeleton pruning and longest path finding."""

    def test_find_longest_path_straight(self):
        """Longest path of a straight skeleton should span full length."""
        skel = np.zeros((50, 200), dtype=bool)
        skel[25, 10:190] = True  # Horizontal line

        path = find_longest_path(skel)
        assert len(path) > 150, f"Path length {len(path)} too short"

    def test_prune_short_branch(self):
        """Short branches should be removed by pruning."""
        skel = np.zeros((50, 200), dtype=bool)
        skel[25, 10:190] = True  # Main line

        # Add short branch (5 px)
        for i in range(1, 6):
            skel[25 - i, 100] = True

        pruned = prune_skeleton(skel, min_branch_length=10)

        # Branch should be removed
        assert not pruned[20, 100], "Short branch should have been pruned"
        # Main line should remain
        assert pruned[25, 50], "Main skeleton should remain"

    def test_keep_long_branch(self):
        """Branches longer than threshold should be preserved."""
        skel = np.zeros((100, 200), dtype=bool)
        skel[50, 10:190] = True  # Main line

        # Add long branch (30 px)
        for i in range(1, 31):
            skel[50 - i, 100] = True

        pruned = prune_skeleton(skel, min_branch_length=10)

        # Branch should be kept
        assert pruned[25, 100], "Long branch should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
