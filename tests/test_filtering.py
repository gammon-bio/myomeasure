"""Unit tests for object filtering."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from myotube.config import Config
from myotube.filtering import filter_myotubes


@pytest.fixture
def config():
    return Config(
        min_object_area=500,
        min_aspect_ratio=2.0,
        min_length_um=20.0,
        max_circularity=0.6,
        min_solidity=0.4,
    )


def _make_labels_with_shapes():
    """Create a label image with a myotube (elongated) and a round blob."""
    labels = np.zeros((300, 300), dtype=np.int32)

    # Label 1: Elongated myotube (200x20 pixels)
    labels[130:150, 50:250] = 1

    # Label 2: Round blob (radius ~25)
    yy, xx = np.ogrid[:300, :300]
    circle = ((yy - 250) ** 2 + (xx - 250) ** 2) < 25 ** 2
    labels[circle] = 2

    # Label 3: Tiny debris (10x5 pixels)
    labels[10:15, 10:20] = 3

    return labels


class TestFiltering:
    def test_keeps_elongated_myotube(self, config):
        """Elongated myotube should pass all filters."""
        labels = _make_labels_with_shapes()
        filtered = filter_myotubes(labels, config)

        # Label 1 (myotube) should be kept
        assert filtered.max() >= 1, "At least one object should remain"
        # Check that the elongated region has a label
        assert filtered[140, 150] > 0, "Myotube region should be labeled"

    def test_removes_round_blob(self, config):
        """Round blob should be removed by aspect ratio or circularity filter."""
        labels = _make_labels_with_shapes()
        filtered = filter_myotubes(labels, config)

        # Round blob region should be background
        assert filtered[250, 250] == 0, "Round blob should be filtered out"

    def test_removes_tiny_debris(self, config):
        """Tiny object should be removed by area filter."""
        labels = _make_labels_with_shapes()
        filtered = filter_myotubes(labels, config)

        assert filtered[12, 15] == 0, "Tiny debris should be filtered out"

    def test_empty_input(self, config):
        """Empty label image should return empty."""
        labels = np.zeros((100, 100), dtype=np.int32)
        filtered = filter_myotubes(labels, config)
        assert filtered.max() == 0

    def test_relabeling_sequential(self, config):
        """Filtered labels should be sequential starting from 1."""
        labels = _make_labels_with_shapes()
        filtered = filter_myotubes(labels, config)

        unique = set(np.unique(filtered)) - {0}
        if unique:
            assert unique == set(range(1, max(unique) + 1)), \
                "Labels should be sequential"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
