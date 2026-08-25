"""Self-contained phantom tests for myotube MEASUREMENT + FILTERING.

Synthetic "phantom" tubes of known width are built in-memory (no disk I/O, no
fixtures dir, no cellpose/torch, no archived modules). Exercises only the
shipping ``myotube`` package:

    myotube.config.Config
    myotube.measurement.{measure_single_myotube, prune_skeleton, find_longest_path}
    myotube.filtering.filter_myotubes

This is the container build gate: ``pytest tests/ -q`` must pass for the image to
build. Run locally the same way (repo root on sys.path via the pyproject
``[tool.pytest.ini_options] pythonpath = ["."]`` setting).
"""

import numpy as np
import pytest
from skimage.draw import disk

from myotube.config import Config
from myotube.filtering import filter_myotubes
from myotube.measurement import (
    find_longest_path,
    measure_single_myotube,
    prune_skeleton,
)


# --------------------------------------------------------------------------- #
# In-memory synthetic phantom generators (return arrays, never touch disk)     #
# --------------------------------------------------------------------------- #
def straight_tube_mask(width=30, length=300, shape=(200, 400)):
    """Binary mask: one horizontal tube of exact ``width`` x ``length`` pixels."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    cy = h // 2
    x0 = (w - length) // 2
    y0 = cy - width // 2
    mask[y0:y0 + width, x0:x0 + length] = True
    return mask


def curved_tube_mask(width=25, shape=(300, 300)):
    """Binary mask: a constant-width arc swept from disks of radius width/2."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    t = np.linspace(0.30 * np.pi, 0.70 * np.pi, 600)
    radius, cx, cy, rad = 100, w // 2, h // 2 + 50, width / 2.0
    for ti in t:
        r = int(round(cy - radius * np.sin(ti)))
        c = int(round(cx + radius * np.cos(ti)))
        rr, cc = disk((r, c), rad, shape=mask.shape)
        mask[rr, cc] = True
    return mask


def shapes_label_image(shape=(300, 300)):
    """Label image: 1 = elongated myotube, 2 = round blob, 3 = tiny debris."""
    labels = np.zeros(shape, dtype=np.int32)
    labels[130:150, 50:250] = 1                                  # elongated 200x20
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    labels[((yy - 250) ** 2 + (xx - 250) ** 2) < 25 ** 2] = 2    # round blob r=25
    labels[10:15, 10:20] = 3                                     # tiny debris 5x10
    return labels


@pytest.fixture
def measure_config():
    return Config(num_diameter_samples=9, skeleton_prune_length=20)


@pytest.fixture
def filter_config():
    return Config(min_object_area=500, min_aspect_ratio=2.0, min_length_um=20.0,
                  max_circularity=0.6, min_solidity=0.4)


# --------------------------------------------------------------------------- #
# MEASUREMENT — diameter accuracy on phantoms of known width                   #
# --------------------------------------------------------------------------- #
class TestStraightTube:
    def test_diameter(self, measure_config):
        res = measure_single_myotube(straight_tube_mask(width=30, length=300),
                                     label_id=1, config=measure_config)
        assert res["mean_diameter"] > 0
        assert abs(res["mean_diameter"] - 30) < 4, res["mean_diameter"]

    def test_length(self, measure_config):
        res = measure_single_myotube(straight_tube_mask(width=30, length=300),
                                     label_id=1, config=measure_config)
        assert 250 < res["length"] < 350, res["length"]


class TestCurvedTube:
    def test_diameter(self, measure_config):
        res = measure_single_myotube(curved_tube_mask(width=25),
                                     label_id=1, config=measure_config)
        assert res["mean_diameter"] > 0
        assert abs(res["mean_diameter"] - 25) < 6, res["mean_diameter"]


# --------------------------------------------------------------------------- #
# SKELETON — longest path + prune                                              #
# --------------------------------------------------------------------------- #
class TestSkeleton:
    def test_longest_path_straight(self):
        skel = np.zeros((50, 200), dtype=bool)
        skel[25, 10:190] = True
        assert len(find_longest_path(skel)) > 150

    def test_prune_short_branch(self):
        skel = np.zeros((50, 200), dtype=bool)
        skel[25, 10:190] = True
        for i in range(1, 6):
            skel[25 - i, 100] = True
        pruned = prune_skeleton(skel, min_branch_length=10)
        assert not pruned[20, 100]      # short branch removed
        assert pruned[25, 50]           # main line kept

    def test_keep_long_branch(self):
        skel = np.zeros((100, 200), dtype=bool)
        skel[50, 10:190] = True
        for i in range(1, 31):
            skel[50 - i, 100] = True
        assert prune_skeleton(skel, min_branch_length=10)[25, 100]


# --------------------------------------------------------------------------- #
# FILTERING — shape-based myotube selection                                    #
# --------------------------------------------------------------------------- #
class TestFiltering:
    def test_keeps_elongated(self, filter_config):
        out = filter_myotubes(shapes_label_image(), filter_config)
        assert out.max() >= 1 and out[140, 150] > 0

    def test_removes_round_blob(self, filter_config):
        assert filter_myotubes(shapes_label_image(), filter_config)[250, 250] == 0

    def test_removes_tiny_debris(self, filter_config):
        assert filter_myotubes(shapes_label_image(), filter_config)[12, 15] == 0

    def test_empty_input(self, filter_config):
        assert filter_myotubes(np.zeros((100, 100), np.int32), filter_config).max() == 0

    def test_sequential_relabel(self, filter_config):
        out = filter_myotubes(shapes_label_image(), filter_config)
        uniq = set(np.unique(out)) - {0}
        if uniq:
            assert uniq == set(range(1, max(uniq) + 1))
