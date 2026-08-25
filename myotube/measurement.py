"""Core diameter measurement: skeleton + distance transform + perpendicular ray-casting.

TRUEFAD-inspired: 9 equidistant perpendicular diameter samples per myotube.
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


def measure_all_myotubes(labels: np.ndarray, config,
                         pixel_size: float = 1.0,
                         unit: str = "pixels") -> List[Dict]:
    """Measure diameter of all labeled myotubes.

    Args:
        labels: Integer-labeled instance mask.
        config: Config object.
        pixel_size: Microns per pixel.
        unit: Measurement unit string.

    Returns:
        List of dicts with measurements per myotube.
    """
    results = []
    n_labels = labels.max()

    if n_labels == 0:
        logger.warning("No myotubes to measure")
        return results

    for label_id in range(1, n_labels + 1):
        mask = (labels == label_id)
        try:
            measurements = measure_single_myotube(
                mask, label_id, config, pixel_size, unit
            )
            results.append(measurements)
        except Exception as e:
            logger.warning(f"Failed to measure myotube {label_id}: {e}")

    return results


def measure_single_myotube(mask: np.ndarray, label_id: int, config,
                           pixel_size: float = 1.0,
                           unit: str = "pixels") -> Dict:
    """Measure a single myotube.

    Steps:
    1. Skeletonize
    2. Prune short branches
    3. Find longest path (medial axis)
    4. Measure diameters at N equidistant points
    5. Compute summary metrics
    """
    # Skeletonize
    skeleton = skeletonize(mask)

    # Prune short branches
    if config.skeleton_prune_length > 0:
        skeleton = prune_skeleton(skeleton, config.skeleton_prune_length)

    # Count branches before finding longest path
    branch_points = _find_branch_points(skeleton)
    n_branches = len(branch_points)

    # Find longest path through skeleton
    path = find_longest_path(skeleton)

    if len(path) < 3:
        logger.debug(f"Myotube {label_id}: skeleton path too short ({len(path)} px)")
        return _empty_result(label_id, mask, pixel_size, unit, n_branches)

    # Distance transform of the mask
    dt = distance_transform_edt(mask)

    # Sample N equidistant points along the path
    n_samples = min(config.num_diameter_samples, len(path))
    indices = np.linspace(0, len(path) - 1, n_samples, dtype=int)
    sample_points = [path[i] for i in indices]

    # Measure diameters
    diameters_ray = []
    diameters_dt = []
    measurement_lines = []  # For visualization

    for idx, (r, c) in enumerate(sample_points):
        # Distance transform diameter
        d_dt = 2.0 * dt[r, c] * pixel_size
        diameters_dt.append(d_dt)

        # Perpendicular ray-casting diameter
        tangent = _compute_local_tangent(path, indices[idx], window=5)
        d_ray, line_pts = _cast_perpendicular_rays(mask, r, c, tangent, pixel_size)
        diameters_ray.append(d_ray)
        measurement_lines.append(line_pts)

    # Use ray-casting as primary measurement
    diameters = np.array(diameters_ray)
    diameters_dt_arr = np.array(diameters_dt)

    # Sanity check: flag large discrepancies
    valid_mask = (diameters > 0) & (diameters_dt_arr > 0)
    if valid_mask.any():
        ratio = diameters[valid_mask] / diameters_dt_arr[valid_mask]
        if np.any(ratio > 2.0) or np.any(ratio < 0.5):
            logger.debug(f"Myotube {label_id}: ray/DT diameter discrepancy detected")

    # Filter out zero diameters (edge cases)
    valid_diameters = diameters[diameters > 0]
    if len(valid_diameters) == 0:
        valid_diameters = diameters_dt_arr[diameters_dt_arr > 0]
    if len(valid_diameters) == 0:
        return _empty_result(label_id, mask, pixel_size, unit, n_branches)

    # Skeleton length
    path_arr = np.array(path)
    diffs = np.diff(path_arr, axis=0)
    skeleton_length = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))) * pixel_size

    # Area in calibrated units
    area = np.sum(mask) * (pixel_size ** 2)

    # Aspect ratio from regionprops-style calculation
    from skimage.measure import regionprops
    props = regionprops(mask.astype(int))
    if props:
        aspect_ratio = props[0].axis_major_length / max(props[0].axis_minor_length, 1)
    else:
        aspect_ratio = 0

    return {
        "label_id": label_id,
        "mean_diameter": float(np.mean(valid_diameters)),
        "median_diameter": float(np.median(valid_diameters)),
        "min_diameter": float(np.min(valid_diameters)),
        "max_diameter": float(np.max(valid_diameters)),
        "std_diameter": float(np.std(valid_diameters)),
        "length": float(skeleton_length),
        "area": float(area),
        "aspect_ratio": float(aspect_ratio),
        "n_branches": n_branches,
        "unit": unit,
        # Store for visualization (not saved to CSV)
        "_skeleton_path": path,
        "_measurement_lines": measurement_lines,
        "_sample_points": sample_points,
    }


def _empty_result(label_id, mask, pixel_size, unit, n_branches):
    area = np.sum(mask) * (pixel_size ** 2)
    return {
        "label_id": label_id,
        "mean_diameter": 0.0,
        "median_diameter": 0.0,
        "min_diameter": 0.0,
        "max_diameter": 0.0,
        "std_diameter": 0.0,
        "length": 0.0,
        "area": float(area),
        "aspect_ratio": 0.0,
        "n_branches": n_branches,
        "unit": unit,
        "_skeleton_path": [],
        "_measurement_lines": [],
        "_sample_points": [],
    }


# ---------------------------------------------------------------------------
# Skeleton operations
# ---------------------------------------------------------------------------

def _get_neighbors(skeleton: np.ndarray, r: int, c: int) -> List[Tuple[int, int]]:
    """Get 8-connected skeleton neighbors of a pixel."""
    h, w = skeleton.shape
    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc]:
                neighbors.append((nr, nc))
    return neighbors


def _classify_skeleton_pixels(skeleton: np.ndarray):
    """Classify skeleton pixels into endpoints, branch points, and body pixels."""
    coords = np.argwhere(skeleton)
    endpoints = []
    branch_points = []

    for r, c in coords:
        n = len(_get_neighbors(skeleton, r, c))
        if n == 1:
            endpoints.append((r, c))
        elif n > 2:
            branch_points.append((r, c))

    return endpoints, branch_points


def _find_branch_points(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """Find branch points (pixels with >2 neighbors)."""
    _, branch_points = _classify_skeleton_pixels(skeleton)
    return branch_points


def prune_skeleton(skeleton: np.ndarray, min_branch_length: int) -> np.ndarray:
    """Remove short branches from skeleton.

    Traces from each endpoint to the nearest branch point.
    If the path is shorter than min_branch_length, remove it.
    Repeats until stable.
    """
    skel = skeleton.copy()

    for _ in range(50):  # Max iterations to prevent infinite loops
        endpoints, branch_points = _classify_skeleton_pixels(skel)
        bp_set = set(branch_points)

        if not endpoints:
            break

        changed = False
        for ep in endpoints:
            # Trace from endpoint toward branch point
            path = _trace_to_branch(skel, ep, bp_set)
            if path is not None and len(path) < min_branch_length:
                # Remove this branch (but not the branch point itself)
                for r, c in path:
                    if (r, c) not in bp_set:
                        skel[r, c] = False
                changed = True

        if not changed:
            break

    return skel


def _trace_to_branch(skeleton: np.ndarray, start: Tuple[int, int],
                     branch_points: set) -> Optional[List[Tuple[int, int]]]:
    """Trace from endpoint to nearest branch point or other endpoint.

    Returns the path (including start, excluding the branch point),
    or None if the trace reaches another endpoint (= main segment, don't prune).
    """
    path = [start]
    visited = {start}
    current = start

    while True:
        neighbors = _get_neighbors(skeleton, current[0], current[1])
        unvisited = [n for n in neighbors if n not in visited]

        if not unvisited:
            # Dead end or isolated point
            return path

        if len(unvisited) == 1:
            nxt = unvisited[0]
            if nxt in branch_points:
                # Reached a branch point; this is a prunable branch
                return path
            path.append(nxt)
            visited.add(nxt)
            current = nxt
        else:
            # Multiple unvisited neighbors = we hit a branch point
            return path

    return path


def find_longest_path(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """Find the longest endpoint-to-endpoint path through the skeleton.

    Uses BFS from each endpoint to find the pair with maximum path length.
    """
    endpoints, _ = _classify_skeleton_pixels(skeleton)

    if len(endpoints) == 0:
        # No endpoints; skeleton might be a loop. Pick any skeleton pixel.
        coords = np.argwhere(skeleton)
        if len(coords) == 0:
            return []
        endpoints = [(coords[0][0], coords[0][1])]

    if len(endpoints) == 1:
        # Single endpoint; trace as far as possible
        return _bfs_farthest(skeleton, endpoints[0])

    # BFS from first endpoint to find farthest point
    far_path = _bfs_farthest(skeleton, endpoints[0])
    if not far_path:
        return []

    # BFS from that farthest point to find the true longest path
    farthest = far_path[-1]
    longest_path = _bfs_farthest(skeleton, farthest)

    return longest_path


def _bfs_farthest(skeleton: np.ndarray,
                  start: Tuple[int, int]) -> List[Tuple[int, int]]:
    """BFS from start to find the farthest reachable skeleton pixel.

    Returns the path from start to the farthest point.
    """
    queue = deque()
    queue.append(start)
    visited = {start: None}  # pixel -> parent

    farthest = start

    while queue:
        current = queue.popleft()
        farthest = current

        for neighbor in _get_neighbors(skeleton, current[0], current[1]):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)

    # Reconstruct path
    path = []
    node = farthest
    while node is not None:
        path.append(node)
        node = visited[node]
    path.reverse()

    return path


# ---------------------------------------------------------------------------
# Diameter measurement
# ---------------------------------------------------------------------------

def _compute_local_tangent(path: List[Tuple[int, int]], idx: int,
                           window: int = 5) -> np.ndarray:
    """Compute local tangent direction at a point along the path.

    Uses a window of neighboring path points to estimate direction.
    Returns unit vector [dr, dc].
    """
    n = len(path)
    half_w = window // 2

    start = max(0, idx - half_w)
    end = min(n - 1, idx + half_w)

    if start == end:
        # Fallback: use immediate neighbors
        if idx > 0:
            start = idx - 1
        if idx < n - 1:
            end = idx + 1

    p_start = np.array(path[start], dtype=float)
    p_end = np.array(path[end], dtype=float)

    tangent = p_end - p_start
    length = np.linalg.norm(tangent)

    if length < 1e-10:
        return np.array([0.0, 1.0])  # Default horizontal

    return tangent / length


def _cast_perpendicular_rays(mask: np.ndarray, r: int, c: int,
                             tangent: np.ndarray, pixel_size: float,
                             max_ray_length: int = 200) -> Tuple[float, Tuple]:
    """Cast rays perpendicular to tangent in both directions.

    Returns (diameter, ((r1,c1), (r2,c2))) where the two points are
    the boundary intersections.
    """
    h, w = mask.shape

    # Perpendicular direction (rotate tangent 90 degrees)
    perp = np.array([-tangent[1], tangent[0]])

    # Cast ray in positive perpendicular direction
    r1, c1, d1 = _cast_ray(mask, r, c, perp, max_ray_length)
    # Cast ray in negative perpendicular direction
    r2, c2, d2 = _cast_ray(mask, r, c, -perp, max_ray_length)

    diameter = (d1 + d2) * pixel_size
    line_pts = ((r1, c1), (r2, c2))

    return diameter, line_pts


def _cast_ray(mask: np.ndarray, r: int, c: int,
              direction: np.ndarray, max_length: int) -> Tuple[int, int, float]:
    """Cast a ray from (r,c) in given direction until leaving the mask.

    Returns (end_r, end_c, distance_in_pixels).
    """
    h, w = mask.shape
    prev_r, prev_c = r, c

    for step in range(1, max_length + 1):
        nr = int(round(r + direction[0] * step))
        nc = int(round(c + direction[1] * step))

        # Check bounds
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            dist = np.sqrt((prev_r - r) ** 2 + (prev_c - c) ** 2)
            return prev_r, prev_c, dist

        if not mask[nr, nc]:
            # Hit boundary
            dist = np.sqrt((prev_r - r) ** 2 + (prev_c - c) ** 2)
            return prev_r, prev_c, dist

        prev_r, prev_c = nr, nc

    dist = np.sqrt((prev_r - r) ** 2 + (prev_c - c) ** 2)
    return prev_r, prev_c, dist
