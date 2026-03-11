# Myotube Diameter Measurement Tool -- Implementation Plan

## Context

C2C12 myotube diameter is a key metric for assessing skeletal muscle differentiation in vitro, but most existing tools (MuscleJ, SMASH, MyoCount) target gradient-contrast or cross-section histology images. This project needs a Python tool purpose-built for **fluorescence images** where myotubes are stained with MF-20 (pan-myosin heavy chain) and AF488 secondary at **10X magnification**. The fluorescence signal simplifies detection vs. brightfield, but introduces challenges: variable MHC distribution, heterogeneous intensity from PFA fixation, and overlapping/touching myotubes in culture.

**Literature-informed approach:** The TRUEFAD method (Nature Sci Rep, 2024) validated skeleton-based measurement with 9 equidistant perpendicular diameter samples per myotube at 97.4% accuracy. Cellpose (Nature Methods, 2021) provides robust instance segmentation that handles intensity heterogeneity (Dice ~0.9). This plan combines both into a Python pipeline with a classical fallback for environments without GPU/Cellpose.

---

## File Structure

```
myotube/
├── measure_myotubes.py          # CLI entry point
├── config.yaml                  # Default parameters
├── requirements.txt             # Core deps (no cellpose)
├── requirements-cellpose.txt    # Optional cellpose + torch
├── myotube/
│   ├── __init__.py
│   ├── config.py                # Config dataclass, YAML + CLI loading
│   ├── io.py                    # Image I/O, batch discovery, scale calibration
│   ├── preprocessing.py         # Normalize, denoise, CLAHE
│   ├── segmentation.py          # Cellpose + classical (adaptive thresh + watershed)
│   ├── filtering.py             # Remove non-myotubes by shape/size
│   ├── measurement.py           # Skeleton + distance transform diameters
│   ├── visualization.py         # QC overlay generation
│   └── pipeline.py              # Orchestrates full processing
└── tests/
    ├── generate_synthetic.py    # Creates test images with known ground truth
    ├── test_measurement.py      # Unit tests for diameter accuracy
    └── test_filtering.py        # Unit tests for object filtering
```

---

## Implementation Steps (in order)

### Step 1: `config.py` -- Configuration Management

Create a `@dataclass Config` with all tunable parameters and a `load_config()` function that reads YAML defaults, applies CLI overrides, and validates.

**Key defaults for 10X MF-20/AF488:**

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `gaussian_sigma` | 2.0 | ~1.3 um at 10X; smooths noise, preserves 10+ um edges |
| `clahe_clip_limit` | 3.0 | Standard; avoids amplifying background noise |
| `clahe_tile_size` | 64 | ~42 um tiles; captures local illumination variation |
| `adaptive_block_size` | 127 | ~82 um neighborhood for local thresholding |
| `adaptive_c` | -5.0 | Slightly below local mean; includes dim myotube edges |
| `morph_open_size` | 5 | ~3 um disk; removes speckle noise |
| `morph_close_size` | 15 | ~10 um disk; fills gaps from sarcomeric MF-20 pattern |
| `min_object_area` | 2000 px | ~845 um2; excludes myoblasts (~200 um2) |
| `min_aspect_ratio` | 3.0 | Myotubes typically 5-20+; myoblasts ~1-2 |
| `min_length_um` | 50.0 | Shortest plausible myotube |
| `max_circularity` | 0.5 | Excludes round blobs |
| `cellpose_model` | "cyto2" | Best for elongated structures |
| `cellpose_diameter` | 80 | ~52 um; approximate median myotube width at 10X |
| `num_diameter_samples` | 9 | TRUEFAD-validated sampling density |
| `skeleton_prune_length` | 20 px | ~13 um; removes noise branches |

---

### Step 2: `io.py` -- Image I/O and Scale Calibration

- `discover_images()`: Find .tif/.tiff/.png/.jpg in directory, natural sort
- `load_image()`: Load via tifffile (16-bit TIFF support) or OpenCV, convert to float64 [0,1]
- `extract_green_channel()`: If RGB/multi-channel, extract channel 1 (green/AF488). If grayscale, pass through
- `parse_pixel_size()`: Try TIFF metadata (ImageJ tags, OME-XML) -> config value -> warn and use pixels
- `save_csv()`: Write pandas DataFrame to CSV

---

### Step 3: `tests/generate_synthetic.py` -- Test Fixtures

Create synthetic images with **known ground-truth diameters** for validation:

1. **Single straight tube** (rectangle, width=30 px) -- expect diameter ~30
2. **Single curved tube** (arc, constant width=25 px) -- tests curvature handling
3. **Two parallel touching tubes** -- tests watershed separation
4. **Y-shaped branching tube** -- tests branch pruning
5. **Mixed scene** (tubes + round blobs + dots) -- tests filtering
6. **Variable intensity tube** (gradient along length) -- tests preprocessing robustness

---

### Step 4: `preprocessing.py` -- Image Preprocessing

Pipeline: `normalize_intensity()` -> `denoise()` -> `enhance_contrast()`

1. **Percentile normalization** (0.5th-99.5th) -- robust to hot pixels/outliers
2. **Gaussian blur** (sigma=2.0) -- suppress shot noise
3. **CLAHE** (clip=3.0, tile=64) -- normalize uneven MF-20 staining intensity across FOV

---

### Step 5: `segmentation.py` -- Two-Tier Segmentation

**Cellpose path (preferred):**
- Import cellpose conditionally (try/except ImportError)
- Use `cyto2` model with `diameter=80`, `flow_threshold=0.4`
- Returns instance-labeled mask directly (handles overlaps natively)

**Classical path (fallback):**
1. Adaptive Gaussian threshold on CLAHE-enhanced image -> binary mask
2. Morphological opening (disk r=5) -> remove small noise
3. Morphological closing (disk r=15) -> fill internal holes from sarcomeric MF-20 pattern
4. `remove_small_objects(min_size=500)` -> eliminate tiny debris
5. Distance transform of binary mask
6. Find local maxima of distance transform (= tube centers)
7. **Marker-controlled watershed** using maxima as seeds -> split touching/overlapping myotubes
8. Return labeled instance mask

**Method selection:** `"auto"` tries Cellpose first, falls back to classical if import fails.

---

### Step 6: `filtering.py` -- Object Filtering

Use `skimage.measure.regionprops` to compute properties, then apply filters:

1. **Area** >= `min_object_area` (2000 px) -- excludes debris and individual myoblasts
2. **Aspect ratio** (major/minor axis) >= 3.0 -- excludes round cells/clusters
3. **Length** (major axis) >= `min_length` (~77 px at 10X) -- excludes short fragments
4. **Circularity** (4*pi*area/perimeter^2) <= 0.5 -- redundant round-object check
5. **Solidity** (area/convex_area) >= 0.5 -- excludes very fragmented shapes

Multiple complementary filters ensure robustness: a myoblast cluster may have large area but will be round (caught by AR); bright debris may be elongated but tiny (caught by area).

---

### Step 7: `measurement.py` -- Core Diameter Logic (TRUEFAD-inspired)

This is the most critical module. For **each labeled myotube**:

#### 7a. Skeletonize
- `skimage.morphology.skeletonize()` (Zhang-Suen thinning) on single-myotube binary mask

#### 7b. Prune skeleton branches
- Find branch points (pixels with >2 neighbors in 8-connectivity)
- Find endpoints (pixels with exactly 1 neighbor)
- Trace from each endpoint to nearest branch point
- Remove branches shorter than `skeleton_prune_length` (20 px)
- Repeat until stable

#### 7c. Find longest path (= medial axis)
- BFS/DFS from each endpoint to find the longest endpoint-to-endpoint path through the skeleton
- This is the main axis of the myotube

#### 7d. Measure diameters at 9 equidistant points
At each sample point along the skeleton path:

1. **Distance transform method (fast):** `diameter_DT = 2 * distance_transform_edt(mask)[point]`
2. **Perpendicular ray-casting (accurate):**
   - Compute local tangent from ~5 neighboring skeleton points
   - Cast rays perpendicular to tangent in both directions until hitting mask boundary
   - `diameter_ray = left_distance + right_distance`
3. Report `diameter_ray` as primary; use `diameter_DT` as sanity check (should agree within ~20%)

**Why both methods:** Distance transform gives inscribed-circle radius (underestimates when skeleton is off-center, e.g., near branch points). Ray-casting gives true cross-sectional width at the perpendicular orientation. The extra code complexity is justified by measurement accuracy.

**Why NOT `regionprops.minor_axis_length`:** This metric uses second moments of the entire shape. For curved myotubes, the "minor axis" of the curve is much larger than the actual tube width. Skeleton-based measurement is the only correct approach for curved/non-linear structures.

#### 7e. Compute summary metrics
- `mean_diameter`, `median_diameter`, `min_diameter`, `max_diameter`, `std_diameter`
- `length` (sum of Euclidean distances along skeleton path)
- `area`, `aspect_ratio`, `n_branches`

#### Output columns:
`image`, `label_id`, `mean_diameter`, `median_diameter`, `min_diameter`, `max_diameter`, `std_diameter`, `length`, `area`, `aspect_ratio`, `n_branches`, `unit`

---

### Step 8: `visualization.py` -- QC Overlays

Generate per-image QC overlay:
- Original image as green-tinted grayscale background
- **Colored contours** around each segmented myotube (random distinct colors)
- **Skeleton** drawn in red
- **Perpendicular measurement lines** in yellow at 9 sample points
- **Mean diameter label** next to each myotube
- Save as PNG

Also generate `create_summary_figure()`: histogram of mean diameters + box plot + summary stats.

---

### Step 9: `pipeline.py` -- Orchestration

```
process_single_image(path, config) -> DataFrame:
    1. load_image + extract_green_channel
    2. parse_pixel_size
    3. preprocess (normalize, denoise, CLAHE)
    4. segment (cellpose or classical)
    5. filter_myotubes
    6. measure_all_myotubes
    7. create_overlay
    8. return measurements

process_batch(input_path, config) -> DataFrame:
    - discover_images()
    - process each with try/except (log errors, continue)
    - concatenate all results
    - save combined CSV + per-image overlays + summary figure
```

---

### Step 10: `measure_myotubes.py` -- CLI Entry Point

```
python measure_myotubes.py path/to/images/
python measure_myotubes.py path/to/images/ --config my_config.yaml
python measure_myotubes.py path/to/images/ --method classical --pixel-size 0.65
python measure_myotubes.py single_image.tif --no-overlay
```

Key CLI args: `input_path`, `--config`, `--method`, `--pixel-size`, `--output-dir`, `--min-area`, `--min-aspect-ratio`, `--no-overlay`, `--verbose`

---

### Step 11: `config.yaml`, `requirements.txt`, `requirements-cellpose.txt`

**requirements.txt** (core, no GPU needed):
```
numpy
scipy
scikit-image
opencv-python-headless
pandas
matplotlib
tifffile
pyyaml
```

**requirements-cellpose.txt** (optional):
```
cellpose>=3.0
```

---

## Edge Case Handling Summary

| Edge Case | Solution |
|-----------|----------|
| **Overlapping/touching myotubes** | Cellpose instance segmentation (preferred); marker-controlled watershed (classical) |
| **Branching myotubes** | Skeleton branch pruning removes short noise branches; longest-path selection follows main axis; `n_branches` reported for flagging |
| **Curved myotubes** | Skeleton-based measurement naturally follows curves; local tangent computation at each sample point |
| **Clustered myoblasts** | Multi-filter exclusion: area + aspect ratio + circularity + length thresholds |
| **Variable fluorescence intensity** | CLAHE preprocessing normalizes local contrast; Cellpose is intensity-agnostic |
| **Sarcomeric MF-20 gaps** | Morphological closing (15 px disk) fills internal holes |
| **Image edge truncation** | Perpendicular rays clamped to image boundary; flagged as "edge-truncated" |
| **No myotubes detected** | Log warning, write empty row to CSV, continue batch |
| **All objects filtered out** | Log warning with suggestion to relax parameters |
| **Cellpose unavailable** | Automatic fallback to classical pipeline |
| **Missing pixel calibration** | Warn user, report in pixels, set `unit: pixels` |

---

## Error Handling Strategy

- **Per-image error isolation:** Each image in try/except; log error + traceback, continue batch
- **Logging:** Python `logging` module; INFO default, DEBUG with `--verbose`; log to console + `processing.log` in output dir
- **Graceful degradation:** Cellpose -> classical fallback; metadata -> config -> pixels fallback

---

## Verification Plan

1. **Unit tests on synthetic images** with known ground-truth diameters (tolerance: within 2 px)
   - Straight tube (30 px wide) -> expect mean_diameter ~30
   - Curved tube (25 px wide) -> expect mean_diameter ~25
   - Verify watershed separates two touching tubes into 2 labels
   - Verify filtering removes round blobs but keeps elongated tubes

2. **Visual QC** on real images: inspect overlays to verify skeletons follow tube centers and perpendicular lines look correct

3. **Cross-validation** against manual measurements or existing tools (Myotube Analyzer) on same images

4. **Batch test**: run on full image directory, verify CSV output is complete and summary statistics are reasonable

---

## Dependencies

- **No GPU required** for classical pipeline
- **Optional GPU** for Cellpose (falls back to CPU automatically)
- All core dependencies are standard scientific Python packages
- No dependency on `skan` (skeleton analysis implemented manually using scipy/scikit-image primitives to minimize dependencies)
