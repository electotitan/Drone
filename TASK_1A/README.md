# Task 1A

Image processing pipeline for the eYRC 26–27 Khojo-Drone arena. Takes a single
photo of the arena, locates it via ArUco corner markers, rectifies it to a
top-down view, builds the grid coordinate system, and reports the location of
every red ("Critical") and yellow ("Stable") survivor.

## Requirements

- Python 3
- OpenCV with the `aruco` module (`pip install opencv-contrib-python`)
- NumPy

Check your OpenCV version before debugging anything else — the ArUco API
changed across versions and is the single most common source of confusion:

```bash
python3 -c "import cv2; print(cv2.__version__)"
```

This script targets the modern `cv2.aruco.ArucoDetector` API (OpenCV ≥ 4.7).

## Usage

```bash
python3 task1a.py --image image_1.jpg
```

| Flag | Default | Meaning |
|---|---|---|
| `--image` | *(required)* | Path to the arena photo |
| `--debug-dir` | `<image_folder>/<image_name>_debug` | Where checkpoint images are saved |
| `--no-debug` | off | Skip saving checkpoint images |

## Output

Writes `<image_name>_results.txt` next to the input image, e.g.
`image_1.jpg` → `image_1_results.txt`:

```
Detected marker IDs: [80, 85, 90, 95]

Critical Survivors: C10, B7, I2
Stable Survivors: E9, H6, D2
```

- Line 1: detected marker IDs, as a Python-style list.
- Line 2: blank.
- Line 3: red survivors ("Critical"), comma-separated, `A1`–`K11` naming.
- Line 4: yellow survivors ("Stable"), same naming.
- If a category is empty the line is still written, with nothing after the colon.

## Pipeline

1. **Corner markers** — detect the four ArUco (4×4_250) markers, IDs
   `80, 85, 90, 95`. Exits if any are missing.
2. **Rectify** — warp the arena to a top-down 900×900 view. The mapping is
   rotation-independent: for each marker, the corner closest to the overall
   centroid of all four marker centers is the one touching the playing
   field, and each marker's centroid position relative to that same
   centroid decides which arena corner it belongs to — so it doesn't matter
   which marker ID physically sits in which corner.
3. **Grid** — the rectified canvas is an exact 900×900 square with 12 equal
   cells by construction, so the 11×11 interior intersections are computed
   directly (multiples of 75 px) rather than detected from the painted lines.
4. **Names** — intersections are named `<column A–K><row 1–11>`, top-left to
   bottom-right.
5. **Survivors** — red and yellow regions are isolated in HSV, each
   contour above a minimum area is kept.
6. **Centroids** — each surviving region is reduced to one point (image
   moments, with a bounding-box fallback for degenerate zero-area regions).
7. **Matching** — each centroid is assigned to its nearest grid
   intersection, clamped to `A1`–`K11` so an edge-hugging survivor can't
   crash the script or produce a nonsense label.

## Debug / checkpoint images

Unless `--no-debug` is passed, each step's output is saved to the debug
folder so you can eyeball the pipeline as you go:

| File | Shows |
|---|---|
| `step1_markers.png` | Detected ArUco markers, boxed and ID-labelled |
| `step2_rectified.png` | The straightened 900×900 arena |
| `step3_grid.png` | Computed grid lines over the rectified arena |
| `step4_labels.png` | All 121 intersection names |
| `step5_outlines.png` | Red/yellow region outlines |
| `step6_centers.png` | Reduced survivor centroids |
| `composite.png` | Grid + outlines + centroids + assigned names, all in one — the fastest way to see why an answer is wrong |

## Tuning

If survivors are missed or over-detected on a different photo (different
lighting, camera, or print), adjust these constants near the top of
`task1a.py`:

- `RED_HSV_RANGES` / `YELLOW_HSV_RANGE` — HSV thresholds for each color
- `MIN_SURVIVOR_AREA` — minimum contour area (in px², on the 900×900
  rectified image) to count as a real survivor rather than noise
