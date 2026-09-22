#!/usr/bin/env python3
"""
Task 1A
Image processing pipeline: locate the arena via ArUco markers, rectify it to
a top-down 900x900 view, build the 12x12 grid coordinate system, find the
red ("Critical") and yellow ("Stable") survivors, and report the grid
intersection nearest each survivor to <image>_results.txt.

Run:
    python3 task1a.py --image image_1.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

REQUIRED_MARKER_IDS = {80, 85, 90, 95}
ARENA_SIZE = 900          # rectified canvas is ARENA_SIZE x ARENA_SIZE
GRID_CELLS = 12           # 12 x 12 cells -> 11 interior lines per axis
COLUMN_LETTERS = "ABCDEFGHIJK"   # 11 letters, A..K

# HSV colour ranges for the two survivor colours. Kept generous (wide S/V
# floor) since lighting on the arena photo can vary; red wraps around hue 0.
RED_HSV_RANGES = [
    ((0, 100, 60), (10, 255, 255)),
    ((170, 100, 60), (180, 255, 255)),
]
YELLOW_HSV_RANGE = ((18, 100, 60), (35, 255, 255))

MIN_SURVIVOR_AREA = 120  # px^2 in the rectified image; filters out noise specks


# ----------------------------------------------------------------------------
# Step 1 - find the corner markers
# ----------------------------------------------------------------------------

def detect_markers(gray_image):
    """Detect the ArUco 4x4_250 markers and return {id: 4x2 corner array}.

    Exits the program if any of the four required IDs (80, 85, 90, 95) is
    not found, per the task's "you cannot continue without all four" rule.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _rejected = detector.detectMarkers(gray_image)

    if ids is None:
        print("ERROR: no ArUco markers detected at all. Cannot continue.")
        sys.exit(1)

    ids = ids.flatten()
    markers = {int(mid): corners[i][0] for i, mid in enumerate(ids)}

    print(f"Detected marker IDs: {sorted(markers.keys())}")

    missing = REQUIRED_MARKER_IDS - set(markers.keys())
    if missing:
        print(f"ERROR: missing required marker ID(s) {sorted(missing)}. "
              f"Cannot continue without all four corner markers.")
        sys.exit(1)

    return markers


def draw_marker_debug(image, markers):
    """Checkpoint image for Step 1: detected markers boxed and labelled."""
    vis = image.copy()
    for mid, corners in markers.items():
        pts = corners.astype(int)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        center = pts.mean(axis=0).astype(int)
        cv2.putText(vis, str(mid), tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


# ----------------------------------------------------------------------------
# Step 2 - straighten the arena (perspective transform)
# ----------------------------------------------------------------------------

def rectify_arena(image, markers, size=ARENA_SIZE):
    """Warp the arena to a top-down size x size view of the playing field.

    Only the four required markers are used. For each marker we take the
    single corner point closest to the overall centre of all four marker
    centres - that is always the corner touching the playing field, no
    matter how the arena/markers happen to be rotated in the photo. Each
    marker is then bucketed into a quadrant (top-left / top-right /
    bottom-right / bottom-left) purely by its position relative to that
    same centre, so the mapping is correct regardless of which marker ID
    physically sits in which corner.
    """
    used = {mid: markers[mid] for mid in REQUIRED_MARKER_IDS}
    centers = {mid: c.mean(axis=0) for mid, c in used.items()}
    overall_center = np.mean(list(centers.values()), axis=0)

    quadrant_pts = {}
    for mid, corners in used.items():
        cen = centers[mid]
        quadrant = ("L" if cen[0] < overall_center[0] else "R") + \
                   ("T" if cen[1] < overall_center[1] else "B")
        dists = np.linalg.norm(corners - overall_center, axis=1)
        inner_corner = corners[np.argmin(dists)]
        quadrant_pts[quadrant] = inner_corner

    if set(quadrant_pts.keys()) != {"LT", "RT", "RB", "LB"}:
        print("ERROR: could not resolve one marker per arena corner "
              "(two markers landed in the same quadrant). Cannot rectify.")
        sys.exit(1)

    src = np.array([quadrant_pts["LT"], quadrant_pts["RT"],
                     quadrant_pts["RB"], quadrant_pts["LB"]], dtype=np.float32)
    dst = np.array([[0, 0], [size - 1, 0],
                     [size - 1, size - 1], [0, size - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(image, M, (size, size))
    return rectified


# ----------------------------------------------------------------------------
# Step 3 & 4 - grid intersections and their names
# ----------------------------------------------------------------------------

def build_grid(size=ARENA_SIZE, cells=GRID_CELLS):
    """Return {name: (x, y)} for all 121 interior grid intersections.

    The rectified image is an exact size x size square with `cells` equal
    cells by construction, so the interior lines fall at exact multiples of
    the cell size - no line detection required.
    """
    cell = size / cells
    grid = {}
    for row in range(1, cells):          # 1..11
        for col in range(1, cells):      # 1..11
            name = f"{COLUMN_LETTERS[col - 1]}{row}"
            x = round(col * cell)
            y = round(row * cell)
            grid[name] = (x, y)
    return grid


def draw_grid_debug(image, size=ARENA_SIZE, cells=GRID_CELLS):
    vis = image.copy()
    cell = size / cells
    for i in range(1, cells):
        pos = round(i * cell)
        cv2.line(vis, (pos, 0), (pos, size), (0, 255, 0), 1)
        cv2.line(vis, (0, pos), (size, pos), (0, 255, 0), 1)
    return vis


def draw_labels_debug(image, grid):
    vis = image.copy()
    for name, (x, y) in grid.items():
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(vis, name, (x + 4, y - 4), cv2.FONT_HERSHEY_PLAIN,
                    0.9, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


# ----------------------------------------------------------------------------
# Step 5 & 6 - find the survivors and reduce each to one point
# ----------------------------------------------------------------------------

def _mask_for_ranges(hsv, ranges):
    mask = None
    for lower, upper in ranges:
        part = cv2.inRange(hsv, lower, upper)
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_survivor_centroids(rectified_bgr, hsv_ranges):
    """Isolate one colour, return (centroids, contours) for each region
    large enough to be a real survivor rather than noise."""
    hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    mask = _mask_for_ranges(hsv, hsv_ranges)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    kept_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_SURVIVOR_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            # Degenerate (zero-area) region: fall back to the bounding
            # box centre so we never crash or drop a real detection.
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w / 2.0, y + h / 2.0
        centroids.append((cx, cy))
        kept_contours.append(cnt)

    return centroids, kept_contours


# ----------------------------------------------------------------------------
# Step 7 - match each centre to its nearest intersection
# ----------------------------------------------------------------------------

def nearest_intersection_name(point, size=ARENA_SIZE, cells=GRID_CELLS):
    """Nearest of the 121 interior intersections to `point`, clamped to the
    valid A1..K11 range so an off-grid centre never produces a crash or a
    nonsense label."""
    cell = size / cells
    cx, cy = point

    col = int(round(cx / cell))
    row = int(round(cy / cell))
    col = min(max(col, 1), cells - 1)
    row = min(max(row, 1), cells - 1)

    return f"{COLUMN_LETTERS[col - 1]}{row}"


# ----------------------------------------------------------------------------
# Results file
# ----------------------------------------------------------------------------

def write_results(output_path, marker_ids, critical_names, stable_names):
    lines = [
        f"Detected marker IDs: {sorted(marker_ids)}",
        "",
        f"Critical Survivors: {', '.join(critical_names)}",
        f"Stable Survivors: {', '.join(stable_names)}",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Khojo Drone Task 1A pipeline")
    parser.add_argument("--image", required=True, help="path to the arena photo")
    parser.add_argument("--debug-dir", default=None,
                         help="where to save checkpoint images "
                              "(default: <image_folder>/<image_name>_debug)")
    parser.add_argument("--no-debug", action="store_true",
                         help="skip saving checkpoint images")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.isfile(image_path):
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    folder = os.path.dirname(os.path.abspath(image_path))
    base = os.path.splitext(os.path.basename(image_path))[0]
    results_path = os.path.join(folder, f"{base}_results.txt")

    debug_dir = args.debug_dir or os.path.join(folder, f"{base}_debug")
    save_debug = not args.no_debug
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: could not read image: {image_path}")
        sys.exit(1)

    # --- Step 1: corner markers -------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    markers = detect_markers(gray)
    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "step1_markers.png"),
                    draw_marker_debug(image, markers))

    # --- Step 2: rectify ----------------------------------------------------
    rectified = rectify_arena(image, markers)
    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "step2_rectified.png"), rectified)

    # --- Step 3 & 4: grid + names -------------------------------------------
    grid = build_grid()
    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "step3_grid.png"),
                    draw_grid_debug(rectified))
        cv2.imwrite(os.path.join(debug_dir, "step4_labels.png"),
                    draw_labels_debug(rectified, grid))

    # --- Step 5 & 6: survivors + centroids ----------------------------------
    red_centroids, red_contours = find_survivor_centroids(rectified, RED_HSV_RANGES)
    yellow_centroids, yellow_contours = find_survivor_centroids(rectified, [YELLOW_HSV_RANGE])

    print(f"Red (Critical) regions found: {len(red_centroids)}")
    print(f"Yellow (Stable) regions found: {len(yellow_centroids)}")

    if save_debug:
        outline_vis = rectified.copy()
        cv2.drawContours(outline_vis, red_contours, -1, (255, 0, 255), 2)
        cv2.drawContours(outline_vis, yellow_contours, -1, (255, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "step5_outlines.png"), outline_vis)

        centers_vis = rectified.copy()
        for cx, cy in red_centroids + yellow_centroids:
            cv2.circle(centers_vis, (int(round(cx)), int(round(cy))), 4,
                        (0, 0, 0), -1)
            cv2.circle(centers_vis, (int(round(cx)), int(round(cy))), 4,
                        (255, 255, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, "step6_centers.png"), centers_vis)

    # --- Step 7: nearest intersection ---------------------------------------
    critical_names = [nearest_intersection_name(p) for p in red_centroids]
    stable_names = [nearest_intersection_name(p) for p in yellow_centroids]

    print(f"Critical Survivors: {', '.join(critical_names)}")
    print(f"Stable Survivors: {', '.join(stable_names)}")

    if save_debug:
        composite = draw_grid_debug(rectified)
        cv2.drawContours(composite, red_contours, -1, (255, 0, 255), 2)
        cv2.drawContours(composite, yellow_contours, -1, (255, 255, 0), 2)
        for (cx, cy), name in zip(red_centroids + yellow_centroids,
                                   critical_names + stable_names):
            pt = (int(round(cx)), int(round(cy)))
            cv2.circle(composite, pt, 4, (255, 255, 255), -1)
            cv2.putText(composite, name, (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
        cv2.imwrite(os.path.join(debug_dir, "composite.png"), composite)
        print(f"Checkpoint images written to: {debug_dir}")

    # --- Results file --------------------------------------------------------
    write_results(results_path, markers.keys(), critical_names, stable_names)
    print(f"Results written to: {results_path}")


if __name__ == "__main__":
    main()
