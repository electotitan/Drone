"""
Arena scanner: takes a frame from the arena's top camera and finds the target markers.

A TARGET MARKER = a red triangle with a yellow circle on top of it (as in the reference
picture). Lone triangles / lone circles are reported but not treated as targets.

Pipeline (numpy + scipy only, no OpenCV):
  1. render the camera frame (MuJoCo offscreen renderer, camera 'top_cam')
  2. colour-segment red and yellow pixels
  3. connected components -> blobs, centroids (pixel)
  4. pixel -> world using the camera's own pose / field of view read from the model
  5. snap to the nearest lattice intersection and name the cell (e.g. 'D5')

Nothing here knows where the markers are: it only sees pixels.
"""
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import mujoco
from scipy import ndimage

import lattice


@dataclass
class Detection:
    cell: str
    x: float                # snapped lattice coordinate (m)
    y: float
    meas_x: float           # raw measurement from the image (m)
    meas_y: float
    snap_err: float         # |measured - snapped| (m)
    kind: str               # 'target' | 'triangle-only' | 'circle-only'
    px: tuple               # pixel centroid (u, v)


@dataclass
class ScanResult:
    frame: np.ndarray
    scale: float            # pixels per metre at floor level
    detections: List[Detection]

    @property
    def targets(self):
        return [d for d in self.detections if d.kind == "target"]


class CameraModel:
    """Pinhole top-down camera built from the MuJoCo camera definition."""
    def __init__(self, model, cam_name="top_cam", size=960):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            raise RuntimeError(f"camera '{cam_name}' not found in the model")
        self.name, self.size = cam_name, size
        self.cam_pos = model.cam_pos[cid].copy()
        self.fovy = float(model.cam_fovy[cid])
        height = self.cam_pos[2] - 0.0                              # camera height above the floor
        self.scale = (size / 2) / (height * np.tan(np.radians(self.fovy) / 2))   # px per metre at floor level

    def pixel_to_world(self, u, v):
        # camera looks straight down, image right = +x, image up = +y (xyaxes "1 0 0 0 1 0")
        return (self.cam_pos[0] + (u - self.size / 2) / self.scale,
                self.cam_pos[1] - (v - self.size / 2) / self.scale)

    def world_to_pixel(self, x, y):
        return (self.size / 2 + (x - self.cam_pos[0]) * self.scale,
                self.size / 2 - (y - self.cam_pos[1]) * self.scale)


def render_frame(model, data, cam_name="top_cam", size=960):
    """Offscreen render. Must be called from the main thread (OpenGL context)."""
    renderer = mujoco.Renderer(model, size, size)
    try:
        renderer.update_scene(data, camera=cam_name)
        return renderer.render().copy()
    finally:
        renderer.close()


def _blobs(mask, min_px):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        area = int(sel.sum())
        if area < min_px:
            continue
        vs, us = np.nonzero(sel)
        out.append(dict(area=area, u=float(us.mean()), v=float(vs.mean())))
    return out


def detect_markers(frame, cam: CameraModel, roi_margin=0.4, max_snap_err=0.30) -> ScanResult:
    img = frame.astype(np.int16)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    red = (R > 200) & (G < 70) & (B < 70)
    yellow = (R > 200) & (G > 200) & (B < 90)

    # minimum blob sizes in physical units: triangle ~0.43 m^2, circle ~0.20 m^2
    px_per_m2 = cam.scale ** 2
    tri_blobs = _blobs(red, int(0.15 * px_per_m2))
    cir_blobs = _blobs(yellow, int(0.05 * px_per_m2))

    half = (lattice.N_COLS // 2) * lattice.CELL + roi_margin       # only the lattice area counts
    def in_roi(b):
        x, y = cam.pixel_to_world(b["u"], b["v"])
        return abs(x) <= half and abs(y) <= half                   # (excludes the red 'eyantra' logo)
    tri_blobs = [b for b in tri_blobs if in_roi(b)]
    cir_blobs = [b for b in cir_blobs if in_roi(b)]

    detections, used_tri = [], set()
    for c in cir_blobs:
        cx, cy = cam.pixel_to_world(c["u"], c["v"])
        match = None
        for k, t in enumerate(tri_blobs):
            tx, ty = cam.pixel_to_world(t["u"], t["v"])
            if k not in used_tri and np.hypot(tx - cx, ty - cy) < 0.35:    # circle sits on the triangle centroid
                match = k
                break
        kind = "target" if match is not None else "circle-only"
        if match is not None:
            used_tri.add(match)
        detections.append(_mk(c, cx, cy, kind, max_snap_err))
    for k, t in enumerate(tri_blobs):
        if k not in used_tri:
            tx, ty = cam.pixel_to_world(t["u"], t["v"])
            detections.append(_mk(t, tx, ty, "triangle-only", max_snap_err))
    return ScanResult(frame=frame, scale=cam.scale, detections=detections)


def _mk(blob, x, y, kind, max_snap_err):
    col, row, err = lattice.world_to_idx(x, y)
    if err > max_snap_err or not lattice.inside(col, row):
        print(f"[scan] warning: blob at ({x:+.2f}, {y:+.2f}) is {err:.2f} m from the nearest lattice point")
    sx, sy = lattice.idx_to_world(col, row)
    return Detection(cell=lattice.label(col, row), x=sx, y=sy, meas_x=x, meas_y=y,
                     snap_err=err, kind=kind, px=(blob["u"], blob["v"]))


def scan_arena(model, data, cam_name="top_cam", size=960) -> ScanResult:
    cam = CameraModel(model, cam_name, size)
    frame = render_frame(model, data, cam_name, size)
    return detect_markers(frame, cam)
