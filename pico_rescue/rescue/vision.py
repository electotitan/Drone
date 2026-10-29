"""
Vision for the rescue mission: ONE top-camera frame -> survivors + obstacles.

  red triangle    -> CRITICAL survivor
  yellow circle   -> STABLE survivor
  big black blob  -> OBSTACLE (a 3 m tall box seen from 15 m: the blob is its whole *silhouette*,
                     top face + visible sides, so it is shifted/enlarged outward by perspective)

numpy + scipy only. Nothing here knows where anything is -- it only sees pixels.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import mujoco
from scipy import ndimage

import lattice

CAM_HEIGHT = 15.0


@dataclass
class Survivor:
    cell: str
    x: float                 # measured world position (m)
    y: float
    phase: str               # 'critical' | 'stable'
    px: tuple


@dataclass
class Scene:
    frame: np.ndarray
    scale: float             # px per metre at floor level
    size: int
    survivors: List[Survivor]
    obstacle_labels: np.ndarray       # int image: 0 = none, k = obstacle blob k
    obstacle_info: list               # per blob: dict(area_m2, centroid_world, bbox_world)

    @property
    def critical(self):
        return [s for s in self.survivors if s.phase == "critical"]

    @property
    def stable(self):
        return [s for s in self.survivors if s.phase == "stable"]


class CameraModel:
    """Pinhole top-down camera from the MuJoCo camera definition (what /camera_info would give)."""
    def __init__(self, model, cam_name="top_cam", size=960):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            raise RuntimeError(f"camera '{cam_name}' not found")
        self.name, self.size = cam_name, size
        self.pos = model.cam_pos[cid].copy()
        self.height = float(self.pos[2])
        self.fovy = float(model.cam_fovy[cid])
        self.scale = (size / 2) / (self.height * np.tan(np.radians(self.fovy) / 2))   # px/m at the floor

    def scale_at(self, whycode_z):
        """px per metre on the horizontal plane that is `whycode_z` below the camera."""
        return self.scale * self.height / whycode_z

    def pixel_to_world(self, u, v, whycode_z=None):
        s = self.scale if whycode_z is None else self.scale_at(whycode_z)
        return (self.pos[0] + (u - self.size / 2) / s, self.pos[1] - (v - self.size / 2) / s)

    def world_to_pixel(self, x, y, whycode_z=None):
        s = self.scale if whycode_z is None else self.scale_at(whycode_z)
        return (self.size / 2 + (x - self.pos[0]) * s, self.size / 2 - (y - self.pos[1]) * s)


def render_frame(model, data, cam_name="top_cam", size=960):
    """Offscreen render of the camera (main thread only: OpenGL context)."""
    r = mujoco.Renderer(model, size, size)
    try:
        r.update_scene(data, camera=cam_name)
        return r.render().copy()
    finally:
        r.close()


def _components(mask, min_px):
    lab, n = ndimage.label(mask)
    keep = []
    for i in range(1, n + 1):
        sel = lab == i
        a = int(sel.sum())
        if a >= min_px:
            vs, us = np.nonzero(sel)
            keep.append((i, a, float(us.mean()), float(vs.mean())))
    return lab, keep


def detect_scene(frame, cam: CameraModel, home_xy=(0.0, 0.0), roi_margin=0.6) -> Scene:
    img = frame.astype(np.int16)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    red = (R > 200) & (G < 70) & (B < 70)
    yellow = (R > 200) & (G > 200) & (B < 90)
    black = (np.max(img, axis=2) < 45) & ((np.max(img, axis=2) - np.min(img, axis=2)) < 18)

    m2 = cam.scale ** 2                                        # px per m^2 at floor scale
    half = (lattice.N_COLS // 2) * lattice.CELL + roi_margin   # only the lattice area counts
    # (this also excludes the red 'eyantra' logo, the 4 corner markers and the white border)
    yy, xx = np.mgrid[0:cam.size, 0:cam.size]
    wx = cam.pos[0] + (xx - cam.size / 2) / cam.scale
    wy = cam.pos[1] - (yy - cam.size / 2) / cam.scale
    roi = (np.abs(wx) <= half) & (np.abs(wy) <= half)

    survivors = []
    for mask, phase, min_m2 in ((red, "critical", 0.15), (yellow, "stable", 0.3)):
        _, blobs = _components(mask & roi, int(min_m2 * m2))
        for _, a, u, v in blobs:
            x, y = cam.pixel_to_world(u, v)
            col, row, _ = lattice.world_to_idx(x, y)
            survivors.append(Survivor(lattice.label(col, row), x, y, phase, (u, v)))

    # Obstacles: big black blobs. The drone itself (dark body, small, at home) is far smaller.
    lab, blobs = _components(black & roi, int(0.6 * m2))
    obst_lab = np.zeros_like(lab)
    info = []
    for k, (i, a, u, v) in enumerate(blobs, 1):
        obst_lab[lab == i] = k
        vs, us = np.nonzero(lab == i)
        x0, y0 = cam.pixel_to_world(us.min(), vs.max())
        x1, y1 = cam.pixel_to_world(us.max(), vs.min())
        cx, cy = cam.pixel_to_world(u, v)
        info.append(dict(id=k, area_m2=a / m2, centroid=(cx, cy), bbox=(x0, y0, x1, y1)))
    return Scene(frame=frame, scale=cam.scale, size=cam.size, survivors=survivors,
                 obstacle_labels=obst_lab, obstacle_info=info)
