"""
Obstacle-aware route planning on the flight plane.

Obstacle model
    The camera sees each 3 m box as a *silhouette* (top face + sides) that is shifted outward by
    perspective. A point of the box at height h shows up at pixel = x * H/(H-h) (H = 15 m). We read
    the black silhouette pixels and map them onto the horizontal plane the drone flies in
    (WhyCode Z = 13, i.e. 2 m above the floor) with that plane's own pixel scale. The true cross-section
    of the box at the flight altitude maps exactly onto itself, so the silhouette mapped this way is a
    SUPERSET of it: always safe, and it also covers the region where the box would hide the drone's
    marker from the camera.
    That region is then inflated by `margin` = drone radius (0.33 m) + tracking slack.

Planning: A* (8-connected) on a 0.1 m grid over the inflated map, then line-of-sight shortcutting.
"""
import heapq
import itertools
import math

import numpy as np
from scipy import ndimage

RES = 0.1


class Planner:
    def __init__(self, scene, cam, plane_z=13.0, margin=0.85, relaxed_margin=0.50, bound=6.3):
        self.cam, self.plane_z, self.margin, self.relaxed_margin, self.bound = cam, plane_z, margin, relaxed_margin, bound
        n = int(round(2 * bound / RES)) + 1
        self.n = n
        xs = np.linspace(-bound, bound, n)
        X, Y = np.meshgrid(xs, xs)                              # [iy, ix]
        s = cam.scale_at(plane_z)
        u = np.clip(np.round(cam.size / 2 + X * s).astype(int), 0, cam.size - 1)
        v = np.clip(np.round(cam.size / 2 - Y * s).astype(int), 0, cam.size - 1)
        mask = scene.obstacle_labels > 0
        mask = ndimage.binary_closing(mask, iterations=2)
        blocked = mask[v, u]
        self.blocked = blocked
        d = ndimage.distance_transform_edt(~blocked) * RES
        edge = np.minimum.reduce([X + bound, bound - X, Y + bound, bound - Y])    # keep inside the arena
        self.clear = np.minimum(d, 10.0)
        self.free = (d >= margin) & (edge >= 0)
        self.free_relaxed = (d >= relaxed_margin) & (edge >= 0)
        self._cache = {}

    # ---- grid helpers -------------------------------------------------------------
    def _idx(self, p):
        return (int(round((p[1] + self.bound) / RES)), int(round((p[0] + self.bound) / RES)))   # (iy, ix)

    def _xy(self, i):
        return (i[1] * RES - self.bound, i[0] * RES - self.bound)

    def clearance(self, p):
        iy, ix = self._idx(p)
        return float(self.clear[iy, ix])

    def _free_map(self, a, b):
        """Strict map, relaxed only in a small disc around the two end points (e.g. a survivor that
        sits close to an obstacle's shadow) -- never relaxed anywhere else."""
        f = self.free.copy()
        for p in (a, b):
            iy, ix = self._idx(p)
            r = int(0.7 / RES)
            y0, y1, x0, x1 = max(iy - r, 0), min(iy + r + 1, self.n), max(ix - r, 0), min(ix + r + 1, self.n)
            f[y0:y1, x0:x1] |= self.free_relaxed[y0:y1, x0:x1]
        return f

    def _seg_free(self, p, q, free):
        L = math.dist(p, q)
        k = max(int(L / 0.04), 1)
        for t in np.linspace(0, 1, k + 1):
            iy, ix = self._idx((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            if not free[iy, ix]:
                return False
        return True

    # ---- A* -----------------------------------------------------------------------
    def path(self, a, b):
        key = (round(a[0], 2), round(a[1], 2), round(b[0], 2), round(b[1], 2))
        if key in self._cache:
            return self._cache[key]
        free = self._free_map(a, b)
        s, g = self._idx(a), self._idx(b)
        if not free[s]:
            # start (e.g. the drone's position at the Switch) is inside the inflated zone: allow leaving it
            free = free.copy(); free[s] = True
        if not free[g]:
            raise RuntimeError(f"goal {b} is not reachable with the required clearance")
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        gscore, parent, cnt = {s: 0.0}, {}, 0
        h = lambda i: math.hypot(i[0] - g[0], i[1] - g[1])
        pq = [(h(s), 0, s)]
        while pq:
            _, _, u = heapq.heappop(pq)
            if u == g:
                break
            for dy, dx in nbrs:
                v = (u[0] + dy, u[1] + dx)
                if not (0 <= v[0] < self.n and 0 <= v[1] < self.n) or not free[v]:
                    continue
                c = gscore[u] + (1.4142 if dy and dx else 1.0)
                if c < gscore.get(v, 1e9):
                    gscore[v], parent[v] = c, u
                    cnt += 1
                    heapq.heappush(pq, (c + h(v), cnt, v))
        if g not in parent and g != s:
            raise RuntimeError(f"no collision-free path from {a} to {b}")
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        cells.reverse()
        pts = [tuple(a)] + [self._xy(c) for c in cells[1:-1]] + [tuple(b)]
        # line-of-sight shortcutting
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._seg_free(pts[i], pts[j], free):
                j -= 1
            out.append(pts[j]); i = j
        length = sum(math.dist(out[k], out[k + 1]) for k in range(len(out) - 1))
        self._cache[key] = (out, length)
        return out, length

    def warm(self, points):
        """Pre-compute every pairwise path (done before lift-off), so the route services that are called
        mid-flight only have to plan from the drone's current position."""
        for a in points:
            for b in points:
                if a != b:
                    self.path(a, b)

    # ---- ordering -----------------------------------------------------------------
    def best_order(self, start, goals, end=None):
        """goals: list of (label, (x, y)). Returns (ordered labels, total path length)."""
        best = None
        for perm in itertools.permutations(goals):
            pts = [start] + [g[1] for g in perm] + ([end] if end is not None else [])
            L = sum(self.path(pts[k], pts[k + 1])[1] for k in range(len(pts) - 1))
            if best is None or L < best[1] - 1e-9:
                best = ([g[0] for g in perm], L)
        return best

    @staticmethod
    def resample(poly, spacing=1.2):
        """Intermediate flight points: every corner, plus extra points so spacing <= `spacing`.
        The two ends are NOT included (they are the previous / next goal)."""
        out = []
        for p, q in zip(poly[:-1], poly[1:]):
            L = math.dist(p, q)
            k = max(int(math.ceil(L / spacing)), 1)
            for t in range(1, k + 1):
                out.append((p[0] + (q[0] - p[0]) * t / k, p[1] + (q[1] - p[1]) * t / k))
        return out[:-1]
