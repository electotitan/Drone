"""
waypoint_service (program 1 of 3)

  subscribes   /image_raw                  ONE camera frame is all it ever uses
  finds        survivors (red triangle = critical, yellow circle = stable) and obstacles
  serves       get_waypoints               -> the CRITICAL route, planned from F6
               get_stable_route (x, y, z)  -> the STABLE route, planned from where the drone is
                                              NOW (the Switch position) and ending at F6

Route format: for every survivor  [fly-past points at Z = 13] -> visit (Z = 13, hold 1 s)
                                                              -> save  (Z = 14, hold 2 s)
Fly-past ('via') goals have hold 0: the drone just has to pass through their box. All routes are
planned around the obstacles (see planner.py).
"""
import threading

import lattice
from messages import (Waypoint, GetWaypointsResponse, GetStableRouteResponse)
from middleware import ServiceServer, Subscriber
from planner import Planner
from vision import detect_scene

HOME_CELL = "F6"
FLIGHT_Z = 13.0          # WhyCode Z of the route (2 m above the floor)
SAVE_Z = 14.0            # WhyCode Z of the Save hold (1 m above the floor)
VISIT_HOLD, SAVE_HOLD, HOME_HOLD = 1.0, 2.0, 1.0
BOX = 0.4                # +/- box on X, Y and Z


class WaypointService:
    def __init__(self, camera, image_topic="/image_raw", margin=0.85, via_spacing=1.2, log=print):
        self.cam, self.margin, self.via_spacing, self.log = camera, margin, via_spacing, log
        self.home = lattice.cell_to_world(HOME_CELL)
        self.scene = None
        self.planner = None
        self._frame = None
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._sub = Subscriber(image_topic, self._on_image)
        self._srv_crit = ServiceServer("get_waypoints", self._get_waypoints)
        self._srv_stable = ServiceServer("get_stable_route", self._get_stable_route)

    def shutdown(self):
        self._srv_crit.shutdown(); self._srv_stable.shutdown()

    # ------------------------------------------------------------------ perception
    def _on_image(self, frame):
        self._frame = frame
        self._frame_ready.set()

    def _ensure_scene(self):
        with self._lock:
            if self.scene is None:
                if not self._frame_ready.wait(10.0):
                    raise RuntimeError("no camera frame on /image_raw")
                self.scene = detect_scene(self._frame, self.cam, home_xy=self.home)
                self.planner = Planner(self.scene, self.cam, plane_z=FLIGHT_Z, margin=self.margin)
                self.planner.warm([tuple(self.home)] + [(s.x, s.y) for s in self.scene.survivors])
                self.log("[waypoint_service] one frame -> "
                         f"{len(self.scene.critical)} critical {[s.cell for s in self.scene.critical]}, "
                         f"{len(self.scene.stable)} stable {[s.cell for s in self.scene.stable]}, "
                         f"{len(self.scene.obstacle_info)} obstacles")

    # ------------------------------------------------------------------ route building
    def _legs(self, start_xy, ordered, phase, end_home, start_Z=FLIGHT_Z):
        P, wps, cur, k = self.planner, [], tuple(start_xy), 0
        if start_Z > FLIGHT_Z + BOX:                          # drone is low (e.g. cancelled mid-Save): climb first
            wps.append(Waypoint("climb", cur[0], cur[1], FLIGHT_Z, "via", 0.0, BOX))
        def vias(a, b):
            nonlocal k
            poly, _ = P.path(a, b)
            for p in P.resample(poly, self.via_spacing):
                k += 1
                wps.append(Waypoint(f"via {k}", p[0], p[1], FLIGHT_Z, "via", 0.0, BOX))
        for s in ordered:
            vias(cur, (s.x, s.y))
            wps.append(Waypoint(f"visit {s.cell}", s.x, s.y, FLIGHT_Z, "visit", VISIT_HOLD, BOX, s.cell, phase))
            wps.append(Waypoint(f"save {s.cell}", s.x, s.y, SAVE_Z, "save", SAVE_HOLD, BOX, s.cell, phase))
            cur = (s.x, s.y)
        if end_home:
            vias(cur, self.home)
            wps.append(Waypoint(f"home {HOME_CELL}", self.home[0], self.home[1], FLIGHT_Z, "home", HOME_HOLD, BOX))
        return wps

    def _order(self, start_xy, survivors, end_xy=None):
        by = {s.cell: s for s in survivors}
        labels, length = self.planner.best_order(tuple(start_xy), [(s.cell, (s.x, s.y)) for s in survivors], end_xy)
        return [by[c] for c in labels], length

    # ------------------------------------------------------------------ services
    def _get_waypoints(self, _req):
        self._ensure_scene()
        ordered, length = self._order(self.home, self.scene.critical)
        wps = self._legs(self.home, ordered, "critical", end_home=False)
        info = f"critical route {' -> '.join(s.cell for s in ordered)}  ({length:.1f} m, {len(wps)} goals)"
        self.log(f"[waypoint_service] {info}")
        return GetWaypointsResponse(waypoints=wps, info=info)

    def _get_stable_route(self, req):
        self._ensure_scene()
        start = (req.x, req.y)
        ordered, length = self._order(start, self.scene.stable, end_xy=self.home)
        wps = self._legs(start, ordered, "stable", end_home=True, start_Z=req.z)
        info = (f"stable route from ({req.x:+.2f}, {req.y:+.2f}, Z={req.z:.2f}): "
                f"{' -> '.join(s.cell for s in ordered)} -> {HOME_CELL}  ({length:.1f} m, {len(wps)} goals)")
        self.log(f"[waypoint_service] {info}")
        return GetStableRouteResponse(waypoints=wps, info=info)
