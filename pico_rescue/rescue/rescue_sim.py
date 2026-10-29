"""
RescueSim -- the MuJoCo world + the Task 2A LQR controller, as seen by the three programs.

Differences from the Task 2C SimNode:
  * WhyCode coordinates: X, Y in world metres, Z = distance DOWN from the top camera (z_world = 15 - Z).
    So Z = 13 is 2 m above the floor and Z = 14 is 1 m above the floor.
  * Motors are OFF until arm() (the drone sits on F6). The first arm() is the lift-off, t = 0.
  * Camera line of sight: every step a ray is cast from the top camera to the drone's marker. If a
    3 m obstacle is in the way the marker is "lost" (marker_visible = False) -- the hold monitors
    treat that exactly like leaving the box.
  * disarm() for landing, a 200 s run cap measured from lift-off, contact detection vs obstacles.

Runs the physics in the thread that calls run() (the main thread), other threads use the
thread-safe methods only.
"""
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "controller"))
from controller import DroneController      # noqa: E402  (Task 2A LQR controller)

DEFAULT_MODEL = os.path.join(HERE, "..", "model", "drone_rescue.xml")
LOG_COLS = "t,X,Y,Z,visible,z_world,armed"


@dataclass
class State:
    t: float
    mission_time: float
    X: float
    Y: float
    Z: float              # WhyCode Z
    z_world: float
    vel: np.ndarray
    visible: bool
    armed: bool


class RescueSim:
    def __init__(self, model_path=DEFAULT_MODEL, speed=0.0, max_run=200.0):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        mujoco.mj_forward(self.model, self.data)
        self.speed, self.max_run = speed, max_run

        self._body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_cam")
        self.cam_pos = self.model.cam_pos[cam].astype(float).copy()
        self.cam_z = float(self.cam_pos[2])

        spawn = self.data.qpos[0:3].copy()
        self.controller = DroneController(self.model, tuple(spawn))

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._sp = spawn.copy()
        self._armed = False
        self.t_takeoff = None
        self.t_disarm = None
        self.timed_out = False
        self._t, self._pos, self._vel, self._vis = 0.0, spawn.copy(), np.zeros(3), True
        self._log = np.zeros((8192, 7))
        self._n = 0

        self._obstacles = {i for i in range(self.model.ngeom)
                           if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("obstacle_")
                           and self.model.geom_contype[i] != 0}
        self.collisions = set()
        self._geomid = np.zeros(1, dtype=np.int32)

    # ------------------------------------------------------------------ conversions
    def to_world_z(self, Z):
        return self.cam_z - Z

    def to_whycode_z(self, z):
        return self.cam_z - z

    # ------------------------------------------------------------------ thread-safe API
    def arm(self):
        """Lift-off. The first call defines mission time t = 0. Returns the sim time."""
        with self._lock:
            if not self._armed:
                self._armed = True
                if self.t_takeoff is None:
                    self.t_takeoff = self._t
                self.t_disarm = None
            return self._t

    def disarm(self):
        with self._lock:
            self._armed = False
            self.t_disarm = self._t

    @property
    def armed(self):
        return self._armed

    def mission_time(self, t=None):
        with self._lock:
            t = self._t if t is None else t
            return 0.0 if self.t_takeoff is None else t - self.t_takeoff

    def set_setpoint(self, X, Y, Z):
        """Target in WhyCode coordinates. Returns (sim_time, sample_index) when it was latched."""
        with self._lock:
            self._sp = np.array([X, Y, self.to_world_z(Z)], dtype=float)
            return self._t, self._n

    def get_state(self):
        with self._lock:
            p = self._pos
            return State(self._t, self.mission_time(), float(p[0]), float(p[1]), float(self.to_whycode_z(p[2])),
                         float(p[2]), self._vel.copy(), bool(self._vis), self._armed)

    def samples_since(self, index):
        with self._lock:
            return self._n, self._log[index:self._n].copy()

    def full_log(self):
        with self._lock:
            return self._log[:self._n].copy()

    def wait_for_time(self, t_target, wall_timeout=10.0):
        end = time.time() + wall_timeout
        with self._cond:
            while self._t < t_target:
                if self._stop.is_set():
                    return False
                rem = end - time.time()
                if rem <= 0:
                    return False
                self._cond.wait(min(rem, 0.1))
            return True

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    @property
    def stopped(self):
        return self._stop.is_set()

    # ------------------------------------------------------------------ camera line of sight
    def _marker_visible(self):
        d = self.data
        target = d.xpos[self._body] + np.array([0.0, 0.0, 0.1])      # marker is on top of the drone
        vec = target - self.cam_pos
        dist = float(np.linalg.norm(vec))
        hit = mujoco.mj_ray(self.model, d, self.cam_pos, vec / dist, None, 1, -1, self._geomid)
        if hit < 0:
            return True
        gid = int(self._geomid[0])
        if self.model.body_rootid[self.model.geom_bodyid[gid]] == self._body:    # first thing hit is the drone itself
            return True
        return hit >= dist - 0.02                                                 # something else, but behind the drone

    # ------------------------------------------------------------------ the loop
    def _step(self):
        m, d = self.model, self.data
        with self._lock:
            sp, armed = self._sp.copy(), self._armed
        if armed:
            self.controller.set_setpoint(sp)
            d.ctrl[:] = self.controller.update(d, self.dt)
        else:
            d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)

        if d.ncon and self._obstacles:
            for k in range(d.ncon):
                c = d.contact[k]
                if c.geom1 in self._obstacles or c.geom2 in self._obstacles:
                    self.collisions.add(round(d.time, 2))

        vis = self._marker_visible()
        with self._lock:
            self._t = d.time
            self._pos = d.qpos[0:3].copy()
            self._vel = d.qvel[0:3].copy()
            self._vis = vis
            if self._n == len(self._log):
                self._log = np.vstack([self._log, np.zeros_like(self._log)])
            self._log[self._n] = (d.time, d.qpos[0], d.qpos[1], self.cam_z - d.qpos[2], float(vis), d.qpos[2], float(self._armed))
            self._n += 1

    def run(self, viewer=False):
        wall0, t_sim0, i = time.time(), self.data.time, 0
        ctx = None
        if viewer:
            from mujoco import viewer as mj_viewer
            ctx = mj_viewer.launch_passive(self.model, self.data)
        try:
            while not self._stop.is_set():
                if ctx is not None and not ctx.is_running():
                    break
                self._step()
                i += 1
                if self.t_takeoff is not None and self._t - self.t_takeoff >= self.max_run:
                    self.timed_out = True                      # "the run ends at 200 s"
                    break
                if i % 10 == 0:
                    with self._cond:
                        self._cond.notify_all()
                    if ctx is not None and i % 40 == 0:
                        ctx.sync()
                    # before lift-off the drone just sits on F6: don't let an unthrottled sim race ahead
                    if self.t_takeoff is None:
                        eff = self.speed if self.speed > 0 else 20.0      # idle on the pad: at most 20x real time
                    else:
                        eff = self.speed                                  # 0 = flat out
                    if eff > 0:
                        lag = wall0 + (self.data.time - t_sim0) / eff - time.time()
                        if lag > 0:
                            time.sleep(lag)
                        else:
                            time.sleep(0)
                    else:
                        time.sleep(0)
        finally:
            if ctx is not None:
                ctx.close()
            self.stop()
