"""
SimNode -- owns the MuJoCo model and the Task 2A (LQR) controller.

Runs the physics + control loop in whatever thread calls run() (the main thread
in run_mission.py, so the MuJoCo viewer works on every OS). Other threads (the
action server) talk to it only through thread-safe methods:

    get_state()                 latest pose / velocity
    set_setpoint(xyz)           retarget the LQR controller
    samples_since(i)            every logged step since index i (t, x, y, z)
    wait_for_time(t)            block until the sim clock reaches t
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

DEFAULT_MODEL = os.path.join(HERE, "..", "model", "drone.xml")


@dataclass
class State:
    t: float
    pos: np.ndarray
    vel: np.ndarray


class SimNode:
    def __init__(self, model_path=DEFAULT_MODEL, initial_setpoint=None, speed=0.0):
        """speed: 1.0 = real time, 2.0 = twice real time, 0 = as fast as possible."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        mujoco.mj_forward(self.model, self.data)
        spawn = self.data.qpos[0:3].copy()
        self.controller = DroneController(self.model, tuple(initial_setpoint if initial_setpoint is not None else spawn))
        self.speed = speed

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._setpoint = np.array(self.controller.sp)
        self._t = 0.0
        self._pos = spawn
        self._vel = np.zeros(3)
        self._log = np.zeros((4096, 4))
        self._n = 0
        self._stop = threading.Event()

        self._obstacle_geoms = {i for i in range(self.model.ngeom)
                                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("obstacle")
                                and self.model.geom_contype[i] != 0}
        self.collisions = set()

    # ------------------------------------------------------------------ thread-safe API
    def get_state(self):
        with self._lock:
            return State(self._t, self._pos.copy(), self._vel.copy())

    def set_setpoint(self, xyz):
        """Returns (sim_time, sample_index) at which the new setpoint was latched."""
        with self._lock:
            self._setpoint = np.array(xyz, dtype=float)
            return self._t, self._n

    def samples_since(self, index):
        with self._lock:
            return self._n, self._log[index:self._n].copy()

    def wait_for_time(self, t_target, wall_timeout=10.0):
        """Block until sim clock >= t_target. False if the sim stopped / timed out."""
        end = time.time() + wall_timeout
        with self._cond:
            while self._t < t_target:
                if self._stop.is_set():
                    return False
                remaining = end - time.time()
                if remaining <= 0:
                    return False
                self._cond.wait(min(remaining, 0.1))
            return True

    def full_log(self):
        with self._lock:
            return self._log[:self._n].copy()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    @property
    def stopped(self):
        return self._stop.is_set()

    # ------------------------------------------------------------------ the loop
    def _step(self):
        m, d = self.model, self.data
        with self._lock:
            sp = self._setpoint
        self.controller.set_setpoint(sp)
        d.ctrl[:] = self.controller.update(d, self.dt)
        mujoco.mj_step(m, d)

        if d.ncon and self._obstacle_geoms:
            for k in range(d.ncon):
                c = d.contact[k]
                if c.geom1 in self._obstacle_geoms or c.geom2 in self._obstacle_geoms:
                    self.collisions.add(round(d.time, 2))

        with self._lock:
            self._t = d.time
            self._pos = d.qpos[0:3].copy()
            self._vel = d.qvel[0:3].copy()
            if self._n == len(self._log):
                self._log = np.vstack([self._log, np.zeros_like(self._log)])
            self._log[self._n] = (d.time, *d.qpos[0:3])
            self._n += 1

    def run(self, viewer=False):
        """Blocking physics loop. Call from the main thread."""
        wall0 = time.time()
        t_sim0 = self.data.time
        i = 0
        if viewer:
            from mujoco import viewer as mj_viewer
            ctx = mj_viewer.launch_passive(self.model, self.data)
        else:
            ctx = None
        try:
            while not self._stop.is_set():
                if ctx is not None and not ctx.is_running():
                    self.stop()
                    break
                self._step()
                i += 1
                if i % 10 == 0:                         # every 0.05 s of sim time
                    with self._cond:
                        self._cond.notify_all()
                    if ctx is not None and i % 40 == 0:
                        ctx.sync()
                    if self.speed > 0:
                        lag = wall0 + (self.data.time - t_sim0) / self.speed - time.time()
                        if lag > 0:
                            time.sleep(lag)
                    else:
                        time.sleep(0)                   # let the server/client threads run
        finally:
            if ctx is not None:
                ctx.close()
            self.stop()
