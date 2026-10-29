import numpy as np
import mujoco

from lqr import (design_lqr, IDX_POS, IDX_VEL, IDX_ATT, IDX_RATE, IDX_INT, N_STATE)

ARM = 0.19                   # motor arm length (m), from drone.xml site positions
YAW_COEF = 0.016             # reactive-yaw coefficient in the actuator <gear>
MAX_THRUST_PER_MOTOR = 5.47  # N, from actuator ctrlrange


def mix(F_total, tau_x, tau_y, tau_z):
    """[F_total, tau_x, tau_y, tau_z] -> [F1, F2, F3, F4] (N), actuator order of
    drone.xml. Derived from the X-layout site positions and the gear vectors."""
    a, b = ARM, YAW_COEF
    u2 = tau_x / a
    u3 = tau_y / a
    u4 = tau_z / b
    f1 = (F_total - u2 - u3 - u4) / 4.0
    f2 = (F_total + u2 + u3 - u4) / 4.0
    f3 = (F_total + u2 - u3 + u4) / 4.0
    f4 = (F_total - u2 + u3 + u4) / 4.0
    return np.array([f1, f2, f3, f4])


def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


class DroneController:
    """Full-state LQR (with integral action) for the Swift Pico in drone.xml.

    One 4x15 gain matrix K, designed offline in lqr.py from the model's own
    mass/inertia, replaces the whole cascade of PIDs:

        x = [pos err, vel, roll/pitch/yaw, body rates, integral of pos err]
        u = -K x = [dF, tau_x, tau_y, tau_z]
        F_total = (m*g + dF) / cos(tilt)         (tilt compensation)
        mixer -> 4 motor thrusts, clipped to [0, 5.47] N

    Practical additions around the linear LQR (it is only optimal near hover):
      * position error is saturated before it enters K, so a big setpoint step
        cannot demand a huge tilt/thrust (acts like a built-in speed limit);
      * the integrator is conditional (only runs when close to the setpoint)
        and clamped, so it cannot wind up during the initial climb;
      * the whole thing is evaluated in the drone's heading frame, so it does
        not care that the drone spawns rotated 180 deg about z.
    """

    def __init__(self, model, setpoint_xyz=(0.0, 0.0, 3.0), Q=None, R=None,
                 max_xy_err=1.2, max_z_err=1.0, integ_zone=0.8, integ_limit=1.5):
        self.model = model
        self.dt = model.opt.timestep
        self.K, self.info = design_lqr(model, self.dt, Q, R)
        self.total_mass = self.info["mass"]
        self.g = self.info["g"]
        self.hover_thrust = self.total_mass * self.g

        self.sp = np.array(setpoint_xyz, dtype=float)
        self.yaw_sp = None   # captured on first update() from the spawn heading

        self.max_xy_err = max_xy_err     # |position error| fed to K is clipped to this (m)
        self.max_z_err = max_z_err
        self.integ_zone = integ_zone     # integrate only when |error| < this (m)
        self.integ_limit = integ_limit   # clamp on each integral state (m*s)
        self._int = np.zeros(3)          # integral of heading-frame position error

        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")

    def set_setpoint(self, xyz):
        """Retarget the controller. The integral states are deliberately kept:
        they hold the estimate of the constant thrust/torque bias (mass error,
        COM offset), which is the same at every waypoint."""
        self.sp = np.array(xyz, dtype=float)

    def reset(self, data=None):
        self._int[:] = 0.0
        self.yaw_sp = None

    def update(self, data, dt):
        pos = data.qpos[0:3]
        R = data.xmat[self._body_id].reshape(3, 3)

        # MuJoCo free joint: qvel[0:3] = linear velocity in WORLD frame,
        #                    qvel[3:6] = angular velocity in BODY frame.
        vel_w = data.qvel[0:3]
        rates_b = data.qvel[3:6]

        # ZYX Euler angles from the rotation matrix
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))

        if self.yaw_sp is None:
            self.yaw_sp = yaw

        # ---- error state, expressed in the heading frame -------------------
        err_w = pos - self.sp
        err_w_sat = err_w.copy()
        xy_norm = np.linalg.norm(err_w[:2])
        if xy_norm > self.max_xy_err:
            err_w_sat[:2] *= self.max_xy_err / xy_norm
        err_w_sat[2] = np.clip(err_w[2], -self.max_z_err, self.max_z_err)

        c, s = np.cos(yaw), np.sin(yaw)
        Rh = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # world -> heading
        err_h = Rh @ err_w_sat
        vel_h = Rh @ vel_w

        # Conditional integration (anti-windup): only near the setpoint
        if np.linalg.norm(err_w) < self.integ_zone:
            self._int += (Rh @ err_w) * dt
            self._int = np.clip(self._int, -self.integ_limit, self.integ_limit)

        x = np.zeros(N_STATE)
        x[IDX_POS] = err_h
        x[IDX_VEL] = vel_h
        x[IDX_ATT] = [roll, pitch, wrap_pi(yaw - self.yaw_sp)]
        x[IDX_RATE] = rates_b
        x[IDX_INT] = self._int

        # ---- LQR law --------------------------------------------------------
        dF, tau_x, tau_y, tau_z = -self.K @ x

        # Tilt-compensated collective thrust
        cos_tilt = max(R[2, 2], 0.5)
        F_total = np.clip((self.hover_thrust + dF) / cos_tilt, 0.0, 4 * MAX_THRUST_PER_MOTOR)

        motor_forces = mix(F_total, tau_x, tau_y, tau_z)
        return np.clip(motor_forces, 0.0, MAX_THRUST_PER_MOTOR)
