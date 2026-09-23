import numpy as np
import mujoco
from pid import PID

ARM = 0.19          # motor arm length (m), from drone.xml site positions
YAW_COEF = 0.016     # reactive-yaw coefficient in the actuator <gear>
MAX_THRUST_PER_MOTOR = 5.47  # N, from actuator ctrlrange

# Mixer matrix inverse (derived analytically from the drone's X-layout and
# gear vector in drone.xml -- see README for the derivation). Maps
# [F_total, tau_x, tau_y, tau_z] -> [F1, F2, F3, F4].
def mix(F_total, tau_x, tau_y, tau_z):
    a, b = ARM, YAW_COEF
    u2 = tau_x / a
    u3 = tau_y / a
    u4 = tau_z / b
    f1 = (F_total - u2 - u3 - u4) / 4.0
    f2 = (F_total + u2 + u3 - u4) / 4.0
    f3 = (F_total + u2 - u3 + u4) / 4.0
    f4 = (F_total - u2 + u3 + u4) / 4.0
    return np.array([f1, f2, f3, f4])


class DroneController:
    """Cascaded PID controller for the Swift Pico model in drone.xml.

    altitude PID  -> vertical force
    xy PID        -> small horizontal force (keeps the drone from drifting)
    thrust vector -> desired body attitude (geometric control)
    attitude PD   -> roll/pitch torque
    yaw PID       -> yaw torque (holds the spawn heading)
    mixer         -> 4 motor thrusts, clipped to actuator range
    """

    def __init__(self, model, setpoint_xyz=(0.0, 0.0, 3.0)):
        self.model = model
        self.total_mass = model.body_mass.sum()
        self.g = -model.opt.gravity[2]
        self.hover_thrust = self.total_mass * self.g

        self.sp_x, self.sp_y, self.sp_z = setpoint_xyz
        self.yaw_sp = None  # captured on first update() call from spawn orientation

        # --- Gains (tuned against this model, see README) ---
        self.alt_pid = PID(kp=1.6, ki=0.08, kd=2.0,
                            out_min=-self.hover_thrust, out_max=4 * MAX_THRUST_PER_MOTOR - self.hover_thrust,
                            i_min=-2.0, i_max=2.0)
        # Outer (position) loop must be much slower than the inner (attitude)
        # loop or the cascade rings / slowly diverges. Attitude gains below
        # are sized from the body's own inertia (Ixx=0.0347, Iyy=0.07,
        # Izz=0.0977 in drone.xml) for ~10 rad/s closed-loop bandwidth,
        # damping ratio ~0.7 -- about 10x the position loop's ~0.9 rad/s.
        self.x_pid = PID(kp=0.8, ki=0.0, kd=1.0, out_min=-1.5, out_max=1.5)
        self.y_pid = PID(kp=0.8, ki=0.0, kd=1.0, out_min=-1.5, out_max=1.5)
        self.roll_pid  = PID(kp=3.5, ki=0.0, kd=0.5, out_min=-1.0, out_max=1.0)
        self.pitch_pid = PID(kp=7.0, ki=0.0, kd=1.0, out_min=-1.0, out_max=1.0)
        self.yaw_pid   = PID(kp=9.8, ki=0.0, kd=1.4, out_min=-0.5, out_max=0.5)

        self._body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")

    def reset(self, data):
        self.alt_pid.reset(); self.x_pid.reset(); self.y_pid.reset()
        self.roll_pid.reset(); self.pitch_pid.reset(); self.yaw_pid.reset()
        self.yaw_sp = None

    def update(self, data, dt):
        x, y, z = data.qpos[0:3]
        R = data.xmat[self._body_id].reshape(3, 3)

        # MuJoCo free-joint qvel convention: qvel[0:3] is the linear velocity
        # of the body origin in the WORLD frame; qvel[3:6] is the angular
        # velocity already expressed in the BODY-LOCAL frame. No extra
        # rotation is needed for either.
        vx_w, vy_w, vz_w = data.qvel[0:3]
        wx_b, wy_b, wz_b = data.qvel[3:6]

        if self.yaw_sp is None:
            self.yaw_sp = np.arctan2(R[1, 0], R[0, 0])

        # --- Altitude loop: desired vertical (world) force ---
        u_z = self.alt_pid.update(self.sp_z - z, vz_w, dt)
        Fz_des = self.hover_thrust + self.total_mass * u_z

        # --- XY loop: small desired horizontal (world) force, gently pulls
        #     the drone back toward the setpoint / spawn column ---
        ax = self.x_pid.update(self.sp_x - x, vx_w, dt)
        ay = self.y_pid.update(self.sp_y - y, vy_w, dt)
        Fx_des = self.total_mass * ax
        Fy_des = self.total_mass * ay

        # --- Thrust vectoring: desired body +Z axis points along the
        #     desired total force vector (standard geometric quad control) ---
        f_des = np.array([Fx_des, Fy_des, Fz_des])
        F_total = np.linalg.norm(f_des)
        F_total = np.clip(F_total, 0.0, 4 * MAX_THRUST_PER_MOTOR)
        bz_des = f_des / max(np.linalg.norm(f_des), 1e-6)

        bz = R[:, 2]
        tilt_err_world = np.cross(bz, bz_des)          # small-angle attitude error
        tilt_err_body = R.T @ tilt_err_world

        tau_x = self.roll_pid.update(tilt_err_body[0], wx_b, dt)
        tau_y = self.pitch_pid.update(tilt_err_body[1], wy_b, dt)

        yaw = np.arctan2(R[1, 0], R[0, 0])
        yaw_err = np.arctan2(np.sin(self.yaw_sp - yaw), np.cos(self.yaw_sp - yaw))
        tau_z = self.yaw_pid.update(yaw_err, wz_b, dt)

        motor_forces = mix(F_total, tau_x, tau_y, tau_z)
        return np.clip(motor_forces, 0.0, MAX_THRUST_PER_MOTOR)
