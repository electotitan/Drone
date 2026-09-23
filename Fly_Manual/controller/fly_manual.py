#!/usr/bin/env python3
"""
Fly the Swift Pico yourself with the keyboard.

This is "self-level" manual control, the same way a beginner-mode hobby
drone works: your keys move the PID's target x/y/z/yaw (see controller.py)
and the existing PID stack (altitude + attitude + mixer) does the actual
balancing. You are commanding *where you want to be*, not raw motor power
-- flying on raw per-motor thrust with no stabilization is a much harder,
twitchier task and not what most people mean by "control the drone"; see
the bottom of this file if you want that instead.

Controls:
    W / S       pitch forward / back   (+x / -x target)
    A / D       roll left / right      (+y / -y target)
    Up / Down   throttle up / down     (z target)
    Q / E       yaw left / right
    R           reset target to the spawn hover point
    Esc         quit

Usage:
    python3 fly_manual.py
    python3 fly_manual.py --step-xy 0.3 --step-z 0.3   # bigger nudges per keypress
"""
import argparse
import math
import os
import time

import mujoco
import mujoco.viewer
import glfw

from controller import DroneController
from motor_spin import PropSpinner

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "model", "drone.xml")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--hover-z", type=float, default=3.0, help="starting/target altitude, m")
    ap.add_argument("--step-xy", type=float, default=0.15, help="x/y target nudge per keypress, m")
    ap.add_argument("--step-z", type=float, default=0.15, help="z target nudge per keypress, m")
    ap.add_argument("--step-yaw-deg", type=float, default=8.0, help="yaw target nudge per keypress, deg")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    spawn_x, spawn_y = 0.0, 0.0
    ctrl = DroneController(model, setpoint_xyz=(spawn_x, spawn_y, args.hover_z))
    spinner = PropSpinner(model)
    mujoco.mj_forward(model, data)
    ctrl.update(data, dt)  # one warm-up call so ctrl.yaw_sp gets captured from the spawn pose
    spawn_yaw = ctrl.yaw_sp

    step_yaw = math.radians(args.step_yaw_deg)

    print(__doc__)
    print(f"Target: x={ctrl.sp_x:+.2f}  y={ctrl.sp_y:+.2f}  z={ctrl.sp_z:+.2f}  "
          f"yaw={math.degrees(ctrl.yaw_sp):+.0f} deg\n")

    def key_callback(key):
        if key == glfw.KEY_W:
            ctrl.sp_x += args.step_xy
        elif key == glfw.KEY_S:
            ctrl.sp_x -= args.step_xy
        elif key == glfw.KEY_A:
            ctrl.sp_y += args.step_xy
        elif key == glfw.KEY_D:
            ctrl.sp_y -= args.step_xy
        elif key == glfw.KEY_UP:
            ctrl.sp_z += args.step_z
        elif key == glfw.KEY_DOWN:
            ctrl.sp_z = max(0.2, ctrl.sp_z - args.step_z)
        elif key == glfw.KEY_Q:
            ctrl.yaw_sp += step_yaw
        elif key == glfw.KEY_E:
            ctrl.yaw_sp -= step_yaw
        elif key == glfw.KEY_R:
            ctrl.sp_x, ctrl.sp_y, ctrl.sp_z, ctrl.yaw_sp = spawn_x, spawn_y, args.hover_z, spawn_yaw
        else:
            return
        print(f"Target: x={ctrl.sp_x:+.2f}  y={ctrl.sp_y:+.2f}  z={ctrl.sp_z:+.2f}  "
              f"yaw={math.degrees(ctrl.yaw_sp):+.0f} deg")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        t0 = time.time()
        while viewer.is_running():
            forces = ctrl.update(data, dt)
            spinner.apply(data, forces)  # spin the propeller meshes to match thrust
            data.ctrl[:] = forces
            mujoco.mj_step(model, data)
            viewer.sync()
            lag = (t0 + data.time) - time.time()
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()

# --- Raw manual mode (no stabilization) -------------------------------
# If you actually want to fly on bare per-motor thrust (no PID helping
# you), skip DroneController entirely and drive data.ctrl (4 values, N,
# clipped to [0, 5.47]) straight from the keyboard, e.g. one key pair
# per motor or a simple mixer of your own. Expect it to flip within a
# second or two without attitude assistance -- the model isn't
# perfectly symmetric (see README), so it won't hover level on its own.
