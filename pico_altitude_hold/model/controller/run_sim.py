#!/usr/bin/env python3
"""
Altitude hold demo / grader for the Swift Pico MuJoCo model (drone.xml).

Runs a cascaded PID controller (controller.py) that lifts the drone to a
target altitude and checks the settling criterion:

    error must be within +/-BAND of the setpoint within SETTLE_DEADLINE
    seconds, and stay within that band for HOLD_DURATION seconds straight.

Usage:
    python3 run_sim.py                  # headless check + CSV + plot
    python3 run_sim.py --viewer         # also opens an interactive 3D view
    python3 run_sim.py --duration 30    # simulate longer than the default
    python3 run_sim.py --setpoint 3.0 --band 0.4
"""
import argparse
import os
import sys
import time

import numpy as np
import mujoco

from controller import DroneController

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "model", "drone.xml")


def check_settling(ts, zs, setpoint, band, hold_duration, dt):
    """First time the error enters +/-band and never leaves it for the next
    hold_duration seconds. Returns (settle_time_or_None, longest streak start)."""
    in_band = np.abs(zs - setpoint) <= band
    hold_steps = int(round(hold_duration / dt))
    n = len(zs)
    for i in range(n - hold_steps):
        if in_band[i:i + hold_steps].all():
            return ts[i]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path to drone.xml")
    ap.add_argument("--setpoint", type=float, default=3.0, help="target altitude, m")
    ap.add_argument("--band", type=float, default=0.4, help="allowed error band, m")
    ap.add_argument("--settle-deadline", type=float, default=5.0,
                     help="error must first enter the band by this time, s")
    ap.add_argument("--hold-duration", type=float, default=10.0,
                     help="must then stay in the band this long, s")
    ap.add_argument("--duration", type=float, default=20.0,
                     help="total simulated time, s (must be >= settle-deadline + hold-duration)")
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer")
    ap.add_argument("--realtime", action="store_true",
                     help="throttle the headless loop to real time (ignored with --viewer)")
    ap.add_argument("--no-plot", action="store_true", help="skip writing altitude_plot.png")
    ap.add_argument("--csv", default=os.path.join(HERE, "log.csv"), help="where to write the run log")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    ctrl = DroneController(model, setpoint_xyz=(0.0, 0.0, args.setpoint))
    mujoco.mj_forward(model, data)

    n_steps = int(args.duration / dt)
    ts = np.zeros(n_steps)
    zs = np.zeros(n_steps)
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)

    def step_once():
        forces = ctrl.update(data, dt)
        data.ctrl[:] = forces
        mujoco.mj_step(model, data)

    if args.viewer:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            i = 0
            t0 = time.time()
            while viewer.is_running() and i < n_steps:
                step_once()
                ts[i] = data.time; zs[i] = data.qpos[2]; xs[i] = data.qpos[0]; ys[i] = data.qpos[1]
                i += 1
                viewer.sync()
                # keep the viewer roughly real-time
                target = t0 + data.time
                lag = target - time.time()
                if lag > 0:
                    time.sleep(lag)
            n_steps = i  # in case the window was closed early
    else:
        t0 = time.time()
        for i in range(n_steps):
            step_once()
            ts[i] = data.time; zs[i] = data.qpos[2]; xs[i] = data.qpos[0]; ys[i] = data.qpos[1]
            if args.realtime:
                target = t0 + data.time
                lag = target - time.time()
                if lag > 0:
                    time.sleep(lag)

    ts, zs, xs, ys = ts[:n_steps], zs[:n_steps], xs[:n_steps], ys[:n_steps]

    settle_time = check_settling(ts, zs, args.setpoint, args.band, args.hold_duration, dt)
    passed = settle_time is not None and settle_time <= args.settle_deadline

    print("\n=== Altitude hold report ===")
    print(f"  setpoint         : {args.setpoint:.2f} m")
    print(f"  band             : +/-{args.band:.2f} m")
    print(f"  final altitude   : {zs[-1]:.3f} m  (error {zs[-1]-args.setpoint:+.3f} m)")
    print(f"  max horizontal drift : {np.max(np.hypot(xs, ys)):.3f} m")
    if settle_time is not None:
        print(f"  entered band and held for {args.hold_duration:.0f}s starting at t = {settle_time:.2f} s")
    else:
        print(f"  never held the band continuously for {args.hold_duration:.0f}s in {args.duration:.0f}s of sim time")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} "
          f"(needed to settle by t <= {args.settle_deadline:.0f}s)")

    with open(args.csv, "w") as f:
        f.write("t,z,x,y\n")
        for t, z, x, y in zip(ts, zs, xs, ys):
            f.write(f"{t:.3f},{z:.5f},{x:.5f},{y:.5f}\n")
    print(f"\nLogged {len(ts)} samples to {args.csv}")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(ts, zs, label="altitude z(t)")
            ax.axhline(args.setpoint, color="k", ls="--", lw=1, label="setpoint")
            ax.axhspan(args.setpoint - args.band, args.setpoint + args.band,
                       color="green", alpha=0.15, label=f"+/-{args.band} m band")
            if settle_time is not None:
                ax.axvline(settle_time, color="red", ls=":", lw=1,
                           label=f"settled at t={settle_time:.2f}s")
            ax.set_xlabel("time (s)"); ax.set_ylabel("altitude (m)")
            ax.set_title("Altitude hold response")
            ax.legend(loc="lower right")
            plot_path = os.path.join(HERE, "altitude_plot.png")
            fig.tight_layout(); fig.savefig(plot_path, dpi=130)
            print(f"Saved plot to {plot_path}")
        except ImportError:
            print("matplotlib not installed -- skipping plot (pip install matplotlib)")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
