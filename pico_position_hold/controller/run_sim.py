#!/usr/bin/env python3
"""
Altitude hold demo / grader for the Swift Pico MuJoCo model (drone.xml).

Runs a cascaded PID controller (controller.py) that drives the drone to a
target x/y/z (Pitch/Roll/Throttle) and checks, on EACH axis independently:

    error must be within +/-BAND of that axis's setpoint within
    SETTLE_DEADLINE seconds, and stay within that band for HOLD_DURATION
    seconds straight.

Usage:
    python3 run_sim.py                          # headless check + CSV + plot
    python3 run_sim.py --viewer                 # also opens an interactive 3D view
    python3 run_sim.py --duration 30             # simulate longer than the default
    python3 run_sim.py --setpoint 2.0 2.0 3.0 --band 0.4   # x y z target
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


def check_settling(ts, values, setpoint, band, hold_duration, dt):
    """First time the error enters +/-band and never leaves it for the next
    hold_duration seconds. Returns settle_time or None."""
    in_band = np.abs(values - setpoint) <= band
    hold_steps = int(round(hold_duration / dt))
    n = len(values)
    for i in range(n - hold_steps):
        if in_band[i:i + hold_steps].all():
            return ts[i]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path to drone.xml")
    ap.add_argument("--setpoint", type=float, nargs=3, default=[0.0, 0.0, 3.0],
                     metavar=("X", "Y", "Z"),
                     help="target x y z (Pitch, Roll, Throttle setpoints), m")
    ap.add_argument("--band", type=float, default=0.4, help="allowed error band, m (same for all 3 axes)")
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

    sp_x, sp_y, sp_z = args.setpoint
    ctrl = DroneController(model, setpoint_xyz=(sp_x, sp_y, sp_z))
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

    axes = [
        ("Pitch (x)",    xs, sp_x),
        ("Roll (y)",     ys, sp_y),
        ("Throttle (z)", zs, sp_z),
    ]
    print("\n=== Position/altitude hold report ===")
    print(f"  band            : +/-{args.band:.2f} m on each axis")
    print(f"  settle deadline : {args.settle_deadline:.0f} s, hold duration: {args.hold_duration:.0f} s\n")

    all_passed = True
    settle_times = {}
    for name, values, sp in axes:
        st = check_settling(ts, values, sp, args.band, args.hold_duration, dt)
        ok = st is not None and st <= args.settle_deadline
        all_passed &= ok
        settle_times[name] = st
        final_err = values[-1] - sp
        status = f"settled at t = {st:.2f} s" if st is not None else "never held the band for the full duration"
        print(f"  {name:14s} setpoint={sp:+.2f}  final={values[-1]:+.3f} (err {final_err:+.3f})  "
              f"{status}  [{'PASS' if ok else 'FAIL'}]")

    print(f"\n  OVERALL RESULT: {'PASS' if all_passed else 'FAIL'} "
          f"(every axis must settle by t <= {args.settle_deadline:.0f}s and hold {args.hold_duration:.0f}s straight)")

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
            fig, axs_ = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
            for sub_ax, (name, values, sp) in zip(axs_, axes):
                st = settle_times[name]
                sub_ax.plot(ts, values, label=f"{name}(t)")
                sub_ax.axhline(sp, color="k", ls="--", lw=1, label="setpoint")
                sub_ax.axhspan(sp - args.band, sp + args.band,
                                color="green", alpha=0.15, label=f"+/-{args.band} m band")
                if st is not None:
                    sub_ax.axvline(st, color="red", ls=":", lw=1, label=f"settled at t={st:.2f}s")
                sub_ax.set_ylabel(f"{name} (m)")
                sub_ax.set_title(name)
                sub_ax.legend(loc="lower right", fontsize=8)
            axs_[-1].set_xlabel("time (s)")
            fig.suptitle("Position / altitude hold response")
            plot_path = os.path.join(HERE, "altitude_plot.png")
            fig.tight_layout(); fig.savefig(plot_path, dpi=130)
            print(f"Saved plot to {plot_path}")
        except ImportError:
            print("matplotlib not installed -- skipping plot (pip install matplotlib)")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
