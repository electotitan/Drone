#!/usr/bin/env python3
"""
Full waypoint-navigation system (no ROS 2):

    waypoint_service  <--call--  MissionClient  --goal-->  NavigateServer  --setpoint-->  SimNode (MuJoCo + LQR)
                                      ^                          |
                                      +------ feedback/result ---+

Usage:
    python3 run_mission.py                  # headless, as fast as possible
    python3 run_mission.py --realtime       # headless, real-time pace
    python3 run_mission.py --viewer         # interactive 3D view (real time)
                                            #   (macOS: run with `mjpython run_mission.py --viewer`)

Afterwards it INDEPENDENTLY re-checks the raw simulation log (not the server's
word): visit order, and +/-band on X, Y, Z for >= hold seconds at every waypoint.
Exit code 0 = all checks pass, 1 = failure.
"""
import argparse
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sim_node import SimNode                                   # noqa: E402
from waypoint_service import make_waypoint_service, WAYPOINTS  # noqa: E402
from action_server import NavigateServer                       # noqa: E402
from action_client import MissionClient                        # noqa: E402
from middleware import GoalStatus                              # noqa: E402


def first_stable_time(log, target, t_from, band, hold):
    """Earliest t >= t_from such that every sample in [t, t+hold] is inside the band."""
    rows = log[log[:, 0] >= t_from]
    inb = np.max(np.abs(rows[:, 1:4] - target), axis=1) <= band
    n_hold = int(round(hold / (rows[1, 0] - rows[0, 0])))
    run = 0
    for i in range(len(rows) - 1, -1, -1):                     # run-length of True going forward
        run = run + 1 if inb[i] else 0
        inb[i] = run > n_hold
    ok = np.where(inb)[0]
    return rows[ok[0], 0] if len(ok) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", action="store_true", help="open the interactive MuJoCo viewer (real time)")
    ap.add_argument("--realtime", action="store_true", help="pace the headless sim at real time")
    ap.add_argument("--speed", type=float, default=None, help="sim speed factor (1 = real time, 0 = max)")
    ap.add_argument("--band", type=float, default=0.4, help="+/- band on X, Y, Z (WhyCode units)")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds the drone must stay inside the band")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-waypoint sim-time timeout (s)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=HERE, help="where to write mission_log.csv / mission_plot.png")
    args = ap.parse_args()

    speed = args.speed if args.speed is not None else (1.0 if (args.viewer or args.realtime) else 0.0)

    sim = SimNode(speed=speed)
    wp_service = make_waypoint_service()
    nav_server = NavigateServer(sim, band=args.band, hold_duration=args.hold, timeout=args.timeout)
    client = MissionClient()

    mission_ok = {"value": False}

    def client_thread():
        try:
            mission_ok["value"] = client.run()
        finally:
            sim.stop()                                          # mission over -> end the physics loop

    t = threading.Thread(target=client_thread, name="mission-client", daemon=True)
    t.start()
    try:
        sim.run(viewer=args.viewer)                             # blocks in the main thread
    except KeyboardInterrupt:
        sim.stop()
    t.join(timeout=5.0)
    nav_server.shutdown(); wp_service.shutdown()

    # ------------------------------------------------------------------ independent verification
    log = sim.full_log()
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "mission_log.csv")
    np.savetxt(csv_path, log, delimiter=",", header="t,x,y,z", comments="", fmt=["%.3f", "%.5f", "%.5f", "%.5f"])

    print("\n=== Mission verification (recomputed from the raw simulation log) ===")
    print(f"  band +/-{args.band} on X, Y, Z; hold >= {args.hold:.1f} s\n")
    checks = []

    expected = [w.name for w in WAYPOINTS]
    visited = [o.waypoint.name for o in client.outcomes]
    order_ok = (visited == expected[:len(visited)]) and len(visited) == len(expected) \
        and [w.name for w in client.requested_order] == expected
    checks.append(order_ok)
    print(f"  visit order      : {' -> '.join(visited) or '(none)'}   [{'PASS' if order_ok else 'FAIL'}]  (expected {' -> '.join(expected)})")

    print(f"\n  {'wp':4s} {'target (x, y, z)':22s} {'settle (server)':>15s} {'settle (log)':>13s} {'held for':>9s} {'max |err| in hold':>20s}  result")
    windows = []
    for o in client.outcomes:
        r, wp = o.result, o.waypoint
        if r is None or o.status != GoalStatus.SUCCEEDED:
            print(f"  {wp.name:4s} {str(wp.xyz):22s}  goal ended with status {o.status}  [FAIL]")
            checks.append(False); continue
        target = np.array(wp.xyz)
        t_log = first_stable_time(log, target, r.start_time, args.band, args.hold)
        if t_log is None:
            print(f"  {wp.name:4s} {str(wp.xyz):22s}  never stable in log  [FAIL]"); checks.append(False); continue
        sel = (log[:, 0] >= t_log) & (log[:, 0] <= t_log + args.hold)
        max_err = np.max(np.abs(log[sel, 1:4] - target), axis=0)
        agree = abs((t_log - r.start_time) - r.settle_time) < 0.02
        ok = bool(max_err.max() <= args.band and agree and r.success)
        checks.append(ok)
        windows.append((wp, r.start_time, t_log, t_log + args.hold))
        print(f"  {wp.name:4s} ({wp.x:+.2f},{wp.y:+.2f},{wp.z:+.2f})  {r.settle_time:12.2f} s {t_log - r.start_time:10.2f} s {args.hold:7.1f} s "
              f"  [{max_err[0]:.3f} {max_err[1]:.3f} {max_err[2]:.3f}]  {'PASS' if ok else 'FAIL'}")

    no_crash = len(sim.collisions) == 0
    checks.append(no_crash)
    print(f"\n  obstacle contacts: {'none' if no_crash else sorted(sim.collisions)[:5]}   [{'PASS' if no_crash else 'FAIL'}]")
    total = log[-1, 0] if len(log) else 0.0
    overall = bool(all(checks) and mission_ok["value"] and len(client.outcomes) == len(expected))
    print(f"  mission time     : {total:.1f} s of simulated flight, {len(log)} samples -> {csv_path}")
    print(f"\n  OVERALL RESULT: {'PASS' if overall else 'FAIL'}")

    # ------------------------------------------------------------------ plot
    if not args.no_plot and len(log):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 9))
            gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1])
            axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
            for ax, col, name in zip(axs, (1, 2, 3), ("X", "Y", "Z")):
                ax.plot(log[:, 0], log[:, col], lw=1.4)
                for wp, t0, ts, te in windows:
                    v = wp.xyz[col - 1]
                    ax.hlines(v, t0, te, colors="k", linestyles="--", lw=0.8)
                    ax.fill_between([t0, te], v - args.band, v + args.band, color="green", alpha=0.12)
                    ax.axvspan(ts, te, color="orange", alpha=0.25)
                    if col == 3:
                        ax.annotate(wp.name, (ts, v + args.band + 0.3), fontsize=9)
                ax.set_ylabel(f"{name} (m)"); ax.grid(alpha=0.3)
            axs[-1].set_xlabel("sim time (s)")
            axs[0].set_title("Position vs time  (green = +/-band, orange = confirmed hold window)")
            axm = fig.add_subplot(gs[:, 1])
            axm.plot(log[:, 1], log[:, 2], lw=1.2)
            for wp, *_ in windows:
                axm.add_patch(plt.Rectangle((wp.x - args.band, wp.y - args.band), 2 * args.band, 2 * args.band,
                                            fill=False, ec="green"))
                axm.plot(wp.x, wp.y, "r+"); axm.annotate(wp.name, (wp.x + 0.15, wp.y + 0.15))
            axm.plot(0, 0, "ko", label="spawn"); axm.set_aspect("equal"); axm.grid(alpha=0.3)
            axm.set_xlabel("x (m)"); axm.set_ylabel("y (m)"); axm.set_title("Top-down path"); axm.legend()
            fig.tight_layout()
            p = os.path.join(args.out, "mission_plot.png")
            fig.savefig(p, dpi=120); print(f"  plot -> {p}")
        except ImportError:
            print("  matplotlib missing -- skipped plot")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
