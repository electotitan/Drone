#!/usr/bin/env python3
"""
Scan-and-navigate mission (no ROS 2).

    1. SCAN     top camera frame -> detect composite markers (red triangle + yellow circle)
    2. PLAN     order the markers, route between them over the lattice (the dotted route in the picture)
    3. SERVE    waypoint_service returns that plan
    4. FLY      MissionClient -> NavigateServer -> LQR controller in MuJoCo, one goal at a time
    5. VERIFY   everything re-checked from the raw simulation log + the physical scene definition

    python3 run_mission.py                         # headless, fast
    python3 run_mission.py --viewer                # 3D viewer (macOS: mjpython run_mission.py --viewer)
    python3 run_mission.py --order clockwise       # other visiting-order rule
    python3 run_mission.py --route direct          # fly marker -> marker, skip the lattice via-points
    python3 run_mission.py --expect C9,D5,H4,J7    # also assert the visiting order of the reference picture

Exit code 0 = all checks pass, 1 = failure.
"""
import os
import sys

# Offscreen rendering for the scan: use EGL on headless Linux (must be set before importing mujoco)
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lattice                                                   # noqa: E402
from sim_node import SimNode                                     # noqa: E402
from arena_scan import scan_arena, CameraModel                   # noqa: E402
from route_planner import order_markers, plan_lattice_route      # noqa: E402
from messages import Waypoint                                    # noqa: E402
from waypoint_service import make_waypoint_service               # noqa: E402
from action_server import NavigateServer                         # noqa: E402
from action_client import MissionClient                          # noqa: E402
from middleware import GoalStatus                                # noqa: E402
import mujoco                                                    # noqa: E402

DEFAULT_SCENE = os.path.join(HERE, "..", "model", "drone_scan.xml")


def build_waypoints(start_cell, ordered_cells, altitude, route, start_hold, marker_hold):
    """start -> [via ...] -> marker 1 -> [via ...] -> marker 2 ... (flight altitude for all)."""
    def wp(name, cell, kind, hold):
        x, y = lattice.cell_to_world(cell)
        return Waypoint(name, round(x, 3), round(y, 3), altitude, kind=kind, cell=cell, hold=hold)

    wps = [wp(f"START {start_cell}", start_cell, "start", start_hold)]
    prev = start_cell
    for i, cell in enumerate(ordered_cells, 1):
        if route == "lattice":
            wps += [wp(f"via {c}", c, "via", 0.0) for c in plan_lattice_route(prev, cell)]
        wps.append(wp(f"{i}: {cell}", cell, "marker", marker_hold))
        prev = cell
    return wps


def first_stable_time(log, target, t_from, band, hold):
    """Earliest t >= t_from such that every sample in [t, t+hold] is inside the band."""
    rows = log[log[:, 0] >= t_from]
    inb = np.max(np.abs(rows[:, 1:4] - target), axis=1) <= band
    n_hold = int(round(hold / (rows[1, 0] - rows[0, 0])))
    run = 0
    for i in range(len(rows) - 1, -1, -1):
        run = run + 1 if inb[i] else 0
        inb[i] = run > n_hold
    ok = np.where(inb)[0]
    return rows[ok[0], 0] if len(ok) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=DEFAULT_SCENE, help="MuJoCo model with the markers (see model/make_scene.py)")
    ap.add_argument("--altitude", type=float, default=13.0, help="flight altitude for every waypoint (m)")
    ap.add_argument("--order", default="left-to-right", choices=["left-to-right", "clockwise", "nearest"])
    ap.add_argument("--route", default="lattice", choices=["lattice", "direct"])
    ap.add_argument("--expect", default="", help="optional: comma-separated cell sequence the scan must produce")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--speed", type=float, default=None, help="sim speed factor (1 = real time, 0 = max)")
    ap.add_argument("--band", type=float, default=0.4, help="+/- band on X, Y, Z")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold at each MARKER")
    ap.add_argument("--start-hold", type=float, default=1.0, help="seconds to hold above the start cell after take-off")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-waypoint sim-time timeout (s)")
    ap.add_argument("--scan-size", type=int, default=960, help="camera frame size (px)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    speed = args.speed if args.speed is not None else (1.0 if (args.viewer or args.realtime) else 0.0)
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------ 1. SCAN (main thread: GL context)
    sim = SimNode(model_path=args.scene, speed=speed)
    cam = CameraModel(sim.model, "top_cam", args.scan_size)
    scan = scan_arena(sim.model, sim.data, "top_cam", args.scan_size)
    spawn = sim.data.qpos[0:3]
    s_col, s_row, _ = lattice.world_to_idx(spawn[0], spawn[1])
    start_cell = lattice.label(s_col, s_row)

    print("=== Arena scan (top camera, %dx%d px, %.1f px/m) ===" % (args.scan_size, args.scan_size, cam.scale))
    print(f"  drone start cell : {start_cell}")
    for d in scan.detections:
        print(f"  {d.kind:14s} cell {d.cell:4s} measured ({d.meas_x:+.3f}, {d.meas_y:+.3f}) m  "
              f"-> lattice ({d.x:+.3f}, {d.y:+.3f})  snap error {d.snap_err * 100:.1f} cm")
    target_cells = [d.cell for d in scan.targets]
    if not target_cells:
        print("  no target markers found -- nothing to do"); sys.exit(1)

    # ------------------------------------------------------------------ 2. PLAN
    ordered = order_markers(target_cells, start_cell, args.order)
    waypoints = build_waypoints(start_cell, ordered, args.altitude, args.route, args.start_hold, args.hold)
    print(f"\n  visiting order ({args.order}): " + " -> ".join(f"{i}: {c}" for i, c in enumerate(ordered, 1)))
    print(f"  plan ({args.route}): " + " > ".join(w.cell for w in waypoints) + f"   [{len(waypoints)} waypoints]\n")

    # ------------------------------------------------------------------ 3/4. SERVE + FLY
    wp_service = make_waypoint_service(waypoints)
    nav_server = NavigateServer(sim, band=args.band, hold_duration=args.hold, timeout=args.timeout)
    client = MissionClient()
    mission_ok = {"value": False}

    def client_thread():
        try:
            mission_ok["value"] = client.run()
        finally:
            sim.stop()

    th = threading.Thread(target=client_thread, name="mission-client", daemon=True)
    th.start()
    try:
        sim.run(viewer=args.viewer)
    except KeyboardInterrupt:
        sim.stop()
    th.join(timeout=5.0)
    nav_server.shutdown(); wp_service.shutdown()

    # ------------------------------------------------------------------ 5. VERIFY
    log = sim.full_log()
    np.savetxt(os.path.join(args.out, "mission_log.csv"), log, delimiter=",", header="t,x,y,z", comments="",
               fmt=["%.3f", "%.5f", "%.5f", "%.5f"])
    with open(os.path.join(args.out, "scan_result.json"), "w") as f:
        json.dump({"start_cell": start_cell, "order": ordered,
                   "detections": [dict(cell=d.cell, kind=d.kind, x=d.x, y=d.y, meas_x=d.meas_x, meas_y=d.meas_y,
                                       snap_err=d.snap_err) for d in scan.detections],
                   "waypoints": [dict(name=w.name, cell=w.cell, kind=w.kind, x=w.x, y=w.y, z=w.z, hold=w.hold)
                                 for w in waypoints]}, f, indent=2)

    print("\n=== Verification ===")
    checks = []

    # (a) scanner vs the physical scene definition (ground truth, used ONLY for grading)
    truth_t, truth_decoy = set(), set()
    for i in range(sim.model.ngeom):
        n = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if n.startswith("target_") and n.endswith("_circle"):
            truth_t.add(n.split("_")[1])
        if n.startswith("decoy_"):
            truth_decoy.add(n.split("_")[1])
    found = set(target_cells)
    ok = found == truth_t
    checks.append(ok)
    print(f"  scan vs scene     : found {sorted(found)}  truth {sorted(truth_t)}   [{'PASS' if ok else 'FAIL'}]")
    worst = max((d.snap_err for d in scan.targets), default=0.0)
    ok = worst < 0.10
    checks.append(ok)
    print(f"  localisation      : worst pixel->world error {worst * 100:.1f} cm (< 10 cm)   [{'PASS' if ok else 'FAIL'}]")
    if truth_decoy:
        wrongly = found & truth_decoy
        ok = not wrongly
        checks.append(ok)
        print(f"  decoys ignored    : {sorted(truth_decoy)} not treated as targets   [{'PASS' if ok else 'FAIL'}]")
    if args.expect:
        exp = [c.strip().upper() for c in args.expect.split(",")]
        ok = ordered == exp
        checks.append(ok)
        print(f"  reference order   : {' -> '.join(ordered)}  expected {' -> '.join(exp)}   [{'PASS' if ok else 'FAIL'}]")

    # (b) visit order == plan
    visited = [o.waypoint.name for o in client.outcomes]
    planned = [w.name for w in waypoints]
    ok = visited == planned and [w.name for w in client.requested_order] == planned
    checks.append(ok)
    print(f"  visit order       : {len(visited)}/{len(planned)} waypoints visited in plan order   [{'PASS' if ok else 'FAIL'}]")

    # (c) per-waypoint flight checks from the raw log
    print(f"\n  {'waypoint':12s} {'target (x, y, z)':22s} {'kind':6s} {'hold':>5s} {'settle (srv)':>12s} {'settle (log)':>12s}  max|err| during hold    result")
    windows = []
    n_via_ok = 0
    for o in client.outcomes:
        r, w = o.result, o.waypoint
        hold = args.hold if w.hold is None else w.hold
        if r is None or o.status != GoalStatus.SUCCEEDED:
            print(f"  {w.name:12s} goal ended with status {o.status}   [FAIL]"); checks.append(False); continue
        target = np.array(w.xyz)
        t_log = first_stable_time(log, target, r.start_time, args.band, hold)
        if t_log is None:
            print(f"  {w.name:12s} never stable in log   [FAIL]"); checks.append(False); continue
        sel = (log[:, 0] >= t_log) & (log[:, 0] <= t_log + hold)
        max_err = np.max(np.abs(log[sel, 1:4] - target), axis=0)
        agree = abs((t_log - r.start_time) - r.settle_time) < 0.02
        ok = bool(max_err.max() <= args.band and agree and r.success)
        checks.append(ok)
        windows.append((w, r.start_time, t_log, t_log + hold))
        if w.kind == "via":
            n_via_ok += ok
            continue
        print(f"  {w.name:12s} ({w.x:+.2f},{w.y:+.2f},{w.z:+.2f})  {w.kind:6s} {hold:4.1f}s {r.settle_time:10.2f} s {t_log - r.start_time:10.2f} s  "
              f"[{max_err[0]:.3f} {max_err[1]:.3f} {max_err[2]:.3f}]   {'PASS' if ok else 'FAIL'}")
    n_via = sum(1 for w in waypoints if w.kind == "via")
    if n_via:
        print(f"  via points        : {n_via_ok}/{n_via} flown through (within +/-{args.band} on X, Y, Z)")

    has_obst = bool(sim._obstacle_geoms)
    if has_obst:
        ok = not sim.collisions
        checks.append(ok)
        print(f"\n  obstacle contacts : {'none' if ok else sorted(sim.collisions)[:5]}   [{'PASS' if ok else 'FAIL'}]")
    total = log[-1, 0] if len(log) else 0.0
    overall = bool(all(checks) and mission_ok["value"] and len(client.outcomes) == len(waypoints))
    print(f"  mission time      : {total:.1f} s simulated, {len(log)} samples")
    print(f"\n  OVERALL RESULT: {'PASS' if overall else 'FAIL'}")

    # ------------------------------------------------------------------ plots
    if not args.no_plot and len(log):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # --- annotated scan frame with plan and flown path (compare with the reference picture)
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.imshow(scan.frame)
            pts = np.array([cam.world_to_pixel(w.x, w.y) for w in waypoints])
            ax.plot(pts[:, 0], pts[:, 1], "-", color="orange", lw=3, label="planned route", zorder=3)
            for w, (u, v) in zip(waypoints, pts):
                if w.kind == "via":
                    ax.plot(u, v, "o", color="orange", ms=6, zorder=4)
            fu, fv = zip(*[cam.world_to_pixel(x, y) for x, y in log[::10, 1:3]])
            ax.plot(fu, fv, "-", color="cyan", lw=1.5, label="flown path", zorder=5)
            for d in scan.detections:
                col = "yellow" if d.kind == "target" else "white"
                ax.add_patch(plt.Circle(d.px, 0.45 * cam.scale, fill=False, ec=col, lw=2, zorder=6))
            for w, (u, v) in zip(waypoints, pts):
                if w.kind == "marker":
                    ax.annotate(w.name, (u + 18, v - 22), color="yellow", fontsize=13, fontweight="bold", zorder=7)
                elif w.kind == "start":
                    ax.annotate(w.name.upper(), (u + 14, v - 14), color="white", fontsize=11, fontweight="bold", zorder=7)
            ax.set_axis_off(); ax.legend(loc="lower left")
            ax.set_title(f"Scan result: {' -> '.join(ordered)}")
            fig.tight_layout(); fig.savefig(os.path.join(args.out, "scan_result.png"), dpi=110); plt.close(fig)

            # --- time series
            fig = plt.figure(figsize=(12, 9))
            gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1])
            axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
            for ax_, col, name in zip(axs, (1, 2, 3), ("X", "Y", "Z")):
                ax_.plot(log[:, 0], log[:, col], lw=1.3)
                for w, t0, ts, te in windows:
                    if w.kind == "via":
                        continue
                    v = w.xyz[col - 1]
                    ax_.hlines(v, t0, te, colors="k", linestyles="--", lw=0.8)
                    ax_.fill_between([t0, te], v - args.band, v + args.band, color="green", alpha=0.12)
                    ax_.axvspan(ts, te, color="orange", alpha=0.25)
                    if col == 3:
                        ax_.annotate(w.name, (ts, v + args.band + 0.3), fontsize=8)
                ax_.set_ylabel(f"{name} (m)"); ax_.grid(alpha=0.3)
            axs[-1].set_xlabel("sim time (s)")
            axs[0].set_title("Position vs time  (orange = confirmed hold at a marker / start)")
            axm = fig.add_subplot(gs[:, 1])
            axm.plot(log[:, 1], log[:, 2], lw=1.2)
            for w in waypoints:
                if w.kind == "via":
                    axm.plot(w.x, w.y, ".", color="gray")
                else:
                    axm.add_patch(plt.Rectangle((w.x - args.band, w.y - args.band), 2 * args.band, 2 * args.band, fill=False, ec="green"))
                    axm.annotate(w.name, (w.x + 0.15, w.y + 0.15), fontsize=9)
            axm.set_aspect("equal"); axm.grid(alpha=0.3); axm.set_xlabel("x (m)"); axm.set_ylabel("y (m)")
            axm.set_title("Top-down path")
            fig.tight_layout(); fig.savefig(os.path.join(args.out, "mission_plot.png"), dpi=115); plt.close(fig)
            print(f"  plots -> {os.path.join(args.out, 'scan_result.png')}, mission_plot.png")
        except ImportError:
            print("  matplotlib missing -- skipped plots")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
