#!/usr/bin/env python3
"""
Task 3 -- full rescue mission, end to end (no ROS 2).

    camera frame --/image_raw--> waypoint_service  (survivors, obstacles, critical route, stable route)
                                     ^   |
                                     |   v
                                 pico_client  --goals / cancel-->  pico_server --setpoints--> MuJoCo + LQR

    python3 run_rescue.py                                   # headless, as fast as possible
    python3 run_rescue.py --viewer                          # 3D viewer (macOS: mjpython run_rescue.py --viewer)
    python3 run_rescue.py --window 25                       # force the Switch to fire mid-goal (testing)
    python3 run_rescue.py --expect-critical I2,J9,B10 --expect-stable B2,J6,F10

Afterwards the run is scored INDEPENDENTLY from the raw simulation log. Exit code 0 = pass.
"""
import os
import sys

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"          # offscreen rendering on headless Linux (before importing mujoco)

import argparse
import json
import threading

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lattice                                                  # noqa: E402
import scoring                                                  # noqa: E402
from messages import Waypoint                                   # noqa: E402
from middleware import Publisher                                # noqa: E402
from pico_client import PicoClient                              # noqa: E402
from pico_server import PicoServer                              # noqa: E402
from rescue_sim import RescueSim, DEFAULT_MODEL, LOG_COLS       # noqa: E402
from vision import CameraModel, render_frame                    # noqa: E402
from waypoint_service import WaypointService, HOME_CELL         # noqa: E402


def ground_truth(model):
    """Physical positions straight from the scene file -- used ONLY for grading."""
    crit, stab, obst = {}, {}, []
    for i in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        p = model.geom_pos[i]
        if n.startswith("critical_"):
            crit[n.split("_")[1]] = (float(p[0]), float(p[1]))
        elif n.startswith("stable_"):
            stab[n.split("_")[1]] = (float(p[0]), float(p[1]))
        elif n.startswith("obstacle_") and model.geom_contype[i] != 0:
            obst.append((n, float(p[0]), float(p[1]), float(model.geom_size[i][0]), float(model.geom_size[i][1])))
    return crit, stab, obst


def min_obstacle_distance(log, obst, z_max=3.0):
    """Smallest distance (m) from the drone's centre to any obstacle footprint while below the obstacle top."""
    low = log[log[:, 5] < z_max]
    best = (1e9, None)
    for name, cx, cy, hx, hy in obst:
        dx = np.maximum(np.abs(low[:, 1] - cx) - hx, 0.0)
        dy = np.maximum(np.abs(low[:, 2] - cy) - hy, 0.0)
        d = np.hypot(dx, dy)
        if len(d) and d.min() < best[0]:
            best = (float(d.min()), name)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=DEFAULT_MODEL, help="MuJoCo model (see model/make_rescue_scene.py)")
    ap.add_argument("--window", type=float, default=55.0, help="critical window after lift-off (s)")
    ap.add_argument("--max-run", type=float, default=200.0, help="run cap after lift-off (s)")
    ap.add_argument("--margin", type=float, default=0.85, help="obstacle clearance used by the planner (m)")
    ap.add_argument("--speed-marks", type=float, default=110.0, help="landing deadline for full speed marks (s)")
    ap.add_argument("--expect-critical", default="", help="optional: cells the scan must report as critical")
    ap.add_argument("--expect-stable", default="", help="optional: cells the scan must report as stable")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--speed", type=float, default=None, help="sim speed factor (1 = real time, 0 = max)")
    ap.add_argument("--scan-size", type=int, default=960)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    speed = args.speed if args.speed is not None else (1.0 if (args.viewer or args.realtime) else 0.0)
    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------ world + ONE camera frame
    sim = RescueSim(args.scene, speed=speed, max_run=args.max_run)
    cam = CameraModel(sim.model, "top_cam", args.scan_size)
    frame = render_frame(sim.model, sim.data, "top_cam", args.scan_size)     # main thread (GL context)
    Publisher("/image_raw").publish(frame)
    print(f"[camera] published 1 frame on /image_raw ({args.scan_size}x{args.scan_size})")

    # ------------------------------------------------------------------ the three programs
    service = WaypointService(cam, margin=args.margin)
    server = PicoServer(sim)
    client = PicoClient(critical_window=args.window)
    def client_main():
        try:
            client.run()
        except Exception as exc:                     # e.g. a service failed: report it, never leave the sim spinning
            print(f"[pico_client] mission aborted: {exc}")
        finally:
            sim.stop()
    th = threading.Thread(target=client_main, name="pico_client", daemon=True)
    th.start()
    try:
        sim.run(viewer=args.viewer)
    except KeyboardInterrupt:
        sim.stop()
    th.join(timeout=5.0)
    server.shutdown(); service.shutdown()

    # ------------------------------------------------------------------ verification
    log = sim.full_log()
    np.savetxt(os.path.join(args.out, "rescue_log.csv"), log, delimiter=",", header=LOG_COLS, comments="",
               fmt=["%.3f", "%.4f", "%.4f", "%.4f", "%d", "%.4f", "%d"])
    crit_truth, stab_truth, obst_truth = ground_truth(sim.model)
    home = lattice.cell_to_world(HOME_CELL)
    sc = service.scene
    checks = []
    def check(ok, text):
        checks.append(bool(ok))
        print(f"  {text}   [{'PASS' if ok else 'FAIL'}]")

    print("\n=== Verification (independent of the client / server) ===")
    print("-- scan (one frame) vs the physical scene")
    found_c, found_s = {s.cell for s in sc.critical}, {s.cell for s in sc.stable}
    check(found_c == set(crit_truth), f"critical survivors found {sorted(found_c)}  truth {sorted(crit_truth)}")
    check(found_s == set(stab_truth), f"stable survivors found   {sorted(found_s)}  truth {sorted(stab_truth)}")
    check(len(sc.obstacle_info) == len(obst_truth), f"obstacles found {len(sc.obstacle_info)}  truth {len(obst_truth)}")
    worst = max([np.hypot(s.x - truth[s.cell][0], s.y - truth[s.cell][1])
                 for ph, truth in (("critical", crit_truth), ("stable", stab_truth))
                 for s in sc.survivors if s.phase == ph and s.cell in truth] or [0.0])
    check(worst < 0.10, f"survivor localisation: worst error {worst * 100:.1f} cm (< 10 cm)")
    if args.expect_critical:
        exp = set(c.strip().upper() for c in args.expect_critical.split(","))
        check(found_c == exp, f"critical set matches the reference map {sorted(exp)}")
    if args.expect_stable:
        exp = set(c.strip().upper() for c in args.expect_stable.split(","))
        check(found_s == exp, f"stable set matches the reference map {sorted(exp)}")

    print("-- safety")
    check(not sim.collisions, f"no contact with any obstacle {'' if not sim.collisions else sorted(sim.collisions)[:4]}")
    dmin, dname = min_obstacle_distance(log, obst_truth)
    print(f"  closest approach to an obstacle footprint (centre of drone): {dmin:.2f} m ({dname}); drone half-width 0.235 m")
    flying = log[(log[:, 6] > 0.5)]
    lost = float((flying[:, 4] < 0.5).mean()) if len(flying) else 0.0
    print(f"  camera lost the marker during {lost * 100:.1f}% of the flight")

    print("-- scoring from the raw log")
    S = scoring.score_run(log, crit_truth, stab_truth, window=args.window)
    t0 = S["t_takeoff"]
    check(t0 is not None, f"lift-off at sim time {t0:.2f} s  (mission t = 0)" if t0 is not None else "no lift-off")
    if t0 is None:
        print("\n  OVERALL RESULT: FAIL"); sys.exit(1)
    fmt = lambda v: "   -  " if v is None else f"{v:6.2f}"
    print(f"\n  {'survivor':9s} {'phase':9s} {'visit (Z=13, 1 s)':>20s} {'save (Z=14, 2 s)':>20s}   settled to (box)   counts")
    errs = lambda v: f"+/-{'-' if v['visit_box'] is None else format(v['visit_box'], '.2f')} / +/-{'-' if v['save_box'] is None else format(v['save_box'], '.2f')}"
    n_crit = n_stab = 0
    for cell, v in S["critical"].items():
        n_crit += v["counts"]
        print(f"  {cell:9s} {'critical':9s} {fmt(v['visit_start'])} -> {fmt(v['visit_end'])}   {fmt(v['save_start'])} -> {fmt(v['save_end'])}   {errs(v):>12s}       "
              f"{'YES' if v['counts'] else 'no'}{'' if v['counts'] or v['save_end'] is None else f'  (save ended after {args.window:.0f} s)'}")
    print(f"  --- THE SWITCH at t = {S['switch']:.2f} s ({S['switch_reason']}) ---")
    for cell, v in S["stable"].items():
        n_stab += v["counts"]
        print(f"  {cell:9s} {'stable':9s} {fmt(v['visit_start'])} -> {fmt(v['visit_end'])}   {fmt(v['save_start'])} -> {fmt(v['save_end'])}   {errs(v):>12s}       "
              f"{'YES' if v['counts'] else 'no'}")
    last_save = max([v["save_end"] for v in list(S["critical"].values()) + list(S["stable"].values())
                     if v["save_end"] is not None] or [0.0])
    L = scoring.landing_report(log, home, t0, t0 + last_save)
    if L.get("landed"):
        print(f"\n  landed on F6 at t = {L['touchdown_time']:.2f} s, {L['xy_error'] * 100:.0f} cm from F6 "
              f"(hover over F6 started t = {L['hover_start'] if L['hover_start'] is None else round(L['hover_start'], 2)} s)")
    else:
        print("\n  did NOT land on F6")

    print("-- the client's view agrees with the log")
    log_saved = {c for c, v in S["critical"].items() if v["counts"]}
    check(set(client.saved_critical) <= log_saved,
          f"critical saved: client {sorted(client.saved_critical)}  log {sorted(log_saved)}  "
          f"(client never claims a survivor the log does not support)")
    if set(client.saved_critical) != log_saved:
        print(f"  note: the log credits {sorted(log_saved - set(client.saved_critical))} that the client (tighter hold box) did not count")
    if client.switch_time is not None:
        lag = client.switch_time - S["switch"]
        same_reason = client.switch_reason.split()[0] == S["switch_reason"].split()[0]
        ok = same_reason and (abs(lag) < 1e-6 if "up" in S["switch_reason"] else -0.02 <= lag <= 0.5)
        check(ok, f"Switch: client t = {client.switch_time:.2f} s ({client.switch_reason})  log t = {S['switch']:.2f} s "
                  f"({S['switch_reason']})  -> client {lag:+.2f} s (may lag by <= 0.5 s: it holds in a tighter box)")
    check(client.landed == bool(L.get("landed")), f"landing: client {client.landed}  log {bool(L.get('landed'))}")
    if L.get("hover_start") is not None or L.get("landed"):
        check(L.get("hover_start") is not None, "1 s hover at Z = 13 over F6 before landing")

    require_all = args.window >= 55.0
    all_saved = (n_crit == len(crit_truth)) and (n_stab == len(stab_truth))
    in_time = bool(L.get("landed")) and L["touchdown_time"] <= args.max_run
    if require_all:
        check(n_crit == len(crit_truth), f"all {len(crit_truth)} critical survivors saved by t = {args.window:.0f} s ({n_crit}/{len(crit_truth)})")
        check(n_stab == len(stab_truth), f"all {len(stab_truth)} stable survivors saved after the Switch ({n_stab}/{len(stab_truth)})")
    check(in_time, f"landed within the {args.max_run:.0f} s run cap")
    fast = in_time and L["touchdown_time"] <= args.speed_marks
    print(f"\n  survivors saved: {n_crit + n_stab}/{len(crit_truth) + len(stab_truth)}  "
          f"(critical {n_crit}/{len(crit_truth)}, stable {n_stab}/{len(stab_truth)})")
    if L.get("landed"):
        print(f"  landing time   : {L['touchdown_time']:.1f} s  -> {'FULL speed marks' if fast else 'no speed marks'} (<= {args.speed_marks:.0f} s)")
    overall = bool(all(checks) and client.ok)
    print(f"\n  OVERALL RESULT: {'PASS' if overall else 'FAIL'}" +
          ("" if require_all else f"   (reduced --window {args.window:.0f}: only consistency, safety and landing are required)"))

    with open(os.path.join(args.out, "rescue_result.json"), "w") as f:
        json.dump(dict(window=args.window, switch=S["switch"], switch_reason=S["switch_reason"], critical=S["critical"],
                       stable=S["stable"], landing=L, saved=n_crit + n_stab, overall=overall,
                       client=dict(saved_critical=client.saved_critical, switch_time=client.switch_time,
                                   switch_reason=client.switch_reason, switch_position=client.switch_position,
                                   cancelled_goal=client.switch_cancelled_goal)), f, indent=2, default=float)

    if not args.no_plot:
        try:
            make_plots(args.out, sim, cam, sc, service, client, log, S, L, t0, home)
        except ImportError:
            print("  matplotlib missing -- skipped plots")
    sys.exit(0 if overall else 1)


def make_plots(out, sim, cam, sc, service, client, log, S, L, t0, home):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = service.planner
    half = cam.size / 2 / cam.scale
    xs = np.linspace(-P.bound, P.bound, P.n)
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.imshow(sc.frame, extent=[-half, half, -half, half])
    ax.contourf(xs, xs, (~P.free).astype(float), levels=[0.5, 1.5], colors=["red"], alpha=0.28, zorder=2)
    ax.contour(xs, xs, P.free.astype(float), levels=[0.5], colors=["red"], linewidths=1.2, zorder=3)
    tsw = t0 + S["switch"]
    c_part = log[(log[:, 6] > 0.5) & (log[:, 0] <= tsw)]
    s_part = log[(log[:, 6] > 0.5) & (log[:, 0] > tsw)]
    ax.plot(c_part[:, 1], c_part[:, 2], color="#ff4040", lw=2.2, label="critical phase", zorder=5)
    ax.plot(s_part[:, 1], s_part[:, 2], color="#ffe000", lw=2.2, label="stable phase + home", zorder=5)
    for ph, col in (("critical", "#ff4040"), ("stable", "#ffe000")):
        for o in client.outcomes:
            w = o.waypoint
            if o.phase == ph and w.kind == "via":
                ax.plot(w.x, w.y, ".", color="white", ms=5, zorder=6)
    for s in sc.survivors:
        ax.add_patch(plt.Circle((s.x, s.y), 0.55, fill=False, ec="white", lw=2, zorder=6))
        ax.annotate(f"{s.cell}\n{s.phase}", (s.x + 0.35, s.y + 0.35), color="white", fontsize=10, fontweight="bold", zorder=7)
    ax.plot(*home, "o", color="#3fa0ff", ms=9, zorder=7); ax.annotate("F6 take-off/landing", (home[0] + 0.3, home[1] - 0.6), color="#9fd0ff", fontsize=10, zorder=7)
    ax.set_xlim(-6.8, 6.8); ax.set_ylim(-6.8, 6.8); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="lower left"); ax.set_title("Rescue mission: flown path (red band = forbidden zone the planner used)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "rescue_map.png"), dpi=105); plt.close(fig)

    # ---- timeline
    fly = log[log[:, 6] > 0.5]
    tm = fly[:, 0] - t0
    fig, axs = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    for a, col, nm in zip(axs, (1, 2, 3), ("X (m)", "Y (m)", "Z  (down from camera)")):
        a.plot(tm, fly[:, col], lw=1.3)
        a.axvspan(0, S["switch"], color="red", alpha=0.07)
        a.axvspan(S["switch"], tm[-1], color="gold", alpha=0.07)
        a.axvline(S["switch"], color="k", ls="--", lw=1)
        a.set_ylabel(nm); a.grid(alpha=0.3)
    axs[2].invert_yaxis()
    axs[2].axhline(13, color="gray", ls=":", lw=1); axs[2].axhline(14, color="gray", ls=":", lw=1)
    for ph, d in (("critical", S["critical"]), ("stable", S["stable"])):
        for cell, v in d.items():
            if v["visit_start"] is not None:
                axs[2].axvspan(v["visit_start"], v["visit_end"], color="limegreen", alpha=0.4)
            if v["save_start"] is not None:
                axs[2].axvspan(v["save_start"], v["save_end"], color="darkorange", alpha=0.4)
                axs[2].annotate(cell, (v["save_start"], 14.55), fontsize=9, ha="left")
    lostm = tm[fly[:, 4] < 0.5]
    if len(lostm):
        axs[2].plot(lostm, np.full(len(lostm), 12.55), "r|", label="marker lost")
    if L.get("landed"):
        for a in axs:
            a.axvline(L["touchdown_time"], color="green", lw=1.2)
    axs[0].set_title(f"Timeline: red = critical phase, yellow = stable/home; dashed = Switch ({S['switch']:.1f} s); "
                     f"green = Visit, orange = Save, green line = touchdown")
    axs[2].set_xlabel("mission time since lift-off (s)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "rescue_timeline.png"), dpi=105); plt.close(fig)
    print(f"  plots -> {os.path.join(out, 'rescue_map.png')}, rescue_timeline.png")


if __name__ == "__main__":
    main()
