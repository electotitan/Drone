"""
Independent scoring of a run, recomputed ONLY from the raw simulation log
(t, X, Y, Z, marker_visible, z_world, armed) and the physical positions of the survivors --
it does not look at what the client or server think happened.

Rules implemented (from the task statement):
  Visit : inside the +/-0.4 box around the survivor at Z = 13 for 1 s, camera seeing the marker
  Save  : then inside the +/-0.4 box at Z = 14 for 2 s, camera seeing the marker
          (a drift out of the box or a lost marker restarts that hold from zero)
  critical survivor counts  <=> its Save is complete by t = window (55 s after lift-off)
  Switch                    =  the Save at all critical survivors is complete, or t = window
  stable survivor counts    <=> its Visit starts after the Switch (and its Save completes)
"""
import numpy as np

VISIT_Z, SAVE_Z, BOX = 13.0, 14.0, 0.4
VISIT_HOLD, SAVE_HOLD = 1.0, 2.0


def takeoff_time(log):
    armed = np.nonzero(log[:, 6] > 0.5)[0]
    return float(log[armed[0], 0]) if len(armed) else None


def earliest_hold(log, center, tol, dur, t_from):
    """Earliest time ts >= t_from such that EVERY sample in [ts, ts+dur] is inside the box and the
    marker is visible. Returns ts or None."""
    rows = log[log[:, 0] >= t_from]
    if len(rows) < 2:
        return None
    dt = rows[1, 0] - rows[0, 0]
    inb = (rows[:, 4] > 0.5) & (np.max(np.abs(rows[:, 1:4] - np.asarray(center)), axis=1) <= tol)
    n = int(round(dur / dt))
    run = 0
    ok = np.zeros(len(rows), bool)
    for i in range(len(rows) - 1, -1, -1):
        run = run + 1 if inb[i] else 0
        ok[i] = run > n
    idx = np.nonzero(ok)[0]
    return float(rows[idx[0], 0]) if len(idx) else None


def tightest_box(log, center, dur, t_from, span=8.0):
    """How well did the drone really settle? The smallest half-width (m) of a box around `center` for which a
    full-length hold exists within `span` s of t_from. The rules only need <= 0.4; smaller = more margin."""
    seg = log[(log[:, 0] >= t_from) & (log[:, 0] <= t_from + span)]
    for tol in np.arange(0.05, BOX + 1e-9, 0.025):
        if earliest_hold(seg, center, tol, dur, t_from) is not None:
            return float(round(tol, 3))
    return None


def save_event(log, xy, t_from):
    """Earliest Visit (>= t_from) followed by the earliest Save after it. Times are absolute sim seconds.
    *_box = tightest box half-width the drone actually held for that long (rules need <= 0.4)."""
    none = dict(visit_start=None, visit_end=None, save_start=None, save_end=None, visit_box=None, save_box=None)
    tv = earliest_hold(log, (xy[0], xy[1], VISIT_Z), BOX, VISIT_HOLD, t_from)
    if tv is None:
        return none
    ev = dict(none, visit_start=tv, visit_end=tv + VISIT_HOLD,
              visit_box=tightest_box(log, (xy[0], xy[1], VISIT_Z), VISIT_HOLD, tv))
    ts = earliest_hold(log, (xy[0], xy[1], SAVE_Z), BOX, SAVE_HOLD, tv + VISIT_HOLD)
    if ts is not None:
        ev.update(save_start=ts, save_end=ts + SAVE_HOLD,
                  save_box=tightest_box(log, (xy[0], xy[1], SAVE_Z), SAVE_HOLD, tv + VISIT_HOLD))
    return ev


def score_run(log, critical_xy, stable_xy, window=55.0):
    """critical_xy / stable_xy: dict cell -> (x, y). Returns a dict with everything in MISSION time (s)."""
    t0 = takeoff_time(log)
    out = dict(t_takeoff=t0, critical={}, stable={}, switch=None, switch_reason="")
    if t0 is None:
        return out
    m = lambda t: None if t is None else t - t0

    for cell, xy in critical_xy.items():
        ev = save_event(log, xy, t0)
        ok = ev["save_end"] is not None and m(ev["save_end"]) <= window
        out["critical"][cell] = dict(**{k: (v if k.endswith('_box') else m(v)) for k, v in ev.items()}, counts=bool(ok))

    done = [v["save_end"] for v in out["critical"].values() if v["counts"]]
    if critical_xy and len(done) == len(critical_xy):
        out["switch"], out["switch_reason"] = max(done), "all critical survivors saved"
    else:
        out["switch"], out["switch_reason"] = window, f"{window:g} s up"

    for cell, xy in stable_xy.items():
        ev = save_event(log, xy, t0 + out["switch"])
        ok = ev["visit_start"] is not None and ev["save_end"] is not None
        out["stable"][cell] = dict(**{k: (v if k.endswith('_box') else m(v)) for k, v in ev.items()}, counts=bool(ok))
    return out


def landing_report(log, home_xy, t_takeoff, last_save_end_abs=None):
    """Landed = motors off, resting on the floor, over F6 (inside the +/-0.4 box). Also checks the
    1 s hover at Z = 13 over F6 that must come before landing."""
    armed = log[:, 6] > 0.5
    if not len(log) or armed[-1]:
        return dict(landed=False)
    t_off = float(log[np.nonzero(armed)[0][-1], 0])
    last = log[-1]
    err = float(np.hypot(last[1] - home_xy[0], last[2] - home_xy[1]))
    t_from = last_save_end_abs if last_save_end_abs is not None else t_takeoff
    hover = earliest_hold(log[log[:, 0] <= t_off], (home_xy[0], home_xy[1], VISIT_Z), BOX, 1.0, t_from)
    return dict(landed=bool(last[5] < 0.25 and err <= BOX), touchdown_time=t_off - t_takeoff, xy_error=err,
                final_z_world=float(last[5]), hover_start=None if hover is None else hover - t_takeoff)
