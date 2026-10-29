"""
pico_client (program 3 of 3)

  Critical phase   call get_waypoints once, send the goals one at a time. The mission clock comes
                   from the server (t = 0 at lift-off) via its feedback. After EVERY goal, and on
                   every feedback, it asks: all critical survivors saved, or 55 s up?
  The Switch       if the 55 s run out in the middle of a goal -> CANCEL that goal (the server holds
                   in place and returns the drone's position). If all critical survivors were saved
                   earlier, the last result already carries the position.
  Stable phase     call get_stable_route with that position, send those goals one at a time,
                   ending with the `home` goal (hover over F6, then the server lands).

A critical survivor counts only if its Save finished by t = window (55 s): the client trusts the
server's `end_time` of the Save goal, not its own polling.
A goal that fails (timeout, marker lost for too long) skips the rest of that survivor's goals; a failed
fly-past or home goal aborts the mission.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

from messages import GetWaypointsRequest, GetStableRouteRequest, NavGoal, NavFeedback, NavResult, Waypoint
from middleware import ActionClient, ServiceClient, GoalStatus
from pico_server import ACTION_NAME


@dataclass
class Outcome:
    phase: str
    waypoint: Waypoint
    status: str
    result: Optional[NavResult]


class PicoClient:
    def __init__(self, critical_window=55.0, log=print, result_timeout=900.0):
        self.window = critical_window
        self.log = log
        self.result_timeout = result_timeout                       # wall seconds
        self.route_srv = ServiceClient("get_waypoints")
        self.stable_srv = ServiceClient("get_stable_route")
        self.nav = ActionClient(ACTION_NAME)

        self.outcomes: List[Outcome] = []
        self.saved_critical = []                                   # cells, in completion order
        self.failed = []                                           # (phase, cell)
        self.switch_time: Optional[float] = None
        self.switch_reason = ""
        self.switch_cancelled_goal = ""
        self.switch_position = None
        self.critical_cells = []
        self.stable_cells = []
        self.landed = False
        self.landing_time = None
        self.ok = False
        self.requested_critical: List[Waypoint] = []
        self.requested_stable: List[Waypoint] = []

        self._mt = 0.0                                             # latest mission time (from feedback)
        self._handle = None
        self._cancel_sent = False
        self._watch = False

    # ------------------------------------------------------------------ helpers
    def _say(self, text, t=None):
        t = self._mt if t is None else t
        self.log(f"[pico_client] t={t:6.2f}  {text}")

    def _on_feedback(self, fb: NavFeedback):
        self._mt = fb.mission_time
        if self._watch and not self._cancel_sent and fb.mission_time >= self.window and self._handle is not None:
            self._cancel_sent = True                               # the Switch, in the middle of a goal
            self._handle.cancel_goal_async()

    def _fly(self, wp: Waypoint, phase: str, watch_clock: bool):
        """Send one goal and block until its result. Returns (status, result) or (None, None)."""
        self._cancel_sent, self._watch, self._handle = False, watch_clock, None
        handle = self.nav.send_goal_async(NavGoal(wp), self._on_feedback).result(5.0)
        if not handle.accepted:
            self._say(f"goal '{wp.name}' was REJECTED")
            return None, None
        self._handle = handle
        fut = handle.get_result_async()
        end = time.time() + self.result_timeout
        while not fut.done():
            if time.time() > end:
                return None, None
            time.sleep(0.002)
        status, res = fut.result()
        self.outcomes.append(Outcome(phase, wp, status, res))
        if res is not None:
            self._mt = max(self._mt, res.end_time)
        return status, res

    def _fly_route(self, waypoints, phase, watch_clock):
        """Fly a route goal by goal. Returns 'done' | 'switch' | 'failed'."""
        skip = None
        n_crit = len(self.critical_cells)
        for wp in waypoints:
            if skip is not None and wp.survivor == skip:
                continue
            skip = None
            if wp.kind == "visit":
                self._say(f"-> {phase} survivor {wp.survivor}: VISIT")
            status, res = self._fly(wp, phase, watch_clock)
            if status is None:
                self._say(f"goal '{wp.name}' did not complete")
                return "failed"
            if status == GoalStatus.CANCELED:
                self.switch_cancelled_goal = wp.name
                return "switch"
            if status != GoalStatus.SUCCEEDED:
                if wp.kind in ("visit", "save"):
                    self._say(f"goal '{wp.name}' FAILED ({res.message if res else status}) -- skipping survivor {wp.survivor}")
                    self.failed.append((phase, wp.survivor))
                    skip = wp.survivor
                    continue
                return "failed"
            if wp.kind == "save":
                if phase == "critical":
                    if res.end_time <= self.window:
                        self.saved_critical.append(wp.survivor)
                        self._say(f"critical survivor {wp.survivor} SAVED  ({len(self.saved_critical)}/{n_crit})", res.end_time)
                    else:
                        self._say(f"critical survivor {wp.survivor}: Save finished at {res.end_time:.2f} s, "
                                  f"after the {self.window:g} s window -- does not count", res.end_time)
                else:
                    self._say(f"stable survivor {wp.survivor} SAVED", res.end_time)
            if wp.kind == "home" and res.landed:
                self.landed, self.landing_time = True, res.end_time
                self._say("LANDED on F6", res.end_time)
            if watch_clock:
                self.switch_position = (res.x, res.y, res.z)
                if len(self.saved_critical) >= n_crit and wp.kind == "save" and wp.phase == "critical":
                    self.switch_time, self.switch_reason = res.end_time, "all critical survivors saved"
                    return "switch"
                if res.end_time >= self.window:
                    return "switch"
        return "done"

    # ------------------------------------------------------------------ the mission
    def run(self) -> bool:
        if not (self.route_srv.wait_for_service(10) and self.stable_srv.wait_for_service(10)
                and self.nav.wait_for_server(10)):
            self.log("[pico_client] services / action server not available")
            return False

        # ---------------- critical phase
        resp = self.route_srv.call(GetWaypointsRequest(), timeout=30.0)
        self.requested_critical = list(resp.waypoints)
        self.critical_cells = sorted({w.survivor for w in resp.waypoints if w.survivor})
        self._say(f"critical route received: {resp.info}", 0.0)
        outcome = self._fly_route(resp.waypoints, "critical", watch_clock=True)
        if outcome == "failed":
            return False

        # ---------------- THE SWITCH
        if self.switch_reason == "":
            self.switch_reason = f"{self.window:g} s up" if self._mt >= self.window else "critical route finished"
        if self.switch_time is None:
            self.switch_time = self.window if "up" in self.switch_reason else self._mt
        if self.switch_cancelled_goal:
            # the cancel result carries the drone's position
            last = self.outcomes[-1].result
            self.switch_position = (last.x, last.y, last.z)
        self._say(f"=== THE SWITCH ({self.switch_reason}) ===  critical saved: {self.saved_critical}"
                  + (f"  | cancelled goal '{self.switch_cancelled_goal}'" if self.switch_cancelled_goal else ""),
                  self.switch_time if not self.switch_cancelled_goal else self._mt)
        x, y, z = self.switch_position
        self._say(f"drone at ({x:+.2f}, {y:+.2f}, Z={z:.2f}) -> requesting the stable route from there")

        # ---------------- stable phase + home
        sresp = self.stable_srv.call(GetStableRouteRequest(x, y, z), timeout=30.0)
        self.requested_stable = list(sresp.waypoints)
        self.stable_cells = sorted({w.survivor for w in sresp.waypoints if w.survivor})
        outcome = self._fly_route(sresp.waypoints, "stable", watch_clock=False)
        self.ok = outcome == "done" and self.landed
        return self.ok
