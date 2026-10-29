"""
pico_server (program 2 of 3) -- action server `nav_to_waypoint`

  goal      NavGoal(waypoint)            fly to it with the Task 2A LQR controller
  feedback  NavFeedback                  current position, error, time in the box, marker visible
  result    NavResult                    success / canceled / landed + the times and final position

What counts as "holding": the drone is inside the +/-tol box on X, Y and Z around the goal AND the
camera can see its marker. Leaving the box or losing the marker restarts the hold from zero. It is
checked on every physics sample (200 Hz). A goal with hold = 0 (fly-past) succeeds the first time the
drone is inside its box.

New in Task 3
  * the FIRST goal lifts the drone off F6; that instant is mission time t = 0
  * cancel: the drone is told to hold where it is, and the result carries its position
  * a `home` goal = hover over F6 for its hold, then LAND (descend, touch down, motors off)
"""
import numpy as np

from messages import NavFeedback, NavResult
from middleware import ActionServer

ACTION_NAME = "nav_to_waypoint"
REST_Z_WORLD = 0.175          # drone body height when sitting on the floor


class PicoServer:
    def __init__(self, sim, feedback_period=0.1, goal_timeout=60.0, land_rate=0.5, hold_margin=0.1, log=print):
        self.sim = sim
        self.hold_margin = hold_margin            # holds are counted inside a box this much TIGHTER than +/-tol
        self.feedback_period = feedback_period
        self.goal_timeout = goal_timeout          # sim seconds per goal
        self.land_rate = land_rate                # m/s descent during landing
        self.log = log
        self._server = ActionServer(ACTION_NAME, self.execute)

    def shutdown(self):
        self._server.shutdown()

    # ------------------------------------------------------------------
    def _result(self, ok, wp, t_start, hold_start, t_now, st, msg, canceled=False, landed=False):
        mt = self.sim.mission_time
        return NavResult(success=ok, canceled=canceled, landed=landed,
                         start_time=mt(t_start), end_time=mt(t_now),
                         hold_start=float("nan") if hold_start is None else mt(hold_start),
                         x=st.X, y=st.Y, z=st.Z, message=msg)

    def execute(self, gh):
        sim = self.sim
        wp = gh.request.waypoint
        target = np.array(wp.xyz, dtype=float)
        if sim.t_takeoff is None:
            sim.arm()                                            # LIFT-OFF: mission time t = 0
            self.log("[pico_server] lift-off from F6  (t = 0)")
        t0, idx = sim.set_setpoint(*wp.xyz)
        in_since, last_t, next_fb = None, t0, t0
        st = sim.get_state()
        quiet = wp.kind == "via"
        holding_goal = wp.hold > 0
        # a hold only counts inside a slightly tighter box, so camera noise / a 1-2 cm error in the
        # survivor's position can never turn a "1.00 s" hold into a failed one
        tol = max(wp.tol - self.hold_margin, 0.05) if holding_goal else wp.tol

        while True:
            if gh.is_cancel_requested:
                st = sim.get_state()
                sim.set_setpoint(st.X, st.Y, st.Z)               # hold in place
                gh.canceled()
                self.log(f"[pico_server] t={st.mission_time:6.2f}  goal '{wp.name}' CANCELED at "
                         f"({st.X:+.2f}, {st.Y:+.2f}, Z={st.Z:.2f}) -- holding")
                return self._result(False, wp, t0, in_since, st.t, st, "canceled", canceled=True)

            if not sim.wait_for_time(last_t + self.feedback_period / 2):
                st = sim.get_state()
                gh.abort()
                return self._result(False, wp, t0, in_since, st.t, st, "simulation stopped")

            idx, rows = sim.samples_since(idx)
            for t, X, Y, Z, vis, zw, _armed in rows:
                last_t = t
                inside = bool(vis) and max(abs(X - target[0]), abs(Y - target[1]), abs(Z - target[2])) <= tol
                if inside:
                    if in_since is None:
                        in_since = t
                    if t - in_since >= wp.hold:
                        st = sim.get_state()
                        gh.succeed()
                        if not quiet:
                            self.log(f"[pico_server] t={sim.mission_time(t):6.2f}  '{wp.name}' done "
                                     f"(hold {wp.hold:.1f} s from t={sim.mission_time(in_since):.2f})")
                        res = self._result(True, wp, t0, in_since, t, st, f"held {wp.hold:.1f} s")
                        if wp.kind == "home":
                            return self._land(gh, wp, res)
                        return res
                else:
                    in_since = None

            if holding_goal and in_since is None:
                self._approach(wp, target)            # arrive gently instead of streaking through the box

            if len(rows) and last_t >= next_fb:
                st = sim.get_state()
                gh.publish_feedback(NavFeedback(
                    mission_time=sim.mission_time(last_t), x=st.X, y=st.Y, z=st.Z,
                    err_x=st.X - target[0], err_y=st.Y - target[1], err_z=st.Z - target[2],
                    in_box_time=0.0 if in_since is None else last_t - in_since, marker_visible=st.visible))
                next_fb = last_t + self.feedback_period

            if last_t - t0 > self.goal_timeout:
                st = sim.get_state()
                gh.abort()
                self.log(f"[pico_server] goal '{wp.name}' TIMEOUT")
                return self._result(False, wp, t0, in_since, last_t, st, "timeout")

    def _approach(self, wp, target):
        """Gentle arrival at a hold goal: inside ~2 m the commanded point is only `0.4 + 0.4 d` metres ahead of
        the drone (d = distance left), so the drone slows down instead of overshooting the box."""
        st = self.sim.get_state()
        dx, dy = target[0] - st.X, target[1] - st.Y
        d = float(np.hypot(dx, dy))
        reach = 0.4 + 0.4 * d
        if d > reach:
            f = reach / d
            self.sim.set_setpoint(st.X + dx * f, st.Y + dy * f, wp.z)
        else:
            self.sim.set_setpoint(*wp.xyz)

    # ------------------------------------------------------------------
    def _land(self, gh, wp, hover_result):
        """After the 1 s hover over F6: ramp the setpoint down, touch down, cut the motors."""
        sim = self.sim
        st = sim.get_state()
        Z, t_prev, slow = st.Z, st.t, None
        self.log(f"[pico_server] t={st.mission_time:6.2f}  hover at F6 complete -> landing")
        while True:
            if not sim.wait_for_time(t_prev + 0.05):
                break
            st = sim.get_state()
            Z = min(Z + self.land_rate * (st.t - t_prev), sim.cam_z)         # Z grows as the drone descends
            t_prev = st.t
            sim.set_setpoint(wp.x, wp.y, Z)
            if st.z_world < REST_Z_WORLD + 0.03 and abs(st.vel[2]) < 0.08:  # on the floor and slow
                sim.disarm()
                break
        # wait for the airframe to come to rest (touchdown can bounce slightly)
        calm_since = None
        while sim.wait_for_time(t_prev + 0.05):
            st = sim.get_state()
            t_prev = st.t
            if abs(st.vel[2]) < 0.05 and np.linalg.norm(st.vel[:2]) < 0.05:
                calm_since = st.t if calm_since is None else calm_since
                if st.t - calm_since > 0.4:
                    break
            else:
                calm_since = None
            if sim.t_disarm is None or st.t - sim.t_disarm > 4.0:
                break
        st = sim.get_state()
        landed = (not sim.armed) and st.z_world < REST_Z_WORLD + 0.05
        hover_result.landed = landed
        hover_result.end_time = sim.mission_time(sim.t_disarm if sim.t_disarm is not None else st.t)
        hover_result.x, hover_result.y, hover_result.z = st.X, st.Y, st.Z
        hover_result.message = "hovered over F6 and landed" if landed else "landing failed"
        self.log(f"[pico_server] t={hover_result.end_time:6.2f}  LANDED at ({st.X:+.2f}, {st.Y:+.2f})"
                 if landed else "[pico_server] landing did not complete")
        return hover_result
