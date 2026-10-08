"""
navigate_to_waypoint action server.

  Goal     : NavigateGoal(waypoint)
  Feedback : NavigateFeedback  -- the drone's current position (10 Hz of SIM time)
  Result   : NavigateResult    -- success + the time taken to settle

"Stabilised" means: |x-x*|, |y-y*| and |z-z*| are ALL <= band (0.4) continuously
for at least hold_duration (3 s). The check runs on every physics sample (200 Hz),
not just on the feedback ticks, so a brief excursion cannot slip through.

settle_time = (sim time the drone entered the band for the final, uninterrupted
               stay) - (sim time the goal's setpoint was applied).
"""
import numpy as np

from messages import NavigateFeedback, NavigateResult
from middleware import ActionServer

ACTION_NAME = "navigate_to_waypoint"


class NavigateServer:
    def __init__(self, sim, band=0.4, hold_duration=3.0, timeout=60.0, feedback_period=0.1):
        self.sim = sim
        self.band = band
        self.hold_duration = hold_duration
        self.timeout = timeout                  # sim seconds before the goal is aborted
        self.feedback_period = feedback_period  # sim seconds
        self._server = ActionServer(ACTION_NAME, self.execute)

    def shutdown(self):
        self._server.shutdown()

    # ------------------------------------------------------------------
    def execute(self, goal_handle):
        wp = goal_handle.request.waypoint
        target = np.array(wp.xyz, dtype=float)
        hold = self.hold_duration if wp.hold is None else wp.hold     # 'via' points: 0 s = just fly through
        quiet = wp.kind == "via"
        if not quiet:
            print(f"[server] goal accepted: {wp.name} -> ({wp.x:+.2f}, {wp.y:+.2f}, {wp.z:+.2f})  hold {hold:.1f} s")

        t0, idx = self.sim.set_setpoint(target)          # controller now flies to the waypoint
        in_band_since = None
        last_t = t0
        next_fb = t0
        state_pos = self.sim.get_state().pos

        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                print(f"[server] {wp.name} canceled")
                return self._result(False, t0, in_band_since, last_t, state_pos, "canceled")

            if not self.sim.wait_for_time(last_t + self.feedback_period / 2):
                goal_handle.abort()
                return self._result(False, t0, in_band_since, last_t, state_pos, "simulation stopped")

            idx, rows = self.sim.samples_since(idx)
            for t, x, y, z in rows:                      # every physics step since last wake-up
                last_t = t
                state_pos = np.array((x, y, z))
                if np.max(np.abs(state_pos - target)) <= self.band:
                    if in_band_since is None:
                        in_band_since = t
                    if t - in_band_since >= hold:
                        goal_handle.succeed()
                        res = self._result(True, t0, in_band_since, t, state_pos,
                                           f"held within +/-{self.band} for {hold:.1f}s")
                        if not quiet:
                            print(f"[server] {wp.name} stabilised: settle_time = {res.settle_time:.2f} s")
                        return res
                else:
                    in_band_since = None                 # left the band: restart the hold timer

            if last_t >= next_fb and len(rows):
                err = state_pos - target
                goal_handle.publish_feedback(NavigateFeedback(
                    sim_time=last_t, elapsed=last_t - t0,
                    x=state_pos[0], y=state_pos[1], z=state_pos[2],
                    err_x=err[0], err_y=err[1], err_z=err[2],
                    in_band_time=0.0 if in_band_since is None else last_t - in_band_since))
                next_fb = last_t + self.feedback_period

            if last_t - t0 > self.timeout:
                goal_handle.abort()
                print(f"[server] {wp.name} TIMEOUT after {self.timeout:.0f} s")
                return self._result(False, t0, in_band_since, last_t, state_pos, "timeout")

    @staticmethod
    def _result(ok, t0, since, t_now, pos, msg):
        settle = (since - t0) if since is not None else float("nan")
        hold = (t_now - since) if since is not None else 0.0
        return NavigateResult(success=ok, settle_time=settle, hold_time=hold,
                              final_x=float(pos[0]), final_y=float(pos[1]), final_z=float(pos[2]),
                              start_time=float(t0), message=msg)
