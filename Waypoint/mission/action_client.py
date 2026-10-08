"""
Mission action client.

  1. calls waypoint_service to get the predefined waypoints
  2. sends them as goals to navigate_to_waypoint, ONE AT A TIME, in exactly the
     order the service returned them (no sorting, no skipping)
  3. waits for the server's result (= drone stabilised) before sending the next
"""
from dataclasses import dataclass
from typing import List, Optional

from messages import GetWaypointsRequest, NavigateGoal, NavigateFeedback, NavigateResult, Waypoint
from middleware import ActionClient, ServiceClient, GoalStatus
from waypoint_service import SERVICE_NAME as WAYPOINT_SERVICE
from action_server import ACTION_NAME


@dataclass
class WaypointOutcome:
    waypoint: Waypoint
    status: str
    result: Optional[NavigateResult]


class MissionClient:
    def __init__(self, verbose=True, result_timeout=600.0):
        self.wp_client = ServiceClient(WAYPOINT_SERVICE)
        self.nav_client = ActionClient(ACTION_NAME)
        self.verbose = verbose
        self.result_timeout = result_timeout            # wall seconds
        self.outcomes: List[WaypointOutcome] = []
        self.requested_order: List[Waypoint] = []
        self._last_print_sec = -1

    def _feedback_cb(self, fb: NavigateFeedback):
        sec = int(fb.elapsed)
        if self.verbose and sec != self._last_print_sec:         # ~1 line per sim second
            self._last_print_sec = sec
            print(f"[client]   t+{fb.elapsed:5.1f}s  pos=({fb.x:+7.3f}, {fb.y:+7.3f}, {fb.z:+7.3f})  "
                  f"err=({fb.err_x:+6.3f}, {fb.err_y:+6.3f}, {fb.err_z:+6.3f})  in-band {fb.in_band_time:.1f}s")

    def run(self) -> bool:
        if not self.wp_client.wait_for_service(10.0):
            print("[client] waypoint_service not available"); return False
        if not self.nav_client.wait_for_server(10.0):
            print("[client] navigate_to_waypoint server not available"); return False

        response = self.wp_client.call(GetWaypointsRequest(), timeout=5.0)
        waypoints = response.waypoints                      # <- order is NEVER changed
        self.requested_order = list(waypoints)
        print(f"[client] waypoint_service returned {len(waypoints)} waypoints: "
              + ", ".join(w.name for w in waypoints))

        for i, wp in enumerate(waypoints, 1):
            print(f"[client] sending goal {i}/{len(waypoints)}: {wp.name} ({wp.x:+.2f}, {wp.y:+.2f}, {wp.z:+.2f})")
            self._last_print_sec = -1
            handle = self.nav_client.send_goal_async(NavigateGoal(wp), self._feedback_cb).result(5.0)
            if not handle.accepted:
                print(f"[client] goal {wp.name} was REJECTED -- stopping")
                self.outcomes.append(WaypointOutcome(wp, GoalStatus.REJECTED, None))
                return False

            status, result = handle.get_result_async().result(self.result_timeout)  # blocks until stabilised
            self.outcomes.append(WaypointOutcome(wp, status, result))
            if status != GoalStatus.SUCCEEDED:
                print(f"[client] {wp.name} finished with status {status} -- stopping")
                return False
            print(f"[client] {wp.name} confirmed stable: settle time {result.settle_time:.2f} s "
                  f"(final {result.final_x:+.3f}, {result.final_y:+.3f}, {result.final_z:+.3f})\n")

        print("[client] all waypoints reached")
        return True
