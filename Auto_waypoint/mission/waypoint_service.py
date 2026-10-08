"""waypoint_service: returns the mission waypoints.

The list is NO LONGER hard-coded: run_mission.py builds it from what the arena scan found
(markers -> ordering -> lattice route) and hands it to this service. The service returns it
exactly as built; the client must visit it in that order.
"""
from typing import List

from messages import Waypoint, GetWaypointsRequest, GetWaypointsResponse
from middleware import ServiceServer

SERVICE_NAME = "waypoint_service"


def make_waypoint_service(waypoints: List[Waypoint]):
    frozen = list(waypoints)

    def handler(_request: GetWaypointsRequest) -> GetWaypointsResponse:
        return GetWaypointsResponse(waypoints=list(frozen))          # same order, copy
    return ServiceServer(SERVICE_NAME, handler)
