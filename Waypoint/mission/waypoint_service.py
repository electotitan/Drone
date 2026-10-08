"""waypoint_service: returns the predefined mission waypoints.

THE ORDER OF THIS LIST IS THE ORDER THE DRONE FLIES.  Both the action client and
the verification rely on it -- keep wp1 first and wp4 last.
Units are the simulator's world / WhyCode units (metres, z up, arena centre = 0,0).
"""
from messages import Waypoint, GetWaypointsRequest, GetWaypointsResponse
from middleware import ServiceServer

SERVICE_NAME = "waypoint_service"

WAYPOINTS = [
    Waypoint("wp1", -2.41, -2.41, 13.00),
    Waypoint("wp2",  2.41,  1.20, 13.00),
    Waypoint("wp3",  1.20, -2.41, 13.00),
    Waypoint("wp4", -1.20,  1.20, 13.00),
]


def make_waypoint_service():
    def handler(_request: GetWaypointsRequest) -> GetWaypointsResponse:
        return GetWaypointsResponse(waypoints=list(WAYPOINTS))      # same order, copy
    return ServiceServer(SERVICE_NAME, handler)
