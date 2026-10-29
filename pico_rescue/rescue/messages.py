"""Message / service / action definitions for the rescue mission (plain dataclasses).

Z is WhyCode Z: distance DOWN from the top camera (camera at world z = 15 m).
Z = 13 -> 2 m above the floor, Z = 14 -> 1 m above the floor. X, Y are world metres.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    z: float                       # WhyCode Z
    kind: str = "via"              # 'via' | 'visit' | 'save' | 'home'
    hold: float = 0.0              # seconds the drone must stay inside the box (0 = fly past)
    tol: float = 0.4               # half-width of the box on X, Y and Z
    survivor: str = ""             # cell of the survivor this goal belongs to ('' for via / home)
    phase: str = ""                # 'critical' | 'stable' | ''

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ---- services ---------------------------------------------------------------------
@dataclass
class GetWaypointsRequest:          # critical route; empty
    pass


@dataclass
class GetWaypointsResponse:
    waypoints: List[Waypoint] = field(default_factory=list)
    info: str = ""


@dataclass
class GetStableRouteRequest:        # where the drone is right now (WhyCode coordinates)
    x: float = 0.0
    y: float = 0.0
    z: float = 13.0


@dataclass
class GetStableRouteResponse:
    waypoints: List[Waypoint] = field(default_factory=list)
    info: str = ""


# ---- action: nav_to_waypoint ---------------------------------------------------------
@dataclass
class NavGoal:
    waypoint: Waypoint


@dataclass
class NavFeedback:
    mission_time: float             # s since lift-off (t = 0)
    x: float                        # current position (WhyCode coordinates)
    y: float
    z: float
    err_x: float
    err_y: float
    err_z: float
    in_box_time: float              # s continuously inside the box so far
    marker_visible: bool


@dataclass
class NavResult:
    success: bool
    canceled: bool = False
    landed: bool = False
    start_time: float = 0.0         # mission time the goal started
    end_time: float = 0.0           # mission time the goal finished (hold completed / canceled)
    hold_start: float = float("nan")  # mission time the final, uninterrupted hold began
    x: float = 0.0                  # position when the goal ended (WhyCode coordinates)
    y: float = 0.0
    z: float = 0.0
    message: str = ""
