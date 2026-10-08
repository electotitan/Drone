"""Message / service / action type definitions (plain dataclasses)."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    z: float

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ---- service: waypoint_service ------------------------------------------------
@dataclass
class GetWaypointsRequest:
    pass                                   # empty request, like std_srvs/Trigger


@dataclass
class GetWaypointsResponse:
    waypoints: List[Waypoint] = field(default_factory=list)


# ---- action: navigate_to_waypoint ----------------------------------------------
@dataclass
class NavigateGoal:
    waypoint: Waypoint


@dataclass
class NavigateFeedback:
    sim_time: float            # simulation clock (s)
    elapsed: float             # s since the goal was accepted
    x: float                   # current drone position
    y: float
    z: float
    err_x: float               # position - target
    err_y: float
    err_z: float
    in_band_time: float        # s the drone has CONTINUOUSLY been inside the +/-band so far


@dataclass
class NavigateResult:
    success: bool
    settle_time: float         # s from goal accepted until the drone entered the band and stayed
    hold_time: float           # s it was then confirmed holding inside the band
    final_x: float
    final_y: float
    final_z: float
    start_time: float = 0.0    # sim time at which the goal's setpoint was applied
    message: str = ""
