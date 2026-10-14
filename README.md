# Drone

Collection of eYRC Swift Pico drone tasks covering image processing, MuJoCo simulation, control (PID/LQR), waypoint missions, and ROS 2 utilities.

## Repository structure

| Path | Purpose |
|---|---|
| `/home/runner/work/Drone/Drone/TASK_1A` | Arena image-processing pipeline (ArUco rectification + survivor detection). |
| `/home/runner/work/Drone/Drone/pico_altitude_hold` | MuJoCo altitude-hold controller (PID). |
| `/home/runner/work/Drone/Drone/pico_position_hold` | MuJoCo 3-axis position-hold controller (PID). |
| `/home/runner/work/Drone/Drone/pico_lqr_hold` | MuJoCo position hold and mission flows using LQR. |
| `/home/runner/work/Drone/Drone/Waypoint` | Waypoint mission flow with service/action-style architecture. |
| `/home/runner/work/Drone/Drone/Auto_waypoint` | Extended autonomous waypoint/scan-planning mission stack. |
| `/home/runner/work/Drone/Drone/Fly_Manual` | Manual-flight simulation tools and keyboard control flow. |
| `/home/runner/work/Drone/Drone/circle_drawer_package` | ROS 2 `turtlesim` circle-drawing package. |
| `/home/runner/work/Drone/Drone/swift_pico_simulation.launch.py` | ROS 2 launch file for Swift Pico + WhyCode + bridge integrations. |

## Quick start

Each module is self-contained. Use the README inside that folder for full details.

### 1) Task 1A (image processing)

```bash
cd /home/runner/work/Drone/Drone/TASK_1A
pip install opencv-contrib-python numpy
python3 task1a.py --image image_1.jpg
```

### 2) MuJoCo control modules

Install dependencies per module, then run the module entrypoint:

```bash
# Altitude hold
cd /home/runner/work/Drone/Drone/pico_altitude_hold/model
pip install -r requirements.txt
cd ../controller
python3 run_sim.py
```

```bash
# Position hold (PID)
cd /home/runner/work/Drone/Drone/pico_position_hold
pip install -r requirements.txt
cd controller
python3 run_sim.py
```

```bash
# Position hold / missions (LQR)
cd /home/runner/work/Drone/Drone/pico_lqr_hold
pip install -r requirements.txt
cd controller
python3 run_sim.py --setpoint 2 2 3
```

## ROS 2 components

- `circle_drawer_package` targets ROS 2 Humble (`colcon build --packages-select circle_drawer`).
- `swift_pico_simulation.launch.py` is a launch entrypoint for Swift Pico simulation integrations (requires the corresponding ROS 2 packages in your workspace).

## Notes

- Most simulation modules use `mujoco`, `numpy`, and optionally `matplotlib`; LQR modules also use `scipy`.
- Waypoint/autonomous mission modules include their own run scripts under each module’s `mission/` directory.
- Start from each module’s README for exact command-line options and verification criteria.