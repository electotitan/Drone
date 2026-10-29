# Swift Pico — Task 3: Rescue Mission (MuJoCo, no ROS 2)

Scan the arena from ONE camera frame, save the 3 critical survivors first (55 s window), then the 3 stable
ones, planning every route around the obstacles, and land on F6. Each survivor gets the same Saving maneuver:
**Visit** (hold 1 s at Z = 13) then **Save** (descend, hold 2 s at Z = 14), inside a ±0.4 box on X, Y, Z with the
camera seeing the drone's marker.

## Folder
```
pico_rescue_task3/
├── requirements.txt          mujoco, numpy, scipy, matplotlib
├── controller/               the LQR controller the mission flies with
│   ├── lqr.py                linear model from drone.xml + Riccati solve -> gain matrix K
│   └── controller.py         DroneController (u = -Kx, mixer, set_setpoint)
├── model/
│   ├── drone.xml, arena.xml  your uploaded originals (unchanged)
│   ├── make_rescue_scene.py  generates the two files below
│   ├── drone_rescue.xml      drone + arena_rescue.xml          <- the mission loads this
│   ├── arena_rescue.xml      your 3 obstacle boxes + critical triangles (I2, J9, B10) + stable circles (B2, J6, F10)
│   └── meshes/               7 assets used by the model (.obj + 2 textures)
└── rescue/
    ├── run_rescue.py         entry point: run everything, score it independently, plot
    ├── waypoint_service.py   program 1: frame -> survivors + obstacles -> critical / stable routes
    ├── pico_client.py        program 2: 55 s clock, the Switch, cancel, stable route, landing
    ├── pico_server.py        program 3: nav_to_waypoint action server (cancel, holds, lift-off, landing)
    ├── vision.py             camera model + survivor / obstacle detection
    ├── planner.py            obstacle-aware A* on the flight plane
    ├── rescue_sim.py         MuJoCo + LQR, WhyCode Z, marker line of sight, arming, 200 s cap
    ├── scoring.py            independent scoring from the raw simulation log
    ├── middleware.py         in-process services / actions / latched topics (replaces ROS 2)
    └── messages.py, lattice.py
```

## Run
```bash
pip install -r requirements.txt
cd rescue
python3 run_rescue.py --realtime --expect-critical I2,J9,B10 --expect-stable B2,J6,F10   # honest timings (~90 s wall)
python3 run_rescue.py                    # headless, as fast as possible (~1 min wall)
python3 run_rescue.py --viewer           # 3D viewer (macOS: mjpython run_rescue.py --viewer)
python3 run_rescue.py --window 25        # force the Switch to fire mid-goal (testing)
python3 run_rescue.py --help
```
Writes `rescue_map.png`, `rescue_timeline.png`, `rescue_log.csv`, `rescue_result.json` next to the script.
Exit code 0 = PASS. On headless Linux the camera render uses EGL automatically.
Other layouts: `cd model && python3 make_rescue_scene.py --critical A3,K9,D8 --stable G2,J5,C11`
(run it with no arguments to restore the reference map).

## Z convention
WhyCode **Z is measured down from the camera** (camera at world z = 15 m): Z = 13 is 2 m above the floor, Z = 14 is
1 m above the floor — below the 3 m obstacle tops. X, Y are world metres. The conversion lives in `RescueSim.to_world_z`.

## The three programs
- **waypoint_service** subscribes to `/image_raw` (one frame): red triangle = critical, yellow circle = stable, big black
  blob = obstacle. Serves `get_waypoints` (critical route from F6) and `get_stable_route(x, y, Z)` (stable route from the
  drone's position at the Switch, ending at F6). Per survivor: fly-past points at Z = 13 -> visit -> save. The cheapest
  survivor order is picked from obstacle-aware path lengths; all pairwise paths are precomputed before lift-off.
- **pico_client** fetches the critical route and sends goals one at a time. The clock is the server's (t = 0 at
  lift-off). After every goal and on every feedback: "all critical saved, or 55 s up?". If 55 s run out mid-goal it
  **cancels** the goal, takes the drone's position from the result and calls `get_stable_route` from there. A critical
  survivor counts only if its Save ended by 55 s. If all three are saved earlier the Switch happens immediately.
  Then it flies the stable route ending with the landing.
- **pico_server** (`nav_to_waypoint`): the first goal lifts off F6 (t = 0; motors are off before). A hold counts only
  while the drone is in the box on X, Y and Z AND the camera sees its marker (checked at 200 Hz); leaving the box or losing
  the marker restarts it. Cancel = hold in place and report the position. The home goal = hover 1 s over F6, then land.

## Obstacles
The camera sees each 3 m box as a silhouette shifted outward by perspective. The planner maps the black pixels onto
the flight plane (Z = 13) with that plane's pixel scale; the true cross-section at flight altitude lies inside that
silhouette, which also covers the region where the box would hide the drone's marker from the camera. It is inflated by
0.85 m (drone radius 0.33 + tracking slack) and A* + line-of-sight smoothing plans around it.

## Holds are counted with margin
The server counts a hold only inside a box 0.1 m tighter (±0.3) and approaches hold goals gently, so a 1–2 cm error can't
turn a 1.00 s hold into a miss. The independent scorer uses the true ±0.4 box: Visits settle to ±0.23–0.25 m and Saves
to ±0.12 m.

## Results (simulation, reference map)
- Scan: 3 critical + 3 stable + 3 obstacles, <= 1.8 cm position error.
- Critical route I2 -> J9 -> B10 all saved by t ~ 37–41 s -> Switch; stable route F10 -> J6 -> B2 -> F6.
- 6/6 saved, landed on F6 at 91.7 s real-time (about 94 s unthrottled; full speed marks need <= 110 s).
- 0 obstacle contacts (closest approach 0.8–0.96 m), marker never lost.
- Also tested: `--window 25 / 22.5 / 4` (Switch mid-goal, incl. mid-Save), a different layout, `--margin 0.05` (collision caught),
  `--max-run 60` (ends un-landed -> FAIL).

## Assumptions and limits
- Visiting order within a phase = shortest total obstacle-aware path (the brief does not fix it). Home is F6 = world (0, 0).
- "Camera loses the marker" = an obstacle blocks the camera-to-drone line of sight. No sensor noise or latency.
- A survivor hidden in an obstacle's camera shadow can't be seen in one frame and is skipped.
- The unthrottled simulator adds artificial dead time at goal hand-offs; use `--realtime` for representative timings.
- `middleware.py` mimics rclpy's service / action / topic shape with plain threads inside one process.
