# Swift Pico — Position Hold with LQR (MuJoCo)

A standalone MuJoCo + Python controller for `drone.xml` that drives the Swift Pico
to a target `(x, y, z)` — **Pitch, Roll, Throttle** in eYRC's terms — using a
**Linear Quadratic Regulator (LQR) with integral action** instead of the previous
cascaded-PID stack. It checks, on **each axis independently**:

> error settles within **±0.4 m** of that axis's setpoint within **5 s**,
> and stays inside that band for **10 s** straight.

Result for the `(2, 2, 3)` step (spawn is `(0, 0, 0.3)`): every axis settles by
**t ≈ 1.7–1.9 s**, overshoot ≈ 0.1 m, and holds with ~0.0001 m error out to 90 s.

## Project layout

```
pico_lqr_hold/
├── model/                 # drone.xml, arena.xml, meshes/, textures/ (unchanged)
│   ├── make_scene.py      # generates drone_scan.xml + arena_scan.xml (markers like the reference picture)
│   ├── drone_scan.xml     # generated: drone + arena_scan.xml   <- the mission loads this
│   └── arena_scan.xml     # generated: composite markers at C9, D5, H4, J7, no obstacles
├── controller/
│   ├── lqr.py             # builds the linear model from drone.xml, solves the Riccati eq.
│   ├── controller.py      # DroneController: x -> u = -Kx -> mixer -> 4 motor thrusts
│   └── run_sim.py         # single-setpoint hold test (Task 2A)
├── mission/               # Task 2C: scan the arena, plan, navigate (service + action server + client)
│   ├── run_mission.py     # entry point: scan -> plan -> fly -> verify -> plots
│   ├── arena_scan.py      # top-camera frame -> marker detection -> world / lattice cell
│   ├── lattice.py         # A..K / 1..11 lattice <-> world coordinates
│   ├── route_planner.py   # visiting order + lattice route (the dotted route in the picture)
│   ├── waypoint_service.py# serves whatever plan the scan produced (no hard-coded list)
│   ├── action_server.py   # navigate_to_waypoint server
│   ├── action_client.py   # fetches waypoints, sends goals one at a time
│   ├── middleware.py      # in-process Service/Action framework (threads + queues, NO ROS 2)
│   ├── messages.py        # Waypoint, NavigateGoal / Feedback / Result dataclasses
│   └── sim_node.py        # MuJoCo + LQR loop with thread-safe state / setpoint access
├── requirements.txt       # mujoco, numpy, scipy, matplotlib
└── README.md
```
`pid.py` is no longer needed and has been removed.

## Run it

```bash
pip install -r requirements.txt        # note: scipy is new (needed for the Riccati solver)
cd controller
python3 lqr.py                          # print the gain matrix K, poles, controllability check
python3 run_sim.py --setpoint 2 2 3     # headless PASS/FAIL report + log.csv + altitude_plot.png
python3 run_sim.py --setpoint 2 2 3 --viewer
python3 run_sim.py --help
```
`run_sim.py` is unchanged apart from wording; the CLI and exit codes (0 = PASS, 1 = FAIL) are the same.

## How the LQR works

**State (15)**, as *errors* from the setpoint, in the drone's heading frame:

| group | states |
|---|---|
| position error | ex, ey, ez |
| velocity | vx, vy, vz |
| attitude | roll φ, pitch θ, yaw ψ (error vs. spawn heading) |
| body rates | p, q, r |
| integral of position error | ∫ex, ∫ey, ∫ez |

**Input (4):** `u = [ΔF, τx, τy, τz]` — collective-thrust deviation from hover and three body torques.

**Linearised model about hover** (`lqr.py`):

```
ẍ = g·θ      ÿ = −g·φ      z̈ = ΔF/m
φ̈ = τx/Ixx   θ̈ = τy/Iyy    ψ̈ = τz/Izz       İ = e
```
Mass, gravity and inertia are **read from the MuJoCo model** (the 1.5 kg body plus the
four small prop bodies → m = 1.525 kg, I ≈ [0.036, 0.071, 0.099] kg·m²), so editing
`drone.xml` re-derives the gains automatically.

**Design:** ZOH-discretise at the simulator timestep (0.005 s, matches how MuJoCo holds
`ctrl`), solve the discrete algebraic Riccati equation, `K = (R + BᵀPB)⁻¹BᵀPA`, and apply
`u = −Kx`. The result is a single 4×15 matrix; the system is fully controllable
(rank 15/15) and the slowest closed-loop pole is ≈ −1 rad/s.

**Around the linear law** (`controller.py`):
- `F_total = (m·g + ΔF) / cos(tilt)` — tilt-compensated thrust.
- The same mixer as before converts `[F, τx, τy, τz]` → `[F1..F4]`, clipped to `[0, 5.47] N`.
- Position error is **saturated** (1.2 m in xy, 1.0 m in z) before it enters `K`. LQR is only
  optimal near hover; this makes large steps behave like a built-in speed limit instead of
  demanding huge tilt.
- The integrator is **conditional** (runs only within 0.8 m of the setpoint) and clamped, so it
  doesn't wind up during the initial climb.
- Everything is in the heading frame, so the 180° spawn yaw doesn't matter.

## Tuning (the LQR way)

There are no per-loop gains to hand-tune. You choose **Q** (how much each state error
hurts) and **R** (how much each input costs) in `default_weights()` in `lqr.py`, using
Bryson's rule: `Q_ii = 1/(acceptable error_i)²`, `R_ii = 1/(acceptable effort_i)²`.

| want | change |
|---|---|
| faster / tighter position response | decrease the "acceptable" position error (raises Q on ex, ey, ez) |
| less overshoot | increase Q on velocity (decrease its "acceptable" value) |
| gentler tilting / less aggressive | increase R on τx, τy (decrease their "acceptable" torque) |
| stronger wind / bias rejection | raise Q on the integral states; raise `integ_limit` |

You can also pass your own matrices: `DroneController(model, sp, Q=..., R=...)`.

## Checks I ran (simulation only)

| test | LQR (this repo) | previous PID |
|---|---|---|
| 7 obstacle-free setpoints, e.g. `(2,2,3)`, `(1,4,3)`, `(-4,-2,3.5)` | all PASS, settle 0.01–3.1 s | 3 PASS, 4 FAIL (x/y settled at 5.2–5.3 s, just past the 5 s deadline) |
| mass +25% / −20% (controller still assumes nominal) | PASS | z sags 1.1–1.4 m (no gravity-offset integrator → FAIL) |
| COM offset (2 cm, −1.5 cm) | PASS | x/y drift 0.5–0.8 m (FAIL) |
| steady 1.8 N horizontal wind from t = 8 s | peak dev 0.29 m, returns to 0 error | flips over |
| steady 3.6 N wind (≈ 24% of weight) | peak dev 0.57 m, back in band in 2.5 s, returns to 0 error | flips over |
| 90 s run at `(2,2,3)` | max error after 10 s: 0.0002 m | — |

Limits worth knowing: constant disturbances beyond roughly 4–5 N exceed what the integrator
clamp / motor headroom can reject (at 5 N the final x error was ≈ 0.39 m). Max tilt during a
`(2,2,3)` step is ≈ 19°, and the motors saturate for about 0.5 s during the initial climb from
0.3 m — expected for a 2.7 m climb in <2 s.

**Caveats:** all of this is in simulation with perfect state feedback (`qpos`/`qvel`). On the real
stack the state comes from WhyCon (x, y, z) and an IMU/estimator, so sensor noise, latency and
a velocity estimate matter; expect to soften the Q on velocity/rates and add filtering. Also,
setpoints whose straight-line path crosses an arena obstacle will crash into it (I hit
`obstacle_1` with `(-3, 4, 2)`) — the controller has no obstacle avoidance; that belongs in a planner.

## Porting to ROS 2 / the real bridge

- `DroneController.update(data, dt)` reads `data.qpos[0:3]`, `data.xmat`, `data.qvel[0:6]`.
  Replace those with your estimator outputs: position (WhyCon + altitude), world-frame velocity,
  roll/pitch/yaw, and body rates (gyro).
- It returns four motor thrusts in Newtons in actuator order `[m1, m2, m3, m4]`, same as before.
- `K` is a constant matrix: for a C++ node, run `python3 lqr.py`, copy the printed 4×15 `K`
  and the formula above — no Riccati solver is needed at runtime.


---

# Task 2C — Scan the arena, then navigate (no ROS 2)

```
top camera -> arena_scan -> order + lattice route -> waypoint_service
                                                          |
SimNode (MuJoCo + LQR) <-- setpoint -- NavigateServer <-- goals -- MissionClient
```

```bash
cd mission
python3 run_mission.py --expect C9,D5,H4,J7     # headless, ~5 s wall; prints scan table + verification
python3 run_mission.py --viewer                 # 3D viewer (macOS: mjpython run_mission.py --viewer)
python3 run_mission.py --order clockwise        # other visiting-order rule (left-to-right | clockwise | nearest)
python3 run_mission.py --route direct           # marker -> marker, skipping the lattice via-points
```
Writes `scan_result.png` (the annotated scan with plan + flown path), `mission_plot.png`, `mission_log.csv`,
`scan_result.json`. Exit code 0 = PASS. On headless Linux the offscreen renderer uses EGL automatically.

## What changed vs. the hard-coded version
The waypoint list is gone. `run_mission.py` now:
1. **Scans** — renders the arena's `top_cam` frame, segments red and yellow pixels, finds blobs, converts pixel -> world using
   the camera pose / field of view read from the model, and snaps to the nearest lattice intersection (A..K, 1..11; F6 = centre).
   A **target** = a red triangle with a yellow circle on it. Lone triangles / circles are reported but ignored. Only the
   lattice area is searched, so the red "eyantra" logo on the floor texture is never mistaken for a marker.
2. **Plans** — orders the targets and builds the route: start cell (from the drone's spawn pose) -> intermediate lattice
   points -> marker 1 -> ... -> marker 4.
3. **Serves** the plan through `waypoint_service`; the client flies it in exactly the returned order.

For the reference picture it finds C9, D5, H4, J7 and plans
`F6 > E7 > D8 > C9 > C8 > C7 > C6 > D5 > E4 > F4 > G4 > H4 > I5 > J6 > J7`, i.e. the dotted route in the picture.

**Waypoint kinds** (all at the flight altitude, default 13 m):
| kind | what the server requires | why |
|---|---|---|
| start (take-off above F6) | within +/-0.4 on X, Y, Z for 1 s | settle after the climb |
| via (intermediate lattice points) | just reach within +/-0.4 (0 s hold), then continue | follow the route without stopping |
| marker | within +/-0.4 on X, Y, Z for **3 s continuously** | the stabilisation requirement |

The client still waits for the server's result before sending the next goal. Settle time of a marker is measured from when
its goal is sent, so it is short (~1 s) because the drone arrives from the previous via-point already close by.

## The scene
`arena.xml` as uploaded has loose circles/triangles and three black pillars, which do not match the reference picture
(and `obstacle_1` sits right on D5, where it would hide marker 2 from the camera). So `model/make_scene.py` generates
`arena_scan.xml` / `drone_scan.xml` with composite markers at C9, D5, H4, J7 and no pillars. Your originals are untouched.
`python3 make_scene.py --markers B3,H8,K2,E10 --tri-only G3 --circle-only C6` builds a different layout (with decoys)
to prove the scanner is not tied to this one. The mission code never reads the scene file; it only sees the camera image.

## Results (simulation)
Reference scene, `--expect C9,D5,H4,J7`: **PASS**
- scan: 4/4 targets found, 1.0–1.4 cm from the lattice point, nothing else detected;
- 15/15 waypoints visited in plan order; 10/10 via points flown through; 36 s of simulated flight;
- markers: |error| <= 0.4 on X, Y and Z for >= 3 s each, settle times 0.85–1.25 s (server and independent log recomputation agree);
- START F6: 7.25 s (the 12.7 m climb).

Also tested: a different layout `B3,H8,K2,E10` with a lone-triangle and a lone-circle decoy (found exactly the 4 targets,
decoys ignored, localisation <= 1.9 cm, all PASS); `--order clockwise|nearest`; `--route direct`; forced timeout -> ABORTED;
second goal while busy -> REJECTED; cancel -> CANCELED.

## Decisions you may want to change
- **Visiting order.** The picture shows the order (C9, D5, H4, J7) but not the rule. Default is `left-to-right`
  (columns A -> K). `clockwise` around the start gives the same order here; `nearest` would go to D5 first. Edit
  `order_markers()` in `route_planner.py` if your task defines it differently.
- **Route between markers.** Every shortest 8-connected path over the lattice is equally short; the planner's tie-break is
  chosen so it reproduces the exact dots in your picture. Via-points do not avoid obstacles (there are none at 13 m).
- **Altitude 13 m** (from the previous task); `--altitude` changes it. Coordinates are the simulator's world metres.
- Perfect state feedback and a noise-free render, as before; real WhyCon/camera noise, lighting and perspective are not modelled.
