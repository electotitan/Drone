# Swift Pico — Position Hold (MuJoCo)

A standalone MuJoCo + Python controller for `drone.xml` that drives the
Swift Pico to a target `(x, y, z)` — **Pitch, Roll, Throttle** in eYRC's
terms — and checks, on **each axis independently**:

> error settles within **±0.4 m** of that axis's setpoint within **5 s**,
> and stays inside that band for **10 s** straight.

Verified against the actual model in this repo for both the trivial case
(hover in place at `(0,0,3)`, since that's where it spawns) and a real
3-axis step (`(2,2,3)`, i.e. Pitch and Roll actually have to move, not
just Throttle): every axis settles by **t ≈ 2.0–2.3 s** and holds cleanly
— tested out to 60 s continuous. See `altitude_plot.png` after you run it
for the per-axis response.

**If your Task 1B target isn't `(2, 2, 3)`**, pass your real setpoint with
`--setpoint X Y Z` (see below) — the controller itself needs no changes,
only the target.

This is independent of the ROS 2 / `mujoco_bridge.cpp` workspace
(`pico_mujoco_ws`) — it talks to the MuJoCo model directly via the Python
API, so you can use it to tune/verify gains before wiring them into the
real bridge. If you want it as a ROS 2 node instead, see "Porting to ROS 2"
below.

## Project layout

```
pico_altitude_hold/
├── model/
│   ├── drone.xml          # your uploaded model (unchanged)
│   ├── arena.xml          # your uploaded arena (unchanged)
│   └── meshes/...         # your uploaded meshes/textures (unchanged)
├── controller/
│   ├── pid.py             # generic PID (anti-windup, output clamp)
│   ├── controller.py       # DroneController: PID stack + mixer
│   └── run_sim.py         # entry point: simulate, verify, plot
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

(`mujoco`, `numpy`, `matplotlib`. matplotlib is only needed for the plot —
the pass/fail check still runs without it.)

## Run it

Headless (fastest, what you'd use in CI / for grading):

```bash
cd controller
python3 run_sim.py
```

Prints a PASS/FAIL report, writes `log.csv` (t, z, x, y) and
`altitude_plot.png`. Exits with code 0 on PASS, 1 on FAIL, so it's
script-friendly.

With the interactive 3D viewer (watch the drone fly):

```bash
python3 run_sim.py --viewer
```

Useful flags:

```bash
python3 run_sim.py --duration 30                       # simulate longer
python3 run_sim.py --setpoint 2.0 2.0 3.0 --band 0.4    # x y z target, different band
python3 run_sim.py --viewer --realtime                  # (headless mode too) throttle to real time
```

Run `python3 run_sim.py --help` for the full list.

## How the controller works (`controller.py`)

Cascaded PID, computed fresh each physics step (`dt = 0.005 s`, from
`arena.xml`'s `<option timestep>`):

1. **Altitude loop** — PID on `z` error with `mg` feedforward
   (`total_mass` is read straight off the model, `1.5 kg` body +
   propeller meshes ≈ `1.525 kg` total) → desired vertical force `Fz`.
2. **XY loop** — PD on `x`/`y` error (Pitch/Roll targets) → a desired
   horizontal force `Fx, Fy`. Besides driving to a non-origin setpoint,
   this also cancels a very slight (~0.1°) mass asymmetry between the
   `m1_link` and `m4_link` meshes that would otherwise cause a slow
   horizontal drift even when holding position at `(0,0)`.
3. **Thrust vectoring** — `[Fx, Fy, Fz]` is treated as the desired total
   thrust vector; its direction is the desired body +Z axis
   (standard geometric quadrotor control), its magnitude (clipped to
   the actuator limit) is the commanded total thrust.
4. **Attitude loop** — roll/pitch PD (gains sized from the body's own
   `Ixx`/`Iyy` in `drone.xml` for a fast, well-damped ~10 rad/s inner
   loop — see note below) drives the actual tilt toward that desired
   direction; yaw PID holds the spawn heading.
5. **Mixer** (`mix()` in `controller.py`) converts
   `[F_total, τx, τy, τz]` into the 4 individual motor thrusts using the
   model's actual motor arm (`0.19 m`) and reactive-yaw coefficient
   (`0.016`, both read from the `<site>`/`<motor gear>` values in
   `drone.xml`), then clips each to `[0, 5.47] N` — the real
   `ctrlrange`.

**Why the loop gains are what they are:** the attitude loop must be
markedly faster/better-damped than the position loop or the cascade
rings — a slowly growing oscillation that looks fine for the first
~15 s and then diverges. The gains in `controller.py` were picked so the
attitude loop's natural frequency (~10 rad/s) is ~10x the position
loop's (~0.9 rad/s); that's what actually gets you the flat, non-diverging
response in `altitude_plot.png`. If you retune anything, keep that
separation in mind, and re-run with `--duration 60` before trusting it —
a 20 s test isn't long enough to catch that failure mode.

## Verifying the criterion yourself

The check (`check_settling()` in `run_sim.py`) is literal, and is run
**separately for x, y, and z**: for each axis it scans the logged values
and finds the *first* time index after which `|value - setpoint| <= band`
holds for every sample over the next `hold_duration` seconds, then checks
that time is `<= settle_deadline`. The overall result is PASS only if all
three axes individually pass. Change `--settle-deadline` / `--hold-duration`
/ `--band` if your grading criterion differs from the one above.

## Porting to ROS 2 / the real bridge

This script talks to MuJoCo directly (`mujoco.MjModel`, `d.ctrl[:] = ...`)
rather than through `pico_mujoco_ws`'s `mujoco_bridge.cpp`, because that
bridge file wasn't part of what you uploaded here, so I can't match its
topic/message interface. To wire this into the real ROS 2 node:
- `DroneController.update(data, dt)` → replace `data.qpos`/`data.qvel`
  reads with your pose/velocity source (e.g. `/whycon/poses` for x,y and
  a filtered altitude estimate for z).
- The four returned motor forces are already in Newtons in actuator
  order `[m1, m2, m3, m4]` — matches `drone.xml`'s `<actuator>` block, so
  they can be published directly if your bridge accepts per-motor thrust,
  or converted to whatever RC/PWM convention `mujoco_bridge.cpp` expects.

If you share `mujoco_bridge.cpp` and the `swift_pico` controller node,
I can adapt this to publish/subscribe on the actual topics instead.
