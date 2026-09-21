# circle_drawer

A ROS 2 Humble package that uses `turtlesim` to draw a circle of **diameter 2.0**, centered at **(5.0, 5.0)**, and leaves the turtle parked exactly at the center when done.

## How it works

The node (`draw_circle`) doesn't drive the turtle with velocity commands — it uses the `/turtle1/teleport_absolute` and `/turtle1/set_pen` services directly, so the circle is geometrically exact rather than approximated by a PID controller:

1. **Pen up**, teleport to the circle's starting point `(6.0, 5.0)` — the rightmost edge of the circle — so no stray line is drawn from the turtle's default spawn position.
2. **Pen down**, then teleport through 360 points around the circle using
   `x = 5.0 + cos(θ)`, `y = 5.0 + sin(θ)` for `θ` from `0` to `2π`.
3. **Pen up**, teleport back to the exact center `(5.0, 5.0)`.

## Package layout

```
circle_drawer/
├── circle_drawer/
│   ├── __init__.py
│   └── draw_circle.py      # the node
├── resource/
│   └── circle_drawer
├── package.xml
├── setup.py
└── setup.cfg
```

## Build

From the root of your ROS 2 workspace (e.g. `~/ros2_ws`), with this package under `src/`:

```bash
colcon build --packages-select circle_drawer
source install/setup.bash
```

## Run

Terminal 1 — start turtlesim:
```bash
ros2 run turtlesim turtlesim_node
```

Terminal 2 — run the node:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run circle_drawer draw_circle
```

## Parameters

Circle center and diameter are set as constants at the top of `draw_circle.py`:

```python
self.cx = 5.0
self.cy = 5.0
self.diameter = 2.0
```

Change these and rebuild (or just re-source, since Python files aren't compiled) to draw a different circle.

## Requirements

- ROS 2 Humble
- `turtlesim` package (`sudo apt install ros-humble-turtlesim` if not already installed)
