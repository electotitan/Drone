#!/usr/bin/env python3
"""
Generate the RESCUE scene (Task 3): your arena.xml with its three black obstacle boxes kept
exactly as they are, and the original loose markers replaced by

    critical survivors  = red triangles    (default I2, J9, B10)
    stable survivors    = yellow circles   (default B2, J6, F10)

    python3 make_rescue_scene.py
    python3 make_rescue_scene.py --critical A3,K8,E11 --stable C2,G7,J10     # other layout

Writes arena_rescue.xml + drone_rescue.xml next to this file (originals untouched).
This describes what is physically on the floor; the mission code never reads it -- it only
sees the top-camera image. (The verifier reads it to grade the scan.)
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 1.203414
COLS = "ABCDEFGHIJK"


def cell_to_world(c):
    col, row = COLS.index(c[0].upper()), int(c[1:])
    assert 1 <= row <= 11, c
    return (col - 5) * CELL, (6 - row) * CELL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--critical", default="I2,J9,B10")
    ap.add_argument("--stable", default="B2,J6,F10")
    args = ap.parse_args()

    arena = open(os.path.join(HERE, "arena.xml")).read()
    i_wb = arena.index("<worldbody>")
    i_markers = arena.index("<!-- Flat ground markers")
    i_cam = arena.index("<!--\n      Top-down whycon camera")
    head = arena[:i_wb].replace('model="khojo_arena"', 'model="khojo_arena_rescue"')
    world_and_obstacles = arena[i_wb:i_markers]          # lights, floors, the 3 obstacle boxes (unchanged)
    camera = arena[i_cam:]                               # top_cam + closing tags

    g = []
    for c in [s.strip() for s in args.critical.split(",") if s.strip()]:
        x, y = cell_to_world(c)
        g.append(f'    <geom name="critical_{c}_tri" type="mesh" mesh="triangle_marker" pos="{x:.4f} {y:.4f} 0.005" '
                 f'rgba="1 0 0 1" contype="0" conaffinity="0"/>')
    for c in [s.strip() for s in args.stable.split(",") if s.strip()]:
        x, y = cell_to_world(c)
        g.append(f'    <geom name="stable_{c}_circle" type="cylinder" size="0.5 0.005" pos="{x:.4f} {y:.4f} 0.005" '
                 f'rgba="1 1 0 1" contype="0" conaffinity="0"/>')
    markers = ("<!-- Survivors. critical (red triangle): " + args.critical + " ; stable (yellow circle): " + args.stable +
               " -->\n" + "\n".join(g) + "\n\n    ")
    open(os.path.join(HERE, "arena_rescue.xml"), "w").write(head + world_and_obstacles + markers + camera)

    drone = open(os.path.join(HERE, "drone.xml")).read()
    assert '<include file="arena.xml"/>' in drone
    open(os.path.join(HERE, "drone_rescue.xml"), "w").write(
        drone.replace('<include file="arena.xml"/>', '<include file="arena_rescue.xml"/>'))
    print(f"wrote arena_rescue.xml + drone_rescue.xml  critical={args.critical}  stable={args.stable}")


if __name__ == "__main__":
    main()
