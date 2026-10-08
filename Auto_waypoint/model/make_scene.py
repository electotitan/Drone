#!/usr/bin/env python3
"""
Generate the "scan" scene: the arena with COMPOSITE MARKERS (a red triangle with a
yellow circle on top) at the given lattice cells, and no obstacles -- matching the
reference picture.

    python3 make_scene.py                       # C9,D5,H4,J7  (the reference picture)
    python3 make_scene.py --markers B3,H8,K2    # any other layout, to test the scanner

Writes  arena_scan.xml  and  drone_scan.xml  next to this file. Your original
arena.xml / drone.xml are NOT modified.

This file describes what is PHYSICALLY printed on the arena floor. The mission code
never reads it: it only ever sees the top-camera image (see mission/arena_scan.py).
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 1.203414                      # metres per lattice interval (see arena.xml comments)
COLS = "ABCDEFGHIJK"


def cell_to_world(label):
    col, row = COLS.index(label[0].upper()), int(label[1:])
    assert 1 <= row <= 11, f"row out of range in {label}"
    return (col - 5) * CELL, (6 - row) * CELL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markers", default="C9,D5,H4,J7", help="comma-separated lattice cells")
    ap.add_argument("--tri-only", default="", help="extra lone red triangles (decoys the scanner must ignore)")
    ap.add_argument("--circle-only", default="", help="extra lone yellow circles (decoys the scanner must ignore)")
    args = ap.parse_args()

    arena = open(os.path.join(HERE, "arena.xml")).read()
    head = arena[:arena.index("<worldbody>")].replace('model="khojo_arena"', 'model="khojo_arena_scan"')

    geoms = []
    def tri(c):
        x, y = cell_to_world(c)
        geoms.append(f'    <geom name="target_{c}_tri" type="mesh" mesh="triangle_marker" pos="{x:.4f} {y:.4f} 0.005" '
                     f'rgba="1 0 0 1" contype="0" conaffinity="0"/>')
    def circ(c):
        x, y = cell_to_world(c)
        geoms.append(f'    <geom name="target_{c}_circle" type="cylinder" size="0.25 0.003" pos="{x:.4f} {y:.4f} 0.012" '
                     f'rgba="1 1 0 1" contype="0" conaffinity="0"/>')
    for c in [s.strip() for s in args.markers.split(",") if s.strip()]:
        tri(c); circ(c)                                   # composite marker = triangle + circle on top
    for c in [s.strip() for s in args.tri_only.split(",") if s.strip()]:
        x, y = cell_to_world(c)
        geoms.append(f'    <geom name="decoy_{c}_tri" type="mesh" mesh="triangle_marker" pos="{x:.4f} {y:.4f} 0.005" '
                     f'rgba="1 0 0 1" contype="0" conaffinity="0"/>')
    for c in [s.strip() for s in args.circle_only.split(",") if s.strip()]:
        x, y = cell_to_world(c)
        geoms.append(f'    <geom name="decoy_{c}_circle" type="cylinder" size="0.25 0.003" pos="{x:.4f} {y:.4f} 0.012" '
                     f'rgba="1 1 0 1" contype="0" conaffinity="0"/>')

    body = f"""<worldbody>
    <light pos="0 0 10" dir="0 0 -1" directional="true" castshadow="false"
           diffuse="0.9 0.9 0.9" specular="0.1 0.1 0.1"/>
    <light pos="0 0 20" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.0 0.0 0.0" castshadow="false"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="floor_mat" contype="1" conaffinity="1"/>
    <geom name="arena_floor" type="plane" size="8.6603 8.6603 0.01" pos="0 0 0.001"
          material="arena_mat" contype="0" conaffinity="0"/>

    <!-- Composite markers (red triangle + yellow circle) at: {args.markers} . No obstacles in this scene. -->
{chr(10).join(geoms)}

    <camera name="top_cam" pos="0 0 15" xyaxes="1 0 0 0 1 0" fovy="60"/>
  </worldbody>
</mujoco>
"""
    open(os.path.join(HERE, "arena_scan.xml"), "w").write(head + body)

    drone = open(os.path.join(HERE, "drone.xml")).read()
    assert '<include file="arena.xml"/>' in drone
    open(os.path.join(HERE, "drone_scan.xml"), "w").write(drone.replace('<include file="arena.xml"/>',
                                                                        '<include file="arena_scan.xml"/>'))
    print(f"wrote arena_scan.xml + drone_scan.xml with markers at {args.markers}")


if __name__ == "__main__":
    main()
