"""
Sequencing + routing of the detected markers.

order_markers(): which marker is visited when.
    left-to-right : columns A -> K (ties: top row first).  Gives C9, D5, H4, J7 for the reference picture.
    clockwise     : sweep clockwise around the start cell, from south. (same result on the reference picture)
    nearest       : greedy nearest-neighbour from the start. (would go D5 first -> differs from the picture)

plan_lattice_route(): between two cells, an 8-connected (straight + diagonal) shortest path over
lattice intersections -- the orange dotted route in the reference picture. All shortest paths are
equally short; the neighbour order below is just the tie-break that reproduces the picture's dots.
"""
import heapq
import math

import lattice

# (d_col, d_row); tie-break order chosen so the planner reproduces the reference picture's route
_NEIGHBOURS = [(-1, 0), (0, -1), (1, 1), (0, 1), (1, -1), (1, 0), (-1, 1), (-1, -1)]
STRAIGHT, DIAG = 10, 14                              # integer octile costs -> exact ties


def order_markers(cells, start_cell, strategy="left-to-right"):
    pts = {c: lattice.parse(c) for c in cells}
    sc = lattice.parse(start_cell)
    if strategy == "left-to-right":
        return sorted(cells, key=lambda c: (pts[c][0], pts[c][1]))
    if strategy == "clockwise":
        # clockwise sweep around the start cell, beginning at "south" (6 o'clock): SW, W, NW, N, NE, E, SE
        def key(c):
            ang_ccw = math.atan2(-(pts[c][1] - sc[1]), pts[c][0] - sc[0])   # row grows downward -> -(d_row)
            cw_from_south = ((-ang_ccw) - math.pi / 2) % (2 * math.pi)
            return (cw_from_south, math.hypot(pts[c][0] - sc[0], pts[c][1] - sc[1]))
        return sorted(cells, key=key)
    if strategy == "nearest":
        left, cur, out = set(cells), sc, []
        while left:
            nxt = min(left, key=lambda c: (math.hypot(pts[c][0] - cur[0], pts[c][1] - cur[1]), pts[c]))
            out.append(nxt); left.remove(nxt); cur = pts[nxt]
        return out
    raise ValueError(f"unknown order strategy '{strategy}'")


def plan_lattice_route(a_cell, b_cell):
    """Intermediate lattice cells (excluding both ends) of the shortest 8-connected path a -> b."""
    a, b = lattice.parse(a_cell), lattice.parse(b_cell)

    def h(p):
        dx, dy = abs(p[0] - b[0]), abs(p[1] - b[1])
        return DIAG * min(dx, dy) + STRAIGHT * (max(dx, dy) - min(dx, dy))

    cnt, pq, g, parent = 0, [(h(a), 0, 0, a)], {a: 0}, {}
    while pq:
        _, _, gc, u = heapq.heappop(pq)
        if u == b:
            break
        if gc > g[u]:
            continue
        for dc, dr in _NEIGHBOURS:
            v = (u[0] + dc, u[1] + dr)
            if not lattice.inside(*v):
                continue
            c = gc + (DIAG if dc and dr else STRAIGHT)
            if c < g.get(v, 1 << 30):
                g[v], parent[v] = c, u
                cnt += 1
                heapq.heappush(pq, (c + h(v), cnt, c, v))
    path = [b]
    while path[-1] != a:
        path.append(parent[path[-1]])
    path.reverse()
    return [lattice.label(*p) for p in path[1:-1]]
