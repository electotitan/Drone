"""Arena lattice: 11 x 11 intersections, columns A..K (left->right), rows 1..11 (top->bottom).
Cell F6 is the arena centre = world (0, 0). World x is right, y is up (as seen by the top camera)."""
CELL = 1.203414          # metres between lattice intersections (from arena.xml)
COLS = "ABCDEFGHIJK"
N_COLS, N_ROWS = 11, 11
CENTRE_COL, CENTRE_ROW = 5, 6


def label(col, row):
    return f"{COLS[col]}{row}"


def parse(lbl):
    return COLS.index(lbl[0].upper()), int(lbl[1:])


def idx_to_world(col, row):
    return (col - CENTRE_COL) * CELL, (CENTRE_ROW - row) * CELL


def cell_to_world(lbl):
    return idx_to_world(*parse(lbl))


def world_to_idx(x, y):
    """Nearest lattice intersection -> (col, row, distance_m_to_it)."""
    col = int(round(x / CELL)) + CENTRE_COL
    row = CENTRE_ROW - int(round(y / CELL))
    cx, cy = idx_to_world(col, row)
    return col, row, float(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5)


def inside(col, row):
    return 0 <= col < N_COLS and 1 <= row <= N_ROWS
