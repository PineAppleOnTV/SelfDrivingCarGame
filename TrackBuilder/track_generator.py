"""
track_generator.py
===================

Generates random, valid closed-loop tracks out of straight, 90-degree corner
and crossing pieces, and renders them as PNG images.

Key ideas
---------
- The track lives on an integer grid. Every grid cell holds exactly one piece:
  a straight, a corner, or a crossing.
- A "cell" is always the same physical size (`TrackConfig.cell_size`), no
  matter what piece sits in it, so pieces always line up. Corner geometry
  (inner radius + track width) fixes that size; a straight piece is centered
  inside its cell (see NOTES below if straight_length != cell_size).
- Crossings can be entered from any of the 4 sides, but a path is only ever
  allowed to go *straight through* a crossing, never turn - exactly as
  requested. A single crossing cell may be used twice by the same loop (once
  along each axis), which is what lets a track cross itself (figure-8 style).
- Track generation respects a limited inventory of pieces if you give it one
  (e.g. "I only have 10 straights, 8 corners, 2 crossings").

NOTES on straight_length vs cell_size
--------------------------------------
cell_size is derived from the corner geometry: cell_size = 2*inner_radius +
track_width. With the defaults (inner_radius=7, track_width=53) that's 67 px.
If straight_length (default 63) is not exactly equal to cell_size, the
straight piece's artwork is centered in its cell, leaving a small gap
(here: 2px on each end) in the border lines where two pieces meet. It's
cosmetic only (background and track are both white), but if you want
perfectly continuous border lines, set:

    straight_length == 2 * corner_inner_radius + track_width

or adjust corner_inner_radius / track_width to match whatever straight_length
you want.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, FrozenSet

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrackConfig:
    track_width: int = 51          # width of the white lane
    border_width: int = 2          # thickness of the black border line
    straight_length: int = 71      # length of a straight piece (direction of travel)
    corner_inner_radius: int = 7   # radius from the pivot to the INSIDE edge of the lane

    # start / finish marker: two short black bars across the lane
    start_mark_width: int = 7      # size of each mark, across the lane
    start_mark_length: int = 2     # thickness of each mark, along the direction of travel
    start_mark_gap: int = 4        # gap between the two marks, along the direction of travel

    @property
    def outer_radius(self) -> int:
        return self.corner_inner_radius + self.track_width

    @property
    def cell_size(self) -> int:
        # Every piece is rendered into a square cell of this size so pieces
        # of any type always connect edge-to-edge on the grid.
        return 2 * self.corner_inner_radius + self.track_width

    @property
    def margin(self) -> int:
        # empty (white) strip between the lane and the edge of a cell
        return self.corner_inner_radius


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

# heading = direction of travel. (row, col) grid, row grows downward.
DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
# turning left/right while moving with a given heading -> new heading
LEFT_OF = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}
AXIS = {"N": "V", "S": "V", "E": "H", "W": "H"}  # which axis a heading belongs to


# ---------------------------------------------------------------------------
# Piece / cell representation
# ---------------------------------------------------------------------------
# A cell's "piece" once finalized is one of:
#   ("straight", "H" | "V")
#   ("corner", frozenset({dir, dir}))   -- the two connected sides
#   ("crossing", None)


Piece = Tuple[str, object]


class GenerationError(Exception):
    pass


def _cell_piece_for_move(entry_dir: str, exit_dir: str) -> Piece:
    """Piece type/orientation implied by arriving via entry_dir and leaving via exit_dir."""
    if exit_dir == entry_dir:
        return ("straight", AXIS[entry_dir])
    if exit_dir == OPPOSITE[entry_dir]:
        raise GenerationError("U-turns are not a valid piece")
    from_side = OPPOSITE[entry_dir]  # the side of the cell we entered from
    to_side = exit_dir  # the side of the cell we exit through
    return ("corner", frozenset({from_side, to_side}))


# ---------------------------------------------------------------------------
# Track topology search
# ---------------------------------------------------------------------------


@dataclass
class GenResult:
    rows: int
    cols: int
    start_cell: Tuple[int, int]
    start_heading: str
    cells: Dict[Tuple[int, int], Piece]
    path: List[Tuple[Tuple[int, int], str]]  # (cell, heading-when-leaving-it), in order


def generate_track(
    piece_counts: Optional[Dict[str, Optional[int]]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    min_length: int = 8,
    allow_crossings: bool = True,
    max_node_visits: int = 200_000,
    rng: Optional[random.Random] = None,
) -> GenResult:
    """
    Randomly build ONE valid closed-loop track.

    piece_counts: e.g. {"straight": 12, "corner": 8, "crossing": 2}.
                  Use None (or omit a key) for "unlimited".
    grid_size:    (rows, cols). If omitted, a size is guessed from piece_counts.
    min_length:   minimum number of cells in the loop.
    allow_crossings: if False, the path never reuses a cell (simple loop only).
    """
    rng = rng or random.Random()
    counts = {"straight": None, "corner": None, "crossing": None}
    if piece_counts:
        counts.update(piece_counts)

    if grid_size is None:
        total = sum(v for v in counts.values() if v) or 16
        side = max(4, math.ceil(math.sqrt(total * 1.6)))
        grid_size = (side, side)
    rows, cols = grid_size

    start_cell = (rows // 2, cols // 2)
    start_heading = "E"
    # The start cell is a fixed W<->E straight. The loop begins by stepping
    # out of its East port; it must close by stepping INTO its East port from
    # the West-adjacent cell, i.e. the closing move also has heading "E".
    closing_heading = start_heading

    # visited[cell] = piece already committed there (or None)
    visited: Dict[Tuple[int, int], Piece] = {}
    remaining = dict(counts)
    node_visits = 0

    def in_bounds(cell):
        r, c = cell
        return 0 <= r < rows and 0 <= c < cols

    def can_afford(kind: str) -> bool:
        n = remaining.get(kind)
        return n is None or n > 0

    def spend(kind: str, delta: int):
        n = remaining.get(kind)
        if n is not None:
            remaining[kind] = n + delta

    # commit_cell / undo_cell mutate `visited` and `remaining`, and return an
    # "undo token" describing exactly how to revert the change.
    def commit_cell(cell, entry_dir, exit_dir):
        existing = visited.get(cell)
        if existing is None:
            piece = _cell_piece_for_move(entry_dir, exit_dir)
            kind = "crossing" if piece[0] == "crossing" else piece[0]
            if piece[0] == "corner" and not can_afford("corner"):
                return None
            if piece[0] == "straight" and not can_afford("straight"):
                return None
            visited[cell] = piece
            spend(piece[0], -1)
            return ("new", cell, piece)
        else:
            # cell already used once - only a straight-through reuse of the
            # OTHER axis is allowed, turning it into a crossing.
            if not allow_crossings:
                return None
            if existing[0] != "straight":
                return None  # corners/crossings can't be reused
            if exit_dir != entry_dir:
                return None  # must go straight through
            if AXIS[entry_dir] == existing[1]:
                return None  # same axis already used - would overlap, not cross
            if not can_afford("crossing"):
                return None
            old_piece = existing
            visited[cell] = ("crossing", None)
            spend("straight", +1)  # give back the straight we'd tentatively spent
            spend("crossing", -1)
            return ("upgrade", cell, old_piece)

    def undo(token):
        if token is None:
            return
        kind, cell, old = token
        if kind == "new":
            piece = visited.pop(cell)
            spend(piece[0], +1)
        else:  # upgrade
            spend("crossing", +1)
            spend("straight", -1)
            visited[cell] = old

    path: List[Tuple[Tuple[int, int], str]] = []

    def dfs(cell, heading, length):
        nonlocal node_visits
        node_visits += 1
        if node_visits > max_node_visits:
            raise GenerationError("search budget exceeded")

        choices = [heading, LEFT_OF[heading], RIGHT_OF[heading]]
        rng.shuffle(choices)
        # try closing the loop first sometimes, to keep tracks compact & varied
        try_close_first = rng.random() < 0.35 and length >= min_length

        def attempt_close():
            if length < min_length:
                return None
            next_cell = (cell[0] + DELTA[closing_heading][0], cell[1] + DELTA[closing_heading][1])
            if next_cell != start_cell:
                return None
            token = commit_cell(cell, heading, closing_heading)
            if token is None:
                return None
            path.append((cell, closing_heading))
            return token

        if try_close_first:
            token = attempt_close()
            if token is not None:
                return True
            # else fall through to normal exploration

        for new_heading in choices:
            next_cell = (cell[0] + DELTA[new_heading][0], cell[1] + DELTA[new_heading][1])
            if not in_bounds(next_cell):
                continue
            if next_cell == start_cell:
                continue  # closing must go through attempt_close(), not here
            if next_cell in visited and visited[next_cell][0] != "straight":
                continue  # can't re-enter a corner/crossing cell
            token = commit_cell(cell, heading, new_heading)
            if token is None:
                continue
            path.append((cell, new_heading))
            if dfs(next_cell, new_heading, length + 1):
                return True
            path.pop()
            undo(token)

        if not try_close_first:
            token = attempt_close()
            if token is not None:
                return True

        return False

    # reserve the start cell itself as a fixed straight running E-W
    visited[start_cell] = ("straight", "H")
    spend("straight", -1)

    # The start cell's own piece is already fixed above, so DFS proper begins
    # at its East neighbour (the cell you're in after taking the first step).
    first_cell = (start_cell[0] + DELTA[start_heading][0], start_cell[1] + DELTA[start_heading][1])
    if not in_bounds(first_cell):
        raise GenerationError("grid too small for the start piece")

    ok = dfs(first_cell, start_heading, 1)
    if not ok:
        raise GenerationError("could not find a closed loop with the given constraints")

    return GenResult(rows, cols, start_cell, start_heading, dict(visited), path)


def generate_track_variants(
    piece_counts: Optional[Dict[str, Optional[int]]] = None,
    n_variants: int = 4,
    grid_size: Optional[Tuple[int, int]] = None,
    min_length: int = 8,
    allow_crossings: bool = True,
    seed: Optional[int] = None,
    max_attempts: int = 200,
) -> List[GenResult]:
    """Try to build n_variants distinct valid tracks."""
    rng = random.Random(seed)
    results: List[GenResult] = []
    signatures = set()
    attempts = 0
    while len(results) < n_variants and attempts < max_attempts:
        attempts += 1
        try:
            res = generate_track(
                piece_counts=piece_counts,
                grid_size=grid_size,
                min_length=min_length,
                allow_crossings=allow_crossings,
                rng=rng,
            )
        except GenerationError:
            continue
        sig = tuple(sorted(res.cells.items(), key=lambda kv: kv[0]))
        sig = (sig,)
        key = repr(sig)
        if key in signatures:
            continue
        signatures.add(key)
        results.append(res)
    return results
