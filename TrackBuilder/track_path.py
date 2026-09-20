"""
track_path.py
==============
Turns a generated track (track_generator.GenResult) into an actual drivable
path, optionally leaning toward a "racing line" (corner-cutting, running
wide before a corner if that gives a smoother line, taking a straighter path
through a chicane instead of chasing every apex) rather than the plain
centerline, and samples that path into evenly-spaced checkpoints for
training an ML model.

Crossings are always taken with a fixed direction of travel - a crossing is
geometrically a straight line (never an arc), so it is physically impossible
for the path to turn left or right there, regardless of any racing-line
offset. The racing line's lateral position is still free to drift smoothly
through a crossing (same as it would through any plain straight cell) if
that's what the corners on either side call for.

Geometry model
--------------
The loop is walked as a sequence of "moves" (one per cell visited - a
crossing used twice contributes two moves, one per pass). Each move is
either a straight line (straight or crossing piece) or a circular arc
(corner piece, radius = corner_inner_radius + track_width/2, sweeping 90
degrees around the same pivot used to draw the piece).

The racing line itself is built by relaxation, not a hand-picked target per
corner: the centerline is sampled finely all the way around the loop, each
sample point is given a lateral degree of freedom (how far it may move
sideways, up to `aggressiveness` x the safety margin), and then each point
is repeatedly pulled toward the midpoint of its two neighbours (clamped to
its allowed range) for a few hundred iterations. This is a standard, simple
curvature-reducing relaxation: pulling a point toward the chord between its
neighbours is exactly what straightens a curve out, and because it only
ever looks at *local* neighbours, it naturally produces the right emergent
behaviour without any special-casing:
  - it cuts to the inside of an isolated corner (a chord across an arc
    always falls inside the arc),
  - it drifts wide beforehand if that makes the overall path straighter,
  - and for a tight run of alternating corners (a chicane) it settles on a
    straighter compromise path instead of snapping to every apex, since
    zig-zagging to each apex is exactly the high-curvature situation this
    relaxation smooths away.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from track_generator import (
    TrackConfig, GenResult, DELTA, OPPOSITE, LEFT_OF, RIGHT_OF, AXIS,
)

# side -> local (x, y) fraction of the cell size, for that side's edge midpoint
_SIDE_MIDPOINT_FRAC = {
    "N": (0.5, 0.0),
    "S": (0.5, 1.0),
    "E": (1.0, 0.5),
    "W": (0.0, 0.5),
}

# For each pair of connected sides on a corner piece: the pivot (as a
# fraction of cell size) and the angle (PIL/standard math convention,
# degrees, measured the same way as track_render._CORNER_GEOMETRY) at which
# the arc touches each of the two sides.
_CORNER_ARC = {
    frozenset({"N", "W"}): {"pivot": (0.0, 0.0), "N": 0.0, "W": 90.0},
    frozenset({"N", "E"}): {"pivot": (1.0, 0.0), "E": 90.0, "N": 180.0},
    frozenset({"S", "E"}): {"pivot": (1.0, 1.0), "S": 180.0, "E": 270.0},
    frozenset({"S", "W"}): {"pivot": (0.0, 1.0), "W": 270.0, "S": 360.0},
}


@dataclass
class _Move:
    cell: Tuple[int, int]
    kind: str        # 'line' | 'arc'
    entry_dir: str
    exit_dir: str


def _ordered_moves(result: GenResult) -> List[_Move]:
    """Full ordered loop of moves, including the fixed start cell. A crossing
    cell used twice appears as two separate moves (one per pass)."""
    moves = [_Move(result.start_cell, "line", result.start_heading, result.start_heading)]
    heading = result.start_heading
    for cell, exit_heading in result.path:
        piece = result.cells[cell]
        kind = "arc" if piece[0] == "corner" else "line"
        moves.append(_Move(cell, kind, heading, exit_heading))
        heading = exit_heading
    return moves


class _Segment:
    """One straight or arc piece of the loop, in absolute canvas coordinates."""

    def __init__(self, move: _Move, cfg: TrackConfig):
        self.move = move
        self.cfg = cfg
        s = cfg.cell_size
        ox, oy = move.cell[1] * s, move.cell[0] * s  # cell origin (col, row) -> (x, y)
        self.origin = (ox, oy)

        if move.kind == "line":
            entry_side = OPPOSITE[move.entry_dir]
            exit_side = move.exit_dir
            ex, ey = _SIDE_MIDPOINT_FRAC[entry_side]
            xx, xy = _SIDE_MIDPOINT_FRAC[exit_side]
            self.p0 = (ox + ex * s, oy + ey * s)
            self.p1 = (ox + xx * s, oy + xy * s)
            self.length = math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])
            # unit tangent, and the unit "left of travel" normal
            dx, dy = (self.p1[0] - self.p0[0]) / self.length, (self.p1[1] - self.p0[1]) / self.length
            self.tangent = (dx, dy)
            self.left_normal = (-dy, dx)
            self.turn_sign = 0
        else:
            entry_side = OPPOSITE[move.entry_dir]
            exit_side = move.exit_dir
            geo = _CORNER_ARC[frozenset({entry_side, exit_side})]
            pfx, pfy = geo["pivot"]
            self.pivot = (ox + pfx * s, oy + pfy * s)
            self.angle0 = math.radians(geo[entry_side])
            self.angle1 = math.radians(geo[exit_side])
            self.r_center = cfg.corner_inner_radius + cfg.track_width / 2.0
            self.length = self.r_center * abs(self.angle1 - self.angle0)
            # +1 if angle increases along travel (one rotational sense), else -1
            self.turn_sign = 1 if self.angle1 > self.angle0 else -1

    def centerline_and_normal(self, t: float) -> Tuple[float, float, float, float]:
        """t in [0,1]. Returns (x, y, nx, ny): the centerline point and the
        unit direction in which a *positive* lateral offset moves it (chosen
        so that, for an arc, positive offset = toward the pivot = tighter)."""
        if self.move.kind == "line":
            cx = self.p0[0] + (self.p1[0] - self.p0[0]) * t
            cy = self.p0[1] + (self.p1[1] - self.p0[1]) * t
            return cx, cy, self.left_normal[0], self.left_normal[1]
        else:
            angle = self.angle0 + (self.angle1 - self.angle0) * t
            cx = self.pivot[0] + self.r_center * math.cos(angle)
            cy = self.pivot[1] + self.r_center * math.sin(angle)
            nx = -self.turn_sign * math.cos(angle)
            ny = -self.turn_sign * math.sin(angle)
            return cx, cy, nx, ny


def _build_segments(result: GenResult, cfg: TrackConfig) -> List[_Segment]:
    moves = _ordered_moves(result)
    return [_Segment(m, cfg) for m in moves]


def _build_fine_centerline(segments: List[_Segment], sample_step: float
                            ) -> List[Tuple[float, float, float, float]]:
    """(x, y, nx, ny) for every fine sample point all the way around the
    loop, cyclic (the last point's neighbour is the first point) with no
    duplicated seam point."""
    points = []
    for seg in segments:
        n_steps = max(1, int(math.ceil(seg.length / sample_step)))
        for k in range(n_steps):  # excludes t=1, which is the next point's t=0
            points.append(seg.centerline_and_normal(k / n_steps))
    return points


def _segment_index_ranges(segments: List[_Segment], sample_step: float) -> List[Tuple[int, int]]:
    """[start, count) fine-point index range for each segment, matching
    _build_fine_centerline's layout exactly."""
    ranges = []
    start = 0
    for seg in segments:
        n_steps = max(1, int(math.ceil(seg.length / sample_step)))
        ranges.append((start, n_steps))
        start += n_steps
    return ranges


def _pre_corner_swing_bias(segments: List[_Segment], sample_step: float, max_offset: float,
                            swing_distance: float, swing_strength: float,
                            turn_in_distance: float,
                            excluded_indices: Optional[set] = None) -> List[float]:
    """An extra, persistent lateral pull applied only in the last
    `swing_distance` pixels of a plain straight/crossing that leads directly
    into a corner - it nudges the line toward the OUTSIDE of that corner
    beforehand (classic "out" before the "in"). It's deliberately *not*
    applied when the preceding piece is itself a corner (a chicane), so it
    can never fight the chicane-straightening behaviour - it only ever
    kicks in when there's a plain, generous straight approach to swing
    across in the first place.

    The push isn't held at full strength all the way to the corner: it
    peaks `turn_in_distance` pixels before the corner and eases back off
    over that last stretch, so the line has already started turning back in
    before it actually reaches the corner, instead of being held hard on
    the outside and then having to snap to the apex all at once.
    turn_in_distance is clamped to stay smaller than swing_distance.

    If the push's window would cover any index in `excluded_indices` (e.g.
    the start-line anchor), that corner is skipped entirely rather than
    fighting a fixed point - a corner that happens to fall shortly after the
    start just doesn't get the outward push."""
    n_seg = len(segments)
    ranges = _segment_index_ranges(segments, sample_step)
    total_points = ranges[-1][0] + ranges[-1][1]
    bias = [0.0] * total_points
    if swing_strength <= 0 or swing_distance <= 0:
        return bias
    turn_in_distance = max(0.0, min(turn_in_distance, swing_distance * 0.95))
    excluded_indices = excluded_indices or set()

    def ease(x):
        x = max(0.0, min(1.0, x))
        return (1 - math.cos(x * math.pi)) / 2.0

    for i, seg in enumerate(segments):
        if seg.move.kind != "arc":
            continue
        prev_i = (i - 1) % n_seg
        prev_seg = segments[prev_i]
        if prev_seg.move.kind != "line" or prev_seg.length < swing_distance * 0.8:
            continue  # only swing out of a plain, sufficiently long straight
        prev_start, prev_count = ranges[prev_i]
        step = prev_seg.length / prev_count
        n_window = max(2, min(prev_count, int(round(swing_distance / step))))
        window_start = prev_start + prev_count - n_window
        window_end = prev_start + prev_count - 1
        if any(window_start <= idx <= window_end for idx in excluded_indices):
            continue  # this corner's approach overlaps a fixed point (e.g. the start anchor) - skip it
        n_turn_in = max(1, min(n_window - 1, int(round(turn_in_distance / step))))
        for k in range(n_window):
            idx = prev_start + prev_count - n_window + k
            d = n_window - 1 - k  # steps before the corner: 0 = at the corner, larger = farther back
            if d >= n_turn_in:
                frac = (n_window - 1 - d) / max(1, n_window - 1 - n_turn_in)  # 0 (far) -> 1 (at turn-in point)
            else:
                frac = d / n_turn_in  # 1 (at turn-in point) -> 0 (right at the corner)
            bias[idx] += -seg.turn_sign * swing_strength * max_offset * ease(frac)
    return bias


def _crossing_linear_ranges(segments: List[_Segment], result: GenResult, sample_step: float
                             ) -> List[Tuple[int, int]]:
    """[start, count) fine-point ranges belonging to a crossing piece - used
    to force the line to be a plain straight (linear in offset) all the way
    across it, exactly like it would connect two fixed points on any other
    straight, rather than being independently pulled around inside it."""
    ranges = _segment_index_ranges(segments, sample_step)
    out = []
    for i, seg in enumerate(segments):
        piece = result.cells.get(seg.move.cell)
        if piece is not None and piece[0] == "crossing":
            out.append(ranges[i])
    return out


def _relax_offsets(points: List[Tuple[float, float, float, float]], max_offset: float,
                    iterations: int = 400, damping: float = 0.5,
                    bias: Optional[List[float]] = None,
                    fixed: Optional[Dict[int, float]] = None,
                    linear_ranges: Optional[List[Tuple[int, int]]] = None) -> List[float]:
    """Repeatedly pulls each point toward the midpoint of its two neighbours
    (projected onto that point's own lateral freedom), plus an optional
    fixed `bias` term (see _pre_corner_swing_bias), clamped to
    [-max_offset, max_offset]. The neighbour-pulling part is a standard
    curvature-reducing relaxation - it's what makes the line cut corners,
    and flatten out chicanes, as one emergent behaviour, since pulling a
    point toward the chord between its neighbours is exactly what
    straightens a curve out.

    `fixed`: {index: offset} - these points never move (used to anchor the
    start line exactly on the centerline).
    `linear_ranges`: [(start, count), ...] - the *interior* points of each
    range (excluding its own two endpoints) are held to a straight-line
    interpolation between those endpoints instead of being pulled by their
    immediate neighbours - used to make crossings behave as a single
    straight from wherever they're entered to wherever they're exited."""
    n = len(points)
    if n < 3 or max_offset <= 0:
        return [0.0] * n
    offsets = [0.0] * n
    bias = bias or [0.0] * n
    fixed = fixed or {}
    for idx, val in fixed.items():
        offsets[idx] = val

    linear_interior: Dict[int, Tuple[int, int]] = {}
    for start, count in (linear_ranges or []):
        if count >= 3:
            end = start + count - 1
            for k in range(start + 1, end):
                linear_interior[k] = (start, end)

    pos = [(cx, cy) for cx, cy, nx, ny in points]
    for idx, val in fixed.items():
        cx, cy, nx, ny = points[idx]
        pos[idx] = (cx + val * nx, cy + val * ny)

    for _ in range(iterations):
        for i in range(n):
            if i in fixed:
                continue  # anchored - never moves
            cx, cy, nx, ny = points[i]
            if i in linear_interior:
                s_idx, e_idx = linear_interior[i]
                t = (i - s_idx) / (e_idx - s_idx)
                target = offsets[s_idx] + (offsets[e_idx] - offsets[s_idx]) * t
            else:
                ip, inx = (i - 1) % n, (i + 1) % n
                tx = (pos[ip][0] + pos[inx][0]) / 2.0
                ty = (pos[ip][1] + pos[inx][1]) / 2.0
                target = (tx - cx) * nx + (ty - cy) * ny + bias[i]  # project onto the normal
            target = max(-max_offset, min(max_offset, target))
            offsets[i] += damping * (target - offsets[i])
            pos[i] = (cx + offsets[i] * nx, cy + offsets[i] * ny)
    return offsets


@dataclass
class Checkpoint:
    index: int
    x: float
    y: float
    heading_deg: float
    distance: float  # cumulative distance along the path, in pixels


def offset_checkpoints(checkpoints: List[Checkpoint], dx: float, dy: float) -> List[Checkpoint]:
    """Shift all checkpoints by (-dx, -dy) - use this to match the coordinate
    system of a trimmed/cropped track image (see track_render.assemble_track's
    return_offset option)."""
    return [Checkpoint(c.index, c.x - dx, c.y - dy, c.heading_deg, c.distance) for c in checkpoints]


def draw_checkpoints(img, checkpoints: List[Checkpoint], radius: float = 1.5,
                      color: Tuple[int, int, int] = (220, 30, 30), labels: bool = False):
    """Draws checkpoint markers directly onto a PIL image (in place)."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for c in checkpoints:
        d.ellipse([c.x - radius, c.y - radius, c.x + radius, c.y + radius], fill=color)
        if labels:
            d.text((c.x + radius + 2, c.y - radius), str(c.index), fill=color)
    return img


def export_checkpoints_csv(checkpoints: List[Checkpoint], path: str):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "heading_deg", "distance"])
        for c in checkpoints:
            w.writerow([c.index, round(c.x, 2), round(c.y, 2), round(c.heading_deg, 2), round(c.distance, 2)])


def generate_checkpoints(
    result: GenResult,
    cfg: TrackConfig,
    interval: float = 100.0,
    aggressiveness: float = 0.6,
    safety_margin: float = 6.0,
    sample_step: float = 2.0,
    smoothing_iterations: int = 400,
    pre_corner_swing: float = 0.4,
    pre_corner_swing_distance: float = 50.0,
    pre_corner_turn_in_distance: float = 25.0,
    start_offset: float = 20.0,
) -> List[Checkpoint]:
    """
    interval:                     distance in pixels between checkpoints along the path.
    aggressiveness:                0 = plain centerline, 1 = allowed to roam as
                                   close to the lane edges as `safety_margin` permits.
    safety_margin:                 pixels kept clear between the racing line and
                                   the lane border at full aggressiveness.
    sample_step:                   how finely (in centerline pixels) to sample
                                   the path - smaller is more accurate and
                                   slower. 2px is plenty for track pieces this size.
    smoothing_iterations:          how many relaxation passes to run. More gives
                                   a smoother line; a few hundred is enough to
                                   converge for tracks of this scale.
    pre_corner_swing:              0 disables it entirely. Otherwise, how hard
                                   (as a fraction of the max lateral offset) the
                                   line is nudged toward the OUTSIDE of a corner
                                   in the stretch immediately before it - the
                                   classic "out" before the "in". Only ever
                                   applied on a plain, sufficiently long
                                   straight/crossing approach, never between
                                   corners in a chicane, so it can't reintroduce
                                   apex-chasing zig-zag there.
    pre_corner_swing_distance:     how many pixels before the corner the
                                   outward push starts building up.
    pre_corner_turn_in_distance:   how many pixels before the corner the
                                   outward push has fully eased off again, so
                                   the line has already started turning back
                                   in before it reaches the corner instead of
                                   holding the outside line right up to the
                                   apex and snapping in all at once. Always
                                   kept smaller than pre_corner_swing_distance
                                   (clamped automatically if you set it too
                                   close to it).
    start_offset:                  checkpoint 0 sits at the middle of the start
                                   straight, shifted backward (against the
                                   direction of travel) by this many pixels.
    """
    segments = _build_segments(result, cfg)
    points = _build_fine_centerline(segments, sample_step)
    max_offset = max(0.0, aggressiveness) * max(0.0, cfg.track_width / 2.0 - safety_margin)

    # The start line is a fixed, physical mark drawn at the exact centre of
    # the start straight - anchor the racing line to sit exactly on the
    # centerline right there, regardless of aggressiveness.
    seg0_start, seg0_count = _segment_index_ranges(segments, sample_step)[0]
    mid_idx = seg0_start + seg0_count // 2

    bias = _pre_corner_swing_bias(segments, sample_step, max_offset,
                                   pre_corner_swing_distance, pre_corner_swing,
                                   pre_corner_turn_in_distance,
                                   excluded_indices={mid_idx})
    linear_ranges = _crossing_linear_ranges(segments, result, sample_step)

    offsets = _relax_offsets(points, max_offset, iterations=smoothing_iterations, bias=bias,
                              fixed={mid_idx: 0.0}, linear_ranges=linear_ranges)

    n = len(points)
    fine = [(cx + o * nx, cy + o * ny) for (cx, cy, nx, ny), o in zip(points, offsets)]

    # cumulative true distance around the loop (cum[n] = total loop length)
    cum = [0.0] * (n + 1)
    for i in range(n):
        x0, y0 = fine[i]
        x1, y1 = fine[(i + 1) % n]
        cum[i + 1] = cum[i] + math.hypot(x1 - x0, y1 - y0)
    total_true_length = cum[n]

    # headings via a central finite difference (cyclic) on the final line
    headings = []
    for i in range(n):
        xm, ym = fine[(i - 1) % n]
        xp, yp = fine[(i + 1) % n]
        headings.append(math.atan2(yp - ym, xp - xm))

    # Checkpoint 0's reference point: the same start-line middle used as the
    # anchor above, shifted backward by start_offset pixels.
    back_steps = int(round(start_offset / sample_step))
    start_idx = (mid_idx - back_steps) % n
    base_distance = cum[start_idx]

    checkpoints = []
    n_checkpoints = max(1, int(total_true_length // interval))
    fi = start_idx
    prev_wrapped = None
    for i in range(n_checkpoints):
        wrapped_d = (base_distance + i * interval) % total_true_length
        if prev_wrapped is not None and wrapped_d < prev_wrapped:
            fi = 0  # crossed the loop-closure point - restart the scan from index 0
        while fi < n - 1 and cum[fi + 1] < wrapped_d:
            fi += 1
        i0, i1 = fi, (fi + 1) % n
        d0, d1 = cum[fi], cum[fi + 1]
        frac = 0.0 if d1 <= d0 else (wrapped_d - d0) / (d1 - d0)
        x = fine[i0][0] + (fine[i1][0] - fine[i0][0]) * frac
        y = fine[i0][1] + (fine[i1][1] - fine[i0][1]) * frac
        dh = (headings[i1] - headings[i0] + math.pi) % (2 * math.pi) - math.pi
        h = headings[i0] + dh * frac
        checkpoints.append(Checkpoint(i, x, y, math.degrees(h) % 360.0, i * interval))
        prev_wrapped = wrapped_d

    return checkpoints