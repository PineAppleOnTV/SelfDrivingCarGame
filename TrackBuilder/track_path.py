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

The racing line is found by optimisation over the whole loop at once, not
by a hand-picked rule per corner. The centerline is sampled all the way
around, each sample may slide sideways (up to `aggressiveness` x the usable
half-width), and the offsets that minimise

    (total squared curvature)  +  corner_cut * (path length)
      + soft outside/inside preferences per turn (see below)

are found with a coarse-to-fine solver (so it converges globally, not just
locally). On top of that, every "turn" (corners turning the same way with
at most one short straight between them) gets soft F1-style preferences:
back to the middle of the road after it, out to the outside shortly before
the next one, and a gentle pull to the inside halfway through. So:
  - a single corner or a U-turn gets outside -> inside apex -> outside,
  - a diagonal "staircase" of alternating corners is driven as a straight
    diagonal through the middle instead of chasing every apex,
  - an S-bend / chicane, or a short straight between a left and a right
    turn, gets the straightest middle-of-the-road compromise.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

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


def _centerline_arclength(segments: List[_Segment], sample_step: float) -> List[float]:
    """Centerline distance from the loop start for every point produced by
    _build_fine_centerline at this sample_step (same layout)."""
    s_vals = []
    s0 = 0.0
    for seg in segments:
        n_steps = max(1, int(math.ceil(seg.length / sample_step)))
        for k in range(n_steps):
            s_vals.append(s0 + seg.length * k / n_steps)
        s0 += seg.length
    return s_vals


def _turn_groups(segments: List[_Segment], merge_gap: float):
    """Groups corners into "turns": consecutive corners turning the same way,
    with at most `merge_gap` px of straight between them, form one turn (e.g.
    a U-turn made of corner + short straight + corner). Returns a list of
    (first_seg, last_seg) segment indices (last may wrap past the end)."""
    n_seg = len(segments)
    arcs = [i for i, s in enumerate(segments) if s.move.kind == "arc"]
    if not arcs:
        return []

    def joins(a, b):  # can arc a be merged with the next arc b?
        if segments[a].turn_sign != segments[b].turn_sign:
            return False
        gap, i = 0.0, (a + 1) % n_seg
        while i != b:
            gap += segments[i].length
            i = (i + 1) % n_seg
        return gap <= merge_gap

    # start the walk on an arc that does NOT join its predecessor, so no
    # group is split across the loop seam
    m = len(arcs)
    start = next((k for k in range(m) if not joins(arcs[k - 1], arcs[k])), 0)
    order = [arcs[(start + k) % m] for k in range(m)]
    groups = [[order[0], order[0]]]
    for a in order[1:]:
        if joins(groups[-1][1], a):
            groups[-1][1] = a
        else:
            groups.append([a, a])
    return [tuple(g) for g in groups]


def _turn_targets(segments: List[_Segment], points, step: float, max_offset: float,
                  distance: float, peak_before: float, anchor: int, merge_gap: float,
                  apex_strength: float = 0.1, settle_distance: float = 100.0):
    """Soft racing-line preferences per turn (see _turn_groups):
      - after a turn: the line is pulled back to the MIDDLE of the road
        (eases in over `settle_distance` px, so the exit itself stays smooth)
        and stays there along the straight,
      - entry: from `distance` px before the next turn the middle pull hands
        over to a pull toward the OUTSIDE of that turn (strongest
        `peak_before` px before it, letting go at the turn-in point),
      - apex: a gentle pull to the INSIDE halfway through the whole turn (so
        a U-turn gets one apex, not two).
    Turns that run straight into a corner of the opposite direction (S-bends,
    diagonal staircases) get no outside/apex pull, and a short straight
    between a left and a right turn only gets the middle pull, so those keep
    a straightened, middle-of-the-road line.
    Returns (weights, targets) arrays, one entry per point."""
    import numpy as np
    n = len(points)
    zeros = np.zeros(n)
    if distance <= 0 or max_offset <= 0:
        return zeros, zeros.copy()
    peak_before = max(0.0, min(peak_before, distance * 0.95))

    ranges = _segment_index_ranges(segments, step)
    seg_of = np.empty(n, dtype=int)
    for si, (start, count) in enumerate(ranges):
        seg_of[start:start + count] = si
    n_seg = len(segments)

    def ease(x):
        x = max(0.0, min(1.0, x))
        return (1 - math.cos(x * math.pi)) / 2.0

    anchor_fade = 1.5 * distance

    def anchor_factor(idx):
        # fade the outside pull near the start line (pinned to the centre), so
        # the line leaves/arrives there with a gentle drift instead of a kink
        k = (idx - anchor) % n
        return ease(min(k, n - k) * step / anchor_fade)

    def entry_weight(d):  # d = px before the turn
        if d >= peak_before:
            return ease((distance - d) / max(1e-9, distance - peak_before))
        return ease(d / max(1e-9, peak_before))

    def inside_vec(arc_pt):  # direction from an arc point toward its corner's pivot
        seg = segments[seg_of[arc_pt]]
        return (seg.pivot[0] - points[arc_pt][0], seg.pivot[1] - points[arc_pt][1])

    def side(idx, ref_arc_pt):
        """+1 if a positive offset at idx moves toward the inside of the corner
        that ref_arc_pt (an adjacent arc point) belongs to, else -1."""
        vx, vy = inside_vec(ref_arc_pt)
        return 1.0 if points[idx][2] * vx + points[idx][3] * vy > 0 else -1.0

    groups = _turn_groups(segments, merge_gap)
    n_g = len(groups)

    def straight_between(g_a, g_b):  # px of straight from the end of turn a to the start of turn b
        i, total = (groups[g_a][1] + 1) % n_seg, 0.0
        while i != groups[g_b][0]:
            total += segments[i].length
            i = (i + 1) % n_seg
        return total

    # A short straight between a left and a right turn: no outside pull for
    # the second turn there, the line just keeps to the middle.
    no_entry = set()
    for g in range(n_g):
        h = (g + 1) % n_g
        if n_g > 1 and segments[groups[g][1]].turn_sign != segments[groups[h][0]].turn_sign                 and straight_between(g, h) < 2.0 * distance:
            no_entry.add(h)

    entry_w = np.zeros(n)   # outside pull before a turn
    entry_t = np.zeros(n)
    center_w = np.zeros(n)  # middle-of-the-road pull after a turn
    apex_w = np.zeros(n)
    apex_t = np.zeros(n)

    for g, (first, last) in enumerate(groups):
        entry = ranges[first][0]
        exit_ = (ranges[last][0] + ranges[last][1]) % n
        before = segments[(first - 1) % n_seg].move.kind
        after = segments[(last + 1) % n_seg].move.kind

        if before == "line" and g not in no_entry:  # outside on the way in
            for k in range(1, int(distance / step) + 1):
                idx = (entry - k) % n
                if segments[seg_of[idx]].move.kind != "line":
                    break
                w = entry_weight(k * step) * anchor_factor(idx)
                if w > entry_w[idx]:
                    entry_w[idx] = w
                    entry_t[idx] = -side(idx, entry) * max_offset
        if after == "line":  # back to the middle after the turn, all along the straight
            k = 0
            while k < n:
                idx = (exit_ + k) % n
                if segments[seg_of[idx]].move.kind != "line":
                    break
                center_w[idx] = max(center_w[idx], ease((k + 1) * step / max(1e-9, settle_distance)))
                k += 1
        if before == "line" and after == "line":  # apex: gently inside, halfway through
            span = (exit_ - entry) % n
            for k in range(span):
                idx = (entry + k) % n
                u = (k + 0.5) / span
                if segments[seg_of[idx]].move.kind == "arc":
                    ref = idx
                else:  # short straight inside a merged turn: use the arc just before it
                    ref = (ranges[seg_of[idx]][0] - 1) % n
                apex_w[idx] = apex_strength * math.sin(math.pi * u)
                apex_t[idx] = side(idx, ref) * max_offset

    # the outside pull takes over from the middle pull as the turn approaches
    center_w *= (1.0 - entry_w)
    w_sum = entry_w + center_w + apex_w
    targets = np.where(w_sum > 0, (entry_w * entry_t + apex_w * apex_t) / np.maximum(w_sum, 1e-12), 0.0)
    return w_sum, targets


def _optimise_level(points, offsets, max_offset: float, step: float, length_weight: float,
                    iterations: int, fixed: Dict[int, float],
                    linear_ranges: List[Tuple[int, int]],
                    setup_strength: float = 0.0, setup_weights=None, setup_targets=None):
    """Projected (damped Jacobi) descent on one sampling level of

        E = sum |p[i-1] - 2 p[i] + p[i+1]|^2 / step^3        (curvature^2 * ds)
          + length_weight * sum |p[i+1] - p[i]|^2 / step     (~ path length)
          + setup_strength * sum w[i] (o[i] - target[i])^2 * step   (wide entry)

    where p[i] = centerline[i] + offset[i] * normal[i] and every offset is
    boxed to [-max_offset, max_offset]. The objective is convex, so this
    always settles on the single best line for the whole loop."""
    import numpy as np
    pts = np.asarray(points, dtype=float)
    c = pts[:, 0:2]
    nrm = pts[:, 2:4]
    o = np.asarray(offsets, dtype=float).copy()

    wc = 1.0 / step ** 3
    wl = length_weight / step
    if setup_weights is None or setup_strength <= 0:
        ws = np.zeros(len(o))
        tgt = np.zeros(len(o))
    else:
        ws = setup_strength * step * np.asarray(setup_weights, dtype=float)
        tgt = np.asarray(setup_targets, dtype=float)
    h = 2.0 * (6.0 * wc + 2.0 * wl) + 2.0 * ws  # exact curvature of E along each point's own normal
    omega = 0.6                       # < 2/2.67 keeps the Jacobi sweep stable for this stencil

    fixed_idx = np.array(list(fixed.keys()), dtype=int)
    fixed_val = np.array(list(fixed.values()), dtype=float)
    lin = [(s, s + cnt - 1) for s, cnt in linear_ranges if cnt >= 3]

    def apply_constraints(o):
        if len(fixed_idx):
            o[fixed_idx] = fixed_val
        for s_idx, e_idx in lin:
            k = np.arange(s_idx + 1, e_idx)
            t = (k - s_idx) / (e_idx - s_idx)
            o[k] = o[s_idx] + (o[e_idx] - o[s_idx]) * t
        return o

    o = apply_constraints(np.clip(o, -max_offset, max_offset))
    for _ in range(iterations):
        p = c + o[:, None] * nrm
        pm, pp = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
        d2 = pm - 2.0 * p + pp                       # D[i]
        g_curv = np.roll(d2, 1, axis=0) - 2.0 * d2 + np.roll(d2, -1, axis=0)
        g = 2.0 * wc * g_curv + 2.0 * wl * (2.0 * p - pm - pp)
        g_n = np.einsum("ij,ij->i", g, nrm) + 2.0 * ws * (o - tgt)
        o -= omega * g_n / h
        np.clip(o, -max_offset, max_offset, out=o)
        o = apply_constraints(o)
    return o


def _optimise_racing_line(segments: List[_Segment], result: GenResult, sample_step: float,
                          max_offset: float, length_weight: float, iterations: int,
                          setup_strength: float = 0.0, setup_distance: float = 0.0,
                          setup_peak_before: float = 0.0, merge_gap: float = 0.0,
                          apex_pull: float = 0.1, settle_distance: float = 100.0):
    """Coarse-to-fine solve: the line is first optimised on a very coarse
    sampling (where information travels around the whole loop in a few
    iterations), then each result is interpolated onto a finer sampling and
    refined, down to sample_step. Returns (points, offsets, anchor_index) at
    the finest level."""
    import numpy as np
    total_len = sum(seg.length for seg in segments)
    steps = [sample_step * f for f in (16, 8, 4, 2, 1)]

    prev_s = prev_o = None
    points = offsets = None
    anchor = 0
    for level, step in enumerate(steps):
        points = _build_fine_centerline(segments, step)
        s_vals = np.array(_centerline_arclength(segments, step))
        ranges = _segment_index_ranges(segments, step)
        seg0_start, seg0_count = ranges[0]
        anchor = seg0_start + seg0_count // 2  # start line: pinned to the centerline
        linear = _crossing_linear_ranges(segments, result, step)

        if prev_o is None:
            init = np.zeros(len(points))
        else:
            init = np.interp(s_vals, prev_s, prev_o, period=total_len)

        # coarse levels are cheap and carry the global shape -> most iterations there
        n_iter = iterations if level == 0 else max(50, iterations // (2 ** level))
        sw, st = _turn_targets(segments, points, step, max_offset,
                               setup_distance, setup_peak_before, anchor, merge_gap, apex_pull,
                               settle_distance)
        offsets = _optimise_level(points, init, max_offset, step, length_weight,
                                  n_iter, {anchor: 0.0}, linear,
                                  setup_strength, sw, st)
        prev_s, prev_o = s_vals, offsets
    return points, list(offsets), anchor


# Converts the user-facing entry_swing (0..~1) into the optimiser's penalty weight.
_ENTRY_SWING_SCALE = 8e-7


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
    smoothing_iterations: int = 3000,
    corner_cut: float = 0.0,
    entry_swing: float = 1.0,
    entry_swing_distance: float = 140.0,
    entry_swing_peak: float = 70.0,
    exit_settle_distance: float = 100.0,
    turn_merge_gap=None,
    apex_pull: float = 0.1,
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
    smoothing_iterations:          optimiser iterations on the coarsest level
                                   (finer levels use fewer). A few thousand
                                   is plenty to converge.
    corner_cut:                    extra "shortest path" preference on top of
                                   smoothness. 0 (default) = pure smooth,
                                   F1-style line; larger (e.g. 1e-4) hugs the
                                   inside of corners more and uses less width.
    entry_swing:                   0 = off. How strongly the line is guided on
                                   straights: back to the MIDDLE of the road
                                   after a turn, then out to the OUTSIDE of the
                                   next turn shortly before it (outside -
                                   inside - outside through the turn). A soft
                                   preference weighed against smoothness;
                                   higher = follows it more strictly, but
                                   swings get more abrupt. Corners turning the same way
                                   with at most one short straight between
                                   them count as one turn (e.g. a U-turn). Not
                                   used between a turn and an opposite turn
                                   that follows quickly (S-bends, staircases),
                                   where the line takes the middle instead.
    entry_swing_distance:          how many px before a turn the line starts
                                   moving from the middle to the outside.
    entry_swing_peak:              how many px before the turn the outside
                                   pull is strongest (it lets go after that
                                   so the line can turn in smoothly).
    exit_settle_distance:          over how many px after a turn the line is
                                   eased back toward the middle of the road.
    turn_merge_gap:                max px of straight between two same-direction
                                   corners for them to count as one turn
                                   (default: about one straight piece).
    apex_pull:                     how strongly the line is pulled to the
                                   INSIDE in the middle of a turn (relative to
                                   entry_swing). Small values are enough - the
                                   smoothness term already finds the apex;
                                   large values make corners look angular.
    start_offset:                 checkpoint 0 sits at the middle of the start
                                   straight, shifted backward (against the
                                   direction of travel) by this many pixels.
    """
    segments = _build_segments(result, cfg)
    max_offset = max(0.0, aggressiveness) * max(0.0, cfg.track_width / 2.0 - safety_margin)

    # The start line is a fixed, physical mark drawn at the exact centre of
    # the start straight - the racing line is anchored to sit exactly on the
    # centerline right there, regardless of aggressiveness.
    points, offsets, mid_idx = _optimise_racing_line(
        segments, result, sample_step, max_offset,
        length_weight=corner_cut, iterations=smoothing_iterations,
        setup_strength=_ENTRY_SWING_SCALE * max(0.0, entry_swing),
        setup_distance=entry_swing_distance, setup_peak_before=entry_swing_peak,
        merge_gap=1.1 * cfg.cell_size if turn_merge_gap is None else turn_merge_gap,
        apex_pull=apex_pull, settle_distance=exit_settle_distance)

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