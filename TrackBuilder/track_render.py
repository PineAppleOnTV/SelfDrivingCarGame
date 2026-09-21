"""
track_render.py
================
Turns a GenResult (from track_generator.py) into a black/white(-ish) PNG
image. The drivable lane surface is pure white (255,255,255); everywhere
else (background, and the non-lane parts of each tile) is (254,254,254) -
visually identical, but a plain pixel-colour check tells you whether a
point is on the track or not.
"""

from typing import Dict, Tuple, Optional
from PIL import Image, ImageDraw

from track_generator import TrackConfig, GenResult, DELTA

WHITE = (255, 255, 255)
# Everything that ISN'T the drivable lane surface (background, and the
# non-lane parts of every tile) is this near-white instead of pure white,
# so "is this pixel on track?" is a plain, exact colour check: (255,255,255)
# = on the lane, (254,254,254) = off it. The two are visually indistinguishable.
BACKGROUND = (254, 254, 254)
BLACK = (0, 0, 0)

# Which tile-corner a corner piece pivots around, and the angle range of its
# arcs, for each pair of connected sides. Angles are PIL's convention: 0 deg
# points along +x (east/right), and angles increase clockwise on screen
# (because the image y-axis points down).
_CORNER_GEOMETRY = {
    frozenset({"N", "W"}): ((0, 0), 0, 90),      # pivot top-left
    frozenset({"N", "E"}): ((1, 0), 90, 180),    # pivot top-right
    frozenset({"S", "E"}): ((1, 1), 180, 270),   # pivot bottom-right
    frozenset({"S", "W"}): ((0, 1), 270, 360),   # pivot bottom-left
}


def _blank_tile(cfg: TrackConfig) -> Image.Image:
    s = cfg.cell_size
    return Image.new("RGB", (s, s), BACKGROUND)


def _band(center: float, bw: int) -> Tuple[int, int]:
    """Pixel range [lo, hi] (both inclusive) that is exactly bw pixels thick,
    biased toward the inside (the extra pixel for an even bw goes on the
    lower/inward side) - this is the convention draw_corner's ring() also
    uses, so a straight/crossing border lines up exactly with a corner
    border of the same width. PIL's rectangle() is inclusive on both ends,
    so a naive [center-bw/2, center+bw/2] box would be 1px too thick.
    """
    hi = int(center) + (bw - 1) // 2
    return hi - bw + 1, hi


def draw_straight_h(cfg: TrackConfig) -> Image.Image:
    """A straight piece, lane running left-to-right (E-W)."""
    img = _blank_tile(cfg)
    d = ImageDraw.Draw(img)
    s = cfg.cell_size
    m = cfg.margin
    bw = cfg.border_width
    gap = (s - cfg.straight_length) / 2.0  # centers the piece in its cell
    x0 = gap
    x1 = s - gap
    top = m                    # inner edge of the lane band (== corner inner radius)
    bot = m + cfg.track_width  # outer edge of the lane band
    d.rectangle([x0, top, x1, bot], fill=WHITE)  # the drivable lane surface
    top_lo, top_hi = _band(top, bw)
    bot_lo, bot_hi = _band(bot, bw)
    d.rectangle([x0, top_lo, x1, top_hi], fill=BLACK)
    d.rectangle([x0, bot_lo, x1, bot_hi], fill=BLACK)
    return img


def draw_corner(cfg: TrackConfig, pair: frozenset) -> Image.Image:
    """A 90 degree corner connecting the two given sides ('N','E','S','W')."""
    img = _blank_tile(cfg)
    d = ImageDraw.Draw(img)
    s = cfg.cell_size
    m = cfg.margin
    tw = cfg.track_width
    bw = cfg.border_width
    r_in = m
    r_out = m + tw

    corner_frac, start_angle, end_angle = _CORNER_GEOMETRY[pair]
    px = corner_frac[0] * s
    py = corner_frac[1] * s

    def bbox(r):
        return [px - r, py - r, px + r, py + r]

    # the drivable lane surface: the annular band between r_in and r_out,
    # restricted to this corner's 90-degree sector
    d.pieslice(bbox(r_out), start_angle, end_angle, fill=WHITE)
    d.pieslice(bbox(r_in), start_angle, end_angle, fill=BACKGROUND)

    # A clean, uniformly-thick arc band is drawn as the difference of two
    # filled pie slices (same pivot & angles) rather than ImageDraw.arc's
    # `width=`, which renders badly at small radii (it turns the tight inner
    # radius into a solid blob instead of a thin ring). The bbox radii are
    # picked (via _band) to land on exactly the same integer pixels a
    # straight/crossing border of the same width would use.
    def ring(r):
        lo, hi = _band(r, bw)
        d.pieslice(bbox(hi), start_angle, end_angle, fill=BLACK)
        d.pieslice(bbox(lo - 1), start_angle, end_angle, fill=WHITE)

    ring(r_out)
    ring(r_in)
    return img


def draw_crossing(cfg: TrackConfig) -> Image.Image:
    """A '+' crossing: lane open in both directions, border only drawn as the
    4 short bracket segments outside the central (shared) square."""
    img = _blank_tile(cfg)
    d = ImageDraw.Draw(img)
    s = cfg.cell_size
    m = cfg.margin
    tw = cfg.track_width
    bw = cfg.border_width
    lo = m               # inner edge of the lane band (same as a straight/corner)
    hi = m + tw          # outer edge of the lane band
    # the drivable lane surface: the '+' shape (both bands, full width/height)
    d.rectangle([0, lo, s, hi], fill=WHITE)
    d.rectangle([lo, 0, hi, s], fill=WHITE)
    lo_lo, lo_hi = _band(lo, bw)
    hi_lo, hi_hi = _band(hi, bw)

    # Same lane-boundary convention as straight/corner. The gap (over
    # [lo, hi]) is exactly the width of the perpendicular lane, so the open
    # driving surface is never crossed by a border stroke.
    d.rectangle([0, lo_lo, lo, lo_hi], fill=BLACK)
    d.rectangle([hi, lo_lo, s, lo_hi], fill=BLACK)
    d.rectangle([0, hi_lo, lo, hi_hi], fill=BLACK)
    d.rectangle([hi, hi_lo, s, hi_hi], fill=BLACK)
    d.rectangle([lo_lo, 0, lo_hi, lo], fill=BLACK)
    d.rectangle([lo_lo, hi, lo_hi, s], fill=BLACK)
    d.rectangle([hi_lo, 0, hi_hi, lo], fill=BLACK)
    d.rectangle([hi_lo, hi, hi_hi, s], fill=BLACK)
    return img


def _rotated(img: Image.Image, degrees: int) -> Image.Image:
    # expand=False keeps the tile square/aligned; our tiles are square so this is exact.
    return img.rotate(-degrees, resample=Image.NEAREST, expand=False)


def draw_start_marker(img: Image.Image, cfg: TrackConfig, axis: str, center_xy: Tuple[float, float]):
    """Paints the two start/finish bars onto `img`, centered at center_xy.
    Each bar is always l (along travel) by w (across the lane) - never
    rotated. Only the gap between the two bars changes orientation:
    axis='H' -> travel is left/right, so the pair is stacked along y
                (perpendicular to travel) - their short (l-length) edges
                face each other, forming one crossable line.
    axis='V' -> travel is up/down, so the pair is stacked along x instead.
    """
    d = ImageDraw.Draw(img)
    cx, cy = center_xy
    w = cfg.start_mark_width      # across the lane
    l = cfg.start_mark_length     # along the direction of travel
    gap = cfg.start_mark_gap
    half_gap = gap / 2.0
    if axis == "H":
        # each bar is l wide (x) by w tall (y) - unrotated - stacked along y
        bar1 = [cx - l / 2.0, cy - half_gap - w, cx + l / 2.0, cy - half_gap]
        bar2 = [cx - l / 2.0, cy + half_gap, cx + l / 2.0, cy + half_gap + w]
    else:
        # each bar is w wide (x) by l tall (y) - unrotated - stacked along x
        bar1 = [cx - half_gap - w, cy - l / 2.0, cx - half_gap, cy + l / 2.0]
        bar2 = [cx + half_gap, cy - l / 2.0, cx + half_gap + w, cy + l / 2.0]
    d.rectangle(bar1, fill=BLACK)
    d.rectangle(bar2, fill=BLACK)


def assemble_track(
    result: GenResult,
    cfg: TrackConfig,
    start_marker: bool = True,
    trim: bool = True,
    padding: int = 8,
    return_offset: bool = False,
):
    """Returns the assembled image, or (image, (dx, dy)) if return_offset is
    True - (dx, dy) is how far the trim shifted the origin, i.e. subtract it
    from any absolute canvas coordinates (e.g. from track_path checkpoints,
    via track_path.offset_checkpoints) to match this image."""
    s = cfg.cell_size
    canvas = Image.new("RGB", (result.cols * s, result.rows * s), BACKGROUND)

    straight_h = draw_straight_h(cfg)
    straight_v = _rotated(straight_h, 90)
    crossing = draw_crossing(cfg)
    corner_cache: Dict[frozenset, Image.Image] = {}

    for (r, c), piece in result.cells.items():
        kind = piece[0]
        if kind == "straight":
            tile = straight_h if piece[1] == "H" else straight_v
        elif kind == "crossing":
            tile = crossing
        else:  # corner
            pair = piece[1]
            if pair not in corner_cache:
                corner_cache[pair] = draw_corner(cfg, pair)
            tile = corner_cache[pair]
        canvas.paste(tile, (c * s, r * s))

    if start_marker:
        r, c = result.start_cell
        cx = c * s + s / 2.0
        cy = r * s + s / 2.0
        # start cell is always a fixed H straight (see track_generator.py)
        draw_start_marker(canvas, cfg, "H", (cx, cy))

    offset = (0, 0)
    if trim:
        rows_used = [r for r, c in result.cells]
        cols_used = [c for r, c in result.cells]
        left = max(0, min(cols_used) * s - padding)
        top = max(0, min(rows_used) * s - padding)
        right = min(canvas.width, (max(cols_used) + 1) * s + padding)
        bottom = min(canvas.height, (max(rows_used) + 1) * s + padding)
        canvas = canvas.crop((left, top, right, bottom))
        offset = (left, top)

    return (canvas, offset) if return_offset else canvas