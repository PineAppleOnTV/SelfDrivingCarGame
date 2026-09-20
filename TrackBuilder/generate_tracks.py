"""
generate_tracks.py
===================
Run this to generate several different random closed-loop tracks as PNGs,
built out of straight / corner / crossing pieces.

Edit the settings below, then just run:  python3 generate_tracks.py
"""

import os
from track_generator import TrackConfig, generate_track_variants
from track_render import assemble_track
from track_path import generate_checkpoints, offset_checkpoints, draw_checkpoints, export_checkpoints_csv

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------

# How many of each piece you actually have. Use None for "as many as needed".
PIECE_COUNTS = {
    "straight": 18,
    "corner": 14,
    "crossing": 2,
}

# How many different track layouts to generate.
NUM_VARIANTS = 4

# Size of the grid tracks are laid out on (rows, cols). Leave as None to let
# it size itself based on PIECE_COUNTS.
GRID_SIZE = None

# Smallest allowed loop length (in pieces). Raise this if you want bigger loops.
MIN_LENGTH = 10

# Whether the track is allowed to cross itself once through a crossing piece
# (figure-8 style). Set to False to forbid crossings entirely, even if you
# have some available.
ALLOW_CROSSINGS = True

# Fix this to an integer to get the same set of tracks again later; leave as
# None for a different random batch every run.
SEED = None

OUTPUT_DIR = "."
TRACKS_DIR = os.path.join(OUTPUT_DIR, "tracks")          # all PNGs go here
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")  # all checkpoint CSVs go here

# --- Checkpoints, for ML training ---------------------------------------
# Set to False to skip checkpoint generation entirely.
GENERATE_CHECKPOINTS = True

# Distance in pixels between consecutive checkpoints along the path.
CHECKPOINT_INTERVAL = 20.0

# 0 = plain centerline down the middle of the lane. 1 = allowed to roam as
# close to the lane edges as CHECKPOINT_SAFETY_MARGIN permits. Crossings
# always keep a fixed (straight) direction of travel regardless of this
# value - a crossing is geometrically a straight line, never an arc, so it
# is physically impossible for the path to turn there - but its lateral
# position is still free to drift smoothly through it.
RACING_LINE_AGGRESSIVENESS = 0.4

# Pixels of clearance kept between the racing line and the lane border at
# full (1.0) aggressiveness.
CHECKPOINT_SAFETY_MARGIN = 2.0

# How hard (0 = off, fraction of the max lateral offset) the line swings
# toward the OUTSIDE of a corner in the straight immediately before it -
# the classic "out" before the "in". Only ever kicks in on a plain,
# sufficiently long straight/crossing approach - never between corners in a
# chicane, so it can't reintroduce apex-chasing zig-zag there.
PRE_CORNER_SWING = 0.3
PRE_CORNER_SWING_DISTANCE = 80.0  # pixels before the corner this ramps up over

# How many pixels before the corner the outward push has fully eased off
# again, so the line has already started turning back in before it reaches
# the corner instead of holding the outside line right up to the apex and
# snapping in all at once. Always kept smaller than PRE_CORNER_SWING_DISTANCE
# (clamped automatically if set too close to it).
PRE_CORNER_TURN_IN_DISTANCE = 25.0

# Checkpoint 0 sits at the middle of the start straight, shifted backward
# (against the direction of travel) by this many pixels.
CHECKPOINT_START_OFFSET = 20.0

# Also save a PNG with the racing line + checkpoints drawn on top, so you can
# eyeball them.
SAVE_CHECKPOINT_PREVIEW = True

# Piece geometry - all pixel values, all independently adjustable.
CONFIG = TrackConfig(
    track_width=53,
    border_width=2,
    straight_length=67,  # == 2*corner_inner_radius + track_width -> seamless joints
    corner_inner_radius=7,
    start_mark_width=7,
    start_mark_length=2,
    start_mark_gap=4,
)

# ---------------------------------------------------------------------------

def main():
    os.makedirs(TRACKS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    variants = generate_track_variants(
        piece_counts=PIECE_COUNTS,
        n_variants=NUM_VARIANTS,
        grid_size=GRID_SIZE,
        min_length=MIN_LENGTH,
        allow_crossings=ALLOW_CROSSINGS,
        seed=SEED,
    )

    if not variants:
        print("Couldn't build any valid track with these settings - try a "
              "larger piece budget, a bigger grid, or a smaller min_length.")
        return

    if len(variants) < NUM_VARIANTS:
        print(f"Only found {len(variants)} distinct layouts (asked for "
              f"{NUM_VARIANTS}) - try loosening the constraints for more.")

    for i, result in enumerate(variants, start=1):
        img, offset = assemble_track(result, CONFIG, return_offset=True)
        path = os.path.join(TRACKS_DIR, f"track_{i}.png")
        img.save(path)
        from collections import Counter
        counts = Counter(p[0] for p in result.cells.values())
        print(f"tracks/track_{i}.png  -  {dict(counts)}  ({img.size[0]}x{img.size[1]}px)")

        if GENERATE_CHECKPOINTS:
            checkpoints = generate_checkpoints(
                result, CONFIG,
                interval=CHECKPOINT_INTERVAL,
                aggressiveness=RACING_LINE_AGGRESSIVENESS,
                safety_margin=CHECKPOINT_SAFETY_MARGIN,
                pre_corner_swing=PRE_CORNER_SWING,
                pre_corner_swing_distance=PRE_CORNER_SWING_DISTANCE,
                pre_corner_turn_in_distance=PRE_CORNER_TURN_IN_DISTANCE,
                start_offset=CHECKPOINT_START_OFFSET,
            )
            checkpoints = offset_checkpoints(checkpoints, offset[0], offset[1])
            csv_path = os.path.join(CHECKPOINTS_DIR, f"track_{i}_checkpoints.csv")
            export_checkpoints_csv(checkpoints, csv_path)
            print(f"  -> {len(checkpoints)} checkpoints saved to checkpoints/track_{i}_checkpoints.csv")

            if SAVE_CHECKPOINT_PREVIEW:
                preview = img.copy()
                draw_checkpoints(preview, checkpoints)
                preview.save(os.path.join(TRACKS_DIR, f"track_{i}_checkpoints.png"))


if __name__ == "__main__":
    main()