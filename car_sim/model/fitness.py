"""
Fitness = distance travelled along the racing line, credited only in
order: a car must reach its own next expected checkpoint (not just
whichever one is spatially nearest) before any later one counts. That's
what stops a car earning progress by cutting across the track, driving
backwards, or exploiting a spot where the track loops close to itself.

Checkpoints wrap: after the last one, the next expected checkpoint goes
back to the first (index 0) - reaching it again counts as finishing a
lap, and progress keeps counting up from there for however many laps a
car manages within one generation.

There's also a small per-second penalty for simply being alive, applied
once at the end using each car's own total alive time - so two cars that
reach the same checkpoint don't score the same if one dawdled to get
there.
"""
import numpy as np

from settings import CHECKPOINT_RADIUS


def advance_checkpoints(checkpoint_xy, car_x, car_y, next_idx, laps_completed):
    """
    checkpoint_xy: (K, 2) checkpoint positions, in order
    car_x, car_y: (N,) current car positions
    next_idx: (N,) int - each car's next expected checkpoint index (0..K-1)
    laps_completed: (N,) int - full laps each car has finished so far

    Advances a car's next_idx (and laps_completed, on wrap) once it gets
    within CHECKPOINT_RADIUS of its own next expected checkpoint - never
    any other checkpoint, in or out of order.

    Returns the updated (next_idx, laps_completed).
    """
    k = checkpoint_xy.shape[0]
    car_xy = np.stack([car_x, car_y], axis=-1)        # (N, 2)
    targets = checkpoint_xy[next_idx]                   # (N, 2) - each car's own next checkpoint
    dist = np.linalg.norm(car_xy - targets, axis=-1)    # (N,)
    reached = dist < CHECKPOINT_RADIUS

    # Reaching checkpoint 0 as a target means a car just closed the loop -
    # that's a completed lap, counted before next_idx advances past it.
    laps_completed = laps_completed + (reached & (next_idx == 0)).astype(np.int64)
    next_idx = np.where(reached, (next_idx + 1) % k, next_idx)
    return next_idx, laps_completed


def progress_along_track(checkpoint_distance, next_idx, laps_completed):
    """
    checkpoint_distance: (K,) cumulative distance at each checkpoint
    next_idx, laps_completed: current per-car state (see advance_checkpoints)

    Total distance travelled so far: completed laps' worth, plus whatever
    of the current lap has been covered. checkpoint_distance[next_idx - 1]
    is the last checkpoint actually reached this lap - and reads correctly
    right after a wrap too, since next_idx - 1 is -1 there, which numpy
    indexes as the last (highest-distance) checkpoint.
    """
    lap_distance = checkpoint_distance[-1]
    return laps_completed * lap_distance + checkpoint_distance[next_idx - 1]


def apply_alive_penalty(progress, time_alive, penalty_per_second):
    """Small per-second cost for simply being alive, so idling or a slower
    line scores worse than reaching the same progress faster."""
    return progress - penalty_per_second * time_alive