import numpy as np
from settings import H_AREAS, V_AREAS, WIDTH_OF_AREA, LENGTH_OF_AREA, STEP_SIZE


class Sensor:
    """
    Batched distance sensors for all N cars at once.

    For every car:
      - one (left, right) pair per entry in H_AREAS -> shape (N, len(H_AREAS), 2)
      - one forward reading per entry in V_AREAS     -> shape (N, len(V_AREAS))

    Values are normalised to [0.0, 1.0] as a fraction of that sensor's max
    range, or -1.0 if no border pixel was found within range.

    The only Python-level loop here is over the *fixed* sensor layout
    (H_AREAS / V_AREAS, a handful of entries known at settings-time) - the
    N cars and the S samples along each ray are numpy array dimensions,
    so this doesn't grow slower as N grows.

    ASSUMPTIONS (flip these constants if readings look off once visualised):
      - H_AREAS values are forward offsets (distance ahead of the car
        centre) at which each left/right pair is anchored; each side scans
        up to the full WIDTH_OF_AREA (not WIDTH_OF_AREA / 2).
      - V_AREAS values are lateral offsets (e.g. left/centre/right) for
        rays that each scan straight ahead up to the full LENGTH_OF_AREA.
      - "right" of the car is +90deg from heading: (-sin(angle), cos(angle)),
        matching the angle convention already used in Player.update().
    """

    def __init__(self, track):
        self.track = track
        self.h_offsets = np.asarray(H_AREAS, dtype=np.float64)  # (H,)
        self.v_offsets = np.asarray(V_AREAS, dtype=np.float64)  # (V,)
        self.h_max_range = float(WIDTH_OF_AREA)
        self.v_max_range = float(LENGTH_OF_AREA)

        max_range = max(self.h_max_range, self.v_max_range)
        n_steps = max(1, int(round(max_range / STEP_SIZE)))
        all_steps = np.arange(1, n_steps + 1, dtype=np.float64) * STEP_SIZE  # (S,)
        self._h_steps = all_steps[all_steps <= self.h_max_range]
        self._v_steps = all_steps[all_steps <= self.v_max_range]

    def read(self, players):
        """players: batched Player (arrays x, y, angle of shape (N,))."""
        centre = np.stack([players.x, players.y], axis=-1)                    # (N, 2)
        angle_rad = np.radians(players.angle)
        forward = np.stack([np.cos(angle_rad), np.sin(angle_rad)], axis=-1)   # (N, 2)
        right = np.stack([-np.sin(angle_rad), np.cos(angle_rad)], axis=-1)    # (N, 2)

        return {
            "horizontal": self._read_horizontal(centre, forward, right),  # (N, H, 2)
            "vertical": self._read_vertical(centre, forward, right),      # (N, V)
        }

    def _read_horizontal(self, centre, forward, right):
        # anchor per (car, H-area): centre + forward_offset along heading -> (N, H, 2)
        anchors = centre[:, None, :] + forward[:, None, :] * self.h_offsets[None, :, None]

        # sample points out to the right / left of each anchor -> (N, H, S, 2)
        offsets = self._h_steps[None, None, :, None] * right[:, None, None, :]
        right_pts = anchors[:, :, None, :] + offsets
        left_pts = anchors[:, :, None, :] - offsets

        right_dist = self._first_hit(right_pts, self._h_steps, self.h_max_range)  # (N, H)
        left_dist = self._first_hit(left_pts, self._h_steps, self.h_max_range)    # (N, H)
        return np.stack([left_dist, right_dist], axis=-1)  # (N, H, 2)

    def _read_vertical(self, centre, forward, right):
        # start per (car, V-area): centre offset sideways -> (N, V, 2)
        starts = centre[:, None, :] + right[:, None, :] * self.v_offsets[None, :, None]

        # sample points straight ahead of each start -> (N, V, S, 2)
        pts = starts[:, :, None, :] + self._v_steps[None, None, :, None] * forward[:, None, None, :]
        return self._first_hit(pts, self._v_steps, self.v_max_range)  # (N, V)

    def _first_hit(self, pts, steps, max_range):
        """pts: (..., S, 2) world coords ordered near->far along the ray.
        Returns (...) normalised first-hit distance in [0, 1], or -1.0 if
        no border was found anywhere along the ray."""
        hits = self.track.is_border_batch(pts[..., 0], pts[..., 1])  # (..., S) bool

        any_hit = hits.any(axis=-1)
        first_idx = hits.argmax(axis=-1)   # first True index; meaningless where any_hit is False
        first_dist = steps[first_idx]

        return np.where(any_hit, first_dist / max_range, -1.0)
