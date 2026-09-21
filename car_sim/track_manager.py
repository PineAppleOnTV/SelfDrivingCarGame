import os
import csv

import numpy as np

from settings import TRACKS, TRACKS_FOLDER, CHECKPOINTS_FOLDER
from track import Track


def _round_to_90(heading_deg):
    """Rounds a 0-360 heading to the nearest of 0/90/180/270 (wraps 360 -> 0)."""
    return round(heading_deg / 90.0) * 90.0 % 360.0


def _load_checkpoints(path):
    """Loads a track_X_checkpoints.csv (index,x,y,heading_deg,distance) into
    a list of dicts, in file order."""
    checkpoints = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            checkpoints.append({
                "index": int(row["index"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "heading_deg": float(row["heading_deg"]),
                "distance": float(row["distance"]),
            })
    return checkpoints


class TrackManager:
    """
    Loads every track in settings.TRACKS (image + checkpoint file) once up
    front, then lets the rest of the game switch between them by index or
    name without touching disk again. Keeping Track itself single-track
    means it doesn't need to know anything about the other 3.
    """

    def __init__(self):
        self.names = list(TRACKS)
        self._tracks = {}
        self._checkpoints = {}

        for name in self.names:
            image_path = os.path.join(TRACKS_FOLDER, f"{name}.png")
            checkpoints_path = os.path.join(CHECKPOINTS_FOLDER, f"{name}_checkpoints.csv")
            self._tracks[name] = Track(image_path)
            self._checkpoints[name] = _load_checkpoints(checkpoints_path)

        self.current_index = 0

    @property
    def current_name(self):
        return self.names[self.current_index]

    @property
    def current_track(self):
        return self._tracks[self.current_name]

    def select(self, index):
        """Switch to a track by index (wraps around, so -1 / len both work)."""
        self.current_index = index % len(self.names)

    def next(self):
        self.select(self.current_index + 1)

    def previous(self):
        self.select(self.current_index - 1)

    def get_start_pose(self):
        """(x, y, heading_deg) for the current track's first checkpoint -
        that's the starting location - with heading rounded to the nearest
        90 degrees."""
        first = self._checkpoints[self.current_name][0]
        return first["x"], first["y"], _round_to_90(first["heading_deg"])

    def get_checkpoints(self):
        """The current track's checkpoints, as the raw list of dicts loaded
        from its CSV - used for drawing the 'view checkpoints' overlay."""
        return self._checkpoints[self.current_name]

    def get_checkpoint_arrays(self):
        """The current track's checkpoints as (xs, ys, distances) numpy
        arrays - used for the training fitness calculation."""
        checkpoints = self.get_checkpoints()
        xs = np.array([c["x"] for c in checkpoints], dtype=np.float64)
        ys = np.array([c["y"] for c in checkpoints], dtype=np.float64)
        distances = np.array([c["distance"] for c in checkpoints], dtype=np.float64)
        return xs, ys, distances

    def get_screen_offset(self, c_x, c_y):
        """Where the current track is blitted for a given screen-center
        point - see Track.screen_offset."""
        return self.current_track.screen_offset(c_x, c_y)

    def draw(self, screen, c_x, c_y):
        self.current_track.draw(screen, c_x, c_y)