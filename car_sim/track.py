import pygame
import pygame.surfarray as surfarray
import numpy as np
from settings import TRACK_COLOR, BORDER_COLOR, OFF_TRACK_COLOR


class Track:
    """
    Loads a track image once and answers batched queries about many
    (x, y) points at once, via precomputed boolean masks.

    Precomputing the masks means every query is a plain boolean array
    lookup instead of comparing a 3-tuple colour against a pixel on
    every call (the original per-pixel `==` comparison against a tuple
    doesn't reduce to a single bool anyway - `.all()` is needed there).
    """

    def __init__(self, image_path, track_color=TRACK_COLOR, border_color=BORDER_COLOR,
                 off_track_color=OFF_TRACK_COLOR):
        self.image = pygame.image.load(image_path).convert()
        pixels = surfarray.array3d(self.image)  # (W, H, 3)
        self.image_size = self.image.get_size()

        self.track_color = track_color
        self.border_color = border_color
        self.off_track_color = off_track_color

        self.track_mask = np.all(pixels == np.array(track_color), axis=-1)          # (W, H) bool
        self.border_mask = np.all(pixels == np.array(border_color), axis=-1)        # (W, H) bool
        self.off_track_mask = np.all(pixels == np.array(off_track_color), axis=-1)  # (W, H) bool

    def get_size(self):
        """Gets the size of the track image"""
        return self.image_size

    def _sample(self, mask, x, y, out_of_bounds_value=False):
        """Shared lookup for x, y arrays of any matching shape. A point
        outside the image returns out_of_bounds_value (False for "is this
        the track/border colour", True for "is this off-track" - going off
        the edge of the image entirely is off-track too)."""
        width, height = self.image_size
        ix = x.astype(np.int64)
        iy = y.astype(np.int64)

        in_bounds = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
        # Clip so the fancy-index below is always valid; in_bounds picks
        # between that lookup and out_of_bounds_value for points that were
        # actually outside.
        ix_clipped = np.clip(ix, 0, width - 1)
        iy_clipped = np.clip(iy, 0, height - 1)

        sampled = mask[ix_clipped, iy_clipped]
        return np.where(in_bounds, sampled, out_of_bounds_value)

    def is_on_track_batch(self, x, y):
        """x, y: numpy arrays of any matching shape. Returns bool array, same shape."""
        return self._sample(self.track_mask, x, y)

    def is_border_batch(self, x, y):
        """x, y: numpy arrays of any matching shape. Returns bool array, same shape."""
        return self._sample(self.border_mask, x, y)

    def is_off_track_batch(self, x, y):
        """x, y: numpy arrays of any matching shape. True where the pixel is
        the off-track background colour, or the point is outside the image
        entirely. This is the fail condition, deliberately separate from
        "not track colour" so anti-aliased edge pixels between the white
        track and black border don't themselves count as off-track."""
        return self._sample(self.off_track_mask, x, y, out_of_bounds_value=True)

    # Single-point convenience wrappers, kept so any one-off/debug call site
    # doesn't need a separate API from the batched one.
    def is_on_track(self, x, y):
        """Returns True or False depending on where the center of the car is"""
        return bool(self.is_on_track_batch(np.array([x]), np.array([y]))[0])

    def is_border(self, x, y):
        """Returns True or False depending on if the pixel is a border pixel"""
        return bool(self.is_border_batch(np.array([x]), np.array([y]))[0])

    def is_off_track(self, x, y):
        """Returns True or False depending on if the pixel is off-track"""
        return bool(self.is_off_track_batch(np.array([x]), np.array([y]))[0])

    def screen_offset(self, c_x, c_y):
        """
        Top-left screen position this track is blitted at when centered on
        (c_x, c_y). World/track-pixel coordinates (car position, checkpoints,
        sensor rays) are always in the track image's own coordinate space;
        this is the one place that converts to screen space for drawing -
        needed because the 4 tracks are different sizes but share one window.
        """
        width, height = self.image_size
        return int(c_x - width / 2), int(c_y - height / 2)

    def draw(self, screen, c_x, c_y):
        """Draw the image on the screen"""
        top_left_x, top_left_y = self.screen_offset(c_x, c_y)
        screen.blit(self.image, (top_left_x, top_left_y))