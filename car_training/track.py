import pygame
import pygame.surfarray as surfarray
import numpy as np
from settings import *

class Track:
    """
    Loads a track image and answers if the car is on the track
    """
    def __init__(self, image_path, track_color=TRACK_COLOR, tolerance=TRACK_COLOR_TOLERANCE):
        """Give the image path for it to save as pygame's display, and store track color and tolerance"""
        self.image = pygame.image.load(image_path).convert()
        self.pixels = surfarray.array3d(self.image)
        self.image_size = self.image.get_size()
        self.track_color = track_color
        self.tolerance = tolerance

    def get_size(self):
        """Gets the size of the track image"""
        return self.image_size

    def is_on_track(self, x, y):
        """Returns True or False depending on where the center of the car is"""
        width, heigth = self.image_size
        ix, iy = int(x), int(y)

        if ix < 0 or iy < 0 or ix > width or iy > heigth:
            return False

        pixel_color = self.pixels[ix, iy]
        return self._colors_close(pixel_color)

    def is_on_track_batch(self, xs, ys):
        """
        Vectorized version of is_on_track - takes arrays of x, y coordinates
        and returns a boolean array, one result per point, computed in one
        numpy operation instead of one Python call per point.
        """
        width, height = self.image.get_size()
        ix = xs.astype(int)
        iy = ys.astype(int)

        in_bounds = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)

        # Clip so out-of-bounds indices don't crash the lookup - their
        # result gets overwritten to False by in_bounds afterward anyway.
        ix_safe = np.clip(ix, 0, width - 1)
        iy_safe = np.clip(iy, 0, height - 1)

        pixel_colors = self.pixels[ix_safe, iy_safe]  # shape (N, 3)
        target = np.array(self.track_color)
        close = np.all(np.abs(pixel_colors.astype(int) - target) <= self.tolerance, axis=1)

        return close & in_bounds

    def _colors_close(self, pixel_color):
        """Compares if the color is close to the track color"""
        return all(
            abs(pixel_color[i] - self.track_color[i]) <= self.tolerance for i in range(3)
        )

    def draw(self, screen):
        """Draw the image on the screen"""
        screen.blit(self.image, (0, 0))
