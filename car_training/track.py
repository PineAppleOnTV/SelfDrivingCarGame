import pygame
import pygame.surfarray as surfarray
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

    def _colors_close(self, pixel_color):
        """Compares if the color is close to the track color"""
        return all(
            abs(pixel_color[i] - self.track_color[i]) <= self.tolerance for i in range(3)
        )

    def draw(self, screen):
        """Draw the image on the screen"""
        screen.blit(self.image, (0, 0))
