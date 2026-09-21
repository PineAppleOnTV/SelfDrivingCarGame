import numpy as np
import pygame


class Controller:
    """
    Common interface for anything that drives the batch of cars - a human
    at the keyboard today, an ML policy later. main.py only ever calls
    get_actions(), so swapping controllers doesn't touch the game loop.
    """

    def get_actions(self, players, sensor_reading):
        """
        players: batched Player (arrays x, y, angle, speed, each shape (N,))
        sensor_reading: dict from Sensor.read(players)
                        {'horizontal': (N, H, 2), 'vertical': (N, V)}
        Returns (throttle, steering): numpy arrays of shape (N,), in [-1, 1].
        """
        raise NotImplementedError


class KeyboardController(Controller):
    """
    Drives every car in the batch with the same keyboard input. Meant for
    N == 1 (you driving one car by hand) - sensor_reading is accepted but
    ignored, purely so the interface already matches what a future
    MLController (which WILL use it) will look like.
    """

    def get_actions(self, players, sensor_reading):
        keys = pygame.key.get_pressed()

        throttle_in = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle_in = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle_in = -1.0

        steering_in = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering_in = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering_in = 1.0

        n = len(players.x)
        throttle = np.full(n, throttle_in, dtype=np.float64)
        steering = np.full(n, steering_in, dtype=np.float64)
        return throttle, steering
