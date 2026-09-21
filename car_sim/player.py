import numpy as np
import pygame
from settings import (
    PLAYER_LENGTH, PLAYER_WIDTH, PLAYER_COLOR, PLAYER_WHEELBASE,
    ACCELERATION, BRAKING, FRICTION, MAX_FORWARD_SPEED, MAX_REVERSE_SPEED,
    MAX_STEER_ANGLE, SPEED_EFFECT,
)


class Player:
    """
    Batched state for N cars at once. Every attribute is a numpy array of
    shape (N,) instead of one Player object per car - same physics as
    before, just applied to all N cars in one set of array ops per step.
    """

    def __init__(self, x, y, start_angle=0.0):
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.y = np.asarray(y, dtype=np.float64).copy()
        n = len(self.x)

        if np.isscalar(start_angle):
            self.angle = np.full(n, float(start_angle), dtype=np.float64)
        else:
            self.angle = np.asarray(start_angle, dtype=np.float64).copy()

        self.speed = np.zeros(n, dtype=np.float64)   # signed px/sec, positive forward
        self.failed = np.zeros(n, dtype=bool)         # per-car "left the track" flag

        self.car_image = pygame.Surface((PLAYER_LENGTH, PLAYER_WIDTH), pygame.SRCALPHA)
        self.car_image.fill(PLAYER_COLOR)

    def update(self, dt, throttle, steering):
        """
        Updates every car's position and speed at once.
        throttle, steering: arrays of shape (N,), each clamped to [-1, 1].
        """
        throttle = np.clip(np.asarray(throttle, dtype=np.float64), -1.0, 1.0)
        steering = np.clip(np.asarray(steering, dtype=np.float64), -1.0, 1.0)

        # Throttle changes speed, based on the acceleration stat.
        # Accelerating = pushing further in the direction already moving;
        # otherwise (opposite direction, or standing still) it's braking.
        accelerating = ((self.speed > 0) & (throttle > 0)) | ((self.speed < 0) & (throttle < 0))
        rate = np.where(accelerating, ACCELERATION, BRAKING)
        self.speed = self.speed + throttle * rate * dt

        # Friction only where there's no throttle input at all.
        coasting = throttle == 0.0
        self.speed = np.where(coasting & (self.speed > 0),
                               np.maximum(0.0, self.speed - FRICTION * dt), self.speed)
        self.speed = np.where(coasting & (self.speed < 0),
                               np.minimum(0.0, self.speed + FRICTION * dt), self.speed)

        # Speed is capped
        self.speed = np.clip(self.speed, MAX_REVERSE_SPEED, MAX_FORWARD_SPEED)

        # Steering with a limiter based on speed (losing grip when going too fast)
        steer_angle = steering * (
            MAX_STEER_ANGLE - (SPEED_EFFECT * (np.abs(self.speed) / MAX_FORWARD_SPEED) * MAX_STEER_ANGLE)
        )
        steer_angle_rad = np.radians(steer_angle)
        # Bicycle model: heading depends on speed, wheelbase, and how far the
        # front wheel is turned - not a flat turn speed.
        angular_velocity = (self.speed / PLAYER_WHEELBASE) * np.tan(steer_angle_rad)
        self.angle = self.angle + np.degrees(angular_velocity) * dt

        self.speed = np.where(self.failed, 0.0, self.speed)

        # Update position for all cars at once
        radians = np.radians(self.angle)
        self.x = self.x + np.cos(radians) * self.speed * dt
        self.y = self.y + np.sin(radians) * self.speed * dt

    def constrain_to(self, bounds_rect):
        """Keeps every car's center inside the box"""
        self.x = np.clip(self.x, bounds_rect.left, bounds_rect.right)
        self.y = np.clip(self.y, bounds_rect.top, bounds_rect.bottom)

    def set_failed(self, mask):
        """mask: bool array (N,) - True marks that car as failed (left the track)."""
        self.failed = self.failed | np.asarray(mask, dtype=bool)
        self.speed = np.where(self.failed, 0.0, self.speed)

    def get_front_position(self):
        """Gets the front position of every car, for the sensors"""
        rad = np.radians(self.angle)
        fx = self.x + np.cos(rad) * (PLAYER_LENGTH / 2)
        fy = self.y + np.sin(rad) * (PLAYER_LENGTH / 2)
        return fx, fy

    def draw(self, screen, offset=(0, 0)):
        """
        pygame drawing isn't vectorizable - this stays a per-car loop. At
        N ~100 this loop is blitting/rendering cost, not simulation cost,
        so it isn't the bottleneck the physics/sensor code was.

        offset: (screen_x, screen_y) for world position (0, 0) - i.e. the
        track's Track.screen_offset(...), so the car is drawn in the same
        place on screen that the (centered) track image is drawn.
        """
        offset_x, offset_y = offset
        for i in range(len(self.x)):
            rotated_image = pygame.transform.rotate(self.car_image, -self.angle[i])
            rotated_rect = rotated_image.get_rect(
                center=(self.x[i] + offset_x, self.y[i] + offset_y)
            )
            screen.blit(rotated_image, rotated_rect)
