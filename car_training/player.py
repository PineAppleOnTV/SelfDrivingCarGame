import pygame
from settings import *
import math

class Player:
    """The 'player' that is controlled."""
    def __init__(self, x=float, y=float, start_angle=0.0):
        self.x = x
        self.y = y
        self.angle = start_angle    # degrees, 0 is along x axis (right)
        self.speed = 0.0    # signed pixels/sec, positive forward, negative backward

        self.car_image = pygame.Surface((PLAYER_LENGTH, PLAYER_WIDTH), pygame.SRCALPHA)
        self.car_image.fill(PLAYER_COLOR)

        self.failed = False # triggers if the car has left the track

    def update(self, dt, throttle, steering):
        """Updates the player location and speed"""
        throttle = max(-1.0, min(1.0, throttle))
        steering = max(-1.0, min(1.0, steering))

        # Throttle changes speed, based on the acceleration stat
        if (self.speed > 0 and throttle > 0) or (self.speed < 0 and throttle < 0):
            self.speed += throttle * ACCELERATION * dt
        else:
            self.speed += throttle * BRAKING * dt

        # If there is no throttle input
        if throttle == 0.0:
            if self.speed > 0.0:
                self.speed = max(0.0, self.speed - FRICTION * dt)
            elif self.speed < 0.0:
                self.speed = min(0.0, self.speed + FRICTION * dt)

        # Speed is capped
        self.speed = max(MAX_REVERSE_SPEED, min(MAX_FORWARD_SPEED, self.speed))

        # Steering with a limiter based on its speed (like loosing grip when going to fast)
        steer_angle = steering * (MAX_STEER_ANGLE - (SPEED_EFFECT * (abs(self.speed)/MAX_FORWARD_SPEED) * MAX_STEER_ANGLE))
        steer_angle_rad = math.radians(steer_angle)
        # Bicycle model: heading depends on speed, wheelbase,
        # and how far the front wheel is turned - not a flat turn speed
        angular_velocity = (self.speed / PLAYER_WHEELBASE) * math.tan(steer_angle_rad)
        self.angle += math.degrees(angular_velocity) * dt

        if self.failed:
            self.speed = 0.0

        # Update the position based on all of these numbers
        radians = math.radians(self.angle)
        self.x += math.cos(radians) * self.speed * dt
        self.y += math.sin(radians) * self.speed * dt

    def constrain_to(self, bounds_rect):
        """Keeps the car center inside the box"""
        self.x = max(bounds_rect.left, min(bounds_rect.right, self.x))
        self.y = max(bounds_rect.top, min(bounds_rect.bottom, self.y))

    def set_failed(self):
        """Triggered if the car leaves the track"""
        self.failed = True
        self.speed = 0.0

    def get_front_position(self):
        """Gets the front position for the sensors"""
        rad = math.radians(self.angle)
        return (
            self.x + math.cos(rad) * (PLAYER_LENGTH / 2),
            self.y + math.sin(rad) * (PLAYER_LENGTH / 2),
        )

    def draw(self, screen):
        """pygame's rotate() spins counter-clockwise for positive angles, but
        our angle grows clockwise on screen (screen y increases downward)
        negating it here keeps the visual rotation matching the physics."""
        rotated_image = pygame.transform.rotate(self.car_image, -self.angle)
        rotated_rect = rotated_image.get_rect(center=(self.x, self.y))
        screen.blit(rotated_image, rotated_rect)