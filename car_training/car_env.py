# car_env.py

import os
import numpy as np
import pygame

from settings import (
    FPS, TRACK_IMAGE_PATH, TRACK_SIZE, TRACK_START_POSITION, TRACK_START_DIRECTION,
    WAYPOINT_RADIUS,
    SENSOR_COUNT, SENSOR_FOV, SENSOR_MAX_DISTANCE, SENSOR_STEP,
    PLAYER_LENGTH, PLAYER_WHEELBASE, MAX_STEER_ANGLE, SPEED_EFFECT,
    ACCELERATION, BRAKING, FRICTION, MAX_FORWARD_SPEED, MAX_REVERSE_SPEED,
)
from player import Player
from sensors import SensorArray, VectorizedSensorArray
from track import Track
from assets.track2coordinates import TRACK_WAYPOINTS


class CarEnv:
    """
    Single car on a track - plain reset()/step() environment.
    Used for watching a trained policy (rendering) - NOT used for training.
    """

    def __init__(self, render=False):
        if not pygame.get_init():
            if render:
                pass
            else:
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()

        if pygame.display.get_surface() is None:
            if render:
                self.screen = pygame.display.set_mode(TRACK_SIZE)
            else:
                pygame.display.set_mode(TRACK_SIZE)
        else:
            self.screen = pygame.display.get_surface()

        if render:
            self.clock = pygame.time.Clock()
        self.render_enabled = render

        self.dt = 1.0 / FPS
        self.track = Track(TRACK_IMAGE_PATH)
        self.sensor_array = SensorArray(SENSOR_COUNT, SENSOR_FOV, SENSOR_MAX_DISTANCE, SENSOR_STEP)

        self.car = Player(
            x=TRACK_START_POSITION[0],
            y=TRACK_START_POSITION[1],
            start_angle=TRACK_START_DIRECTION,
        )

    def reset(self):
        self.car.x, self.car.y = TRACK_START_POSITION
        self.car.angle = TRACK_START_DIRECTION
        self.car.speed = 0.0
        self.car.failed = False
        return self._get_observation()

    def step(self, action):
        throttle, steering = float(action[0]), float(action[1])
        self.car.update(self.dt, throttle, steering)
        self.car.constrain_to(pygame.Rect(0, 0, *self.track.get_size()))

        on_track = self.track.is_on_track(self.car.x, self.car.y)

        if on_track:
            reward = -0.01
            done = False
        else:
            self.car.set_failed()
            reward = -1.0
            done = True

        obs = self._get_observation()
        info = {}
        return obs, reward, done, info

    def _get_observation(self):
        front_x, front_y = self.car.get_front_position()
        normalized_sensors = self.sensor_array.sense(front_x, front_y, self.car.angle, self.track)

        speed_range = MAX_FORWARD_SPEED if self.car.speed >= 0 else -MAX_REVERSE_SPEED
        normalized_speed = self.car.speed / speed_range

        return np.array(normalized_sensors + [normalized_speed], dtype=np.float32)

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((30, 30, 30))
        self.track.draw(self.screen)
        self.car.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)


class MultiCarEnv:
    """
    Many cars simulated together with fully vectorized (numpy array-based)
    state and physics - used for fast parallel training. No Player objects
    involved here; position/speed/angle for ALL cars live in single arrays.
    """

    def __init__(self, num_cars):
        self.num_cars = num_cars

        if not pygame.get_init():
            if pygame.display.get_surface() is None:
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode(TRACK_SIZE)

        self.dt = 1.0 / FPS
        self.track = Track(TRACK_IMAGE_PATH)
        self.sensors = VectorizedSensorArray(SENSOR_COUNT, SENSOR_FOV, SENSOR_MAX_DISTANCE, SENSOR_STEP)

        self.bounds_w, self.bounds_h = self.track.get_size()
        self.waypoints = np.array(TRACK_WAYPOINTS, dtype=np.float32)

        n = num_cars
        self.x = np.full(n, float(TRACK_START_POSITION[0]), dtype=np.float32)
        self.y = np.full(n, float(TRACK_START_POSITION[1]), dtype=np.float32)
        self.angle = np.full(n, float(TRACK_START_DIRECTION), dtype=np.float32)
        self.speed = np.zeros(n, dtype=np.float32)
        self.next_waypoint_idx = np.zeros(n, dtype=int)

    def reset(self):
        self._reset_mask(np.ones(self.num_cars, dtype=bool))
        return self._get_all_observations()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        throttle = np.clip(actions[:, 0], -1.0, 1.0)
        steering = np.clip(actions[:, 1], -1.0, 1.0)

        same_dir = ((self.speed > 0) & (throttle > 0)) | ((self.speed < 0) & (throttle < 0))
        accel = np.where(same_dir, throttle * ACCELERATION, throttle * BRAKING)
        self.speed += accel * self.dt

        zero_throttle = throttle == 0.0
        self.speed = np.where(zero_throttle & (self.speed > 0),
                               np.maximum(0.0, self.speed - FRICTION * self.dt), self.speed)
        self.speed = np.where(zero_throttle & (self.speed < 0),
                               np.minimum(0.0, self.speed + FRICTION * self.dt), self.speed)

        self.speed = np.clip(self.speed, MAX_REVERSE_SPEED, MAX_FORWARD_SPEED)

        steer_angle = steering * (MAX_STEER_ANGLE - SPEED_EFFECT * (np.abs(self.speed) / MAX_FORWARD_SPEED) * MAX_STEER_ANGLE)
        angular_velocity = (self.speed / PLAYER_WHEELBASE) * np.tan(np.radians(steer_angle))
        self.angle += np.degrees(angular_velocity) * self.dt

        rad = np.radians(self.angle)
        self.x += np.cos(rad) * self.speed * self.dt
        self.y += np.sin(rad) * self.speed * self.dt

        self.x = np.clip(self.x, 0, self.bounds_w)
        self.y = np.clip(self.y, 0, self.bounds_h)

        on_track = self.track.is_on_track_batch(self.x, self.y)

        targets = self.waypoints[self.next_waypoint_idx]
        dist_to_target = np.hypot(self.x - targets[:, 0], self.y - targets[:, 1])
        reached = on_track & (dist_to_target < WAYPOINT_RADIUS)

        rewards = np.where(on_track, -0.01 + reached * 8.0, -3.0).astype(np.float32)
        dones = ~on_track

        self.next_waypoint_idx = np.where(reached,
                                           (self.next_waypoint_idx + 1) % len(self.waypoints),
                                           self.next_waypoint_idx)

        self._reset_mask(dones)

        observations = self._get_all_observations()
        infos = [{} for _ in range(self.num_cars)]
        return observations, rewards, dones, infos

    def _reset_mask(self, mask):
        self.x[mask] = TRACK_START_POSITION[0]
        self.y[mask] = TRACK_START_POSITION[1]
        self.angle[mask] = TRACK_START_DIRECTION
        self.speed[mask] = 0.0
        self.next_waypoint_idx[mask] = 0

    def _get_all_observations(self):
        rad = np.radians(self.angle)
        front_x = self.x + np.cos(rad) * (PLAYER_LENGTH / 2)
        front_y = self.y + np.sin(rad) * (PLAYER_LENGTH / 2)

        normalized_sensors = self.sensors.sense_batch(front_x, front_y, self.angle, self.track)

        speed_range = np.where(self.speed >= 0, MAX_FORWARD_SPEED, -MAX_REVERSE_SPEED)
        normalized_speed = (self.speed / speed_range)[:, None]

        return np.concatenate([normalized_sensors, normalized_speed], axis=1).astype(np.float32)
