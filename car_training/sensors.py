# sensors.py

import math
import numpy as np


class SensorArray:
    """
    A fan of raycast sensors simulating a forward-facing camera's field of view.
    Single-car version - used by CarEnv (rendering/watching).
    """

    def __init__(self, count=5, fov_degrees=120, max_distance=200, step=4):
        self.max_distance = max_distance
        self.step = step

        if count == 1:
            self.angle_offsets = [0.0]
        else:
            half_fov = fov_degrees / 2
            self.angle_offsets = [-half_fov + i * (fov_degrees / (count - 1)) for i in range(count)]

    def sense(self, origin_x, origin_y, heading_degrees, track):
        """Casts all rays from origin x, y facing heading_degrees.
        Returns a list of NORMALIZED distances (0-1), in order of offsets."""
        return [self._cast_ray(origin_x, origin_y, heading_degrees + offset, track)
                for offset in self.angle_offsets]

    def _cast_ray(self, x, y, angle_degrees, track):
        rad = math.radians(angle_degrees)
        dx, dy = math.cos(rad), math.sin(rad)

        distance = 0
        while distance <= self.max_distance:
            test_x = x + dx * distance
            test_y = y + dy * distance
            if not track.is_on_track(test_x, test_y):
                return float(distance / self.max_distance)
            distance += self.step

        return 1.0

    def get_endpoints(self, origin_x, origin_y, heading_degrees, distances):
        """
        Given NORMALIZED distances from sense(), returns the actual (x, y)
        endpoint of each ray - purely for debug drawing.
        """
        endpoints = []
        for offset, norm_distance in zip(self.angle_offsets, distances):
            rad = math.radians(heading_degrees + offset)
            pixel_distance = norm_distance * self.max_distance
            endpoints.append((
                origin_x + math.cos(rad) * pixel_distance,
                origin_y + math.sin(rad) * pixel_distance,
            ))
        return endpoints


class VectorizedSensorArray:
    """
    Same raycasting concept as SensorArray, but casts ALL rays for ALL
    cars simultaneously - one array operation per marching step, instead
    of one Python-level call per (car, ray, step) combination.
    Used by MultiCarEnv (training).
    """

    def __init__(self, count=5, fov_degrees=120, max_distance=200, step=4):
        self.max_distance = max_distance
        self.step = step

        if count == 1:
            self.angle_offsets = np.array([0.0])
        else:
            half_fov = fov_degrees / 2
            self.angle_offsets = np.array([
                -half_fov + i * (fov_degrees / (count - 1))
                for i in range(count)
            ])

    def sense_batch(self, origin_x, origin_y, heading_deg, track):
        """
        origin_x, origin_y, heading_deg: arrays of shape (num_cars,)
        Returns NORMALIZED distances (0-1), shape (num_cars, num_sensors) -
        matches SensorArray.sense()'s convention exactly.
        """
        num_cars = len(origin_x)
        num_sensors = len(self.angle_offsets)

        ray_angles = heading_deg[:, None] + self.angle_offsets[None, :]  # (cars, sensors)
        rad = np.radians(ray_angles)
        dx = np.cos(rad)
        dy = np.sin(rad)

        results = np.ones((num_cars, num_sensors), dtype=np.float32)  # default: clear all the way (1.0)
        active = np.ones((num_cars, num_sensors), dtype=bool)

        distance = 0.0
        while distance <= self.max_distance:
            test_x = origin_x[:, None] + dx * distance
            test_y = origin_y[:, None] + dy * distance

            on_track = track.is_on_track_batch(test_x.ravel(), test_y.ravel()).reshape(num_cars, num_sensors)

            newly_off = active & (~on_track)
            results[newly_off] = distance / self.max_distance
            active &= on_track

            if not active.any():
                break

            distance += self.step

        return results