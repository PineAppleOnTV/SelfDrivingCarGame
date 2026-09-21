"""
Drives a single, already-trained genome - used by 'watch best model
drive' mode. Duck-types the same get_actions(players, sensor_reading)
call signature as controller.Controller, without importing it, so
model/ doesn't need to depend on the game's controller module.
"""
import numpy as np

from model.network import forward_batch, build_observations, genome_size


class MLController:
    def __init__(self, genome):
        self.genome = genome
        self.last_steering = np.zeros(genome_size(genome))

    def get_actions(self, players, sensor_reading):
        obs = build_observations(players, sensor_reading, self.last_steering)
        throttle, steering = forward_batch(self.genome, obs)
        self.last_steering = steering
        return throttle, steering
