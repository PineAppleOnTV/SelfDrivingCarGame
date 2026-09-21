"""
Runs a Population through repeated generations on whichever track the
TrackManager is currently pointed at: step every car with the current
population's network, advance each car's own next-expected-checkpoint
state, and once a generation ends (everyone's failed, or the time limit
is hit) turn that into a fitness score, breed the next generation, and
start again.

Unlike the rest of model/, this file does import the game's Player and
Sensor - it's the orchestration layer that actually runs a batch of cars,
so that dependency is expected, not something to avoid.
"""
import os

import numpy as np

from settings import POPULATION_SIZE, GENERATION_TIME_LIMIT, MODEL_FOLDER, ALIVE_PENALTY_PER_SECOND
from player import Player
from sensor import Sensor
from model.population import Population
from model.fitness import advance_checkpoints, progress_along_track, apply_alive_penalty
from model.network import save_genome


class Trainer:
    def __init__(self, track_manager):
        self.track_manager = track_manager
        self.population = Population(POPULATION_SIZE)
        self.sensor = Sensor(track_manager.current_track)
        self.best_fitness_ever = -np.inf

        self.players = None
        self.next_idx = None
        self.laps_completed = None
        self.time_alive = None
        self.time_in_generation = 0.0
        self._spawn_generation()

    def _spawn_generation(self):
        start_x, start_y, start_angle = self.track_manager.get_start_pose()
        n = self.population.size
        self.players = Player(
            x=np.full(n, start_x), y=np.full(n, start_y), start_angle=start_angle
        )

        # Every car starts sitting on checkpoint 0, so checkpoint 1 is the
        # first one that actually needs reaching.
        self.next_idx = np.ones(n, dtype=np.int64)
        self.laps_completed = np.zeros(n, dtype=np.int64)
        self.time_alive = np.zeros(n)
        self.time_in_generation = 0.0
        self.population.reset_state()

    def update(self, dt):
        """Advances the simulation by one step; evolves to the next
        generation on its own once one ends."""
        sensor_reading = self.sensor.read(self.players)
        throttle, steering = self.population.get_actions(self.players, sensor_reading)
        self.players.update(dt, throttle, steering)

        off_track = self.track_manager.current_track.is_off_track_batch(
            self.players.x, self.players.y
        )
        self.players.set_failed(off_track)

        checkpoint_xs, checkpoint_ys, checkpoint_distance = self.track_manager.get_checkpoint_arrays()
        checkpoint_xy = np.stack([checkpoint_xs, checkpoint_ys], axis=-1)

        # Failed cars are frozen in place - don't let a checkpoint that
        # happens to sit near where they went off track keep advancing them.
        active = ~self.players.failed
        new_next_idx, new_laps = advance_checkpoints(
            checkpoint_xy, self.players.x, self.players.y, self.next_idx, self.laps_completed
        )
        self.next_idx = np.where(active, new_next_idx, self.next_idx)
        self.laps_completed = np.where(active, new_laps, self.laps_completed)
        self.time_alive = self.time_alive + dt * active

        self.time_in_generation += dt
        all_failed = bool(np.all(self.players.failed))
        time_up = self.time_in_generation >= GENERATION_TIME_LIMIT
        if all_failed or time_up:
            self._end_generation(checkpoint_distance)

    def _end_generation(self, checkpoint_distance):
        progress = progress_along_track(checkpoint_distance, self.next_idx, self.laps_completed)
        fitness = apply_alive_penalty(progress, self.time_alive, ALIVE_PENALTY_PER_SECOND)

        best_this_gen = float(fitness.max())
        if best_this_gen > self.best_fitness_ever:
            self.best_fitness_ever = best_this_gen
            self._save_best(fitness)

        self.population.evolve(fitness)
        self._spawn_generation()

    def _save_best(self, fitness):
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        genome = self.population.best_genome(fitness)
        path = os.path.join(MODEL_FOLDER, f"{self.track_manager.current_name}_best.npz")
        save_genome(path, genome)