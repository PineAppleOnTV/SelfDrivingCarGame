"""
A population of driving networks, evolved via selective breeding: no
gradients anywhere - each genome just drives for a generation, the best
performers are kept, and the next generation is bred from them via
crossover + mutation.
"""
import numpy as np

from settings import ELITE_FRACTION, MUTATION_RATE, MUTATION_STRENGTH
from model.network import random_genome, forward_batch, build_observations, extract_single


class Population:
    """
    Duck-types the same get_actions(players, sensor_reading) call
    signature the game's Controller classes use, so it can drive the
    batch of training cars directly - without model/ needing to import
    controller.Controller.
    """

    def __init__(self, size, rng=None):
        self.size = size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.genome = random_genome(self.rng, size)
        self.last_steering = np.zeros(size)
        self.generation = 0

    def reset_state(self):
        """Call whenever the cars respawn - 'last steering' shouldn't carry
        over from the previous generation's run."""
        self.last_steering = np.zeros(self.size)

    def get_actions(self, players, sensor_reading):
        obs = build_observations(players, sensor_reading, self.last_steering)
        throttle, steering = forward_batch(self.genome, obs)
        self.last_steering = steering
        return throttle, steering

    def best_genome(self, fitness):
        """The single fittest network this generation, as a genome with
        leading dim 1 - directly usable by forward_batch() or save_genome()."""
        best_index = int(np.argmax(fitness))
        return extract_single(self.genome, best_index)

    def evolve(self, fitness):
        """
        fitness: (N,) - higher is better. Replaces self.genome with the
        next generation, bred from this generation's top performers.
        """
        order = np.argsort(fitness)[::-1]  # best first
        n_elite = max(1, int(round(self.size * ELITE_FRACTION)))
        elite_indices = order[:n_elite]

        new_genome = {key: np.empty_like(value) for key, value in self.genome.items()}

        # Elitism: the single best network carries over untouched, so a
        # generation's result can never be lost to unlucky mutation.
        best_index = int(order[0])
        for key in new_genome:
            new_genome[key][0] = self.genome[key][best_index]

        for i in range(1, self.size):
            parent_a = elite_indices[self.rng.integers(0, n_elite)]
            parent_b = elite_indices[self.rng.integers(0, n_elite)]
            for key in new_genome:
                mask = self.rng.random(self.genome[key].shape[1:]) < 0.5
                child = np.where(mask, self.genome[key][parent_a], self.genome[key][parent_b])

                mutate_mask = self.rng.random(child.shape) < MUTATION_RATE
                noise = self.rng.normal(0.0, MUTATION_STRENGTH, size=child.shape)
                child = np.where(mutate_mask, child + noise, child)

                new_genome[key][i] = child

        self.genome = new_genome
        self.generation += 1
        self.reset_state()
