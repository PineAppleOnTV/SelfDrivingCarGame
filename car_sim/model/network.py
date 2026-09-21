"""
Small fixed-architecture MLP used as the driving policy:

    17 inputs -> 32 (tanh) -> 32 (tanh) -> 2 outputs (tanh)

Inputs: 6 horizontal sensor pairs (12), 3 vertical sensors (3), current
speed, and last steering command - matching Sensor.read()'s shape plus
Player's own state. Outputs: throttle, steering, both in [-1, 1].

A "genome" is a dict of weight/bias arrays, each with a leading dimension
N. Every genome in this module is always batched this way - N == 1 for a
single saved network (watch mode), N == population size for training -
so there's exactly one forward-pass code path for both.
"""
import numpy as np

from settings import H_AREAS, V_AREAS, MAX_FORWARD_SPEED

INPUT_SIZE = len(H_AREAS) * 2 + len(V_AREAS) + 2   # horizontal L/R + vertical + speed + last steering
HIDDEN_SIZE = 32
OUTPUT_SIZE = 2                                     # throttle, steering


def random_genome(rng, n):
    """A fresh set of n randomly-initialised networks."""

    def layer(fan_in, fan_out):
        scale = 1.0 / np.sqrt(fan_in)
        w = rng.normal(0.0, scale, size=(n, fan_in, fan_out))
        b = np.zeros((n, fan_out))
        return w, b

    w1, b1 = layer(INPUT_SIZE, HIDDEN_SIZE)
    w2, b2 = layer(HIDDEN_SIZE, HIDDEN_SIZE)
    w3, b3 = layer(HIDDEN_SIZE, OUTPUT_SIZE)
    return {"W1": w1, "b1": b1, "W2": w2, "b2": b2, "W3": w3, "b3": b3}


def genome_size(genome):
    """Number of networks stacked in this genome (its leading dimension)."""
    return genome["W1"].shape[0]


def forward_batch(genome, obs):
    """
    genome: dict of stacked weight/bias arrays, each with leading dim N.
    obs: (N, INPUT_SIZE) observations, one row per network.
    Returns (throttle, steering), each shape (N,), both in [-1, 1].
    """
    h1 = np.tanh(np.einsum("ni,nih->nh", obs, genome["W1"]) + genome["b1"])
    h2 = np.tanh(np.einsum("nh,nhk->nk", h1, genome["W2"]) + genome["b2"])
    out = np.tanh(np.einsum("nk,nko->no", h2, genome["W3"]) + genome["b3"])
    return out[:, 0], out[:, 1]


def build_observations(players, sensor_reading, last_steering):
    """
    Flattens sensor readings + car state into the (N, INPUT_SIZE) array the
    network expects. Speed is normalised as a fraction of max forward speed;
    last_steering is already in [-1, 1] since that's the network's own
    previous output, so it needs no further scaling.
    """
    n = len(players.x)
    horizontal = sensor_reading["horizontal"].reshape(n, -1)  # (N, H*2)
    vertical = sensor_reading["vertical"]                      # (N, V)
    speed = (players.speed / MAX_FORWARD_SPEED)[:, None]       # (N, 1)
    steering = last_steering[:, None]                          # (N, 1)
    return np.concatenate([horizontal, vertical, speed, steering], axis=1).astype(np.float64)


def extract_single(genome, index):
    """Pulls one network out of a batched genome, keeping a leading dim of 1
    so it can be fed straight back into forward_batch()."""
    return {key: value[index:index + 1].copy() for key, value in genome.items()}


def save_genome(path, genome):
    """genome should have leading dim 1 (see extract_single / Population.best_genome)."""
    np.savez(path, **genome)


def load_genome(path):
    """Returns a genome dict with leading dim 1, ready for forward_batch()."""
    data = np.load(path)
    return {key: data[key] for key in data.files}
