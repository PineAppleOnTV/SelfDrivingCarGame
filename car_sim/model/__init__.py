"""
Machine-learning side of the sim, kept separate from the game code.

Only settings.py is a dependency going this direction - everything here
takes plain arrays / duck-typed objects (a Player-shaped thing, a
Sensor.read()-shaped dict) as input, so it doesn't need to import Player,
Track, Sensor, TrackManager or Controller at all. model/trainer.py is the
one exception: it orchestrates an actual training run, so it does import
Player and Sensor to build the population's cars.
"""
