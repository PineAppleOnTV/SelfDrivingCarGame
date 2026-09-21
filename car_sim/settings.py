CHECKPOINTS_FOLDER = "../TrackBuilder/checkpoints"
TRACKS_FOLDER = "../TrackBuilder/tracks"
TRACKS = [
    "track_1",
    "track_2",
    "track_3",
    "track_4"
]
SIZE_TO_FIT_ALL = (1920, 1080)
BORDER_COLOR = (0, 0, 0)
TRACK_COLOR = (255, 255, 255)
OFF_TRACK_COLOR = (254, 254, 254)  # background fill outside the track boundary - driving here fails

#Player settings
PLAYER_LENGTH = 50          # vehicle length
PLAYER_WIDTH = 40           # vehicle width
PLAYER_COLOR = (70, 220, 50)
PLAYER_WHEELBASE = 38       # distance between front and rear axle (pixels)

#Movement physics (physics, seconds, degrees)
MAX_FORWARD_SPEED = 600     # pixels/sec at full throttle
MAX_REVERSE_SPEED = -150    # pixels/sec at full reverse throttle
ACCELERATION = 350          # pixels/sec^2 at full throttle
BRAKING = 500               # pixels/sec^2 at full opposite throttle
FRICTION = 180              # pixels/sec^2 decay when throttle = 0
MAX_STEER_ANGLE = 40        # degrees how far the front wheels can turn
SPEED_EFFECT = 0.6          # less steering at speed (0 is no effect 1 is no steering at max speed)

#Area's of interest
STEP_SIZE = 3

#Horizontal
WIDTH_OF_AREA = 150
H_AREAS = [
    0,
    20,
    100,
    120,
    200,
    220
]

# Vertical
LENGTH_OF_AREA = 300
V_AREAS = [
    -20,
    0,
    20
]

# --- Machine learning (model/) ---
MODEL_FOLDER = "model/saved"   # trained genomes are saved here, one best-of per track
POPULATION_SIZE = 50
GENERATION_TIME_LIMIT = 20.0   # seconds before a generation is cut off even if cars are still alive
ELITE_FRACTION = 0.1           # fraction of the population used as breeding parents
MUTATION_RATE = 0.15           # fraction of weights perturbed per child
MUTATION_STRENGTH = 0.4        # stddev of the mutation noise added to a perturbed weight

CHECKPOINT_RADIUS =  12         # pixels - how close a car must get to its next checkpoint to count
                                # as reaching it. Checkpoints in these tracks are spaced ~35px apart,
                                # so this is deliberately a bit larger than that spacing - too small
                                # and a car following a slightly different line than the exact racing
                                # line can get permanently stuck unable to ever reach one waypoint.
ALIVE_PENALTY_PER_SECOND = 12  # small per-second cost just for being alive, so idling/slow lines
                                # score worse than reaching the same checkpoint faster. Progress is
                                # in the same pixel units as the checkpoint distance column (up to
                                # ~600/sec at full speed), so this is a light tax, not a hard cap.

TRAIN_SPEED_MULTIPLIER = 4     # simulated seconds per real second while training (adjustable live
                                # with +/-) - runs this many fixed-timestep physics steps per
                                # rendered frame instead of one, so training doesn't wait on real time
MAX_TRAIN_SPEED_MULTIPLIER = 128