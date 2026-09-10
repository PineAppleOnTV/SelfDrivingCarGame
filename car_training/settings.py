#Frame settings
FPS = 60

#Track settings
TRACK_IMAGE_PATH = "assets/track2.png"
TRACK_SIZE = (1600, 900)                # Track size(used to set the window correctly)
TRACK_START_POSITION = (100, 450)        # Start position for the car
TRACK_START_DIRECTION = 270.0           # Direction in degrees how the car starts (0 is right)
TRACK_COLOR = (30, 30, 30)              # Track color (Used to see what the track is)
TRACK_COLOR_TOLERANCE = 30              # Tolerence on pixel color to be seen as track

#Player settings
PLAYER_LENGTH = 40          # vehicle length
PLAYER_WIDTH = 20           # vehicle width
PLAYER_COLOR = (70, 220, 50)
PLAYER_WHEELBASE = 35       # distance between front and rear axle (pixels)

#Movement physics (physics, seconds, degrees)
MAX_FORWARD_SPEED = 600     # pixels/sec at full throttle
MAX_REVERSE_SPEED = -150    # pixels/sec at full reverse throttle
ACCELERATION = 350          # pixels/sec^2 at full throttle
BRAKING = 500               # pixels/sec^2 at full opposite throttle
FRICTION = 180              # pixels/sec^2 decay when throttle = 0
MAX_STEER_ANGLE = 40        # degrees how far the front wheels can turn
SPEED_EFFECT = 0.6          # the effect speed has on steering (0 is no effect 1.0 is no steering at max speed)

#Sensor settings
SENSOR_COUNT = 7            # amount of sensors present
SENSOR_FOV = 120            # field of view that the sensors cover
SENSOR_MAX_DISTANCE = 250   # Max distance a sensor can see
SENSOR_STEP = 6             # Control how many pixels a jump is between checks
