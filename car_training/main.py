import sys
import pygame

from settings import *
from player import Player
from controls import KeyBoardController
from track import Track
from sensors import SensorArray

def main():
    """Runs the game loop"""

    pygame.init()
    # Sets up the screen width the size from settings
    screen = pygame.display.set_mode(TRACK_SIZE)
    pygame.display.set_caption("Learning Game")
    clock = pygame.time.Clock()


    track = Track(TRACK_IMAGE_PATH)
    # Sets up the player with a starting location
    player = Player(
        x=TRACK_START_POSITION[0],
        y=TRACK_START_POSITION[1],
        start_angle=TRACK_START_DIRECTION
    )
    sensors = SensorArray(SENSOR_COUNT, SENSOR_FOV, SENSOR_MAX_DISTANCE, SENSOR_STEP)

    # Sets up the keyboard controller
    controller = KeyBoardController()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0 # Converts millisecond to seconds (in float)

        # Quits if the user presses the red cross
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Start of game loop
        throttle, steering, reset = controller.get_action()
        # Reset player value when pressed
        if reset:
            player.x, player.y, player.angle, player.speed = (
                TRACK_START_POSITION[0],
                TRACK_START_POSITION[1],
                TRACK_START_DIRECTION,
                0.0
            )
            player.failed = False

        player.update(dt=dt, throttle=throttle, steering=steering)
        front_x, front_y = player.get_front_position()
        sensor_readings = sensors.sense(front_x, front_y, player.angle, track)
        player.constrain_to(screen.get_rect())

        # Sets the speed to 0 and triggers a failed mode inside the player
        if not track.is_on_track(player.x, player.y):
            player.set_failed()

        screen.fill((30, 30, 30))
        track.draw(screen)
        player.draw(screen)
        endpoints = sensors.get_endpoints(front_x, front_y, player.angle, [distance * SENSOR_MAX_DISTANCE for distance in sensor_readings])
        for end_x, end_y in endpoints:
            pygame.draw.line(screen, (255, 0, 0), (front_x, front_y), (end_x, end_y), 1)
        pygame.display.flip()

        #Limits to given FPS
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__=="__main__":
    main()
