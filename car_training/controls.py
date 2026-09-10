import pygame

class KeyBoardController:
    """
    Turns keyboard state into a car-style action: (throttle, steering),
    both floats in [-1.0, 1.0].

    throttle:  +1.0 = full accel forward, -1.0 = full accel reverse, 0 = coast
    steering:  +1.0 = full turn right,    -1.0 = full turn left,    0 = straight
    """

    def get_action(self):
        """Gets the movement from keypresses"""
        keys = pygame.key.get_pressed()

        throttle = 0.0
        # forward motion on w
        if keys[pygame.K_w]:
            throttle += 1.0
        # reverse motion on s
        if keys[pygame.K_s]:
            throttle -= 1.0

        steering = 0.0
        # Steering left on a
        if keys[pygame.K_a]:
            steering -= 1
        # Steering right on d
        if keys[pygame.K_d]:
            steering += 1

        # Reset
        if keys[pygame.K_r]:
            return throttle, steering, True

        return throttle, steering, False
