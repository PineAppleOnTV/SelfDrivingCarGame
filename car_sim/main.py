import os

import pygame

from settings import SIZE_TO_FIT_ALL, MODEL_FOLDER, TRAIN_SPEED_MULTIPLIER, MAX_TRAIN_SPEED_MULTIPLIER
from track_manager import TrackManager
from player import Player
from sensor import Sensor
from keyboard.controller import KeyboardController
from model.ml_controller import MLController
from model.network import load_genome
from model.trainer import Trainer

FIXED_DT = 1.0 / 60.0  # one simulated step's worth of time, used for training's fast-forward stepping

TRACK_SWITCH_HINT = "[ ]: prev/next track   1-4: pick track   M: menu   Esc: quit"
CONTROLS_TEXT_KEYBOARD = f"WASD/arrows: drive   R: reset   {TRACK_SWITCH_HINT}"
CONTROLS_TEXT_WATCH = f"C: toggle checkpoints   R: reset   {TRACK_SWITCH_HINT}"
CONTROLS_TEXT_TRAIN = f"+/-: sim speed   {TRACK_SWITCH_HINT}"

MENU_OPTIONS = ["Keyboard driven", "Train model", "Watch best model drive"]
MENU_MODES = ["keyboard", "train", "watch"]

BACKGROUND_COLOR = (30, 30, 30)
TEXT_COLOR = (230, 230, 230)
HINT_COLOR = (150, 150, 150)
HIGHLIGHT_COLOR = (255, 220, 90)


def spawn_player(track_manager):
    """One car, placed at the current track's first checkpoint."""
    start_x, start_y, start_angle = track_manager.get_start_pose()
    return Player(x=[start_x], y=[start_y], start_angle=start_angle)


def draw_hud(screen, font, text):
    screen.blit(font.render(text, True, TEXT_COLOR), (12, 12))


def draw_checkpoints(screen, track_manager, offset):
    """The 'view checkpoints' overlay for watch mode: a dot per checkpoint,
    the start checkpoint picked out in a different colour."""
    offset_x, offset_y = offset
    for checkpoint in track_manager.get_checkpoints():
        pos = (int(checkpoint["x"] + offset_x), int(checkpoint["y"] + offset_y))
        color = (255, 90, 90) if checkpoint["index"] == 0 else (90, 170, 255)
        pygame.draw.circle(screen, color, pos, 5)


def handle_track_switch_keys(event, track_manager):
    """Returns True if this event switched the current track."""
    if event.key == pygame.K_RIGHTBRACKET:
        track_manager.next()
        return True
    if event.key == pygame.K_LEFTBRACKET:
        track_manager.previous()
        return True
    if pygame.K_1 <= event.key <= pygame.K_4:
        track_manager.select(event.key - pygame.K_1)
        return True
    return False


def show_menu(screen, font, clock):
    """Blocking mini-loop: returns the chosen mode name, or None to quit."""
    selected = 0
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                elif event.key in (pygame.K_UP, pygame.K_w):
                    selected = (selected - 1) % len(MENU_OPTIONS)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % len(MENU_OPTIONS)
                elif event.key == pygame.K_RETURN:
                    return MENU_MODES[selected]
                elif pygame.K_1 <= event.key <= pygame.K_3:
                    return MENU_MODES[event.key - pygame.K_1]

        screen.fill(BACKGROUND_COLOR)
        screen.blit(font.render("RL Car Sim - choose a mode", True, TEXT_COLOR), (60, 60))
        for i, option in enumerate(MENU_OPTIONS):
            color = HIGHLIGHT_COLOR if i == selected else TEXT_COLOR
            prefix = "> " if i == selected else "  "
            screen.blit(font.render(f"{prefix}{i + 1}. {option}", True, color), (80, 120 + i * 36))
        hint_y = 120 + len(MENU_OPTIONS) * 36 + 20
        screen.blit(font.render("Up/Down + Enter, or press 1-3   Esc: quit", True, HINT_COLOR), (60, hint_y))
        pygame.display.flip()


def show_message(screen, font, clock, message):
    """Blocking: shows a message until any key is pressed, then returns to
    the menu (or quits, if the window was closed)."""
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                return "menu"

        screen.fill(BACKGROUND_COLOR)
        screen.blit(font.render(message, True, TEXT_COLOR), (60, 100))
        screen.blit(font.render("Press any key to go back to the menu", True, HINT_COLOR), (60, 140))
        pygame.display.flip()


def run_keyboard(screen, font, clock, track_manager):
    controller = KeyboardController()
    screen_center = (SIZE_TO_FIT_ALL[0] / 2, SIZE_TO_FIT_ALL[1] / 2)

    while True:  # outer loop: rebuilds everything whenever the track changes
        sensor = Sensor(track_manager.current_track)
        players = spawn_player(track_manager)
        track_changed = False

        while not track_changed:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    elif event.key == pygame.K_m:
                        return "menu"
                    elif event.key == pygame.K_r:
                        players = spawn_player(track_manager)
                    elif handle_track_switch_keys(event, track_manager):
                        track_changed = True

            if track_changed:
                break

            sensor_reading = sensor.read(players)
            throttle, steering = controller.get_actions(players, sensor_reading)
            players.update(dt, throttle, steering)

            off_track = track_manager.current_track.is_off_track_batch(players.x, players.y)
            players.set_failed(off_track)

            offset = track_manager.get_screen_offset(*screen_center)
            screen.fill(BACKGROUND_COLOR)
            track_manager.draw(screen, *screen_center)
            players.draw(screen, offset=offset)
            draw_hud(screen, font, f"Track: {track_manager.current_name}    {CONTROLS_TEXT_KEYBOARD}")
            pygame.display.flip()


def run_watch(screen, font, clock, track_manager):
    screen_center = (SIZE_TO_FIT_ALL[0] / 2, SIZE_TO_FIT_ALL[1] / 2)

    while True:  # outer loop: reloads whenever the track changes
        model_path = os.path.join(MODEL_FOLDER, f"{track_manager.current_name}_best.npz")
        if not os.path.exists(model_path):
            return show_message(
                screen, font, clock,
                f"No saved model yet for {track_manager.current_name} - train one first."
            )

        controller = MLController(load_genome(model_path))
        sensor = Sensor(track_manager.current_track)
        players = spawn_player(track_manager)
        show_checkpoints = False
        track_changed = False

        while not track_changed:
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    elif event.key == pygame.K_m:
                        return "menu"
                    elif event.key == pygame.K_c:
                        show_checkpoints = not show_checkpoints
                    elif event.key == pygame.K_r:
                        players = spawn_player(track_manager)
                        controller.last_steering[:] = 0.0
                    elif handle_track_switch_keys(event, track_manager):
                        track_changed = True

            if track_changed:
                break

            sensor_reading = sensor.read(players)
            throttle, steering = controller.get_actions(players, sensor_reading)
            players.update(dt, throttle, steering)

            off_track = track_manager.current_track.is_off_track_batch(players.x, players.y)
            players.set_failed(off_track)

            offset = track_manager.get_screen_offset(*screen_center)
            screen.fill(BACKGROUND_COLOR)
            track_manager.draw(screen, *screen_center)
            if show_checkpoints:
                draw_checkpoints(screen, track_manager, offset)
            players.draw(screen, offset=offset)
            draw_hud(screen, font, f"Track: {track_manager.current_name}    {CONTROLS_TEXT_WATCH}")
            pygame.display.flip()


def run_train(screen, font, clock, track_manager):
    screen_center = (SIZE_TO_FIT_ALL[0] / 2, SIZE_TO_FIT_ALL[1] / 2)
    speed_multiplier = TRAIN_SPEED_MULTIPLIER

    while True:  # outer loop: a track switch starts a fresh population
        trainer = Trainer(track_manager)
        track_changed = False

        while not track_changed:
            # clock.tick(60) here only paces rendering/input polling at ~60Hz -
            # it does NOT set the simulated timestep. Each substep below always
            # advances by a fixed FIXED_DT regardless of real elapsed time, and
            # running `speed_multiplier` of them per rendered frame is what
            # makes training run faster than real time (see FIXED_DT comment
            # above): 1 rendered frame now covers speed_multiplier simulated
            # frames, instead of exactly one.
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "quit"
                    elif event.key == pygame.K_m:
                        return "menu"
                    elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                        speed_multiplier = min(speed_multiplier * 2, MAX_TRAIN_SPEED_MULTIPLIER)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        speed_multiplier = max(speed_multiplier // 2, 1)
                    elif handle_track_switch_keys(event, track_manager):
                        track_changed = True

            if track_changed:
                break

            for _ in range(speed_multiplier):
                trainer.update(FIXED_DT)

            offset = track_manager.get_screen_offset(*screen_center)
            screen.fill(BACKGROUND_COLOR)
            track_manager.draw(screen, *screen_center)
            trainer.players.draw(screen, offset=offset)

            alive = int((~trainer.players.failed).sum())
            hud = (
                f"Track: {track_manager.current_name}    "
                f"Gen {trainer.population.generation}    "
                f"Alive {alive}/{trainer.population.size}    "
                f"Best ever {trainer.best_fitness_ever:.0f}    "
                f"Speed {speed_multiplier}x    "
                f"{CONTROLS_TEXT_TRAIN}"
            )
            draw_hud(screen, font, hud)
            pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode(SIZE_TO_FIT_ALL)
    pygame.display.set_caption("RL Car Sim")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    track_manager = TrackManager()

    mode = "menu"
    while mode != "quit":
        if mode == "menu":
            chosen = show_menu(screen, font, clock)
            if chosen is None:
                break
            mode = chosen

        if mode == "keyboard":
            mode = run_keyboard(screen, font, clock, track_manager)
        elif mode == "train":
            mode = run_train(screen, font, clock, track_manager)
        elif mode == "watch":
            mode = run_watch(screen, font, clock, track_manager)

    pygame.quit()


if __name__ == "__main__":
    main()