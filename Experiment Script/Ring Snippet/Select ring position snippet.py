### virtual environment: alien_features
### installed packages in the virtual environment:
# - Psychopy v2026.1.2
# - Python v3.10.12

## This code snippet is designed to allow the user to select a ring position on the screen using the mouse. The selected position will be stored in the variable `ring_position`. The code uses the PsychoPy library to create a visual window and handle mouse interactions.
## This function can be used for learning the material and for testing memories.
## If for learning, set parameter `draggable = False` to prevent the user from dragging the bar, `button_visible = False' and set `initial_angle` to a specific value to control where the bar starts.
## if for testing, set `draggable = True` to allow the user to drag the bar, `button_visible = True` to show the submit button, and set `initial_angle` to None to randomize the starting position of the bar.

import random
import math
from psychopy import visual, event, core


def ring_position_response(
    win=None,
    ring_radius=0.30,
    ring_line_width=4,
    bar_length=0.10,
    bar_width=0.003,
    submit_pos=(0, -0.42),
    submit_size=(0.22, 0.08),
    units="height",
    bg_color=None,
    ring_color="white",
    bar_color="red",
    button_color="darkgray",
    button_text_color="white",
    draggable=True,
    initial_angle=None,
    show_angle_text=True,
    button_visible=True,
    allow_escape=True,
):
    """
    Display a ring with a radial draggable bar for angular position selection.

    The midpoint of the bar lies exactly on the ring.
    The participant can drag the bar if draggable=True.
    The response is recorded only after clicking and releasing on Submit.

    Angle definition:
        0 degrees   = right
        90 degrees  = top
        180 degrees = left
        270 degrees = bottom

    Returns
    -------
    response_angle : int or None
        Selected angle in whole degrees, 0 to 359.
        None if ESC is pressed.
    start_angle : int
        Initial angle or random start angle.
    """

    created_window = False

    if win is None:
        win_kwargs = dict(
            size=(1000, 800),
            units=units,
            fullscr=False
        )
        if bg_color is not None:
            win_kwargs["color"] = bg_color

        win = visual.Window(**win_kwargs)
        created_window = True

    mouse = event.Mouse(win=win)
    if initial_angle is not None:
        start_angle = initial_angle % 360
    else:
        start_angle = random.randint(0, 359)
    current_angle = float(start_angle)

    ring = visual.Circle(
        win=win,
        radius=ring_radius,
        edges=512,
        lineColor=ring_color,
        fillColor=None,
        lineWidth=ring_line_width,
        pos=(0, 0),
        interpolate=True
    )

    bar = visual.Rect(
        win=win,
        width=bar_width,
        height=bar_length,
        fillColor=bar_color,
        lineColor=bar_color,
        interpolate=True
    )

    angle_text = visual.TextStim(
        win=win,
        text="",
        pos=(0, -0.36),
        height=0.035,
        color="white"
    )

    submit_button = visual.Rect(
        win=win,
        width=submit_size[0],
        height=submit_size[1],
        pos=submit_pos,
        fillColor=button_color,
        lineColor="white",
        lineWidth=2
    )

    submit_label = visual.TextStim(
        win=win,
        text="Submit",
        pos=submit_pos,
        height=0.035,
        color=button_text_color
    )

    def angle_to_xy(angle_deg, radius):
        theta = math.radians(angle_deg)
        return radius * math.cos(theta), radius * math.sin(theta)

    def xy_to_angle(x, y):
        return math.degrees(math.atan2(y, x)) % 360

    def update_bar(angle_deg):
        # Bar midpoint exactly on ring
        x, y = angle_to_xy(angle_deg, ring_radius)
        bar.pos = (x, y)

        # Radial orientation
        bar.ori = 90 - angle_deg

    def point_inside_rect(point, rect_stim):
        """
        Manual rectangle hit-test.
        More predictable than mouse.isPressedIn during drag states.
        Assumes the submit button is not rotated.
        """
        px, py = point
        rx, ry = rect_stim.pos
        half_w = rect_stim.width / 2
        half_h = rect_stim.height / 2

        return (
            rx - half_w <= px <= rx + half_w and
            ry - half_h <= py <= ry + half_h
        )

    update_bar(current_angle)

    dragging = False
    response_angle = None

    # Prevent old mouse press from immediately doing something
    mouse.clickReset()

    while mouse.getPressed()[0]:
        event.getKeys()
        ring.draw()
        bar.draw()
        if button_visible:
            submit_button.draw()
            submit_label.draw()
        win.flip()

    previous_left_pressed = False
    submit_press_started = False

    while True:
        if allow_escape and "escape" in event.getKeys():
            response_angle = None
            break

        mx, my = mouse.getPos()
        left_pressed = mouse.getPressed()[0]

        mouse_on_button = point_inside_rect((mx, my), submit_button)

        # Detect new mouse press
        new_press = left_pressed and not previous_left_pressed

        # Detect mouse release
        new_release = (not left_pressed) and previous_left_pressed

        # -------------------------
        # Start submit click
        # -------------------------
        # Only start a submit click if:
        # - button is visible
        # - user is not dragging
        # - new press begins inside the button
        if (
            button_visible and
            new_press and
            mouse_on_button and
            not dragging
        ):
            submit_press_started = True

        # -------------------------
        # Start or continue dragging
        # -------------------------
        if draggable:
            bx, by = bar.pos
            distance_to_bar = math.sqrt((mx - bx) ** 2 + (my - by) ** 2)
            grab_radius = max(bar_length * 0.8, 0.03)

            # Start dragging only if the press begins near the bar,
            # not if the press begins on the submit button.
            if new_press and distance_to_bar <= grab_radius and not mouse_on_button:
                dragging = True

            if dragging and left_pressed:
                current_angle = xy_to_angle(mx, my)
                update_bar(current_angle)

            if dragging and new_release:
                dragging = False

        # -------------------------
        # Submit only on release
        # -------------------------
        # This prevents the dragging mouse press from closing the window.
        if (
            button_visible and
            submit_press_started and
            new_release and
            mouse_on_button and
            not dragging
        ):
            response_angle = int(round(current_angle)) % 360
            break

        # If press began on button but release occurred elsewhere, cancel submit
        if submit_press_started and new_release and not mouse_on_button:
            submit_press_started = False

        if show_angle_text:
            angle_text.text = f"{int(round(current_angle)) % 360}°"

        # Draw
        ring.draw()
        bar.draw()

        if show_angle_text:
            angle_text.draw()

        if button_visible:
            submit_button.draw()
            submit_label.draw()

        win.flip()

        previous_left_pressed = left_pressed

    core.wait(0.1)

    if created_window:
        win.close()

    return response_angle, start_angle
response, initial = ring_position_response()
print(f"Selected angle: {response}°, Initial angle: {initial}°")