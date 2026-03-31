### virtual environment: alien_features
### installed packages in the virtual environment:
# - Psychopy v2026.1.2
# - Python v3.10.12

### This code snippet creates a color ring using Psychopy. The color ring is made up of multiple segments, each with a different color. The colors are generated using the color space CIELCh, which allows for easy creation of a spectrum of colors.
### In this example, L = 65 (lightness), C = 40 (chroma), and H (hue) varies from 0 to 360 degrees in steps of 1 degree.
### The recorded hue is the true hue corresponding to a non rotated ring where red is approciamtely at 0/360 degrees. The ring is randomly rotated each time the program is run, so the screen angle of the selector will not directly correspond to the hue value. The user can click and drag a selector around the ring to choose a color, and then click a submit button to output the selected color's properties.
### Note: The code includes some best-effort attempts to raise the window on top when it starts, but this may not work on all platforms or backends. The user may need to manually focus the window if it does not appear on top.
### The ring is drawn as a series of ShapeStim objects, which are essentially quadrilaterals that form the segments of the ring. The color of each segment is calculated using the lch_to_psychopy_rgb function, which converts from CIELCh to the RGB color space used by PsychoPy.
### The selector is drawn as a line and a knob that follows the mouse position when dragging. The selected color is previewed in a circle, and the hue value is displayed as text below it. The user can submit their selection by clicking the submit button, which will print the selected color's properties to the console.
### The printed results include the screen angle of the selector, the random ring rotation, the true hue value, the L and C values, and the RGB color in PsychoPy's -1 to 1 range.


from psychopy import visual, event, core
from psychopy.hardware import keyboard
import numpy as np
import random
import math

# =========================
# Color conversion helpers
# =========================

# D65 reference white
XN = 95.047
YN = 100.000
ZN = 108.883


def lch_to_lab(L, C, h_deg):
    h_rad = np.deg2rad(h_deg)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)
    return L, a, b


def lab_to_xyz(L, a, b):
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def f_inv(t):
        delta = 6 / 29
        if t > delta:
            return t ** 3
        return 3 * (delta ** 2) * (t - 4 / 29)

    x = XN * f_inv(fx)
    y = YN * f_inv(fy)
    z = ZN * f_inv(fz)
    return x, y, z


def xyz_to_linear_rgb(x, y, z):
    x /= 100.0
    y /= 100.0
    z /= 100.0

    r_lin = x * 3.2406 + y * -1.5372 + z * -0.4986
    g_lin = x * -0.9689 + y * 1.8758 + z * 0.0415
    b_lin = x * 0.0557 + y * -0.2040 + z * 1.0570
    return np.array([r_lin, g_lin, b_lin], dtype=float)


def linear_to_srgb(rgb_lin):
    out = np.empty(3, dtype=float)
    for i, c in enumerate(rgb_lin):
        if c <= 0.0031308:
            out[i] = 12.92 * c
        else:
            out[i] = 1.055 * (c ** (1 / 2.4)) - 0.055
    return out


def lch_to_psychopy_rgb(L, C, h_deg):
    L_, a_, b_ = lch_to_lab(L, C, h_deg)
    x, y, z = lab_to_xyz(L_, a_, b_)
    rgb_lin = xyz_to_linear_rgb(x, y, z)
    rgb = linear_to_srgb(rgb_lin)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb * 2.0 - 1.0


# =========================
# Geometry helpers
# =========================

def pol_to_cart(r, ang_deg):
    a = np.deg2rad(ang_deg)
    return np.array([r * np.cos(a), r * np.sin(a)], dtype=float)


def angle_from_xy(x, y):
    ang = np.degrees(np.arctan2(y, x))
    if ang < 0:
        ang += 360
    return ang


def point_in_rect(pt, center, w, h):
    x, y = pt
    cx, cy = center
    return (cx - w / 2 <= x <= cx + w / 2) and (cy - h / 2 <= y <= cy + h / 2)


def distance(x, y):
    return math.hypot(x, y)


# =========================
# Parameters
# =========================

L_VAL = 65
C_VAL = 40

WIN_SIZE = (1100, 760)

RING_RADIUS = 190
RING_WIDTH = 34
INNER_R = RING_RADIUS - RING_WIDTH / 2
OUTER_R = RING_RADIUS + RING_WIDTH / 2
RING_TOL = 28

N_SEGMENTS = 360

PREVIEW_POS = (320, 40)
PREVIEW_RADIUS = 72

SUBMIT_POS = (320, -185)
SUBMIT_W = 175
SUBMIT_H = 66

BG = (-0.72, -0.72, -0.72)

# Random ring rotation each run
ring_rotation = random.uniform(0, 360)

# Initial selector position in screen coordinates
selector_angle_screen = random.uniform(0, 360)


# =========================
# Window and input devices
# =========================

win = visual.Window(
    size=WIN_SIZE,
    units='pix',
    monitor='testMonitor',
    color=BG,
    colorSpace='rgb',
    allowGUI=True,
    fullscr=False,
    waitBlanking=True
)

mouse = event.Mouse(win=win)
mouse.setVisible(True)
mouse.clickReset()

kb = keyboard.Keyboard()

# Try to activate / raise the window if backend supports it
# This is best-effort only.
try:
    if hasattr(win, "winHandle") and win.winHandle is not None:
        wh = win.winHandle

        # pyglet-style methods if available
        if hasattr(wh, "activate"):
            wh.activate()
        if hasattr(wh, "set_visible"):
            wh.set_visible(True)
        if hasattr(wh, "minimize") and False:
            wh.minimize()  # disabled intentionally
except Exception:
    pass

# Show one frame first
startup_text = visual.TextStim(
    win,
    text='Loading color ring...',
    color='white',
    height=28
)
startup_text.draw()
win.flip()
core.wait(0.15)

event.clearEvents()
kb.clearEvents()


# =========================
# Build hue ring as sectors
# =========================

ring_sectors = []

for i in range(N_SEGMENTS):
    hue = i
    col = lch_to_psychopy_rgb(L_VAL, C_VAL, hue)

    a1 = i + ring_rotation
    a2 = i + 1 + ring_rotation

    p1o = pol_to_cart(OUTER_R, a1)
    p2o = pol_to_cart(OUTER_R, a2)
    p2i = pol_to_cart(INNER_R, a2)
    p1i = pol_to_cart(INNER_R, a1)

    sector = visual.ShapeStim(
        win=win,
        vertices=np.array([p1o, p2o, p2i, p1i]),
        fillColor=col,
        lineColor=col,
        lineWidth=0,
        colorSpace='rgb',
        closeShape=True,
        interpolate=True
    )
    ring_sectors.append(sector)


# Optional thin outlines to make the annulus edge look cleaner
outer_outline = visual.Circle(
    win,
    radius=OUTER_R,
    edges=256,
    lineColor=(0.2, 0.2, 0.2),
    lineWidth=1,
    fillColor=None,
    colorSpace='rgb'
)

inner_outline = visual.Circle(
    win,
    radius=INNER_R,
    edges=256,
    lineColor=BG,
    lineWidth=2,
    fillColor=None,
    colorSpace='rgb'
)


# =========================
# Selector
# =========================

selector_line = visual.Line(
    win,
    start=(0, 0),
    end=(0, 0),
    lineColor='white',
    colorSpace='named',
    lineWidth=6
)

selector_knob = visual.Circle(
    win,
    radius=9,
    pos=(0, 0),
    fillColor='white',
    lineColor='black',
    lineWidth=1
)


def update_selector_geometry():
    p1 = pol_to_cart(INNER_R - 10, selector_angle_screen)
    p2 = pol_to_cart(OUTER_R + 10, selector_angle_screen)
    selector_line.start = p1
    selector_line.end = p2
    selector_knob.pos = p2


# =========================
# Preview / labels
# =========================

preview_circle = visual.Circle(
    win,
    radius=PREVIEW_RADIUS,
    pos=PREVIEW_POS,
    fillColor=(0, 0, 0),
    lineColor='white',
    lineWidth=2,
    colorSpace='rgb'
)

preview_label = visual.TextStim(
    win,
    text='Selected color',
    pos=(PREVIEW_POS[0], PREVIEW_POS[1] + 112),
    color='white',
    height=24
)

hue_text = visual.TextStim(
    win,
    text='',
    pos=(PREVIEW_POS[0], PREVIEW_POS[1] - 114),
    color='white',
    height=22
)

instruction_text = visual.TextStim(
    win,
    text='Click and drag the white marker around the ring. Release, then click Submit. Press Esc to quit.',
    pos=(0, 324),
    color='white',
    height=22,
    wrapWidth=980
)


selected_hue = None
selected_rgb = None


def update_selected_color():
    global selected_hue, selected_rgb
    selected_hue = (selector_angle_screen - ring_rotation) % 360
    selected_rgb = lch_to_psychopy_rgb(L_VAL, C_VAL, selected_hue)
    preview_circle.fillColor = selected_rgb
    preview_circle.lineColor = selected_rgb
    hue_text.text = f"L={L_VAL}, C={C_VAL}, h={selected_hue:.1f}°"


update_selector_geometry()
update_selected_color()


# =========================
# Submit button
# =========================

submit_rect = visual.Rect(
    win,
    width=SUBMIT_W,
    height=SUBMIT_H,
    pos=SUBMIT_POS,
    fillColor=(-0.35, -0.35, -0.35),
    lineColor='white',
    lineWidth=2,
    colorSpace='rgb'
)

submit_text = visual.TextStim(
    win,
    text='Submit',
    pos=SUBMIT_POS,
    color='white',
    height=28
)


# =========================
# Interaction helpers
# =========================

def mouse_on_ring(mouse_pos):
    x, y = mouse_pos
    d = distance(x, y)
    return abs(d - RING_RADIUS) <= RING_TOL


def draw_scene():
    for s in ring_sectors:
        s.draw()
    outer_outline.draw()
    inner_outline.draw()
    selector_line.draw()
    selector_knob.draw()
    preview_circle.draw()
    preview_label.draw()
    hue_text.draw()
    submit_rect.draw()
    submit_text.draw()
    instruction_text.draw()


# =========================
# Main loop
# =========================

dragging = False
submitted = False
prev_left = False

clock = core.Clock()

while not submitted:
    # Keyboard first
    keys = kb.getKeys(keyList=['escape'], waitRelease=False, clear=True)
    if keys:
        print("User cancelled with Escape.")
        win.close()
        core.quit()

    mouse_pos = mouse.getPos()
    left = mouse.getPressed(getTime=False)[0]

    # Mouse-down edge
    if left and not prev_left:
        if mouse_on_ring(mouse_pos):
            dragging = True
            selector_angle_screen = angle_from_xy(mouse_pos[0], mouse_pos[1])
            update_selector_geometry()
            update_selected_color()
        elif point_in_rect(mouse_pos, SUBMIT_POS, SUBMIT_W, SUBMIT_H):
            submitted = True

    # Drag update
    if dragging and left:
        selector_angle_screen = angle_from_xy(mouse_pos[0], mouse_pos[1])
        update_selector_geometry()
        update_selected_color()

    # Mouse-up edge
    if prev_left and not left:
        dragging = False

    prev_left = left

    draw_scene()
    win.flip()


# =========================
# Output
# =========================

print("Submitted color:")
print(f"screen angle = {selector_angle_screen:.2f} degrees")
print(f"ring rotation = {ring_rotation:.2f} degrees")
print(f"true hue = {selected_hue:.2f} degrees")
print(f"L = {L_VAL}")
print(f"C = {C_VAL}")
print(f"rgb (PsychoPy -1..1) = {selected_rgb}")

win.close()
core.quit()