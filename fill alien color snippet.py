from psychopy import visual, event, core
from psychopy.hardware import keyboard
import numpy as np
import random
import math
from PIL import Image

# =========================
# Color conversion helpers
# =========================

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
    return rgb * 2.0 - 1.0  # PsychoPy rgb (-1..1)


# =========================
# Geometry helpers
# =========================

def pol_to_cart(r, ang_deg):
    a = np.deg2rad(ang_deg)
    return np.array([r * np.cos(a), r * np.sin(a)], dtype=float)


def angle_from_xy(x, y, center=(0, 0)):
    dx = x - center[0]
    dy = y - center[1]
    ang = np.degrees(np.arctan2(dy, dx))
    if ang < 0:
        ang += 360
    return ang


def point_in_rect(pt, center, w, h):
    x, y = pt
    cx, cy = center
    return (cx - w / 2 <= x <= cx + w / 2) and (cy - h / 2 <= y <= cy + h / 2)


def distance_xy(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


# =========================
# Parameters
# =========================

L_VAL = 65
C_VAL = 40

WIN_SIZE = (1200, 800)
BG = (-0.72, -0.72, -0.72)

RING_CENTER = (250, 20)
RING_RADIUS = 190
RING_WIDTH = 34
INNER_R = RING_RADIUS - RING_WIDTH / 2
OUTER_R = RING_RADIUS + RING_WIDTH / 2
N_SEGMENTS = 360

# More forgiving grab margin so dragging doesn't feel "narrow"
RING_GRAB_MARGIN = 36

ALIEN_POS = (-250, 20)
ALIEN_MAX_W = 420
ALIEN_MAX_H = 520

SUBMIT_POS = (250, -235)
SUBMIT_W = 175
SUBMIT_H = 66

WHITE_THRESHOLD = 235

# IMPORTANT: set False if your images become upside-down
IMAGE_FLIP_VERT = True

ring_rotation = random.uniform(0, 360)
selector_angle_screen = random.uniform(0, 360)

# =========================
# Window/input
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
kb = keyboard.Keyboard()

startup_text = visual.TextStim(win, text='Loading...', color='white', height=28)
startup_text.draw()
win.flip()

# =========================
# Load images
# =========================

fill_path = "fill alien 131.png"
outline_path = "outline alien 131.png"

fill_rgba_base = np.array(Image.open(fill_path).convert("RGBA"), dtype=np.uint8)
outline_rgba = np.array(Image.open(outline_path).convert("RGBA"), dtype=np.uint8)

h, w = fill_rgba_base.shape[:2]
scale = min(ALIEN_MAX_W / w, ALIEN_MAX_H / h)
alien_size = (w * scale, h * scale)

fill_rgb = fill_rgba_base[:, :, :3]
fill_alpha = fill_rgba_base[:, :, 3]

white_mask = (
    (fill_rgb[:, :, 0] >= WHITE_THRESHOLD) &
    (fill_rgb[:, :, 1] >= WHITE_THRESHOLD) &
    (fill_rgb[:, :, 2] >= WHITE_THRESHOLD) &
    (fill_alpha > 0)
)

white_shade = np.mean(fill_rgb.astype(np.float32) / 255.0, axis=2)
white_shade = np.clip(white_shade, 0.0, 1.0)

# Optional speed-up: pre-index white pixels
mask_rows, mask_cols = np.where(white_mask)
mask_shade = white_shade[mask_rows, mask_cols]

# Precompute 360 hue RGB triplets
hue_rgb_psy = np.array([lch_to_psychopy_rgb(L_VAL, C_VAL, h) for h in range(360)], dtype=np.float32)
hue_rgb_srgb01 = np.clip((hue_rgb_psy + 1.0) / 2.0, 0.0, 1.0)


def make_tinted_fill_image(hue_idx):
    srgb01 = hue_rgb_srgb01[hue_idx]
    out = fill_rgba_base.copy()
    out_rgb = out[:, :, :3]

    # tint only masked pixels
    tint = (mask_shade[:, None] * srgb01[None, :]) * 255.0
    out_rgb[mask_rows, mask_cols] = np.clip(tint, 0, 255).astype(np.uint8)

    out[:, :, :3] = out_rgb
    return out


# Initial tinted image
initial_hue_idx = int(round((selector_angle_screen - ring_rotation) % 360)) % 360
fill_stim = visual.ImageStim(
    win=win,
    image=make_tinted_fill_image(initial_hue_idx),
    pos=ALIEN_POS,
    size=alien_size,
    units='pix',
    interpolate=True,
    flipVert=IMAGE_FLIP_VERT
)

outline_stim = visual.ImageStim(
    win=win,
    image=outline_rgba,
    pos=ALIEN_POS,
    size=alien_size,
    units='pix',
    interpolate=True,
    flipVert=IMAGE_FLIP_VERT
)

# =========================
# Hue ring
# =========================

ring_sectors = []
for i in range(N_SEGMENTS):
    col = hue_rgb_psy[i]
    a1 = i + ring_rotation
    a2 = i + 1 + ring_rotation

    p1o = np.array(RING_CENTER) + pol_to_cart(OUTER_R, a1)
    p2o = np.array(RING_CENTER) + pol_to_cart(OUTER_R, a2)
    p2i = np.array(RING_CENTER) + pol_to_cart(INNER_R, a2)
    p1i = np.array(RING_CENTER) + pol_to_cart(INNER_R, a1)

    ring_sectors.append(
        visual.ShapeStim(
            win=win,
            vertices=np.array([p1o, p2o, p2i, p1i]),
            fillColor=col,
            lineColor=col,
            lineWidth=0,
            colorSpace='rgb',
            closeShape=True,
            interpolate=True
        )
    )

outer_outline = visual.Circle(
    win, radius=OUTER_R, pos=RING_CENTER, edges=256,
    lineColor=(0.2, 0.2, 0.2), lineWidth=1, fillColor=None, colorSpace='rgb'
)

inner_outline = visual.Circle(
    win, radius=INNER_R, pos=RING_CENTER, edges=256,
    lineColor=BG, lineWidth=2, fillColor=None, colorSpace='rgb'
)

selector_line = visual.Line(win, start=(0, 0), end=(0, 0), lineColor='white', lineWidth=6)
selector_knob = visual.Circle(win, radius=11, pos=(0, 0), fillColor='white', lineColor='black', lineWidth=1.5)

hue_text = visual.TextStim(win, text='', pos=(RING_CENTER[0], RING_CENTER[1] + 255), color='white', height=22)
instruction_text = visual.TextStim(
    win,
    text='Click/drag on ring (or knob). Hue updates continuously while dragging. Esc to quit.',
    pos=(0, 350),
    color='white',
    height=22,
    wrapWidth=1100
)

submit_rect = visual.Rect(
    win, width=SUBMIT_W, height=SUBMIT_H, pos=SUBMIT_POS,
    fillColor=(-0.35, -0.35, -0.35), lineColor='white', lineWidth=2, colorSpace='rgb'
)
submit_text = visual.TextStim(win, text='Submit', pos=SUBMIT_POS, color='white', height=28)

selected_hue = None
selected_rgb = None
current_hue_idx = None


def update_selector_geometry():
    p1 = np.array(RING_CENTER) + pol_to_cart(INNER_R - 10, selector_angle_screen)
    p2 = np.array(RING_CENTER) + pol_to_cart(OUTER_R + 10, selector_angle_screen)
    selector_line.start = p1
    selector_line.end = p2
    selector_knob.pos = p2


def update_selected_color():
    global selected_hue, selected_rgb, current_hue_idx
    selected_hue = (selector_angle_screen - ring_rotation) % 360
    hue_idx = int(round(selected_hue)) % 360
    selected_rgb = hue_rgb_psy[hue_idx]
    hue_text.text = f"L={L_VAL}, C={C_VAL}, h={selected_hue:.1f}°"

    if hue_idx != current_hue_idx:
        # Use setImage() for reliable real-time texture updates
        fill_stim.setImage(make_tinted_fill_image(hue_idx))
        current_hue_idx = hue_idx


def mouse_near_ring(mouse_pos):
    dx = mouse_pos[0] - RING_CENTER[0]
    dy = mouse_pos[1] - RING_CENTER[1]
    d = math.hypot(dx, dy)
    return (INNER_R - RING_GRAB_MARGIN) <= d <= (OUTER_R + RING_GRAB_MARGIN)


def mouse_on_knob(mouse_pos):
    kx, ky = selector_knob.pos
    return distance_xy(mouse_pos[0], mouse_pos[1], kx, ky) <= (selector_knob.radius + 10)


def draw_scene():
    fill_stim.draw()
    outline_stim.draw()
    for s in ring_sectors:
        s.draw()
    outer_outline.draw()
    inner_outline.draw()
    selector_line.draw()
    selector_knob.draw()
    hue_text.draw()
    submit_rect.draw()
    submit_text.draw()
    instruction_text.draw()


update_selector_geometry()
update_selected_color()

# =========================
# Main loop
# =========================

dragging = False
submitted = False
prev_left = False

while not submitted:
    if kb.getKeys(keyList=['escape'], waitRelease=False, clear=True):
        print("User cancelled with Escape.")
        win.close()
        core.quit()

    mouse_pos = mouse.getPos()
    left = mouse.getPressed(getTime=False)[0]

    # mouse press (edge)
    if left and not prev_left:
        if point_in_rect(mouse_pos, SUBMIT_POS, SUBMIT_W, SUBMIT_H):
            submitted = True
        elif mouse_near_ring(mouse_pos) or mouse_on_knob(mouse_pos):
            dragging = True
            selector_angle_screen = angle_from_xy(mouse_pos[0], mouse_pos[1], center=RING_CENTER)
            update_selector_geometry()
            update_selected_color()

    # while held: keep updating continuously
    if dragging and left:
        selector_angle_screen = angle_from_xy(mouse_pos[0], mouse_pos[1], center=RING_CENTER)
        update_selector_geometry()
        update_selected_color()

    # release
    if prev_left and not left:
        dragging = False

    prev_left = left

    draw_scene()
    win.flip()

print("Submitted color:")
print(f"screen angle = {selector_angle_screen:.2f}")
print(f"ring rotation = {ring_rotation:.2f}")
print(f"true hue = {selected_hue:.2f}")
print(f"L={L_VAL}, C={C_VAL}")
print(f"rgb (PsychoPy -1..1) = {selected_rgb}")

win.close()
core.quit()