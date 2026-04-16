### virtual environment: alien_features
### installed packages in the virtual environment:
# - Psychopy v2026.1.2
# - Python v3.10.12



### This is an update version based on 'color ring snippet.py'. Instead of fill in a shape on the screen, this version fill the color of imported alien images. 
### This filling effect is implemented by prepare two alien images. One image is the outline image, e.g., 'outline alien 130.png'. The outline image is the black drawings of the alien with black and white eyes, on transparent background.
### The fill layer is an fill image, e.g., 'fill alien 130.png'. The fill image is white covers all the areas of the alien including the outlines on a transparent background. 
### On the screen, the fill layer is under the outline layer. The window first draws the fill image, then converts it into a greyscale masks using numpy array, therefore the greyscale is essentially a texture.
### Greyscale is used to indicate to the Psychopy which pixels need to be tinted (colored) by selected hue. 

### In order to accurately display the color, the better practice is to do Monitor Calibration. It is difficult to dirrect do calibration with codes.
### The current practice is to use the Psychopy Monitor Center to create a moniter with specific parameters, then load in the Monitor with scripts before everything. 
### The related monitor loading parts are currently marked out.

### The color ring parameters are the same with 'color ring snippet.py'. It also adopts CIELCh. In this example, L = 65 (lightness), C = 40 (chroma), and H (hue) varies from 0 to 360 degrees in steps of 1 degree.
### The color ring is randomly rotated each time the script is run. 











# from psychopy import visual, event, core, monitors
# import numpy as np
# import math
# from PIL import Image

# # Load a monitor you created in Monitor Center
# mon = monitors.Monitor('LabDisplay1')
# mon.setCurrent(-1)  # use most recent stored calibration

# print("Loaded monitor:", mon.name)
# print("Width (cm):", mon.getWidth())
# print("Distance (cm):", mon.getDistance())
# print("Size (pix):", mon.getSizePix())
# print("Gamma grid:", mon.getGammaGrid())

# win = visual.Window(
#     size=(1200, 800),
#     monitor=mon,
#     units='pix',
#     color=(-0.72, -0.72, -0.72),
#     colorSpace='rgb',
#     allowGUI=True,
#     fullscr=False,
#     waitBlanking=True,
#     useRetina=True
# )


from psychopy import visual, event, core
import numpy as np
import math
from PIL import Image

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


def angle_from_xy(x, y, center=(0.0, 0.0)):
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


def distance_to_center(x, y, center):
    return math.hypot(x - center[0], y - center[1])


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

ALIEN_POS = (-250, 20)
ALIEN_MAX_W = 420
ALIEN_MAX_H = 520

SUBMIT_POS = (250, -235)
SUBMIT_W = 175
SUBMIT_H = 66

WHITE_THRESHOLD = 180
IMAGE_FLIP_VERT = False  # fix upside-down display

ring_rotation = np.random.uniform(0, 360)
selector_angle_screen = 0.0


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
    waitBlanking=True,
    useRetina=True
)

mouse = event.Mouse(win=win)
mouse.setVisible(True)


# =========================
# Load and prepare images
# =========================

fill_path = "fill alien 131.png"
outline_path = "outline alien 131.png"

fill_rgba = np.array(Image.open(fill_path).convert("RGBA"), dtype=np.uint8)
outline_rgba = np.array(Image.open(outline_path).convert("RGBA"), dtype=np.uint8)

h_img, w_img = fill_rgba.shape[:2]
scale = min(ALIEN_MAX_W / w_img, ALIEN_MAX_H / h_img)
alien_size = (w_img * scale, h_img * scale)

fill_rgb = fill_rgba[:, :, :3]
fill_alpha = fill_rgba[:, :, 3]

white_mask = (
    (fill_rgb[:, :, 0] >= WHITE_THRESHOLD) &
    (fill_rgb[:, :, 1] >= WHITE_THRESHOLD) &
    (fill_rgb[:, :, 2] >= WHITE_THRESHOLD) &
    (fill_alpha > 0)
)

# Create a grayscale intensity texture only in the recolorable region.
# Black outside the mask, grayscale inside.
gray_tex = np.zeros((h_img, w_img, 4), dtype=np.uint8)

# preserve shading from original bright pixels
shade = np.mean(fill_rgb.astype(np.float32), axis=2)
shade = np.clip(shade, 0, 255).astype(np.uint8)

gray_tex[:, :, 0] = np.where(white_mask, shade, 0)
gray_tex[:, :, 1] = np.where(white_mask, shade, 0)
gray_tex[:, :, 2] = np.where(white_mask, shade, 0)
gray_tex[:, :, 3] = np.where(white_mask, fill_alpha, 0)

gray_fill_image = Image.fromarray(gray_tex, mode="RGBA")
outline_image = Image.fromarray(outline_rgba, mode="RGBA")

print("Masked pixels:", np.count_nonzero(white_mask))


# =========================
# Precompute hue colors
# =========================

hue_rgb_psy = np.array(
    [lch_to_psychopy_rgb(L_VAL, C_VAL, hh) for hh in range(360)],
    dtype=np.float32
)

current_hue_idx = 0
selected_hue = 0.0
selected_rgb = hue_rgb_psy[0]


# =========================
# Stimuli
# =========================

fill_stim = visual.ImageStim(
    win=win,
    image=gray_fill_image,
    pos=ALIEN_POS,
    size=alien_size,
    units='pix',
    interpolate=True,
    flipVert=IMAGE_FLIP_VERT,
    color=selected_rgb,
    colorSpace='rgb'
)

outline_stim = visual.ImageStim(
    win=win,
    image=outline_image,
    pos=ALIEN_POS,
    size=alien_size,
    units='pix',
    interpolate=True,
    flipVert=IMAGE_FLIP_VERT
)

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
    win=win,
    radius=OUTER_R,
    pos=RING_CENTER,
    edges=256,
    lineColor=(0.2, 0.2, 0.2),
    lineWidth=1,
    fillColor=None,
    colorSpace='rgb'
)

inner_outline = visual.Circle(
    win=win,
    radius=INNER_R,
    pos=RING_CENTER,
    edges=256,
    lineColor=BG,
    lineWidth=2,
    fillColor=None,
    colorSpace='rgb'
)

selector_line = visual.Line(
    win,
    start=(0, 0),
    end=(0, 0),
    lineColor='white',
    lineWidth=4
)


hue_text = visual.TextStim(
    win,
    text='',
    pos=(RING_CENTER[0], RING_CENTER[1] + 255),
    color='white',
    height=22
)

instruction_text = visual.TextStim(
    win,
    text='Click and drag around the ring to recolor the alien. Release, then click Submit. Esc quits.',
    pos=(0, 350),
    color='white',
    height=22,
    wrapWidth=1100
)

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
# Update helpers
# =========================


def update_selector_geometry():
    eps = 1.0
    p1 = np.array(RING_CENTER) + pol_to_cart(INNER_R + eps, selector_angle_screen)
    p2 = np.array(RING_CENTER) + pol_to_cart(OUTER_R - eps, selector_angle_screen)
    selector_line.start = p1
    selector_line.end = p2
    


def update_selected_color_from_angle():
    global selected_hue, selected_rgb, current_hue_idx

    selected_hue = (selector_angle_screen - ring_rotation) % 360
    hue_idx = int(round(selected_hue)) % 360
    current_hue_idx = hue_idx
    selected_rgb = hue_rgb_psy[hue_idx]

    fill_stim.color = selected_rgb
    hue_text.text = f"L={L_VAL}, C={C_VAL}, h={selected_hue:.1f}°"


def mouse_on_ring(mouse_pos):
    d = distance_to_center(mouse_pos[0], mouse_pos[1], RING_CENTER)
    return INNER_R <= d <= OUTER_R


def draw_scene():
    fill_stim.draw()
    outline_stim.draw()

    for s in ring_sectors:
        s.draw()

    outer_outline.draw()
    inner_outline.draw()
    selector_line.draw()
    hue_text.draw()

    submit_rect.draw()
    submit_text.draw()
    instruction_text.draw()


# Initialize
update_selector_geometry()
update_selected_color_from_angle()

# =========================
# Main loop
# =========================

dragging = False
submitted = False
prev_left = False

while not submitted:
    if 'escape' in event.getKeys():
        win.close()
        core.quit()

    mouse_pos = mouse.getPos()
    left = mouse.getPressed(getTime=False)[0]

    if left and not prev_left:
        if point_in_rect(mouse_pos, SUBMIT_POS, SUBMIT_W, SUBMIT_H):
            submitted = True
        elif mouse_on_ring(mouse_pos):
            dragging = True
            selector_angle_screen = angle_from_xy(mouse_pos[0], mouse_pos[1], center=RING_CENTER)
            update_selector_geometry()
            update_selected_color_from_angle()

    if dragging and left:
        selector_angle_screen = angle_from_xy(mouse_pos[0], mouse_pos[1], center=RING_CENTER)
        update_selector_geometry()
        update_selected_color_from_angle()

    if prev_left and not left:
        dragging = False

    prev_left = left

    draw_scene()
    win.flip()

# =========================
# Output
# =========================

print("Submitted color:")
print(f"screen angle = {selector_angle_screen:.2f}")
print(f"ring rotation = {ring_rotation:.2f}")
print(f"true hue = {selected_hue:.2f}")
print(f"L={L_VAL}, C={C_VAL}")
print(f"rgb (PsychoPy -1..1) = {selected_rgb}")

win.close()
core.quit()