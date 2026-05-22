

# ============Import Modules===========

from psychopy import monitors, visual, event, core, gui
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np
import random
import math
from PIL import Image
from datetime import datetime


# ============ Define parameters ===========
MONITOR = 'AlienMemoryMonitor' ## set the name of the monitor that you created with create_monitor.py
WIN_SIZE = ()
WIN_BG = () ## set window background color in rbg format, e.g. (0, 0, 0) for black, (1, 1, 1) for white, (-1, -1, -1) for black in rgb space
RETINA = True ## set to True if using a retina display, False otherwise. This will ensure that the stimuli are displayed at the correct size on retina displays, which have a higher pixel density.
Allow_escape = True

##### Learning Trial Parameters #####
ENCODING_TIME = 4.0 ## time to show each alien during encoding phase in seconds
BLANK_TIME = 0.5 ## time to show blank screen between encoding and practice phases in seconds
INTERVAL_TIME = 1.0 ## time to show blank screen between learning trials in seconds
FIXATION_TIME = np.random.uniform(0.75,1.25) ## time to show fixation cross before each trial in seconds (randomized between 0.75 and 1.25 second)  
BLOCK_BREAK_TIME = 90.0 ## time to show break screen between learning blocks in seconds 

###### Stimulus Parameters ######
ALIEN_SIZE = 100 ## size of the alien images in pixels
MASK_THRESHOLD = 180 

ALIEN_M_NAMES = ["Ethan Miller", "Liam Chen", "Noah Johnson", "Mason Clark", "Aiden Brown", "Lucas Walker", "Owen Bennett", "Leo Collins", "Caleb Turner", "Wyatt Reed", "Nathan Scott", "Ezra Hayes","Hudson Gray", "Carter Price", "Grayson Cole", "Asher Bell", "Dylan Foster", "Ryan Cooper", "Roman Blake", "Hunter Murphy", "Aaron Cox", "Thomas Watson", "Theo Lawson", "Gavin Pierce", "Jason Howard"]
ALIEN_F_NAMES = ["Sophie Lee", "Olivia Smith", "Emma Davis", "Ava Mitchell", "Mia Taylor", "Harper Bailey", "Ella Ward", "Grace Hughes", "Zoe Ramirez","Lily Adams", "Riley Griffin","Avery Diaz","Paige Larson", "Kayla Dunn", "Katie Ross", "Jessica Lane", "Lauren Hill", "Rachel Wood", "Nicole Grant", "Leah Perry", "Amber Cruz", "Allison Ford", "Julia Banks", "Chloe Dean", "Hailey Moore"]
ALIEN_PLANETS = ["Phyethia", "Teraris", "Luxidon", "Arcteron"]  

ALIEN_PATH_LEARNING = ""
ALIEN_PATH_PRACTICE = ""
ALIEN_PATH_GEN = ""
IMAGE_FLIP_VERT = False ## set to True if the alien images need to be flipped vertically, False otherwise. This will ensure that the images are displayed correctly based on how they were created. 

##### Encoding Screen Parameters #####
ALIEN_POS = (0,0)

ALIEN_NAME_POS = ()
ALIEN_NAME_COLOR = ()
ALIEN_NAME_HEIGHT = 15

ALIEN_PLANET_POS = ()
ALIEN_PLANET_COLOR = ()
ALIEN_PLANET_HEIGHT = 30

##### Testing Screen Parameters ######
TEST_ALIEN_POS = (- WIN_SIZE[0] * 1/3, 0)
TEST_ALIEN_NAME_POS = (TEST_ALIEN_POS[0], ALIEN_NAME_POS[1])
TEST_QUESTION_HEIGHT = ALIEN_PLANET_HEIGHT
TEST_QUESTION_COLOR = ()
TEST_QUESTION_POS = (WIN_SIZE[0] * 1/3, WIN_SIZE[1] * 1/3)
PLANET_TEST_QUESTION = 'Where does this alien come from ?'
COLOR_TEST_QUESTION = "What is the color of this alien ?"
RESIDENCE_TEST_QUESTION = "What is the residence place of this alien ?"
TEST_LABEL_SIZE = ()
TEST_LABEL_DISTANCE_TO_CENTER = ()
TEST_LABEL_CENTER = (WIN_SIZE[0] * 1/3, 0)
TEST_SUBMIT_POS = (WIN_SIZE[0] * 1/3, - WIN_SIZE[1]*3/4)
TEST_SUBMIT_SIZE = ()




###### Color Parameters ######
# D65 reference white
XN = 95.047
YN = 100.000
ZN = 108.883
# This script adopts CIE LCh color space, which is a cylindrical representation of the CIE Lab color space. In CIE LCh, colors are represented by three parameters: L* (lightness), C* (chroma), and h° (hue angle). The hue angle is measured in degrees, with 0° corresponding to red, 90° to yellow, 180° to green, and 270° to blue. The chroma represents the intensity of the color, with higher values indicating more saturated colors. The lightness represents the brightness of the color, with higher values indicating lighter colors.
# For this experiment, L and C values will be held constant while h° values will be manipulated to create different colors for the aliens. The specific L and C values, as well as the range of h° values, can be adjusted based on the desired color palette for the experiment.
L_VALUE = 65 ## set the lightness value for the alien colors. This will control how light or dark the colors of the aliens are. Higher values will result in lighter colors, while lower values will result in darker colors. The range of L values is typically from 0 to 100, where 0 is black and 100 is white.
C_VALUE = 40 ## set the chroma value for the alien colors. This will control how saturated the colors of the aliens are. Higher values will result in more saturated colors, while lower values will result in more desaturated colors. The range of C values can vary depending on the specific color space being used, but typically it can range from 0 to around 100 or more, where 0 is completely desaturated (gray) and higher values indicate more saturation.


###### Ring Parameters ######
# Color ring parameters
TEST_RING_CENTER = (WIN_SIZE[0] * 1/3, 0)
COLOR_RING_RADIUS = 10
COLOR_RING_WIDTH = 2
COLOR_RING_ROTATION = True
COLOR_RING_SEGMENTS = 360
SELECTOR_COLOR = 'white'



# Residence ring parameters
RESIDENCE_RING_THIKNESS = 2
RESIDENCE_RING_RADIUS = 10
RESIDENCE_RING_COLOR = ()
RESIDENCE_RING_EDGES = 512 #the number of line segments to draw the circle shape, higher number means smoother and rounder
RESIDENCE_BAR_WIDTH = 0.003
RESIDENCE_BAR_LENGTH = 0.1
RESIDENCE_BAR_COLOR = 'black'


###### MODE PARAMETERS ######
MODES = ['debug','demo','experiment']
EXPERIMENT_GROUPS = ['1','2'] ## set the group assignemnt for the experiment. '1' for immediate test group, '2' for 1-day-delay test group. 


###### Structure Parameters ######

### number of trials and blocks for each phase of the experiment. These parameters will be used to create the structure of the experiment.
NUM_PRACTICE_PER_GROUP = 1
NUM_PRACTICE_GROUPS = 4

NUM_LEARNING_BLOCKS = 3
NUM_LEARNING_GROUPS = 4
NUM_ALIEN_LEARNING_PER_GROUP = 8 ## number of aliens assigned to each group during learning phase.

NUM_TESTING_PER_GROUP = 8
NUM_TESTING_GROUP = 4

NUM_GENERALIZATION_PER_GROUP = 2
NUM_GENERALIZATION_GROUPS = 4

### Phase settings for the working memory test, the memory test, generalization test, color reconstruction. 
WK_MEMORY_STATUS = True ## set to True if working memory is included in the experiment, False otherwise.
TESTING_FIRST = True ## Set to true of memory test comes before generalization test, false if generalization test comes first.
COLOR_RECONSTRUCTION = True ## set to True if color reconstruction task is included in the experiment, False otherwise.


# ============ Define Key functions ===========

### record function
def record_response(
    data,
    trial_no,
    block,
    trial_start_time,
    correct_response,
    stimuli,
    response,
    rt,
    trial_end_time,
    initial_hue = None,
    ring_rotation = None,
    initial_residence = None,
):
    trial_data = {
        "participant_id": exp_info["participant_id"],
        "group": exp_info["group"],
        "condition": exp_info["condition"],
        "session": exp_info["session"],
        "trial_no": trial_no,
        "block": block,
        "trial_start_time": trial_start_time,
        "correct_response": correct_response,
        "stimuli": stimuli,
        "response": response,
        "rt": rt,
        "initial_hue": initial_hue,
        "ring_rotation": ring_rotation,
        "initial_residence":initial_residence,
        "trial_end_time": trial_end_time
    }

    data.append(trial_data)

### Learning trial function. This function will control the flow of each learning trial (and practice learning trials), including the encoding phase, the practice phase, and the inter-trial interval. It will also handle the presentation of stimuli and the collection of responses during the practice phase. The parameters defined above will be used to control the timing and structure of the learning trials.
#def learning_trial(alien_image, alien_name, alien_planet, practice=False):
    ## 1. Fixation Cross

def display_fixation_cross(stage, duration):
    fixation = win.TextStim(
        win = win,
        text = "+",
        color = (1, 1, 1),
        colorSpace = 'rgb',
        height = 50
    )
    fixation_N_Frames = time_to_frame(duration) ## show fixation cross for a random duration between 0.75 and 1.25 seconds that are converted to the number of frames for timing accuracy.
    for fixation_Frame in (fixation_N_Frames):
        fixation.draw()
        win.flip()
    stage += 1
    return stage 

def alien_fill_image(alien_image, alien_pos=(0,0)):
    fill_stim = visual.ImageStim(
        win=win,
        image=fill_image(ALIEN_PATH_LEARNING, alien_image),
        pos=alien_pos,
        size=ALIEN_SIZE,
        units='pix',
        interpolate=True,
        flipVert=IMAGE_FLIP_VERT,
        colorSpace='rgb'
    )
    return fill_stim
def alien_outline_image(alien_image, alien_pos=(0,0)):
    outline_stim = visual.ImageStim(
        win=win,
        image=outline_image(ALIEN_PATH_LEARNING + alien_image),
        pos=alien_pos,
        size=ALIEN_SIZE,
        units='pix',
        interpolate=True,
        flipVert=IMAGE_FLIP_VERT      
   )
    return outline_stim

    
def update_alien_fill_color(alien_stim, new_color):
    alien_stim.color = new_color
    alien_stim.draw()
    win.flip()


def alien_text(text_content, text_pos, height, textColor):
    alien_text_stim = visual.TextStim(
        win = win,
        text = text_content,
        color = textColor,
        colorSpace = 'rgb',
        height = height,
        pos = text_pos
    )
    return alien_text_stim

def residence_circle(position):
    residence_circle_stim = visual.Circle(
        win = win,
        radius = RESIDENCE_RING_RADIUS,
        edges = RESIDENCE_RING_EDGES,
        lineColor = RESIDENCE_RING_COLOR,
        fillColor = None,
        lineWidth = RESIDENCE_RING_THIKNESS,
        interpolate = True,
        pos = position
    )
    return residence_circle_stim
    
def residence_circle_bar():
    residence_bar_stim = visual.Rect(
        win = win,
        width = RESIDENCE_BAR_WIDTH,
        height = RESIDENCE_BAR_LENGTH,
        fillColor = RESIDENCE_BAR_COLOR,
        lineColor = RESIDENCE_BAR_COLOR,
        interpolate = True)
    return residence_bar_stim


def encoding_screen_draw(alien_image, alien_color, alien_pos, alien_fake_name, alien_planet, residence_ring_pos ,residence_angle): 
    fill_stim = alien_fill_image(alien_image, alien_pos)
    update_alien_fill_color(fill_stim, alien_color)
    fill_stim.draw()
    outline_stim = alien_outline_image(alien_image, alien_pos)
    outline_stim.draw()
    fake_name_stim = alien_text(alien_fake_name, ALIEN_NAME_POS, ALIEN_NAME_HEIGHT, ALIEN_NAME_COLOR)
    fake_name_stim.draw()
    planet_text_stim = alien_text(alien_planet, ALIEN_PLANET_POS, ALIEN_PLANET_HEIGHT, ALIEN_PLANET_COLOR )
    planet_text_stim.draw()
    residence_circle_stim = residence_circle(residence_ring_pos)
    residence_circle_stim.draw()
    residence_bar_stim = residence_circle_bar()
    update_residence_bar(residence_circle_stim, residence_angle)
    residence_bar_stim.draw()

def encoding_screen_present(stage, alien_image, alien_color, alien_pos, alien_fake_name, alien_planet, duration, residence_ring_pos,residence_angle): 
    nFrames = time_to_frame(duration)
    for frame in range(nFrames):
        encoding_screen_draw(alien_image, alien_color, alien_pos, alien_fake_name, alien_planet, residence_ring_pos ,residence_angle)
        win.flip()

    stage+=1
    return stage
    
def blank_screen_present(stage,duration):
    emptyText = visual.TextStim(
        win = win,
        text = "")
    nFrames = time_to_frame(duration)
    for frame in range(nFrames):
        emptyText.draw()
        win.flip()
    stage += 1
    return stage

def test_labels():
    labels_pos = calculate_pos()
    labels_stim = []
    labels_list = ALIEN_PLANETS
    np.random.shuffle(labels_list)
    for i in range(4):
        pos = labels_pos[i]
        test_label = visual.TextStim(
        text = labels_list[i],
        pos = pos
        )
        labels_stim.append((test_label,pos))
    return labels_stim

def draw_test_labels(stim): #draw all four labels
    labels_stim = stim
    for i in range(4):
        label = labels_stim[i][0]
        label.draw()


def detect_label_selection(mouse, labels_stim):   
    pt = mouse.getPos()
    click_label = False 
    selected_label = next((label for label in labels_stim if point_in_rect(pt,label[1], TEST_LABEL_SIZE[0], TEST_LABEL_SIZE[1])), None)
    if selected_label is not None:
        click_label = True
    return click_label, selected_label

def planet_test_screen():
    test_alien = alien_outline_image(alien, TEST_ALIEN_POS)
    test_alien.draw()
    test_alien_name = visual.TextStim(
        text = alien_name,
        pos = TEST_ALIEN_NAME_POS,
        height = ALIEN_NAME_HEIGHT,
        color = ALIEN_NAME_COLOR
    )
    test_alien_name.draw()
    test_question = PLANET_TEST_QUESTION
    test_question_stim = visual.TextStim(
        text = test_question,
        pos = TEST_QUESTION_POS,
        color = TEST_QUESTION_COLOR,
        height = TEST_QUESTION_HEIGHT
    )
    test_question_stim.draw()
    labels_stim = test_labels()
    draw_test_labels(labels_stim)
    rt_clock = core.Clock()
    win.callOnFlip(rt_clock.reset)
    win.flip()
    trial_start_time = now_time()
    while True:
        test_alien.draw()
        test_alien_name.draw()
        test_question_stim.draw()
        draw_test_labels(labels_stim)
        win.flip()
        mouse.clickRest()
        if mouse.getPress()[0]:
            click_label, selected_label = detect_label_selection(mouse, labels_stim)
            if click_label:
                selected_planet = selected_label.text
                rt = rt_clock.getTime()
                trial_end_time = now_time()
                record_response(data, trial_no, block, trial_start_time, planet, alien, selected_planet,rt, trial_end_time)
                break
    if practice:
        practice_no = +1
        return practice_no
    
def create_color_ring(
    win,
    ring_center,
    ring_radius,
    ring_width,
    hue_rgb_psy,
    ring_rotation=0.0,
    n_segments=360
):
    """
    Create a circular color ring made from 1-degree wedge segments.

    Returns
    -------
    ring_sectors : list
        List of PsychoPy ShapeStim ring segments.

    inner_r : float
        Inner radius of the ring.

    outer_r : float
        Outer radius of the ring.
    """

    inner_r = ring_radius - ring_width / 2
    outer_r = ring_radius + ring_width / 2

    ring_sectors = []

    for i in range(n_segments):
        col = hue_rgb_psy[i % len(hue_rgb_psy)]

        a1 = i + ring_rotation
        a2 = i + 1 + ring_rotation

        p1o = np.array(ring_center) + pol_to_cart(outer_r, a1)
        p2o = np.array(ring_center) + pol_to_cart(outer_r, a2)
        p2i = np.array(ring_center) + pol_to_cart(inner_r, a2)
        p1i = np.array(ring_center) + pol_to_cart(inner_r, a1)

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

    return ring_sectors, inner_r, outer_r

def draw_color_ring(ring_sectors):
    for sector in ring_sectors:
        sector.draw()

def color_test_screen():
    fill_stim = alien_fill_image(alien, TEST_ALIEN_POS)
    outline_stim = alien_outline_image(alien, TEST_ALIEN_POS)
    test_alien_name = visual.TextStim(
        text = alien_name,
        pos = TEST_ALIEN_NAME_POS,
        height = ALIEN_NAME_HEIGHT,
        color = ALIEN_NAME_COLOR
    )

    test_question = COLOR_TEST_QUESTION
    test_question_stim = visual.TextStim(
        text = test_question,
        pos = TEST_QUESTION_POS,
        color = TEST_QUESTION_COLOR,
        height = TEST_QUESTION_HEIGHT
    )
    

    initial_hue = int(round(np.random.uniform(0, 360)))
    ring_rotation = int(round(np.random.uniform(0, 360)))

    initial_angle = (initial_hue + ring_rotation) % 360
    hue_rgb_psy = create_hue_rgb_psy()
    ring_sectors, INNER_R, OUTER_R = create_color_ring(win, TEST_RING_CENTER, COLOR_RING_RADIUS, COLOR_RING_WIDTH, hue_rgb_psy, ring_rotation, COLOR_RING_SEGMENTS)
    outer_outline = visual.Circle(
        win = win,
        radius = OUTER_R,
        pos = TEST_RING_CENTER,
        edges = 256,
        lineColor = (0.2, 0.2, 0.2),
        lineWidth = 1,
        fillColor = None,
        colorSpace = 'rgb'
    )
    inner_outline = visual.Circle(
        win = win,
        radius = INNER_R,
        pos = TEST_RING_CENTER,
        edges = 256,
        lineColor = WIN_BG,
        lineWidth = 2,
        fillColor = None,
        colorSpace = 'rgb'
    )
    selector_line = visual.Line(
        win = win,
        start = (0, 0),
        end = (0, 0),
        lineColor = SELECTOR_COLOR,
        lineWidth =4
    )
    submit_rect = visual.Rect(
        win = win,
        width = TEST_SUBMIT_SIZE[0],
        height = TEST_SUBMIT_SIZE[1],
        pos = TEST_SUBMIT_POS,
        fillColor = (-0.35, -0.35, -0.35),
        lineColor = 'white',
        lineWidth = 2,
        colorSpace = 'rgb'
    )
    submit_text = visual.TextStim(
        win = win,
        text = 'Confirm',
        pos = TEST_SUBMIT_POS,
        color = 'white',
        height = 28
    )
    p1, p2 = update_color_selector_geometry(TEST_RING_CENTER, INNER_R, OUTER_R, initial_angle)
    selector_line.start = p1
    selector_line.end = p2
    initial_rgb = hue_rgb_psy[initial_hue]
    update_alien_fill_color(fill_stim, initial_rgb)
    fill_stim.draw()
    outline_stim.draw()
    test_alien_name.draw()
    test_question_stim.draw()
    draw_color_ring(ring_sectors)
    outer_outline.draw()
    inner_outline.draw()
    selector_line.draw()
    submit_rect.draw()
    submit_text.draw()
    rt_clock = core.Clock()
    win.callOnFlip(rt_clock.reset)
    win.flip()
    trial_start_time = now_time()
    submitted = False
    current_hue_idx = initial_hue
    dragging = False
    prev_left = False

    while not submitted:
        mouse.clickRest()
        pt = mouse.getPos()
        left = mouse.getPressed()[0]

        new_press = left and not prev_left
        new_release = prev_left and not left

        if new_press:
            if point_in_rect(pt, TEST_SUBMIT_POS, TEST_SUBMIT_SIZE[0], TEST_SUBMIT_SIZE[1]):
                submitted = True
            elif mouse_on_ring(pt): 
                dragging = True
                angle = angle_from_xy(pt[0],pt[1], TEST_RING_CENTER)
                p1, p2 = update_color_selector_geometry(TEST_RING_CENTER, INNER_R, OUTER_R, angle)
                selector_line.start = p1
                selector_line.end = p2
                current_hue_idx, selected_rgb = update_selected_color_from_angle(angle,hue_rgb_psy,ring_rotation)
                update_alien_fill_color(fill_stim, selected_rgb)      
        if dragging and left:
            angle = angle_from_xy(pt[0],pt[1], TEST_RING_CENTER)
            p1, p2 = update_color_selector_geometry(TEST_RING_CENTER, INNER_R, OUTER_R, angle)
            selector_line.start = p1
            selector_line.end = p2
            current_hue_idx, selected_rgb = update_selected_color_from_angle(angle,hue_rgb_psy,ring_rotation)
            update_alien_fill_color(fill_stim, selected_rgb)      
        if dragging and new_release:
            dragging = False

        prev_left = left
        fill_stim.draw()
        outline_stim.draw()
        test_alien_name.draw()
        test_question_stim.draw()
        draw_color_ring(ring_sectors)
        outer_outline.draw()
        inner_outline.draw()
        selector_line.draw()
        submit_rect.draw()
        submit_text.draw()
        win.flip()
    selected_hue = current_hue_idx
    rt = rt_clock.getTime()
    trial_end_time = now_time()
    record_response(data, trial_no, block, trial_start_time, color, alien, selected_hue, rt, trial_end_time, initial_hue, ring_rotation)
    if practice:
        practice_no += 1

def residence_test_screen():
    outline_stim = alien_outline_image(alien, TEST_ALIEN_POS)
    test_alien_name = visual.TextStim(
        text = alien_name,
        pos = TEST_ALIEN_NAME_POS,
        height = ALIEN_NAME_HEIGHT,
        color = ALIEN_NAME_COLOR
    )

    test_question = RESIDENCE_TEST_QUESTION
    test_question_stim = visual.TextStim(
        text = test_question,
        pos = TEST_QUESTION_POS,
        color = TEST_QUESTION_COLOR,
        height = TEST_QUESTION_HEIGHT
    )

    residence_ring_stim = residence_circle(TEST_RING_CENTER)
    initial_bar_angle = int(round(np.random.uniform(0, 360)))
    residence_bar_stim = residence_circle_bar()
    update_residence_bar(residence_bar_stim,initial_bar_angle)

    submit_rect = visual.Rect(
        win = win,
        width = TEST_SUBMIT_SIZE[0],
        height = TEST_SUBMIT_SIZE[1],
        pos = TEST_SUBMIT_POS,
        fillColor = (-0.35, -0.35, -0.35),
        lineColor = 'white',
        lineWidth = 2,
        colorSpace = 'rgb'
    )
    submit_text = visual.TextStim(
        win = win,
        text = 'Confirm',
        pos = TEST_SUBMIT_POS,
        color = 'white',
        height = 28
    )

    outline_stim.draw()
    test_alien_name.draw()
    test_question_stim.draw()
    residence_ring_stim.draw()
    residence_bar_stim.draw()
    submit_rect.draw()
    submit_text.draw()

    rt_clock = core.Clock()
    win.callOnFlip(rt_clock.reset)
    win.flip()
    trial_start_time = now_time()
    submitted = False
    current_angle = initial_bar_angle
    dragging = False
    prev_left = False

    while not submitted:
        mouse.clickRest()
        pt = mouse.getPos()
        mx = pt[0]
        my = pt[1]

        bar_pos = residence_bar_stim.pos
        bx = bar_pos[0]
        by = bar_pos[1]

        mouse_on_submit = False
        distance_to_bar = math.sqrt((mx - bx) ** 2 + (my - by) ** 2)
        grab_radius = max(RESIDENCE_BAR_LENGTH  * 0.8, 0.03)
        left = mouse.getPressed()[0]

        new_press = left and not prev_left
        new_release = prev_left and not left

        if new_press:
            if point_in_rect(pt, TEST_SUBMIT_POS, TEST_SUBMIT_SIZE[0], TEST_SUBMIT_SIZE[1]):
                submitted = True
                mouse_on_submit = True
            elif distance_to_bar <= grab_radius and not mouse_on_submit:
                dragging = True
        if dragging and left:
            current_angle = angle_from_xy(mx, my, TEST_RING_CENTER)
            update_residence_bar(residence_bar_stim, current_angle)
        if dragging and new_release:
            dragging = False

        prev_left = left
        outline_stim.draw()
        test_alien_name.draw()
        test_question_stim.draw()
        residence_ring_stim.draw()
        residence_bar_stim.draw()
        submit_rect.draw()
        submit_text.draw()
        win.flip()

    selected_residence = current_angle
    rt = rt_clock.getTime()
    trial_end_time = now_time()
    record_response(data, trial_no, block, trial_start_time, residence_angle, alien, selected_residence, rt, trial_end_time, initial_residence = initial_bar_angle)
    if practice:
        practice_no += 1
        
def practice_flow(data, trial_no, block, alien, alien_name, color, planet, residence, interval):
    practice_no = 0
    for i in range(len(practice_order)):
        practice_order[]
    

    


########### Helper function #############

### Timestamp functions
def now_date():
    return datetime.now().strftime("%Y-%m-%d")

def now_time():
    return datetime.now().strftime("%H:%M:%S")

def now_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

### Color conversion functions. These functions will be used to convert between different color spaces (e.g., from CIE LCh to RGB) and to manipulate the colors of the alien stimuli based on the defined L, C, and h° values. These functions are important for ensuring that the colors of the aliens are displayed correctly on the monitor and that they match the intended color palette for the experiment.

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

    # PsychoPy rgb range is -1 to 1
    return rgb * 2.0 - 1.0

### Geometry functions. These functions handle the conversion between polar, cartesian. 
##These two functions handle the conversion between polar, cartesian. 

def pol_to_cart(r, ang_deg):
    """
    Convert polar coordinates to Cartesian coordinates.

    Angle is in degrees.
    0 degrees points right.
    90 degrees points up.
    """
    a = np.deg2rad(ang_deg)
    return np.array([r * np.cos(a), r * np.sin(a)], dtype=float)

def angle_from_xy(x, y, center=(0.0, 0.0)):
    """
    Convert an x/y mouse position to a screen angle in degrees.
    """
    dx = x - center[0]
    dy = y - center[1]

    ang = np.degrees(np.arctan2(dy, dx))

    if ang < 0:
        ang += 360

    return ang

## check if the mouse position (pt) is within a rectangle (e.g., the submit button) or on the edge. 
def point_in_rect(pt, center, w, h):
    x, y = pt ## get the x and y values of the position pt
    cx, cy = center # center is the center of the given rectangle

    return (
        cx - w / 2 <= x <= cx + w / 2 and
        cy - h / 2 <= y <= cy + h / 2
    )

## calculate the distance from the point(xy)
def distance_to_center(x, y, center):
    return math.hypot(x - center[0], y - center[1])

## update the position of selector (the bar on the ring that can select the color). It is used to draw the line of bar with the starting position and ending position. 
def update_color_selector_geometry(ring_center, inner_ring, outer_ring, selector_angle):
    eps = 1.0 # The epsillon offset to keep the selector bar within the color ring

    p1 = np.array(ring_center) + pol_to_cart(
        inner_ring + eps,
        selector_angle
    )

    p2 = np.array(ring_center) + pol_to_cart(
        outer_ring - eps,
        selector_angle
    )
    return p1,p2


## This function checks whether the mouse is on the selector bar when draggable
def mouse_on_ring(mouse_pos, ring_center, inner_ring, outer_ring):
    d = distance_to_center(
        mouse_pos[0],
        mouse_pos[1],
        ring_center
    )
    return inner_ring <= d <= outer_ring

## This function converts the selection (polar system) to the rgb value that can be used for Psychopy drawing
def update_selected_color_from_angle(selector_angle, hue_rgb_psy, ring_rotation = 0):
  
    selected_hue = (selector_angle - ring_rotation) % 360

    hue_idx = int(round(selected_hue)) % 360
    current_hue_idx = hue_idx

    selected_rgb = hue_rgb_psy[hue_idx]
    return current_hue_idx, selected_rgb

def update_residence_bar(bar, angle):
    # Bar midpoint exactly on ring
    x, y = pol_to_cart(angle, RESIDENCE_RING_RADIUS)
    bar.pos = (x, y)
    # Radial orientation
    bar.ori = 90 - angle

## This function updates the actions when mouse is clicked left
def mouse_press_on_the_ring(mouse, mouse_on_ring, ring_center):
    mouse_pos = mouse.getPos()
    left = mouse.getPressed(getTime=False)[0] #getPressed return the state of left, middle, and right buttons. It typically return a list with three binary digits, 0 not pressed, 1 pressed, for example [1, 0, 0]
    if left and mouse_on_ring:
        selector_angle = angle_from_xy(mouse_pos[0], mouse_pos[1], ring_center)
        return left, selector_angle
    else:
        return left



## This function precomputes the hue color, to allow for smooth updating of colors
def create_hue_rgb_psy():
    hue_rgb_psy = np.array([lch_to_psychopy_rgb(L_VALUE, C_VALUE, hh) for hh in range(360)]),
    return hue_rgb_psy

## This function coverts the time_duration to frames, to allowed for more accurate time control
def time_to_frame(time_in_seconds):
    return math.ceil(time_in_seconds * fresh_rate)

## This function generates the initial fill image
def fill_image(image_path, image_name, mask_color = (0, 0, 0)):
    fill_path = image_path + "fill_layer/" + image_name ## set the path to the fill layer images. These images should be created in advance and should be the same size as the alien images. The fill layer images should have a transparent background and the alien shape filled with white (255, 255, 255) in rgb space. This will allow us to use the color parameter in the ImageStim to change the color of the alien images during the experiment.
    fill_rgba = np.array(Image.open(fill_path).convert("RGBA"), dtype=np.uint8)
    h_img, w_img = fill_rgba.shape[:2]
    fill_rgb = fill_rgba[:, :, :3]
    fill_alpha = fill_rgba[:, :, 3]
    white_mask = (
    (fill_rgb[:, :, 0] >= MASK_THRESHOLD) &
    (fill_rgb[:, :, 1] >= MASK_THRESHOLD) &
    (fill_rgb[:, :, 2] >= MASK_THRESHOLD) &
    (fill_alpha > 0)
    )
    gray_tex = np.zeros((h_img, w_img, 4), dtype=np.uint8)

    shade = np.mean(fill_rgb.astype(np.float32), axis=2)
    shade = np.clip(shade, 0, 255).astype(np.uint8)

    gray_tex[:, :, 0] = np.where(white_mask, shade, 0)
    gray_tex[:, :, 1] = np.where(white_mask, shade, 0)
    gray_tex[:, :, 2] = np.where(white_mask, shade, 0)
    gray_tex[:, :, 3] = np.where(white_mask, fill_alpha, 0)

    gray_fill_image = Image.fromarray(gray_tex, mode="RGBA")
    return gray_fill_image

## This function generates outline image. 
def outline_image(image_path, image_name):
    outline_path = image_path + "outline_layer/" + image_name ## set the path to the outline layer images. These images should be created in advance and should be the same size as the alien images. The outline layer images should have a transparent background and the alien shape filled with white (255, 255, 255) in rgb space. This will allow us to use the color parameter in the ImageStim to change the color of the alien images during the experiment.
    outline_rgba = np.array(Image.open(outline_path).convert("RGBA"), dtype=np.uint8)
    
    get_outline_image = Image.fromarray(outline_rgba, mode="RGBA")
    return get_outline_image


## This function calculate the position of labels 
def calculate_pos():
    center_x, center_y = TEST_LABEL_CENTER
    distance_x, distance_y = TEST_LABEL_DISTANCE_TO_CENTER
    parameters = [(-1,1),(-1,-1),(1,-1),(1,1)]
    pos = []
    for i in range(4):
        pos_x = center_x + parameters[i] * (distance_x + 1/2 * TEST_LABEL_SIZE[0])
        pos_y = center_y + parameters[i] * (distance_y + 1/2 * TEST_LABEL_SIZE[1])
        pos.append((pos_x, pos_y))
    return pos

# ============ Main Script ===========

exp_info = {
    "participant_id": "",
    "group": ["1", "2", "3"], # 1. color 2. residence 3. no rule
    "condition": ["A", "B"], # A. Immediate Test B. Delayed Test
    "session": "1" # Session 1 includes learning and working memory test 2. Session 2 includes main memory test, generalization and color construction
}
dlg = gui.DlgFromDict(exp_info, title="Participant Information")
if not dlg.OK:
    core.quit()

exp_info["experiment_date"] = now_date()
exp_info["experiment_start_time"] = now_time()
exp_info["experiment_start_datetime"] = now_datetime()

### Create the monitor object (this will load the properties from the saved monitor object created with create_monitor.py)
my_monitor = monitors.Monitor(MONITOR)
my_monitor.setCurrent(MONITOR) ## set this monitor as the current monitor so that when we create a window, it will use the properties from this monitor

### Create a window

win = visual.Window(
    monitors=my_monitor,
    size=WIN_SIZE,
    units='pix',
    color=WIN_BG,
    colorSpace='rgb',
    allowGUI=True,
    fullscr=False,
    waitBlanking=True,
    useRetina=RETINA
)

fresh_rate = win.getActualFrameRate()

### Create a keyboard object to check for key presses
kb = keyboard.Keyboard()

### Instructions
instructions_text = visual.TextStim(
    win = win,
    text = "Welcome to the Alien Memory Experiment!\n\nIn this experiment, you will be shown a series of images of aliens. Your task is to remember the aliens and their features.\n\nPress any key to start the experiment.",
    color = (1, 1, 1),
    colorSpace = 'rgb',
    height = 30,
    wrapWidth = 800
)
instructions_text.draw()
win.flip()
event.waitKeys() ## wait for a key press to start the experiment


