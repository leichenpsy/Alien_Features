

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
BLANK_TIME = 0.5 ## time to show blank screen between encoding and practice phases in seconds, and between practice and feedback in seconds
INTERVAL_TIME = 1.0 ## time to show blank screen between learning trials in seconds
FIXATION_TIME = np.random.uniform(0.75,1.25) ## time to show fixation cross before each trial in seconds (randomized between 0.75 and 1.25 second)  
BLOCK_BREAK_TIME = 90.0 ## time to show break screen between learning blocks in seconds 
FEEDBACK_TIME = 1.5 # Time to re-present the encoding screen after all 3 practice in seconds
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
COLOR_RING_UNIT = 0.1 #(degree)
TEST_RING_CENTER = (WIN_SIZE[0] * 1/3, 0)
COLOR_RING_RADIUS = 10
COLOR_RING_WIDTH = 2
COLOR_RING_ROTATION = True
COLOR_RING_SEGMENTS = 360/COLOR_RING_UNIT
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

def saveData(names, values, data):
    data.update((zip(names, values)))
    return data


### Learning trial function. This function will control the flow of each learning trial (and practice learning trials), including the encoding phase, the practice phase, and the inter-trial interval. It will also handle the presentation of stimuli and the collection of responses during the practice phase. The parameters defined above will be used to control the timing and structure of the learning trials.
#def learning_trial(alien_image, alien_name, alien_planet, practice=False):
    ## 1. Fixation Cross

def display_fixation_cross():
    duration = np.random.uniform(0.75,1.25)
    fixation = win.TextStim(
        win = win,
        text = "+",
        color = (1, 1, 1),
        colorSpace = 'rgb',
        height = 50
    )
    fixation_N_Frames = time_to_frame(duration) ## show fixation cross for a random duration between 0.75 and 1.25 seconds that are converted to the number of frames for timing accuracy.
    fix_start = now_time()
    for fixation_Frame in (fixation_N_Frames):
        fixation.draw()
        win.flip()
    fix_end = now_time()
    fix_duration = duration
    return fix_start, fix_end, fix_duration 

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

def encoding_screen_present(alien_image, alien_color, alien_pos, alien_fake_name, alien_planet, duration, residence_ring_pos,residence_angle): 
    nFrames = time_to_frame(duration)
    encoding_start = now_time()
    for frame in range(nFrames):
        encoding_screen_draw(alien_image, alien_color, alien_pos, alien_fake_name, alien_planet, residence_ring_pos ,residence_angle)
        win.flip()
    encoding_end = now_time()
    encoding_duration = encoding_end - encoding_start
   
    return encoding_start, encoding_end, encoding_duration
    
def blank_screen_present(duration):
    emptyText = visual.TextStim(
        win = win,
        text = "")
    nFrames = time_to_frame(duration)
    for frame in range(nFrames):
        emptyText.draw()
        win.flip()
    

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

def planet_test_screen(practiceNo, practice = False):
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
                if practice:
                    saveData(['practiceNo','practice_planet_start_time','practice_planet_alien', 'practice_planet_correct', 'practice_planet_selected', 'practice_planet_rt', 'practice_end_time'],[practiceNo,trial_start_time, alien, planet, selected_planet,rt, trial_end_time], trial_data)
                else:
                    saveData(['test_planet_start_time', 'test_planet_alien', 'test_planet_correct', 'test_planet_selected', 'test_planet_rt', 'test_end_time'], [trial_start_time, alien, planet, selected_planet,rt, trial_end_time], trial_data)

            break
    
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

def color_test_screen(practiceNo = None, practice = False):
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
    

    initial_hue = np.round(np.random.uniform(0, 360), 1)
    ring_rotation = np.round(np.random.uniform(0, 360), 1)

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
    initial_rgb = hue_rgb_psy[initial_hue/COLOR_RING_UNIT]
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
    if practice:
        saveData(['practice_color_No', 'practice_color_start_time', 'practice_color_alien', 'practice_color_correct', 'practice_color_selected', 'practice_color_rt', 'practice_color_end_time', 'practice_color_ring_initial_hue','practice_color_ring_rotation'], [practiceNo, trial_start_time, alien, color, selected_hue, rt, trial_end_time, initial_hue, ring_rotation], trial_data)
    else:
        saveData(['test_color_start_time','test_color_alien', 'test_color_correct', 'test_color_selected', 'test_color_rt', 'test_color_end_time', 'test_color_ring_initial_hue','test_color_ring_rotation'], [trial_start_time, alien, color, selected_hue, rt, trial_end_time, initial_hue, ring_rotation], trial_data)
   

def residence_test_screen(practiceNo = None, practice = False):
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
    initial_bar_angle = np.round(np.random.uniform(0, 360),1)
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

    selected_residence = np.round(current_angle,1)
    rt = rt_clock.getTime()
    trial_end_time = now_time()
    if practice:
        saveData(['practice_residence_no', 'practice_residence_start_time', 'practice_residence_alien', 'practice_residence_correct', 'practice_residence_selected', 'practice_residence_rt', 'practice_residence_end_time', 'practice_residence_ring_initial_angle'],[practiceNo, trial_start_time, alien, residence, selected_residence, rt, trial_end_time, initial_bar_angle], trial_data)
    else:
        saveData(['test_residence_start_time', 'test_residence_alien', 'test_residence_correct', 'test_residence_selected', 'test_residence_rt', 'test_residence_end_time', 'test_residence_ring_initial_angle'][trial_start_time, alien, residence, selected_residence, rt, trial_end_time, initial_bar_angle], trial_data)
    
def generatePracticeOrder(planet, color, residence):
    list_1 = [planet, color, residence]
    random.shuffle(list_1)
    practice_set_1 = []
    practice_set_1.append(list_1)
    list_2 = [list_1[1], list_1[2],list_1[0]]
    practice_set_1.append(list_2)
    list_3 = [list_1[2], list_1[0], list_1[1]]
    practice_set_1.append(list_3)
    random.shuffle(practice_set_1)
    practice_set_2 = []
    for i in range(3):
        old_list = practice_set_1[i]
        new_list = [old_list[0], old_list[2], old_list[1]]
        practice_set_2.append(new_list)
    return practice_set_1, practice_set_2
    

        
def practice_flow(practiceOrder):
    practice_start = now_time()
    for i in range(len(practiceOrder)):
        practiceNo = i + 1
        practiceOrder[i](practiceNo, True)
    practice_end = now_time()
    practice_duration = practice_end - practice_start
    return practice_start, practice_end, practice_duration

def learning_trial(data, trial_no, block, alien, alien_name, color, planet, residence, practiceOrder):
    trial_data = {
    "participant_id": exp_info["participant_id"],
    "group": exp_info["group"],
    "condition": exp_info["condition"],
    "session": exp_info["session"],
    "trial_no": trial_no,
    "block": block,
    }
    learning_trial_start = now_time()
    fix_start, fix_end, fix_duration = display_fixation_cross()
    encoding_start, encoding_end, encoding_duration = encoding_screen_present(alien, color, ALIEN_POS, alien_name, planet, ENCODING_TIME, ALIEN_POS, residence)
    saveData(['trial_no', 'block','learning_trial_start','fix_start','fix_end','fix_duration', 'encoding_start', 'encoding_end', 'encoding_duration'], [trial_no, block,learning_trial_start,fix_start, fix_end, fix_duration, encoding_start, encoding_end, encoding_duration], trial_data)
    blank_screen_present(BLANK_TIME)
    mouse = event.Mouse(win=win)
    practice_start, practice_end, practice_duration = practice_flow(practiceOrder)
    blank_screen_present(BLANK_TIME)
    feedback_start, feedback_end, feedback_duration = encoding_screen_present(alien, color, ALIEN_POS, alien_name, planet, FEEDBACK_TIME, ALIEN_POS, residence)
    trial_end_time = now_time()
    saveData(['practice_start', 'practice_end', 'practice_duration', 'feedback_start', 'feedback_end', 'feedback_duration','trial_end_time'], [practice_start, practice_end, practice_duration, feedback_start, feedback_end, feedback_duration, trial_end_time], trial_data)
    data.append(trial_data)

def study_block(data, block, practice_order, no_trials):
    for i in range(no_trials):
        trial_no = i+1
        learning_trial(data, block, )

def generateStudySequence():


def generateColors():



def generateStudyMaterials():
    
    
def sample_points_around_center(
    m,
    n,
    k,
    kappa=15,
    max_offset=35,
    jitter_range=3,
    mean_tolerance=1.5,
    min_per_side=None,
    avoid_exact_symmetry=True,
    max_attempts=50000,
    seed=None
):
    """
    Generate n integer hue samples around mean center m.

    Requirements:
        - Center is jittered by random integer in [-jitter_range, +jitter_range].
        - Samples are drawn from a von Mises distribution around the jittered center.
        - Final hues are rounded to nearest integer.
        - Each sampled hue is distinct.
        - Pairwise circular distance is at least k degrees.
        - Circular mean is close to the jittered center.
        - Exact symmetry is avoided.

    Parameters
    ----------
    m : int or float
        Intended category center in degrees, e.g., 0, 90, 180, 270.

    n : int
        Number of samples to generate.

    k : int or float
        Minimum pairwise distance between samples, in degrees.

    kappa : float
        Concentration of von Mises distribution.
        Larger values produce tighter samples around the center.

    max_offset : int or float
        Maximum allowed absolute angular offset from the jittered center.

    jitter_range : int
        Integer jitter range. If 3, center jitter is sampled from [-3, 3].

    mean_tolerance : float
        Maximum allowed circular-mean error from the jittered center.

    min_per_side : int or None
        Minimum number of samples required on each side of the center.
        If None, uses floor(n / 3), which allows mildly asymmetric samples.

    avoid_exact_symmetry : bool
        If True, reject samples with too many mirror-paired offsets.

    max_attempts : int
        Number of rejection-sampling attempts.

    seed : int or None
        Random seed.

    Returns
    -------
    dict
        Contains jittered center, offsets, hues, mean hue, and diagnostics.
    """

    rng = np.random.default_rng(seed)

    if min_per_side is None:
        min_per_side = max(1, n // 3)

    for attempt in range(1, max_attempts + 1):

        # Integer center jitter, e.g. -3, -2, ..., 2, 3
        center_jitter = rng.integers(
            -jitter_range,
            jitter_range + 1
        )

        target_center = int(round(m + center_jitter)) % 360

        offsets = []
        points = []

        inner_attempts = 0
        max_inner_attempts = 10000

        while len(points) < n and inner_attempts < max_inner_attempts:
            inner_attempts += 1

            # Draw angular offset from von Mises centered at 0.
            sample_rad = rng.vonmises(mu=0, kappa=kappa)
            offset = int(round(np.rad2deg(sample_rad)))

            # Convert to signed offset in [-180, 180)
            offset = int(circular_signed_diff_deg(offset, 0))

            if offset == 0:
                continue

            if abs(offset) > max_offset:
                continue

            candidate_point = int(round(target_center + offset)) % 360

            # Distinct hue requirement.
            if candidate_point in points:
                continue

            # Minimum pairwise distance requirement.
            too_close = False
            for old_point in points:
                if circular_abs_diff_deg(candidate_point, old_point) < k:
                    too_close = True
                    break

            if too_close:
                continue

            points.append(candidate_point)
            offsets.append(offset)

        if len(points) < n:
            continue

        mean_point = circular_mean_deg(points)
        mean_error = circular_abs_diff_deg(mean_point, target_center)

        if mean_error > mean_tolerance:
            continue

        n_negative = sum(o < 0 for o in offsets)
        n_positive = sum(o > 0 for o in offsets)

        # Ensure the sample set is not all on one side.
        if n_negative < min_per_side or n_positive < min_per_side:
            continue

        if avoid_exact_symmetry and has_exact_symmetry(offsets):
            continue

        return {
            "base_center": m,
            "center_jitter": int(center_jitter),
            "target_center": int(target_center),
            "points": sorted([int(h) for h in hues]),
            "offsets": sorted([int(o) for o in offsets]),
            "mean_point": round(float(mean_hue), 3),
            "mean_error": round(float(mean_error), 3),
            "min_pairwise_distance": round(float(min_pairwise_distance_deg(hues)), 3),
            "n_negative": int(n_negative),
            "n_positive": int(n_positive),
            "attempts": attempt
        }

    raise RuntimeError(
        "Could not generate valid hue samples. "
        "Try increasing max_attempts, increasing mean_tolerance, "
        "reducing k, reducing kappa, or increasing max_offset."
    )

def generate_four_point_sets(
    n,
    k,
    centers=None,
    labels=None,
    kappa=15,
    max_offset=35,
    jitter_range=3,
    mean_tolerance=1.5,
    within_k=None,
    between_k=10,
    boundary_margin=5,
    max_outer_attempts=2000,
    seed=None
):
    """
    Generate four hue sample sets around 0, 90, 180, and 270 degrees.

    Each set:
        - has n integer hue samples
        - is sampled from a von Mises distribution
        - has jittered center in [-3, 3] by default
        - avoids exact symmetry
        - has pairwise distances at least k degrees

    Then the four sets are checked for clear category boundaries.

    Parameters
    ----------
    n : int
        Number of samples per color category.

    k : int
        Minimum within-set distance.

    centers : list or None
        Default: [0, 90, 180, 270]

    labels : list or None
        Default: ["red", "yellow", "green", "blue"]

    within_k : int or None
        Within-set spacing used in boundary check.
        If None, uses k.

    between_k : int
        Minimum distance between samples from different sets.

    boundary_margin : int
        Minimum distance from any sample to its category boundary.

    max_outer_attempts : int
        Number of times to regenerate all four sets.

    seed : int or None
        Random seed.

    Returns
    -------
    dict
        Contains generated sample sets and boundary report.
    """

    if centers is None:
        centers = [0, 90, 180, 270]

    if labels is None:
        labels = ["red", "yellow", "green", "blue"]

    if len(centers) != 4 or len(labels) != 4:
        raise ValueError("centers and labels must both have length 4.")

    if within_k is None:
        within_k = k

    rng = np.random.default_rng(seed)

    for outer_attempt in range(1, max_outer_attempts + 1):

        sample_sets = {}

        # Use independent seeds for each color set.
        for label, center in zip(labels, centers):
            color_seed = int(rng.integers(0, 2**32 - 1))

            sample_sets[label] = sample_points_around_center(
                m=center,
                n=n,
                k=k,
                kappa=kappa,
                max_offset=max_offset,
                jitter_range=jitter_range,
                mean_tolerance=mean_tolerance,
                avoid_exact_symmetry=True,
                seed=color_seed
            )

        boundary_report = check_four_set_boundaries(
            sample_sets=sample_sets,
            within_k=within_k,
            between_k=between_k,
            boundary_margin=boundary_margin
        )

        if boundary_report["passed"]:
            return {
                "sample_sets": sample_sets,
                "boundary_report": boundary_report,
                "outer_attempts": outer_attempt
            }

    raise RuntimeError(
        "Could not generate four valid color sets. "
        "Try increasing max_outer_attempts, increasing mean_tolerance, "
        "reducing k, reducing between_k, reducing boundary_margin, "
        "reducing kappa, or increasing max_offset."
    )
    


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

    hue_idx = np.round(selected_hue,1) % 360
    current_hue_idx = hue_idx

    selected_rgb = hue_rgb_psy[hue_idx/COLOR_RING_UNIT]
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
    hue_rgb_psy = np.array([lch_to_psychopy_rgb(L_VALUE, C_VALUE, hh) for hh in range(360/COLOR_RING_UNIT)]),
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


## This function calculate the position of labels in planet test
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

def round_unique_hues(hues, min_sep=1):
    """
    Round hues to integer degrees and ensure they are unique.
    """
    rounded = [int(round(h)) % 360 for h in hues]

    if len(set(rounded)) != len(rounded):
        return None

    # Check circular minimum separation
    for i in range(len(rounded)):
        for j in range(i + 1, len(rounded)):
            d = abs(((rounded[i] - rounded[j] + 180) % 360) - 180)
            if d < min_sep:
                return None

    return rounded

def circular_mean_deg(angles):
    """
    Circular mean of angles in degrees, returned in [0, 360).
    """
    angles = np.asarray(angles)
    radians = np.deg2rad(angles)

    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))

    return np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360


def circular_signed_diff_deg(a, b):
    """
    Signed circular difference a - b in degrees, returned in [-180, 180).
    """
    return ((a - b + 180) % 360) - 180


def circular_abs_diff_deg(a, b):
    """
    Absolute circular distance between two angles in degrees.
    """
    return abs(circular_signed_diff_deg(a, b))


def min_pairwise_distance_deg(hues):
    """
    Minimum circular distance among all pairs in one hue set.
    """
    hues = list(hues)

    if len(hues) < 2:
        return np.inf

    min_dist = np.inf

    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            d = circular_abs_diff_deg(hues[i], hues[j])
            min_dist = min(min_dist, d)

    return min_dist


def has_exact_symmetry(offsets):
    """
    Returns True if offsets contain exact mirror pairs around zero.

    Example:
        [-20, -10, 10, 20] -> True-ish symmetric
    """
    offsets = list(offsets)
    offset_set = set(offsets)

    mirror_count = 0

    for o in offsets:
        if o != 0 and -o in offset_set:
            mirror_count += 1

    # Each mirrored pair is counted twice.
    mirror_pairs = mirror_count // 2

    return mirror_pairs >= len(offsets) // 2

def circular_midpoint_deg(a, b):
    """
    Circular midpoint from angle a to angle b along the shortest path.
    """
    diff = circular_signed_diff_deg(b, a)
    return (a + diff / 2) % 360


def angle_in_clockwise_arc(x, start, end):
    """
    Returns True if x lies on clockwise arc from start to end.
    Angles are in degrees.
    """
    return ((x - start) % 360) <= ((end - start) % 360)


def boundary_interval_for_center(center, left_neighbor, right_neighbor):
    """
    Compute the angular interval belonging to one category center.

    The interval is bounded by:
        midpoint(left_neighbor, center)
        midpoint(center, right_neighbor)

    Returns
    -------
    left_boundary, right_boundary
    """
    left_boundary = circular_midpoint_deg(left_neighbor, center)
    right_boundary = circular_midpoint_deg(center, right_neighbor)

    return left_boundary, right_boundary


def check_four_set_boundaries(
    sample_sets,
    within_k,
    between_k=10,
    boundary_margin=5
):
    """
    Check boundary clarity for four hue sample sets.

    Expected input format
    ---------------------
    sample_sets should be a dict like:

    {
        "red": {
            "target_center": 359,
            "hues": [...]
        },
        "yellow": {
            "target_center": 92,
            "hues": [...]
        },
        "green": {
            "target_center": 181,
            "hues": [...]
        },
        "blue": {
            "target_center": 268,
            "hues": [...]
        }
    }

    Requirements checked
    --------------------
    1. Each set has within-set spacing >= within_k.
    2. Any two hues from different sets are separated by >= between_k.
    3. Each hue lies inside its category's boundary interval.
    4. Each hue is at least boundary_margin degrees away from boundaries.

    Returns
    -------
    dict
        Diagnostic report.
    """

    names = list(sample_sets.keys())

    if len(names) != 4:
        raise ValueError("sample_sets must contain exactly 4 sets.")

    centers = {
        name: sample_sets[name]["target_center"]
        for name in names
    }

    # Sort centers circularly by angle.
    sorted_names = sorted(names, key=lambda name: centers[name])

    report = {
        "passed": True,
        "within_set": {},
        "between_set": {},
        "boundaries": {},
        "violations": []
    }

    # 1. Check within-set spacing.
    for name in names:
        hues = sample_sets[name]["hues"]
        min_dist = min_pairwise_distance_deg(hues)

        ok = min_dist >= within_k

        report["within_set"][name] = {
            "min_distance": round(float(min_dist), 3),
            "required": within_k,
            "passed": bool(ok)
        }

        if not ok:
            report["passed"] = False
            report["violations"].append(
                f"{name}: within-set minimum distance {min_dist:.2f} < {within_k}"
            )

    # 2. Check between-set spacing.
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a = names[i]
            name_b = names[j]

            points_a = sample_sets[name_a]["points"]
            points_b = sample_sets[name_b]["points"]

            min_dist = np.inf

            for ha in points_a:
                for hb in points_b:
                    d = circular_abs_diff_deg(ha, hb)
                    min_dist = min(min_dist, d)

            ok = min_dist >= between_k

            pair_name = f"{name_a}-{name_b}"

            report["between_set"][pair_name] = {
                "min_distance": round(float(min_dist), 3),
                "required": between_k,
                "passed": bool(ok)
            }

            if not ok:
                report["passed"] = False
                report["violations"].append(
                    f"{pair_name}: between-set minimum distance {min_dist:.2f} < {between_k}"
                )

    # 3. Check category boundary intervals.
    for idx, name in enumerate(sorted_names):
        center = centers[name]

        left_name = sorted_names[(idx - 1) % 4]
        right_name = sorted_names[(idx + 1) % 4]

        left_center = centers[left_name]
        right_center = centers[right_name]

        left_boundary, right_boundary = boundary_interval_for_center(
            center=center,
            left_neighbor=left_center,
            right_neighbor=right_center
        )

        points = sample_sets[name]["points"]

        point_reports = []

        for p in points:
            inside = angle_in_clockwise_arc(
                p,
                left_boundary,
                right_boundary
            )

            dist_to_left = circular_abs_diff_deg(h, left_boundary)
            dist_to_right = circular_abs_diff_deg(h, right_boundary)
            min_boundary_dist = min(dist_to_left, dist_to_right)

            margin_ok = min_boundary_dist >= boundary_margin

            point_ok = inside and margin_ok

            point_reports.append({
                "point": int(p),
                "inside": bool(inside),
                "min_boundary_distance": round(float(min_boundary_dist), 3),
                "passed": bool(point_ok)
            })

            if not point_ok:
                report["passed"] = False
                report["violations"].append(
                    f"{name}: hue {h} failed boundary check. "
                    f"inside={inside}, margin={min_boundary_dist:.2f}"
                )

        report["boundaries"][name] = {
            "center": int(center),
            "left_neighbor": left_name,
            "right_neighbor": right_name,
            "left_boundary": round(float(left_boundary), 3),
            "right_boundary": round(float(right_boundary), 3),
            "boundary_margin": boundary_margin,
            "points": point_reports
        }

    return report
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


