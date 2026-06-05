

# ============Import Modules===========

from psychopy import monitors, visual, event, core, gui
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np
import random
import math
import csv
from PIL import Image
from datetime import datetime
from collections import Counter, defaultdict
import os


# ============ Define parameters ===========
MONITOR = 'AlienMemoryMonitor' ## set the name of the monitor that you created with create_monitor.py
WIN_SIZE = ()
WIN_BG = () ## set window background color in rbg format, e.g. (0, 0, 0) for black, (1, 1, 1) for white, (-1, -1, -1) for black in rgb space
RETINA = True ## set to True if using a retina display, False otherwise. This will ensure that the stimuli are displayed at the correct size on retina displays, which have a higher pixel density.
Allow_escape = True
N_BLOCKS = 3

##### Learning Trial Parameters #####
ENCODING_TIME = 4.0 ## time to show each alien during encoding phase in seconds
BLANK_TIME = 0.5 ## time to show blank screen between encoding and practice phases in seconds, and between practice and feedback in seconds
INTERVAL_TIME = 1.0 ## time to show blank screen between learning trials in seconds
FIXATION_TIME = np.random.uniform(0.75,1.25) ## time to show fixation cross before each trial in seconds (randomized between 0.75 and 1.25 second)  
BLOCK_BREAK_TIME = 90.0 ## time to show break screen between learning blocks in seconds 
FEEDBACK_TIME = 1.5 # Time to re-present the encoding screen after all 3 practice in seconds
STUDY_SESSION_BREAK_TIME = 300.0 ## time to show break screen between study session and working memory session in seconds. 
STUDY_TEST_INTERVAL = 1200.0 ## time between the end of the study session and the beginning of the test session in seconds for the immediate test group. 
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
COLOR_BAR_COLOR = 'black'
COLOR_BAR_WIDTH = 0.03



# Residence ring parameters
RESIDENCE_RING_THIKNESS = 2
RESIDENCE_RING_RADIUS = 10
RESIDENCE_RING_COLOR = ()
RESIDENCE_RING_EDGES = 512 #the number of line segments to draw the circle shape, higher number means smoother and rounder
RESIDENCE_BAR_WIDTH = 0.03
RESIDENCE_BAR_LENGTH = 0.1
RESIDENCE_BAR_COLOR = 'black'


###### MODE PARAMETERS ######
MODES = ['debug','demo','experiment']
GROUPS = ['1','2'] ## set the group assignemnt for the experiment. '1' for immediate test group, '2' for 1-day-delay test group. 
CONDITIONS = ['C', 'R', 'N'] # C: group by color R: group by residence cluster N: no rule

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
WM_MEMORY_STATUS = True ## set to True if working memory is included in the experiment, False otherwise.
TESTING_FIRST = True ## Set to true of memory test comes before generalization test, false if generalization test comes first.
COLOR_RECONSTRUCTION = True ## set to True if color reconstruction task is included in the experiment, False otherwise.


### Working memory test settings
WM_N_TRIALS = 12
DIGIT_LENGTH = 8
DIGIT_DISPLAY_TIME = 2.0
WM_RETENTION_INTERVAL = 3.0
WM_INTER_TRIAL_TIME = 1.0

### main test settings
TEST_BLOCK_BREAK_TIME = 60.0 ## time to show break screen between test blocks in seconds.
TEST_INTER_TRIAL_TIME = 1.0 ## time to show blank screen between test trials in seconds.
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
        image=fill_image(alien_image),
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
        image=outline_image(alien_image),
        pos=alien_pos,
        size=ALIEN_SIZE,
        units='pix',
        interpolate=True,
        flipVert=IMAGE_FLIP_VERT      
   )
    return outline_stim

    
def update_alien_fill_color(alien_stim, new_color):
    alien_stim.color = new_color



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

def residence_ring(position):
    residence_ring_stim = visual.Circle(
        win = win,
        radius = RESIDENCE_RING_RADIUS,
        edges = RESIDENCE_RING_EDGES,
        lineColor = RESIDENCE_RING_COLOR,
        fillColor = None,
        lineWidth = RESIDENCE_RING_THIKNESS,
        interpolate = True,
        pos = position
    )
    return residence_ring_stim
    
def residence_ring_bar():
    residence_bar_stim = visual.Line(
        win = win,
        start = (0, 0),
        end = (0, 0),
        lineColor = RESIDENCE_BAR_COLOR,
        lineWidth = RESIDENCE_BAR_WIDTH
    )
    return residence_bar_stim



def encoding_screen_draw(alien_info): 
    alien = alien_info['alien']
    alien_folder = alien_info['alien_folder']
    alien_pos = ALIEN_POS
    alien_color = alien_info['alien_color']
    alien_planet = alien_info['alien_planet']
    alien_residence = alien_info['alien_residence']
    alien_name = alien_info['alien_name']

    fill_alien_image = ALIEN_PATH_LEARNING + alien_folder + 'fill_layer' + alien
    fill_stim = alien_fill_image(fill_alien_image, alien_pos)
    update_alien_fill_color(fill_stim, alien_color)
    fill_stim.draw()
    outline_alien_image = ALIEN_PATH_LEARNING + alien_folder + 'outline_layer' + alien
    outline_stim = alien_outline_image(outline_alien_image, alien_pos)
    outline_stim.draw()
    fake_name_stim = alien_text(alien_name, ALIEN_NAME_POS, ALIEN_NAME_HEIGHT, ALIEN_NAME_COLOR)
    fake_name_stim.draw()
    planet_text_stim = alien_text(alien_planet, ALIEN_PLANET_POS, ALIEN_PLANET_HEIGHT, ALIEN_PLANET_COLOR )
    planet_text_stim.draw()
    residence_ring_stim = residence_ring(ALIEN_POS)
    residence_ring_stim.draw()
    residence_bar_stim = residence_ring_bar()
    update_residence_bar(ALIEN_POS, residence_bar_stim, alien_residence)
    residence_bar_stim.draw()

def encoding_screen_present(alien_info): 
    nFrames = time_to_frame(ENCODING_TIME)
    encoding_start = now_time()
    for frame in range(nFrames):
        encoding_screen_draw(alien_info)
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
    selected_label = next((label for label in labels_stim if point_in_rect(pt,label.pos, TEST_LABEL_SIZE[0], TEST_LABEL_SIZE[1])), None)
    if selected_label is not None:
        click_label = True
    return click_label, selected_label

def planet_test_screen(practiceNo, alien_info, data, practice = False):
    alien = alien_info['alien']
    alien_name = alien_info['alien_name']
    alien_folder = alien_info['alien_folder']
    alien_planet = alien_info['alien_planet']
    alien_path = ALIEN_PATH_LEARNING + alien_folder + 'outline_layer' + alien
    test_alien = alien_outline_image(alien_path, TEST_ALIEN_POS)
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
    stimuli_start_time = now_time()
    mouse = event.Mouse(visible=True)
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
                    saveData(['practiceNo','practice_planet_stimuli_start','practice_planet_alien', 'practice_planet_correct', 'practice_planet_selected', 'practice_planet_rt', 'practice_end_time'],[practiceNo,stimuli_start_time, alien, alien_planet, selected_planet,rt, trial_end_time], data)
                else:
                    saveData(['test_planet_stimuli_start', 'test_planet_alien', 'test_planet_correct', 'test_planet_selected', 'test_planet_rt', 'test_end_time'], [stimuli_start_time, alien, alien_planet, selected_planet,rt, trial_end_time], data)

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

def color_test_screen(practiceNo, alien_info, data, practice = False):
    alien = alien_info['alien']
    alien_name = alien_info['alien_name']
    alien_folder = alien_info['alien_folder']
    alien_color = alien_info['alien_color']
    alien_outline_path = ALIEN_PATH_LEARNING + alien_folder + 'outline_layer' + alien
    alien_fill_path = ALIEN_PATH_LEARNING + alien_folder + 'fill_layer' + alien
    fill_stim = alien_fill_image(alien_fill_path, TEST_ALIEN_POS)
    outline_stim = alien_outline_image(alien_outline_path, TEST_ALIEN_POS)
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
        lineColor = COLOR_BAR_COLOR,
        lineWidth = COLOR_BAR_WIDTH
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
    mouse = event.Mouse(visible=True)
    rt_clock = core.Clock()
    win.callOnFlip(rt_clock.reset)
    win.flip()
    stimuli_start_time = now_time()
    submitted = False
    current_hue_idx = initial_hue/COLOR_RING_UNIT
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
            elif mouse_on_bar(pt, selector_line.start, selector_line.end) and not point_in_rect(pt, TEST_SUBMIT_POS, TEST_SUBMIT_SIZE[0], TEST_SUBMIT_SIZE[1]): 
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
        saveData(['practice_color_No', 'practice_color_stimuli_start_time', 'practice_color_alien', 'practice_color_correct', 'practice_color_selected', 'practice_color_rt', 'practice_color_end_time', 'practice_color_ring_initial_hue','practice_color_ring_rotation'], [practiceNo, stimuli_start_time, alien, alien_color, selected_hue, rt, trial_end_time, initial_hue, ring_rotation], data)
    else:
        saveData(['test_color_stimuli_start_time','test_color_alien', 'test_color_correct', 'test_color_selected', 'test_color_rt', 'test_color_end_time', 'test_color_ring_initial_hue','test_color_ring_rotation'], [stimuli_start_time, alien, alien_color, selected_hue, rt, trial_end_time, initial_hue, ring_rotation], data)
   

def residence_test_screen(practiceNo, alien_info, data, practice = False):
    alien = alien_info['alien']
    alien_name = alien_info['alien_name']
    alien_folder = alien_info['alien_folder']
    alien_residence = alien_info['alien_residence']
    outline_path = ALIEN_PATH_LEARNING + alien_folder + 'outline_layer' + alien
    outline_stim = alien_outline_image(outline_path, TEST_ALIEN_POS)
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

    residence_ring_stim = residence_ring(TEST_RING_CENTER)
    initial_bar_angle = np.round(np.random.uniform(0, 360),1)
    residence_bar_stim = residence_ring_bar()
    update_residence_bar(TEST_RING_CENTER,residence_bar_stim,initial_bar_angle)

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
    stimuli_start_time = now_time()
    submitted = False
    current_angle = initial_bar_angle
    dragging = False
    prev_left = False
    mouse = event.Mouse(visible=True)

    while not submitted:
        mouse.clickRest()
        pt = mouse.getPos()
        mx = pt[0]
        my = pt[1]

        bar_start = residence_bar_stim.start
        bar_end = residence_bar_stim.end

        mouse_on_submit = False
        left = mouse.getPressed()[0]

        new_press = left and not prev_left
        new_release = prev_left and not left

        if new_press:
            if point_in_rect(pt, TEST_SUBMIT_POS, TEST_SUBMIT_SIZE[0], TEST_SUBMIT_SIZE[1]):
                submitted = True
                mouse_on_submit = True
            elif mouse_on_bar(pt, bar_start, bar_end, 0.03) and not mouse_on_submit:
                dragging = True
        if dragging and left:
            current_angle = angle_from_xy(mx, my, TEST_RING_CENTER)
            update_residence_bar(TEST_RING_CENTER, residence_bar_stim, current_angle)
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
        saveData(['practice_residence_no', 'practice_residence_stimuli_start_time', 'practice_residence_alien', 'practice_residence_correct', 'practice_residence_selected', 'practice_residence_rt', 'practice_residence_end_time', 'practice_residence_ring_initial_angle'],[practiceNo, stimuli_start_time, alien, alien_residence, selected_residence, rt, trial_end_time, initial_bar_angle], data)
    else:
        saveData(['test_residence_stimuli_start_time', 'test_residence_alien', 'test_residence_correct', 'test_residence_selected', 'test_residence_rt', 'test_residence_end_time', 'test_residence_ring_initial_angle'][stimuli_start_time, alien, alien_residence, selected_residence, rt, trial_end_time, initial_bar_angle], data)
    

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
    idx1, idx2 = random.sample(range(len(practice_set_2)), 2)
    practice_set_2[idx1], practice_set_2[idx2] = practice_set_2[idx2], practice_set_2[idx1]
    practice_order = []
    for i in range(3):
        practice_set = [practice_set_1[i], practice_set_2[i]]
        practice_order.append(practice_set)
    random.shuffle(practice_order)
    return practice_order

def practice_by_order(content, practiceNo, alien_info, trial_data):
    if content == 'plant':
        planet_test_screen(practiceNo, alien_info, trial_data, True)
    elif content == 'color':
        color_test_screen(practiceNo, alien_info, trial_data,True)
    elif content == 'residence':
        residence_test_screen(practiceNo, alien_info, trial_data, True)
      
def practice_flow(alien_info, trial_data):
    practice_start = now_time()
    practiceOrder = alien_info['alien_practice_order']
    for i in range(len(practiceOrder)):
        practiceNo = i + 1
        practiceContent = practiceOrder[i]
        practice_by_order(practiceContent, practiceNo, alien_info, trial_data)
    practice_end = now_time()
    practice_duration = practice_end - practice_start
    return practice_start, practice_end, practice_duration

def learning_trial(data, trial_no, block, block_start, alien_info):
    trial_data = {
    "trial_no": trial_no,
    "block": block,
    'block_start': block_start
    }
    trial_data.update(exp_info)
    learning_trial_start = now_time()
    fix_start, fix_end, fix_duration = display_fixation_cross()
    encoding_start, encoding_end, encoding_duration = encoding_screen_present(alien_info)
    saveData(['learning_trial_start','fix_start','fix_end','fix_duration', 'encoding_start', 'encoding_end', 'encoding_duration'], [learning_trial_start,fix_start, fix_end, fix_duration, encoding_start, encoding_end, encoding_duration], trial_data)
    blank_screen_present(BLANK_TIME)
    practice_start, practice_end, practice_duration = practice_flow(alien_info, trial_data)
    blank_screen_present(BLANK_TIME)
    feedback_start, feedback_end, feedback_duration = encoding_screen_present(alien_info)
    trial_end_time = now_time()
    trial_duration = trial_end_time - learning_trial_start
    saveData(['practice_start', 'practice_end', 'practice_duration', 'feedback_start', 'feedback_end', 'feedback_duration','trial_end_time','trial_duration'], [practice_start, practice_end, practice_duration, feedback_start, feedback_end, feedback_duration, trial_end_time, trial_duration], trial_data)
    data.append(trial_data)

def study_block(data, dic, block, practice_order):
    ## generate study materials
    study_stimuli = generate_stimuli_for_block(dic, block, practice_order)
    ## generate study sequence
    study_sequence = make_shuffled_list(study_stimuli)
    block_start = now_time()
    ## control trial loops    
    for i in range(len(study_sequence)):
        trial_no = i + 1
        alien_info = {
            'alien': study_sequence[i]['alien'],
            'alien_folder' : study_sequence[i]['alien_folder'],
            'alien_name' : study_sequence[i]['name'],
            'alien_color' : study_sequence[i]['color'],
            'alien_planet' : study_sequence[i]['planet'],
            'alien_residence' : study_sequence[i]['residence'],
            'alien_practice_order' : study_sequence[i]['practice_order']}
        learning_trial(data, trial_no, block, block_start, alien_info)
    block_end = now_time()
    block_duration = block_end - block_start
    for trial in data:
        trial['block_end'] = block_end
        trial['block_duration'] = block_duration
    
def break_screen(duration):
    break_text = visual.TextStim(
        win = win,
        text = "Take a short break!",
        color = 'white',
        height = 30
    )
    break_text.draw()
    win.flip()
    for frame in range(time_to_frame(duration)):
        break_text.draw()
        win.flip()

def break_end_screen():
    break_end_text = visual.TextStim(
        win = win,
        text = "Break is over. Press any key to continue.",
        color = 'white',
        height = 30
    )
    break_end_text.draw()
    win.flip()
    event.waitKeys()
   
    
def generate_until_valid(
    set_labels,
    base_means_deg,
    n=8,
    kappa=80.0,
    side_offset_deg=10.0,
    mean_jitter_deg=2,
    unit=0.1,
    min_within_pairwise_dist_deg=4,
    max_within_span_deg=70.0,
    min_between_set_dist_deg=30.0,
    min_center_dist_deg=60.0,
    max_attempts=100000,
    rng=None,
    verbose=False,
    avoid_within_set_symmetry=True,
    symmetry_pair_tolerance_deg=1.0
):
    """
    Generate samples repeatedly until all requirements are met.

    Internally, this uses metadata such as jittered centers for validation.
    Publicly, it returns only a simple dictionary:

    {
        label_1: [points...],
        label_2: [points...],
        ...
    }

    Returns
    -------
    dict
        Simple sample dictionary with labels as keys and point lists as values.
    """
    if rng is None:
        rng = np.random.default_rng()

    for attempt in range(1, max_attempts + 1):
        internal_sample = generate_circle_point_sets_internal(
            set_labels=set_labels,
            base_means_deg=base_means_deg,
            n=n,
            kappa=kappa,
            side_offset_deg=side_offset_deg,
            mean_jitter_deg=mean_jitter_deg,
            unit=unit,
            rng=rng,
            min_within_pairwise_dist_deg=min_within_pairwise_dist_deg,
        )

        is_valid = check_circle_point_sets_internal(
            internal_sample=internal_sample,
            min_within_pairwise_dist_deg=min_within_pairwise_dist_deg,
            max_within_span_deg=max_within_span_deg,
            min_between_set_dist_deg=min_between_set_dist_deg,
            min_center_dist_deg=min_center_dist_deg,
            avoid_within_set_symmetry=avoid_within_set_symmetry,
            symmetry_pair_tolerance_deg=symmetry_pair_tolerance_deg,
            verbose=verbose,
        )

        if is_valid:
            return simplify_sample(internal_sample)

    raise RuntimeError(
        f"Could not generate a valid sample after {max_attempts} attempts. "
        f"Try relaxing constraints, increasing kappa, or reducing n."
        f"or lowering symmetry_pair_tolerance_deg."
    )

######## Working Memory Trial Function. This function will control the flow of each working memory trial, including the presentation of the digit sequence, the retention interval, and the recall phase. It will also handle the collection of responses during the recall phase and the scoring of those responses based on their accuracy compared to the presented digit sequence. The parameters defined above will be used to control the timing and structure of the working memory trials.

def working_memory_trial(trial_no,session_start,data):
    trial_data = {
        "trial_no": trial_no,
        'session': 'working_memory test',
        'wm_session_start': session_start
    }
    trial_data.update(exp_info)
    wm_trial_start = now_time()
    fixation_start, fixation_end, fixation_duration = display_fixation_cross()
    saveData(['wm_trial_start', 'fixation_start', 'fixation_end', 'fixation_duration'], [wm_trial_start, fixation_start, fixation_end, fixation_duration], trial_data)
    digit_sequence = generate_digit_sequence(DIGIT_LENGTH)
    display_digit_sequence(digit_sequence, DIGIT_DISPLAY_TIME)
    retention_interval_start = now_time()
    blank_screen_present(WM_RETENTION_INTERVAL)
    retention_interval_end = now_time()
    recall_response, rt = collect_recall_response(DIGIT_LENGTH)
    recall_end_time = now_time()
    span, accuracy = score_recall_response(digit_sequence, recall_response)
    wm_trial_end = now_time()
    wm_trial_duration = wm_trial_end - wm_trial_start
    saveData(['digit_sequence', 'retention_interval_start', 'retention_interval_end', 'recall_response', 'recall_rt', 'recall_end_time', 'recall_span', 'recall_accuracy', 'wm_trial_end', 'wm_trial_duration'], [digit_sequence, retention_interval_start, retention_interval_end, recall_response, rt, recall_end_time, span, accuracy, wm_trial_end, wm_trial_duration], trial_data)
    data.append(trial_data)

def generate_digit_sequence(length):
    return ''.join(random.choices('0123456789', k=length))
def display_digit_sequence(sequence, display_time):
    digit_stim = visual.TextStim(
        win = win,
        text = sequence,
        color = 'white',
        height = 50
    )
    nFrames = time_to_frame(display_time)
    for frame in range(nFrames):
        digit_stim.draw()
        win.flip()
def collect_recall_response(expected_length):
    response = ''
    rt_clock = core.Clock()
    win.callOnFlip(rt_clock.reset)
    while True:
        keys = event.getKeys()
        for key in keys:
            if key in '0123456789' and len(response) < expected_length:
                response += key
            elif key == 'backspace' and len(response) > 0:
                response = response[:-1]
            elif key == 'return' and len(response) == expected_length:
                rt = rt_clock.getTime()
                return response, rt
        # Display the current response
        response_stim = visual.TextStim(
            win = win,
            text = response,
            color = 'white',
            height = 50
        )
        response_stim.draw()
        win.flip()
def score_recall_response(correct_sequence, participant_response):
    correct_digits = sum(1 for c, r in zip(correct_sequence, participant_response) if c == r)
    span = 0
    for c, r in zip(correct_sequence, participant_response):
        if c == r:
            span += 1
        else:
            break
    return span, correct_digits / len(correct_sequence)

def run_working_memory_session(data, n_trials):
    session_start = now_time()
    intro_text = visual.TextStim(
        win = win,
        text = "In the following session, you will see a sequence of digits. Try to remember in correct order. After a short delay, you will be asked to recall the digits by typing them in (order matters). Press Enter when you are done. \nPress any key to start.",
        color = 'white',
        height = 30,
        wrapWidth = 1.5
    )
    intro_text.draw()
    win.flip()
    event.waitKeys()
    for i in range(n_trials):
        trial_no = i + 1
        working_memory_trial(trial_no, session_start, data)
        if trial_no < n_trials:
            blank_screen_present(WM_INTER_TRIAL_TIME)
    session_end = now_time()
    session_duration = session_end - session_start
    for trial in data:
        trial['wm_session_end'] = session_end
        trial['wm_session_duration'] = session_duration



######### Main test session functions. #################

def read_stimuli_from_csv(participant_id):
    stimuli_path = 'stimuli_participant_' + participant_id + '.csv'
    stimuli_df = pd.read_csv(stimuli_path)
    return stimuli_df


def shuffle_test_stimuli_dataframe(
    df,
    test_contents=("planet", "color", "residence"),
    max_attempts=500000,
    random_state=None
):
    """
    Convert a pandas DataFrame into a shuffled list of dictionaries.

    For each original row:
    - Make 3 total copies.
    - Add key 'testContent'.
    - Assign one of: 'planet', 'color', 'residence' to each copy.

    Shuffle rows with these constraints:
    1. Rows with the same 'planet' value cannot appear consecutively
       more than 2 times.
    2. Rows with the same 'testContent' value cannot appear consecutively
       more than 2 times.
    3. Rows with the same key-values except 'testContent' cannot be consecutive
       and must be separated by at least 2 other rows.
    4. For rows with the same 'alien' value, the order positions of
       'color', 'residence', and 'planet' are approximately balanced.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    test_contents : tuple
        Test content labels to assign to duplicated rows.
    max_attempts : int
        Maximum number of full shuffle attempts.
    random_state : int or None
        Random seed.

    Returns
    -------
    list of dict
        Shuffled rows as dictionaries.
    """

    rng = random.Random(random_state)

    # -----------------------------
    # Step 1: Convert dataframe to list of dictionaries
    # -----------------------------
    base_rows = df.to_dict(orient="records")

    # -----------------------------
    # Step 2: Expand each row into 3 copies with testContent
    # -----------------------------
    expanded_rows = []

    for row_id, row in enumerate(base_rows):
        contents = list(test_contents)
        rng.shuffle(contents)

        for content in contents:
            new_row = dict(row)
            new_row["testContent"] = content
            new_row["_baseRowID"] = row_id
            expanded_rows.append(new_row)

    # -----------------------------
    # Helper functions
    # -----------------------------
    def same_base_row(row1, row2):
        """
        True if two rows are identical except for testContent
        and internal bookkeeping keys.
        """
        ignore_keys = {"testContent", "_baseRowID"}

        keys1 = set(row1.keys()) - ignore_keys
        keys2 = set(row2.keys()) - ignore_keys

        if keys1 != keys2:
            return False

        return all(row1[k] == row2[k] for k in keys1)

    def violates_recent_constraints(candidate, shuffled):
        """
        Check constraints 1, 2, and 3.
        """

        # Constraint 1:
        # Same planet value cannot appear more than 2 times consecutively.
        if len(shuffled) >= 2:
            if (
                shuffled[-1].get("planet") == candidate.get("planet")
                and shuffled[-2].get("planet") == candidate.get("planet")
            ):
                return True

        # Constraint 2:
        # Same testContent cannot appear more than 2 times consecutively.
        if len(shuffled) >= 2:
            if (
                shuffled[-1].get("testContent") == candidate.get("testContent")
                and shuffled[-2].get("testContent") == candidate.get("testContent")
            ):
                return True

        # Constraint 3:
        # Same original row must be separated by at least 2 other rows.
        # Therefore it cannot appear within the previous 2 positions.
        for previous_row in shuffled[-2:]:
            if same_base_row(candidate, previous_row):
                return True

        return False

    def alien_order_penalty(candidate, shuffled, alien_position_counts):
        """
        Lower penalty means better balance.

        For each alien, we track how often each testContent has appeared
        as the 1st, 2nd, or 3rd occurrence among rows with that alien.
        """

        alien = candidate.get("alien")
        content = candidate.get("testContent")

        if alien is None:
            return 0

        previous_alien_count = sum(
            1 for row in shuffled if row.get("alien") == alien
        )

        position = previous_alien_count % len(test_contents)

        return alien_position_counts[content][position]

    def update_alien_position_counts(row, shuffled, alien_position_counts):
        alien = row.get("alien")
        content = row.get("testContent")

        if alien is None:
            return

        previous_alien_count = sum(
            1 for existing in shuffled if existing.get("alien") == alien
        )

        position = previous_alien_count % len(test_contents)

        alien_position_counts[content][position] += 1

    # -----------------------------
    # Step 3: Try constrained random shuffle
    # -----------------------------
    best_shuffle = None
    best_score = float("inf")

    for attempt in range(max_attempts):

        remaining = expanded_rows[:]
        rng.shuffle(remaining)

        shuffled = []
        alien_position_counts = {
            content: Counter() for content in test_contents
        }

        success = True

        while remaining:
            valid_candidates = [
                row for row in remaining
                if not violates_recent_constraints(row, shuffled)
            ]

            if not valid_candidates:
                success = False
                break

            # Prefer candidates that improve alien-order balance
            min_penalty = min(
                alien_order_penalty(row, shuffled, alien_position_counts)
                for row in valid_candidates
            )

            best_candidates = [
                row for row in valid_candidates
                if alien_order_penalty(row, shuffled, alien_position_counts)
                == min_penalty
            ]

            candidate = rng.choice(best_candidates)

            update_alien_position_counts(
                candidate,
                shuffled,
                alien_position_counts
            )

            shuffled.append(candidate)
            remaining.remove(candidate)

        if success:
            return [
                {
                    k: v
                    for k, v in row.items()
                    if not k.startswith("_")
                }
                for row in shuffled
            ]

        # Keep best partial result as fallback
        if len(shuffled) > 0:
            score = len(expanded_rows) - len(shuffled)
            if score < best_score:
                best_score = score
                best_shuffle = shuffled

    raise RuntimeError(
        "Could not generate a valid shuffle. "
        "Try increasing max_attempts or relaxing constraints."
    )

def memory_test_session():
    memory_test_data = []
    test_session_start = now_time()
    stimuli_df = read_stimuli_from_csv(exp_info['participant_id'])
    test_sequence = shuffle_test_stimuli_dataframe(stimuli_df)
    test_session_start = now_time()
    test_intro = visual.TextStim(
        win = win,
        text = "Now you will be tested on what you learned about the aliens. For each trial, you will be asked a question about one of the aliens. Try to answer as accurately as possible. There will be 4 blocks of test. You will take one minute break after each block. \n\nPress any key to start.",
        color = 'white',
        height = 30,
        wrapWidth = 1.5
    )
    test_intro.draw()
    win.flip()
    event.waitKeys()
    block_start = now_time()
    block_no = 1
    for i in range(len(test_sequence)):
        if i == 8 or i == 16 or i == 24:
            block_end = now_time()
            block_duration = block_end - block_start
            for trial in memory_test_data:
                trial['test_block_end'] = block_end
                trial['test_block_duration'] = block_duration
            break_screen(TEST_BLOCK_BREAK_TIME)
            break_end_screen()
            block_start = now_time()
            block_no += 1
        trial_info_dic = test_sequence[i]
        trial_no = i + 1
        run_memory_test_trial(memory_test_data, block_no, block_start, trial_no, trial_info_dic)
        if i < len(test_sequence) - 1 and i not in [7, 15, 23]:
            blank_screen_present(TEST_INTER_TRIAL_TIME)
    test_session_end = now_time()
    test_session_duration = test_session_end - test_session_start
    for trial in memory_test_data:
        trial['test_session_start'] = test_session_start
        trial['test_session_end'] = test_session_end
        trial['test_session_duration'] = test_session_duration
    write_data_csv(f"test_data_participant_{exp_info['participant_id']}.csv", memory_test_data)



def run_memory_test_trial(data, block_no, block_start, trial_no, trial_info):
    trial_start_time = now_time()
    fix_start, fix_end, fix_duration = display_fixation_cross()
    trial_data = {
        "test_block_no": block_no,
        "test_block_start": block_start,
        "test_trial_no": trial_no,
        'test_fixation_start': fix_start,
        'test_fixation_end': fix_end,
        'test_fixation_duration': fix_duration,
        'trial_start_time': trial_start_time
    }
    trial_data.update(exp_info)
    test_content = trial_info['testContent']
    if test_content == 'planet':
        planet_test_screen(trial_no, trial_info, trial_data)
    elif test_content == 'color':
        color_test_screen(trial_no, trial_info, trial_data)
    elif test_content == 'residence':
        residence_test_screen(trial_no, trial_info, trial_data)
    data.append(trial_data)      


def generate_generalization_materials(rows, seed=None):
    """
    Generate generalization dictionaries from a list of dictionaries.

    Assumptions
    -----------
    - rows is a list of dictionaries.
    - All dictionaries share the same 'condition' value: 'C', 'R', or 'N'.
    - Each dictionary contains at least:
        'condition', 'planet', 'alien_folder', 'color', 'residence'
    - 'color' and 'residence' are circular angle values in degrees.
    - Angles such as 0.1 and 359.9 are treated as close.

    Returns
    -------
    list of dict
        A list of generated dictionaries.
        For each planet group, four dictionaries are generated.
    """
    rng = random.Random(seed)

    # -----------------------------
    # Circular helper functions
    # -----------------------------

    def wrap_angle(x):
        """Wrap angle to [0, 360)."""
        return x % 360

    def circular_distance(a, b):
        """Smallest circular distance between two angles."""
        diff = abs((a - b) % 360)
        return min(diff, 360 - diff)

    def circular_signed_delta(a, b, direction):
        """
        Distance from a to b in a chosen direction.
        direction='clockwise' or direction='counterclockwise'
        """
        if direction == "clockwise":
            return (b - a) % 360
        else:
            return (a - b) % 360

    def circular_mean(values):
        """Mean of circular angle values."""
        radians = [math.radians(float(v)) for v in values]
        sin_sum = sum(math.sin(r) for r in radians)
        cos_sum = sum(math.cos(r) for r in radians)

        if sin_sum == 0 and cos_sum == 0:
            return 0

        mean_angle = math.degrees(math.atan2(sin_sum, cos_sum))
        return wrap_angle(mean_angle)

    def circular_quantiles(values, quantiles=(1 / 3, 2 / 3)):
        """
        Approximate circular quantiles.

        Values are unwrapped around the largest circular gap,
        then ordinary linear quantiles are computed.
        """
        values = sorted(wrap_angle(float(v)) for v in values)

        if len(values) == 1:
            return [values[0] for _ in quantiles]

        gaps = []
        for i in range(len(values)):
            current = values[i]
            nxt = values[(i + 1) % len(values)]
            gap = (nxt - current) % 360
            gaps.append(gap)

        cut_index = gaps.index(max(gaps))
        start_index = (cut_index + 1) % len(values)

        ordered = values[start_index:] + values[:start_index]

        unwrapped = [ordered[0]]
        for v in ordered[1:]:
            while v < unwrapped[-1]:
                v += 360
            unwrapped.append(v)

        result = []
        n = len(unwrapped)

        for q in quantiles:
            pos = q * (n - 1)
            low = int(math.floor(pos))
            high = int(math.ceil(pos))

            if low == high:
                value = unwrapped[low]
            else:
                weight = pos - low
                value = unwrapped[low] * (1 - weight) + unwrapped[high] * weight

            result.append(wrap_angle(value))

        return result

    def closest_row(rows_, key, value):
        """Find row whose key value is circularly closest to value."""
        return min(
            rows_,
            key=lambda r: circular_distance(float(r[key]), value)
        )

    def furthest_value(rows_, key, value):
        """Find value in rows whose key value is circularly furthest from value."""
        return float(
            max(
                rows_,
                key=lambda r: circular_distance(float(r[key]), value)
            )[key]
        )

    def circular_jitter(value, amount):
        return wrap_angle(value + amount)

    def make_jitter():
        return rng.uniform(-3, 3)

    # -----------------------------
    # Group rows by planet
    # -----------------------------

    planet_groups = defaultdict(list)

    for row in rows:
        planet_groups[row["planet"]].append(row)

    planets = list(planet_groups.keys())

    if not rows:
        return []

    overall_condition = rows[0]["condition"]

    # -----------------------------
    # Determine generation rule
    # -----------------------------

    if overall_condition in ["C", "R"]:
        planet_rules = {
            planet: overall_condition
            for planet in planets
        }

    elif overall_condition == "N":
        shuffled_planets = planets[:]
        rng.shuffle(shuffled_planets)

        half = len(shuffled_planets) // 2

        planet_rules = {}

        for planet in shuffled_planets[:half]:
            planet_rules[planet] = "C"

        for planet in shuffled_planets[half:]:
            planet_rules[planet] = "R"

    else:
        raise ValueError("condition must be 'C', 'R', or 'N'")

    # -----------------------------
    # Helper: neighbor group by direction
    # -----------------------------

    def group_center_color(group_rows):
        return circular_mean([float(r["color"]) for r in group_rows])

    planet_centers = {
        planet: group_center_color(group_rows)
        for planet, group_rows in planet_groups.items()
    }

    def neighbor_planet(current_planet, direction):
        """
        Find neighboring planet in color space in the chosen direction.
        """
        current_center = planet_centers[current_planet]

        candidates = [
            p for p in planets
            if p != current_planet
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda p: circular_signed_delta(
                current_center,
                planet_centers[p],
                direction
            )
        )

    def closest_boundary_values(current_rows, neighbor_rows, direction):
        """
        Find the two closest color values between current group
        and neighbor group in the specified circular direction.
        """
        best_current = None
        best_neighbor = None
        best_distance = float("inf")

        for current_row in current_rows:
            c = float(current_row["color"])

            for neighbor_row in neighbor_rows:
                n = float(neighbor_row["color"])

                d = circular_signed_delta(c, n, direction)

                if 0 < d < best_distance:
                    best_distance = d
                    best_current = c
                    best_neighbor = n

        if best_current is None:
            for current_row in current_rows:
                c = float(current_row["color"])

                for neighbor_row in neighbor_rows:
                    n = float(neighbor_row["color"])

                    d = circular_distance(c, n)

                    if d < best_distance:
                        best_distance = d
                        best_current = c
                        best_neighbor = n

        return best_current, best_neighbor, best_distance

    # -----------------------------
    # Generate one planet group
    # -----------------------------

    def generate_for_planet(planet, group_rows, direction, gen_rule):
        generated = []

        color_values = [float(r["color"]) for r in group_rows]
        residence_values = [float(r["residence"]) for r in group_rows]

        alien_folder = group_rows[0]["alien_folder"]

        base_info = {
            "direction": direction,
            "planet": planet,
            "alien_folder": alien_folder,
            "gen_rule": gen_rule
        }

        # -------------------------
        # center-center
        # -------------------------

        center_trial = dict(base_info)
        center_trial.update({
            "gen_condition": "center-center",
            "new_color": circular_mean(color_values),
            "new_residence": circular_mean(residence_values),
            "percentile": 1 / 2,
            "jitter": 0
        })

        generated.append(center_trial)

        # -------------------------
        # inCluster-close / inCluster-far
        # -------------------------

        if gen_rule == "C":
            q1, q2 = circular_quantiles(
                color_values,
                quantiles=(1 / 3, 2 / 3)
            )
        else:
            q1, q2 = circular_quantiles(
                residence_values,
                quantiles=(1 / 3, 2 / 3)
            )

        percentiles = [1 / 3, 2 / 3]
        quantile_values = [q1, q2]

        paired = list(zip(percentiles, quantile_values))
        rng.shuffle(paired)

        for gen_condition, pair in zip(
            ["inCluster-close", "inCluster-far"],
            paired
        ):
            percentile, quantile_value = pair
            jitter = make_jitter()

            trial = dict(base_info)
            trial["gen_condition"] = gen_condition
            trial["percentile"] = percentile
            trial["jitter"] = jitter

            if gen_rule == "C":
                new_color = quantile_value

                nearest = closest_row(group_rows, "color", new_color)
                extracted_residence = float(nearest["residence"])

                if gen_condition == "inCluster-close":
                    new_residence = circular_jitter(
                        extracted_residence,
                        jitter
                    )
                else:
                    far_residence = furthest_value(
                        group_rows,
                        "residence",
                        extracted_residence
                    )
                    new_residence = circular_jitter(
                        far_residence,
                        jitter
                    )

                trial["new_color"] = wrap_angle(new_color)
                trial["new_residence"] = wrap_angle(new_residence)

            elif gen_rule == "R":
                new_residence = quantile_value

                nearest = closest_row(group_rows, "residence", new_residence)
                extracted_color = float(nearest["color"])

                if gen_condition == "inCluster-close":
                    new_color = circular_jitter(
                        extracted_color,
                        jitter
                    )
                else:
                    far_color = furthest_value(
                        group_rows,
                        "color",
                        extracted_color
                    )
                    new_color = circular_jitter(
                        far_color,
                        jitter
                    )

                trial["new_color"] = wrap_angle(new_color)
                trial["new_residence"] = wrap_angle(new_residence)

            generated.append(trial)

        # -------------------------
        # outCluster-center
        # -------------------------

        out_trial = dict(base_info)
        out_trial["gen_condition"] = "outCluster-center"
        out_trial["percentile"] = 4 / 3

        jitter = make_jitter()
        out_trial["jitter"] = jitter

        neighbor = neighbor_planet(planet, direction)

        if neighbor is None:
            interval_one_third = 0
            boundary_color = circular_mean(color_values)
        else:
            neighbor_rows = planet_groups[neighbor]

            boundary_color, _, interval = closest_boundary_values(
                group_rows,
                neighbor_rows,
                direction
            )

            interval_one_third = interval / 3

        if gen_rule == "C":
            if direction == "clockwise":
                new_color = boundary_color + interval_one_third + jitter
            else:
                new_color = boundary_color - interval_one_third + jitter

            new_residence = circular_mean(residence_values)

            out_trial["new_color"] = wrap_angle(new_color)
            out_trial["new_residence"] = wrap_angle(new_residence)

        elif gen_rule == "R":
            residence_centers = {
                p: circular_mean([float(r["residence"]) for r in rs])
                for p, rs in planet_groups.items()
            }

            current_center = residence_centers[planet]

            candidate_planets = [
                p for p in planets
                if p != planet
            ]

            if not candidate_planets:
                boundary_residence = circular_mean(residence_values)
                interval_one_third_r = 0
            else:
                neighbor_r = min(
                    candidate_planets,
                    key=lambda p: circular_signed_delta(
                        current_center,
                        residence_centers[p],
                        direction
                    )
                )

                best_current = None
                best_distance = float("inf")

                for current_row in group_rows:
                    c = float(current_row["residence"])

                    for neighbor_row in planet_groups[neighbor_r]:
                        n = float(neighbor_row["residence"])

                        d = circular_signed_delta(c, n, direction)

                        if 0 < d < best_distance:
                            best_distance = d
                            best_current = c

                if best_current is None:
                    boundary_residence = circular_mean(residence_values)
                    interval_one_third_r = 0
                else:
                    boundary_residence = best_current
                    interval_one_third_r = best_distance / 3

            if direction == "clockwise":
                new_residence = boundary_residence + interval_one_third_r + jitter
            else:
                new_residence = boundary_residence - interval_one_third_r + jitter

            new_color = circular_mean(color_values)

            out_trial["new_color"] = wrap_angle(new_color)
            out_trial["new_residence"] = wrap_angle(new_residence)

        generated.append(out_trial)

        return generated

    # -----------------------------
    # Generate trials for all planets
    # -----------------------------

    generated_by_planet = {}

    direction = rng.choice(["clockwise", "counterclockwise"])

    for planet in planets:
        group_rows = planet_groups[planet]
        gen_rule = planet_rules[planet]

        generated_by_planet[planet] = generate_for_planet(
            planet=planet,
            group_rows=group_rows,
            direction=direction,
            gen_rule=gen_rule
        )

    return generated_by_planet

def generate_generalization_sequence(materials, seed=None):
    stimuli = add_alien(materials, ['list_1', 'list_2', 'list_3', 'list_4'], ALIEN_PATH_GEN)
    sequence = make_shuffled_list(stimuli)
    return sequence

def generalization_test_session(stimuli):
    generalization_session_data = []
    gen_stimuli = generate_generalization_materials(stimuli)
    gen_sequence = generate_generalization_sequence(gen_stimuli)
    gen_session_start = now_time()
    gen_intro = visual.TextStim(
        win = win,
        text = "Now you will be tested on a new alien that you haven't seen before. Based on what you learned about the previous aliens, try to guess where this new alien lives. \n\nPress any key to start.",
        color = 'white',
        height = 30,
        wrapWidth = 1.5
    )
    gen_intro.draw()
    win.flip()
    event.waitKeys()
 
    for i in range(len(gen_sequence)):
        trial_info_dic = gen_sequence[i]
        trial_no = i + 1
        trial_data = run_generalization_trial(trial_no, gen_session_start, trial_info_dic)
        generalization_session_data.append(trial_data)
        if i < len(gen_sequence) - 1:
            blank_screen_present(TEST_INTER_TRIAL_TIME)
    gen_session_end = now_time()
    for trial_data in generalization_session_data:
        trial_data['gen_session_end'] = gen_session_end
        trial_data['gen_session_duration'] = gen_session_end - gen_session_start
    write_data_csv('generalization_participant_' + exp_info['participant_id'] + '.csv', generalization_session_data)


def run_generalization_trial(trial_no, gen_session_start, alien_info):
    trial_start_time = now_time()   
    fix_start, fix_end, fix_duration = display_fixation_cross()
    trial_data = {
        'gen_session_start': gen_session_start,
        'gen_trial_no': trial_no,
        'gen_trial_start_time': trial_start_time,
        'gen_fixation_start': fix_start,
        'gen_fixation_end': fix_end,
        'gen_fixation_duration': fix_duration
    }
    trial_data.update(exp_info)
    display_generalization_screen(alien_info, trial_data)
    return trial_data

def display_generalization_screen(alien_info, trial_data):
    alien_folder = alien_info['alien_folder']
    alien = alien_info['alien']
    alien_color = alien_info['new_color']
    alien_residence = alien_info['new_residence']
    alien_planet = alien_info['planet']
    direction = alien_info['direction']
    percentile = alien_info['percentile']
    jitter = alien_info['jitter']    
    gen_condition = alien_info['gen_condition']
    gen_rule = alien_info['gen_rule']
    alien_pos = TEST_ALIEN_POS
    fill_alien_image = ALIEN_PATH_GEN + alien_folder + '/fill_layer' + alien
    fill_stim = alien_fill_image(fill_alien_image, alien_pos)
    update_alien_fill_color(fill_stim, alien_color)
    fill_stim.draw()
    outline_alien_image = ALIEN_PATH_GEN + alien_folder + 'outline_layer' + alien
    outline_stim = alien_outline_image(outline_alien_image, alien_pos)
    outline_stim.draw()
    residence_ring_stim = residence_ring(alien_pos)
    residence_ring_stim.draw()
    residence_bar_stim = residence_ring_bar()
    update_residence_bar(alien_pos, residence_bar_stim, alien_residence)
    residence_bar_stim.draw()
    test_question_stim = visual.TextStim(
        text = 'Where does this alien most likely to live?',
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
    stimuli_start_time = now_time()
    mouse = event.Mouse(visible=True)
    while True:
        fill_stim.draw()
        outline_stim.draw()
        test_question_stim.draw()
        residence_ring_stim.draw()
        residence_bar_stim.draw()
        draw_test_labels(labels_stim)
        win.flip()
        mouse.clickRest()
        if mouse.getPress()[0]:
            click_label, selected_label = detect_label_selection(mouse, labels_stim)
            if click_label:
                selected_planet = selected_label.text
                rt = rt_clock.getTime()
                trial_end_time = now_time()
                saveData(['gen_stimuli_start_time', 'trial_end_time', 'selected_planet', 'rt', 'correct_planet', 'new_color', 'new_residence', 'direction', 'percentile', 'jitter', 'gen_condition', 'gen_rule'], [stimuli_start_time, trial_end_time, selected_planet, rt, alien_planet, alien_color, alien_residence, direction, percentile, jitter, gen_condition, gen_rule], trial_data)
                break
    

def color_anchor_trial(color, trial_data):
    test_color = color['color']
    color_name = color['color_name']
    fill_stim = visual.Rect(
        win = win,
        width = 200,
        height = 200,
        pos = TEST_ALIEN_POS,
    )
    test_question_stim = visual.TextStim(
        text = 'Please select the color that matches what you think of' + color_name,
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
        lineColor = COLOR_BAR_COLOR,
        lineWidth = COLOR_BAR_WIDTH
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
    test_question_stim.draw()
    draw_color_ring(ring_sectors)
    outer_outline.draw()
    inner_outline.draw()
    selector_line.draw()
    submit_rect.draw()
    submit_text.draw()
    mouse = event.Mouse(visible=True)
    rt_clock = core.Clock()
    win.callOnFlip(rt_clock.reset)
    win.flip()
    trial_start_time = now_time()
    submitted = False
    current_hue_idx = initial_hue/COLOR_RING_UNIT
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
            elif mouse_on_bar(pt, selector_line.start, selector_line.end) and not point_in_rect(pt, TEST_SUBMIT_POS, TEST_SUBMIT_SIZE[0], TEST_SUBMIT_SIZE[1]): 
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
    saveData(['trial_start_time', 'trial_end_time', 'selected_hue', 'rt', 'test_color', 'color_name'], [trial_start_time, trial_end_time, selected_hue, rt, test_color, color_name], trial_data)


def color_anchor_session():
    colors = [
        {'color': 0, 'color_name': 'red'},
        {'color': 90, 'color_name': 'yellow'},
        {'color': 180, 'color_name': 'green'},
        {'color': 270, 'color_name': 'blue'}]
    color_anchor_data = []
    color_anchor_session_start = now_time()
    color_anchor_intro = visual.TextStim(
        win = win,
        text = "In this part, you will be asked to select the color that matches your knowledge of each color. \n\nPress any key to start.",
        color = 'white',
        height = 30,
        wrapWidth = 1.5
    )
    color_anchor_intro.draw()
    win.flip()
    event.waitKeys()
    random.shuffle(colors)
    for i in range(len(colors)):
        trial_no = i + 1
        trial_data = {
            'color_anchor_session_start': color_anchor_session_start,
            'color_anchor_trial_no': trial_no}
        trial_data.update(exp_info)
        color_anchor_trial(colors[i], trial_data)
        color_anchor_data.append(trial_data)
        if i < len(colors) - 1:
            blank_screen_present(TEST_INTER_TRIAL_TIME)
    color_anchor_session_end = now_time()
    for trial_data in color_anchor_data:
        trial_data['color_anchor_session_end'] = color_anchor_session_end
        trial_data['color_anchor_session_duration'] = color_anchor_session_end - color_anchor_session_start
    write_data_csv(f"color_anchor_participant_{exp_info['participant_id']}.csv", color_anchor_data)

def run_post_exp_survey(win):
    """
    Run a post-experiment survey in PsychoPy.

    Parameters
    ----------
    win : psychopy.visual.Window
        Existing PsychoPy window.

    Returns
    -------
    dict
        Survey answers saved as a dictionary.
    """

    mouse = event.Mouse(win=win)

    # -----------------------------
    # Survey state
    # -----------------------------

    answers = {
        "thought_old_aliens_between_sessions": {
            "yes_no": None,
            "explanation": ""
        },
        "found_regularity_old_aliens": {
            "yes_no": None,
            "explanation": ""
        },
        "used_strategy_memory_test": {
            "yes_no": None,
            "explanation": ""
        }
    }

    active_textbox_key = None

    # -----------------------------
    # Helper functions
    # -----------------------------

    def is_complete():
        """
        Confirm is enabled only when:
        - all yes/no questions are answered
        - if yes, explanation is non-empty
        """
        for key, value in answers.items():
            if value["yes_no"] is None:
                return False

            if value["yes_no"] == "yes" and value["explanation"].strip() == "":
                return False

        return True

    def point_on_button(pos, rect):
        """
        Check whether mouse position is inside a rectangle.

        rect = {
            "x": center x,
            "y": center y,
            "w": width,
            "h": height
        }
        """
        x, y = pos
        return (
            rect["x"] - rect["w"] / 2 <= x <= rect["x"] + rect["w"] / 2
            and
            rect["y"] - rect["h"] / 2 <= y <= rect["y"] + rect["h"] / 2
        )

    def draw_button(label, rect, selected=False, enabled=True):
        """
        Draw a button and return its clickable rect.
        """
        if selected:
            fill_color = "dodgerblue"
            line_color = "white"
            text_color = "white"
        elif enabled:
            fill_color = "gray25"
            line_color = "white"
            text_color = "white"
        else:
            fill_color = "gray15"
            line_color = "gray40"
            text_color = "gray50"

        box = visual.Rect(
            win,
            width=rect["w"],
            height=rect["h"],
            pos=(rect["x"], rect["y"]),
            fillColor=fill_color,
            lineColor=line_color,
            lineWidth=2
        )

        text = visual.TextStim(
            win,
            text=label,
            pos=(rect["x"], rect["y"]),
            color=text_color,
            height=0.035,
            wrapWidth=rect["w"] * 0.9
        )

        box.draw()
        text.draw()

    def draw_textbox(rect, text_value, active=False, enabled=True):
        """
        Draw a text input box.
        """
        if active:
            line_color = "yellow"
        else:
            line_color = "white"

        if not enabled:
            fill_color = "gray10"
            text_color = "gray45"
            line_color = "gray35"
        else:
            fill_color = "gray20"
            text_color = "white"

        box = visual.Rect(
            win,
            width=rect["w"],
            height=rect["h"],
            pos=(rect["x"], rect["y"]),
            fillColor=fill_color,
            lineColor=line_color,
            lineWidth=2
        )

        display_text = text_value
        if active:
            display_text += "|"

        if display_text.strip() == "":
            display_text = "Type your answer here..." if enabled else ""

        text = visual.TextStim(
            win,
            text=display_text,
            pos=(rect["x"], rect["y"]),
            color=text_color,
            height=0.03,
            wrapWidth=rect["w"] * 0.92,
            alignText="left",
            anchorHoriz="center",
            anchorVert="center"
        )

        box.draw()
        text.draw()

    # -----------------------------
    # Layout
    # -----------------------------

    title = visual.TextStim(
        win,
        text="Post-experiment Survey",
        pos=(0, 0.42),
        color="white",
        height=0.055,
        bold=True
    )

    instruction = visual.TextStim(
        win,
        text="Please answer all questions. If you select Yes, please explain.",
        pos=(0, 0.36),
        color="white",
        height=0.032
    )

    questions = [
        {
            "key": "thought_old_aliens_between_sessions",
            "text": "1. Did you think of any old aliens in the time between learning sessions and memory test?",
            "y": 0.22
        },
        {
            "key": "found_regularity_old_aliens",
            "text": "2. Did you find any regularity from old aliens?",
            "y": 0.02
        },
        {
            "key": "used_strategy_memory_test",
            "text": "3. Did you use any strategy to complete the memory test?",
            "y": -0.18
        }
    ]

    clickable_items = {}

    for q in questions:
        key = q["key"]
        y = q["y"]

        clickable_items[(key, "yes")] = {
            "x": -0.22,
            "y": y - 0.055,
            "w": 0.13,
            "h": 0.055
        }

        clickable_items[(key, "no")] = {
            "x": -0.06,
            "y": y - 0.055,
            "w": 0.13,
            "h": 0.055
        }

        clickable_items[(key, "textbox")] = {
            "x": 0.37,
            "y": y - 0.055,
            "w": 0.58,
            "h": 0.07
        }

    confirm_rect = {
        "x": 0,
        "y": -0.42,
        "w": 0.28,
        "h": 0.07
    }

    # -----------------------------
    # Main loop
    # -----------------------------

    submitted = False

    while not submitted:
        title.draw()
        instruction.draw()

        # Draw each question
        for q in questions:
            key = q["key"]
            y = q["y"]

            q_text = visual.TextStim(
                win,
                text=q["text"],
                pos=(-0.75, y),
                color="white",
                height=0.032,
                wrapWidth=1.45,
                alignText="left",
                anchorHoriz="left"
            )
            q_text.draw()

            # Yes button
            draw_button(
                "Yes",
                clickable_items[(key, "yes")],
                selected=answers[key]["yes_no"] == "yes",
                enabled=True
            )

            # No button
            draw_button(
                "No",
                clickable_items[(key, "no")],
                selected=answers[key]["yes_no"] == "no",
                enabled=True
            )

            # Textbox enabled only if Yes
            textbox_enabled = answers[key]["yes_no"] == "yes"

            draw_textbox(
                clickable_items[(key, "textbox")],
                answers[key]["explanation"],
                active=active_textbox_key == key,
                enabled=textbox_enabled
            )

        # Draw confirm button
        confirm_enabled = is_complete()

        draw_button(
            "Confirm",
            confirm_rect,
            selected=False,
            enabled=confirm_enabled
        )

        if not confirm_enabled:
            warning = visual.TextStim(
                win,
                text="Answer all required questions before submitting.",
                pos=(0, -0.49),
                color="gray70",
                height=0.027
            )
            warning.draw()

        win.flip()

        # -------------------------
        # Keyboard input
        # -------------------------

        keys = event.getKeys()

        for keypress in keys:
            if keypress == "escape":
                core.quit()

            if active_textbox_key is not None:
                if keypress == "backspace":
                    answers[active_textbox_key]["explanation"] = (
                        answers[active_textbox_key]["explanation"][:-1]
                    )

                elif keypress == "space":
                    answers[active_textbox_key]["explanation"] += " "

                elif keypress == "return":
                    active_textbox_key = None

                elif len(keypress) == 1:
                    answers[active_textbox_key]["explanation"] += keypress

                # Basic punctuation support
                elif keypress == "period":
                    answers[active_textbox_key]["explanation"] += "."

                elif keypress == "comma":
                    answers[active_textbox_key]["explanation"] += ","

                elif keypress == "minus":
                    answers[active_textbox_key]["explanation"] += "-"

                elif keypress == "apostrophe":
                    answers[active_textbox_key]["explanation"] += "'"

                elif keypress == "slash":
                    answers[active_textbox_key]["explanation"] += "/"

        # -------------------------
        # Mouse input
        # -------------------------

        if mouse.getPressed()[0]:
            mouse_pos = mouse.getPos()

            # Check Yes/No buttons
            for q in questions:
                key = q["key"]

                if point_on_button(mouse_pos, clickable_items[(key, "yes")]):
                    answers[key]["yes_no"] = "yes"
                    active_textbox_key = key

                elif point_on_button(mouse_pos, clickable_items[(key, "no")]):
                    answers[key]["yes_no"] = "no"
                    answers[key]["explanation"] = ""
                    if active_textbox_key == key:
                        active_textbox_key = None

                elif point_on_button(mouse_pos, clickable_items[(key, "textbox")]):
                    if answers[key]["yes_no"] == "yes":
                        active_textbox_key = key

            # Check Confirm button
            if point_on_button(mouse_pos, confirm_rect):
                if is_complete():
                    submitted = True

            # Wait until mouse is released to avoid repeated clicks
            while mouse.getPressed()[0]:
                core.wait(0.01)

    save_survey_answers_to_csv(answers, f"post_exp_survey_participant_{exp_info['participant_id']}.csv")

def save_survey_answers_to_csv(survey_answers, filename):
    """
    Save post-experiment survey answers to a CSV file.

    Parameters
    ----------
    survey_answers : dict
        Dictionary returned by run_post_exp_survey().

    filename : str
        Output CSV filename.
    """

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Header
        writer.writerow([
            "question_key",
            "yes_no",
            "explanation"
        ])

        # Rows
        for question_key, response in survey_answers.items():
            writer.writerow([
                question_key,
                response["yes_no"],
                response["explanation"]
            ])









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

def mouse_on_bar(mouse_pos, line_start, line_end, tolerance=5):
    mx, my = mouse_pos
    x1, y1 = line_start
    x2, y2 = line_end
    line_len = math.hypot(x2 - x1, y2 - y1)
    if line_len == 0:
        return math.hypot(mx - x1, my - y1) <= tolerance
    t = ((mx - x1) * (x2 - x1) + (my - y1) * (y2 - y1)) / (line_len ** 2)
    t = max(0, min(1, t))
    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)
    return math.hypot(mx - closest_x, my - closest_y) <= tolerance


## This function converts the selection (polar system) to the rgb value that can be used for Psychopy drawing
def update_selected_color_from_angle(selector_angle, hue_rgb_psy, ring_rotation = 0):
  
    selected_hue = (selector_angle - ring_rotation) % 360

    hue_idx = np.round(selected_hue,1) % 360
    current_hue_idx = hue_idx

    selected_rgb = hue_rgb_psy[hue_idx/COLOR_RING_UNIT]
    return current_hue_idx, selected_rgb

def update_residence_bar(ring_center, bar, angle):
    length = RESIDENCE_BAR_LENGTH/2
    # Bar midpoint exactly on ring
    p1 = pol_to_cart(angle, RESIDENCE_RING_RADIUS + length) + np.array(ring_center)
    p2 = pol_to_cart(angle, RESIDENCE_RING_RADIUS - length) + np.array(ring_center)
    bar.start = p1
    bar.end = p2

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
    n_units = int(round(360 / COLOR_RING_UNIT))

    hue_rgb_psy = np.array([
        lch_to_psychopy_rgb(L_VALUE, C_VALUE, i * COLOR_RING_UNIT)
        for i in range(n_units)
    ])

    return hue_rgb_psy

## This function coverts the time_duration to frames, to allowed for more accurate time control
def time_to_frame(time_in_seconds):
    return math.ceil(time_in_seconds * fresh_rate)

## This function generates the initial fill image
def fill_image(image_path):
    fill_path = image_path  ## set the path to the fill layer images. These images should be created in advance and should be the same size as the alien images. The fill layer images should have a transparent background and the alien shape filled with white (255, 255, 255) in rgb space. This will allow us to use the color parameter in the ImageStim to change the color of the alien images during the experiment.
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
def outline_image(image_path):
    outline_path = image_path  ## set the path to the outline layer images. These images should be created in advance and should be the same size as the alien images. The outline layer images should have a transparent background and the alien shape filled with white (255, 255, 255) in rgb space. This will allow us to use the color parameter in the ImageStim to change the color of the alien images during the experiment.
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

######### helper functions for generate sampled points

def wrap_angle_deg(angle):
    """
    Wrap angle to [0, 360).
    """
    return angle % 360.0


def angular_distance_deg(a, b):
    """
    Compute the smallest circular angular distance between two angles.
    """
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def quantize_angle(angle, unit=0.1, wrap=True):
    """
    Quantize an angle or value to the nearest unit.
    """
    value = round(angle / unit) * unit
    value = round(value, 10)

    if wrap:
        value = wrap_angle_deg(value)

    return value

def signed_angular_offset_deg(angle, center):
    """
    Signed shortest angular offset from center to angle, in degrees.

    Returns a value in [-180, 180).

    Examples
    --------
    center = 90
    angle = 82  -> -8
    angle = 98  -> +8
    """
    return ((angle - center + 180.0) % 360.0) - 180.0

def check_within_set_mirror_symmetry(
    internal_sample,
    symmetry_pair_tolerance_deg=1.0,
    verbose=False,
):
    """
    Reject a set if a left point and a right point are approximately
    mirror-symmetric around the jittered set center.

    Example rejected:
        center = 90
        left point = 82
        right point = 98

        offsets are -8 and +8.
        abs(offset_left + offset_right) = 0.
    """
    set_labels = internal_sample["set_labels"]
    sets = internal_sample["sets"]
    info = internal_sample["generated_info"]

    for label in set_labels:
        center = info[label]["jittered_mean_deg"]

        left_points = sets[label]["left_points"]
        right_points = sets[label]["right_points"]

        for lp in left_points:
            left_offset = signed_angular_offset_deg(lp, center)

            for rp in right_points:
                right_offset = signed_angular_offset_deg(rp, center)

                # Perfect mirror symmetry means:
                # left_offset + right_offset == 0
                symmetry_error = abs(left_offset + right_offset)

                if symmetry_error <= symmetry_pair_tolerance_deg:
                    if verbose:
                        print(
                            f"Failed within-set mirror symmetry for {label}: "
                            f"center={center}, "
                            f"left_point={lp}, right_point={rp}, "
                            f"left_offset={left_offset:.2f}, "
                            f"right_offset={right_offset:.2f}, "
                            f"symmetry_error={symmetry_error:.2f}, "
                            f"tolerance={symmetry_pair_tolerance_deg}"
                        )
                    return False

    return True

def sample_von_mises_deg(mean_deg, kappa, rng=None):
    """
    Sample one angle in degrees from a von Mises distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_rad = np.deg2rad(mean_deg)
    sample_rad = rng.vonmises(mean_rad, kappa)

    return wrap_angle_deg(np.rad2deg(sample_rad))


def circular_span_deg(points):
    """
    Compute the smallest circular arc span containing all points.
    """
    points = np.sort(np.asarray(points, dtype=float) % 360.0)

    if len(points) <= 1:
        return 0.0

    gaps = np.diff(points)
    wrap_gap = 360.0 - points[-1] + points[0]
    gaps = np.append(gaps, wrap_gap)

    largest_gap = np.max(gaps)

    return 360.0 - largest_gap


def sample_vonmises_with_min_spacing(
    center_deg,
    count,
    kappa,
    existing_points=None,
    min_dist_deg=3.0,
    unit=0.1,
    rng=None,
    max_tries=10000,
):
    """
    Sample angles around a center using a von Mises distribution,
    while enforcing minimum circular spacing from existing points.
    """
    if rng is None:
        rng = np.random.default_rng()

    if existing_points is None:
        existing_points = []

    accepted = []
    center_rad = np.deg2rad(center_deg)

    for _ in range(count):
        found = False

        for _try in range(max_tries):
            angle_rad = rng.vonmises(center_rad, kappa)
            angle_deg = quantize_angle(np.rad2deg(angle_rad), unit=unit)

            all_previous = existing_points + accepted

            if all(
                angular_distance_deg(angle_deg, prev) >= min_dist_deg
                for prev in all_previous
            ):
                accepted.append(angle_deg)
                found = True
                break

        if not found:
            raise RuntimeError(
                f"Could not sample {count} points around {center_deg}° "
                f"with min_dist_deg={min_dist_deg} and kappa={kappa}."
            )

    return accepted


def generate_circle_point_sets_internal(
    set_labels,
    base_means_deg,
    n,
    kappa=80.0,
    side_offset_deg=8.0,
    mean_jitter_deg=3.0,
    unit=0.1,
    rng=None,
    min_within_pairwise_dist_deg=3.0,
):
    """
    Generate circular point sets with internal metadata.

    This function is intended for internal validation only.

    Returns
    -------
    dict
        Internal sample containing:
        - set labels
        - all generated points
        - left/right points
        - jittered centers
        - left/right centers
    """
    if rng is None:
        rng = np.random.default_rng()

    base_means_deg = np.asarray(base_means_deg, dtype=float)

    if len(base_means_deg) != len(set_labels):
        raise ValueError("base_means_deg and set_labels must have the same length.")

    if n % 2 != 0:
        raise ValueError("n must be even so that half the points are left and half are right.")

    generated_sets = {}
    generated_info = {}

    for label, base_mean in zip(set_labels, base_means_deg):
        # Random jitter for this set center
        mean_jitter = rng.uniform(-mean_jitter_deg, mean_jitter_deg)
        mean_jitter = quantize_angle(mean_jitter, unit=unit, wrap=False)

        # Jittered center mean
        jittered_mean = quantize_angle(
            base_mean + mean_jitter,
            unit=unit,
            wrap=True,
        )

        # Left and right centers
        left_center = quantize_angle(
            jittered_mean - side_offset_deg,
            unit=unit,
            wrap=True,
        )

        right_center = quantize_angle(
            jittered_mean + side_offset_deg,
            unit=unit,
            wrap=True,
        )

        left_points = sample_vonmises_with_min_spacing(
            center_deg=left_center,
            count=n // 2,
            kappa=kappa,
            existing_points=[],
            min_dist_deg=min_within_pairwise_dist_deg,
            unit=unit,
            rng=rng,
            max_tries=10000,
        )

        right_points = sample_vonmises_with_min_spacing(
            center_deg=right_center,
            count=n - n // 2,
            kappa=kappa,
            existing_points=left_points,
            min_dist_deg=min_within_pairwise_dist_deg,
            unit=unit,
            rng=rng,
            max_tries=10000,
        )

        points = left_points + right_points

        generated_sets[label] = {
            "all_points": points,
            "left_points": left_points,
            "right_points": right_points,
        }

        generated_info[label] = {
            "base_mean_deg": float(base_mean),
            "mean_jitter_deg": mean_jitter,
            "jittered_mean_deg": jittered_mean,
            "left_center_deg": left_center,
            "right_center_deg": right_center,
        }

    internal_sample = {
        "set_labels": list(set_labels),
        "sets": generated_sets,
        "generated_info": generated_info,
        "parameters": {
            "base_means_deg": base_means_deg.tolist(),
            "n_per_set": n,
            "kappa": kappa,
            "side_offset_deg": side_offset_deg,
            "mean_jitter_deg": mean_jitter_deg,
            "unit": unit,
            "min_within_pairwise_dist_deg": min_within_pairwise_dist_deg,
        },
    }

    return internal_sample

def check_circle_point_sets_internal(
    internal_sample,
    min_within_pairwise_dist_deg=1.8,
    max_within_span_deg=45.0,
    min_between_set_dist_deg=30.0,
    min_center_dist_deg=70.0,
    verbose=False,
    avoid_within_set_symmetry=True,
    symmetry_pair_tolerance_deg=1.0
):
    """
    Check whether generated point sets meet requirements.

    This function expects the internal sample structure.
    """
    set_labels = internal_sample["set_labels"]
    sets = internal_sample["sets"]
    info = internal_sample["generated_info"]

    # 1. Check within-set pairwise distances
    for label in set_labels:
        points = sets[label]["all_points"]

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = angular_distance_deg(points[i], points[j])

                if dist < min_within_pairwise_dist_deg:
                    if verbose:
                        print(
                            f"Failed within-set pairwise distance for {label}: "
                            f"points {points[i]} and {points[j]}, "
                            f"distance={dist:.2f}, "
                            f"required>={min_within_pairwise_dist_deg}"
                        )
                    return False

        # 2. Check within-set span
        span = circular_span_deg(points)

        if span > max_within_span_deg:
            if verbose:
                print(
                    f"Failed within-set span for {label}: "
                    f"span={span:.2f}, "
                    f"max allowed={max_within_span_deg}"
                )
            return False

    # 3. Check between-set point separation
    for a in range(len(set_labels)):
        for b in range(a + 1, len(set_labels)):
            label_a = set_labels[a]
            label_b = set_labels[b]

            points_a = sets[label_a]["all_points"]
            points_b = sets[label_b]["all_points"]

            for p_a in points_a:
                for p_b in points_b:
                    dist = angular_distance_deg(p_a, p_b)

                    if dist < min_between_set_dist_deg:
                        if verbose:
                            print(
                                f"Failed between-set distance for {label_a} and {label_b}: "
                                f"points {p_a} and {p_b}, "
                                f"distance={dist:.2f}, "
                                f"required>={min_between_set_dist_deg}"
                            )
                        return False

    # 4. Check between-center distances
    for a in range(len(set_labels)):
        for b in range(a + 1, len(set_labels)):
            label_a = set_labels[a]
            label_b = set_labels[b]

            center_a = info[label_a]["jittered_mean_deg"]
            center_b = info[label_b]["jittered_mean_deg"]

            dist = angular_distance_deg(center_a, center_b)

            if dist < min_center_dist_deg:
                if verbose:
                    print(
                        f"Failed center distance for {label_a} and {label_b}: "
                        f"centers {center_a} and {center_b}, "
                        f"distance={dist:.2f}, "
                        f"required>={min_center_dist_deg}"
                    )
                return False
            
    # 5. Avoid mirror symmetry within each set
    if avoid_within_set_symmetry:
        if not check_within_set_mirror_symmetry(
            internal_sample=internal_sample,
            symmetry_pair_tolerance_deg=symmetry_pair_tolerance_deg,
            verbose=verbose,
        ):
            return False

    return True

def simplify_sample(internal_sample):
    """
    Convert an internal sample into the simple public return format.

    Returns
    -------
    dict
        Dictionary with labels as keys and generated point lists as values.
    """
    return {
        label: internal_sample["sets"][label]["all_points"]
        for label in internal_sample["set_labels"]
    }

###### helper function to pair color and residence
import random


def pair_two_dicts(dict1, dict2, field_names=None, seed=None):
    """
    Pair items from dict1 with items from dict2 without replacement.

    Parameters
    ----------
    dict1 : dict
        Dictionary with 4 keys. Each key maps to a list of 8 items.

    dict2 : dict
        Dictionary with 4 keys. Each key maps to a list of 8 items.

    field_names : list of str, optional
        A list of 4 strings used as output field names.

        Default:
            ["dict1_key", "dict1_item", "dict2_key", "dict2_item"]

        Example:
            ["cue_key", "cue_color", "target_key", "target_color"]

    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Result grouped by keys of dict1.
    """

    if field_names is None:
        field_names = ["dict1_key", "dict1_item", "dict2_key", "dict2_item"]

    if len(field_names) != 4:
        raise ValueError("field_names must contain exactly 4 strings.")

    if not all(isinstance(x, str) for x in field_names):
        raise ValueError("All field_names must be strings.")

    f_dict1_key, f_dict1_item, f_dict2_key, f_dict2_item = field_names

    rng = random.Random(seed)

    # Validate input
    if len(dict1) != 4:
        raise ValueError("dict1 must contain exactly 4 keys.")

    if len(dict2) != 4:
        raise ValueError("dict2 must contain exactly 4 keys.")

    for k, v in dict1.items():
        if len(v) != 8:
            raise ValueError(f"dict1[{k!r}] must contain exactly 8 items.")

    for k, v in dict2.items():
        if len(v) != 8:
            raise ValueError(f"dict2[{k!r}] must contain exactly 8 items.")

    # Shuffle copies of dict1 lists
    shuffled_dict1 = {
        k: rng.sample(v, len(v))
        for k, v in dict1.items()
    }

    # Shuffle copies of dict2 lists
    shuffled_dict2 = {
        k: rng.sample(v, len(v))
        for k, v in dict2.items()
    }

    dict1_keys = list(dict1.keys())
    dict2_keys = list(dict2.keys())

    # Split each dict2 list into 4 chunks of size 2.
    # Each dict1 key receives 2 items from each dict2 key.
    dict2_chunks = {}

    for k2 in dict2_keys:
        items = shuffled_dict2[k2]

        dict2_chunks[k2] = {
            k1: items[i * 2:(i + 1) * 2]
            for i, k1 in enumerate(dict1_keys)
        }

    result = {}

    for k1 in dict1_keys:
        items1 = shuffled_dict1[k1]

        candidate_pairs = []

        for k2 in dict2_keys:
            for item2 in dict2_chunks[k2][k1]:
                candidate_pairs.append({
                    f_dict2_key: k2,
                    f_dict2_item: item2
                })

        rng.shuffle(candidate_pairs)

        result[k1] = []

        for item1, pair_info in zip(items1, candidate_pairs):
            result[k1].append({
                f_dict1_key: k1,
                f_dict1_item: item1,
                f_dict2_key: pair_info[f_dict2_key],
                f_dict2_item: pair_info[f_dict2_item],
            })

    return result

def add_group_values(grouped_dict, new_key, values):
    """
    Add a new key-value pair to each dictionary inside a grouped result.

    Parameters
    ----------
    grouped_dict : dict
        Dictionary where each key maps to a list of dictionaries.

        Example:
            {
                "A": [{...}, {...}],
                "B": [{...}, {...}],
                "C": [{...}, {...}],
                "D": [{...}, {...}],
            }

    new_key : str
        New key to add to each inner dictionary.

    values : list
        A list of 4 values.

        The first value is assigned to the first group,
        the second value to the second group,
        the third value to the third group,
        the fourth value to the fourth group.

    Returns
    -------
    dict
        A modified copy of grouped_dict.
    """

    if len(grouped_dict) != 4:
        raise ValueError("grouped_dict must contain exactly 4 groups.")

    if len(values) != 4:
        raise ValueError("values must contain exactly 4 items.")

    if not isinstance(new_key, str):
        raise ValueError("new_key must be a string.")

    group_keys = list(grouped_dict.keys())

    result = {}

    for group_key, value in zip(group_keys, values):
        result[group_key] = []

        for item_dict in grouped_dict[group_key]:
            new_item_dict = item_dict.copy()
            new_item_dict[new_key] = value
            result[group_key].append(new_item_dict)

    return result


def regroup_by_key(input_dict, key="planet"):
    """
    Reorganize a dictionary from being grouped by original keys
    to being grouped by planet.

    Parameters
    ----------
    input_dict : dict
        Example:

        {
            "A": [
                {"group": "A", "value": 1, "planet": "Mars"},
                {"group": "A", "value": 2, "planet": "Venus"},
            ],
            "B": [
                {"group": "B", "value": 3, "planet": "Mars"},
                {"group": "B", "value": 4, "planet": "Jupiter"},
            ],
        }

    planet_key : str
        The key inside each small dictionary that stores the planet name.

    Returns
    -------
    dict
        Example:

        {
            "Mars": [
                {"group": "A", "value": 1, "planet": "Mars"},
                {"group": "B", "value": 3, "planet": "Mars"},
            ],
            "Venus": [
                {"group": "A", "value": 2, "planet": "Venus"},
            ],
            "Jupiter": [
                {"group": "B", "value": 4, "planet": "Jupiter"},
            ],
        }
    """

    regrouped = defaultdict(list)

    for group_records in input_dict.values():
        for record in group_records:
            planet = record[key]
            regrouped[planet].append(record)

    return dict(regrouped)


def pair_by_preassigned_planets(
    dict1,
    dict2,
    planets,
    field_names=None,
    seed=None,
):
    """
    Pair two 4x8 dictionaries using preassigned planets.

    Method
    ------
    1. For each group/list in dict1 and dict2:
       randomly assign planets so each planet appears exactly twice.

    2. Pair dict1 and dict2 items only when they have the same planet.

    3. Use a Latin-square rotation so that each dict1 group pairs with
       exactly 2 items from each dict2 group.

    Parameters
    ----------
    dict1 : dict
        Dictionary with 4 keys. Each key maps to 8 items.

    dict2 : dict
        Dictionary with 4 keys. Each key maps to 8 items.

    planets : list
        List of 4 planet values.

    field_names : list of str, optional
        Names for the output fields, excluding "planet".

        Default:
            [
                "dict1_group",
                "dict1_item",
                "dict2_group",
                "dict2_item",
            ]

    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Final result grouped by planet.
    """

    if field_names is None:
        field_names = [
            "dict1_group",
            "dict1_item",
            "dict2_group",
            "dict2_item",
        ]

    if len(field_names) != 4:
        raise ValueError("field_names must contain exactly 4 strings.")

    if "planet" in field_names:
        raise ValueError('"planet" is added automatically; do not include it in field_names.')

    if len(dict1) != 4:
        raise ValueError("dict1 must contain exactly 4 keys.")

    if len(dict2) != 4:
        raise ValueError("dict2 must contain exactly 4 keys.")

    if len(planets) != 4:
        raise ValueError("planets must contain exactly 4 items.")

    for k, v in dict1.items():
        if len(v) != 8:
            raise ValueError(f"dict1[{k!r}] must contain exactly 8 items.")

    for k, v in dict2.items():
        if len(v) != 8:
            raise ValueError(f"dict2[{k!r}] must contain exactly 8 items.")

    rng = random.Random(seed)

    f_d1_group, f_d1_item, f_d2_group, f_d2_item = field_names

    dict1_keys = list(dict1.keys())
    dict2_keys = list(dict2.keys())

    # Randomize group order for Latin-square mapping
    rng.shuffle(dict1_keys)
    rng.shuffle(dict2_keys)

    # --------------------------------------------------
    # Helper: assign planets to items in each group
    # --------------------------------------------------
    def assign_planets_to_dict(input_dict):
        assigned = {}

        for group_key, items in input_dict.items():
            shuffled_items = list(items)
            rng.shuffle(shuffled_items)

            planet_pool = []
            for planet in planets:
                planet_pool.extend([planet, planet])

            rng.shuffle(planet_pool)

            assigned[group_key] = {planet: [] for planet in planets}

            for item, planet in zip(shuffled_items, planet_pool):
                assigned[group_key][planet].append(item)

        return assigned

    assigned1 = assign_planets_to_dict(dict1)
    assigned2 = assign_planets_to_dict(dict2)

    # --------------------------------------------------
    # Pair using Latin-square rotation
    # --------------------------------------------------
    grouped_by_planet = {planet: [] for planet in planets}

    for p_idx, planet in enumerate(planets):
        for i, d1_key in enumerate(dict1_keys):
            d2_key = dict2_keys[(i + p_idx) % 4]

            d1_items = assigned1[d1_key][planet]
            d2_items = assigned2[d2_key][planet]

            rng.shuffle(d1_items)
            rng.shuffle(d2_items)

            for item1, item2 in zip(d1_items, d2_items):
                grouped_by_planet[planet].append({
                    f_d1_group: d1_key,
                    f_d1_item: item1,
                    f_d2_group: d2_key,
                    f_d2_item: item2,
                    "planet": planet,
                })

    # Shuffle rows within each planet group
    for planet in planets:
        rng.shuffle(grouped_by_planet[planet])

    return grouped_by_planet

def add_name(dic, male_name_list, female_name_list):
    random.shuffle(male_name_list)
    random.shuffle(female_name_list)
    male_idx = 0
    female_idx = 0
    for value in dic.values():
        random.shuffle(value)
        for i in range(len(value)):
            if i % 2 == 0:
                value[i]['name'] = female_name_list[female_idx]
                female_idx += 1
            elif i % 2 == 1:
                value[i]['name'] = male_name_list[male_idx]
                male_idx += 1
    return dic

def add_alien(dic, folder_names, path):
    learning_path = path
    random.shuffle(folder_names)
    folder_idx = 0
    for value in dic.values():
        image_paths = os.listdir(learning_path + folder_names[folder_idx])
        images = [f for f in os.listdir(image_paths) if os.path.isfile(os.path.join(image_paths, f))]
        random.shuffle(image_paths)
        for i in range(len(value)):
            value[i]['alien'] = images[i]
            value[i]['alien_folder'] = folder_names[folder_idx]
        folder_idx += 1
    return dic

def add_order_group(dic):
    for value in dic.values():
        random.shuffle(value)
        for i in range(len(value)):
            if i < len(value)/2:
                value[i]['practice_order_group'] = 1
            else:
                value[i]['practice_order_group'] = 2
    return dic

def add_others(dic, new_dic):
    for value in dic.values():
        for i in range(len(value)):
            value[i].update(new_dic)
    return dic

def create_stimuli_csv(data, file_name):
    """
    Save a dictionary of lists of dictionaries to a CSV file.

    Parameters
    ----------
    data : dict
        A dictionary where each value is a list of small dictionaries.

        Example:

        {
            "Mars": [
                {"name": "A1", "degree": 1, "planet": "Mars"},
                {"name": "B1", "degree": 2, "planet": "Mars"},
            ],
            "Venus": [
                {"name": "A2", "degree": 3, "planet": "Venus"},
                {"name": "B2", "degree": 4, "planet": "Venus"},
            ],
        }

    file_name : str
        Name of the CSV file to save.

        Example:
            "output.csv"

    Returns
    -------
    None
    """

    # Flatten all small dictionaries into one list
    rows = []
    save_file_name = 'stimuli_participant_' + file_name
    for group_list in data.values():
        rows.extend(group_list)

    # If there are no rows, create an empty file and return
    if not rows:
        with open(save_file_name, "w", newline="", encoding="utf-8") as f:
            pass
        return

    # Use the keys of the first small dictionary as CSV headers
    headers = list(rows[0].keys())

    with open(save_file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        writer.writeheader()
        writer.writerows(rows)


def make_shuffled_list(grouped, max_attempts=100000, seed=None):
    """
    grouped is a dictionary like:
    {
        "A": [dict1, dict2, ...],
        "B": [dict1, dict2, ...],
        "C": [dict1, dict2, ...],
        "D": [dict1, dict2, ...],
    }

    Returns a shuffled long list where no 3 consecutive dictionaries
    come from the same original group.
    """

    rng = random.Random(seed)

    # Add group information temporarily
    long_list = []
    for group_name, dict_list in grouped.items():
        for d in dict_list:
            long_list.append({
                "_group": group_name,
                "data": d
            })

    def is_valid(lst):
        for i in range(len(lst) - 2):
            if (
                lst[i]["_group"] == lst[i + 1]["_group"] ==
                lst[i + 2]["_group"]
            ):
                return False
        return True

    for _ in range(max_attempts):
        shuffled = long_list[:]
        rng.shuffle(shuffled)

        if is_valid(shuffled):
            # Remove temporary group labels before returning
            return [item["data"] for item in shuffled]

    raise ValueError("Could not find a valid shuffle.")

def generate_stimuli_for_block(dic, block, practice_order):
    for value in dic.values():
        for i in range(len(value)):
            if value[i]['practice_order_group'] == 1:
                value[i]['practice_order'] = practice_order[block-1][0]
            elif value[i]['practice_order_group'] == 2:
                value[i]['practice_order'] = practice_order[block-1][1]
    return dic

def write_data_csv(file_name, data):
    """
    Save a list of dictionaries to a CSV file.

    Parameters
    ----------
    file_name : str
        Name of the CSV file to save.

        Example:
            "output.csv"

    data : list of dict
        A list where each item is a dictionary representing a row.

        Example:
            [
                {"name": "A1", "degree": 1, "planet": "Mars"},
                {"name": "B1", "degree": 2, "planet": "Mars"},
                {"name": "A2", "degree": 3, "planet": "Venus"},
                {"name": "B2", "degree": 4, "planet": "Venus"},
            ]
    """
    if not data:
        with open(file_name, "w", newline="", encoding="utf-8") as f:
            pass
        return

    headers = list(data[0].keys())

    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)


# ============ Main Script ===========

exp_info = {
    "participant_id": "",
    "condition": ["C", "R", "N"], # C. color R. residence N. no rule
    "group": ["1", "2"], # 1. Immediate Test 2. Delayed Test
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

if exp_info['session'] == '1':
    ### Create color and residence stimuli
    initial_list= [0, 90, 180, 270]
    color_samples = generate_until_valid(['red','yellow','green','blue'],initial_list, n = 8,
        kappa=80.0,
        side_offset_deg=10.0,
        mean_jitter_deg=2,
        unit=0.1,
        min_within_pairwise_dist_deg=4,
        max_within_span_deg=70.0,
        min_between_set_dist_deg=30.0,
        min_center_dist_deg=60.0)

    residence_rotation = np.round(np.random.uniform(0,90),1)
    residence_initial = [(degree + residence_rotation) % 360 for degree in initial_list]
    residence_samples = generate_until_valid(['1st', '2nd', '3rd', '4th'], residence_initial, n = 8,
        kappa=80.0,
        side_offset_deg=10.0,
        mean_jitter_deg=2,
        unit=0.1,
        min_within_pairwise_dist_deg=4,
        max_within_span_deg=70.0,
        min_between_set_dist_deg=30.0,
        min_center_dist_deg=60.0)      

    random.shuffle(ALIEN_PLANETS)
    if exp_info['condition'] == 'C':
        paired_stimuli = pair_two_dicts(color_samples, residence_samples, field_names=['color_group', 'color_degree', 'residence_group', 'residence_degree'])
    elif exp_info['condition'] == 'R':
        paired_stimuli = pair_two_dicts(residence_samples, color_samples, field_names= ['residence_group', 'residence_degree', 'color_group', 'color_degree'])
    if exp_info['condition'] == 'C' or exp_info['condition'] == 'R':
        add_planet = add_group_values(paired_stimuli, 'planet', ALIEN_PLANETS)
        color_residence_planet = regroup_by_key(add_planet, 'planet')
    else:
        color_residence_planet = pair_by_preassigned_planets(color_samples, residence_samples, ALIEN_PLANETS, field_names=['color_group', 'color_degree', 'residence_group', 'residence_degree'])

    name_added = add_name(color_residence_planet, ALIEN_M_NAMES, ALIEN_F_NAMES)
    alien_added = add_alien(name_added,['list_1', 'list_2', 'list_3', 'list_4'], ALIEN_PATH_LEARNING)
    order_group_added = add_order_group(alien_added)
    full_stimuli = add_others(order_group_added,exp_info)
    create_stimuli_csv(full_stimuli, exp_info["participant_id"])                            


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


    ### Create practice order
    practice_order = generatePracticeOrder('planet', 'color', 'residence')
    random.shuffle(practice_order)

    ### Study blocks
    study_session_start = now_time()
    exp_info['study_session_start'] = study_session_start
    learning_data = []
    for i in range(N_BLOCKS):
        block = i+1
        study_block(learning_data, full_stimuli, block, practice_order)
        if block != N_BLOCKS:
            break_screen(BLOCK_BREAK_TIME)
            break_end_screen()
        if block == N_BLOCKS:
            study_session_end = now_time()
            study_session_duration = study_session_end - study_session_start
            for record in learning_data:
                record['study_session_end'] = study_session_end
                record['study_session_duration'] = study_session_duration
            write_data_csv(f"learning_data_participant_{exp_info['participant_id']}.csv", learning_data)
            break_screen(STUDY_SESSION_BREAK_TIME)
            long_break_start = now_time()
            if WM_MEMORY_STATUS:
                wm_data = []
                break_end_screen()
                run_working_memory_session(wm_data, WM_N_TRIALS)
                write_data_csv(f"working_memory_data_participant_{exp_info['participant_id']}.csv", wm_data)
    current_time = now_time()
    if exp_info['group'] == '1':
        left_break_time = STUDY_TEST_INTERVAL - (current_time - study_session_end)
        if left_break_time > 0:
            break_screen(left_break_time)
        break_end_screen()
        test_session_start = current_time
        exp_info['test_session_start'] = test_session_start
        test_data = []
        run_session_2(test_data)
        test_session_end = now_time()
        test_session_duration = test_session_end - test_session_start
        for record in test_data:
            record['test_session_end'] = test_session_end
            record['test_session_duration'] = test_session_duration
        write_data_csv(f"test_data_participant_{exp_info['participant_id']}.csv", test_data)
    elif exp_info['group'] == '2':
        session1_end_time = now_time()
        session1_end_text = visual.TextStim(
            win = win,
            text = "End of Session 1.\n\nPlease contact the experimenter to schedule Session 2.\n\n Press any key to end the experiment.",
            color = (1, 1, 1),
            colorSpace = 'rgb',
            height = 30,
            wrapWidth = 800
        )
        session1_end_text.draw()
        win.flip()
        event.waitKeys()

    elif exp_info['session'] == '2':
        test_session_start = now_time()
        exp_info['test_session_start'] = test_session_start
        test_data = []
        run_session_2(test_data, full_stimuli)
        test_session_end = now_time()
        test_session_duration = test_session_end - test_session_start
        for record in test_data:
            record['test_session_end'] = test_session_end
            record['test_session_duration'] = test_session_duration
        write_data_csv(f"test_data_participant_{exp_info['participant_id']}.csv", test_data)




