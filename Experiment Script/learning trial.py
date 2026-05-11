

# ============Import Modules===========

from psychopy import monitors, visual, event, core
from psychopy.hardware import keyboard
import numpy as np
import random
import math
from PIL import Image

# ============ Define parameters ===========
MONITOR = 'AlienMemoryMonitor' ## set the name of the monitor that you created with create_monitor.py
WIN_SIZE = ()
WIN_BG = () ## set window background color in rbg format, e.g. (0, 0, 0) for black, (1, 1, 1) for white, (-1, -1, -1) for black in rgb space

##### Learning Trial Parameters #####
ENCODING_TIME = 4.0 ## time to show each alien during encoding phase in seconds
BLANK_TIME = 0.5 ## time to show blank screen between encoding and practice phases in seconds
INTERVAL_TIME = 1.0 ## time to show blank screen between learning trials in seconds
FIXATION_TIME = np.random.uniform(0.75,1.25) ## time to show fixation cross before each trial in seconds (randomized between 0.75 and 1.25 second)  

###### Stimulus Parameters ######
ALIEN_SIZE = 100 ## size of the alien images in pixels
ALIEN_M_NAMES = ["Ethan Miller", "Liam Chen", "Noah Johnson", "Mason Clark", "Aiden Brown", "Lucas Walker", "Owen Bennett", "Leo Collins", "Caleb Turner", "Wyatt Reed", "Nathan Scott", "Ezra Hayes","Hudson Gray", "Carter Price", "Grayson Cole", "Asher Bell", "Dylan Foster", "Ryan Cooper", "Roman Blake", "Hunter Murphy", "Aaron Cox", "Thomas Watson", "Theo Lawson", "Gavin Pierce", "Jason Howard"]
ALIEN_F_NAMES = ["Sophie Lee", "Olivia Smith", "Emma Davis", "Ava Mitchell", "Mia Taylor", "Harper Bailey", "Ella Ward", "Grace Hughes", "Zoe Ramirez","Lily Adams", "Riley Griffin","Avery Diaz","Paige Larson", "Kayla Dunn", "Katie Ross", "Jessica Lane", "Lauren Hill", "Rachel Wood", "Nicole Grant", "Leah Perry", "Amber Cruz", "Allison Ford", "Julia Banks", "Chloe Dean", "Hailey Moore"]









# ============ Define Key functions ===========




# ============ Main Script ===========

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
    useRetina=True
)

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


