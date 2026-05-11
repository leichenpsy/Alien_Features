

# ============Import Modules===========

from psychopy import visual, event, core
from psychopy.hardware import keyboard
import numpy as np
import random
import math
from PIL import Image

# ============ Define parameters ===========
WIN_SIZE = ()
WIN_BG = () ## set window background color





# ============ Define Key functions ===========




# ============ Main Script ===========

### Create a window

win = visual.Window(
    size=WIN_SIZE,
    units='pix',
    monitor='testMonitor',
    color=WIN_BG,
    colorSpace='rgb',
    allowGUI=True,
    fullscr=False,
    waitBlanking=True
)

