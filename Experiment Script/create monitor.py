### This script creates a monitor object and saves it to disk. It has to be run and only needs to be run once on the machine that runs the experiment created with Psychopy.
### Before running this script, make sure to change the parameters in step 2 to match the physical properties of the monitor that will be used to run the experiment. If you don't do this, then 'deg' units won't work correctly in your experiment.

### The saved monitor object will be stored in the default location for Psychopy monitor objects, which is typically in a folder called 'monitors' within the Psychopy installation directory (a hidden folder under .psychopy). When you create a window in your experiment and specify the name of this monitor, Psychopy will load the properties from this saved monitor object, allowing you to use 'deg' units correctly based on the physical properties you set here.
### For the alien feature experiment, later in the main experiment script, this monitor will be referenced by name. 
### The later script will retrieve this object from disk and use the properties you set here to calculate the correct size of stimuli in degrees of visual angle, which is important for ensuring that the stimuli are perceived at the intended size regardless of the physical monitor used.
### These data is important for creating the win for the experiment and for ensuring that the stimuli are presented at the correct size and location on the screen.

### For better accuracy, can use gammaMotionNull.py and gammaMotionAnalysis.py scripts to measure the actual gamma values for the monitor and then input those values here in the gamma grid. This will help ensure that the stimuli are displayed with the correct luminance levels, which is important for visual experiments.
### These two files usually can be found in psychopy/demos/coder/experiment control/
### But I also place copies of these files in the same folder as this script for easy access.

from psychopy import monitors
import numpy as np

# 1. Create the object
my_monitor = monitors.Monitor(name='AlienMemoryMonitor')

# 2.1 YOU MUST SET THESE! Otherwise, 'deg' units won't work correctly
my_monitor.setWidth(50)         # Width of screen in cm
my_monitor.setDistance(60)      # Distance from eyes in cm
my_monitor.setSizePix([1920, 1080]) # Resolution in pixels

# 2.2 Set the Gamma Grid (4 rows: Grey, Red, Green, Blue; 6 columns: min, max, gamma, etc.)
# Example: Setting just gamma values for R, G, B
gamma_grid = np.array([
    [0, 100, 2.2, 0, 0, 0], # Grey
    [0, 30, 2.1, 0, 0, 0],  # Red
    [0, 60, 2.3, 0, 0, 0],  # Green
    [0, 10, 2.2, 0, 0, 0]   # Blue
])


# 3. SAVE TO DISK (This stops the warning)
my_monitor.saveMon() 



