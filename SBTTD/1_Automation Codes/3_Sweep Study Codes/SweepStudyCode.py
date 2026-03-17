## This code is intended to hasten the sweep study processs ##

import numpy as np
import pathlib
from pathlib import Path
import matplotlib as pyplot
import os


#%% Creating the various directories in each subdirectory # 

## Creating Directories for Mach Study ##
dirc_path = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\37_Mesh and CFD Setup\6_Optimized Case Sweep Study\3_Cases\0_15_SinePhase") # Parent directory
mach_list = np.arange(1.5,4.0 + 0.5, 0.5)



# Iterate through each item in the parent directory
for item in dirc_path.iterdir():
    if item.is_dir():  # Check if the item is a directory
        print(f"Processing directory: {item.name}")
        
        # Create new directories inside this subdirectory
        for mach in mach_list:
            new_dir = item / f"{item.name}_Mach {mach}"
            new_dir.mkdir(parents=True, exist_ok=True)
        
        
            print(f"Created: {new_dir} \n")


# 

#%% Copy and Pasting iCFD++ cases into each subdirectory ##




