import numpy as np 
import tecplot as tp
from tecplot.constant import *
import os
import shutil
import pathlib
from pathlib import Path



#--------- Functions ---------#
def subdirFile_finder(base_dir,file_name):
    
    # A subdirectory checker to determine which files don't have the file specified by the user # 
    subdirs_check = list(base_dir.rglob(file_name))
        
        
    # Another for loop to determine which subdirectories do not have the file specified by the user # 
    for subdir_check in subdirs_check:
        if not subdir_check.exists():
            print("ERROR!")
            print("=="*50)
            
            print(f"'{subdir_check}' does not exist! (No {file_name} found)")
            print("\n")
            print(f"Folder raising error: {subdir_check.parent.name}")
            
            print("=="*50)
            print("\n")


    # Getting all the subdirectories with the certain file specified by the user # 
    subdirs = list(base_dir.rglob(file_name))

    
    # Pre-Allocating Posix variable # 
    subdirs_posix = []
    
    
    for idx, subdir in enumerate(subdirs):
        subdirs_posix.append(subdir)
        
        
        
    return subdirs, subdirs_posix


### Creating a code that allows me to automatically generate Mach Contours with a certain layout format ###


### Specifying the base directory: CHANGE HERE IF YOUR DIRECTORY IS DIFFERENT  ### 
StudyDir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\37_Mesh and CFD Setup\6_Optimized Case Sweep Study\5_Results\Run3\h_l_0.03") #CHANGE DIRECTORY FOR THE CFD++ RESULTS HERE!!!!
file_name = "mcfd_tec.bin"


# Getting the subdirectories within the base directories # 
subdirs,subdirs_posix = subdirFile_finder(StudyDir,file_name)


# Defining Image quality # 
resNumber = 4096


### Importing files and layout ... Exporting PNG file ###

# Defining the desitination of pre-set directories # 

# Image export destination #  
pngDest = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\37_Mesh and CFD Setup\6_Optimized Case Sweep Study\5_Results\Run3\Mach Contours\h_l_0.03") # CHANGE DIRECTORY FOR THE LOCATION OF THE CONTOURS HERE!!!!


# Layout # 
fileLayoutName = "machLayout.lay" # CHANGE HERE IF THE LAYOUT NAME IS DIFFERENT 
layoutDest = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\tecplot layouts") / fileLayoutName # Change layout directory if needed. 



# Connecting to the sesssion # 
tp.session.connect()


### Exporting Mach contours ### 

for subdir_posix in subdirs_posix:
    
    # Clearing the current layout # 
    tp.new_layout()


      
    # Loding the layout # 
    tp.load_layout(layoutDest)


    # Loading file into tecplot # 
    tp.data.load_tecplot(subdir_posix, read_data_option = ReadDataOption.Replace, reset_style = False)

           

    # Getting plot and contour objects # 
    frame = tp.active_frame()
    plot = frame.plot()
    legend = plot.contour(0).legend


    # Get actual data range #
    frame = tp.active_frame()
    dataset = frame.dataset
    mach_var = dataset.variable('M')
    mach_min = min(zone.values(mach_var).min() for zone in dataset.zones())
    mach_max = max(zone.values(mach_var).max() for zone in dataset.zones())


    # Reset contour levels and contour range image as well #
    plot = frame.plot()
    plot.contour(0).levels.reset_levels(mach_min, mach_max, (mach_max - mach_min) / 10)


    tp.macro.execute_command(f'''$!GlobalContour 1
        ColorMapFilter
        {{
            ColorMapDistribution = Continuous
            ContinuousColor
            {{
                CMin = {mach_min}
                CMax = {mach_max}
            }}
        }}''')
        
        
     
        
    # Changing contour setting styles # 
    contour = plot.contour(0)
    contour.levels.reset_to_nice(num_levels= 10)
    #legend.position = (80, 85)  # (x%, y%) - centered horizontally, near top
        
    
    
  
    
    # Saving the contour as an image in PNG format # 
    tp.export.save_png(f'{pngDest}\{subdir_posix.parent.name}.png', resNumber, supersample = 3)
    
  
    
    




