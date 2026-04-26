
import xarray as xr
import numpy as np
from pathlib import Path
import tecplot as tp

# --- add these imports near the top ---
import re

# Tkinter Library (GUI) #import tkinter as tk
from tkinter import Tk
from tkinter.filedialog import askdirectory




###### =============== Functions =============== #######
def dict_to_ds_1d(data):
    """{var: np.array} -> xarray.Dataset with dim 'n'."""
    return xr.Dataset({k: (("n",), np.asarray(v)) for k, v in data.items()})




# Assigning directory so it can be used throughout the entire code # 
def assign_dir(title='Select Folder'):
    """
    Open a directory selection dialog.
    
    Why this works:
    1. We explicitly create and manage the root window
    2. We hide it so the user only sees the dialog
    3. We force the dialog to the front
    4. We clean up after ourselves
    """
    # Create root window (required parent for dialogs)
    root = Tk()
    
    # Hide it - we only want the dialog, not an empty window
    root.withdraw()
    
    # CRITICAL: Bring to front (fixes "invisible dialog" issue)
    root.attributes('-topmost', True)
    
    # For Spyder/Jupyter: force an update to process the window
    root.update()
    
    # NOW open the dialog
    path = askdirectory(title=title)
    
    # Clean up
    root.destroy()
    
    if path:
        print(f"Selected: {path}")
        return path
    else:
        print("Cancelled")
        return None





# Creating a function that imports the data  # 
def bigImport(base_dir,fileName):
    
    # Finding all the h_l_names from the folders # 
    h_l_names = []
    for subdir in base_dir.glob("*/*/"):
        if subdir.is_dir():
            h_l_names.append(subdir.name)


    # Root directory to import mcfd_tec.bin files # 
    subDirs1 = [p for p in base_dir.iterdir() if p.is_dir()]
    
    # Finding all mcfd_tec.bin files and getting variables # 
    subDirs2 = [p for d in subDirs1 for p in d.iterdir() if p.is_dir()]  # flattened
    file_paths = [p / fileName for p in subDirs2]
    
    
    # Extracting variables from tecplot # 
    test = tp.data.load_tecplot(file_paths[0].as_posix())


    # Extracting Values from Certain Zones # 
    section_zone = test.zone("Section")
    
    
    # All variable names in the dataset
    var_names = [v.name for v in test.variables()]
    
    
    #### Extracting Variables from Section Zone ####
    
    # Build dict: {var_name: numpy_array}
    data = {}
    for var in var_names:
        try:
            data[var] = section_zone.values(test.variable(var)).as_numpy_array()
        except Exception as e:
            print(f"Skipping {var}: not found")
    
    
    
    
    
    # Getting case names using regex library #
    case_re = re.compile(r"h_l_(?P<hl>[\d.]+)_p0_(?P<p0>\d+)bar")
    
    
    
    # Pre-Allocating dictionaries for each section # 
    ds_by_case = {} 
    ds_by_case_quad = {}
    ds_by_case_inlet = {}         # key: case name (e.g., 'h_l_0.01_p0_1bar'), value: xarray.Dataset
    ds_by_case_leftWall = {}
    ds_by_case_rightWall = {}
    index_by_case = {}       # key -> (h_l, p0_bar)
    
    
    # Looping through each file path # 
    for file_path in file_paths:
        if not file_path.is_file():
            continue
    
        # Load each case (you were loading only file_paths[0])
        tp.new_layout() # Creating a new tecplot layout. 
        test = tp.data.load_tecplot(file_path.as_posix())
        
        
        # Extracting zones !!!! Should be less hardcoded. Works for now however.... # 
        section_zone = test.zone("Section")
        cells_zone = test.zone("QUADRILATERAL_cells")
        inlet_zone = test.zone("Inlet")
        
        # Getting all the variables available in the dataset with PyTecplot # 
        var_names = [v.name for v in test.variables()]
        
        # Grab values into a plain dict
        data = {}
        data_cells = {}
        data_inlet = {}
        
        # for loop to get all the ariabes in each section #
        for var in var_names:
            try:
                data[var] = section_zone.values(test.variable(var)).as_numpy_array()
                data_cells[var] = cells_zone.values(test.variable(var)).as_numpy_array()
                data_inlet[var] = inlet_zone.values(test.variable(var)).as_numpy_array()
            except Exception:
                pass
    
        # Making a dataset for each case # 
        ds_case = dict_to_ds_1d(data)
        ds_case_quad = dict_to_ds_1d(data_cells)
        ds_case_inlet = dict_to_ds_1d(data_inlet)
        
        # key by the folder name that contains the file, e.g. 'h_l_0.01_p0_1bar'
        case_name = file_path.parent.name
        ds_by_case[case_name] = ds_case
        ds_by_case_quad[case_name] = ds_case_quad
        ds_by_case_inlet[case_name] = ds_case_inlet
    
        # parse h_l and p0 for optional stacking later
        m = case_re.fullmatch(case_name)
        if m:
            h_l = float(m.group("hl"))
            p0_bar = float(m.group("p0"))
            index_by_case[case_name] = (h_l, p0_bar)
    
    return ds_by_case, ds_by_case_quad, ds_by_case_inlet


# This is a funciton that imports data, but gives you the choice to type out the zone name that you want to extract  # 
def Import_choice_3D(base_dir,fileName , h_l_name, sec_name1 , main_name2, inlet_name3):
    import os
    import tecplot as tp
    from tecplot.constant import PlotType, SliceSource
    from utils.parameterComputation import variableImporterMasked
    ##### Instructions #####
    # h_l_name ---> define the h/l name that you want for this case. #
    # sec_name1 ---> that's the name that you specified for the wavy section # 
    # main_name 2 --> that's the anme that you specified for the main mesh. That would be quad_cells for 2D meshes and BRICK_cells for 3D meshes. #
    # inlet_name3 --> the name of the inlet zone. Make sure that captilization is good for all names. Otherwise, an error will come up. #
    ##### End of instrucitons ####
    

    # Finding the file paths with a specific extension #
    file_paths = list(base_dir.glob("**/*.bin"))

    # Extracting variables from tecplot # 
    test = tp.data.load_tecplot(file_paths[0].as_posix())


    # Extracting Values from Certain Zones #
    section_zone = test.zone(sec_name1)

    # Defining the active frame #
    act_frame = tp.active_frame().plot()



    # Extracting Slices  # 
    extracted_slice = extracted_slice = tp.data.extract.extract_slice(
        origin=(0, 0, 0),
        normal=(0, 0, 1),
        source=SliceSource.VolumeZones,
        dataset=test)



    # Getting the data from the extracted zone #
    section_zone = test.zone(extracted_slice)


    # All variable names in the dataset
    var_names = [v.name for v in test.variables()]


    #### Extracting Variables from Section Zone ####

    # Build dict: {var_name: numpy_array}
    data = {}
    for var in var_names:
        try:
            data[var] = section_zone.values(test.variable(var)).as_numpy_array()
        except Exception as e:
            print(f"Skipping {var}: not found")

    
    

    
    
    # Pre-Allocating dictionaries for each section # 
    ds_by_case = {} 
    ds_by_case_quad = {}
    ds_by_case_inlet = {}         # key: case name (e.g., 'h_l_0.01_p0_1bar'), value: xarray.Dataset

    
    # Looping through each file path # 
    for file_path in file_paths:
        if not file_path.is_file():
            continue
    
        # Load each case (you were loading only file_paths[0])
        tp.new_layout() # Creating a new tecplot layout. 
        test = tp.data.load_tecplot(file_path.as_posix())
        
        
        # Extracting zones !!!! Should be less hardcoded. Works for now however.... # 
        section_zone = test.zone(sec_name1)
        cells_zone = test.zone(main_name2)
        inlet_zone = test.zone(inlet_name3)
        
        # Getting all the variables available in the dataset with PyTecplot # 
        var_names = [v.name for v in test.variables()]
        
        # Grab values into a plain dict
        data = {}
        data_cells = {}
        data_inlet = {}
        
        # for loop to get all the Variabes in each section #
        for var in var_names:
            try:
                data[var] = section_zone.values(test.variable(var)).as_numpy_array()
                data_cells[var] = cells_zone.values(test.variable(var)).as_numpy_array()
                data_inlet[var] = inlet_zone.values(test.variable(var)).as_numpy_array()
            except Exception:
                pass
    
        # Making a dataset for each case # 
        ds_case = dict_to_ds_1d(data)
        ds_case_quad = dict_to_ds_1d(data_cells)
        ds_case_inlet = dict_to_ds_1d(data_inlet)
        
        # key by the folder name that contains the file, e.g. 'h_l_0.01_p0_1bar'
        case_name = h_l_name
        ds_by_case[case_name] = ds_case
        ds_by_case_quad[case_name] = ds_case_quad
        ds_by_case_inlet[case_name] = ds_case_inlet
    

            
            
            
    
    return ds_by_case, ds_by_case_quad, ds_by_case_inlet




        
        


"""
#------------------------------------------------------------------------------------------------------------------------------------#
    Saving all dictionaries from the previous run. This saves time since you will not have to post-process the results every time...
#------------------------------------------------------------------------------------------------------------------------------------#

"""

import pickle
import shutil
from datetime import date


def runSaver(ds_by_case, ds_by_case_quad, ds_by_case_inlet, base_dir_dic = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\data\processed\Mach Study")):

    
    # Getting today's date in MM_DD_YYYY format
    today = date.today()
    formatted_date = f"{today.month:02d}_{today.day:02d}_{today.year}"
    
    
    # Create save directory (remove if exists)
    save_dir = base_dir_dic / formatted_date
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir()
    
    # Saving cases # 
    with open(save_dir / "ds_by_case.pkl", "wb") as f:
        pickle.dump(ds_by_case, f)
    with open(save_dir / "ds_by_case_quad.pkl", "wb") as f:
        pickle.dump(ds_by_case_quad, f)
    with open(save_dir / "ds_by_case_inlet.pkl", "wb") as f:
        pickle.dump(ds_by_case_inlet, f)
        return 



    


    
#%% 

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                            Here you can load your data from previously saved run
#------------------------------------------------------------------------------------------------------------------------------------#

"""

 
import pickle
import types
from datetime import date
import xarray as xr
import numpy as np
from pathlib import Path
import tecplot as tp

# --- add these imports near the top ---
import re
import numpy as np
import xarray as xr
import tecplot as tp
from pathlib import Path


#### Functions ####


# Converting dictionaries to 1d datasets # 
def dict_to_ds_1d(data):
    """{var: np.array} -> xarray.Dataset with dim 'n'."""
    return xr.Dataset({k: (("n",), np.asarray(v)) for k, v in data.items()})


# A function that lists all the top directories #
def list_top_directories(base_dir_dic) :
    p = Path(base_dir_dic)
    top_dircs = [entry for entry in p.iterdir() if entry.is_dir()]
    
    # Pre-allocating Variables # 
    top_dirc_names = []
    
    # For loop to determine the top directory names # 
    for top_dirc in top_dircs:
        top_dirc_names.append(top_dirc.parts[-1])
        
    return top_dircs , top_dirc_names



# Loads the runs into python # 
def runLoader(load_dir_dic = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\data\processed\Mach Study"):

    ### Loading the top directory and the top directory names #  
    top_dircs, top_dirc_names = list_top_directories(load_dir_dic)
    
    
    # Finding the latest date # 
    latest_date = max(top_dirc_names)
    
    
    # Defining the latest saved file # 
    latest_date_dir = Path(load_dir_dic + '//' + latest_date)
      
    
    # Loading all the data in automatically based on the latest date # 
    
    with open(rf"{latest_date_dir}\ds_by_case.pkl", "rb") as f:
        ds_by_case = pickle.load(f)
    with open(rf"{latest_date_dir}\ds_by_case_quad.pkl", "rb") as f:
        ds_by_case_quad = pickle.load(f)
    with open(rf"{latest_date_dir}\ds_by_case_inlet.pkl", "rb") as f:
        ds_by_case_inlet = pickle.load(f)
        return ds_by_case, ds_by_case_quad, ds_by_case_inlet




#%% Loads residuals 

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------- your directory setup ----------------


def file_pathFinder(fileName_info , rootDir_info = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\8_Mach_Sweep_Study_2(Solution)\4_Mach_Reruns")):

    subDirs_info  = [p for p in rootDir_info.iterdir() if p.is_dir()]
    file_paths_info = [p / fileName_info for p in subDirs_info]
    return file_paths_info

# ---------------- parser for mcfd info files ----------------
def load_minfo_step_force(path, use="step"):
    """
    Read a 'minfo1_e2' (mcfd.info-like) file and return x (step or time) and x_force arrays.
    Lines starting with '#' are ignored. Data rows are: Step  Time  X-force
    """
    xs, fs = [], []
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                # rows look like: 1  0.0000000e+000  2.1942451e+005
                try:
                    step = float(parts[0])
                    time = float(parts[1])
                    x_force = float(parts[2])
                except ValueError:
                    continue
                xs.append(step if use == "step" else time)
                fs.append(x_force)
    except FileNotFoundError:
        return np.array([]), np.array([])
    return np.asarray(xs), np.asarray(fs)




