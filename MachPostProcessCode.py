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

# Tkinter Library (GUI) #
import tkinter as tk
from tkinter import filedialog
from tkinter import Tk
from tkinter.filedialog import askdirectory
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


### Post-Processing for the Stagnation Pressure sweep study ###
tp.session.connect()

# Defined Values # 
base_dir = Path(rf"{assign_dir()}")
fileName = "mcfd_tec.bin"
ds_by_case, ds_by_case_quad, ds_by_case_inlet = bigImport(base_dir,fileName)







        
        






#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
    Saving all dictionaries from the previous run. This saves time since you will not have to post-process the results every time...
#------------------------------------------------------------------------------------------------------------------------------------#

"""



def runSaver(base_dir_dic = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Python Results\Mach Study") ):
    import pickle
    import shutil
    from datetime import date
    from pathlib import Path
    
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
def runLoader(load_dir_dic = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Python Results\Mach Study"):

    ### Loading the top directory and the top directory names #  
    top_dircs, top_dirc_names = list_top_directories(load_dir_dic)
    
    
    # Finding the latest date # 
    latest_date = max(top_dirc_names)
    
    
    # Defining the latest saved file # 
    latest_date_dir = Path(load_dir_dic + '//' + latest_date)
    
    test = rf"{latest_date_dir}\ds_by_case.pkl"
    
    
    # Loading all the data in automatically based on the latest date # 
    
    with open(rf"{latest_date_dir}\ds_by_case.pkl", "rb") as f:
        ds_by_case = pickle.load(f)
    with open(rf"{latest_date_dir}\ds_by_case_quad.pkl", "rb") as f:
        ds_by_case_quad = pickle.load(f)
    with open(rf"{latest_date_dir}\ds_by_case_inlet.pkl", "rb") as f:
        ds_by_case_inlet = pickle.load(f)
    with open(rf"{latest_date_dir}\index_by_case.pkl", "rb") as f:
        index_by_case = pickle.load(f)
        return ds_by_case, ds_by_case_quad, ds_by_case_inlet, index_by_case



# Saving the files in a certain format #
ds_by_case, ds_by_case_quad, ds_by_case_inlet,_ = runLoader()
        



#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                            Computing Parameters: Reynolds Number, y+ value, Residuals, Net Mass Flux
#------------------------------------------------------------------------------------------------------------------------------------#

"""

# Computing Reynolds number for each case #
import numpy as np

def global_max_from_dict(arrs, ignore_nan=True):
    """
    arrs: dict[str, np.ndarray]
    Returns: (key, index_or_indices, value)
    """
    best_key = None
    best_idx = None
    best_val = -np.inf

    for k, a in arrs.items():
        if a.size == 0:
            continue
        if ignore_nan:
            # skip arrays that are all-NaN
            if np.isnan(a).all():
                continue
            flat_idx = np.nanargmax(a)
            val = a.reshape(-1)[flat_idx]
        else:
            flat_idx = np.argmax(a)
            val = a.reshape(-1)[flat_idx]

        if val > best_val:
            best_val = val
            best_key = k
            best_idx = np.unravel_index(flat_idx, a.shape)  # handles 1D/2D/ND

    if best_key is None:
        raise ValueError("No valid elements found (arrays empty or all-NaN).")

    return best_key, best_idx, best_val




"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                                Computing Reynolds Number
#------------------------------------------------------------------------------------------------------------------------------------#

"""


# All unit Conversions #
# Unit conversion setup # 


# Getting Reynolds Number # 
Re = {}
Re_wall = {} 


# Getting All Variables to compute Reynolds number # 


for key in ds_by_case:
    mu_ref = 1.78e-5 #Pa*s
    T = ds_by_case[key]["T"].data
    T_ref = 300 # kelvin
    S = 120 #kelvin
    
    mu = (mu_ref * (T/T_ref)**(1.5)) * ((T_ref + S)/ (T + S))
    U = ds_by_case_inlet[key]["U"].data
    rho = ds_by_case_inlet[key]["R"].data
    X = 0.1
    print(np.mean(rho))
    print(np.mean(U))
    print()
    Re[key] = (np.mean(rho) * (np.mean(U) ) * X ) / np.mean(mu) # this is just a test. Re wall is not a thing. Well, it is, but you use boundary layer thickness to compute that stuff not the length of the entire thing...
   
    
ReMAX = global_max_from_dict(Re)




"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                                Computing Y+
#------------------------------------------------------------------------------------------------------------------------------------#

"""

# Finding the y+ values and also the max y+ value # 
y_plus = {}
for key in ds_by_case:
   y_plus[key] =  ds_by_case[key]["Y_plus"].data
    


y_plusMAX = global_max_from_dict(y_plus)
print(f"The global Max y_plus value is: {y_plusMAX[2]:.2f}\n")



"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                        Computing Wall shear Stress(tau_x)
#------------------------------------------------------------------------------------------------------------------------------------#

"""

# Finding the wall shear stress along the wall # 
tau_x = {}
tau_y = {}
X_geom = {}
tau_separation = {}
tau_separation_idx = {}

# Masking for x geometry # 

for key in ds_by_case:
    tau_x[key] = ds_by_case[key]["Tau_x"].data
    tau_y[key] = ds_by_case[key]["Tau_y"].data
    X_geom[key] = ds_by_case[key]["X"].data
    y_plus[key] = ds_by_case[key]["Y_plus"].data
    # Masking # 
    mask = (X_geom[key] >= 0) & (X_geom[key] <= 0.1)
    X_geom[key], tau_y[key], tau_x[key],y_plus[key] = X_geom[key][mask] , tau_y[key][mask], tau_x[key][mask], y_plus[key][mask]
    
    
    
    #### COMPUTING THE FIRST POINTS AT WHICH A TAU_Y GOES BELOW ZERO ######
    first_index_tau = np.argmax(tau_x[key] < 0)
    tau_separation[key] = tau_x[key][first_index_tau] #finds the first point at which separation occurs for each h_l case and pressure
    tau_separation_idx[key] = first_index_tau
        
#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Processing Convergence criteria results(X-force Vs X) 
#------------------------------------------------------------------------------------------------------------------------------------#

"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------- your directory setup ----------------
rootDir_info = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\8_Mach_Sweep_Study_2(Solution)\4_Mach_Reruns")
subDirs_info  = [p for p in rootDir_info.iterdir() if p.is_dir()]
fileName_info = "minfo1_e2"
file_paths_info = [p / fileName_info for p in subDirs_info]



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

# ---------------- plotting ----------------
paths = [p for p in file_paths_info if p.exists()]
if not paths:
    print("No minfo1_e2 files found.")
else:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(paths)))

    for color, p in zip(colors, paths):
        x, f = load_minfo_step_force(p, use="step")  # use="time" to plot vs Time instead
        if x.size == 0:
            print(f"Skipping (no data): {p}")
            continue

        # label from the case folder name (parent directory)
        label = p.parent.name
        # sort by x in case rows are out of order
        order = np.argsort(x)
        ax.plot(x[order], f[order], lw=2, color=color, label=label)

    ax.set_title(r"X-force vs Iterations",fontsize = 18)
    ax.set_xlabel("Iterations",fontsize = 14)                 # change to "Time [s]" if use="time"
    ax.set_ylabel("X-force", fontsize = 14)
    ax.grid(True, which = "major", alpha = 0.3)
    ax.grid(True,which ="minor", alpha = 0.3)
    ax.legend(title="Cases", loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    plt.show()

#%%



"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                Plotting Processed Results: y+, Shear Stress[Pa], and location of separation
#------------------------------------------------------------------------------------------------------------------------------------#
"""
    
## Plotting the results of tau_x to see how they differ from one another ##
import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.figure(figsize = (8,6))
# --- Group the cases by h_l value ---
cases_by_hl = {}
for key in ds_by_case:
    # Match either:
    #   h_l_0.01 (numeric: digits.digits)
    #   h_l_x    (letter pattern)
    hl_match = re.match(r"(h_l_(?:\d+\.\d+|[a-zA-Z]+))", key)
    if hl_match:
        hl = hl_match.group(1)
        cases_by_hl.setdefault(hl, []).append(key)


"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                     Plotting Wall Shear stress
#------------------------------------------------------------------------------------------------------------------------------------#
 
"""  

 
# --- Plot for each h_l group: Plotting tau_y(wall shear stress) ---
for hl, case_keys in cases_by_hl.items():
    plt.figure(figsize=(8,6))

    
    for i , key in enumerate(case_keys):
        # Use colormap # 
        cmap = cm.get_cmap("cividis",len(case_keys))
        plt.plot(X_geom[key], tau_x[key], label=key , color = cmap(i),linewidth = 2)
    

    plt.title(fr"$\tau_x$ vs X for {hl}",fontsize = 21)
    plt.axhline(y=0, color='r', linestyle='--', label='Separation')
    plt.xlabel("X [m]" , fontsize = 16)
    plt.ylabel(r"$\tau_x$ [Pa]", fontsize = 16)
    plt.grid(True, which="both")
    plt.legend(loc = "center left",bbox_to_anchor = (1,0.5))
    plt.show()
    

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                                 Plotting y+
#------------------------------------------------------------------------------------------------------------------------------------#
""" 

  
    
# Plotting Y+ values on top of each other # 
for hl, case_keys in cases_by_hl.items():
    plt.figure(figsize=(8,6))

    
    for i , key in enumerate(case_keys):
        # Use colormap # 
        cmap = cm.get_cmap("cividis",len(case_keys))
        plt.plot(X_geom[key], y_plus[key], label=key , color = cmap(i),linewidth = 2)
    

    plt.title(fr"$y^+$ vs X for {hl}",fontsize = 21)
    plt.xlabel("X [m]" , fontsize = 16)
    plt.ylabel(r"$y^+$", fontsize = 16)
    plt.grid(True, which="both")
    plt.legend(loc = "center left",bbox_to_anchor = (1,0.5))
    plt.show()    


#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                        Plotting Separation Points
#------------------------------------------------------------------------------------------------------------------------------------#
"""   




import numpy as np
from scipy.interpolate import UnivariateSpline



for key in ds_by_case:
    # Pre-allocating Variables #
    tau_y_list = []
    sep_length = []
    
    
    #### GPT Test ####
    # your arrays
    x = X_geom[key]
    y = tau_y[key]
    
    # x, y are your 1D arrays
    s = UnivariateSpline(x, y, s=0)   # s=0 -> exact through points (use s>0 to smooth)
    x_zeros = s.roots()               # continuous x-positions where y=0
    idx_nearest = np.searchsorted(x, x_zeros)
    
    
    #Obtaining the separation lenght based on the scipy function # 
    for i in range(len(idx_nearest) -1):
        np.array(tau_y_list.append(tau_y[key][idx_nearest[i]:idx_nearest[i+1]]))
        sep_length.append(X_geom[key][idx_nearest[i+1]] - X_geom[key][idx_nearest[i]])
      
        
    #Getting the separation location and the respective tau, respectively # 
    sep_location = [X_geom[key][idx_nearest[k]] for k in range(len(idx_nearest))]
    tau_y_location = [tau_y[key][idx_nearest[k]] for k in range(len(idx_nearest))]
    
    
    # Plotting Graphs 
    plt.plot(X_geom[key],tau_y[key], label = 'Tau_y')
    plt.scatter(sep_location,tau_y_location,color = 'red', label = 'Separation Points')
    plt.grid(True,which = "both")
    plt.show()


#%%  
"""
#------------------------------------------------------------------------------------------------------------------------------------#
            Plotting the residuals and also plotting the net mass flow. Will have to learn how to do that from CFD++
#------------------------------------------------------------------------------------------------------------------------------------#
""" 



import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_residuals(path, skiprows=3, ncols=4):
    # Read all columns as strings, whitespace-separated
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skiprows, engine="python")
    # Coerce anything non-numeric to NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    # Keep only numeric columns
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < ncols:
        raise ValueError(f"Found only {num.shape[1]} numeric columns in {path}, need {ncols}.")
    # Most residual files have iteration/index columns first; residuals are usually the last 4 numeric cols
    resid = num.iloc[:, -ncols:].to_numpy()
    return resid



# Root directory to import mcfd_tec.bin files # 
rootDir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Results\Mach Study 2") # this is the root directory to the parametric study solution files
subDirs1 = [p for p in rootDir.iterdir() if p.is_dir()]
fileName = "mcfd.rhsgi"
subDirs2_rhsgi = [p for d in subDirs1 for p in d.iterdir() if p.is_dir()]  # flattened
file_paths_rhsgi = [p / fileName for p in subDirs2_rhsgi]


# Residuals labels # 
Resid_labels = ["energy", "mass", "x-momentum", "y-momentum"]

for file_path_rhsgi in file_paths_rhsgi:
    try: 
        resid = load_residuals(file_path_rhsgi.as_posix(), skiprows=3, ncols=4)
    
        # Normalize each column by its first entry (avoid divide-by-zero)
        denom = resid[0, :].copy()
        denom[denom == 0] = 1.0
        resid = resid / denom
    
        iterations = np.arange(1, resid.shape[0] + 1)
    
        plt.figure(figsize=(8, 6))
        for j in range(resid.shape[1]):
            plt.semilogy(iterations, resid[:, j], linewidth=2)
        plt.title(f"Residuals Vs Iterations: {file_path_rhsgi.parent.name}", fontsize=24)
        plt.legend(Resid_labels)
        plt.xlabel("Iterations")
        plt.ylabel("Residuals")
        plt.grid(True, which="both")
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.show()
    except: 
        print(f"Couldn't Find rhsgi file for {file_path_rhsgi.parent.name}\n")



residMAX = max(resid[-1])
print(f"Maximum residual: {residMAX:.2e} \n")




    
#%% 
"""
#------------------------------------------------------------------------------------------------------------------------------------#
     Getting all residuals and all convergence criterions(net mass flow) to be able to see if the simuilations converged properly
#------------------------------------------------------------------------------------------------------------------------------------#
""" 


# Root directory to import mcfd_tec.bin files # 
rootDir_flux = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\8_Mach_Sweep_Study_2(Solution)\4_Mach_Reruns") # this is the root directory to the parametric study solution files
subDirs1_flux = [p for p in rootDir_flux.iterdir() if p.is_dir()]
fileName_flux= "minfo1_e1"



subDirs2_flux = [p for d in subDirs1_flux for p in d.iterdir() if p.is_dir()]  # flattened
file_paths_flux = [p / fileName_flux for p in subDirs2_flux]







## Load data ## 
mass_flux_end = {}

for file_path_flux in file_paths_flux:
    df = pd.read_csv(file_path_flux, sep=r"\s+", comment="#")
    df.rename(columns = {"mass_flux":"misc","infout1":"iterations","d":"mass_flux"},inplace = True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(df["iterations"][:-3],df["mass_flux"][:-3])
    
    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,1))
    
    plt.xlabel("Iterations")
    plt.ylabel("Net Mass Flux")
    plt.title(f"Net Mass Flux Vs Iterations: {file_path_flux.parent.name}")
    plt.grid(True,which = "both")
    plt.tick_params(axis='both', which='major', labelsize=18)

    # Creating a dictonary for the different mass flux # 
    mass_flux_end[file_path_flux.parent.name] = df["mass_flux"][:-3].iloc[-1]

plt.show()



mass_fluxMAX_key = max(mass_flux_end, key = mass_flux_end.get)
mass_fluxMAX_val = mass_flux_end[mass_fluxMAX_key]
print(f"Highest last iteration mass flux: {mass_fluxMAX_val:.2e} at {mass_fluxMAX_key}\n")

massFluxCriterion = 0.8

# Defining Color for more clear print # 
RED = '\033[31m'
RESET = '\033[0m'


for key,value in mass_flux_end.items():
    if value > massFluxCriterion:
        diff = value - massFluxCriterion
        if  diff > 1.0:
            print(f"{key} does not meet criteria(net mass flux < 0.8). Off by {RED}{diff:.1f}{RESET} and net mass flux is {RED}{value}{RESET}\n")
        else: 
            print(f"{key} does not meet criteria(net mass flux < 0.8). Off by {diff:.1f} and net mass flux is {value}\n")
            
            
            #%%

#### ------ Creating a set for h_l of 0.02 only and seeing how the percentage difference changes with various h_l ----- ####

# Getting values for h_l 0.02 for mass flux percentage #
rootDir_info_inlet = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Results\Mach Study 2\h_l_0.02")
subDirs_info_inlet  = [p for p in rootDir_info.iterdir() if p.is_dir()]
fileName_flux_inlet = "minfo1_e3"
file_paths_info = [p / fileName_flux_inlet for p in subDirs_info]

subDirs_info_totFlux = [p for p in rootDir_info.iterdir() if p.is_dir()]
fileName_totFlux = "minfo1_e1"
file_paths_totFLux = [p / fileName_totFlux for p in subDirs_info_totFlux]


# Plotting results for h_l 0.02 # 
# Rebuild file lists from your subdir lists (fixing the earlier variable typo)
file_paths_inlet = [(p / fileName_flux_inlet) for p in subDirs_info_inlet]
file_paths_total = [(p / fileName_totFlux)    for p in subDirs_info_totFlux]

def read_minfo_flexible(path: Path):
    """
    Flexible reader for minfo1_* files.
    - Skips comment/blank lines.
    - On each data line, parses *numeric* tokens only.
    - If >=3 numbers: first is 'iter', last is 'val'.
      If exactly 2 numbers: treat last as 'val' and auto-iterate (1,2,3,...).
    Returns arrays (iter, val) as float.
    """
    iters, vals = [], []
    auto_i = 0
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                nums = []
                for tok in s.replace(",", " ").split():
                    try:
                        nums.append(float(tok))
                    except ValueError:
                        continue
                if len(nums) >= 3:
                    iters.append(nums[0])
                    vals.append(nums[-1])
                elif len(nums) == 2:
                    auto_i += 1
                    iters.append(float(auto_i))
                    vals.append(nums[-1])
                else:
                    continue
    except FileNotFoundError:
        return np.array([]), np.array([])
    iters = np.asarray(iters, dtype=float)
    vals  = np.asarray(vals,  dtype=float)

    # sort by iteration, drop duplicate iters (keep last occurrence)
    if iters.size:
        order = np.argsort(iters)
        iters, vals = iters[order], vals[order]
        # remove duplicates
        uniq, idx = np.unique(iters, return_index=True)
        iters, vals = iters[idx], vals[idx]
    return iters, vals

# Map case folder -> file path for e1 (net) and e3 (inlet)
total_by_case = {p.parent.name: p for p in file_paths_total if p.exists()}
inlet_by_case = {p.parent.name: p for p in file_paths_inlet if p.exists()}

common_cases = sorted(set(total_by_case) & set(inlet_by_case))
if not common_cases:
    print("No case has BOTH minfo1_e1 and minfo1_e3 under rootDir_info.")
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = cm.get_cmap("viridis", len(common_cases))
    plotted = 0

    for i, case in enumerate(common_cases):
        p_e1 = total_by_case[case]   # minfo1_e1 (net)
        p_e3 = inlet_by_case[case]   # minfo1_e3 (inlet)

        it_e1, v_e1 = read_minfo_flexible(p_e1)
        it_e3, v_e3 = read_minfo_flexible(p_e3)

        if it_e1.size == 0 or it_e3.size == 0:
            print(f"Skipping {case}: empty data (e1:{it_e1.size}, e3:{it_e3.size})")
            continue

        # Align by iteration (inner join)
        # Build dicts for fast lookup then intersect keys
        d_e1 = dict(zip(it_e1, v_e1))
        d_e3 = dict(zip(it_e3, v_e3))
        it_common = np.array(sorted(set(d_e1.keys()) & set(d_e3.keys())), dtype=float)
        if it_common.size == 0:
            print(f"Skipping {case}: no overlapping iterations")
            continue

        net   = np.array([d_e1[it] for it in it_common], dtype=float)
        inlet = np.array([d_e3[it] for it in it_common], dtype=float)

        mask = np.isfinite(net) & np.isfinite(inlet) & (np.abs(inlet) > 0)
        if not np.any(mask):
            print(f"Skipping {case}: all denom invalid/zero")
            continue

        it_plot = it_common[mask]
        pct     = 100.0 * np.abs(net[mask]) / np.abs(inlet[mask])

        ax.plot(it_plot, pct, lw=2, color=cmap(i), label=case)
        plotted += 1

    ax.set_title("Mass-flux Imbalance vs Iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"Imbalance [%]  =  $|\dot m_{\rm total}| / |\dot m_{\rm inlet}| \times 100$")
    ax.grid(True, which="both", alpha=0.35)
    if plotted:
        ax.legend(title="Cases", loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        print("No curves plotted — check the messages above for which cases were skipped.")
    fig.tight_layout()
    plt.show()

# ==== Print last-iteration imbalance % for each case ====
print("\n=== Last-iteration mass-flux imbalance (|e1|/|e3| * 100) ===")

results = []
for case in common_cases:
    p_e1 = total_by_case[case]   # minfo1_e1 (net/total)
    p_e3 = inlet_by_case[case]   # minfo1_e3 (inlet)

    it_e1, v_e1 = read_minfo_flexible(p_e1)
    it_e3, v_e3 = read_minfo_flexible(p_e3)
    if it_e1.size == 0 or it_e3.size == 0:
        print(f"Skipping {case}: empty data")
        continue

    # align by iteration (inner join via dicts)
    d1 = dict(zip(it_e1, v_e1))
    d3 = dict(zip(it_e3, v_e3))
    it_common = np.array(sorted(set(d1) & set(d3)), dtype=float)
    if it_common.size == 0:
        print(f"Skipping {case}: no overlapping iterations")
        continue

    net   = np.array([d1[it] for it in it_common], dtype=float)
    inlet = np.array([d3[it] for it in it_common], dtype=float)
    mask  = np.isfinite(net) & np.isfinite(inlet) & (np.abs(inlet) > 0)

    if not np.any(mask):
        print(f"Skipping {case}: no valid ratio (zero/NaN inlet)")
        continue

    # last valid sample (highest iteration with a valid ratio)
    last_idx = np.where(mask)[0][-1]
    last_it  = it_common[last_idx]
    last_pct = 100.0 * np.abs(net[last_idx]) / np.abs(inlet[last_idx])

    results.append((case, last_it, last_pct))

# pretty print (sorted by case name; change key to sort by % if you prefer)
for case, last_it, last_pct in sorted(results, key=lambda t: t[0]):
    it_disp = int(round(last_it)) if abs(last_it - round(last_it)) < 1e-6 else last_it
    print(f"{case:35s}  iter {it_disp:>7}  ->  {last_pct:8.3f}%")

# (optional) save to CSV
import pandas as pd
pd.DataFrame(results, columns=["case","last_iter","last_pct"]).to_csv("mass_flux_imbalance_last.csv", index=False)

    
#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                        Getting all Boundary Layer profiles using Total enthalpy method 
#------------------------------------------------------------------------------------------------------------------------------------#
""" 

"""
Optimized Boundary Layer Analysis
Key optimizations explained inline
"""
import numpy as np
import tecplot as tp
from tecplot.constant import PlotType
import time 
from pathlib import Path

# ============================================================================
# OPTIMIZATION 1: Move functions outside loop (avoid re-definition overhead)
# ============================================================================
def find_bl_edge_curvature(y, dy_dx, curvature_threshold=0.0001):
    """Find where second derivative (curvature) indicates settling."""
    d2y_dx2 = np.gradient(dy_dx, y)
    peak_idx = np.argmax(dy_dx)
    
    max_curvature = np.max(np.abs(d2y_dx2[peak_idx:]))
    if max_curvature == 0:
        return y[-1], len(y) - 1
        
    normalized_curvature = np.abs(d2y_dx2[peak_idx:]) / max_curvature
    settled = normalized_curvature < curvature_threshold
    
    if np.any(settled):
        window = 5
        for i in range(len(settled) - window):
            if np.all(settled[i:i+window]):
                bl_idx = peak_idx + i
                return y[bl_idx], bl_idx
    
    return y[-1], len(y) - 1


### Defining the file path directory ###
base_dir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\7_Parametric Study\1_DeltaPstag Simulations")
rootDir = base_dir # this is the root directory to the parametric study solution files
subDirs1 = [p for p in rootDir.iterdir() if p.is_dir()]

fileName = "mcfd_tec.bin"
subDirs2 = [p for d in subDirs1 for p in d.iterdir() if p.is_dir()]  # flattened
file_paths = [p / fileName for p in subDirs2]




# ============================================================================
# OPTIMIZATION 2: Pre-compute constants once
# ============================================================================
IN_TO_MM = 25.4
MM_TO_M = 1e-3
BL_H_MM = 3.0
BL_H_IN = BL_H_MM * 0.0393701

start = time.time()


# Pre-allocated dictionaries
delta_n_mm_3_dict = {}
tau_w_dict = {}

# Tunables
DEBUG_PLOTS = False
stride = 1  # OPTIMIZATION TIP: Increase to 2-4 for faster testing
num_points = 800  # OPTIMIZATION TIP: Lower to 100-150 if acceptable

# ============================================================================
# Connect to Tecplot ONCE
# ============================================================================
try:
    tp.session.connect()
except Exception:
    pass

# ============================================================================
# OPTIMIZATION 3: Process in batches or parallel (pseudo-code structure)
# ============================================================================
# For true speedup, consider using concurrent.futures:
# from concurrent.futures import ProcessPoolExecutor
# with ProcessPoolExecutor(max_workers=4) as executor:
#     results = executor.map(process_case, cases)

# ============================================================================
# Main processing loop
# ============================================================================
for idx, key in enumerate(ds_by_case):
    if idx == 0: 
        print(f"{'='*40}\n{key}\n{'='*40}")
    
    print(f"Processing iteration {idx}...")
    
    # Load layout once per case
    tp.new_layout()
    tp.data.load_tecplot(file_paths[idx].as_posix())
    fr = tp.active_frame()
    fr.plot_type = PlotType.Cartesian2D

    # ========================================================================
    # OPTIMIZATION 4: Vectorize geometry processing
    # ========================================================================
    x0 = ds_by_case[key]["X"].data.astype(float).ravel()
    y0 = ds_by_case[key]["Y"].data.astype(float).ravel()
    
    # Filter bad values once
    good = np.isfinite(x0) & np.isfinite(y0)
    x0, y0 = x0[good], y0[good]

    # Reference values (computed once)
    P_ref = float(ds_by_case_inlet[key]["P"].data.ravel()[0])
    rho_ref = float(ds_by_case_inlet[key]["R"].data.ravel()[0])
    U_ref = float(ds_by_case_inlet[key]["U"].data.ravel()[0])
    V_ref = float(ds_by_case_inlet[key]["V"].data.ravel()[0])
    h0_ref = float(ds_by_case_inlet[key]["Enthalpy_total"].data.ravel()[0])

    # ========================================================================
    # OPTIMIZATION 5: Vectorized normal calculations
    # ========================================================================
    dx_ds = np.gradient(x0)
    dy_ds = np.gradient(y0)
    nx, ny = -dy_ds, dx_ds
    norm = np.hypot(nx, ny)
    
    # Avoid division by zero efficiently
    norm = np.where(norm < 1e-12, np.nan, norm)
    ux, uy = nx / norm, ny / norm

    # Rake endpoints (apply stride here)
    xf = x0 + BL_H_IN * ux
    yf = y0 + BL_H_IN * uy
    ok = np.isfinite(xf) & np.isfinite(yf)
    
    # Apply stride to reduce number of profiles
    x_start = x0[ok][::stride]
    y_start = y0[ok][::stride]
    x_end = xf[ok][::stride]
    y_end = yf[ok][::stride]

    # Build arrays efficiently
    n = min(len(x_start), len(y_start), len(x_end), len(y_end))
    x_pairs = np.column_stack((x_start[:n], x_end[:n]))
    y_pairs = np.column_stack((y_start[:n], y_end[:n]))

    # Pre-allocate output arrays
    N = x_pairs.shape[0]
    delta_n_mm_3_dict[key] = np.full(N, np.nan, dtype=float)
    tau_w_dict[key] = np.full(N, np.nan, dtype=float)

    # ========================================================================
    # OPTIMIZATION 6: Use suspend() to avoid UI redraws (you already have this!)
    # ========================================================================
    with tp.session.suspend():
        for i in range(N):
            # Convert to tuple once
            p0 = (float(x_pairs[i, 0]), float(y_pairs[i, 0]), 0.0)
            p1 = (float(x_pairs[i, 1]), float(y_pairs[i, 1]), 0.0)
            
            # Extract line
            line = tp.data.extract.extract_line([p0, p1], num_points=num_points)
            
            # Check wall shear stress
            tau_x_wall = line.values("Tau_x").as_numpy_array()[0]
            
            if tau_x_wall <= 0:
                print(f"  Point {i} skipped (separation)")
                continue
            
            # ================================================================
            # OPTIMIZATION 7: Extract all needed variables at once
            # ================================================================
            # This is already efficient, but ensure you're not calling .values() multiple times
            x_BL = line.values("X").as_numpy_array()
            y_BL = line.values("Y").as_numpy_array()
            U = line.values("U").as_numpy_array()
            V = line.values("V").as_numpy_array()
            P = line.values("P").as_numpy_array()
            Gam = line.values("Gamma").as_numpy_array()
            vort_z = line.values("Vort_z").as_numpy_array()
            Mutur = line.values("Mutur").as_numpy_array()
            Mut_ovr_Mu = line.values("Mut_ovr_Mu").as_numpy_array()
            
            # Compute mu efficiently (avoid division by zero)
            mu = np.where(Mut_ovr_Mu != 0, Mutur / Mut_ovr_Mu, Mutur)
            
            # Store wall shear stress
            tau_w_dict[key][i] = tau_x_wall

            # ================================================================
            # OPTIMIZATION 8: Ensure correct array order once
            # ================================================================
            if y_BL[0] > y_BL[-1]:
                # Reverse all arrays at once
                y_BL = y_BL[::-1]
                U = U[::-1]
                V = V[::-1]
                vort_z = vort_z[::-1]
            
            y_wall0 = y_BL[0]
            y_corr = y_BL - y_wall0
            
            # ================================================================
            # OPTIMIZATION 9: Compute derivatives efficiently
            # ================================================================
            domega_dy = np.gradient(vort_z, y_corr)
            y_edge_3, y_edge_idx_3 = find_bl_edge_curvature(y_corr, domega_dy)
            
            # Store result
            delta_n_mm_3_dict[key][i] = float(y_edge_3) * IN_TO_MM
        
        # Summary after processing all profiles
        valid = np.isfinite(delta_n_mm_3_dict[key])
        if np.any(valid):
            delta_valid = delta_n_mm_3_dict[key][valid]
            print(f"{key}: N={N}, δ(mm) ∈ [{np.min(delta_valid):.3f}, {np.max(delta_valid):.3f}]")

# ============================================================================
# Report timing
# ============================================================================
end = time.time()
elapsed = end - start
print(f"\n{'='*50}")
print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/3600:.2f} hours)")
print(f"{'='*50}")









#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Post-processing the boundary layer thickness results
#------------------------------------------------------------------------------------------------------------------------------------#
"""





# === Separation/Attachment from sign of Tau_x (no splines), ignoring edge pairs ===
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- helpers ---
def get_hl(section_key: str):
    # First try numeric pattern: h_l_0.180
    m = re.search(r'h_l_([0-9.]+)', section_key)
    if m:
        return float(m.group(1))
    
    # Then check for h_l_x pattern (return a placeholder value or string)
    if 'h_l_x' in section_key:
        return 'x'  # or return a specific numeric value if you know what h/L this represents
    
    return None

def clean_xy_from_ds(section_key):
    """Use X and Tau_x from the SAME dataset; drop NaN/Inf, sort by X, average duplicate X."""
    x = np.asarray(ds_by_case[section_key]["X"].data).ravel()
    y = np.asarray(ds_by_case[section_key]["Tau_x"].data).ravel()
    
    # Filter to x range [0, 0.1]
    lower_bound = 0
    upper_bound = 0.1
    
    condition_1 = (x >= lower_bound)   # ✓ Fixed: use lower_bound
    condition_2 = (x <= upper_bound)   # ✓ Renamed for clarity
    
    boolean_mask = condition_1 & condition_2
    x = x[boolean_mask]
    y = y[boolean_mask]  # ✓ Fixed: apply mask to both arrays
    
    # Remove NaN/Inf
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    # Sort by x
    s = np.argsort(x)
    x, y = x[s], y[s]

    # Average duplicate x values
    xu, inv = np.unique(x, return_inverse=True)
    yu = np.bincount(inv, weights=y) / np.bincount(inv)
    return xu, yu

def x_at_zero(i, x, y):
    """Linear interpolation of zero between samples i and i+1."""
    x0, x1 = x[i], x[i+1]
    y0, y1 = y[i], y[i+1]
    if y1 == y0:
        return x0  # plateau fallback
    t = y0 / (y0 - y1)
    return x0 + t * (x1 - x0)

# --- outputs (your names) ---
idx_separation = {}
x_sep_points = {}
x_all = {}
sep_location = {}
tau_w_zeros = {}

sep_length_mm = {}
sep_length_nonDim = {}

x_attach = {}
y_attach = {}
x_sep = {}
y_sep = {}



# --- main loop ---
for section_key in ds_by_case:
    if get_hl(section_key) is None:
        continue

    # cleaned series from SAME dataset
    x_clean, tau_clean = clean_xy_from_ds(section_key)
    x_all[section_key] = x_clean

    # sign of tau: negative segments define separated regions
    
    neg = tau_clean < 0
    edge = np.diff(neg.astype(np.int8))

    # +1: +→- (enter negative) = SEP,  -1: -→+ (exit negative) = ATTACH
    i_sep    = np.where(edge == +1)[0]
    i_attach = np.where(edge == -1)[0]

    # interpolate exact zero x-positions
    x_sep_i    = np.array([x_at_zero(i, x_clean, tau_clean) for i in i_sep], dtype=float)
    x_attach_i = np.array([x_at_zero(i, x_clean, tau_clean) for i in i_attach], dtype=float)

    # handle boundary negatives (start/end inside a negative interval)
    if neg[0]:
        if x_sep_i.size == 0 or (x_attach_i.size and x_attach_i[0] < x_sep_i[0]):
            x_sep_i = np.r_[x_clean[0], x_sep_i]
    if neg[-1]:
        if x_attach_i.size == 0 or (x_sep_i.size and x_sep_i[-1] > x_attach_i[-1]):
            x_attach_i = np.r_[x_attach_i, x_clean[-1]]

    # pair in order (guaranteed alternating by construction)
    n = min(x_sep_i.size, x_attach_i.size)
    x_sep_i, x_attach_i = x_sep_i[:n], x_attach_i[:n]

    # --------- NEW: drop first/last pairs if they touch the domain edges ---------
    if n > 0:
        xmin, xmax = x_clean[0], x_clean[-1]
        xrng = max(xmax - xmin, 1.0)
        tol = 1e-6 * xrng  # tolerance for "at the boundary"

        # drop leading boundary pair
        if abs(x_sep_i[0] - xmin) <= tol or abs(x_attach_i[0] - xmin) <= tol:
            x_sep_i    = x_sep_i[1:]
            x_attach_i = x_attach_i[1:]
            n = min(x_sep_i.size, x_attach_i.size)

        # drop trailing boundary pair
        if n > 0 and (abs(x_sep_i[-1] - xmax) <= tol or abs(x_attach_i[-1] - xmax) <= tol):
            x_sep_i    = x_sep_i[:-1]
            x_attach_i = x_attach_i[:-1]
            n = min(x_sep_i.size, x_attach_i.size)
    # ---------------------------------------------------------------------------

    # store sets
    x_sep[section_key]    = x_sep_i
    y_sep[section_key]    = np.zeros_like(x_sep_i)
    x_attach[section_key] = x_attach_i
    y_attach[section_key] = np.zeros_like(x_attach_i)

    # all zero-locations (for plotting like before)
    if n > 0:
        x0_all = np.sort(np.concatenate([x_sep_i, x_attach_i]))
    else:
        x0_all = np.array([], dtype=float)
    sep_location[section_key] = x0_all
    tau_w_zeros[section_key]  = np.zeros_like(x0_all)

    # separation length (sum of SEP→ATTACH spans)
    sep_len = float(np.sum(np.abs(x_sep_i - x_attach_i))) if n > 0 else 0.0
    sep_length_mm[section_key]     = sep_len
    sep_length_nonDim[section_key] = (sep_len / x_clean[-1]) if x_clean.size else np.nan

    # optional: index-based separation mask in cleaned data
    idx_separation[section_key] = np.where(tau_clean <= 0)[0]

    # keep alias if you use it elsewhere
    x_sep_points[section_key] = x_sep_i

    # ---- plot (cleaned data) ----
    plt.plot(x_clean, tau_clean, label=r'$\tau_y$')
    if x_sep_i.size:
        plt.scatter(x_sep_i, np.zeros_like(x_sep_i), color='red', s=36, label='SEP', zorder=3)
    if x_attach_i.size:
        plt.scatter(x_attach_i , np.zeros_like(x_attach_i), color='green', s=36, label='ATTACH', zorder=3)
    plt.axhline(0, linestyle='--', color='black', label='Separation Line')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.grid(True, which="both")
    plt.title(rf"$\tau_x$ vs X [in]: {section_key}")
    plt.ylabel(r"$\tau_x$ [Pa]")
    plt.xlabel("X [m]")
    plt.tight_layout()
    plt.show()



#%% 

"""
#------------------------------------------------------------------------------------------------------------------------------------#
    Evaluating the following values: MAX BL Thickness Vs Re, Sep Length vs Re, BL thickness Vs X (With shear stress filtering)
#------------------------------------------------------------------------------------------------------------------------------------#
"""



fig_save_dir = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\07_Python Codes\01_Python Automation Codes\03_Post-Processing Code\2_Mach Number Code\1_Graphs\1_Separation Length"

# Pre-Allocating Variables # 
Re_L = {}
keys = ds_by_case.keys()


for section_key in ds_by_case:
    # Computing Reynolds number in respect of length # 
    #mu = (1 / ds_by_case_inlet[key]["Mut_ovr_Mu"].data) * ds_by_case_inlet[section_key]["Mutur"].data
    U = ds_by_case_inlet[section_key]["U"].data
    rho = ds_by_case_inlet[section_key]["R"].data
    
    #Re_L[section_key] =  (np.mean(rho) * (np.mean(U) ) * ds_by_case[section_key]["X"].data * in_to_mm ) / np.mean(mu)

    # Filtering delta_n_mm to remove points at which separation does occur # 
    #delta_n_mm_filtered = np.delete(delta_n_mm[section_key] , idx_separation[section_key])
    #delta_n_mm_sep = np.array(delta_n_mm[section_key])[idx_separation[section_key]]
    
    #Re_L_filtered = np.delete(Re_L[section_key] , idx_separation[section_key])
    #Re_L_sep = Re_L[section_key][idx_separation[section_key]]
    
    
    # Reynolds Number Vs Boundary layer thickness # 
    #plt.plot(Re_L[section_key] , delta_n_mm[section_key], color = 'black',linewidth = 2)
    plt.scatter(ds_by_case[section_key]["X"].data[:len(delta_n_mm_3_dict[section_key])], delta_n_mm_3_dict[section_key],label = 'Attached')
    #plt.scatter(Re_L_sep , delta_n_mm_sep , color = 'red', label = 'Separated')
    plt.legend()
    
    plt.title(f"BL Thickness Vs Re: {section_key}")
    plt.xlabel(r"$X$")
    plt.ylabel(r"$\delta_e [mm]$")
    plt.grid(True, which = "both")
    #plt.savefig(os.path.join(fig_save_dir,f"{section_key}_bl_Re.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.cla()

#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                Finding MAXIMAS AND MINIMAS OF THE GEOMETRY 
#------------------------------------------------------------------------------------------------------------------------------------#
"""
# Prior to plotting results, I am going to find the maximas and minimas to accurately represent the separation length
# The definition of the separation length will be the separation length of the second wave
# To do so, the maximas and minimas of each respective geometry will be found and evaluated.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks  # <-- simple peak/valley finder

# --- Pre-allocating Dictionaries ---

# In mm
y_max, x_max = {}, {}
y_min, x_min = {}, {}

# In inches
x_max_in, y_max_in = {}, {}
x_min_in, y_min_in = {}, {}

# Unit conversion


# Using a for loop to find the y_max and x_max of each respective geometry
for pressure_key in ds_by_case:

    # --- Units in mm ---
    x_all = np.asarray(ds_by_case[pressure_key]["X"].data)
    y_all = np.asarray(ds_by_case[pressure_key]["Y"].data)

    # clean + sort by x so indices run left→right
    mask = np.isfinite(x_all) & np.isfinite(y_all)
    x_all, y_all = x_all[mask], y_all[mask]
    order = np.argsort(x_all)
    x_all, y_all = x_all[order], y_all[order]

    # SIMPLE peak picking
    i_max, _ = find_peaks(y_all)       # local maxima
    i_min, _ = find_peaks(-y_all)      # local minima

    # (If you need a tiny bit more robustness, uncomment one of these one-liners)
    # i_max, _ = find_peaks(y_all, distance=20)                       # enforce min spacing
    # i_max, _ = find_peaks(y_all, prominence=0.05*np.ptp(y_all))     # ignore tiny ripples
    # i_max, _ = find_peaks(y_all, plateau_size=1)                    # detect flat tops
    # i_min, _ = find_peaks(-y_all, distance=20)
    # i_min, _ = find_peaks(-y_all, prominence=0.05*np.ptp(y_all))
    # i_min, _ = find_peaks(-y_all, plateau_size=1)

    # store mm
    y_max[pressure_key] = y_all[i_max]
    x_max[pressure_key] = x_all[i_max] 
    #x_max[pressure_key] = 0.015
    y_min[pressure_key] = y_all[i_min]
    x_min[pressure_key] = x_all[i_min]


print(x_max)

#%%



"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                Plotting Separation Length Vs Re_L
#------------------------------------------------------------------------------------------------------------------------------------#
"""



# Plot Re vs Lsep/Lwidth with markers colored by Mach and h/L in the legend
# Plot Re vs Lsep/Lwidth with markers colored by Mach (1.5–4.5).
# h/L groups are shown as separate lines with a legend.

# Re vs Lsep/Lwidth with markers colored by Mach (1.5…4.5) and lines per h/L.

# This version avoids regex brittleness by matching known Mach tags directly.


import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable



# ---------------- helpers ---------------- #
def get_hl(key: str):
    # Handles negative numbers, decimals, scientific notation
    m = re.search(r'h[_-]?l[_-]?(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', key, flags=re.I)
    return float(m.group(1)) if m else None

# Simple & robust Mach extractor:
# checks for 'mach_1.5', 'mach_2.0', … and also 'mach_1_5' style.
MACH_LEVELS = np.arange(1.5, 4.5 , 0.5)  # 1.5, 2.0, …, 4.5
def extract_mach_from_filename(key: str, mach_levels: list) -> float:
    """
    Extract Mach number from filename.
    
    Args:
        key: String containing Mach number (e.g., 'h_l_0.030_Mach_1.0' or 'h_l_0.030_Mach_1_5')
        mach_levels: List of possible Mach values to search for
    
    Returns:
        float: Extracted Mach number or np.nan if not found
    
    Examples:
        >>> extract_mach_from_filename("h_l_0.030_Mach_1.0", [1.0, 1.5, 2.0])
        1.0
        >>> extract_mach_from_filename("h_l_0.030_Mach_2_5", [1.0, 1.5, 2.0, 2.5])
        2.5
    """
    s = key.lower()
    
    for mv in mach_levels:
        # Create pattern variations
        # For mv=2.5: creates "mach_2.5" and "mach_2_5"
        # For mv=1.0: creates "mach_1.0" and "mach_1_0"
        tag_dot = f"mach_{mv:.1f}"        # e.g., "mach_2.5"
        tag_us = tag_dot.replace(".", "_") # e.g., "mach_2_5"
        
        # Check if either pattern exists in the filename
        if tag_dot in s or tag_us in s:
            return float(f"{mv:.1f}")
    
    return np.nan




# ---------------- filter + group by h/L ----------------
min_hl, max_hl = 0.02, 0.09

keys = ds_by_case.keys()
filtered_keys = [
    k for k in keys
    if (get_hl(k) is not None and min_hl <= get_hl(k) <= max_hl and "Mach_0.5" not in k)
]

groups = defaultdict(list)
for k in filtered_keys:
    groups[get_hl(k)].append(k)

# ---------------- discrete Mach mapping for colorbar ----------------
# Create discrete bins with boundaries half-way between the levels.
M0_bounds = np.linspace(1.5, 4.0, len(MACH_LEVELS) + 1)
cmap_mach = cm.get_cmap("viridis", len(MACH_LEVELS))   # 7 distinct colors
norm = BoundaryNorm(M0_bounds, cmap_mach.N)



# line colors per h/L (legend)
cmap_lines = cm.get_cmap("plasma", len(groups))

# ---------------- plotting ----------------
fig, ax = plt.subplots(figsize=(8, 6))
scatter_ref = None
unmatched = 0

for i, (hl, key_list) in enumerate(sorted(groups.items())):
    xs, ys, ms = [], [], []
    print(hl)
    for k in key_list:
        if k in Re and k in sep_length_nonDim:
            
            # Finding separation legnth after the first wave #
            
            
            # Computing the filtered x_sep and y_sep # 
            mask_x = (x_sep[k] > x_max[k][0]) & (x_sep[k] < x_max[k][1])
            mask_y = (y_sep[k] > y_max[k][0]) & (y_sep[k] < y_max[k][1])
            
            x_sep_filtered = x_sep[k][mask_x]
            y_sep_filtered = y_sep[k][mask_y]
            
            x_attach_filtered = x_attach[k][mask_x]
            y_attach_filtered = y_attach[k][mask_y]
            
            # Non-dimensionalized Separation Length # 
            sep_length_nonDim_filtered = np.sum(abs(x_sep_filtered - x_attach_filtered))/(0.1)
            
            
            yval = sep_length_nonDim_filtered


            xs.append(Re[k])
            ys.append(yval)

            dv = extract_mach_from_filename(k,MACH_LEVELS)
            ms.append(dv)
            if not np.isfinite(dv):
                unmatched += 1

    if not xs:
        continue

    xs = np.asarray(xs); ys = np.asarray(ys); ms = np.asarray(ms)

    # line: use finite x/y, sorted by Re
    mask_xy = np.isfinite(xs) & np.isfinite(ys)
    xs_line, ys_line = xs[mask_xy], ys[mask_xy]
    order = np.argsort(xs_line)
    ax.plot(xs_line[order], ys_line[order],
            color=cmap_lines(i), lw=4, label=f"h/L = {hl:.2f}")

    # markers: color by Mach  #
    mask_col = mask_xy & np.isfinite(ms)
    if np.any(mask_col):
        sc = ax.scatter(xs[mask_col], ys[mask_col], c=ms[mask_col],
                        cmap=cmap_mach, norm=norm,
                        s=70, marker='o', edgecolor='k', linewidths= 1.35,
                        zorder=5, alpha=0.98)
        scatter_ref = sc
        

        
        
        
    # (Optional) show unmatched as neutral markers so you can spot them
    mask_neu = mask_xy & ~np.isfinite(ms)
    if np.any(mask_neu):
        ax.scatter(xs[mask_neu], ys[mask_neu],
                   color='white', edgecolor='k', linewidths=0.35,
                   s=46, marker='o', zorder=4, alpha=0.9)

# ---------------- colorbar with exact Pressure ticks ----------------
if scatter_ref is not None:
    cbar = fig.colorbar(scatter_ref, ax=ax, pad=0.02, ticks=MACH_LEVELS)
    cbar.set_label("Mach Number", fontsize = 18)
    cbar.ax.tick_params(labelsize = 18)
else:
    # fallback colorbar so layout stays stable; also tell you why
    sm = ScalarMappable(norm=norm, cmap=cmap_mach); sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, ticks=MACH_LEVELS).set_label("Mach")
    print("Warning: no markers received a Mach color. Check key naming.")




# ---------------- cosmetics ----------------
ax.set_title("Normalized Separation Length vs Reynolds Number", fontsize = 24, pad = 15)
ax.set_xlabel("Reynolds Number", fontsize = 21)
ax.set_ylabel(r"$L_{separation}/L_{Length}$", fontsize = 21)
ax.tick_params(labelsize = 14)

ax.grid(True, which="both")
ax.legend(title="Cases", loc = 'best', bbox_to_anchor = (0.4, 0.3), fontsize = 12)
fig.tight_layout()
plt.show()

# Optional: see how many keys didn’t contain a Mach tag
if unmatched:
    print(f"{unmatched} case(s) had no recognizable Mach tag (e.g., 'mach_2.5').")
    
    

#%% Plotting separation length versus Mach number only!
# Plot Lsep/Lwidth vs Mach Number with lines for each h/L

import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------- helpers ---------------- #
def get_hl(key: str):
    m = re.search(r'h[_-]?l[_-]?(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', key, flags=re.I)
    return float(m.group(1)) if m else None

MACH_LEVELS = np.arange(1.5, 4.5, 0.5)  # 1.5, 2.0, …, 4.0

def extract_mach_from_filename(key: str, mach_levels: list) -> float:
    s = key.lower()
    for mv in mach_levels:
        tag_dot = f"mach_{mv:.1f}"
        tag_us = tag_dot.replace(".", "_")
        if tag_dot in s or tag_us in s:
            return float(f"{mv:.1f}")
    return np.nan

# ---------------- filter + group by h/L ----------------
min_hl, max_hl = 0.02, 0.09

keys = ds_by_case.keys()
filtered_keys = [
    k for k in keys
    if (get_hl(k) is not None and min_hl <= get_hl(k) <= max_hl and "Mach_0.5" not in k)
]

groups = defaultdict(list)
for k in filtered_keys:
    groups[get_hl(k)].append(k)

# ---------------- style definitions ----------------
# Colors for each h/L value
hl_values = sorted(groups.keys())
cmap_lines = cm.get_cmap("viridis", len(hl_values))

# Different markers for each h/L
markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

# ---------------- plotting ----------------
fig, ax = plt.subplots(figsize=(10, 7))

for i, (hl, key_list) in enumerate(sorted(groups.items())):
    mach_vals = []
    sep_lengths = []
    
    for k in key_list:
        if k in Re and k in sep_length_nonDim:
            
            # Computing the filtered x_sep and y_sep
            mask_x = (x_sep[k] > x_max[k][0]) & (x_sep[k] < x_max[k][1])
            mask_y = (y_sep[k] > y_max[k][0]) & (y_sep[k] < y_max[k][1])
            
            x_sep_filtered = x_sep[k][mask_x]
            y_sep_filtered = y_sep[k][mask_y]
            
            x_attach_filtered = x_attach[k][mask_x]
            y_attach_filtered = y_attach[k][mask_y]
            
            # Non-dimensionalized Separation Length
            sep_length_nonDim_filtered = np.sum(abs(x_sep_filtered - x_attach_filtered)) / 0.1
            
            # Extract Mach number
            mach = extract_mach_from_filename(k, MACH_LEVELS)
            
            if np.isfinite(mach) and np.isfinite(sep_length_nonDim_filtered):
                mach_vals.append(mach)
                sep_lengths.append(sep_length_nonDim_filtered)
    
    if not mach_vals:
        continue
    
    mach_vals = np.asarray(mach_vals)
    sep_lengths = np.asarray(sep_lengths)
    
    # Sort by Mach number for proper line connection
    order = np.argsort(mach_vals)
    mach_sorted = mach_vals[order]
    sep_sorted = sep_lengths[order]
    
    # Plot line + markers
    ax.plot(mach_sorted, sep_sorted,
            color=cmap_lines(i),
            marker=markers[i % len(markers)],
            markersize=12,
            linewidth=2.5,
            markeredgecolor='black',
            markeredgewidth=1.2,
            label=f"h/l = {hl:.2f}")

# ---------------- formatting ----------------
ax.set_xlabel("Mach Number", fontsize=21)
ax.set_ylabel(r"$L_{separation} / L_{length}$", fontsize=21)
ax.set_title("Normalized Separation Length vs Mach Number", fontsize=28, fontweight='bold')
ax.tick_params(labelsize=20)

# Set x-axis to show all Mach values
ax.set_xticks(MACH_LEVELS)
ax.set_xlim([MACH_LEVELS[0] - 0.2, MACH_LEVELS[-1] + 0.2])

ax.grid(True, alpha=0.3)
ax.legend(title="h/l", fontsize=12, title_fontsize=14, loc='best')

fig.tight_layout()
plt.savefig('sep_length_vs_mach.png', dpi=200, bbox_inches='tight')
plt.show()


















#%%
#%% Single h/L Geometry Analysis (by name pattern)
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
def extract_mach_from_filename(s, mach_levels):
    """Extract Mach number from key string, handling multiple formats."""
    s_lower = s.lower()  # Case-insensitive matching
    
    for mv in mach_levels:
        # Format variations to check:
        # "mach_2.5", "mach_2_5", "mach 2.5", "mach2.5"
        patterns = [
            f"mach_{mv:.1f}",           # mach_2.5
            f"mach_{mv:.1f}".replace(".", "_"),  # mach_2_5
            f"mach {mv:.1f}",           # mach 2.5 (your format)
            f"mach{mv:.1f}",            # mach2.5
        ]

        
        for pat in patterns:
            if pat in s_lower:
                return float(mv)
    
    return np.nan
# Define Mach levels and colormap
MACH_LEVELS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
M0_bounds = np.linspace(1.5, 4.0, len(MACH_LEVELS) + 1)
cmap_mach = cm.get_cmap("viridis", len(MACH_LEVELS))
norm = BoundaryNorm(M0_bounds, cmap_mach.N)

# Specify the h/L pattern to match in the key names
target_pattern = "h_l_x"
key_list = [k for k in ds_by_case.keys() if target_pattern in k]
key_list = [k for k in key_list if "Mach 1.0" not in k]  # Exclude Mach 1.0

print(f"Found {len(key_list)} cases matching '{target_pattern}':")
for k in key_list:
    print(f"  - {k}")

# Process and plot
fig, ax = plt.subplots(figsize=(8, 6))
xs, ys, ms = [], [], []

for k in key_list:
    if k in Re and k in sep_length_nonDim:
        mask_x = (x_sep[k] > x_max[k][0]) & (x_sep[k] < x_max[k][1])
        
        x_sep_filtered = x_sep[k][mask_x]
        x_attach_filtered = x_attach[k][mask_x]
        
        sep_length_normalized = np.sum(np.abs(x_sep_filtered - x_attach_filtered)) / ds_by_case[k]["X"][-1].data
        
        xs.append(Re[k])
        ys.append(sep_length_normalized)
        ms.append(extract_mach_from_filename(k, MACH_LEVELS))

# Convert to arrays
xs = np.asarray(xs)
ys = np.asarray(ys)
ms = np.asarray(ms)

# Plot line (sorted by Re)
mask_xy = np.isfinite(xs) & np.isfinite(ys)
order = np.argsort(xs[mask_xy])
ax.plot(xs[mask_xy][order], ys[mask_xy][order], 'b-', lw=2)

# Scatter colored by Mach
mask_col = mask_xy & np.isfinite(ms)
if np.any(mask_col):
    sc = ax.scatter(xs[mask_col], ys[mask_col], c=ms[mask_col],
                    cmap=cmap_mach, norm=norm,
                    s=60, marker='o', edgecolor='k', linewidths=0.5, zorder=5)
    cbar = fig.colorbar(sc, ax=ax, ticks=MACH_LEVELS)
    cbar.set_label("Mach Number", fontsize=14)
else:
    # Fallback: create colorbar even if no scatter points matched
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(norm=norm, cmap=cmap_mach)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=MACH_LEVELS)
    cbar.set_label("Mach Number", fontsize=14)
    print("Warning: No Mach values matched. Check extract_mach_from_filename().")

ax.set_title(f"Separation Length vs Re for {target_pattern}", fontsize=16)
ax.set_xlabel("Re", fontsize=14)
ax.set_ylabel(r"$L_{separation}/L_{width}$", fontsize=14)
ax.grid(True)
fig.tight_layout()
plt.show()

#%% Process and plot this single h/L group
fig, ax = plt.subplots(figsize=(8, 6))

xs, ys, ms = [], [], []

for k in key_list:
    if k in Re and k in sep_length_nonDim:
        # Filter to region between first and third wave peaks
        mask_x = (x_sep[k] > x_max[k][0]) & (x_sep[k] < x_max[k][1])
        
        x_sep_filtered = x_sep[k][mask_x]
        x_attach_filtered = x_attach[k][mask_x]
        
        # Normalized separation length
        sep_length_normalized = np.sum(np.abs(x_sep_filtered - x_attach_filtered)) / ds_by_case[k]["X"][-1].data
        
        xs.append(Re[k])
        ys.append(sep_length_normalized)
        ms.append(extract_mach_from_filename(k, MACH_LEVELS))

# Convert to arrays
xs = np.asarray(xs)
ys = np.asarray(ys)
ms = np.asarray(ms)

# Plot line (sorted by Re)
mask_xy = np.isfinite(xs) & np.isfinite(ys)
order = np.argsort(xs[mask_xy])
ax.plot(xs[mask_xy][order], ys[mask_xy][order], 'b-', lw=2)

# Scatter colored by Mach
mask_col = mask_xy & np.isfinite(ms)
if np.any(mask_col):
    sc = ax.scatter(xs[mask_col], ys[mask_col], c=ms[mask_col],
                    cmap=cmap_mach, norm=norm,
                    s=60, marker='o', edgecolor='k', linewidths=0.5, zorder=5)
    cbar = fig.colorbar(sc, ax=ax, ticks=MACH_LEVELS)
    cbar.set_label("Mach Number", fontsize=14)

ax.set_title(f"Separation Length vs Re for h/L = {target_hl}", fontsize=16)
ax.set_xlabel("Re", fontsize=14)
ax.set_ylabel(r"$L_{separation}/L_{width}$", fontsize=14)
ax.grid(True)
fig.tight_layout()
plt.show()





#%%


"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Plotting the location of the first separation sensitivity
#------------------------------------------------------------------------------------------------------------------------------------#
"""



temp_keys = cases_by_hl["h_l_0.04"]

plt.figure(figsize=(8,6))
for temp_key in temp_keys:
    # Computing the filtered x_sep and y_sep # 
    mask_x = (x_sep[temp_key] > x_max[temp_key][0]) & (x_sep[temp_key] < x_max[temp_key][1])
    mask_y = (y_sep[temp_key] > y_max[temp_key][0]) & (y_sep[temp_key] < y_max[temp_key][1])
    
    
    firstSepPointX = x_attach[temp_key][mask_x]
    firstSepPointY = y_sep[temp_key][mask_y]
    
    x_temp = ds_by_case[temp_key]["X"].data
    y_temp = ds_by_case[temp_key]["Y"].data
    Re_temp = Re[temp_key]
    
    
    plt.plot(x_temp,y_temp)
    plt.scatter(firstSepPointX,firstSepPointY, color = 'red', label = f"Re = {Re_temp}",zorder=5)
plt.xlabel("X (inches)")
plt.ylabel("Y (inches)")
plt.title("All Cases on One Graph")
plt.legend()
plt.grid(True)
plt.show()

#%% GPT: Getting the first point at which separation occurs. 
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- your helper kept as-is ---
def y_at_x_on_polyline(x, y, x_star):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    xdiff = x - x_star
    seg_idx = np.where((xdiff[:-1] * xdiff[1:]) <= 0)[0]
    if seg_idx.size:
        k = seg_idx
        x0, x1 = x[k], x[k+1]
        y0, y1 = y[k], y[k+1]
        vertical = (x1 == x0)
        out = np.empty(k.size, float)
        if np.any(~vertical):
            t = (x_star - x0[~vertical]) / (x1[~vertical] - x0[~vertical])
            out[~vertical] = y0[~vertical] + t * (y1[~vertical] - y0[~vertical])
        if np.any(vertical):
            out[vertical] = 0.5*(y0[vertical] + y1[vertical])
        seg_dist = np.minimum(np.abs(xdiff[k]), np.abs(xdiff[k+1]))
        return out[np.argmin(seg_dist)]
    j = int(np.argmin(np.abs(xdiff)))
    return y[j]

# ---- inputs ----
temp_keys = cases_by_hl["h_l_0.09"]



# Track which legend labels we’ve already used
seen_sep   = set()   # Mach values that had a separation point plotted
seen_nosep = set()   # Mach values that had no separation in the window

plt.figure()
for i, temp_key in enumerate(temp_keys):
    # parse Pressure values (for legend & dedupe)
    m = re.search(r"Mach_([0-9]*\.?[0-9]+)", temp_key)
    mach_string = m.group(1) if m else "?"
    mach_val = float(mach_string) if m else np.nan

    # separation x-locations and window bounds (same units!)
    xsep = np.asarray(x_sep[temp_key]).ravel()
    xmax = np.asarray(x_max[temp_key]).ravel()

    # geometry curve
    x_temp = np.asarray(ds_by_case[temp_key]["X"].data)
    y_temp = np.asarray(ds_by_case[temp_key]["Y"].data)
    x_temp_in = x_temp
    y_temp_in = y_temp

    # plot the geometry once per case (fine if repeated)
    plt.plot(x_temp_in, y_temp_in, color="red", linewidth=3.0)

    # window mask: between the first two maxima
    if xmax.size >= 2:
        lo, hi = np.sort(xmax[:2])
        mask_new = np.isfinite(xsep) & (xsep > lo) & (xsep < hi)
        x_sep_filtered = xsep[mask_new]
    else:
        x_sep_filtered = np.array([], dtype=float)

    Re_temp = Re[temp_key]

    if x_sep_filtered.size:
        # Take leftmost separation in the window
        firstSepPointX = float(np.min(x_sep_filtered))
        firstSepPointY = y_at_x_on_polyline(x_temp_in, y_temp_in, firstSepPointX)

        # Label each Mach ONCE for "sep" cases
        label = f"M = {mach_string}"
        if mach_val in seen_sep:
            label = "_nolegend_"
        else:
            seen_sep.add(mach_val)

        plt.scatter(firstSepPointX, firstSepPointY,
                    label=label, zorder=5, color=cmap(i), edgecolor="k", linewidths=0.3)
    else:
        # Optional: mark mid-window with a small x so reader sees "no sep" region
        if xmax.size >= 2:
            midx = 0.5*(lo + hi)
            midy = y_at_x_on_polyline(x_temp, y_temp, midx)
            plt.scatter([midx], [midy], marker="x", color="0.6", zorder=4, s=30)

        # Add ONE legend entry per Mach for "(no sep)"
        label = f"M = {mach_string} (no sep)"
        if mach_val in seen_nosep:
            label = "_nolegend_"
        else:
            seen_nosep.add(mach_val)

        # create a legend handle without adding a visible extra point:
        plt.scatter([], [], color=cmap(i), label=label)

plt.xlabel("X [m]", fontsize = 18)
plt.ylabel("Y [m]", fontsize = 18)
plt.title(r"First Separation Point: $h_{L}$ = 0.09",fontsize = 21)
plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
plt.grid(True)
plt.tight_layout()
plt.show()


 #%% First point separation but for all h/ls in one SUBPLOT! ####################


"""
#------------------------------------------------------------------------------------------------------------------------------------#
                         Plotting the location of the first separation sensitivity in a subplot for all h/ls
#------------------------------------------------------------------------------------------------------------------------------------#
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
# --- your helper kept as-is ---
def y_at_x_on_polyline(x, y, x_star):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    xdiff = x - x_star
    seg_idx = np.where((xdiff[:-1] * xdiff[1:]) <= 0)[0]
    if seg_idx.size:
        k = seg_idx
        x0, x1 = x[k], x[k+1]
        y0, y1 = y[k], y[k+1]
        vertical = (x1 == x0)
        out = np.empty(k.size, float)
        if np.any(~vertical):
            t = (x_star - x0[~vertical]) / (x1[~vertical] - x0[~vertical])
            out[~vertical] = y0[~vertical] + t * (y1[~vertical] - y0[~vertical])
        if np.any(vertical):
            out[vertical] = 0.5*(y0[vertical] + y1[vertical])
        seg_dist = np.minimum(np.abs(xdiff[k]), np.abs(xdiff[k+1]))
        return out[np.argmin(seg_dist)]
    j = int(np.argmin(np.abs(xdiff)))
    return y[j]

# ---- inputs ----
h_l_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

# Create subplot grid (2 rows x 4 columns for 8 h/l values)
fig, axes = plt.subplots(2, 4, figsize=(28, 14))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

# Create a shared colormap for Mach numbers across all subplots
# Adjust the number based on how many Mach cases you have
n_mach_cases = 9  # Adjust this to match your data
cmap = cm.get_cmap("Spectral", 6)

for idx, h_l in enumerate(h_l_values):
    ax = axes[idx]
    
    # Get the key for this h/l value
    h_l_key = f"h_l_{h_l:.2f}"
    temp_keys = cases_by_hl[h_l_key]
    
 
    
    # Track which legend labels we've already used (reset for each subplot)
    seen_sep = set()
    seen_nosep = set()
    
    for i, temp_key in enumerate(temp_keys):
        # Parse Mach values (for legend & dedupe)
        m = re.search(r"Mach_([0-9]*\.?[0-9]+)", temp_key)
        mach_string = m.group(1) if m else "?"
        mach_val = float(mach_string) if m else np.nan
        
        # Separation x-locations and window bounds
        xsep = np.asarray(x_sep[temp_key]).ravel()
        xmax = np.asarray(x_max[temp_key]).ravel()
        
        # alpha list # 
        alpha_list = np.linspace(0.8,0.3,len(temp_key))
        
        
        # Geometry curve
        min_l = 0.0
        max_l = 0.1
        
        jitter_strength = 0.1
        
        x_temp = np.asarray(ds_by_case[temp_key]["X"].data)
        y_temp = np.asarray(ds_by_case[temp_key]["Y"].data)
        
        # Creating an alpha list to make the scatter plot more clear #
        
        
        mask = (x_temp < max_l) & (x_temp > min_l)
        x_temp_in = x_temp[mask]
        y_temp_in = y_temp[mask]
        
        # Plot the geometry once per case
        ax.plot(x_temp_in, y_temp_in, color="blue", linewidth= 4.0)
        
        # Window mask: between the first two maxima
        if xmax.size >= 2:
            lo, hi = np.sort(xmax[:2])
            mask_new = np.isfinite(xsep) & (xsep > lo) & (xsep < hi)
            x_sep_filtered = xsep[mask_new]
        else:
            x_sep_filtered = np.array([], dtype=float)
        
        Re_temp = Re[temp_key]
        
        if x_sep_filtered.size:
            # Take leftmost separation in the window
            firstSepPointX = float(np.min(x_sep_filtered))
            firstSepPointY = y_at_x_on_polyline(x_temp_in, y_temp_in, firstSepPointX)
            
            # Label each Mach ONCE for "sep" cases
            label = f"{mach_string}"
            if mach_val in seen_sep:
                label = "_nolegend_"
            else:
                seen_sep.add(mach_val)
            
            ax.scatter(firstSepPointX, firstSepPointY,
                       label=label, zorder=5, color=cmap(i), 
                       edgecolor="k", linewidths=1, s=650, alpha = alpha_list[i])
        else:
            # Mark mid-window with a small x for "no sep" region
            if xmax.size >= 2:
                midx = 0.5 * (lo + hi)
                midy = y_at_x_on_polyline(x_temp, y_temp, midx)
                ax.scatter([midx], [midy], marker="x", color="0.3", zorder=4, s=600)
            
            # Add ONE legend entry per Mach for "(no sep)"
            label = f"M = {mach_string} (no sep)"
            if mach_val in seen_nosep:
                label = "_nolegend_"
            else:
                seen_nosep.add(mach_val)
            
            ax.scatter([], [], color=cmap(i), label=label)
    
    # Subplot formatting
    ax.set_title(f"h/l = {h_l:.2f}", fontsize=48, fontweight='bold')
    ax.set_xlabel("X [m]", fontsize=34)
    ax.set_ylabel("Y [m]", fontsize=34)
    ax.tick_params(labelsize=34)
    ax.grid(True)


# Add a single shared legend outside the subplots
# Collect handles and labels from the last subplot (or any subplot)
handles, labels = axes[-1].get_legend_handles_labels()

# Add a custom handle for the "x" marker (no separation cases)
no_sep_handle = Line2D([0], [0], marker='x', color='0.3', linestyle='', 
                       markersize=15, markeredgewidth=2, label='No separation')
handles.append(no_sep_handle)
labels.append('No separation')


# Add a single shared legend outside the subplots
# Collect handles and labels from the last subplot (or any subplot)
#handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.10, 0.5), 
           fontsize=38, title="Mach Number", title_fontsize = 48)

# Main title for the entire figure
fig.suptitle("First Separation Point vs h/l", fontsize=58, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(right=0.88)  # Make room for the legend on the right
plt.savefig('separation_points_subplots.png', dpi=150, bbox_inches='tight')
plt.show()

  

#%% Improved case


"""
#------------------------------------------------------------------------------------------------------------------------------------#
                         Plotting the location of the first separation sensitivity in a subplot for all h/ls
#------------------------------------------------------------------------------------------------------------------------------------#
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# --- your helper kept as-is ---
def y_at_x_on_polyline(x, y, x_star):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    xdiff = x - x_star
    seg_idx = np.where((xdiff[:-1] * xdiff[1:]) <= 0)[0]
    if seg_idx.size:
        k = seg_idx
        x0, x1 = x[k], x[k+1]
        y0, y1 = y[k], y[k+1]
        vertical = (x1 == x0)
        out = np.empty(k.size, float)
        if np.any(~vertical):
            t = (x_star - x0[~vertical]) / (x1[~vertical] - x0[~vertical])
            out[~vertical] = y0[~vertical] + t * (y1[~vertical] - y0[~vertical])
        if np.any(vertical):
            out[vertical] = 0.5*(y0[vertical] + y1[vertical])
        seg_dist = np.minimum(np.abs(xdiff[k]), np.abs(xdiff[k+1]))
        return out[np.argmin(seg_dist)]
    j = int(np.argmin(np.abs(xdiff)))
    return y[j]

# ---- inputs ----
h_l_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

# =============================================================================
# NEW: Define consistent marker styles and offsets for each Mach number
# =============================================================================
# You'll need to adjust this based on your actual Mach numbers
mach_style_map = {
    1.5: {'marker': 'o', 'offset_idx': 0},   # circle
    2.0: {'marker': 's', 'offset_idx': 1},   # square
    2.5: {'marker': '^', 'offset_idx': 2},   # triangle up
    3.0: {'marker': 'D', 'offset_idx': 3},   # diamond
    3.5: {'marker': 'v', 'offset_idx': 4},   # triangle down
    4.0: {'marker': 'p', 'offset_idx': 5},   # pentagon
}

# Number of unique Mach values for offset calculation
n_mach = len(mach_style_map)

# Create subplot grid (2 rows x 4 columns for 8 h/l values)
fig, axes = plt.subplots(2, 4, figsize=(28, 14))
axes = axes.flatten()

# Create a shared colormap for Mach numbers across all subplots
cmap = cm.get_cmap("Spectral", n_mach)

for idx, h_l in enumerate(h_l_values):
    ax = axes[idx]
    
    # Get the key for this h/l value
    h_l_key = f"h_l_{h_l:.2f}"
    temp_keys = cases_by_hl[h_l_key]
    
    # =============================================================================
    # NEW: Calculate offset magnitude based on this subplot's y-range
    # This ensures offsets scale appropriately with the geometry amplitude
    # =============================================================================
    # Estimate y-range from the h/l value (amplitude scales with h/l)
    estimated_amplitude = h_l * 0.1  # wavelength is ~0.1m based on your plots
    offset_magnitude = estimated_amplitude * 0.08  # 8% of amplitude
    
    # Create systematic offsets centered around zero
    offsets = np.linspace(-offset_magnitude, offset_magnitude, n_mach)
    
    # Track which legend labels we've already used (reset for each subplot)
    seen_sep = set()
    seen_nosep = set()
    
    # =============================================================================
    # NEW: Track no-separation cases for text annotation
    # =============================================================================
    no_sep_mach_list = []
    
    for i, temp_key in enumerate(temp_keys):
        # Parse Mach values (for legend & dedupe)
        m = re.search(r"Mach_([0-9]*\.?[0-9]+)", temp_key)
        mach_string = m.group(1) if m else "?"
        mach_val = float(mach_string) if m else np.nan
        
        # =============================================================================
        # NEW: Get marker style and offset for this Mach number
        # =============================================================================
        if mach_val in mach_style_map:
            marker_style = mach_style_map[mach_val]['marker']
            y_offset = offsets[mach_style_map[mach_val]['offset_idx']]
            color_idx = mach_style_map[mach_val]['offset_idx']
        else:
            marker_style = 'o'  # default
            y_offset = 0
            color_idx = i % n_mach
        
        # Separation x-locations and window bounds
        xsep = np.asarray(x_sep[temp_key]).ravel()
        xmax = np.asarray(x_max[temp_key]).ravel()
        
        # Geometry curve
        min_l = 0.0
        max_l = 0.1
        
        x_temp = np.asarray(ds_by_case[temp_key]["X"].data)
        y_temp = np.asarray(ds_by_case[temp_key]["Y"].data)
        
        mask = (x_temp < max_l) & (x_temp > min_l)
        x_temp_in = x_temp[mask]
        y_temp_in = y_temp[mask]
        
        # Plot the geometry once per case
        ax.plot(x_temp_in, y_temp_in, color="blue", linewidth=4.0)
        
        # Window mask: between the first two maxima
        if xmax.size >= 2:
            lo, hi = np.sort(xmax[:2])
            mask_new = np.isfinite(xsep) & (xsep > lo) & (xsep < hi)
            x_sep_filtered = xsep[mask_new]
        else:
            x_sep_filtered = np.array([], dtype=float)
        
        if x_sep_filtered.size:
            # Take leftmost separation in the window
            firstSepPointX = float(np.min(x_sep_filtered))
            firstSepPointY = y_at_x_on_polyline(x_temp_in, y_temp_in, firstSepPointX)
            
            # Label each Mach ONCE for "sep" cases
            label = f"M = {mach_string}"
            if mach_val in seen_sep:
                label = "_nolegend_"
            else:
                seen_sep.add(mach_val)
            
            # =============================================================================
            # MODIFIED: Apply systematic y-offset and use unique marker per Mach
            # =============================================================================
            ax.scatter(firstSepPointX, 
                       firstSepPointY + y_offset,  # <-- systematic offset applied
                       label=label, 
                       zorder=5, 
                       color=cmap(color_idx), 
                       marker=marker_style,        # <-- unique marker per Mach
                       edgecolor="black", 
                       linewidths=1.5, 
                       s=250,                      # slightly smaller for cleaner look
                       alpha=0.9)
        else:
            # =============================================================================
            # MODIFIED: Better "no separation" indication
            # =============================================================================
            no_sep_mach_list.append(mach_string)
            
            # Option 1: Place X marker at window midpoint with offset
            if xmax.size >= 2:
                midx = 0.5 * (lo + hi)
                midy = y_at_x_on_polyline(x_temp, y_temp, midx)
                
                # Use the same color as the Mach number but with X marker
                ax.scatter([midx], [midy + y_offset], 
                           marker="x", 
                           color=cmap(color_idx),  # <-- color-coded X
                           edgecolor=cmap(color_idx),
                           zorder=4, 
                           s=400,
                           linewidths=3)
            
            # Add ONE legend entry per Mach for "(no sep)"
            label = f"M = {mach_string} (no sep)"
            if mach_val in seen_nosep:
                label = "_nolegend_"
            else:
                seen_nosep.add(mach_val)
            
            ax.scatter([], [], color=cmap(color_idx), marker='x', 
                       s=100, linewidths=2, label=label)
    
    # =============================================================================
    # NEW: Add text annotation showing which Mach numbers don't separate
    # =============================================================================
    if no_sep_mach_list:
        no_sep_text = "No sep: M = " + ", ".join(sorted(set(no_sep_mach_list)))
        ax.text(0.98, 0.02, no_sep_text, 
                transform=ax.transAxes,
                fontsize=20, 
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                          edgecolor='gray', alpha=0.8))
    
    # Subplot formatting
    ax.set_title(f"h/l = {h_l:.2f}", fontsize=48, fontweight='bold')
    ax.set_xlabel("X [m]", fontsize=34)
    ax.set_ylabel("Y [m]", fontsize=34)
    ax.tick_params(labelsize=34)
    ax.grid(True)

# =============================================================================
# MODIFIED: Build a cleaner shared legend with marker shapes
# =============================================================================
legend_handles = []
for mach_val in sorted(mach_style_map.keys()):
    style = mach_style_map[mach_val]
    handle = Line2D([0], [0], 
                    marker=style['marker'], 
                    color='w',
                    markerfacecolor=cmap(style['offset_idx']),
                    markeredgecolor='black',
                    markersize=20, 
                    linestyle='',
                    label=f'M = {mach_val}')
    legend_handles.append(handle)

# Add "no separation" marker to legend
no_sep_handle = Line2D([0], [0], marker='x', color='gray', linestyle='', 
                       markersize=18, markeredgewidth=3, label='No separation')
legend_handles.append(no_sep_handle)

fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1.10, 0.5), 
           fontsize=38, title="Mach Number", title_fontsize=48)

# Main title for the entire figure
fig.suptitle("First Separation Point vs h/l", fontsize=58, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(right=0.88)
plt.savefig('separation_points_subplots.png', dpi=150, bbox_inches='tight')
plt.show()


#%% Attempt 2---

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                         Figure A: Summary Line Plot - First Separation Location vs h/l
                         Figure B: Heatmap - Separation Occurrence Matrix
#------------------------------------------------------------------------------------------------------------------------------------#
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- your helper kept as-is ---
def y_at_x_on_polyline(x, y, x_star):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    xdiff = x - x_star
    seg_idx = np.where((xdiff[:-1] * xdiff[1:]) <= 0)[0]
    if seg_idx.size:
        k = seg_idx
        x0, x1 = x[k], x[k+1]
        y0, y1 = y[k], y[k+1]
        vertical = (x1 == x0)
        out = np.empty(k.size, float)
        if np.any(~vertical):
            t = (x_star - x0[~vertical]) / (x1[~vertical] - x0[~vertical])
            out[~vertical] = y0[~vertical] + t * (y1[~vertical] - y0[~vertical])
        if np.any(vertical):
            out[vertical] = 0.5*(y0[vertical] + y1[vertical])
        seg_dist = np.minimum(np.abs(xdiff[k]), np.abs(xdiff[k+1]))
        return out[np.argmin(seg_dist)]
    j = int(np.argmin(np.abs(xdiff)))
    return y[j]

# ---- inputs ----
h_l_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
mach_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Style definitions
mach_colors = {
    1.5: '#E31A1C',  # Red
    2.0: '#FF7F00',  # Orange
    2.5: '#33A02C',  # Green
    3.0: '#1F78B4',  # Blue
    3.5: '#6A3D9A',  # Purple
    4.0: '#000000',  # Black
}

mach_markers = {
    1.5: 'o',   # circle
    2.0: 's',   # square
    2.5: '^',   # triangle up
    3.0: 'D',   # diamond
    3.5: 'v',   # triangle down
    4.0: 'p',   # pentagon
}

# =============================================================================
# STEP 1: Extract data into organized structure
# =============================================================================
# Dictionary to store results: {mach_val: {'h_l': [...], 'x_sep': [...], 'separates': [...]}}
results_by_mach = {m: {'h_l': [], 'x_sep': [], 'separates': []} for m in mach_values}

for h_l in h_l_values:
    h_l_key = f"h_l_{h_l:.2f}"
    temp_keys = cases_by_hl[h_l_key]
    
    for temp_key in temp_keys:
        # Parse Mach value
        m = re.search(r"Mach_([0-9]*\.?[0-9]+)", temp_key)
        if not m:
            continue
        mach_val = float(m.group(1))
        
        if mach_val not in mach_values:
            continue
        
        # Get separation data
        xsep = np.asarray(x_sep[temp_key]).ravel()
        xmax = np.asarray(x_max[temp_key]).ravel()
        
        # Window mask: between the first two maxima
        if xmax.size >= 2:
            lo, hi = np.sort(xmax[:2])
            mask_new = np.isfinite(xsep) & (xsep > lo) & (xsep < hi)
            x_sep_filtered = xsep[mask_new]
        else:
            x_sep_filtered = np.array([], dtype=float)
        
        # Store results
        results_by_mach[mach_val]['h_l'].append(h_l)
        
        if x_sep_filtered.size:
            firstSepPointX = float(np.min(x_sep_filtered))
            results_by_mach[mach_val]['x_sep'].append(firstSepPointX)
            results_by_mach[mach_val]['separates'].append(True)
        else:
            results_by_mach[mach_val]['x_sep'].append(np.nan)
            results_by_mach[mach_val]['separates'].append(False)

# =============================================================================
# FIGURE A: Summary Line Plot
# =============================================================================
fig_a, ax_a = plt.subplots(figsize=(12, 8))

for mach_val in mach_values:
    data = results_by_mach[mach_val]
    
    # Get h/l and x_sep values where separation occurs
    h_l_sep = [h for h, sep in zip(data['h_l'], data['separates']) if sep]
    x_sep_vals = [x for x, sep in zip(data['x_sep'], data['separates']) if sep]
    
    if h_l_sep:  # Only plot if there's data
        # Sort by h/l for proper line connection
        sorted_pairs = sorted(zip(h_l_sep, x_sep_vals))
        h_l_sorted, x_sep_sorted = zip(*sorted_pairs)
        
        ax_a.plot(h_l_sorted, x_sep_sorted,
                  color=mach_colors[mach_val],
                  marker=mach_markers[mach_val],
                  markersize=14,
                  linewidth=2.5,
                  markeredgecolor='black',
                  markeredgewidth=1,
                  label=f'M = {mach_val}')

ax_a.set_xlabel('h/l', fontsize=26)
ax_a.set_ylabel('First Separation Location, x [m]', fontsize=26)
ax_a.set_title('First Separation Point Location vs h/l', fontsize=34, fontweight='bold')
ax_a.tick_params(labelsize=21)
ax_a.legend(title='Mach Number', fontsize=18, title_fontsize=21, loc='best')
ax_a.grid(True, alpha=0.3)
ax_a.set_xlim([min(h_l_values) - 0.005, max(h_l_values) + 0.005])

plt.tight_layout()
plt.savefig('figure_a_separation_vs_hl.png', dpi=200, bbox_inches='tight')
plt.show()
#%%

# =============================================================================
# FIGURE A: Summary Line Plot (with outlier filtering)
# =============================================================================
fig_a, ax_a = plt.subplots(figsize=(12, 8))

# Define outliers to exclude: list of (mach, h_l) tuples
outliers_to_exclude = [
    (3.5, 0.04),  # Known bad separation detection
    # Add more tuples here if needed: (mach_value, h_l_value),
]

for mach_val in mach_values:
    data = results_by_mach[mach_val]
    
    # Get h/l and x_sep values where separation occurs AND not an outlier
    h_l_sep = []
    x_sep_vals = []
    
    for h, x, sep in zip(data['h_l'], data['x_sep'], data['separates']):
        if sep:  # Only if separation occurs
            # Check if this point is in the outlier list
            if (mach_val, h) not in outliers_to_exclude:
                h_l_sep.append(h)
                x_sep_vals.append(x)
    
    if h_l_sep:  # Only plot if there's data
        # Sort by h/l for proper line connection
        sorted_pairs = sorted(zip(h_l_sep, x_sep_vals))
        h_l_sorted, x_sep_sorted = zip(*sorted_pairs)
        
        ax_a.plot(h_l_sorted, x_sep_sorted,
                  color=mach_colors[mach_val],
                  marker=mach_markers[mach_val],
                  markersize=14,
                  linewidth=2.5,
                  markeredgecolor='black',
                  markeredgewidth=1,
                  label=f'M = {mach_val}')

ax_a.set_xlabel('h/l', fontsize=26)
ax_a.set_ylabel('First Separation Location, x [m]', fontsize=26)
ax_a.set_title('First Separation Point Location vs h/l', fontsize=34, fontweight='bold')
ax_a.tick_params(labelsize=21)
ax_a.legend(title='Mach Number', fontsize=18, title_fontsize=21, loc='best')
ax_a.grid(True, alpha=0.3)
ax_a.set_xlim([min(h_l_values) - 0.005, max(h_l_values) + 0.005])

plt.tight_layout()
plt.savefig('figure_a_separation_vs_hl.png', dpi=200, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# FIGURE B: Heatmap - Separation Occurrence Matrix
# =============================================================================
# Create matrix: rows = h/l, columns = Mach
separation_matrix = np.zeros((len(h_l_values), len(mach_values)))

for i, h_l in enumerate(h_l_values):
    for j, mach_val in enumerate(mach_values):
        data = results_by_mach[mach_val]
        # Find the index for this h/l
        try:
            idx = data['h_l'].index(h_l)
            separation_matrix[i, j] = 1 if data['separates'][idx] else 0
        except ValueError:
            separation_matrix[i, j] = np.nan  # No data for this combination

fig_b, ax_b = plt.subplots(figsize=(10, 8))

# Custom colormap: Red for no separation, Green for separation
from matplotlib.colors import ListedColormap
cmap_binary = ListedColormap(['#2ECC71', '#E74C3C'])  # Red, Green

im = ax_b.imshow(separation_matrix, cmap=cmap_binary, aspect='auto', vmin=0, vmax=1)

# Set tick labels
ax_b.set_xticks(np.arange(len(mach_values)))
ax_b.set_yticks(np.arange(len(h_l_values)))
ax_b.set_xticklabels([f'{m}' for m in mach_values], fontsize=21)
ax_b.set_yticklabels([f'{h:.2f}' for h in h_l_values], fontsize=21)

# Set minor ticks at cell boundaries (offset by 0.5 from cell centers)
ax_b.set_xticks(np.arange(-0.5, len(mach_values), 1), minor=True)
ax_b.set_yticks(np.arange(-0.5, len(h_l_values), 1), minor=True)

# Draw grid lines at minor ticks
ax_b.grid(which='minor', color='black', linestyle='-', linewidth=2)

# Remove minor tick marks (keep only the lines)
ax_b.tick_params(which='minor', length=0)

# Add text annotations in each cell
for i in range(len(h_l_values)):
    for j in range(len(mach_values)):
        value = separation_matrix[i, j]
        text = "Sep" if value == 1 else "No Sep"
        text_color = 'white'
        ax_b.text(j, i, text, ha='center', va='center', 
                  fontsize=16, fontweight='bold', color=text_color)

ax_b.set_xlabel('Mach Number', fontsize=21)
ax_b.set_ylabel('h/l', fontsize=21)
ax_b.set_title('Flow Separation Occurrence Map', fontsize=32, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax_b, ticks=[0.25, 0.75])
cbar.ax.set_yticklabels(['No Separation', 'Separation'], fontsize=18)

plt.tight_layout()
plt.savefig('figure_b_separation_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()








#%% Analyzing this data # 
temp_keys = cases_by_hl["h_l_0.04"]

## Hypothesis 1: The pressure at the wall dictates where the shock stabilizes. It is hypothesized that the separation point occur at which the pressure on the left side of the shock overcomes the pressure on the right side of the shock. Pleft - Pright = 0....
# To prove hypothesis 1, a series of graphs will be generated to determine if this hypothesis is right. #

###===================  PRE-ALLOCATING VARIABLES  ===================###
P = {}
x = {}
y = {}
P_x = {}
dp_dx = {}



###===================  DEFINING THE FIGURE  ===================###
fig, ax1 = plt.subplots()
ax2 = ax1.twinx() # Geometry subplot
ax3 = ax1.twinx() # Separation point subplot



###===================  DEFINING THE COLOR SCHEME FOR THE PLOT  ===================###
cmap_lines = cm.get_cmap("plasma", len(temp_keys))
cmap_points = cm.get_cmap("plasma",len(temp_keys))



for i,temp_key in enumerate(temp_keys):
    ###===================  PROCESSING DATA  ===================###
    P[temp_key] = ds_by_case[temp_key]["P"].data
    x[temp_key] = ds_by_case[temp_key]["X"].data
    y[temp_key] = ds_by_case[temp_key]["Y"].data
    P_x[temp_key] = ds_by_case[temp_key]["P_x"]
    dp_dx[temp_key] = np.gradient(P[temp_key],x[temp_key])
    
    # separation x-locations and window bounds #
    xsep = np.asarray(x_sep[temp_key]).ravel()
    xmax = np.asarray(x_max[temp_key]).ravel()
    
    
    # window mask: between the first two maxima
    if xmax.size >= 2:
        lo, hi = np.sort(xmax[:2])
        mask_new = np.isfinite(xsep) & (xsep > lo) & (xsep < hi)
        x_sep_filtered = xsep[mask_new]
    else:
        x_sep_filtered = np.array([], dtype=float)
        
    # Getting the first point separation # 
    firstSepPointX = float(np.min(x_sep_filtered))
    firstSepPointY = y_at_x_on_polyline(x[temp_key], y[temp_key], firstSepPointX)
    firstSepPointP = y_at_x_on_polyline(x[temp_key], P[temp_key], firstSepPointX)

    ###===================  SCATTER PLOT OF THE SEPARATION POINTS ALONG THE PRESSURE LINES ===================###
    if i == 0:
        ax1.scatter(firstSepPointX, firstSepPointP, color = "red", label = "Separation Point")
        ax1.axvline(x = firstSepPointX, linestyle = '--', color = cmap_lines(i))
    else: 
        ax1.scatter(firstSepPointX, firstSepPointP, color = "red")
        ax1.axvline(x = firstSepPointX, linestyle = '--', color = cmap_lines(i))
        
        
    ###===================  PLOTTING PRESSURE VS X ===================###
    ax1.plot(x[temp_key],P[temp_key], color = cmap_lines(i), label = f"{temp_key}")
    
    ax1.set_title("Pressure Vs X")
    ax1.set_xlabel("X[inches]")
    ax1.set_ylabel("Pressure[Pa]")
    
    
    ###===================  PLOTTING THE WAVY GEOMETRY ===================###
    #ax2.plot(x[temp_keys[0]],y[temp_keys[0]], color = "black")  
    #ax2.set_xlabel("X[inches]")
    #ax2.set_ylabel("Y[inches]")
    
    

    #ax2.scatter(firstSepPointX, firstSepPointY, color = cmap_points(i), label = f"{temp_key}")
    

    ax1.legend(bbox_to_anchor=(1.25,0.5), loc = "upper left")
    #ax2.axvline(x = firstSepPointX, linestyle = "--", linewidth = 0.5, color = cmap_lines(i))
    #ax2.set_xlabel("X[inches]")
    #ax2.set_ylabel("Y[inches]")    

    
    ax1.grid(True, which = "both")
      
    
    
plt.tight_layout()
plt.show()


#%% SUBPLOT GRAPHS 


###===================  GETTING ALL h_l KEYS  ===================###
h_l_keys = list(cases_by_hl.keys())
n_plots = len(h_l_keys)

###===================  CREATING LARGER SUBPLOTS  ===================###
n_cols = 2  
n_rows = int(np.ceil(n_plots / n_cols))

# INCREASED SIZE HERE
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8*n_rows), dpi=100)
axes = axes.flatten()

###===================  OUTER LOOP: ITERATE THROUGH EACH h_l  ===================###
for plot_idx, h_l_key in enumerate(h_l_keys):
    
    temp_keys = cases_by_hl[h_l_key]
    
    P = {}
    x = {}
    y = {}
    P_x = {}
    dp_dx = {}
    
    ax1 = axes[plot_idx]
    
    cmap_lines = cm.get_cmap("plasma", len(temp_keys))
    cmap_points = cm.get_cmap("plasma", len(temp_keys))
    
    for i, temp_key in enumerate(temp_keys):
        P[temp_key] = ds_by_case[temp_key]["P"].data
        x[temp_key] = ds_by_case[temp_key]["X"].data
        y[temp_key] = ds_by_case[temp_key]["Y"].data
        P_x[temp_key] = ds_by_case[temp_key]["P_x"]
        dp_dx[temp_key] = np.gradient(P[temp_key], x[temp_key])
        
        xsep = np.asarray(x_sep[temp_key]).ravel()
        xmax = np.asarray(x_max[temp_key]).ravel()
        
        if xmax.size >= 2:
            lo, hi = np.sort(xmax[:2])
            mask_new = np.isfinite(xsep) & (xsep > lo) & (xsep < hi)
            x_sep_filtered = xsep[mask_new]
        else:
            x_sep_filtered = np.array([], dtype=float)
        
        if len(x_sep_filtered) > 0:
            firstSepPointX = float(np.min(x_sep_filtered))
            firstSepPointY = y_at_x_on_polyline(x[temp_key], y[temp_key], firstSepPointX)
            firstSepPointP = y_at_x_on_polyline(x[temp_key], P[temp_key], firstSepPointX)
            
            if i == 0:
                ax1.scatter(firstSepPointX, firstSepPointP, color="red", 
                           label="Separation Point", zorder=5, s=100)  # Larger points
            else:
                ax1.scatter(firstSepPointX, firstSepPointP, color="red", zorder=5, s=100)
            
            ax1.axvline(x=firstSepPointX, linestyle='--', color=cmap_lines(i), 
                       alpha=0.5, linewidth=1.5)
        
        ax1.plot(x[temp_key], P_x[temp_key], color=cmap_lines(i), 
                label=f"{temp_key}", linewidth=2)  # Thicker lines
    
    # LARGER FONT SIZES
    ax1.set_title(f"Pressure Vs X - {h_l_key}", fontsize=16, fontweight='bold')
    ax1.set_xlabel("X [inches]", fontsize=14)
    ax1.set_ylabel("Pressure [Pa]", fontsize=14)
    ax1.tick_params(labelsize=12)  # Larger tick labels
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=10)

for idx in range(n_plots, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()


#%% Every graph is separate no subplot 
###===================  GETTING ALL h_l KEYS  ===================###
h_l_keys = list(cases_by_hl.keys())

###===================  LOOP THROUGH EACH h_l AND CREATE SEPARATE FIGURE  ===================###
for h_l_key in h_l_keys:
    
    temp_keys = cases_by_hl[h_l_key]
    
    ###===================  PRE-ALLOCATING VARIABLES  ===================###
    P = {}
    x = {}
    y = {}
    P_x = {}
    dp_dx = {}
    
    ###===================  CREATE NEW FIGURE FOR THIS h_l  ===================###
    fig, ax1 = plt.subplots(figsize=(16, 10))
    
    ###===================  DEFINING THE COLOR SCHEME FOR THE PLOT  ===================###
    cmap_lines = cm.get_cmap("plasma", len(temp_keys))
    cmap_points = cm.get_cmap("plasma", len(temp_keys))
    
    ###===================  INNER LOOP: ITERATE THROUGH CASES FOR THIS h_l  ===================###
    for i, temp_key in enumerate(temp_keys):
        ###===================  PROCESSING DATA  ===================###
        P[temp_key] = ds_by_case[temp_key]["P"].data
        x[temp_key] = ds_by_case[temp_key]["X"].data
        y[temp_key] = ds_by_case[temp_key]["Y"].data
        P_x[temp_key] = ds_by_case[temp_key]["P_x"]
        dp_dx[temp_key] = np.gradient(P[temp_key], x[temp_key])
        
        # separation x-locations and window bounds #
        xsep = np.asarray(x_sep[temp_key]).ravel()
        xmax = np.asarray(x_max[temp_key]).ravel()
        
        # window mask: between the first two maxima
        if xmax.size >= 2:
            lo, hi = np.sort(xmax[:2])
            mask_new = np.isfinite(xsep) & (xsep > lo) & (xsep < hi)
            x_sep_filtered = xsep[mask_new]
        else:
            x_sep_filtered = np.array([], dtype=float)
        
        # Getting the first point separation # 
        if len(x_sep_filtered) > 0:
            firstSepPointX = float(np.min(x_sep_filtered))
            firstSepPointY = y_at_x_on_polyline(x[temp_key], y[temp_key], firstSepPointX)
            firstSepPointP = y_at_x_on_polyline(x[temp_key], P[temp_key], firstSepPointX)
            
            ###===================  SCATTER PLOT OF THE SEPARATION POINTS  ===================###
            if i == 0:
                ax1.scatter(firstSepPointX, firstSepPointP, color="red", 
                           label="Separation Point", zorder=5, s=100)
            else:
                ax1.scatter(firstSepPointX, firstSepPointP, color="red", zorder=5, s=100)
            
            #ax1.axvline(x=firstSepPointX, linestyle='--', color=cmap_lines(i), 
             #          alpha=0.5, linewidth=1.5)
        
        ###===================  PLOTTING PRESSURE VS X  ===================###
        ax1.plot(x[temp_key], P[temp_key], color=cmap_lines(i), linewidth=2, label = f"{temp_key}")
    
    ###===================  FORMATTING THIS FIGURE  ===================###
    ax1.set_title(f"Pressure Vs X - {h_l_key}", fontsize=16, fontweight='bold')
    ax1.set_xlabel("X [inches]", fontsize=14)
    ax1.set_ylabel("Pressure [Pa]", fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=10)
    
    plt.tight_layout()
    plt.show()


#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Comparing h_l_x pressure profiles and sine wave pressure profile 
#------------------------------------------------------------------------------------------------------------------------------------#
"""

h_l_x_cases = cases_by_hl["h_l_x"]

# Extracting data # 

h_l_4_P = ds_by_case["h_l_0.04_Mach_1.5"]["P"].data
h_l_x_P = ds_by_case["h_l_x_Mach_1.5"]["P"].data

h_l_x_xNonDim = (ds_by_case["h_l_x_Mach 1.5"]["X"] - ds_by_case["h_l_x_Mach 1.5"]["X"][0])  / (ds_by_case["h_l_x_Mach 1.5"]["X"][-1] - ds_by_case["h_l_x_Mach 1.5"]["X"][0])
h_l_4_xNonDim = ds_by_case["h_l_0.040_Mach_1.5"]["X"] / np.max(ds_by_case["h_l_0.040_Mach_1.5"]["X"])

h_l_x_y = ds_by_case["h_l_x_Mach 1.5"]["Y"] 
h_l_4_y = ds_by_case["h_l_0.040_Mach_1.5"]["Y"] 


# Plotting #
plt.plot(h_l_x_xNonDim, h_l_x_P, label = "h/l = x")
plt.plot(h_l_4_xNonDim, h_l_4_P, label = "h/l = 0.040")
plt.title(r"$Pressure$ Vs $X/X_{max}$",fontsize = 18)
plt.xlabel(r"$X/X_{max}$",fontsize = 14)
plt.ylabel(r"Pressure [Pa]", fontsize = 14)
plt.grid()
plt.legend()
plt.show()


"""
Simple Power Extraction from 2D Pressure Profiles
==================================================

Given: x, y, p arrays from CFD wall data
Compute: Tangential force, Torque, Power

Physics:
    F_theta = -∫ p * (dy/dx) dx
    tau = F_theta * R
    P = tau * omega
"""

import numpy as np
import pandas as pd


def compute_power_2D(x, y, p, R, omega):
    """
    Power per unit span from 2D pressure profile.
    
    Returns: F_theta [N/m], tau [N], P [W/m]
    """
    dydx = np.gradient(y, x)
    
    if hasattr(np, 'trapezoid'):
        F_theta = -np.trapezoid(p * dydx, x)
    else:
        F_theta = -np.trapz(p * dydx, x)
    
    tau = F_theta * R      # [N·m/m] = [N]
    P = tau * omega        # [W/m]
    
    return {
     'F_theta': F_theta,      # [N/m] per unit span
     'tau': tau,              # [N·m] torque
     'P': P,                  # [W] power
     'P_kW': P / 1000         # [kW] power
     }



def compute_force_2D(x, y, p, R):
    """
    Tangential force and torque from 2D pressure profile.
    
    Returns: F_theta [N/m], tau [N·m/m]
    """
    dydx = np.gradient(y, x)
    
    if hasattr(np, 'trapezoid'):
        F_theta = -np.trapezoid(p * dydx, x)
    else:
        F_theta = -np.trapz(p * dydx, x)
    
    tau = F_theta * R
    
    return {
        'F_theta': F_theta,          # [N] total tangential force
        'tau': tau, # [N/m] per unit span 
        }



def compute_torque_2D_norm(x, y, p, R):
    """
    Tangential force and torque per unit span from 2D pressure profile.
    
    Parameters
    ----------
    x, y : arrays - wall coordinates [m]
    p : array - wall pressure [Pa]
    R : float - radius from rotation axis [m]
    
    Returns
    -------
    dict with:
        F_theta : tangential force per unit span [N/m]
        tau : torque per unit span per unit length [N·m/m²]
    """
    dydx = np.gradient(y, x)
    
    if hasattr(np, 'trapezoid'):
        F_theta = -np.trapezoid(p * dydx, x)
    else:
        F_theta = -np.trapz(p * dydx, x)
    
    tau = F_theta * R
    
    # Normalize by projected length
    L = x[-1] - x[0]
    
    return {
        'F_theta': F_theta / L,   # [N/m²] per unit span per unit length
        'tau': tau / L            # [N·m/m²] per unit span per unit length
    }



def load_csv_data(filepath, x_col='x', y_col='y', p_col='p'):
    """Load pressure profile from CSV file."""
    df = pd.read_csv(filepath)
    return df[x_col].values, df[y_col].values, df[p_col].values





def load_tecplot_data(filepath):
    """
    Load pressure profile from Tecplot ASCII file.
    Assumes columns are: X, Y, Pressure (or similar).
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                values = [float(v) for v in line.split()]
                if len(values) >= 3:
                    data.append(values[:3])
            except ValueError:
                continue
    
    data = np.array(data)
    # Sort by x
    idx = np.argsort(data[:, 0])
    return data[idx, 0], data[idx, 1], data[idx, 2]




# Load your CFD data

# Define rotor parameters
x = ds_by_case["h_l_0.04_Mach_1.5"]["X"]
y = h_l_4_y
p = h_l_4_P
u_inf = np.mean(ds_by_case_inlet["h_l_0.040_Mach_1.5"]["U"].data)

x2 = ds_by_case["h_l_x_Mach_1.5"]["X"]
y2 = h_l_x_y
p2 = h_l_x_P
u_inf2 = np.mean(ds_by_case_inlet["h_l_x_Mach_1.5"]["U"].data)

R = 0.15  # radius [m]
lambda_opt = 0.4                             
omega =  (lambda_opt * u_inf) / R       # RPM to rad/s


# Computing results # 
results = compute_torque_2D_norm(x, y, p, R)
results2 = compute_torque_2D_norm(x2,y2,p2,R)


print(f"Torque: {results['tau']:.4f} N·m")
print(f"Torque: {results2['tau']:.4f} N·m")



#%%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_torque_table_mach(df, output_path='torque_table_mach.png', title=None):
    """
    Generate a formatted pivot table image from the torque comparison DataFrame.
    
    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns: 'Mach', 'h/l', 'tau_h_l_x [N·m/m²]', 'tau_h_l [N·m/m²]'
    output_path : str
        Path to save the output image
    title : str, optional
        Custom title for the table
    
    Returns
    -------
    None (saves image to output_path)
    
    How it works:
    -------------
    1. Extract unique Mach numbers and h/l values from the DataFrame
    2. Get tau_h_l_x values (one per Mach - these go in the first data row)
    3. Pivot tau_h_l values so Mach becomes columns, h/l becomes rows
    4. Build the table structure and apply formatting
    """
    
    # Step 1: Extract unique values (sorted for consistent ordering)
    mach_numbers = sorted(df['Mach'].unique())
    h_l_values = sorted(df['h/l'].unique())
    
    # Step 2: Get tau_h_l_x values for each Mach number
    # These are constant for each Mach, so we take the first occurrence
    tau_x_by_mach = df.groupby('Mach')['tau_h_l_x [N·m/m²]'].first().to_dict()
    
    # Step 3: Pivot the tau_h_l values
    # This transforms: rows of (Mach, h/l, tau) -> matrix[h/l][Mach] = tau
    pivot = df.pivot(index='h/l', columns='Mach', values='tau_h_l [N·m/m²]')
    
    # Step 4: Build table structure
    # Column headers: empty cell + Mach labels
    col_labels = [""] + [f"M = {m}" for m in mach_numbers]
    
    # Build data rows
    table_data = []
    
    # First row: tau_h_l_x values (the "x" case filtered to 0-0.09m)
    first_row = ["h/l = Optimal"] + [f"{tau_x_by_mach[m]:.1f}" for m in mach_numbers]
    table_data.append(first_row)
    
    # Subsequent rows: tau_h_l for each h/l value
    for h_l in h_l_values:
        row = [f"h/l = {h_l:.2f}"] + [f"{pivot.loc[h_l, m]:.1f}" for m in mach_numbers]
        table_data.append(row)
    
    # Step 5: Create the figure and table
    # Figure size scales with number of columns
    fig_width = max(14, len(mach_numbers) * 1.5)
    fig_height = max(6, len(h_l_values) * 0.6 + 2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')  # Hide axes - we only want the table
    
    # Set title
    if title is None:
        title = "Torque per Unit Span per Unit Length [N·m/m²]\n(h/l = x filtered to 0-0.09m, varying Mach number)"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
    
    # Create table object
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # Step 6: Apply styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)  # (width_scale, height_scale)
    
    # Style header row (row index 0 in the table object)
    # Blue background (#2E75B6), white bold text
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor('#2E75B6')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Style first data row (the tau_x row) - light green (#C6EFCE)
    # Row index 1 in table object (0 is header)
    for j in range(len(col_labels)):
        table[(1, j)].set_facecolor('#C6EFCE')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.3)
    plt.close()
    
    print(f"Table saved to: {output_path}")
# =============================================================================
# Torque calculation function
# =============================================================================

def compute_torque_2D_norm(x, y, p, R):
    """
    Tangential force and torque per unit span per unit length from 2D pressure profile.
    
    Parameters
    ----------
    x, y : arrays - wall coordinates [m]
    p : array - wall pressure [Pa]
    R : float - radius from rotation axis [m]
    
    Returns
    -------
    dict with:
        F_theta : tangential force per unit area [N/m²]
        tau : torque per unit span per unit length [N·m/m²]
    """
    dydx = np.gradient(y, x)
    
    if hasattr(np, 'trapezoid'):
        F_theta = np.trapezoid(p * dydx, x)
    else:
        F_theta = np.trapz(p * dydx, x)
    
    tau = F_theta * R
    
    # Normalize by projected length
    L = x[-1] - x[0]
    
    return {
        'F_theta_norm': F_theta / L,
        'tau': tau / L,
        'F_theta': F_theta
    }


# =============================================================================
# Setup
# =============================================================================

R = 0.15  # Set your radius here [m]

mach_numbers = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
h_l_values = ['0.02', '0.03', '0.04', '0.05', '0.06', 
              '0.07', '0.08', '0.09']


# =============================================================================
# Build comparison table
# =============================================================================

results = []
resultsNorm = []


for mach in mach_numbers:
    # h_l_x key (note: space instead of underscore)
    h_l_x_key = f"h_l_x_Mach_{mach}"
    print(h_l_x_key)
    
    # Extract h_l_x data
    x_hlx = ds_by_case[h_l_x_key]["X"].data
    y_hlx = ds_by_case[h_l_x_key]["Y"].data
    p_hlx = ds_by_case[h_l_x_key]["P"].data
    
    # Filter to x from 0 to 0.09
    mask = (x_hlx >= 0) & (x_hlx <= 0.1)
    x_hlx = x_hlx[mask]
    y_hlx = y_hlx[mask]
    p_hlx = p_hlx[mask]
    
    # Compute torque for h_l_x
    hlx_result = compute_torque_2D_norm(x_hlx, y_hlx, p_hlx, R)
    
    # Compare to each h_l
    for h_l in h_l_values:
        h_l_key = f"h_l_{h_l}_Mach_{mach}"
        
        # Extract h_l data
        x_hl = ds_by_case[h_l_key]["X"].data
        y_hl = ds_by_case[h_l_key]["Y"].data
        p_hl = ds_by_case[h_l_key]["P"].data
        
        mask = (x_hl >= 0) & (x_hl <= 0.1)
        x_hl = x_hl[mask]
        y_hl = y_hl[mask]
        p_hl = p_hl[mask]
        
        # Compute torque for h_l
        hl_result = compute_torque_2D_norm(x_hl, y_hl, p_hl, R)
        
        # Store comparison
        results.append({
            'Mach': mach,
            'h/l': float(h_l),
            'tau_h_l_x [N·m/m²]': hlx_result['F_theta'],
            'tau_h_l [N·m/m²]': hl_result['F_theta'],
            'Difference [%]': (hlx_result['F_theta'] - hl_result['F_theta']) / abs(hl_result['F_theta']) * 100 if hl_result['F_theta'] != 0 else np.nan
        })
        
        resultsNorm.append({
            'Mach': mach,
            'h/l': float(h_l),
            'tau_h_l_x [N·m/m²]': hlx_result['F_theta_norm'],
            'tau_h_l [N·m/m²]': hl_result['F_theta_norm'],
            'Difference [%]': (hlx_result['F_theta_norm'] - hl_result['F_theta_norm']) / abs(hl_result['F_theta_norm']) * 100 if hl_result['F_theta_norm'] != 0 else np.nan
        })

# Create DataFrame
df_comparison = pd.DataFrame(results)
print(df_comparison.to_string(index=False))

df_comparisonNorm = pd.DataFrame(resultsNorm)

# =============================================================================
# Generate the formatted table image
# =============================================================================
generate_torque_table_mach(
    df_comparison, 
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Graphs\Mach Study\axialForce_comparison_table.png',
    title="Axial Force [N]\n(h/l = Optimal versus 0.02-0.09, varying Mach Number)"
)


generate_torque_table_mach(
    df_comparisonNorm, 
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Graphs\Mach Study\axialForce_comparison_Norm_table.png',
    title="Axial Force Per Unit Length [N/m]\n(h/l = Optimal versus 0.02-0.09, varying Mach Number)"
)




# =============================================================================
# Pressure profile comparison plots
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, mach in enumerate(mach_numbers):
    ax = axes[i]
    
    # h_l_x data
    h_l_x_key = f"h_l_x_Mach_{mach}"
    x_hlx = ds_by_case[h_l_x_key]["X"].data
    p_hlx = ds_by_case[h_l_x_key]["P"].data
    y_hlx = ds_by_case[h_l_x_key]["Y"].data
    
    # Filter to x from 0 to 0.09
    mask = (x_hlx >= 0) & (x_hlx <= 0.09)
    x_hlx = x_hlx[mask]
    p_hlx = p_hlx[mask]
    y_hlx = y_hlx[mask]
    
    x_hlx_norm = (x_hlx - x_hlx[0]) / (x_hlx[-1] - x_hlx[0])
    
    ax.plot(x_hlx_norm, p_hlx, 'k-', linewidth=2, label='h/l = x')
    
    # Plot all h_l cases
    colors = plt.cm.viridis(np.linspace(0, 1, len(h_l_values)))
    
    for j, h_l in enumerate(h_l_values):
        h_l_key = f"h_l_{h_l}_Mach_{mach}"
        
        x_hl = ds_by_case[h_l_key]["X"].data
        p_hl = ds_by_case[h_l_key]["P"].data
        x_hl_norm = (x_hl - x_hl[0]) / (x_hl[-1] - x_hl[0])
        
        ax.plot(x_hl_norm, p_hl, color=colors[j], alpha=0.7, label=f'h/l = {h_l}')
    
    ax.set_title(f'Mach {mach}', fontsize=21)
    ax.set_xlabel(r'$X/X_{max}$', fontsize = 18)
    ax.set_ylabel('Pressure [Pa]', fontsize = 18)
    ax.grid(True, alpha=0.3)
    
    # Only show legend on first plot
    if i == 0:
        ax.legend(fontsize=9, loc='best')

plt.tight_layout()
plt.savefig('pressure_comparison_all_mach.png', dpi=300, bbox_inches='tight')
plt.show()

#%% Creating a code that plots the axial force Versus varying h/l values ##


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =============================================================================
# Plot generation function (Mach number version)
# =============================================================================
def generate_axial_force_plot_mach(df, output_path='axial_force_plot_mach.png', title=None, 
                                    ylabel="Axial Force [N]", show_optimal=True):
    """
    Generate a plot of axial force vs h/l for different Mach numbers.
    
    Parameters
    ----------
    df : pandas DataFrame
        Must contain columns: 'Mach', 'h/l', 'tau_h_l_x [N·m/m²]', 'tau_h_l [N·m/m²]'
    output_path : str
        Path to save the output image
    title : str, optional
        Custom title for the plot
    ylabel : str
        Label for y-axis
    show_optimal : bool
        Whether to show the optimal h/l reference lines
    
    Returns
    -------
    None (saves image to output_path)
    """
    
    # Extract unique values
    mach_numbers = sorted(df['Mach'].unique())
    h_l_values = sorted(df['h/l'].unique())
    
    # Get tau_h_l_x (optimal) values for each Mach number
    tau_x_by_mach = df.groupby('Mach')['tau_h_l_x [N·m/m²]'].first().to_dict()
    
    # Pivot for easy plotting
    pivot = df.pivot(index='h/l', columns='Mach', values='tau_h_l [N·m/m²]')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colormap for Mach lines
    cmap = cm.get_cmap('viridis', len(mach_numbers))
    
    # Plot each Mach number as a separate line
    for i, mach in enumerate(mach_numbers):
        color = cmap(i)
        
        # Plot h/l values
        y_vals = [pivot.loc[h_l, mach] for h_l in h_l_values]
        ax.plot(h_l_values, y_vals, 'o-', color=color, linewidth=2, 
                markersize=8, label=f'M = {mach}')
        
        # Plot optimal value as a horizontal dashed line
        if show_optimal:
            ax.axhline(y=tau_x_by_mach[mach], color=color, linestyle='--', 
                       alpha=0.5, linewidth=1.5)
            
            # Add marker at far right for optimal
            ax.scatter(h_l_values[-1] + 0.005, tau_x_by_mach[mach], 
                       marker='*', s=150, color=color, edgecolor='k', 
                       linewidths=0.5, zorder=5)
    
    # Add annotation for optimal lines
    if show_optimal:
        ax.annotate('★ = Optimal h/l', xy=(0.98, 0.02), xycoords='axes fraction',
                    fontsize=11, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Formatting
    if title is None:
        title = "Axial Force vs h/l\n(Varying Mach Number)"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("h/l", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Mach Number", title_fontsize=12, fontsize=11,
              loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def generate_axial_force_plot_dual_mach(df, output_path='axial_force_dual_plot_mach.png', 
                                         title=None, ylabel="Axial Force [N/m]"):
    """
    Generate a dual-panel plot:
    - Left: Axial force vs h/l (lines for each Mach number)
    - Right: Axial force vs Mach number (lines for each h/l)
    
    This helps visualize trends in both directions.
    """
    
    # Extract unique values
    mach_numbers = sorted(df['Mach'].unique())
    h_l_values = sorted(df['h/l'].unique())
    
    # Get optimal values
    tau_x_by_mach = df.groupby('Mach')['tau_h_l_x [N·m/m²]'].first().to_dict()
    
    # Pivot tables
    pivot_hl = df.pivot(index='h/l', columns='Mach', values='tau_h_l [N·m/m²]')
    pivot_mach = df.pivot(index='Mach', columns='h/l', values='tau_h_l [N·m/m²]')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Colormaps
    cmap_mach = cm.get_cmap('viridis', len(mach_numbers))
    cmap_hl = cm.get_cmap('plasma', len(h_l_values))
    
    # Pstatic list # 
    pstatic_list = np.array([2.724e+05 , 1.278e+05, 58528, 27224, 13111, 6586.1])
    
    # Computing the shock relations subsquent to the shock # 
    
    
    
    
    
    
    # =========================================================================
    # Left plot: Force vs h/l (one line per Mach number)
    # =========================================================================
    for i, mach in enumerate(mach_numbers):
        color = cmap_mach(i)
        print(pstatic_list[i])
        print(mach)
        y_vals = [pivot_hl.loc[h_l, mach] for h_l in h_l_values]
        ax1.plot(h_l_values, y_vals / P_values[i], 'o-', color=color, linewidth=2, 
                 markersize=8, label=f'M = {mach}') # Divide by y_values to be able to non-dim the solution CHANGE HERE 
        
        # Optimal reference line
        #ax1.plot(0.05, tau_x_by_mach[mach], color=color, marker = 's', markersize = 10)
        #ax1.axhline(y=tau_x_by_mach[mach], color=color, linestyle='--', 
                    #alpha=0.4, linewidth=1.5) # DO THE SAME THING HERE
    
    ax1.set_title("Axial Force / P_after vs h/l", fontsize=28, fontweight='bold')
    ax1.set_xlabel("h/l", fontsize = 21)
    ax1.set_ylabel(ylabel, fontsize=18)
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3)
    
    ax1.axvline(x = 0.05, color='black', linestyle = '--', linewidth = 2, label = 'Optimal')
    ax1.legend(title="Mach", title_fontsize=16, fontsize=14, loc='best', ncol=2)
    
    
    
    
    
    
    # =========================================================================
    # Right plot: Force vs Mach number (one line per h/l)
    # =========================================================================
    # First plot optimal
    optimal_vals = [tau_x_by_mach[mach] for mach in mach_numbers]
    ax2.plot(mach_numbers, optimal_vals, 'k-', linewidth=3, markersize=10, 
             marker='*', label='Optimal', zorder=10)
    
    # Then plot each h/l
    for i, h_l in enumerate(h_l_values):
        color = cmap_hl(i)
        y_vals = [pivot_mach.loc[mach, h_l] for mach in mach_numbers]
        
        
        # Normalize the y_value! #
        ax2.plot(mach_numbers, y_vals , 'o-', color=color, linewidth=2, 
                 markersize=6, label=f'h/l = {h_l:.2f}')
    
    ax2.set_title("Axial Force vs Mach Number", fontsize=28, fontweight='bold')
    ax2.set_xlabel("Mach Number", fontsize=21)
    ax2.set_ylabel("Axial Force [N/m]", fontsize=18)
    ax2.tick_params(labelsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="h/l", title_fontsize=16, fontsize=14, loc='best', ncol=2)
    
    # Main title
    if title is None:
        title = "Axial Force Trends: h/l and Mach Number Effects"
    fig.suptitle(title, fontsize=34, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Dual plot saved to: {output_path}")





# =============================================================================
# Usage
# =============================================================================

# Single plot: Force vs h/l
generate_axial_force_plot_mach(
    df_comparison, 
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Graphs\Mach Study\axialForce_vs_hl_plot.png',
    title="Axial Force vs h/l\n(Varying Mach Number)",
    ylabel="Axial Force [N/m]",
    show_optimal=True
)

# Normalized version
generate_axial_force_plot_mach(
    df_comparisonNorm, 
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Graphs\Mach Study\axialForceNorm_vs_hl_plot.png',
    title="Axial Force per Unit Length vs h/l\n(Varying Mach Number)",
    ylabel="Axial Force per Unit Length [N/m]",
    show_optimal=True
)

# Dual panel plot showing both perspectives
generate_axial_force_plot_dual_mach(
    df_comparison, 
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Graphs\Mach Study\axialForce_dual_plot.png',
    title="Axial Force Trends",
    ylabel="Axial Force / Pstatic [m]"
)



#%% Testing capturing shocks ###
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d


temp_key = "h_l_0.03_Mach_2.5"



# Sorting the variables # 
P = ds_by_case_quad[temp_key]["P_x"].data
P_indx = np.argsort(P)


# Set the variables # 
x_raw = ds_by_case_quad[temp_key]["X"].data[P_indx]
y_raw = ds_by_case_quad[temp_key]["Y"].data[P_indx]
mask = (0 <= x_raw) & (0.1 >= x_raw)




x1 = x_raw[mask]
y1 = y_raw[mask]
Px = np.sort(ds_by_case_quad[temp_key]["P_x"].data[mask])

# Imposing the second mask # 
mask2 = Px >= 0 

x = x1[mask2]
y = y1[mask2]
Px_raw = Px[mask2]


sigma = 20
Px_filtered = gaussian_filter1d(Px[mask2],sigma)

dp_dx_3 = np.gradient(np.gradient(Px_filtered,x))


# Finding the inflection point # 
inflection_indices = np.where(np.diff(np.sign(dp_dx_3)))[0]


plt.scatter(x,dp_dx_3)
plt.xlabel("X [m]", fontsize = 18)
plt.ylabel("d3P/dx3", fontsize = 18)
plt.title("d3P/dx3 Vs X", fontsize = 21)
plt.grid()
plt.show()


# Scatter plotting # 
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()


# First Axis # 
ax1.scatter(x,Px_filtered)
ax1.set_xlabel("X [m]", fontsize = 18)
ax1.set_ylabel("dP/dx", fontsize = 18)
ax1.set_title("dP/dx Vs X", fontsize = 21)

# Second Axis # 
ax2.set_xlabel("X[m]",fontsize = 18)
ax2.set_ylabel("Y[m]",fontsize = 18)



plt.grid()
plt.show()

# Scatter plot x vs y #
plt.scatter(x1,y1)
plt.xlabel("X [m]", fontsize = 18)
plt.ylabel("Y [m]", fontsize = 18)
plt.title("X Vs Y", fontsize = 21)
plt.grid()
plt.show()

#%%

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt

temp_key = "h_l_0.03_Mach_2.5"

# =============================================================================
# ENVELOPE EXTRACTION FUNCTION
# =============================================================================
def extract_envelope(x, y, n_bins=200, envelope='both'):
    """
    Extract the upper and/or lower envelope of scattered data.
    """
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    bin_indices = np.digitize(x, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    y_upper = np.full(n_bins, np.nan)
    y_lower = np.full(n_bins, np.nan)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            y_upper[i] = np.max(y[mask])
            y_lower[i] = np.min(y[mask])
    
    valid = ~np.isnan(y_upper)
    x_env = bin_centers[valid]
    y_upper = y_upper[valid]
    y_lower = y_lower[valid]
    
    if envelope == 'upper':
        return x_env, y_upper
    elif envelope == 'lower':
        return x_env, y_lower
    else:
        return x_env, y_upper, y_lower

# =============================================================================
# FUNCTION: Find inflection points after each maximum
# =============================================================================
def find_inflection_after_peaks(x, y, peaks_idx, inflection_idx):
    """
    For each peak, find the first inflection point that occurs after it.
    
    Parameters:
    -----------
    x : array - x coordinates
    y : array - y values
    peaks_idx : array - indices of peaks (maxima)
    inflection_idx : array - indices of all inflection points
    
    Returns:
    --------
    results : list of dicts with peak and corresponding inflection point info
    """
    results = []
    
    for peak_i in peaks_idx:
        peak_x = x[peak_i]
        peak_y = y[peak_i]
        
        # Find inflection points that come AFTER this peak
        inflections_after = inflection_idx[inflection_idx > peak_i]
        
        if len(inflections_after) > 0:
            # Get the first inflection point after the peak
            first_inflection_i = inflections_after[0]
            inflection_x = x[first_inflection_i]
            inflection_y = y[first_inflection_i]
            
            results.append({
                'peak_idx': peak_i,
                'peak_x': peak_x,
                'peak_y': peak_y,
                'inflection_idx': first_inflection_i,
                'inflection_x': inflection_x,
                'inflection_y': inflection_y,
                'delta_x': inflection_x - peak_x  # Distance from peak to inflection
            })
        else:
            # No inflection point found after this peak
            results.append({
                'peak_idx': peak_i,
                'peak_x': peak_x,
                'peak_y': peak_y,
                'inflection_idx': None,
                'inflection_x': None,
                'inflection_y': None,
                'delta_x': None
            })
    
    return results

# =============================================================================
# DATA EXTRACTION
# =============================================================================

# Pre-Allocating Variables # 
P_values = []

for case_key in cases_by_hl.keys():
    for case_by_hl in cases_by_hl[case_key]:
        x_raw = ds_by_case_quad[case_by_hl]["X"].data
        y_raw = ds_by_case_quad[case_by_hl]["Y"].data
        Px_raw = ds_by_case_quad[case_by_hl]["P_x"].data
        P_raw = ds_by_case_quad[case_by_hl]["P"].data
        
        # First mask: spatial domain
        mask1 = (x_raw >= 0) & (x_raw <= 0.1)
        x1 = x_raw[mask1]
        y1 = y_raw[mask1]
        Px1 = Px_raw[mask1]
        P1 = P_raw[mask1]
        # Second mask: positive pressure gradients only
        mask2 = Px1 >= 0
        x = x1[mask2]
        y = y1[mask2]
        Px = Px1[mask2]
        P = P1[mask2]
        
        
        # =============================================================================
        # EXTRACT ENVELOPE
        # =============================================================================
        n_bins = 500
        x_env, Px_upper, Px_lower = extract_envelope(x, Px, n_bins=n_bins, envelope='both')
        
        # Smooth the envelope
        sigma_env = 3
        Px_upper_smooth = gaussian_filter1d(Px_upper, sigma_env)
        Px_lower_smooth = gaussian_filter1d(Px_lower, sigma_env)
        
        # =============================================================================
        # FIND PEAKS (MAXIMA)
        # =============================================================================
        peaks_idx, _ = find_peaks(Px_upper_smooth, 
                                  prominence=0.05 * np.max(Px_upper_smooth),
                                  distance=10)
        
        # =============================================================================
        # FIND ALL INFLECTION POINTS (zero crossings of d²P/dx²)
        # =============================================================================
        d2P_dx2 = np.gradient(Px_upper_smooth, x_env)
        all_inflection_idx = np.where(np.diff(np.sign(d2P_dx2)))[0]
        
        # =============================================================================
        # FIND INFLECTION POINTS AFTER EACH PEAK
        # =============================================================================
        results = find_inflection_after_peaks(x_env, Px_upper_smooth, peaks_idx, all_inflection_idx)
        
        # Extract indices for plotting
        inflection_after_peak_idx = [r['inflection_idx'] for r in results if r['inflection_idx'] is not None]
        
        # =============================================================================
        # PRINT RESULTS
        # =============================================================================
        print("="*70)
        print(f"PEAK AND INFLECTION POINT ANALYSIS: {temp_key}")
        print("="*70)
        
        for i, r in enumerate(results):
            print(f"\nPeak {i+1}:")
            print(f"   Peak location:       x = {r['peak_x']:.5f} m, dP/dx = {r['peak_y']:.2e}")
            if r['inflection_idx'] is not None:
                print(f"   Inflection after:    x = {r['inflection_x']:.5f} m, dP/dx = {r['inflection_y']:.2e}")
                print(f"   Distance (Δx):       {r['delta_x']:.5f} m")
            else:
                print(f"   Inflection after:    None found")
        
        # =============================================================================
        # PLOT: dP/dx envelope with peaks and inflection points
        # =============================================================================
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Raw scatter (faded)
        ax1.scatter(x, Px, alpha=0.15, s=8, c='blue', label='Raw data')
        
        # Upper envelope
        ax1.plot(x_env, Px_upper_smooth, 'k-', linewidth=2.5, label='Upper envelope')
        
        # Mark peaks (maxima)
        ax1.scatter(x_env[peaks_idx], Px_upper_smooth[peaks_idx], 
                    c='red', s=250, zorder=5, marker='v', 
                    edgecolor='black', linewidth=1.5,
                    label=f'Peaks (n={len(peaks_idx)})')
        
        # Mark inflection points after peaks
        ax1.scatter(x_env[inflection_after_peak_idx], Px_upper_smooth[inflection_after_peak_idx], 
                    c='lime', s=250, zorder=5, marker='o', 
                    edgecolor='black', linewidth=1.5,
                    label=f'Inflection after peak (n={len(inflection_after_peak_idx)})')
        
        # Draw arrows connecting peaks to their inflection points
        for r in results:
            if r['inflection_idx'] is not None:
                ax1.annotate('', 
                             xy=(r['inflection_x'], r['inflection_y']),
                             xytext=(r['peak_x'], r['peak_y']),
                             arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        
        ax1.set_xlabel("X [m]", fontsize=18)
        ax1.set_ylabel("dP/dx", fontsize=18)
        ax1.set_title(f"Peaks and Inflection Points — {case_by_hl}", fontsize=21, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # =============================================================================
        # PLOT: d²P/dx² showing the relationship
        # =============================================================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Top: dP/dx envelope
        ax1.plot(x_env, Px_upper_smooth, 'b-', linewidth=2, label='Upper envelope (dP/dx)')
        ax1.scatter(x_env[peaks_idx], Px_upper_smooth[peaks_idx], 
                    c='red', s=200, zorder=5, marker='v', edgecolor='black', label='Peaks')
        ax1.scatter(x_env[inflection_after_peak_idx], Px_upper_smooth[inflection_after_peak_idx], 
                    c='lime', s=200, zorder=5, marker='o', edgecolor='black', label='Inflection after peak')
        ax1.set_ylabel("dP/dx", fontsize=16)
        ax1.set_title(f"Peak → Inflection Analysis — {case_by_hl}", fontsize=18, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom: d²P/dx² 
        ax2.plot(x_env, d2P_dx2, 'b-', linewidth=1.5, label='d²P/dx²')
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax2.scatter(x_env[all_inflection_idx], d2P_dx2[all_inflection_idx], 
                    c='gray', s=80, alpha=0.5, label='All zero crossings')
        ax2.scatter(x_env[inflection_after_peak_idx], d2P_dx2[inflection_after_peak_idx], 
                    c='lime', s=150, zorder=5, marker='o', edgecolor='black',
                    label='Inflection after peak')
        
        # Mark where peaks occur on the d²P/dx² plot
        for peak_i in peaks_idx:
            ax2.axvline(x_env[peak_i], color='red', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel("X [m]", fontsize=16)
        ax2.set_ylabel("d²P/dx²", fontsize=16)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        
        P_values.append(P[inflection_after_peak_idx[0]])
        
 
#%%




"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Using Small Perturbation Theory and comparing to RANS Simulations
#------------------------------------------------------------------------------------------------------------------------------------#
"""


"""
COMPLETE PHASE-CORRECTED SMALL PERTURBATION THEORY CODE
========================================================
This script combines symbolic mathematics and numerical computation to analyze
supersonic flow over wavy walls using linearized potential flow theory.
"""

import numpy as np
import sympy as sp 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sympy import symbols, init_printing, Integral, sqrt , pprint, simplify
from pygasflow import isentropic_solver


# ================================================================
# SECTION 1: SYMBOLIC SETUP AND FUNCTION CREATION
# ================================================================



# Functions #
def find_wall_source_point(x, y, beta, y_wall_func, l):
    """Find the wall source point for a given point (x,y) in the flow field"""
    if y <= y_wall_func(np.clip(x, 0, l)):
        return x
    
    def equation(x_wall):
        y_wall_val = y_wall_func(x_wall)
        return x_wall - (x - beta * (y - y_wall_val))
    
    x_wall_guess = x - beta * y
    x_wall_guess = np.clip(x_wall_guess, 0, l)
    
    try:
        x_wall_solution = fsolve(equation, x_wall_guess)[0]
        return np.clip(x_wall_solution, 0, l)
    except:
        return x_wall_guess

def compute_phi_corrected(x, y, f_indefinite_func, y_wall_func, beta, l, h, V_infty, B, C):
    """
    Compute velocity potential at points (x,y) using zone of influence method.
    
    Parameters:
    -----------
    x, y : array-like
        Coordinates where to compute phi
    f_indefinite_func : callable
        Function requiring (x, h, l, V_infty, B)
    y_wall_func : callable
        Wall shape function
    beta : float
        √(M²-1)
    l, h : float
        Geometry parameters
    V_infty : float
        Freestream velocity
    B : float
        M²-1
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    if x.ndim == 2:
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_wall = find_wall_source_point(x[i,j], y[i,j], beta, y_wall_func, l)
                # Pass all 5 required parameters
                result[i,j] = f_indefinite_func(x_wall, h, l, V_infty, B, C)
        return result
    else:
        result = np.zeros_like(x)
        for i in range(len(x)):
            x_wall = find_wall_source_point(x[i], y[i], beta, y_wall_func, l)
            # Pass all 5 required parameters
            result[i] = f_indefinite_func(x_wall, h, l, V_infty, B)
        return result


def compute_torque_2D_norm(x, y, p, R):
    """
    Tangential force and torque per unit span per unit length from 2D pressure profile.
    
    Parameters
    ----------
    x, y : arrays - wall coordinates [m]
    p : array - wall pressure [Pa]
    R : float - radius from rotation axis [m]
    
    Returns
    -------
    dict with:
        F_theta : tangential force per unit area [N/m²]
        tau : torque per unit span per unit length [N·m/m²]
    """
    dydx = np.gradient(y, x)
    
    if hasattr(np, 'trapezoid'):
        F_theta = np.trapezoid(p * dydx, x)
    else:
        F_theta = np.trapz(p * dydx, x)
    
    tau = F_theta * R
    
    # Normalize by projected length
    L = x[-1] - x[0]
    
    return {
        'F_theta_norm': F_theta / L,
        'tau': tau / L,
        'F_theta': F_theta
    }



#### End Functions ####

# ====== SETUP ====== #
h_l_values = np.arange(0.02,0.1,0.01) # Defining the h_l values that we have 



results_list = []







counter = 0 
for k, h_l in enumerate(h_l_values):
    #h_l = 0.02
    N = 1 
    l = 0.1
    h = h_l * l 
    num_of_points = 1000
    
    
    # Defining the geometry # 
    lam = l / (2 * N + 1)*2 
    x_wave = np.linspace(0, l, num_of_points)
    y_wave = h * np.sin(2 * np.pi * x_wave / lam)
    #y_wave = h*np.cos((2*np.pi*x_wave) / l ) # Textbook Problem 
    
    
    # Defining variables and equations # 
    x_variable = sp.Symbol('x_variable')
    y_variable = sp.Symbol('y_variable')
    y_wall = 0  
    h_variable = sp.Symbol('h_variable')
    l_variable = sp.Symbol("l_variable")
    
    
    # The equation fo the wall is defined here # 
    y_equation = h_variable*sp.sin((2 * sp.pi * x_variable)/lam) # Re-define if you have changed y_wave
    
    
    # Range of Mach numbers # 
    M_infty_range = np.arange(1.5,4.5,0.5)
    
    
 
    
    # For loop to to evaluate the symbolic functions # 
    
    
    for M_infty in M_infty_range:
        # Defining the flow domain # 
        #M_infty = 2.0 # Change according to the Mach number you're optimzing to 
        
        B = M_infty**2 - 1
        
        # Flow properties (replace with your actual data)
        gamma = 1.4  # Standard air
        R = 287
        
        # Flow results # 
        flow_results = isentropic_solver("m",M_infty)
        P_P0 = flow_results[1]
        rho_rho0 = flow_results[2]
        T_T0 = flow_results[3]
        
        # Ambient Conditions # 
        T0 = 300 #kelvin 
        P0 = 1e6 #Pa
        
        # Getting static conditions # 
        T_infty = T_T0 * T0  # K
        p_infty = P_P0 * P0  # Pa
        rho_infty = p_infty / (R* T_infty)
        
        #### End of initial conditions ####
        
        # Derived properties
        a_infty = np.sqrt(gamma * R * T_infty)
        V_infty = a_infty * M_infty
        

        # Plotting the geometry # 
        plt.plot(x_wave,y_wave)
        plt.xlabel("X[m]", fontsize = 16)
        plt.ylabel("Y[m]", fontsize = 16)
        plt.title("Wavy Section Geometry",fontsize = 24)
        plt.grid()
        plt.show()
        
        
        
        # ====== SYMBOLIC MATHEMATICS ====== #
        
        dy_dx = sp.diff(y_equation, x_variable)
        V_infty_variable = sp.Symbol("V_infty_variable")
        B_variable = sp.Symbol("B_variable")
        
        dphi_dy_wall = dy_dx * V_infty_variable
        df_dx = dphi_dy_wall / -sp.sqrt(B_variable)
        
        # Integration with phase correction
        C = sp.Symbol('C')
        y_variable = sp.Symbol('y_variable')
        f_indefinite = sp.integrate(df_dx, x_variable) + C
        phi_xy = sp.simplify(f_indefinite).subs(x_variable, x_variable - B_variable * y_variable)
        
        # Defining general equation with constant # 
        phi_xy_general = phi_xy
        phi_xy_wall = phi_xy_general.subs(y_variable, 0)
        
        dphi_dx = sp.diff(phi_xy_general, x_variable)
        dphi_dy = sp.diff(phi_xy_general, y_variable)
        
        Cp = ((-2/V_infty_variable) * dphi_dx)
        Cp_wall = Cp.subs(y_variable, 0)
        
        # Finding Values from the velocity potential equations # 
        u_prime = dphi_dx
        v_prime = dphi_dy
        V_x = V_infty_variable + u_prime
        V_y = v_prime
        
        # Pritning symbolic results # 
        pprint(Cp_wall)
        
        # ====== CREATE NUMERICAL FUNCTIONS ====== #

        
        # CRITICAL: Wall shape function
        y_wall_func = sp.lambdify(x_variable, y_equation.subs([(h_variable, h), (l_variable, l)]), 'numpy')
        
        # f_indefinite function (5 parameters)
        f_indefinite_func = sp.lambdify((x_variable, h_variable, l_variable, V_infty_variable, B_variable, C), 
                                         f_indefinite, 'numpy')
        
        # Velocity potential functions (7 parameters)
        phi_xy_general_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable, C), 
                                           phi_xy_general, 'numpy')
        
        phi_xy_wall_func = sp.lambdify((x_variable, h_variable, l_variable, V_infty_variable, B_variable, C),
                                        phi_xy_wall, 'numpy')
        
        # Partial derivative functions (7 parameters)
        dphi_dx_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                    dphi_dx, 'numpy')
        
        dphi_dy_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                    dphi_dy, 'numpy')
        
        # Pressure coefficient functions
        Cp_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                               Cp, 'numpy')
        
        Cp_wall_func = sp.lambdify((x_variable, h_variable, l_variable, V_infty_variable, B_variable),
                                    Cp_wall, 'numpy')
        
        # Velocity perturbation functions
        u_prime_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                    u_prime, 'numpy')
        
        v_prime_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                    v_prime, 'numpy')
        
        # Total velocity component functions
        V_x_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                V_x, 'numpy')
        
        V_y_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                V_y, 'numpy')
        

        
        # ================================================================
        # SECTION 2: NUMERICAL COMPUTATION AND VISUALIZATION
        # ================================================================
 
        
        
        # Getting wall values # 
        
        
        # ====== ZONE OF INFLUENCE ====== #
        
        
        
           
        # ====== DOMAIN ====== #
        
        x_min, x_max = 0, l
        y_min = -h - 0.01
        y_max = 0.3
        
        x_grid = np.linspace(x_min, x_max, 150)
        y_grid = np.linspace(y_min, y_max, 150)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        

        phi_corrected = compute_phi_corrected(X_grid, Y_grid, f_indefinite_func, y_wall_func, 
                                              np.sqrt(B), l, h, V_infty, B, 5)
        
        # Velocity components
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        
        u_prime = np.gradient(phi_corrected, dx, axis=1)
        v_prime = np.gradient(phi_corrected, dy, axis=0)
        
        u_total = V_infty + u_prime
        v_total = v_prime
        V_total = np.sqrt(u_total**2 + v_total**2)
        
        # Pressure coefficient
        Cp_grid = (-2/V_infty) * u_prime
        
        # Pressure
        pressure_ratio = 1 + (gamma * M_infty**2 / 2) * Cp_grid
        P_local = p_infty * pressure_ratio
        
        # Temperature (linearized)
        temperature_ratio = 1 - (gamma - 1) * M_infty**2 * Cp_grid / 2
        temperature_ratio = np.clip(temperature_ratio, 0.5, 2.0)
        T_local = T_infty * temperature_ratio
        
        # Mach number
        M_local = V_total / a_infty
        
        # Mask below wall
        mask = np.zeros_like(X_grid, dtype=bool)
        for i, x_val in enumerate(x_grid):
            y_wall_at_x = y_wall_func(x_val)
            mask[:, i] = Y_grid[:, i] >= y_wall_at_x
        
        Cp_masked = np.where(mask, Cp_grid, np.nan)
        P_masked = np.where(mask, P_local, np.nan)
        M_masked = np.where(mask, M_local, np.nan)
        

        
        # ====== CHARACTERISTIC SOURCE POINTS ====== #
        
        def get_wall_critical_points(l, n):
            """Generate critical points along the wall for characteristic lines"""
            wavelength = l / n
            points = []
            
            for i in range(n):
                points.append(i * wavelength)
                points.append(i * wavelength + wavelength/4)
                points.append(i * wavelength + wavelength/2)
                points.append(i * wavelength + 3*wavelength/4)
            points.append(n * wavelength)
            
            all_points = []
            for i in range(len(points)-1):
                all_points.append(points[i])
                all_points.append(points[i] + (points[i+1] - points[i])/3)
                all_points.append(points[i] + 2*(points[i+1] - points[i])/3)
            all_points.append(points[-1])
            
            return np.array([p for p in all_points if 0.01 <= p <= l-0.01])
        
        wall_x_sources = get_wall_critical_points(l, n)
        wall_y_sources = y_wall_func(wall_x_sources)
        
        # ================================================================
        # SECTION 3: VISUALIZATION
        # ================================================================
        

        '''
        # ====== PLOT 1: PRESSURE COEFFICIENT (Cp) ====== #
        fig_height = 8
        fig_width = 8 
        
        plt.figure(figsize=(fig_height, fig_width))
        
        contour = plt.contourf(X_grid, Y_grid, Cp_masked, levels=40, cmap='RdBu_r')
        cbar = plt.colorbar(contour, label='Cp(x, y)', pad=0.02)
        
        # Wall
        plt.fill_between(x_wave, y_min, y_wave, color='gray', alpha=0.8, label='Wall', zorder=5)
        plt.plot(x_wave, y_wave, 'k-', linewidth=2, zorder=6)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='y=0')
        
        # Characteristics
        wall_colors = plt.cm.plasma((wall_y_sources - wall_y_sources.min()) / 
                                     (wall_y_sources.max() - wall_y_sources.min()))
        
        for i, x_wall in enumerate(wall_x_sources):
            y_wall_val = y_wall_func(x_wall)
            y_char = np.linspace(y_wall_val, y_max, 100)
            x_char = x_wall + np.sqrt(B) * (y_char - y_wall_val)
            valid = (x_char >= x_min) & (x_char <= x_max)
            
            if np.any(valid):
                plt.plot(x_char[valid], y_char[valid], color=wall_colors[i], 
                        alpha=0.5, linewidth=1.0, zorder=3)
        
        plt.scatter(wall_x_sources, wall_y_sources, c=wall_y_sources, cmap='plasma', 
                   s=30, edgecolors='white', linewidth=1, zorder=7, label='Sources')
        
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Pressure Coefficient Cp(x, y)', fontsize=21, fontweight='bold')
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True, alpha=0.2)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.savefig('cp_contour.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ====== PLOT 2: PRESSURE (P) ====== #
        
        plt.figure(figsize=(fig_height, fig_width))
        
        contour = plt.contourf(X_grid, Y_grid, P_masked, levels=40, cmap='coolwarm')
        cbar = plt.colorbar(contour, label='P(x, y) [Pa]', pad=0.02)
        
        plt.fill_between(x_wave, y_min, y_wave, color='gray', alpha=0.8, label='Wall', zorder=5)
        plt.plot(x_wave, y_wave, 'k-', linewidth=2, zorder=6)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='y=0')
        
        for i, x_wall in enumerate(wall_x_sources):
            y_wall_val = y_wall_func(x_wall)
            y_char = np.linspace(y_wall_val, y_max, 100)
            x_char = x_wall + np.sqrt(B) * (y_char - y_wall_val)
            valid = (x_char >= x_min) & (x_char <= x_max)
            
            if np.any(valid):
                plt.plot(x_char[valid], y_char[valid], color=wall_colors[i], 
                        alpha=0.5, linewidth=1.0, zorder=3)
        
        plt.scatter(wall_x_sources, wall_y_sources, c=wall_y_sources, cmap='plasma', 
                   s=30, edgecolors='white', linewidth=1, zorder=7, label='Sources')
        
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Pressure P(x, y)', fontsize=21, fontweight='bold')
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True, alpha=0.2)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.savefig('pressure_contour.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ====== PLOT 3: MACH NUMBER (M) ====== #
        
        plt.figure(figsize=(fig_height, fig_width))
        
        contour = plt.contourf(X_grid, Y_grid, M_masked, levels=40, cmap='jet')
        cbar = plt.colorbar(contour, label='M(x, y)', pad=0.02)
        
        plt.fill_between(x_wave, y_min, y_wave, color='gray', alpha=0.8, label='Wall', zorder=5)
        plt.plot(x_wave, y_wave, 'k-', linewidth=2, zorder=6)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='y=0')
        
        for i, x_wall in enumerate(wall_x_sources):
            y_wall_val = y_wall_func(x_wall)
            y_char = np.linspace(y_wall_val, y_max, 100)
            x_char = x_wall + np.sqrt(B) * (y_char - y_wall_val)
            valid = (x_char >= x_min) & (x_char <= x_max)
            
            if np.any(valid):
                plt.plot(x_char[valid], y_char[valid], color='white', 
                        alpha=0.3, linewidth=0.8, zorder=3)
        
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.title('Mach Number M(x, y)', fontsize=21, fontweight='bold')
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True, alpha=0.2)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()
        plt.savefig('mach_contour.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n" + "="*70)
        print("COMPLETE! THREE CONTOUR PLOTS GENERATED")
        print("="*70)
        print("Files saved:")
        print("  1. cp_contour.png - Pressure Coefficient")
        print("  2. pressure_contour.png - Absolute Pressure")
        print("  3. mach_contour.png - Mach Number Distribution")
        print("="*70)
        '''
        
    
        """
        #------------------------------------------------------------------------------------------------------------------------------------#
                                            Comparing small perturbations theory with RANS Simulation
        #------------------------------------------------------------------------------------------------------------------------------------#
        """
        ### %Comparing Small perturbation theory with RANS viscous simulations amoung all cases ###
    
    
    
        # Getting all the keys of the dictionray so I am able to plot through all of them # 
        keys = list(ds_by_case.keys())
        n_cases = len(keys)
    
    
    
        # Masking to the desired points [0,0.1]
        x_wall_RANS = ds_by_case[keys[counter]]["X"].data 
        x_min = 0 
        x_max = l
        mask = (x_min < x_wall_RANS) & (x_wall_RANS < x_max)
    
        # Defining pressure and y at the wall based on the mask
        P_wall_RANS = ds_by_case[keys[counter]]["P"].data[mask]
        y_wall_RANS = ds_by_case[keys[counter]]["Y"].data[mask] 
        x_wall_RANS = x_wall_RANS[mask]
    
    
        # Define points along the wall
        x_wall = np.linspace(0, l, len(P_wall_RANS))
        y_wall = y_wall_func(x_wall)
    
        Cp_wall_results = Cp_wall_func(x_wall, h, l, V_infty, B)
        P_wall = Cp_wall_results*0.5*rho_infty*V_infty**2 + p_infty
        P_diff = (np.abs(P_wall_RANS - P_wall) / P_wall_RANS) * 100
    
    
        # Computing torque for both cases #
        R = 0 # No torque is actually be extraced here. 
        hl_RANS = compute_torque_2D_norm(x_wall_RANS, y_wall_RANS, P_wall_RANS, R)
        hl_smallPert = compute_torque_2D_norm(x_wall, y_wall, P_wall, R)
        
        # Computing the axial force 
        axialForce_RANS = hl_RANS['F_theta']
        axialForce_smallPert = hl_smallPert['F_theta']
        
        ##### Creating Plots ###### 
        
        # Imposing figure axes # 
        fig, (ax1,ax2) = plt.subplots(1,2,figsize = (10, 6))
        
        
        # Plot on ax1 (P vs X) #
        ax1.plot(x_wall_RANS, P_wall_RANS, label="RANS", linewidth = 3)
        ax1.plot(x_wall, P_wall, label="Small Perturbation", linestyle='--', linewidth = 3)
        ax1.set_title(f"{keys[counter]}: $P_{{wall}}$ Vs X", fontsize=14)
        ax1.set_xlabel("X [m]", fontsize=12)
        ax1.set_ylabel(r"$P_{wall}$ [Pa]", fontsize=12)
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()
        
    
    
        # P_difference Vs X Plot ax2 # 
        ax2.plot(x_wall, P_diff)
        ax2.set_xlim([0,l])
        ax2.set_title(r"$P_{difference}$ Vs X", fontsize = 24)
    
        ax2.set_xlabel("X [m]", fontsize = 18)
        ax2.set_ylabel(r"$P_{difference}$[%]", fontsize = 18) 
    
        ax2.grid(True, which = "both")
        ax2.legend()
        '''
        # P_difference Vs X Plot ax2 # 
        ax3.plot(x_wall_RANS, axialForce_RANS)
        ax3.set_plot(x_wall,axialForce_smallPert)
        
        ax3.xlim([0,l])
        ax3.set_title(r"$P_{difference}$ Vs X", fontsize = 24)
    
        ax3.set_xlabel("X [m]", fontsize = 18)
        ax3.set_ylabel(r"$P_{difference}$[%]", fontsize = 18) 
    
        ax3.grid(True, which = "both")
        ax3.legend()
        '''
        axialForce_diff =  (1 - (axialForce_RANS / axialForce_smallPert)) * 100
        results_list.append({
            'h/l': h_l,
            'M_infty': M_infty,
            'F_axial_RANS [N/m]': axialForce_RANS,
            'F_axial_SmallPert [N/m]': axialForce_smallPert,
            'Difference [%]': axialForce_diff,
            'Case_Key': keys[counter]
        })
        
        
        # Convert list of dictionaries to DataFrame
        df_results = pd.DataFrame(results_list)

        pivot_table = df_results.pivot_table(
            values='Difference [%]',
            index='h/l',
            columns='M_infty',
            aggfunc='mean'  # In case of duplicates
        )
                
        plt.show()
        
        
        # Adding to the counter # 
        counter += 1
        
#%%

# Exporting Table into excel that compares M\infty Axial force, percentage difference, and the case. # 
df_results.to_csv(r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Graphs\axial_force_comparison.csv', index=False)
pivot_table.to_csv(r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Graphs\axial_force_pivot.csv') 





#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                        Creating shock-expansion theory to understand the physics at the wall 
#------------------------------------------------------------------------------------------------------------------------------------#
"""


import seaborn as sns

# Imposing h/l # 
min_hl = 0 
max_hl = 0.1

 
# Corodinates # 
temp_key = "h_l_0.02_Mach_1.5"
x_temp = ds_by_case[temp_key]["X"].data
y_temp = ds_by_case[temp_key]["Y"].data


# Imposed mask # 
mask = (min_hl <= x_temp) & (max_hl >= x_temp)

# imposing mask #
x_temp = x_temp[mask]
y_temp = y_temp[mask]

# Getting the gradient to compute the respective angle # 
dy_dx = np.gradient(y_temp,x_temp)
theta = np.degrees((np.arctan(dy_dx)))


# Getting th



# Getting dtheta/dx to determine if the flow is compressing or expanding # 
dtheta_dx = np.gradient(theta)

# Plotting the x and y values # 
ax = sns.lineplot(x = x_temp,y = y_temp, color = 'black')
sns.set_style("whitegrid")

ax.set_xlabel('X [m]', fontsize = 14)
ax.set_ylabel('Y [m]', fontsize = 14)
ax.set_title("X Vs Y", fontsize = 24)


plt.show()




































#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                        Creating shock-expansion theory to understand the physics at the wall 
#------------------------------------------------------------------------------------------------------------------------------------#
"""


# ====== FUNCTIONS ====== # 
def shock_expansion_wavy_wall(x_wall, y_wall, M_inf, P_inf, gamma=1.4):
    """
    Shock-expansion theory for wavy wall with detachment detection.
    
    Returns:
        P_wall: pressure distribution (up to detachment point)
        M_wall: Mach number distribution
        detachment_info: dict with detachment information if occurred
    """
    N = len(x_wall)
    P_wall = np.zeros(N)
    M_wall = np.zeros(N)
    
    # Initial conditions
    P_wall[0] = P_inf
    M_wall[0] = M_inf
    
    # Track if detachment occurs
    detachment_occurred = False
    detachment_info = None
    
    for i in range(1, N):
        # Calculate local wall angles
        if i == 1:
            theta_old = 0  # Assume aligned with freestream at start
        else:
            theta_old = np.arctan2(y_wall[i] - y_wall[i-1], 
                                   x_wall[i] - x_wall[i-1])
        
        if i < N - 1:
            theta_new = np.arctan2(y_wall[i+1] - y_wall[i], 
                                   x_wall[i+1] - x_wall[i])
        else:
            theta_new = theta_old
        
        # Deflection angle
        delta_theta = theta_new - theta_old
        
        if delta_theta > 1e-6:  # Compression
            # Oblique shock relations
            M_new, P_new, detached, shock_info = oblique_shock(
                M_wall[i-1], P_wall[i-1], delta_theta, gamma
            )
            
            if detached:
                # Shock detachment detected!
                detachment_occurred = True
                detachment_info = {
                    'location_index': i,
                    'x_location': x_wall[i],
                    'y_location': y_wall[i],
                    'Mach_number': M_wall[i-1],
                    'deflection_angle_deg': shock_info['theta_requested'],
                    'max_deflection_deg': shock_info['theta_max'],
                    'message': (f"Shock detachment at x = {x_wall[i]:.3f}, "
                              f"y = {y_wall[i]:.3f}\n"
                              f"Local Mach number: {M_wall[i-1]:.3f}\n"
                              f"Deflection angle: {shock_info['theta_requested']:.2f}° "
                              f"exceeds maximum {shock_info['theta_max']:.2f}°\n"
                              f"A detached bow shock is expected beyond this point.")
                }
                
                # Truncate arrays to detachment point
                P_wall = P_wall[:i]
                M_wall = M_wall[:i]
                x_wall = x_wall[:i]
                y_wall = y_wall[:i]
                
                print("\n" + "="*70)
                print("⚠️  SHOCK DETACHMENT DETECTED  ⚠️")
                print("="*70)
                print(detachment_info['message'])
                print("="*70 + "\n")
                
                break
            
            M_wall[i] = M_new
            P_wall[i] = P_new
            
        elif delta_theta < -1e-6:  # Expansion
            # Prandtl-Meyer expansion (always attached)
            M_wall[i], P_wall[i] = prandtl_meyer_expansion(
                M_wall[i-1], P_wall[i-1], abs(delta_theta), gamma
            )
        else:  # No change
            M_wall[i] = M_wall[i-1]
            P_wall[i] = P_wall[i-1]
    
    return P_wall, M_wall, detachment_info
    


# OBLIQUE SHOCK FUNCITON # 
def oblique_shock(M1, P1, theta, gamma=1.4):
    """
    Apply oblique shock relations.
    
    Returns:
        M2: exit Mach number (or None if detached)
        P2: exit pressure (or None if detached)
        detached: boolean flag
        shock_info: dictionary with additional info
    """
    # Solve for shock angle
    beta, detached, theta_max = solve_shock_angle(M1, theta, gamma)
    
    if detached:
        # Return info about detachment
        shock_info = {
            'detached': True,
            'theta_requested': np.degrees(theta),
            'theta_max': np.degrees(theta_max),
            'M1': M1
        }
        return None, None, detached, shock_info
    
    # Normal component of Mach number
    Mn1 = M1 * np.sin(beta)
    
    # Normal shock relations
    Mn2_sq = ((gamma - 1)*Mn1**2 + 2) / (2*gamma*Mn1**2 - (gamma - 1))
    Mn2 = np.sqrt(Mn2_sq)
    
    # Pressure ratio
    P2 = P1 * (1 + (2*gamma/(gamma + 1)) * (Mn1**2 - 1))
    
    # Exit Mach number
    M2 = Mn2 / np.sin(beta - theta)
    
    shock_info = {
        'detached': False,
        'beta': np.degrees(beta),
        'theta': np.degrees(theta),
        'theta_max': np.degrees(theta_max)
    }
    
    return M2, P2, detached, shock_info




# SOLVE SHOCK ANGLE FUNCTION # 
def solve_shock_angle(M1, theta, gamma=1.4):
    """
    Solve theta-beta-Mach relation iteratively.
    
    Returns:
        beta: shock angle (radians), or None if detached
        detached: boolean flag indicating if shock is detached
        theta_max: maximum possible deflection angle
    """
    from scipy.optimize import fsolve
    
    # First, check if deflection exceeds maximum
    theta_max, beta_at_max = calculate_max_deflection_angle(M1, gamma)
    
    if theta > theta_max:
        # Shock will be detached
        return None, True, theta_max
    
    # Shock is attached, solve for shock angle
    def theta_beta_relation(beta):
        num = 2 / np.tan(beta) * (M1**2 * np.sin(beta)**2 - 1)
        den = M1**2 * (gamma + np.cos(2*beta)) + 2
        return np.arctan(num / den) - theta
    
    # Initial guess: weak shock solution
    mu = np.arcsin(1/M1)  # Mach angle
    beta_guess = mu + theta/2
    
    # Ensure guess is in valid range
    beta_guess = np.clip(beta_guess, mu + 0.01, np.pi/2 - 0.01)
    
    try:
        beta = fsolve(theta_beta_relation, beta_guess)[0]
        
        # Verify solution is physical (weak shock, not strong shock)
        # Weak shock: beta < beta_at_max
        if beta > beta_at_max:
            # Strong shock solution - typically want weak shock
            # Re-solve with different initial guess
            beta_guess = (mu + theta_max) / 2
            beta = fsolve(theta_beta_relation, beta_guess)[0]
        
        return beta, False, theta_max
        
    except:
        # Solver failed - likely at or near detachment
        return None, True, theta_max



# MAX DEFLECTION ANGLE CALCULATION FUNCTION # 
def calculate_max_deflection_angle(M1, gamma=1.4):
    """
    Calculate maximum deflection angle for given Mach number.
    Beyond this angle, shock detaches.
    
    Returns:
        theta_max: maximum deflection angle (radians)
        beta_at_max: shock angle at maximum deflection (radians)
    """
    from scipy.optimize import minimize_scalar
    
    def theta_from_beta(beta):
        """Calculate deflection angle from shock angle"""
        num = 2 / np.tan(beta) * (M1**2 * np.sin(beta)**2 - 1)
        den = M1**2 * (gamma + np.cos(2*beta)) + 2
        return np.arctan(num / den)
    
    # The maximum occurs somewhere between Mach angle and 90°
    mu = np.arcsin(1/M1)  # Mach angle (minimum possible)
    
    # Find maximum by minimizing negative theta
    result = minimize_scalar(lambda b: -theta_from_beta(b), 
                            bounds=(mu, np.pi/2 - 0.01),
                            method='bounded')
    
    beta_at_max = result.x
    theta_max = theta_from_beta(beta_at_max)
    
    return theta_max, beta_at_max


# PRANDTL-MEYER EXPANSION FUNCTION # 
def prandtl_meyer_expansion(M1, P1, theta, gamma):
    """Apply Prandtl-Meyer expansion"""
    # Prandtl-Meyer function
    def nu(M):
        a = np.sqrt((gamma + 1) / (gamma - 1))
        b = np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))
        c = np.sqrt(M**2 - 1)
        return a * np.arctan(b) - np.arctan(c)
    
    # Initial and final PM angles
    nu1 = nu(M1)
    nu2 = nu1 + theta  # Expansion turns away, so add theta
    
    # Solve for M2
    from scipy.optimize import fsolve
    M2 = fsolve(lambda M: nu(M) - nu2, M1 + 0.5)[0]
    
    # Isentropic pressure ratio
    P2 = P1 * ((1 + (gamma-1)/2 * M1**2) / 
               (1 + (gamma-1)/2 * M2**2))**(gamma/(gamma-1))
    
    return M2, P2




# ====== SETUP ====== #
h_l = 0.02
h = 1/15
n = 5
l = h / h_l
num_of_points = 1500 
x_wave = np.linspace(0, l, num_of_points)
y_wave = h * np.sin((n * np.pi * x_wave) / l)


# ====== FREESTREAM CONDITIONS ===== #
M_inf = 1.6 
P_inf = 9e5 # Pascals 
gamma = 1.4 



# ======= COMPUTING WALL CONDITIONS ====== #
P_wall,M_wall,detachment_info = shock_expansion_wavy_wall(x_wave, y_wave, M_inf, P_inf, gamma)


# ====== PLOTTING RESULTS ======= # 


# Pressure Vs X # 
plt.plot(x_wave, P_wall)
plt.title("Pressure Vs X")
plt.ylabel("Pressure[Pa]")
plt.xlabel("X")
plt.grid()
plt.show()


# Mach Vs X # 
plt.plot(x_wave, M_wall)
plt.title("Mach Vs X")
plt.ylabel("Mach")
plt.xlabel("X")
plt.grid()
plt.show()


#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                        Automatically detecting shock waves and their angles from Schlieren imaging
#------------------------------------------------------------------------------------------------------------------------------------#
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, feature
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           IMAGE LOADING AND PREPROCESSING
#------------------------------------------------------------------------------------------------------------------------------------#

def load_schlieren_image(image_path):
    """Load and convert Schlieren image to grayscale."""
    img = io.imread(image_path)
    
    if len(img.shape) == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img
    
    img_gray = img_gray.astype(float)
    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())
    
    print(f"Image loaded: {img_gray.shape[1]} x {img_gray.shape[0]} pixels")
    return img_gray


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           SHOCK DETECTION
#------------------------------------------------------------------------------------------------------------------------------------#

def detect_shocks_canny(img_gray, sigma=1.5, low_threshold=0.1, high_threshold=0.3):
    """Detect shock waves using Canny edge detection."""
    edges = feature.canny(
        img_gray,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    
    print(f"Canny detected {np.sum(edges)} edge pixels")
    return edges


def detect_shocks_sobel(img_gray, threshold=0.2, sigma=1.5):
    """Detect shock waves using Sobel edge detection."""
    img_smooth = ndimage.gaussian_filter(img_gray, sigma=sigma)
    
    gradient_x = ndimage.sobel(img_smooth, axis=1)
    gradient_y = ndimage.sobel(img_smooth, axis=0)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    edges = gradient_magnitude > threshold
    
    print(f"Sobel detected {np.sum(edges)} edge pixels")
    return edges


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           COORDINATE EXTRACTION AND CLUSTERING
#------------------------------------------------------------------------------------------------------------------------------------#

def extract_shock_coordinates(edges, x_scale, y_scale, flip_y=False):
    """Convert edge pixels to physical coordinates."""
    y_pixels, x_pixels = np.where(edges)
    
    # Convert to physical coordinates
    x_physical = x_pixels * x_scale
    
    if flip_y:
        # Flip y-axis to match CFD convention (y increases upward)
        image_height = edges.shape[0]
        y_physical = (image_height - y_pixels) * y_scale
    else:
        # Keep image convention (y increases downward from top)
        y_physical = y_pixels * y_scale
    
    print(f"Extracted {len(x_physical)} shock points")
    return x_physical, y_physical


def cluster_shocks(x_points, y_points, eps=0.5, min_samples=20):
    """Cluster shock points using DBSCAN."""
    shock_points = np.column_stack((x_points, y_points))
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(shock_points)
    
    n_shocks = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"DBSCAN found {n_shocks} distinct shocks")
    print(f"Noise points: {n_noise}")
    
    return labels


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           SHOCK ANGLE CALCULATION
#------------------------------------------------------------------------------------------------------------------------------------#

def calculate_shock_angles(x_points, y_points, labels, y_threshold=None, method='ransac', 
                          min_points=10, min_extent=0.5):
    """Calculate shock angles for each detected shock."""
    
    shock_data = []
    
    for shock_id in set(labels):
        if shock_id == -1:  # Skip noise
            continue
        
        mask = (labels == shock_id)
        x_shock = x_points[mask]
        y_shock = y_points[mask]
        
        # Filter out wall points if threshold provided
        if y_threshold is not None:
            wall_filter = y_shock > y_threshold
            x_shock_filtered = x_shock[wall_filter]
            y_shock_filtered = y_shock[wall_filter]
        else:
            x_shock_filtered = x_shock
            y_shock_filtered = y_shock
        
        # Skip if not enough points after filtering
        if len(x_shock_filtered) < min_points:
            print(f"Shock {shock_id}: skipped (only {len(x_shock_filtered)} points, need {min_points})")
            continue
        
        # Calculate spatial extent of shock
        x_extent = x_shock_filtered.max() - x_shock_filtered.min()
        y_extent = y_shock_filtered.max() - y_shock_filtered.min()
        total_extent = np.sqrt(x_extent**2 + y_extent**2)
        
        # Skip if shock is too small (likely noise)
        if total_extent < min_extent:
            print(f"Shock {shock_id}: skipped (extent {total_extent:.3f} < {min_extent})")
            continue
        
        # Fit line to calculate angle
        if method == 'ransac':
            X = x_shock_filtered.reshape(-1, 1)
            y = y_shock_filtered
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
        else:  # polyfit
            coeffs = np.polyfit(x_shock_filtered, y_shock_filtered, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
        
        # Calculate angle
        beta_rad = np.arctan(slope)
        beta_deg = np.degrees(beta_rad)
        
        # Calculate mean position for sorting
        x_mean = np.mean(x_shock_filtered)
        
        shock_data.append({
            'original_id': shock_id,
            'x_mean': x_mean,
            'x_all': x_shock,
            'y_all': y_shock,
            'x_filtered': x_shock_filtered,
            'y_filtered': y_shock_filtered,
            'angle': beta_deg,
            'slope': slope,
            'intercept': intercept,
            'n_points': len(x_shock_filtered),
            'n_total': len(x_shock)
        })
    
    # Sort shocks left to right
    shock_data.sort(key=lambda s: s['x_mean'])
    
    return shock_data


#------------------------------------------------------------------------------------------------------------------------------------#
#                                                           VISUALIZATION
#------------------------------------------------------------------------------------------------------------------------------------#

def plot_shock_results(img_gray, shock_data, x_scale, y_scale, y_threshold=None, flip_y=False):
    """Plot shock detection and angle measurement results."""
    
    height, width = img_gray.shape
    
    plt.figure(figsize=(14, 8))
    
    # Show image with shock overlay
    # Always show image in its natural orientation
    extent = [0, width * x_scale, height * y_scale, 0]  # [left, right, bottom, top]
    plt.imshow(img_gray, cmap='gray', extent=extent, aspect='auto', alpha=0.5)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(shock_data)))
    
    for shock_num, shock in enumerate(shock_data):
        x_shock_i = shock['x_all']
        y_shock_i = shock['y_all']
        x_shock_filtered = shock['x_filtered']
        y_shock_filtered = shock['y_filtered']
        slope = shock['slope']
        intercept = shock['intercept']
        beta_deg = shock['angle']
        
        shock_label = shock_num + 1
        
        print(f"Shock {shock_label}:")
        print(f"  Angle β = {beta_deg:.2f}°")
        print(f"  Slope = {slope:.4f}")
        print(f"  Mean x-position = {shock['x_mean']:.3f}")
        print(f"  Points used for fitting = {len(x_shock_filtered)} (out of {len(x_shock_i)} total)")
        print()
        
        # Plot all detected shock points (faded)
        plt.scatter(x_shock_i, y_shock_i, s=20, color=colors[shock_num], 
                   alpha=0.3, label=f'Shock {shock_label} (all points)')
        
        # Highlight points used for fitting
        plt.scatter(x_shock_filtered, y_shock_filtered, s=30, 
                   color=colors[shock_num], edgecolors='black', linewidths=1,
                   label=f'Shock {shock_label} fit points (β={beta_deg:.1f}°)')
        
        # Plot fitted line
        x_line = np.array([x_shock_i.min(), x_shock_i.max()])
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, '--', color=colors[shock_num], linewidth=2.5)
    
    # Draw wall filter line if used
    if y_threshold is not None:
        plt.axhline(y=y_threshold, color='red', linestyle=':', 
                   linewidth=1, alpha=0.7, label=f'Wall filter (y={y_threshold})')
    
    plt.xlabel(r"$X$ [inches]", fontsize=14)
    plt.ylabel(r"$Y$ [inches]", fontsize=14)
    plt.title("Shock Angle Measurement from Schlieren Image", fontsize=24)
    plt.legend(loc='best', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------------------------------------------------#
#                                                           MAIN ANALYSIS
#------------------------------------------------------------------------------------------------------------------------------------#

# ===== USER PARAMETERS - MODIFY THESE ===== #

# Image file path
IMAGE_PATH = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\18_Schilierean Imaging\schilieren.png"



# Physical dimensions of imaged area
PHYSICAL_WIDTH = 5.0   # inches (or your preferred units)
PHYSICAL_HEIGHT = 5.0   # inches



# Edge detection parameters
EDGE_METHOD = 'canny'   # 'canny' or 'sobel'
SIGMA = 1.5             # Gaussian smoothing
THRESHOLD = 0.2         # Edge threshold (only for sobel)



# Clustering parameters
DBSCAN_EPS = 0.3        # Maximum distance between points in same shock
DBSCAN_MIN_SAMPLES = 20 # Minimum points to form a shock



# Wall filtering
Y_THRESHOLD = None      # Set to exclude wall region (e.g., 0.5), or None for no filtering



# Fitting method
FITTING_METHOD = 'ransac'  # 'ransac' (robust) or 'polyfit' (fast)



# False shock rejection filters
MIN_POINTS_PER_SHOCK = 50   # Minimum points for valid shock (increase to reject small clusters)
MIN_SHOCK_EXTENT = 1.0      # Minimum shock length in physical units (increase to reject small features)



# Image orientation
FLIP_Y_AXIS = False         # Set to True if you want y to increase upward (CFD style)



# ===== RUN ANALYSIS ===== #

if __name__ == "__main__":
    
    print("="*70)
    print("SCHLIEREN SHOCK ANGLE ANALYSIS")
    print("="*70)
    print()
    
    # Load image
    img_gray = load_schlieren_image(IMAGE_PATH)
    
    # Calculate pixel to physical scale
    height_pixels, width_pixels = img_gray.shape
    x_scale = PHYSICAL_WIDTH / width_pixels
    y_scale = PHYSICAL_HEIGHT / height_pixels
    print(f"Scale: {x_scale:.6f} x {y_scale:.6f} units/pixel")
    print()
    
    # Detect edges
    if EDGE_METHOD == 'canny':
        edges = detect_shocks_canny(img_gray, sigma=SIGMA, 
                                   low_threshold=THRESHOLD*0.5,
                                   high_threshold=THRESHOLD)
    else:
        edges = detect_shocks_sobel(img_gray, threshold=THRESHOLD, sigma=SIGMA)
    
    # Extract coordinates
    x_points, y_points = extract_shock_coordinates(edges, x_scale, y_scale, flip_y=FLIP_Y_AXIS)
    
    # Cluster shocks
    labels = cluster_shocks(x_points, y_points, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    
    # Calculate angles
    print()
    shock_data = calculate_shock_angles(x_points, y_points, labels, 
                                       y_threshold=Y_THRESHOLD, 
                                       method=FITTING_METHOD,
                                       min_points=MIN_POINTS_PER_SHOCK,
                                       min_extent=MIN_SHOCK_EXTENT)
    
    # Print summary
    print()
    print("="*70)
    print("SUMMARY OF SHOCK ANGLES (Left to Right)")
    print("="*70)
    for i, shock in enumerate(shock_data):
        print(f"Shock {i+1}: β = {shock['angle']:.2f}°")
    print("="*70)
    print()
    
    # Plot results
    plot_shock_results(img_gray, shock_data, x_scale, y_scale, 
                      y_threshold=Y_THRESHOLD, flip_y=FLIP_Y_AXIS)
    
    # Export results to CSV (optional)
    import pandas as pd
    df = pd.DataFrame([{
        'Shock': i+1,
        'Angle_deg': shock['angle'],
        'Slope': shock['slope'],
        'X_position': shock['x_mean'],
        'N_points': shock['n_points']
    } for i, shock in enumerate(shock_data)])
    
    df.to_csv('shock_angles_schlieren.csv', index=False)
    print("Results saved to: shock_angles_schlieren.csv")



#%%

#------------------------------------------------------------------------------------------------------------------------------------#
#                        Automatically detecting shock waves and their angles from Schlieren imaging
#                                          MODIFIED VERSION WITH FILTERING
#------------------------------------------------------------------------------------------------------------------------------------#
"""
MODIFICATIONS:
1. Shock angles now reported as positive (absolute value)
2. Added spatial filtering to remove unwanted features
3. Increased default quality filters
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, feature
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           IMAGE LOADING AND PREPROCESSING
#------------------------------------------------------------------------------------------------------------------------------------#

def load_schlieren_image(image_path):
    """Load and convert Schlieren image to grayscale."""
    img = io.imread(image_path)
    
    if len(img.shape) == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img
    
    img_gray = img_gray.astype(float)
    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())
    
    print(f"Image loaded: {img_gray.shape[1]} x {img_gray.shape[0]} pixels")
    return img_gray


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           SHOCK DETECTION
#------------------------------------------------------------------------------------------------------------------------------------#

def detect_shocks_canny(img_gray, sigma=1.5, low_threshold=0.1, high_threshold=0.3):
    """Detect shock waves using Canny edge detection."""
    edges = feature.canny(
        img_gray,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    
    print(f"Canny detected {np.sum(edges)} edge pixels")
    return edges


def detect_shocks_sobel(img_gray, threshold=0.2, sigma=1.5):
    """Detect shock waves using Sobel edge detection."""
    img_smooth = ndimage.gaussian_filter(img_gray, sigma=sigma)
    
    gradient_x = ndimage.sobel(img_smooth, axis=1)
    gradient_y = ndimage.sobel(img_smooth, axis=0)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    edges = gradient_magnitude > threshold
    
    print(f"Sobel detected {np.sum(edges)} edge pixels")
    return edges


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           COORDINATE EXTRACTION AND CLUSTERING
#------------------------------------------------------------------------------------------------------------------------------------#

def extract_shock_coordinates(edges, x_scale, y_scale, flip_y=False):
    """Convert edge pixels to physical coordinates."""
    y_pixels, x_pixels = np.where(edges)
    
    # Convert to physical coordinates
    x_physical = x_pixels * x_scale
    
    if flip_y:
        # Flip y-axis to match CFD convention (y increases upward)
        image_height = edges.shape[0]
        y_physical = (image_height - y_pixels) * y_scale
    else:
        # Keep image convention (y increases downward from top)
        y_physical = y_pixels * y_scale
    
    print(f"Extracted {len(x_physical)} shock points")
    return x_physical, y_physical


def cluster_shocks(x_points, y_points, eps=0.5, min_samples=20):
    """Cluster shock points using DBSCAN."""
    shock_points = np.column_stack((x_points, y_points))
    
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(shock_points)
    
    n_shocks = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"DBSCAN found {n_shocks} distinct shocks")
    print(f"Noise points: {n_noise}")
    
    return labels


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           SHOCK ANGLE CALCULATION
#------------------------------------------------------------------------------------------------------------------------------------#

def calculate_shock_angles(x_points, y_points, labels, y_threshold=None, method='ransac', 
                          min_points=10, min_extent=0.5):
    """Calculate shock angles for each detected shock."""
    
    shock_data = []
    
    for shock_id in set(labels):
        if shock_id == -1:  # Skip noise
            continue
        
        mask = (labels == shock_id)
        x_shock = x_points[mask]
        y_shock = y_points[mask]
        
        # Filter out wall points if threshold provided
        if y_threshold is not None:
            wall_filter = y_shock > y_threshold
            x_shock_filtered = x_shock[wall_filter]
            y_shock_filtered = y_shock[wall_filter]
        else:
            x_shock_filtered = x_shock
            y_shock_filtered = y_shock
        
        # Skip if not enough points after filtering
        if len(x_shock_filtered) < min_points:
            print(f"Shock {shock_id}: skipped (only {len(x_shock_filtered)} points, need {min_points})")
            continue
        
        # Calculate spatial extent of shock
        x_extent = x_shock_filtered.max() - x_shock_filtered.min()
        y_extent = y_shock_filtered.max() - y_shock_filtered.min()
        total_extent = np.sqrt(x_extent**2 + y_extent**2)
        
        # Skip if shock is too small (likely noise)
        if total_extent < min_extent:
            print(f"Shock {shock_id}: skipped (extent {total_extent:.3f} < {min_extent})")
            continue
        
        # Fit line to calculate angle
        if method == 'ransac':
            X = x_shock_filtered.reshape(-1, 1)
            y = y_shock_filtered
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X, y)
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
        else:  # polyfit
            coeffs = np.polyfit(x_shock_filtered, y_shock_filtered, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
        
        # ===== MODIFICATION: Calculate angle as positive value ===== #
        beta_rad = np.arctan(slope)
        beta_deg = np.degrees(beta_rad)
        
        # Convention: report shock angle as positive (angle from horizontal)
        # This is standard in supersonic flow analysis
        beta_deg = abs(beta_deg)
        # =========================================================== #
        
        # Calculate mean position for sorting
        x_mean = np.mean(x_shock_filtered)
        
        shock_data.append({
            'original_id': shock_id,
            'x_mean': x_mean,
            'x_all': x_shock,
            'y_all': y_shock,
            'x_filtered': x_shock_filtered,
            'y_filtered': y_shock_filtered,
            'angle': beta_deg,
            'slope': slope,
            'intercept': intercept,
            'n_points': len(x_shock_filtered),
            'n_total': len(x_shock)
        })
    
    # Sort shocks left to right
    shock_data.sort(key=lambda s: s['x_mean'])
    
    return shock_data


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           NEW: SHOCK FILTERING FUNCTIONS
#------------------------------------------------------------------------------------------------------------------------------------#

def filter_shocks_by_region(shock_data, x_min=None, x_max=None, y_min=None, y_max=None):
    """
    Filter shocks based on spatial region.
    
    This is useful to exclude unwanted features in specific areas of the image.
    For example, excluding expansion waves that occur downstream.
    
    Parameters:
    -----------
    shock_data : list
        List of shock dictionaries from calculate_shock_angles()
    x_min, x_max : float, optional
        Minimum and maximum x-coordinates to include
    y_min, y_max : float, optional
        Minimum and maximum y-coordinates to include
    
    Returns:
    --------
    filtered_shocks : list
        Filtered list of shocks within the specified region
    """
    filtered_shocks = []
    
    for shock in shock_data:
        x_mean = shock['x_mean']
        
        # Apply spatial filters
        if x_min is not None and x_mean < x_min:
            print(f"Filtered out shock at x={x_mean:.3f} (below x_min={x_min})")
            continue
        if x_max is not None and x_mean > x_max:
            print(f"Filtered out shock at x={x_mean:.3f} (above x_max={x_max})")
            continue
        
        # Could add y filters here if needed
        # if y_min is not None and shock['y_mean'] < y_min:
        #     continue
        
        filtered_shocks.append(shock)
    
    return filtered_shocks


def filter_shocks_by_angle(shock_data, min_abs_angle=None, max_abs_angle=None):
    """
    Filter shocks based on their angle.
    
    This helps distinguish between strong compression shocks (steep angles)
    and weak expansion waves (shallow angles).
    
    Parameters:
    -----------
    shock_data : list
        List of shock dictionaries
    min_abs_angle : float, optional
        Minimum absolute angle in degrees (e.g., 30 to exclude shallow features)
    max_abs_angle : float, optional
        Maximum absolute angle in degrees
    
    Returns:
    --------
    filtered_shocks : list
        Shocks within the specified angle range
    """
    filtered_shocks = []
    
    for shock in shock_data:
        angle = abs(shock['angle'])
        
        if min_abs_angle is not None and angle < min_abs_angle:
            print(f"Filtered out shock with angle={shock['angle']:.1f}° (below min={min_abs_angle}°)")
            continue
        if max_abs_angle is not None and angle > max_abs_angle:
            print(f"Filtered out shock with angle={shock['angle']:.1f}° (above max={max_abs_angle}°)")
            continue
        
        filtered_shocks.append(shock)
    
    return filtered_shocks


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           VISUALIZATION
#------------------------------------------------------------------------------------------------------------------------------------#

def plot_shock_results(img_gray, shock_data, x_scale, y_scale, y_threshold=None, flip_y=False):
    """Plot shock detection and angle measurement results."""
    
    height, width = img_gray.shape
    
    plt.figure(figsize=(14, 8))
    
    # Show image with shock overlay
    # Always show image in its natural orientation
    extent = [0, width * x_scale, height * y_scale, 0]  # [left, right, bottom, top]
    plt.imshow(img_gray, cmap='gray', extent=extent, aspect='auto', alpha=0.5)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(shock_data)))
    
    for shock_num, shock in enumerate(shock_data):
        x_shock_i = shock['x_all']
        y_shock_i = shock['y_all']
        x_shock_filtered = shock['x_filtered']
        y_shock_filtered = shock['y_filtered']
        slope = shock['slope']
        intercept = shock['intercept']
        beta_deg = shock['angle']
        
        shock_label = shock_num + 1
        
        print(f"Shock {shock_label}:")
        print(f"  Angle β = {beta_deg:.2f}°")
        print(f"  Slope = {slope:.4f}")
        print(f"  Mean x-position = {shock['x_mean']:.3f}")
        print(f"  Points used for fitting = {len(x_shock_filtered)} (out of {len(x_shock_i)} total)")
        print()
        
        # Plot all detected shock points (faded)
        plt.scatter(x_shock_i, y_shock_i, s=20, color=colors[shock_num], 
                   alpha=0.3, label=f'Shock {shock_label} (all points)')
        
        # Highlight points used for fitting
        plt.scatter(x_shock_filtered, y_shock_filtered, s=30, 
                   color=colors[shock_num], edgecolors='black', linewidths=1,
                   label=f'Shock {shock_label} fit points (β={beta_deg:.1f}°)')
        
        # Plot fitted line
        x_line = np.array([x_shock_i.min(), x_shock_i.max()])
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, '--', color=colors[shock_num], linewidth=2.5)
    
    # Draw wall filter line if used
    if y_threshold is not None:
        plt.axhline(y=y_threshold, color='red', linestyle=':', 
                   linewidth=1, alpha=0.7, label=f'Wall filter (y={y_threshold})')
    
    plt.xlabel(r"$X$ [inches]", fontsize=14)
    plt.ylabel(r"$Y$ [inches]", fontsize=14)
    plt.title("Shock Angle Measurement from Schlieren Image", fontsize=24)
    plt.legend(loc='best', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------------------------------------------------#
#                                           MAIN ANALYSIS
#------------------------------------------------------------------------------------------------------------------------------------#

# ===== USER PARAMETERS - MODIFY THESE ===== #

# Image file path
IMAGE_PATH = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\18_Schilierean Imaging\schilieren.png"



# Physical dimensions of imaged area
PHYSICAL_WIDTH = 5.0   # inches (or your preferred units)
PHYSICAL_HEIGHT = 5.0   # inches



# Edge detection parameters
EDGE_METHOD = 'canny'   # 'canny' or 'sobel'
SIGMA = 1.5             # Gaussian smoothing
THRESHOLD = 0.2         # Edge threshold (only for sobel)



# Clustering parameters
DBSCAN_EPS = 0.3        # Maximum distance between points in same shock
DBSCAN_MIN_SAMPLES = 20 # Minimum points to form a shock



# Wall filtering
Y_THRESHOLD = None      # Set to exclude wall region (e.g., 0.5), or None for no filtering



# Fitting method
FITTING_METHOD = 'ransac'  # 'ransac' (robust) or 'polyfit' (fast)



# ===== MODIFIED: STRONGER QUALITY FILTERS ===== #
MIN_POINTS_PER_SHOCK = 200   # INCREASED from 50 - rejects small disconnected features
MIN_SHOCK_EXTENT = 2.0       # INCREASED from 1.0 - rejects short features



# ===== NEW: SPATIAL FILTERING ===== #
# Only consider shocks in this x-range (set to None to disable)
X_MIN_FILTER = None     # Minimum x-position (e.g., 0.5)
X_MAX_FILTER = 2.5      # Maximum x-position (e.g., 2.5 to exclude expansion region)
                         # Set to None to disable



# ===== NEW: ANGLE FILTERING ===== #
# Only consider shocks with angles in this range (set to None to disable)
MIN_ANGLE_FILTER = 30    # Minimum angle in degrees (e.g., 30 to exclude shallow waves)
MAX_ANGLE_FILTER = None  # Maximum angle in degrees
                         # Set to None to disable



# Image orientation
FLIP_Y_AXIS = False         # Set to True if you want y to increase upward (CFD style)



# ===== RUN ANALYSIS ===== #

if __name__ == "__main__":
    
    print("="*70)
    print("SCHLIEREN SHOCK ANGLE ANALYSIS (MODIFIED VERSION)")
    print("="*70)
    print()
    
    # Load image
    img_gray = load_schlieren_image(IMAGE_PATH)
    
    # Calculate pixel to physical scale
    height_pixels, width_pixels = img_gray.shape
    x_scale = PHYSICAL_WIDTH / width_pixels
    y_scale = PHYSICAL_HEIGHT / height_pixels
    print(f"Scale: {x_scale:.6f} x {y_scale:.6f} units/pixel")
    print()
    
    # Detect edges
    if EDGE_METHOD == 'canny':
        edges = detect_shocks_canny(img_gray, sigma=SIGMA, 
                                   low_threshold=THRESHOLD*0.5,
                                   high_threshold=THRESHOLD)
    else:
        edges = detect_shocks_sobel(img_gray, threshold=THRESHOLD, sigma=SIGMA)
    
    # Extract coordinates
    x_points, y_points = extract_shock_coordinates(edges, x_scale, y_scale, flip_y=FLIP_Y_AXIS)
    
    # Cluster shocks
    labels = cluster_shocks(x_points, y_points, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    
    # Calculate angles
    print()
    shock_data = calculate_shock_angles(x_points, y_points, labels, 
                                       y_threshold=Y_THRESHOLD, 
                                       method=FITTING_METHOD,
                                       min_points=MIN_POINTS_PER_SHOCK,
                                       min_extent=MIN_SHOCK_EXTENT)
    
    # ===== NEW: APPLY ADDITIONAL FILTERS ===== #
    print()
    print("Applying additional filters...")
    
    # Filter by spatial region
    if X_MIN_FILTER is not None or X_MAX_FILTER is not None:
        shock_data = filter_shocks_by_region(shock_data, 
                                            x_min=X_MIN_FILTER, 
                                            x_max=X_MAX_FILTER)
    
    # Filter by angle
    if MIN_ANGLE_FILTER is not None or MAX_ANGLE_FILTER is not None:
        shock_data = filter_shocks_by_angle(shock_data,
                                           min_abs_angle=MIN_ANGLE_FILTER,
                                           max_abs_angle=MAX_ANGLE_FILTER)
    
    print(f"Final number of shocks after filtering: {len(shock_data)}")
    print()
    
    # Print summary
    print()
    print("="*70)
    print("SUMMARY OF SHOCK ANGLES (Left to Right)")
    print("="*70)
    for i, shock in enumerate(shock_data):
        print(f"Shock {i+1}: β = {shock['angle']:.2f}°")
    print("="*70)
    print()
    
    # Plot results
    plot_shock_results(img_gray, shock_data, x_scale, y_scale, 
                      y_threshold=Y_THRESHOLD, flip_y=FLIP_Y_AXIS)
    
    # Export results to CSV (optional)
    import pandas as pd
    df = pd.DataFrame([{
        'Shock': i+1,
        'Angle_deg': shock['angle'],
        'Slope': shock['slope'],
        'X_position': shock['x_mean'],
        'N_points': shock['n_points']
    } for i, shock in enumerate(shock_data)])
    
    df.to_csv('shock_angles_schlieren.csv', index=False)
    print("Results saved to: shock_angles_schlieren.csv")
    
    
    #%%  Quickly getting the pressure gradient 

# Functions #
def get_hl(key: str):
    # Handles negative numbers, decimals, scientific notation
    m = re.search(r'h[_-]?l[_-]?(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', key, flags=re.I)
    return float(m.group(1)) if m else None



####  Plotting pressure gradient of h/l = 0.03 at M = 1.5  ####

# Defining key # 
temp_key = "h_l_0.05_Mach_4.0"

# Defining x #
x = ds_by_case[temp_key]["X"].data

# Masking # 
l = 0.1
x_min = 0 
x_max = l
mask = (x_min < x) & (x < x_max)

# Applying the mask # 
x = x[mask] 
tau_x_var = tau_x[temp_key]


try:
    dp_dx = ds_by_case[temp_key]["P_x"].data[mask]
except:
    P = ds_by_case[temp_key]["P"].data[mask]
    dp_dx = np.gradient(P,x,edge_order = 2)



# Defining cosmetics for uniformity #  
fsize_label = 18
fsize_title = 28
lWidth = 3
tickSize = 14
# Defining the colors of each axis # 
color1 = 'tab:red'
color2 = 'tab:blue'


# Creating the plot #
fig , ax1 = plt.subplots()


#### Plot 1: dp/dx Vs X ####
ax1.plot(x,dp_dx, color = color1, label = "dp/dx",lw = lWidth)
ax1.set_xlabel("X [m]" , fontsize = fsize_label)
ax1.set_ylabel("dp/dx [Pa/m]", fontsize = fsize_label, color = color1)


# Chaning tick parameters 
ax1.tick_params(axis = 'y', labelcolor = color1, labelsize = tickSize)
ax1.tick_params(axis = 'x', labelsize = tickSize)



#### Plot 2: taux vs X ####
ax2 = ax1.twinx()
ax2.plot(x,tau_x_var, color = color2, label = r"\$tau_{x}$" , lw = lWidth)
ax2.set_xlabel("X [m]", fontsize = fsize_label)
ax2.set_ylabel(r"$\tau_{x}$ [Pa]", fontsize = fsize_label, color = color2)\


# Changing tick parameters #     
ax2.tick_params(axis = 'y', labelcolor = color2, labelsize = tickSize)

# Getting the h/l # 
temp_hl = get_hl(temp_key)
MACH_LEVELS = np.arange(1.5, 4.5 , 0.5)
M_temp = extract_mach_from_filename(temp_key, MACH_LEVELS)


# Title and grid # 
plt.title(fr"$\frac{{dp}}{{dx}}$ and $\tau_x$ vs X, h/l = {temp_hl}, M = {M_temp} ", fontsize=fsize_title)
ax1.grid(True, alpha=0.3)



plt.grid()
plt.show()


#%% Plotting the difference between the pressure at the wall between optimized geometry and ideal case #


# Plotting Pressure at the wall for h/l 0.05 # 
keys = cases_by_hl["h_l_0.05"]
keys_optm = cases_by_hl["h_l_x"]

# Defining the x values of the geometry #
x = ds_by_case["h_l_0.05_Mach_1.5"]["X"].data
x_optm = ds_by_case["h_l_x_Mach_1.5"]["X"].data

# Defining values for clarity # 
line_width_coeff = 5
xfontSize = 21
yfontSize = 21
titlefontSize = 28
tickfontSize = 16

for idx, key in enumerate(keys):
    Px_wall = ds_by_case[key]["P_x"].data
    Px_wall_optm = ds_by_case[keys_optm[idx]]["P_x"].data
    
    plt.plot(x, Px_wall,label = "dP/dx Nominal",linewidth = line_width_coeff)
    plt.plot(x_optm,Px_wall_optm, label = "dP/dx Optimial", linewidth = line_width_coeff)
    plt.xlim([0 , 0.1])
    plt.grid()
    
    plt.xlabel("X[m]", fontsize =  xfontSize)
    plt.ylabel("$dP/dx$ [N/m^2]", fontsize = yfontSize)
    plt.title(f"Pressure Vs X: {key}", fontsize = titlefontSize)
    plt.tick_params( labelsize = tickfontSize)
    plt.legend()
    
    plt.show()
    
for idx, key in enumerate(keys):
    P_wall = ds_by_case[key]["P"].data
    P_wall_optm = ds_by_case[keys_optm[idx]]["P"].data
    
    plt.plot(x, P_wall,label = "Pressure Nominal",linewidth = line_width_coeff)
    plt.plot(x_optm,P_wall_optm, label = "Pressure Optimial", linewidth = line_width_coeff)
    plt.xlim([0 , 0.1])
    plt.grid()
    
    plt.xlabel("X[m]", fontsize =  xfontSize)
    plt.ylabel("$P_{wall}$ [Pa]", fontsize = yfontSize)
    plt.title(f"Pressure Vs X: {key}", fontsize = titlefontSize)
    plt.tick_params( labelsize = tickfontSize)
    plt.legend()
    
    plt.show()


#%% 
import matplotlib.pyplot as plt
import numpy as np

# Exporting results for Hannah # 
temp_key = "h_l_0.02_Mach_1.5"
x_opt = ds_by_case[temp_key]["X"].data
y_opt = ds_by_case[temp_key]["Y"].data
mask = (0 <= x_opt) & (0.1 >= x_opt)
x_new = x_opt[mask]
y_new = y_opt[mask]
z_new = np.zeros_like(x_new)  # Fixed: use zeros_like to match array shape


plt.plot(x_new, y_new)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('h/l = 0.02')
plt.axis('equal')
plt.grid(True)
plt.show()
#%%
# Writing results #
filename = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\34_Hannah Proejct\h_l_0_02_geom_coords.txt" 

# Method 1: Using a loop (explicit, easy to understand)
with open(filename, 'w') as file:
    # Note: SolidWorks doesn't need a header - just X Y Z values
    # If you want a header for reference, uncomment the next line:
    # file.write("X Y Z\n")
    
    for i in range(len(x_new)):
        file.write(f"{x_new[i]:.10f} {y_new[i]:.10f} {z_new[i]:.10f}\n")

print(f"Exported {len(x_new)} points to {filename}")

# Method 2 (Alternative): Using numpy.savetxt - more compact
# Uncomment below if you prefer this approach:
# 
# data = np.column_stack((x_new, y_new, z_new))
# np.savetxt(filename, data, fmt='%.10f', delimiter=' ')



   