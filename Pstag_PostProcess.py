# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 09:37:46 2025

@author: hhsabbah
"""
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
def dict_to_ds_1d(data):
    """{var: np.array} -> xarray.Dataset with dim 'n'."""
    return xr.Dataset({k: (("n",), np.asarray(v)) for k, v in data.items()})


"""
#-----------------------------------------------------------------------------------------------------------------------------#
    Getting quantitative reuslts from the data that was extracted, e.g. Reynolds number, Wall shear stress, etc etc. 
#-----------------------------------------------------------------------------------------------------------------------------#

"""


### Post-Processing for the Stagnation Pressure sweep study ###
tp.session.connect()
base_dirStag = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\7_Parametric Study\1_DeltaPstag Simulations")

# Finding all the h_l_names from the folders # 
h_l_names = []
for subdir in base_dirStag.glob("*/*/"):
    if subdir.is_dir():
        h_l_names.append(subdir.name)


# Root directory to import mcfd_tec.bin files # 
rootDir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\7_Parametric Study\1_DeltaPstag Simulations") # this is the root directory to the parametric study solution files
subDirs1 = [p for p in rootDir.iterdir() if p.is_dir()]


# Finding all mcfd_tec.bin files and getting variables # 
fileName = "mcfd_tec.bin"
subDirs2 = [p for d in subDirs1 for p in d.iterdir() if p.is_dir()]  # flattened
file_paths = [p / fileName for p in subDirs2]
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

#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
    Saving all dictionaries from the previous run. This saves time since you will not have to post-process the results every time...
#------------------------------------------------------------------------------------------------------------------------------------#

"""

# Saving the case for the future so I don't have to wait a year to load my data again # 
import pickle
import os
save_dir = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\07_Python Codes\01_Python Automation Codes\03_Post-Processing Code"

# Saving # 
with open(os.path.join(save_dir,"ds_by_case.pkl"), "wb") as f:
    pickle.dump(ds_by_case, f)
with open(os.path.join(save_dir,"ds_by_case_quad.pkl"), "wb") as f:
    pickle.dump(ds_by_case_quad, f)
with open(os.path.join(save_dir,"ds_by_case_inlet.pkl"), "wb") as f:
    pickle.dump(ds_by_case_inlet, f)
with open(os.path.join(save_dir,"index_by_case.pkl"), "wb") as f:
    pickle.dump(index_by_case, f)

#%% 

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                            Here you can load your data from previously saved run
#------------------------------------------------------------------------------------------------------------------------------------#

"""
with open("ds_by_case.pkl", "rb") as f:
    ds_by_case = pickle.load(f)
with open("ds_by_case_quad.pkl", "rb") as f:
    ds_by_case_quad = pickle.load(f)
with open("ds_by_case_inlet.pkl", "rb") as f:
    ds_by_case_inlet = pickle.load(f)
with open("index_by_case.pkl", "rb") as f:
    index_by_case = pickle.load(f)

        
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
mm_to_in = 0.0393701
in_to_mm = 1 / 0.0393701
in_to_m = 0.0254

# Getting Reynolds Number # 
Re = {}
Re_wall = {} 


# Getting All Variables to compute Reynolds number # 


for key in ds_by_case:
    mu = (1 / ds_by_case_inlet[key]["Mut_ovr_Mu"].data) * ds_by_case_inlet[key]["Mutur"].data
    U = ds_by_case_inlet[key]["U"].data
    rho = ds_by_case_inlet[key]["R"].data
    X = ds_by_case[key]["X"].data[-1]
    Re[key] = (np.mean(rho) * (np.mean(U) ) * np.mean(X) * in_to_m ) / np.mean(mu) # this is just a test. Re wall is not a thing. Well, it is, but you use boundary layer thickness to compute that stuff not the length of the entire thing...
   
    
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
                                                        Computing Wall shear Stress(tau_y)
#------------------------------------------------------------------------------------------------------------------------------------#

"""

# Finding the wall shear stress along the wall # 
tau_x = {}
tau_y = {}
X_geom = {}
tau_separation = {}
tau_separation_idx = {}

for key in ds_by_case:
    tau_x[key] = ds_by_case[key]["Tau_x"].data
    tau_y[key] = ds_by_case[key]["Tau_y"].data
    X_geom[key] = ds_by_case[key]["X"].data * in_to_mm
    
    
    #### COMPUTING THE FIRST POINTS AT WHICH A TAU_Y GOES BELOW ZERO ######
    first_index_tau = np.argmax(tau_y[key] < 0)
    tau_separation[key] = tau_y[key][first_index_tau] #finds the first point at which separation occurs for each h_l case and pressure
    tau_separation_idx[key] = first_index_tau
        


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
    # Extract h_l part from key, e.g., "h_l_0.01" from "h_l_0.01_p0_1bar"
    hl_match = re.match(r"(h_l_\d+\.\d+)", key)
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
        plt.plot(X_geom[key], tau_y[key], label=key , color = cmap(i),linewidth = 2)
    

    plt.title(fr"$\tau_y$ vs X for {hl}",fontsize = 21)
    plt.axhline(y=0, color='r', linestyle='--', label='Separation')
    plt.xlabel("X [mm]" , fontsize = 16)
    plt.ylabel(r"$\tau_y$ [Pa]", fontsize = 16)
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
    plt.xlabel("X [mm]" , fontsize = 16)
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
            Plotting the residuals and also plotting the net mass flow. Will ahve to learn how to do that from CFD++
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
rootDir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\7_Parametric Study\1_DeltaPstag Simulations") # this is the root directory to the parametric study solution files
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
rootDir_flux = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\7_Parametric Study\1_DeltaPstag Simulations") # this is the root directory to the parametric study solution files
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
    plt.plot(df["iterations"],df["mass_flux"])
    
    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,1))
    
    plt.xlabel("Iterations")
    plt.ylabel("Net Mass Flux")
    plt.title(f"Net Mass Flux Vs Iterations: {file_path_flux.parent.name}")
    plt.grid(True,which = "both")
    plt.tick_params(axis='both', which='major', labelsize=18)

    # Creating a dictonary for the different mass flux # 
    mass_flux_end[file_path_flux.parent.name] = df["mass_flux"].iloc[-1]

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

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                        Getting all Boundary Layer profiles using Total enthalpy method 
#------------------------------------------------------------------------------------------------------------------------------------#
""" 
######## Getting all boundary layer profiles... ########
from tecplot.constant import PlotType
from itertools import tee
import numpy as np
import sympy as sp


### Functions ####
import numpy as np



def flatten_after_peak_robust(y, f, *,
                              smooth=9,       # odd, moving-average on TAIL
                              window=9,       # persistence window (points)
                              frac=0.01,      # relative slope threshold (1% of tail max)
                              abs_floor=None, # absolute slope floor; if None, auto from noise
                              early_tail=0.6  # only search first 60% of the tail
                              ):
    """
    Pick the first index AFTER THE GLOBAL PEAK where the curve has 'flattened'.

    Robust to:
      - float32 quantization (promotes to float64, smooths tail)
      - huge dynamic range (normalizes y, uses relative threshold)
      - duplicate/unsorted y (sorts & collapses duplicates)

    Returns an index into y,f sorted by y.
    """
    y = np.asarray(y).ravel().astype(np.float64)
    f = np.asarray(f).ravel().astype(np.float64)

    # 1) sort by y and collapse duplicate y's (median of f)
    order = np.argsort(y)
    y, f = y[order], f[order]
    # unique y collapse
    uy, idx_start = np.unique(y, return_index=True)
    if len(uy) != len(y):
        # group by unique y and take median of f in each group
        # (vectorized-ish)
        groups = np.split(f, np.r_[idx_start[1:], len(f)])
        f = np.array([np.median(g) for g in groups])
        y = uy

    n = len(y)
    if n < max(5, window+2):
        return 0

    # 2) normalize y for stable gradient scale
    y0, y1 = y[0], y[-1]
    if y1 == y0:
        return 0
    yN = (y - y0) / (y1 - y0)

    # 3) find global peak on original f
    i_peak = int(np.argmax(f))
    if i_peak >= n - 2:
        return i_peak

    # 4) tail arrays after the peak
    y_t  = yN[i_peak:]
    f_t  = f[i_peak:]
    nt   = len(y_t)

    # 5) moving-average smoothing on the tail only (to beat ±32 quantization)
    if smooth and smooth > 1 and len(f_t) >= 3:
        k = int(smooth) | 1
        ker = np.ones(k)/k
        f_ts = np.convolve(f_t, ker, mode="same")
    else:
        f_ts = f_t

    # --- NEW: sanity checks before taking the gradient ---
    # Need at least 3 points for a stable gradient (edge_order=1 needs 2, but 3 is safer)
    if len(f_ts) < 3:
        return i_peak  # tail too short to judge "flatness" robustly

    # Ensure the coordinate array matches the data array length
    if f_ts.shape != y_t.shape:
        m = min(f_ts.size, y_t.size)
        f_ts = f_ts[:m]
        y_t  = y_t[:m]


    # 6) gradient on the normalized coordinate
    g = np.abs(np.gradient(f_ts, y_t))

    # 7) thresholds: relative + absolute floor
    # relative based on tail (ignoring first few points after peak to skip the steepest drop)
    head_cut = max(2, nt//20)  # ignore first ~5% of tail for thresholding
    g_tail_max = np.nanmax(g[head_cut:]) if head_cut < nt else np.nanmax(g)
    rel_eps = frac * max(1e-15, g_tail_max)

    # absolute floor: detect quantization step if not provided
    if abs_floor is None:
        # Typical floor ~ few LSBs of tail slopes: use robust estimate from small values
        small = np.sort(g)[max(0, nt - max(50, nt//4)):]  # smallest quartile
        abs_eps = 5.0 * (np.median(small) if small.size else 0.0)
        abs_eps = max(abs_eps, 1e-12)
    else:
        abs_eps = abs_floor

    eps = max(rel_eps, abs_eps)  # must be below BOTH scales

    # 8) persistence mask and early window to avoid drifting too far
    m = int(max(1, min(window, nt)))
    early = int(max(1, min(int(early_tail * nt), nt)))

    flat = (g[:early] <= eps).astype(int)
    if m > 1:
        run = np.convolve(flat, np.ones(m, dtype=int), mode="valid")
        j = np.where(run == m)[0]
        if j.size:
            return i_peak + int(j[0])  # first persistent flat
    else:
        j0 = np.argmax(flat == 1)
        if flat[j0] == 1:
            return i_peak + int(j0)

    # 9) fallback: pick minimum |slope| in early tail
    j_fb = int(np.argmin(g[:early]))
    return i_peak + j_fb


# Constants # 
mm_to_m = 0.001


# Pre-Allocating Variables # 
delta_n_m = {k: [] for k in ds_by_case}  # pre-seed keys # Boundary layer thickness in meters
delta_n_mm = {k: [] for k in ds_by_case} # Boundary Layer thickness in mm
tau_w_dict = {k: [] for k in ds_by_case}

for idx,key in enumerate(ds_by_case):
    # --- Inputs from your dataset ---
    x0 = ds_by_case[key]["X"].data.astype(float).ravel()
    y0 = ds_by_case[key]["Y"].data.astype(float).ravel()
    
    
    # (Optional) ensure no NaNs in geometry itself
    good = (~np.isnan(x0)) & (~np.isnan(y0))
    x0 = x0[good]
    y0 = y0[good]
    
    
    
    
    
    # --- Parameters ---
    bl_h_mm = 10  # 10 mm -> meters if your X,Y are in meters. Keep units consistent!
    bl_h = bl_h_mm * 0.0393701 # units in inches now...
    z_const = 0.0        # 2D case: stick all lines in z=0 plane
    stride = 1           # use every point; increase to 2,3,... to thin the rake if needed
    
    
    
    # --- 1. Tangent and unit normal (stable; avoids divide-by-zero) ---
    # Central differences with same length as x0,y0
    dx_ds = np.gradient(x0)
    dy_ds = np.gradient(y0)
    
    # Normal vector (not yet unit): n = (-dy, +dx)
    nx = -dy_ds
    ny =  dx_ds
    
    # Normalize
    norm = np.hypot(nx, ny)
    eps = 1e-12
    norm = np.where(norm < eps, np.nan, norm)  # mark degenerate spots as NaN
    
    ux = nx / norm   # unit normal x
    uy = ny / norm   # unit normal y
    
    # --- 2. Build start/end points for each small line segment ---
    # Start at surface point Pi = (x0, y0); end at Pi + bl_h * n_hat
    xf = x0 + bl_h * ux
    yf = y0 + bl_h * uy
    
    # Drop any pairs where normal was undefined
    valid = (~np.isnan(xf)) & (~np.isnan(yf))
    x_start = x0[valid][::stride]
    y_start = y0[valid][::stride]
    x_end   = xf[valid][::stride]
    y_end   = yf[valid][::stride]
    
    
    
    
    # At this point you have N pairs: (x_start[i], y_start[i]) -> (x_end[i], y_end[i])
    # Interleave starts and ends: [y0[0], y_f[0], y0[1], y_f[1], ...]
    y_new = np.ravel(np.column_stack((y0, y_end)))
    x_new = np.ravel(np.column_stack((x0, x_end)))
    
    
    
    
    
    
    # Plotting X Vs Y # 
    plt.figure(figsize = (12,40))
    plt.plot(x_new,y_new, color = 'red',label = 'BL Extration')
    plt.plot(x0,y0, color = 'blue',label = 'Geometry')
    plt.scatter(x_start,y_start, color = 'green',label = 'Initial Points')
    plt.title(f"X Vs Y w/ Perpendicular lines: {key}")
    plt.xlabel("X[inches]")
    plt.ylabel("Y[inches]")
    plt.grid(True, which = 'both')
    
    # Force equal scaling between x and y axes
    plt.gca().set_aspect('equal')
    plt.show()
    
    
    # Creating points in zip for pytecplot extraction
    points_in = zip(x_new,y_new)
    
    
    # x_new, y_new are 1D arrays like: [x0_start, x0_end, x1_start, x1_end, ...]
    n = min(len(x_new), len(y_new))
    if n % 2:                   # drop a dangling last value if the length is odd
        n -= 1
    
    x_pairs = np.asarray(x_new[:n]).reshape(-1, 2)   # shape: (N, 2)
    y_pairs = np.asarray(y_new[:n]).reshape(-1, 2)   # shape: (N, 2)
    
    
    # If you need 3D points for Tecplot:
    z0 = 0.0
    it1 = [(x_pairs[i, 0], y_pairs[i, 0], z0) for i in range(x_pairs.shape[0])]
    it2 = [(x_pairs[i, 1], y_pairs[i, 1], z0) for i in range(x_pairs.shape[0])]
    
    
    
    
    # Defining the number of points I want in the extraction # 
    num_points = 300
    
    # Connecting to the Tecplot session # 
    tp.session.connect() # Connecting to the open tecplot application 
    tp.new_layout() # Creating a new tecplot layout.
    tp.data.load_tecplot(file_paths[idx].as_posix())
    
    
    
    
    # Obtain Reference values for reconstruction of the BL (to obtain BL thickness) # 
    P_ref = ds_by_case_inlet[key]["P"].data.astype(float).ravel()[0]
    rho_ref = ds_by_case_inlet[key]["R"].data.astype(float).ravel()[0]
    U_ref = ds_by_case_inlet[key]["U"].data.astype(float).ravel()[0]
    V_ref = ds_by_case_inlet[key]["V"].data.astype(float).ravel()[0]
    nu = ds_by_case_inlet[key]["Nu"].data.astype(float).ravel()[0]
    mu_ref = (1 / ds_by_case_inlet[key]["Mut_ovr_Mu"].data) * ds_by_case_inlet[key]["Mutur"].data
    
    

    
    
    for idx2, (curr, nxt) in enumerate(zip(it1, it2)):
    
    
        # Extracting data from the line
        line = tp.data.extract.extract_line([curr, nxt], num_points=num_points)
    
        # Read arrays (your unit conversions unchanged)
        x_BL = line.values("X").as_numpy_array() 
        y_BL = line.values("Y").as_numpy_array() 
        u_BL = line.values("U").as_numpy_array()
        
        P = line.values("P").as_numpy_array()
        P_total = line.values("P_total").as_numpy_array()
        T = line.values("T").as_numpy_array()
        rho = line.values("R").as_numpy_array()
        mu = (1 / line.values("Mut_ovr_Mu").as_numpy_array()) * line.values("Mutur").as_numpy_array()
        
        
        U = u_BL
        V = line.values("V").as_numpy_array()
        W = line.values("W").as_numpy_array()
        
        tau_y = line.values("Tau_y").as_numpy_array()
        tau_w = tau_y[0]
        tau_w_dict[key].append(tau_y[0])
        cf_CFD = (tau_w/(0.5*rho_ref*U_ref**2))
        
        
        # Getting the local reconstruction of the boundary layer using total enthalpy equation # 
        gamma = line.values("Gamma").as_numpy_array() 
        h0 = ( (gamma)/(gamma-1) ) * (P/rho) + 0.5*(U**2 + V**2) # calculating the total enthalpy
        U_I = np.sqrt( ( (2*gamma) / (gamma-1) ) * ( (P_ref/rho_ref) - (P / rho) ) + U_ref**2 + V_ref**2 - V**2) 
        
        # Threshold to when the boundary layer thickness edge has been achived # 
        dh0dy = np.gradient(h0,y_BL)
        # after you compute h0 and y_BL
        # after computing h0 and y_BL
        
        # after computing h0 and y_BL
        # after computing h0 and y_BL
        y_index = flatten_after_peak_robust(
            y_BL, h0,
            smooth =  9,   # try 7–11 if tail is noisy/quantized
            window= 9,        # same as smoothing is a good start
            frac=0.00001,           # 1% of (tail) max slope
            abs_floor=None,      # let it auto-detect; or set e.g. 50.0 for your ±32 case
            early_tail=0.9       # only search first 60% of the post-peak tail
        )


        
        
        # Computing the edge velocity 
        n = 99 # can change...
        U_e = U_I[y_index] * (n / 100)
        U_e = float(U_e)
        
       
        # --- ensure samples run from wall→outer ---
        if y_BL[0] > y_BL[-1]:
            x_BL = x_BL[::-1]; y_BL = y_BL[::-1]; u_BL = u_BL[::-1]
    
        # Datum: start at 0 at the wall
        y_BL_wall = y_BL[0]
        y_BL_corr = y_BL - y_BL_wall
        
        # For plotting purposes, converting x from inches to mm for a better reference ... #
        x_BL_mm = curr[0] * in_to_mm
        


        
        
        # Computing the boundary layer thickness based on the corrected y_bl datum #
        delta_n_m[key].append(float(y_BL_corr[y_index]) * in_to_mm * mm_to_m) # mm
        delta_n_mm[key].append(float(y_BL_corr[y_index]) * in_to_mm)
        delta_n_now = delta_n_mm[key][idx2] 
        
        
    
        print(f"--------------------------------------\n"
              f" Iter{idx2}:\n Current: {curr}\n Next: {nxt}\n"
              f" U_e = {U_e:.2f} m/s \n"
              f" BL Thickness = {delta_n_now:.2f} mm\n"
              f"--------------------------------------")
        
#%% GPT Test BL thickness extraction with quicker code...


import numpy as np
import tecplot as tp
from tecplot.constant import PlotType
import time 


start = time.time()
# --- Tunables ---
DEBUG_PLOTS = False       # turn on only when debugging
stride = 2                # 1 = every surface point; try 2–4+ to speed up
num_points = 200          # was 300; lower if resolution still OK
use_unique_collapse = False  # skip expensive unique/median in flatten()

in_to_mm = 25.4
mm_to_m  = 1e-3

def flatten_after_peak_fast(y, f, smooth=9, window=9, frac=1e-5, abs_floor=None, early_tail=0.9, do_unique=False):
    y = np.asarray(y, np.float64).ravel()
    f = np.asarray(f, np.float64).ravel()

    # Optional: unique collapse (expensive). Keep OFF unless you truly need it.
    if do_unique:
        order = np.argsort(y)
        y, f = y[order], f[order]
        uy, idx_start = np.unique(y, return_index=True)
        if len(uy) != len(y):
            # vectorized-ish median collapse
            splits = np.r_[idx_start[1:], len(f)]
            groups = np.split(f, splits)
            f = np.array([np.median(g) for g in groups], dtype=np.float64)
            y = uy

    n = len(y)
    if n < max(5, window+2):
        return 0

    # normalize coordinate for stable gradient
    y0, y1 = y[0], y[-1]
    if y1 == y0:
        return 0
    yN = (y - y0) / (y1 - y0)

    i_peak = int(np.argmax(f))
    if i_peak >= n - 2:
        return i_peak

    y_t = yN[i_peak:]
    f_t = f[i_peak:]
    nt  = len(y_t)

    if smooth and smooth > 1 and nt >= 3:
        k = int(smooth) | 1
        ker = np.ones(k)/k
        f_ts = np.convolve(f_t, ker, mode="same")
    else:
        f_ts = f_t

    # length guard
    m = min(f_ts.size, y_t.size)
    f_ts = f_ts[:m]
    y_t  = y_t[:m]

    g = np.abs(np.gradient(f_ts, y_t))

    head_cut   = max(2, nt//20)
    g_tail_max = np.nanmax(g[head_cut:]) if head_cut < nt else np.nanmax(g)
    rel_eps    = frac * max(1e-15, g_tail_max)

    if abs_floor is None:
        small   = np.sort(g)[max(0, nt - max(50, nt//4)):]  # smallest quartile
        abs_eps = 5.0 * (np.median(small) if small.size else 0.0)
        abs_eps = max(abs_eps, 1e-12)
    else:
        abs_eps = abs_floor

    eps   = max(rel_eps, abs_eps)
    mwin  = int(max(1, min(window, nt)))
    early = int(max(1, min(int(early_tail * nt), nt)))

    flat = (g[:early] <= eps).astype(np.uint8)
    if mwin > 1:
        run = np.convolve(flat, np.ones(mwin, dtype=np.uint8), mode="valid")
        j = np.where(run == mwin)[0]
        if j.size:
            return i_peak + int(j[0])
    else:
        j0 = np.argmax(flat == 1)
        if flat[j0] == 1:
            return i_peak + int(j0)

    j_fb = int(np.argmin(g[:early]))
    return i_peak + j_fb

# -------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------
# Connect ONCE per session
try:
    tp.session.connect()
except Exception:
    pass  # OK if already connected or running in batch

for idx, key in enumerate(ds_by_case):
    # Load layout/data ONCE per case
    tp.new_layout()
    tp.data.load_tecplot(file_paths[idx].as_posix())
    fr = tp.active_frame()
    fr.plot_type = PlotType.Cartesian2D

    # --- pull geometry once ---
    x0 = ds_by_case[key]["X"].data.astype(float).ravel()
    y0 = ds_by_case[key]["Y"].data.astype(float).ravel()
    good = np.isfinite(x0) & np.isfinite(y0)
    x0, y0 = x0[good], y0[good]

    # --- references ONCE per case ---
    P_ref   = float(ds_by_case_inlet[key]["P"].data.ravel()[0])
    rho_ref = float(ds_by_case_inlet[key]["R"].data.ravel()[0])
    U_ref   = float(ds_by_case_inlet[key]["U"].data.ravel()[0])
    V_ref   = float(ds_by_case_inlet[key]["V"].data.ravel()[0])

    # --- surface normals (vectorized) ---
    dx_ds = np.gradient(x0)
    dy_ds = np.gradient(y0)
    nx, ny = -dy_ds, dx_ds
    norm = np.hypot(nx, ny)
    norm[norm < 1e-12] = np.nan
    ux, uy = nx / norm, ny / norm

    # --- rake endpoints (thinned) ---
    bl_h_mm = 10.0
    bl_h_in = bl_h_mm * 0.0393701
    xf = x0 + bl_h_in * ux
    yf = y0 + bl_h_in * uy
    ok = np.isfinite(xf) & np.isfinite(yf)
    x_start = x0[ok][::stride]; y_start = y0[ok][::stride]
    x_end   = xf[ok][::stride]; y_end   = yf[ok][::stride]

    # build paired lines (N, 2)
    n = min(len(x_start), len(y_start), len(x_end), len(y_end))
    x_pairs = np.column_stack((x_start[:n], x_end[:n]))
    y_pairs = np.column_stack((y_start[:n], y_end[:n]))

    # pre-size outputs (fewer appends)
    N = x_pairs.shape[0]
    delta_n_mm[key] = np.empty(N, dtype=float)
    delta_n_m[key]  = np.empty(N, dtype=float)
    tau_w_dict[key] = np.empty(N, dtype=float)

    if DEBUG_PLOTS:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x0, y0, color='C0', lw=1.2, label='Geometry')
        for i in range(N):
            ax.plot(x_pairs[i], y_pairs[i], color='C3', lw=0.6, alpha=0.7)
        ax.set_aspect('equal', adjustable='box'); ax.legend(); ax.grid(True)
        plt.show()

    # --- HOT LOOP: suspend to avoid redraws/UI chatter ---
    with tp.session.suspend():
        # (Optional) also: tp.macro.execute_command('REDRAWCONTROL SETREDRAWINACTIVE')
        for i in range(N):
            p0 = (float(x_pairs[i, 0]), float(y_pairs[i, 0]), 0.0)
            p1 = (float(x_pairs[i, 1]), float(y_pairs[i, 1]), 0.0)

            line = tp.data.extract.extract_line([p0, p1], num_points=num_points)

            # Grab only needed vars
            y_BL = line.values("Y").as_numpy_array()
            U    = line.values("U").as_numpy_array()
            V    = line.values("V").as_numpy_array()
            P    = line.values("P").as_numpy_array()
            Rho  = line.values("R").as_numpy_array()
            Gam  = line.values("Gamma").as_numpy_array()
            tauy = line.values("Tau_y").as_numpy_array()

            tau_w = float(tauy[0])
            tau_w_dict[key][i] = tau_w

            # total enthalpy and ideal inviscid profile
            h0 = (Gam/(Gam-1.0)) * (P/Rho) + 0.5*(U*U + V*V)
            UI = np.sqrt( (2*Gam)/(Gam-1.0) * ((P_ref/rho_ref) - (P/Rho)) + U_ref*U_ref + V_ref*V_ref - V*V )

            # robust index after peak where h0 flattens
            y_index = flatten_after_peak_fast(y_BL, h0,
                                              smooth=9, window=9, frac=1e-5,
                                              abs_floor=None, early_tail=0.9,
                                              do_unique=use_unique_collapse)

            # ensure wall→outer order
            if y_BL[0] > y_BL[-1]:
                y_BL = y_BL[::-1]

            y_wall0   = y_BL[0]
            y_corr    = y_BL - y_wall0
            delta_mm  = float(y_corr[y_index]) * in_to_mm
            delta_m   = delta_mm * mm_to_m

            delta_n_mm[key][i] = delta_mm
            delta_n_m[key][i]  = delta_m

    # End with tp.session.resume() automatically (context manager)

    # (Optional) print a tiny summary instead of per-line spam
    print(f"{key}: N={N}, delta(mm) ~ [{np.nanmin(delta_n_mm[key]):.3g}, {np.nanmax(delta_n_mm[key]):.3g}]")

    
# Total time # 
end = time.time()
print(f"Total Elapsed Time: {end - start:.2f} seconds")

      
#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Post-processing the boundary layer thickness results
#------------------------------------------------------------------------------------------------------------------------------------#
"""


from scipy.interpolate import splrep, PPoly, splev
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def get_hl(section_key):
    # Match the number that comes right after 'h_l_'
    m = re.search(r'h_l_([0-9.]+)', section_key)
    if m:
        return float(m.group(1))
    return None

hl_groups = defaultdict(lambda: {"Re": [], "Sep": []})


cases_by_hl = {}
for key in ds_by_case:
    # Extract h_l part from key, e.g., "h_l_0.01" from "h_l_0.01_p0_1bar"
    hl_match = re.match(r"(h_l_\d+\.\d+)", key)
    if hl_match:
        hl = hl_match.group(1)
        cases_by_hl.setdefault(hl, []).append(key)
        

# Creating a criterion that highlights when separation occurs. To do so, we just find the index at which it happens as it was previously done. #
   #Obtaining the separation lenght based on the scipy function # 
   
# Pre-Allocating Variables # 
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

for section_key in ds_by_case:
    
    
    hl = get_hl(section_key)
    if hl is None:
        continue  # skip if pattern not found


    
    x_all[section_key] = X_geom[section_key]

    
    # x, y are your 1D arrays
    tck = splrep(x_all[section_key],tau_w_dict[section_key], s = 0)
    
    # Converting the polynomial to a ppoly object # 
    ppoly = PPoly.from_spline(tck)
    
    # Finding zeros # 
    x_zeros = ppoly.roots(extrapolate = False)
    y_zeros = splev(x_zeros,tck) 
    
    # finding the gradients at those points # 
    eps = 1e-8 * (x_all[section_key].ptp())  # tiny step relative to domain
    yL = splev(x_zeros - eps, tck)
    yR = splev(x_zeros + eps, tck)
    
    
    
    
    
    # If statement to showcase that if the value lenght of the sep points is odd, then remove the last zero # 
    if len(x_zeros) % 2 != 0:
        if yR[0] < 0:
            sep_location[section_key] = x_zeros[1:]
            tau_w_zeros[section_key] = y_zeros[1:]
            
            x_zeros = x_zeros[1:]
            y_zeros = y_zeros[1:]
            
            # finding the gradients at those points # 
            eps = 1e-8 * (x_all[section_key].ptp())  # tiny step relative to domain
            yL_new = splev(x_zeros - eps, tck)
            yR_new = splev(x_zeros + eps, tck)
            
        elif yR[-1] > 0: 
            sep_location[section_key] = x_zeros[:-1]
            tau_w_zeros[section_key] = y_zeros[:-1]
            
            x_zeros = x_zeros[:-1]
            y_zeros = y_zeros[:-1]
            
            # finding the gradients at those points # 
            eps = 1e-8 * (x_all[section_key].ptp())  # tiny step relative to domain
            yL_new = splev(x_zeros - eps, tck)
            yR_new = splev(x_zeros + eps, tck)
    else: 
        sep_location[section_key] = x_zeros
        tau_w_zeros[section_key] = y_zeros
        
        yR_new = yR
    
    
    # Finding attached and separated values # 
    x_attach[section_key] = x_zeros[np.where(yR_new > 0)]
    y_attach[section_key] = y_zeros[np.where(yR_new > 0)]
    
    x_sep[section_key] = x_zeros[np.where(yR_new < 0)]
    y_sep[section_key] = y_zeros[np.where(yR_new < 0)]
    
    sep_length_mm[section_key] = np.sum(abs(x_sep[section_key] - x_attach[section_key]))
    sep_length_nonDim[section_key] = (np.sum(abs(x_sep[section_key] - x_attach[section_key])))/(x_all[section_key][-1])
    
    # Obtaining indices at which the shear stress at the wall is below zero #
    tau_w_dict_nparray = np.array(tau_w_dict[section_key], dtype = float) # Converting list to a numpy array to use np.where
    idx_separation[section_key] = np.where( tau_w_dict_nparray <= 0  ) # Criterion at which 2D separation occurs at the wall # 
    idx_separation[section_key] = np.array(idx_separation[section_key]).flatten() # Converting from tuple ---> numpy array and then flattening the array 
    
    

    # Plotting Wall shear stress Vs X-coordinate #
    plt.plot(x_all[section_key] * 0.0393701 ,tau_w_dict[section_key], label = r'$\tau_y$')
    
    
    # Plotting separation point locations VS Separation shear stress values # 
    plt.scatter(sep_location[section_key] * 0.0393701,tau_w_zeros[section_key],color = 'red', label = 'Separation Points')
    
    
    # Plotting y line # 
    plt.axhline(0, linestyle = '--', color = 'black', label = 'Separation Line')
    
    
    # Describing the plot # 
    plt.legend(bbox_to_anchor = (1.05,1), loc = 'upper right' , borderaxespad=0.)
    plt.grid(True, which = "both")
    plt.title(r"$\tau_{y}$ Vs X[in]")
    plt.ylabel(r"$\tau_y[Pa]$")
    plt.xlabel("X[in]")
    plt.show()

#%% 

"""
#------------------------------------------------------------------------------------------------------------------------------------#
    Evaluating the following values: MAX BL Thickness Vs Re, Sep Length vs Re, BL thickness Vs X (With shear stress filtering)
#------------------------------------------------------------------------------------------------------------------------------------#
"""



fig_save_dir = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\21_ANSYS Workflow Automation\7_Parametric Study\1_Stagnation Pressure Change\1_Numerical Data"

# Pre-Allocating Variables # 
Re_L = {}


for section_key in ds_by_case:
    # Computing Reynolds number in respect of length # 
    mu = (1 / ds_by_case_inlet[key]["Mut_ovr_Mu"].data) * ds_by_case_inlet[key]["Mutur"].data
    U = ds_by_case_inlet[key]["U"].data
    rho = ds_by_case_inlet[key]["R"].data
    
    Re_L[section_key] =  (np.mean(rho) * (np.mean(U) ) * x_all[section_key] * in_to_mm ) / np.mean(mu)
    
    # Filtering delta_n_mm to remove points at which separation does occur # 
    delta_n_mm_filtered = np.delete(delta_n_mm[section_key] , idx_separation[section_key])
    delta_n_mm_sep = np.array(delta_n_mm[section_key])[idx_separation[section_key]]
    
    Re_L_filtered = np.delete(Re_L[section_key] , idx_separation[section_key])
    Re_L_sep = Re_L[section_key][idx_separation[section_key]]
    
    
    # Reynolds Number Vs Boundary layer thickness # 
    plt.plot(Re_L[section_key] , delta_n_mm[section_key], color = 'black',linewidth = 2)
    plt.scatter(Re_L_filtered , delta_n_mm_filtered,label = 'Attached')
    plt.scatter(Re_L_sep , delta_n_mm_sep , color = 'red', label = 'Separated')
    plt.legend()
    
    plt.title(f"BL Thickness Vs Re: {section_key}")
    plt.xlabel(r"$Re_{L}$")
    plt.ylabel(r"$\delta_e [mm]$")
    plt.grid(True, which = "both")
    plt.savefig(os.path.join(fig_save_dir,f"{section_key}_bl_Re.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.cla()




# --- Plot per h/L case ---
plt.figure(figsize=(8,6))
for hl, data in sorted(hl_groups.items()):
    R = np.asarray(data["Re"])
    S = np.asarray(data["Sep"])
    order = np.argsort(R)  # sort so lines don’t zig-zag
    plt.plot(R[order], S[order], marker='o', label=f'h/L = {hl}')

plt.title("Normalized Separation Length vs Re (grouped by h/L)")
plt.xlabel("Re")
plt.ylabel(r"$L_{sep}/L_{width}$")
plt.grid(True, which="both")
plt.legend(title="Cases")
plt.savefig(os.path.join(fig_save_dir,"separation_length_all.png"),dpi=300,bbox_inches = "tight")
plt.show()


# --- Individual plots per h/L ---
for hl, data in sorted(hl_groups.items()):
    R = np.asarray(data["Re"])
    S = np.asarray(data["Sep"])
    order = np.argsort(R)

    plt.figure(figsize=(8,6))
    plt.plot(R[order], S[order], marker='o', color='black')
    plt.title(f"Normalized Separation Length vs Re (h/L = {hl})")
    plt.xlabel("Re")
    plt.ylabel(r"$L_{sep}/L_{width}$")
    plt.grid(True, which="both")
    # Save each figure with h/L in the filename
    plt.savefig(os.path.join(fig_save_dir,f"sep_length_hl_{hl:.2f}.png"), dpi=300, bbox_inches="tight")
    plt.show()




#%%


"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Plotting the location of the first separation sensitivity
#------------------------------------------------------------------------------------------------------------------------------------#
"""



temp_keys = cases_by_hl["h_l_0.02"]

plt.figure(figsize=(8,6))
for temp_key in temp_keys:
    firstSepPointX = x_attach[temp_key][1] *  0.0393701
    firstSepPointY = y_sep[temp_key][1] *  0.0393701
    
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

#%% GPT
import numpy as np

def y_at_x_on_polyline(x, y, x_star):
    """
    Piecewise-linear interpolation on a polyline (x, y) to get y(x_star).
    If multiple segments cross x_star, picks the closest crossing.
    If x_star is outside the x-range, returns the nearest node's y.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # guard against NaNs
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    xdiff = x - x_star
    # segments [k,k+1] where x crosses x_star (or touches exactly)
    seg_idx = np.where((xdiff[:-1] * xdiff[1:] ) <= 0)[0]

    if seg_idx.size:
        # choose the crossing with the smallest distance to x_star
        # (handles cases with multiple crossings)
        k_candidates = seg_idx
        # for each candidate, compute a param t in [0,1] and the |dx| span
        x0 = x[k_candidates]
        x1 = x[k_candidates + 1]
        y0 = y[k_candidates]
        y1 = y[k_candidates + 1]

        # handle vertical segments (x1 == x0): pick midpoint in y
        vertical = (x1 == x0)
        out = np.empty(k_candidates.size, dtype=float)

        if np.any(~vertical):
            t = (x_star - x0[~vertical]) / (x1[~vertical] - x0[~vertical])
            out[~vertical] = y0[~vertical] + t * (y1[~vertical] - y0[~vertical])
        if np.any(vertical):
            out[vertical] = 0.5 * (y0[vertical] + y1[vertical])

        # pick the interpolated y from the segment with smallest |xdiff| near the crossing
        # metric: min(|x_k - x_star|, |x_{k+1} - x_star|)
        seg_dist = np.minimum(np.abs(xdiff[k_candidates]), np.abs(xdiff[k_candidates + 1]))
        best = np.argmin(seg_dist)
        return out[best]

    # no crossing: snap to nearest node
    j = int(np.argmin(np.abs(xdiff)))
    return y[j]

temp_keys = cases_by_hl["h_l_0.02"]

for i, temp_key in enumerate(temp_keys):
    # Your separation X (already in inches)
    firstSepPointX = x_attach[temp_key][1] * 0.0393701

    # Geometry
    x_temp = ds_by_case[temp_key]["X"].data
    y_temp = ds_by_case[temp_key]["Y"].data

    # Reynolds Number # 
    Re_temp = Re[temp_key]
    
    # ⚠️ Ensure units match:
    # If x_temp/y_temp are in meters, convert to inches to match firstSepPointX.
    # Comment these out if X/Y are already in inches.
    x_temp_in = x_temp
    y_temp_in = y_temp 

    # Compute the correct Y on the wall at this X
    firstSepPointY = y_at_x_on_polyline(x_temp_in, y_temp_in, firstSepPointX)
    
    # Color map # 
    cmap = cm.get_cmap("cividis",len(case_keys))
    
    #(optional) plot or store
    plt.plot(x_temp_in, y_temp_in, color = "red",linewidth = 3.0)
    plt.scatter(firstSepPointX, firstSepPointY, label = f"Re = {Re_temp:.2f}",zorder=5, color = cmap(i))

plt.xlabel("X (inches)")
plt.ylabel("Y (inches)")
plt.title(r"First Separation Point: $h_{L}$ = 0.02")
plt.legend(loc = "center left",bbox_to_anchor = (1.05,0.5),borderaxespad = 0.)
plt.grid(True)
plt.show()




   

