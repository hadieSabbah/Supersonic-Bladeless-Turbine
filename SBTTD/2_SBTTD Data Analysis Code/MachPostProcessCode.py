import xarray as xr
import numpy as np
from pathlib import Path
import sympy as sp
import tecplot as tp
import os 
import matplotlib.pyplot as plt

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                    Data importing and exporting
#------------------------------------------------------------------------------------------------------------------------------------#

"""


#%%
# Automatically changing the working directory # 
new_dirc = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\2_SBTTD Data Analysis Code"
os.chdir(new_dirc)

#%%
# Importing modules # 
# In your notebook/script, run this FIRST:
import utils.plotting

from utils.parameterComputation import variableImporterMasked, ReCompute, yplusThreshold
from utils.dataload_util import assign_dir, bigImport, runSaver, runLoader, file_pathFinder, load_minfo_step_force
from utils.plotting import plotter, plotter_multi_all, plotter_multiPerCase, subplotter, plot_scaled_axialForce_vs_hl,plot_BL_thickness,plot_BL_location_tecplot,plot_BL_thickness_subplots, plot_mach_contours_per_hl, plot_viscous_vs_inviscid_contours, subplotter_multiPerCase
from utils.models import analyze_geometries, get_first_shock_pressures, offsetGeomPoints, smallPertSolver, find_sepLength, max_min_finder,mach_vs_sepLength, smallPertSolver_with_SE, smallPertSolver_combined, compute_power_2D , compute_force_2D , compute_torque_2D_norm , load_csv_data, load_tecplot_data, generate_torque_table_mach , compute_torque_2D_norm, generate_axial_force_plot_mach, generate_axial_force_plot_dual_mach


#%%
### Connecting to the session # 
tp.session.connect()

#%%
#### Importing and processing raw data ####


# Defining values #
base_dir = Path(rf"{assign_dir()}")
fileName = "mcfd_tec.bin"
ds_by_case, ds_by_case_quad, ds_by_case_inlet = bigImport(base_dir,fileName)




#%% Importing processed data ###

ds_by_case,ds_by_case_quad, ds_by_case_inlet = runLoader(load_dir_dic = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\data\processed\Mach Study") 




#%% Saving data if needed ###


runSaver(ds_by_case, ds_by_case_quad, ds_by_case_inlet)



#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                        Computing ALL Variables and putting them in a dictionary 
#------------------------------------------------------------------------------------------------------------------------------------#

"""


# Defining Variables # 
min_l = 0 
max_l = 0.1

# Data at the wall masked #
y_plus, tau_x, tau_y, tau_separation, tau_separation_idx, \
x, y, T, P, Px, Py, P0, rho, mach, \
omega_z, u, v, q_dot = \
    variableImporterMasked(ds_by_case, min_l, max_l)

# Data at the entire quadrant # 
y_plus_quad, tau_x_quad, tau_y_quad, tau_separation_quad, tau_separation_idx_quad, \
x_quad, y_quad, T_quad, P_quad, Px_quad, Py_quad, P0_quad, rho_quad, mach_quad, \
omega_z_quad, u_quad, v_quad, q_dot_quad = \
    variableImporterMasked(ds_by_case_quad, 0, 0, mask_input = False)

# Data at the inlet # 
y_plus_inlet, tau_x_inlet, tau_y_inlet, tau_separation_inlet, tau_separation_idx_inlet, \
x_inlet, y_inlet, T_inlet, P_inlet, Px_inlet, Py_inlet, P0_inlet, rho_inlet, mach_inlet, \
omega_z_inlet, u_inlet, v_inlet, q_dot_inlet = \
    variableImporterMasked(ds_by_case_inlet, 0, 0, mask_input = False)
    



#%% 
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                            Computing the shear stress at the wall and saving it as a dictionary
#------------------------------------------------------------------------------------------------------------------------------------#

"""

# Pre-allocating variable # 
tau_wall = {}


for section_key in ds_by_case:
    
    # Getting discrete points from the geometry # 
    dx = np.gradient(x[section_key])
    dy = np.gradient(y[section_key])
    
    
    # Computing the normal unit tangent # 
    ds = np.sqrt(dx**2 + dy**2)
    tx = dx / ds
    ty = dy / ds
    
    # Getting the shear stress at the wall # 
    tau_wall[section_key] = tau_x[section_key] * tx + tau_y[section_key] * ty
    
    


#%% 

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                    Exporting coordinates 
#------------------------------------------------------------------------------------------------------------------------------------#

"""

def saveTxtSW2D(x,y,output_dir):
    
    z = np.zeros(len(x))
    # Store
    points_xyz = np.column_stack([x, y + 0.01, z])

    # Export individual curve .txt file (SolidWorks XYZ format)
    filename = "h_l_0_02_curve_mm.txt"
    filepath = os.path.join(output_dir, filename)
    np.savetxt(filepath, points_xyz, fmt='%.6f', delimiter='\t')
       
        

# Saving as txt # 
saveTxtSW2D(x['h_l_0.02_Mach_2.0'],y['h_l_0.02_Mach_2.0'], r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\34_Hannah Proejct")




#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                    Finding missing casees 
#------------------------------------------------------------------------------------------------------------------------------------#

"""

"""
Simple Variable Checker
=======================
Find which cases are missing a specific variable.
"""

def find_missing_cases(ds_by_case, variable_name):
    """
    Return list of case names that are missing the specified variable.
    
    Parameters
    ----------
    ds_by_case : dict
        Your dictionary of cases
    variable_name : str
        The variable to check for (e.g., "Mutur", "Qdot", "P_total")
    
    Returns
    -------
    list
        Case names missing the variable
    """
    missing = []
    for case in ds_by_case:
        if variable_name not in ds_by_case[case].keys():
            missing.append(case)
    return missing


# Example usage:
# missing = find_missing_cases(ds_by_case, "Mutur")
# print(f"Cases missing Mutur: {missing}")




# Function to print the cases with missing variables # 
missing = find_missing_cases(ds_by_case, "P_x")
print(missing)  # ['M2.0_P0_0.5', 'M3.0_P0_1.2', ...]
    
    
    


#%% 
    
    
import matplotlib as pyplot
from tecplot.constant import PlotType

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                 Boundary layer thickness detection
#------------------------------------------------------------------------------------------------------------------------------------#
"""

# Connecting to Tecplot #
tp.session.connect()

# Pre-Allocating Variables for boundary layer edge #
delta_n_dict = {}
tau_w_dict = {}

# Defining variables #
Nx = 500
Ny = 1000
bl_h = 3 / 1000
stride = 1
num_points = Ny

# Defining all the directories #
base_dir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\data\raw\Mach Study 2")
rootDir = base_dir
subDirs1 = [p for p in rootDir.iterdir() if p.is_dir()]

fileName = "mcfd_tec.bin"
subDirs2 = [p for d in subDirs1 for p in d.iterdir() if p.is_dir()]
file_paths = [p / fileName for p in subDirs2]


for idx, key in enumerate(ds_by_case.keys()):

    # Import case into tecplot #
    tp.new_layout()
    tp.data.load_tecplot(file_paths[idx].as_posix())
    fr = tp.active_frame()
    fr.plot_type = PlotType.Cartesian2D

    # Finding the gradient for x and y #
    slice_cut_int = 0
    dx_ds = np.gradient(x[key][slice_cut_int:-1])
    dy_ds = np.gradient(y[key][slice_cut_int:-1])

    # Finding the Normal values for line extraction #
    nx, ny = -dy_ds, dx_ds
    norm = np.hypot(nx, ny)
    norm = np.where(norm < 1e-12, np.nan, norm)
    ux, uy = nx / norm, ny / norm

    # Raking End points #
    x_final = x[key][slice_cut_int:-1] + bl_h * ux
    y_final = y[key][slice_cut_int:-1] + bl_h * uy
    ok = np.isfinite(x_final) & np.isfinite(y_final)

    # Apply Stride #
    x_start = x[key][slice_cut_int:-1][ok][::stride]
    y_start = y[key][slice_cut_int:-1][ok][::stride]
    x_end = x_final[ok][::stride]
    y_end = y_final[ok][::stride]

    # Precompute tangent vectors for all stride points (outside inner loop) #
    dx_arr = dx_ds[ok][::stride]
    dy_arr = dy_ds[ok][::stride]
    ds_arr = np.sqrt(dx_arr**2 + dy_arr**2)
    tx_arr = dx_arr / ds_arr
    ty_arr = dy_arr / ds_arr

    # Build arrays #
    n = min(len(x_start), len(y_start), len(x_end), len(y_end))
    x_pairs = np.column_stack((x_start[:n], x_end[:n]))
    y_pairs = np.column_stack((y_start[:n], y_end[:n]))

    # Pre-allocate output arrays
    N = x_pairs.shape[0]
    delta_n_dict[key] = np.full(N, np.nan, dtype=float)
    tau_w_dict[key] = np.full(N, np.nan, dtype=float)

    # Pre-allocate lists for BL edge points (batch zone creation) #
    edge_x_list = []
    edge_y_list = []
    
    #Test #
    
    
    print(50 * "==")
    print(f"Processing {file_paths[idx]}")
    print(50 * "==")
    print("\n")

    for i in range(N):
        p0 = (float(x_pairs[i, 0]), float(y_pairs[i, 0]), 0.0)
        p1 = (float(x_pairs[i, 1]), float(y_pairs[i, 1]), 0.0)

        # Single line extraction (no duplicate call) #
        line = tp.data.extract.extract_line([p0, p1], num_points=num_points)

        # Shear stress at the wall #
        tau_x_line = line.values("Tau_x").as_numpy_array()[0]
        tau_y_line = line.values("Tau_y").as_numpy_array()[0]

        # Shear stress projected onto wall tangent #
        tau_wall_line = tau_x_line * tx_arr[i] + tau_y_line * ty_arr[i]
        
        print(f"Iteration {i}:")
        print(f"shear stress = {tau_wall_line:.4f} Pa")

        if tau_wall_line <= 0:
            print("------------Point skipped due to separation!\n")
            continue

        # Store wall shear stress #
        tau_w_dict[key][i] = tau_wall_line

        # Grab BL profile variables from the already-extracted line #
        x_BL       = line.values("X").as_numpy_array()
        y_BL       = line.values("Y").as_numpy_array()
        omega_z_Bl = line.values("Vort_z").as_numpy_array()

        # Finding the location at which the vorticity is almost equal to zero #
        
        # Old threshold # 
        #y_index = np.abs(omega_z_Bl).argmin()
        
        # Vorticity gradient along the rake (d_omega/dn) #
        d_omega = np.gradient(omega_z_Bl)
        
        # Normalize by peak gradient to make threshold geometry-independent #
        d_omega_norm = np.abs(d_omega) / (np.abs(d_omega).max() + 1e-30)
        
        # BL edge: first point (beyond ~10% of rake) where normalized gradient < threshold #
        threshold = 0.02
        start_idx = max(1, num_points // 10)  # skip near-wall region
        candidates = np.where(d_omega_norm[start_idx:] < threshold)[0]
        
        if len(candidates) > 0:
            y_index = candidates[0] + start_idx
        else:
            y_index = num_points - 1  # fallback: use rake tip



        # Ensure wall→outer order #
        if y_BL[0] > y_BL[-1]:
            y_BL = y_BL[::-1]

        y_wall0 = y_BL[0]
        y_corr = y_BL - y_wall0

        # Saving the boundary layer edge values #
        delta_mm = float(y_corr[y_index]) * 1000
        delta_n_dict[key][i] = delta_mm
        print(f"BL thickness = {delta_mm:.3f} mm")
        
        print(20*"==")

        # Collect BL edge coordinates for batch plotting #
        edge_x_list.append(float(x_BL[y_index]))
        edge_y_list.append(float(y_BL[y_index]))
        





#%% 

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                Plotting boundary layer thickness results 
#------------------------------------------------------------------------------------------------------------------------------------#

"""
x_start_dict = {key: x[key][slice_cut_int:-1] for key in x}
plot_BL_thickness(delta_n_dict, x_start_dict, save=True)



#%% Subplotting the boundary layer results ###


"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                SUB-Plotting boundary layer thickness results 
#------------------------------------------------------------------------------------------------------------------------------------#
"""
plot_BL_thickness_subplots(delta_n_dict, x_start_dict, save=False)





#%% Plotting the results on tecplot 


"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                        Plotting the location of the BL edge in Tecplot
#------------------------------------------------------------------------------------------------------------------------------------#
"""


from tecplot.constant import PlotType, GeomShape, Color
import tecplot as tp






edge_x_dict = {}
edge_y_dict = {}

bl_h          = 3 / 1000
slice_cut_int = 0
stride        = 1

for key in delta_n_dict:
    dx_ds = np.gradient(x[key][slice_cut_int:-1])
    dy_ds = np.gradient(y[key][slice_cut_int:-1])

    nx, ny = -dy_ds, dx_ds
    norm   = np.hypot(nx, ny)
    norm   = np.where(norm < 1e-12, np.nan, norm)
    ux, uy = nx / norm, ny / norm

    x_final = x[key][slice_cut_int:-1] + bl_h * ux
    y_final = y[key][slice_cut_int:-1] + bl_h * uy
    ok      = np.isfinite(x_final) & np.isfinite(y_final)

    x_s = x[key][slice_cut_int:-1][ok][::stride]
    y_s = y[key][slice_cut_int:-1][ok][::stride]

    delta_arr = delta_n_dict[key]
    valid     = np.isfinite(delta_arr)
    delta_m   = delta_arr[valid] / 1000.0

    edge_x_dict[key] = x_s[valid] + delta_m * ux[ok][::stride][valid]
    edge_y_dict[key] = y_s[valid] + delta_m * uy[ok][::stride][valid]
    
    
  
    
plot_BL_location_tecplot(edge_x_dict, edge_y_dict, file_paths, ds_by_case)



#%% Saving the boundary layer height results ##

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                        Saving BL results
#------------------------------------------------------------------------------------------------------------------------------------#
"""


from datetime import date
from pathlib import Path

# --- Export BL thickness results ---
base_dir = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\data\processed\Boundary Layer Data\Mach Study")
today = date.today().strftime("%Y-%m-%d")
output_dir = base_dir / f"BL_results_{today}"
output_dir.mkdir(exist_ok=True)

np.savez(
    output_dir / "delta_n_dict.npz",
    **delta_n_dict
)

print(f"Saved delta_n_dict to {output_dir / 'delta_n_dict.npz'}")

#%% Reload results # 
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                        Reloading BL results
#------------------------------------------------------------------------------------------------------------------------------------#
"""



loaded = np.load(output_dir / "delta_n_dict.npz", allow_pickle=False)
delta_n_dict = {key: loaded[key] for key in loaded.files}


#%% Automatic reloading ###

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                        Auto-Reloading BL results
#------------------------------------------------------------------------------------------------------------------------------------#
"""

from datetime import date
from pathlib import Path

# --- Reload latest BL results ---
all_bl_dirs = sorted(base_dir.glob("BL_results_*"))  # finds all dated folders

latest_dir = max(all_bl_dirs, key=lambda p: date.fromisoformat(p.name.split("_")[-1]))

loaded = np.load(latest_dir / "delta_n_dict.npz", allow_pickle=False)
delta_n_dict = {key: loaded[key] for key in loaded.files}

print(f"Loaded delta_n_dict from {latest_dir}")







#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                 Plotting the Mach contours and inviscid contours
#------------------------------------------------------------------------------------------------------------------------------------#
"""


VISCOUS_DIR  = r"\\oitrspprd.hpc.ncsu.edu\rsstu\users\j\jbraun2\yip_afosr\hhsabbah\32_Geometry Code\Results\2_Contours\1_Viscous Simulations\1_Mach Study\1_Mach Contour"
INVISCID_DIR = r"\\oitrspprd.hpc.ncsu.edu\rsstu\users\j\jbraun2\yip_afosr\hhsabbah\32_Geometry Code\Results\2_Contours\2_Inviscid Simulations\1_Mach Study\1_Mach Contour"

# Function 1 — all Mach contours for h/l = 0.03, 3 columns
plot_mach_contours_per_hl(
    hl_value    = 0.03,
    viscous_dir = VISCOUS_DIR,
    ncols       = 3,
    save        = True,
    save_dir    = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\37_Mesh and CFD Setup\8_Proccessed Contours Results\1_Mach Sweep Study"
)


#%%

# Function 2 — viscous vs inviscid for h/l = 0.05

mach_range = [1.5,4.0]
plot_viscous_vs_inviscid_contours(
    hl_value     = 0.06,
    viscous_dir  = VISCOUS_DIR,
    inviscid_dir = INVISCID_DIR,
    save         = True,
    save_dir     = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\37_Mesh and CFD Setup\8_Proccessed Contours Results\2_Inviscid Comparison",
    mach_range = mach_range
)


#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Processing Convergence criteria results(X-force Vs X) 
#------------------------------------------------------------------------------------------------------------------------------------#

"""



# ---- Defining the file name and the root directory  ---- #
rootDir_info = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Results\Mach Study 2")
fileName_info = "minfo1_e2"



file_paths_info = file_pathFinder(fileName_info , rootDir_info)



# For loop tos ave the residual variables # 
for file_path in file_paths_info: 
    x_resid_info, f_resid_info = load_minfo_step_force(file_path.as_posix(), use="step")  # use="time" to plot vs Time instead
    if x_resid_info.size == 0:
        print(f"Skipping (no data): {file_path} \n")
        continue
    
    # Finding the lab from the folder name #
    label = file_path.parent.name
 

# Check density gradient magnitudes for one case
key = 'h_l_0.06_Mach_3.5'

x_arr = x_quad[key]
y_arr = y_quad[key]
rho_arr = rho_quad[key]

# At one x-station
x_loc = 0.05
mask = np.abs(x_arr - x_loc) < 0.001

y_local = y_arr[mask]
rho_local = rho_arr[mask]

sort_idx = np.argsort(y_local)
y_sorted = y_local[sort_idx]
rho_sorted = rho_local[sort_idx]

dy = np.diff(y_sorted)
d_rho = np.diff(rho_sorted)
dy[dy == 0] = 1e-10

rho_grad = np.abs(d_rho / dy)

print(f"Max density gradient: {np.max(rho_grad):.2f}")
print(f"Mean density gradient: {np.mean(rho_grad):.2f}")
print(f"99th percentile: {np.percentile(rho_grad, 99):.2f}")





#%%



"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                Plotting Processed Results: y+, Shear Stress[Pa], and location of separation
#------------------------------------------------------------------------------------------------------------------------------------#
"""
    
## Plotting the results of tau_x to see how they differ from one another ##
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re

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


h_l_list = np.arange(0.02,0.09 + 0.01,0.01)


for h_l in h_l_list:
    fig, ax = plotter_multiPerCase(
        x_dict=x,
        y_dict=tau_wall,
        x_string='x',
        y_string=r'$\tau_{wall}$',
        unit_x='[m]',
        unit_y='[Pa]',
        filter_param='h_l',
        filter_value= round(h_l,2),
        vary_param='mach',
        cmap_name='cividis'
    )

        



#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                                 Plotting y+
#------------------------------------------------------------------------------------------------------------------------------------#
""" 

  
    
# Plotting Y+ values on top of each other # 

for h_l in h_l_list:
    fig, ax = plotter_multiPerCase(
        x_dict=x,
        y_dict=y_plus,
        x_string='x',
        y_string=r'$y^{+}$',
        unit_x='[m]',
        unit_y='',
        filter_param='h_l',
        filter_value= round(h_l,2),
        vary_param='mach',
        cmap_name='cividis'
    )


#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                        Plotting Separation Points
#------------------------------------------------------------------------------------------------------------------------------------#
"""   




import numpy as np
from scipy.interpolate import UnivariateSpline



for key in x.keys():
    # Pre-allocating Variables #
    tau_y_list = []
    sep_length = []
    
    # your arrays
    x_data = x[key]
    y_data = tau_wall[key]
    
    # x, y are your 1D arrays
    s = UnivariateSpline(x_data, y_data, s=0)
    x_zeros = s.roots()
    idx_nearest = np.searchsorted(x_data, x_zeros)
    
    # Obtaining the separation length
    for i in range(len(idx_nearest) - 1):
        np.array(tau_y_list.append(tau_y[key][idx_nearest[i]:idx_nearest[i+1]]))
        sep_length.append(x[key][idx_nearest[i+1]] - x[key][idx_nearest[i]])
    
    # Getting the separation location and respective tau
    sep_location = [x[key][idx_nearest[k]] for k in range(len(idx_nearest))]
    tau_y_location = [tau_wall[key][idx_nearest[k]] for k in range(len(idx_nearest))]
    
    # Use plotter but get fig, ax to add scatter points
    fig, ax = plotter(x_data, y_data, 'x', r'$\tau_{wall}$', '[m]', '[Pa]', 
                      save=False, return_axes = True)
    
    # Add separation points
    ax.scatter(sep_location, tau_y_location, color='red', s=50, 
               label='Separation Points', zorder=5)
    ax.legend()
    
    plt.show()
    
    
    
    
    
    
    
#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                      Plotting pressure gradients across the wavy section
#------------------------------------------------------------------------------------------------------------------------------------#
""" 

import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import re



h_l_list = np.arange(2, 9 + 1, 1) * 0.01

for h_l in h_l_list:
    plotter_multiPerCase(x, Px, "x", "$P_{x}$", "[m]", "[Pa]", filter_param = "h_l", filter_value = h_l,  labels=None, cmap_name='cividis', save=False, title=None)


subplotter_multiPerCase(
    x, Px,
    x_string='x',
    y_string=r'$\frac{dp}{dx}$',
    unit_x='[m]',
    unit_y='[Pa/m]',
    filter_param='h_l',
    filter_values=h_l_list,
    vary_param='mach',
    save=False
)


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

def load_mcfd_info1(root_dir, outlet_selector=2):
    """
    Parse iCFD++ mcfd.info1 boundary flux files across multiple case folders.
    Returns dicts compatible with plotter_multi_all() and subplotter_multiPerCase().

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing case subfolders, each with mcfd.info1
    outlet_selector : int
        Selector number for the outlet boundary (default: 2)

    Returns
    -------
    iter_dict : dict
        {case_name: np.array of iteration numbers}
    flux_dict : dict
        {case_name: np.array of outlet mass flux values}
    """
    root_dir = Path(root_dir)
    case_dirs = sorted([p for p in root_dir.rglob("mcfd.info1")])

    iter_dict = {}
    flux_dict = {}

    for info1_path in case_dirs:
        case_name = info1_path.parent.name
        iters, fluxes = [], []
        current_iter = None
        current_sel  = None
        in_nondim    = False
        sel_flux     = {}

        with open(info1_path) as f:
            for line in f:
                line = line.rstrip()

                m = re.match(r'^nt\s+(\d+)', line)
                if m:
                    if current_iter is not None:
                        iters.append(current_iter)
                        fluxes.append(sel_flux.get(outlet_selector, np.nan))
                    current_iter = int(m.group(1))
                    sel_flux, current_sel, in_nondim = {}, None, False
                    continue

                m = re.match(r'^For selector\s+(\d+)', line)
                if m:
                    current_sel = int(m.group(1))
                    in_nondim   = False
                    continue

                if 'nondimensional' in line:
                    in_nondim = True
                    continue
                if 'dimensional' in line and 'non' not in line:
                    in_nondim = False
                    continue

                if in_nondim and current_sel is not None and 'mass   flux' in line:
                    if current_sel not in sel_flux:   # first = nondim total
                        sel_flux[current_sel] = float(line.split()[2])

        # flush last iteration
        if current_iter is not None:
            iters.append(current_iter)
            fluxes.append(sel_flux.get(outlet_selector, np.nan))

        iter_dict[case_name] = np.array(iters)
        flux_dict[case_name] = np.abs(np.array(fluxes))

    return iter_dict, flux_dict

def load_info0(path):
    """
    Parse iCFD++ mcfd.info0 residual file.
    Columns: iter | dt | L2(rho) | L2(rhou) | L2(rhoe) | CFL | tau | misc
    Returns dict of numpy arrays.
    """
    data = np.loadtxt(path)
    return {
        "iter":    data[:, 0],
        "L2_rho":  data[:, 2],
        "L2_rhou": data[:, 3],
        "L2_rhoe": data[:, 4],
        "tau":     data[:, 6],
    }


def load_info1(path, outlet_selector=2):
    """
    Parse iCFD++ mcfd.info1 boundary flux file.
    Extracts outlet mass flux (nondimensional) per iteration.
    
    Selector 1 = inlet (fixed BC), 2 = outlet (tracks convergence), 3 = wall (~0)
    """
    iters, outlet_flux = [], []
    current_iter  = None
    current_sel   = None
    in_nondim     = False
    sel_flux      = {}

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            m = re.match(r'^nt\s+(\d+)', line)
            if m:
                if current_iter is not None:
                    iters.append(current_iter)
                    outlet_flux.append(sel_flux.get(outlet_selector, np.nan))
                current_iter = int(m.group(1))
                sel_flux, current_sel, in_nondim = {}, None, False
                continue

            m = re.match(r'^For selector\s+(\d+)', line)
            if m:
                current_sel = int(m.group(1))
                in_nondim   = False
                continue

            if 'nondimensional' in line:
                in_nondim = True;  continue
            if 'dimensional' in line and 'non' not in line:
                in_nondim = False; continue

            if in_nondim and current_sel is not None and 'mass   flux' in line:
                if current_sel not in sel_flux:          # first = nondim total
                    sel_flux[current_sel] = float(line.split()[2])

    if current_iter is not None:                         # flush last iteration
        iters.append(current_iter)
        outlet_flux.append(sel_flux.get(outlet_selector, np.nan))

    return {
        "iter":         np.array(iters),
        "outlet_mflux": np.abs(np.array(outlet_flux)),
    }


def icfd_convergence_plotter(root_dir, case_labels=None, save=False):
    """
    Plot iCFD++ convergence (residuals + mass flux) for all cases under root_dir.
    Mirrors the style of residual_plotter() and mass_flux_analyzer().

    Expects structure:
        root_dir/
            case_A/   <- contains mcfd.info0 and mcfd.info1
            case_B/
            ...

    Parameters
    ----------
    root_dir   : str or Path
    case_labels: dict, optional  {folder_name: display_label}
    save       : bool
    """
    root_dir  = Path(root_dir)
    case_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])

    if not case_dirs:
        print("No subdirectories found.")
        return

    # ── collect all data first ──────────────────────────────────────────────
    cases = []
    for d in case_dirs:
        p0 = d / "mcfd.info0"
        p1 = d / "mcfd.info1"
        if not p0.exists() or not p1.exists():
            print(f"Skipping {d.name}: missing info0 or info1")
            continue
        label = case_labels.get(d.name, d.name) if case_labels else d.name
        cases.append({
            "label": label,
            "res":   load_info0(p0),
            "flux":  load_info1(p1),
        })

    if not cases:
        print("No valid cases found.")
        return

    cmap = cm.get_cmap("cividis", len(cases))

    # ── 1. Separate plot per case ────────────────────────────────────────────
    for i, c in enumerate(cases):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)
        color = cmap(i)

        # Residuals
        ax = axes[0]
        ax.semilogy(c["res"]["iter"], c["res"]["L2_rho"],  color=color,          lw=2, label=r"L2($\rho$)")
        ax.semilogy(c["res"]["iter"], c["res"]["L2_rhou"], color=color, ls="--", lw=2, label=r"L2($\rho u$)")
        ax.semilogy(c["res"]["iter"], c["res"]["L2_rhoe"], color=color, ls=":",  lw=2, label=r"L2($\rho e$)")
        ax.set_ylabel("L2 Residual")
        ax.set_title(f"Residuals — {c['label']}", fontsize=14)
        ax.legend(frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=12)

        # Mass flux
        ax = axes[1]
        inlet_ref = c["flux"]["outlet_mflux"][0]
        ax.plot(c["flux"]["iter"], c["flux"]["outlet_mflux"], color=color, lw=2, label="|ṁ| outlet")
        ax.axhline(inlet_ref, color="gray", ls=":", lw=1.5, label=f"Inlet ref = {inlet_ref:.3f}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("|ṁ| [kg/s]")
        ax.set_title(f"Mass Flux Convergence — {c['label']}", fontsize=14)
        ax.legend(frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=12)

        plt.tight_layout()

        if save:
            dirc = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Convergence")
            dirc.mkdir(parents=True, exist_ok=True)
            plt.savefig(dirc / f"convergence_{c['label']}.png", dpi=600, bbox_inches="tight")
            plt.savefig(dirc / f"convergence_{c['label']}.pdf", bbox_inches="tight")

        plt.show()

    # ── 2. Combined plot — all cases together ────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for i, c in enumerate(cases):
        color = cmap(i)
        lbl   = c["label"]
        axes[0].semilogy(c["res"]["iter"],  c["res"]["L2_rho"],          color=color, lw=2, label=f"{lbl} — L2(ρ)")
        axes[0].semilogy(c["res"]["iter"],  c["res"]["L2_rhou"], ls="--", color=color, lw=2)
        axes[0].semilogy(c["res"]["iter"],  c["res"]["L2_rhoe"], ls=":",  color=color, lw=2)
        axes[1].plot(c["flux"]["iter"], c["flux"]["outlet_mflux"],        color=color, lw=2, label=lbl)

    axes[0].set_ylabel("L2 Residual");     axes[0].set_title("Residuals — All Cases", fontsize=14)
    axes[1].set_ylabel("|ṁ| [kg/s]");     axes[1].set_title("Mass Flux — All Cases", fontsize=14)
    axes[1].set_xlabel("Iteration")

    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=12)

    plt.tight_layout()

    if save:
        plt.savefig(dirc / "convergence_combined.png", dpi=600, bbox_inches="tight")
        plt.savefig(dirc / "convergence_combined.pdf", bbox_inches="tight")

    plt.show()


def load_mcfd_net_mass_flux(root_dir, inlet_selector=1, outlet_selector=2):
    """
    Parse mcfd.info1 and compute net mass flux per iteration.
    
    Net mass flux = (ṁ_outlet + ṁ_inlet) / A_outlet
    Approaches 0 as simulation converges.

    Returns
    -------
    iter_dict : dict  {case_name: iterations array}
    flux_dict : dict  {case_name: net mass flux array [kg/m²·s]}
    """
    root_dir = Path(root_dir)
    iter_dict, flux_dict = {}, {}

    for info1_path in sorted(root_dir.rglob("mcfd.info1")):
        case_name = info1_path.parent.name
        iters, net_fluxes = [], []

        current_iter = None
        current_sel  = None
        in_nondim    = False
        sel_flow     = {}   # selector -> mass flow [kg/s]
        sel_area     = {}   # selector -> area [m²]

        with open(info1_path) as f:
            for line in f:
                line = line.rstrip()

                m = re.match(r'^nt\s+(\d+)', line)
                if m:
                    if current_iter is not None:
                        ṁ_in  = sel_flow.get(inlet_selector,  np.nan)
                        ṁ_out = sel_flow.get(outlet_selector, np.nan)
                        A_out = sel_area.get(outlet_selector, np.nan)
                        iters.append(current_iter)
                        net_fluxes.append((ṁ_out + ṁ_in) / A_out)
                    current_iter = int(m.group(1))
                    sel_flow, sel_area = {}, {}
                    current_sel, in_nondim = None, False
                    continue

                m = re.match(r'^For selector\s+(\d+)', line)
                if m:
                    current_sel = int(m.group(1))
                    in_nondim   = False
                    continue

                if 'nondimensional' in line:
                    in_nondim = True
                    continue
                if 'dimensional' in line and 'non' not in line:
                    in_nondim = False
                    continue

                if in_nondim and current_sel is not None:
                    if 'mass   flux' in line and current_sel not in sel_flow:
                        sel_flow[current_sel] = float(line.split()[2])
                    if line.strip().startswith('areas') and current_sel not in sel_area:
                        sel_area[current_sel] = abs(float(line.split()[-1]))

        # flush last iteration
        if current_iter is not None:
            ṁ_in  = sel_flow.get(inlet_selector,  np.nan)
            ṁ_out = sel_flow.get(outlet_selector, np.nan)
            A_out = sel_area.get(outlet_selector, np.nan)
            iters.append(current_iter)
            net_fluxes.append((ṁ_out + ṁ_in) / A_out)

        iter_dict[case_name] = np.array(iters)
        flux_dict[case_name] = np.array(net_fluxes)

    return iter_dict, flux_dict




#### Plotting cases # 

root = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\data\raw\Mach Study 2"

iter_dict, flux_dict = load_mcfd_info1(root, outlet_selector=2)

iter_dict, flux_dict = load_mcfd_net_mass_flux(root)

# Separate plot per h/l case
subplotter_multiPerCase(iter_dict, flux_dict,
                        x_string="Iteration", y_string="Net Mass Flux",
                        unit_x="[-]", unit_y=r"[kg/m²·s]",
                        filter_param="h_l",
                        filter_values=["0.02", "0.04", "0.06","0.08"],  # your h/l values
                        vary_param="mach",
                        overall_title="Net Mass Flux Convergence Study")



#%%
# Combined plot — all cases on one axes
plotter_multi_all(iter_dict, flux_dict,
                  x_string="Iteration", y_string="|ṁ| outlet",
                  unit_x="[-]", unit_y="[kg/s]",
         
                  
         title="Mass Flux Convergence — Mesh Study")

# Separate plot per h/l case
subplotter_multiPerCase(iter_dict, flux_dict,
                        x_string="Iteration", y_string="|ṁ| outlet",
                        unit_x="[-]", unit_y="[kg/s]",
                        filter_param="h_l",
                        filter_values=["0.05", "0.10", "0.15"],
                        vary_param="mach")
    




#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Post-processing the boundary layer thickness results
#------------------------------------------------------------------------------------------------------------------------------------#
"""





# === Separation/Attachment from sign of Tau_x (no splines), ignoring edge pairs ===
sep_length, sep_length_nonDim, x_sep, y_sep, x_attach, y_attach = find_sepLength(ds_by_case,x,tau_wall)


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
    #plt.scatter(ds_by_case[section_key]["X"].data[:len(delta_n_mm_3_dict[section_key])], delta_n_mm_3_dict[section_key],label = 'Attached')
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
# To do so, the maximas and minimas of each respective geometry will be found.

x_max, x_min, y_max, y_min = max_min_finder(ds_by_case,x,y)

#%%



"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                Plotting Separation Length Vs Re_L
#------------------------------------------------------------------------------------------------------------------------------------#
"""

mach = mach_vs_sepLength(ds_by_case, x, y, sep_length_nonDim)









#%% GPT: Getting the first point at which separation occurs.


 
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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



#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
         Figure A: Separation Length vs h/l  (uses find_sepLength outputs)
         Figure B: Heatmap - Separation Occurrence Matrix
         Figure C: First Separation Location vs h/l (original figure, kept for reference)
#------------------------------------------------------------------------------------------------------------------------------------#

HOW THIS WORKS:
===============
Instead of manually detecting separation from x_sep arrays and filtering by geometry
maxima (x_max), this version uses the outputs from find_sepLength(), which already does
the heavy lifting:
    - sep_length[key]         : total separation length in [m]
    - sep_length_nonDim[key]  : separation length / domain length (non-dimensional)
    - x_sep[key]              : array of separation point x-locations
    - x_attach[key]           : array of reattachment point x-locations

The key insight: find_sepLength() identifies separation as regions where tau_x < 0,
interpolates exact zero-crossings, and computes the sum of all SEP→ATTACH spans.
This is more robust than the previous approach of filtering x_sep by geometry maxima.

WHAT YOU NEED TO HAVE RUN BEFORE THIS:
=======================================
1. variableImporterMasked()  → gives you x, tau_x, etc.
2. find_sepLength(ds_by_case, x, tau_x)  → gives you sep_length, sep_length_nonDim,
                                            x_sep, y_sep, x_attach, y_attach

USAGE:
======
    # Step 1: Get your variables
    y_plus, tau_x, tau_y, ... = variableImporterMasked(ds_by_case, min_l, max_l)
    
    # Step 2: Compute separation lengths
    sep_length, sep_length_nonDim, x_sep, y_sep, x_attach, y_attach = \
        find_sepLength(ds_by_case, x, tau_x)
    
    # Step 3: Run this plotting code (below)
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

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
# STEP 1: Extract data into organized structure using find_sepLength outputs
# =============================================================================


results_by_mach = {m: {'h_l': [], 'sep_len': [], 'sep_len_nonDim': [],
                        'x_sep_first': [], 'separates': []} for m in mach_values}

for h_l in h_l_values:
    h_l_key = f"h_l_{h_l:.2f}"
    temp_keys = cases_by_hl[h_l_key]
    
    for temp_key in temp_keys:
        # Parse Mach value from the case key
        m = re.search(r"Mach_([0-9]*\.?[0-9]+)", temp_key)
        if not m:
            continue
        mach_val = float(m.group(1))
        
        if mach_val not in mach_values:
            continue
        
        # ---- Pull data from find_sepLength outputs ----
        # sep_length[temp_key] is the TOTAL separation length (sum of all bubbles)
        # x_sep[temp_key] is the array of separation START locations
        
        sep_len_val = sep_length.get(temp_key, 0.0)
        sep_len_nd_val = sep_length_nonDim.get(temp_key, 0.0)
        x_sep_arr = np.asarray(x_sep.get(temp_key, [])).ravel()
        
        # Determine if separation occurs:
        # sep_length > 0 means tau_x went negative somewhere in the domain
        does_separate = (sep_len_val > 0) and (x_sep_arr.size > 0)
        
        # Store results
        results_by_mach[mach_val]['h_l'].append(h_l)
        results_by_mach[mach_val]['sep_len'].append(sep_len_val)
        results_by_mach[mach_val]['sep_len_nonDim'].append(sep_len_nd_val)
        results_by_mach[mach_val]['separates'].append(does_separate)
        
        # Also store first separation location for Figure C
        if does_separate:
            results_by_mach[mach_val]['x_sep_first'].append(float(np.min(x_sep_arr)))
        else:
            results_by_mach[mach_val]['x_sep_first'].append(np.nan)


# =============================================================================
# FIGURE A: Separation Length vs h/l
# =============================================================================


# --- Figure A1: Dimensional separation length ---
fig_a1, ax_a1 = plt.subplots(figsize=(12, 8))

# Define outliers to exclude: list of (mach, h_l) tuples
outliers_to_exclude = [
    # (3.5, 0.04),  # Example: known bad separation detection
    # Add more tuples here if needed
]

for mach_val in mach_values:
    data = results_by_mach[mach_val]
    
    # Filter: only where separation occurs AND not an outlier
    h_l_sep = []
    sep_vals = []
    
    for h, sl, sep in zip(data['h_l'], data['sep_len'], data['separates']):
        if sep and (mach_val, h) not in outliers_to_exclude:
            h_l_sep.append(h)
            sep_vals.append(sl)
    
    if h_l_sep:
        sorted_pairs = sorted(zip(h_l_sep, sep_vals))
        h_sorted, sl_sorted = zip(*sorted_pairs)
        
        ax_a1.plot(h_sorted, sl_sorted,
                   color=mach_colors[mach_val],
                   marker=mach_markers[mach_val],
                   markersize=14,
                   linewidth=2.5,
                   markeredgecolor='black',
                   markeredgewidth=1,
                   label=f'M = {mach_val}')

ax_a1.set_xlabel('h/l', fontsize=26)
ax_a1.set_ylabel('Total Separation Length [m]', fontsize=26)
ax_a1.set_title('Separation Length vs h/l', fontsize=34, fontweight='bold')
ax_a1.tick_params(labelsize=21)
ax_a1.legend(title='Mach Number', fontsize=18, title_fontsize=21, loc='best')
ax_a1.grid(True, alpha=0.3)
ax_a1.set_xlim([min(h_l_values) - 0.005, max(h_l_values) + 0.005])

plt.tight_layout()
plt.savefig('figure_a1_sepLength_vs_hl.png', dpi=200, bbox_inches='tight')
plt.show()


# --- Figure A2: Non-dimensional separation length ---
fig_a2, ax_a2 = plt.subplots(figsize=(12, 8))

for mach_val in mach_values:
    data = results_by_mach[mach_val]
    
    h_l_sep = []
    sep_nd_vals = []
    
    for h, sl_nd, sep in zip(data['h_l'], data['sep_len_nonDim'], data['separates']):
        if sep and (mach_val, h) not in outliers_to_exclude:
            h_l_sep.append(h)
            sep_nd_vals.append(sl_nd)
    
    if h_l_sep:
        sorted_pairs = sorted(zip(h_l_sep, sep_nd_vals))
        h_sorted, sl_nd_sorted = zip(*sorted_pairs)
        
        ax_a2.plot(h_sorted, sl_nd_sorted,
                   color=mach_colors[mach_val],
                   marker=mach_markers[mach_val],
                   markersize=14,
                   linewidth=2.5,
                   markeredgecolor='black',
                   markeredgewidth=1,
                   label=f'M = {mach_val}')

ax_a2.set_xlabel('h/l', fontsize=26)
ax_a2.set_ylabel(r'$L_{sep} / L_{domain}$', fontsize=26)
ax_a2.set_title('Non-Dimensional Separation Length vs h/l', fontsize=34, fontweight='bold')
ax_a2.tick_params(labelsize=21)
ax_a2.legend(title='Mach Number', fontsize=18, title_fontsize=21, loc='best')
ax_a2.grid(True, alpha=0.3)
ax_a2.set_xlim([min(h_l_values) - 0.005, max(h_l_values) + 0.005])

plt.tight_layout()
plt.savefig('figure_a2_sepLength_nonDim_vs_hl.png', dpi=200, bbox_inches='tight')
plt.show()


# =============================================================================
# FIGURE B: Heatmap - Separation Occurrence Matrix (unchanged logic)
# =============================================================================

separation_matrix = np.zeros((len(h_l_values), len(mach_values)))

for i, h_l in enumerate(h_l_values):
    for j, mach_val in enumerate(mach_values):
        data = results_by_mach[mach_val]
        try:
            idx = data['h_l'].index(h_l)
            separation_matrix[i, j] = 1 if data['separates'][idx] else 0
        except ValueError:
            separation_matrix[i, j] = np.nan  # No data for this combination

fig_b, ax_b = plt.subplots(figsize=(10, 8))

cmap_binary = ListedColormap(['#2ECC71', '#E74C3C'])  # Green = No Sep, Red = Sep

im = ax_b.imshow(separation_matrix, cmap=cmap_binary, aspect='auto', vmin=0, vmax=1)

ax_b.set_xticks(np.arange(len(mach_values)))
ax_b.set_yticks(np.arange(len(h_l_values)))
ax_b.set_xticklabels([f'{m}' for m in mach_values], fontsize=21)
ax_b.set_yticklabels([f'{h:.2f}' for h in h_l_values], fontsize=21)

# Minor ticks for grid
ax_b.set_xticks(np.arange(-0.5, len(mach_values), 1), minor=True)
ax_b.set_yticks(np.arange(-0.5, len(h_l_values), 1), minor=True)
ax_b.grid(which='minor', color='black', linestyle='-', linewidth=2)
ax_b.tick_params(which='minor', length=0)

# Cell annotations
for i in range(len(h_l_values)):
    for j in range(len(mach_values)):
        value = separation_matrix[i, j]
        text = "Sep" if value == 1 else "No Sep"
        ax_b.text(j, i, text, ha='center', va='center',
                  fontsize=16, fontweight='bold', color='white')

ax_b.set_xlabel('Mach Number', fontsize=21)
ax_b.set_ylabel('h/l', fontsize=21)
ax_b.set_title('Flow Separation Occurrence Map', fontsize=32, fontweight='bold')

cbar = plt.colorbar(im, ax=ax_b, ticks=[0.25, 0.75])
cbar.ax.set_yticklabels(['No Separation', 'Separation'], fontsize=18)

plt.tight_layout()
plt.savefig('figure_b_separation_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()


# =============================================================================
# FIGURE C: First Separation Location vs h/l (your original Figure A, preserved)
# =============================================================================

fig_c, ax_c = plt.subplots(figsize=(12, 8))

for mach_val in mach_values:
    data = results_by_mach[mach_val]
    
    h_l_sep = []
    x_first_vals = []
    
    for h, xf, sep in zip(data['h_l'], data['x_sep_first'], data['separates']):
        if sep and (mach_val, h) not in outliers_to_exclude:
            h_l_sep.append(h)
            x_first_vals.append(xf)
    
    if h_l_sep:
        sorted_pairs = sorted(zip(h_l_sep, x_first_vals))
        h_sorted, xf_sorted = zip(*sorted_pairs)
        
        ax_c.plot(h_sorted, xf_sorted,
                  color=mach_colors[mach_val],
                  marker=mach_markers[mach_val],
                  markersize=14,
                  linewidth=2.5,
                  markeredgecolor='black',
                  markeredgewidth=1,
                  label=f'M = {mach_val}')

ax_c.set_xlabel('h/l', fontsize=26)
ax_c.set_ylabel('First Separation Location, x [m]', fontsize=26)
ax_c.set_title('First Separation Point Location vs h/l', fontsize=34, fontweight='bold')
ax_c.tick_params(labelsize=21)
ax_c.legend(title='Mach Number', fontsize=18, title_fontsize=21, loc='best')
ax_c.grid(True, alpha=0.3)
ax_c.set_xlim([min(h_l_values) - 0.005, max(h_l_values) + 0.005])

plt.tight_layout()
plt.savefig('figure_c_first_sep_location_vs_hl.png', dpi=200, bbox_inches='tight')
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

h_l_x_xNonDim = (ds_by_case["h_l_x_Mach_1.5"]["X"] - ds_by_case["h_l_x_Mach_1.5"]["X"][0])  / (ds_by_case["h_l_x_Mach_1.5"]["X"][-1] - ds_by_case["h_l_x_Mach_1.5"]["X"][0])
h_l_4_xNonDim = ds_by_case["h_l_0.04_Mach_1.5"]["X"] / np.max(ds_by_case["h_l_0.04_Mach_1.5"]["X"])

h_l_x_y = ds_by_case["h_l_x_Mach_1.5"]["Y"] 
h_l_4_y = ds_by_case["h_l_0.04_Mach_1.5"]["Y"] 


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




# Load your CFD data

# Define rotor parameters
x = ds_by_case["h_l_0.04_Mach_1.5"]["X"]
y = h_l_4_y
p = h_l_4_P
u_inf = np.mean(ds_by_case_inlet["h_l_0.04_Mach_1.5"]["U"].data)

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
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\Axial Force Calculations\axialForce_comparison_table.png',
    title="Axial Force [N]\n(h/l = Optimal versus 0.02-0.09, varying Mach Number)"
)


generate_torque_table_mach(
    df_comparisonNorm, 
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\Axial Force Calculations\axialForce_comparison_Norm_table.png',
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
    x_hlx = x[h_l_x_key]
    p_hlx = P[h_l_x_key]
    y_hlx = y[h_l_x_key]
    
    
    x_hlx_norm = (x_hlx - x_hlx[0]) / (x_hlx[-1] - x_hlx[0])
    
    ax.plot(x_hlx_norm, p_hlx, 'k-', linewidth=2, label='h/l = x')
    
    # Plot all h_l cases
    colors = plt.cm.viridis(np.linspace(0, 1, len(h_l_values)))
    
    for j, h_l in enumerate(h_l_values):
        h_l_key = f"h_l_{h_l}_Mach_{mach}"
        
        x_hl = x[h_l_key]
        p_hl = P[h_l_key]
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
# Usage
# =============================================================================

# Single plot: Force vs h/l
generate_axial_force_plot_mach(
    df_comparison, 
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\axialForce_vs_hl_plot.png',
    title="Axial Force vs h/l\n(Varying Mach Number)",
    ylabel="Axial Force [N/m]",
    show_optimal=True
)

# Normalized version
generate_axial_force_plot_mach(
    df_comparisonNorm, 
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\axialForceNorm_vs_hl_plot.png',
    title="Axial Force per Unit Length vs h/l\n(Varying Mach Number)",
    ylabel="Axial Force per Unit Length [N/m]",
    show_optimal=True
)

# Dual panel plot showing both perspectives
generate_axial_force_plot_dual_mach(
    df_comparison,
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\axialForce_dual_plot.png',
    title="Axial Force Trends",
    ylabel=r"Axial Force / $P_{static}$"
)



   
#%%

   

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =============================================================================
# Helper function to create DataFrame from your dictionaries
# =============================================================================







# Step 1: Create DataFrame from your dictionaries
df_axial_force = create_axial_force_dataframe(
    tau_x_dict=tau_x,
    x_dict=x
)

# Step 2: Generate single plot (Force vs h/l) - shown in terminal
generate_axial_force_plot_mach(
    df=df_axial_force,
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\axialForce_vs_hl.png',
    title="Axial Force vs h/l\n(Varying Mach Number)",
    ylabel="Axial Force [N/m]",
    show_optimal=True,
    show_plot=True  # Shows in terminal
)

# Step 3: Generate dual plot WITH normalization by first_shock_pressures
generate_axial_force_plot_dual_mach(
    df=df_axial_force,
    first_shock_pressures = first_shock_pressures,  # Your dictionary
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\axialForce_dual_normalized.png',
    title="Axial Force Trends (Normalized)",
    ylabel="Axial Force / P_shock [m]",
    normalize_by_shock=True,  # Uses first_shock_pressures
    show_plot=True  # Shows in terminal
)

# Step 4: Generate dual plot WITHOUT normalization
generate_axial_force_plot_dual_mach(
    df=df_axial_force,
    first_shock_pressures = first_shock_pressures,
    output_path=r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\axialForce_dual.png',
    title="Axial Force Trends",
    ylabel="Axial Force [N/m]",
    normalize_by_shock=False,  # No normalization
    show_plot=True
)












        
#%%




"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    Using Small Perturbation Theory and comparing to RANS Simulations
#------------------------------------------------------------------------------------------------------------------------------------#
"""


# ====== SETUP ====== #
h_l_values = np.arange(0.02,0.1,0.01) # Defining the h_l values that we have 


#axialForceScaled = smallPertSolver(h_l_values, ds_by_case, plotting = True)



# Same as before, just one extra return value
axialForceScaled, axialForceScaled_SE = smallPertSolver_with_SE(
    h_l_values, ds_by_case, plotting=True
)




#%%
axialForceScaled, axialForceScaled_SE, axialForceScaled_combined = \
    smallPertSolver_combined(h_l_values, ds_by_case, plotting=False)


#%%



# ====== SETUP ====== #
h_l_values = np.arange(0.02,0.1,0.01) # Defining the h_l values that we have 


# Plotting the results # 
plot_scaled_axialForce_vs_hl(axialForceScaled, h_l_values)
        

#%%

# Understanding how to optimize using small perturbation theory for maximum Axial force(expected torque) #
x0 = 2
x = np.linspace(0,1,1000)
h = 5 

y = h* np.exp((-x/x0)**2)

plt.plot(x,y)





#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                        Creating shock-expansion theory to understand the physics at the wall 
#------------------------------------------------------------------------------------------------------------------------------------#
"""

# Analyze all geometries at once
results = analyze_geometries(x, y, p_inf=1.0)  # p_inf in your units

# Get just the first shock pressures as a dictionary
first_shock_pressure_ratios = get_first_shock_pressures(results)

# getting actual first shock pressure # 
first_shock_pressures = {}


for key,_ in P_inlet.items(): 
    first_shock_pressures[key] = first_shock_pressure_ratios[key] * np.mean(P_inlet[key])









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
    
    
   





   