import xarray as xr
import numpy as np
from pathlib import Path
import sympy as sp
import tecplot as tp
import os 


"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                    Data importing and exporting
#------------------------------------------------------------------------------------------------------------------------------------#

"""



# Automaticalyl changing the working directory # 
new_dirc = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\2_SBTTD Data Analysis Code"
os.chdir(new_dirc)



#%%

# Importing modules # 


from utils.parameterComputation import variableImporterMasked, ReCompute, yplusThreshold
from utils.dataload_util import assign_dir, bigImport, runSaver, runLoader, file_pathFinder, load_minfo_step_force
from utils.plotting import plotter, plotter_multi_all, plotter_multiPerCase, subplotter, plot_scaled_axialForce_vs_hl, subplotter_multiPerCase
from utils.models import analyze_geometries, get_first_shock_pressures, offsetGeomPoints, smallPertSolver, find_sepLength, max_min_finder,mach_vs_sepLength, smallPertSolver_with_SE, smallPertSolver_combined


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

ds_by_case,ds_by_case_quad, ds_by_case_inlet = runLoader() 




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
    
    


#%% Exporting Corodinates for the hoes ####



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
missing = find_missing_cases(ds_by_case, "P_total")
print(missing)  # ['M2.0_P0_0.5', 'M3.0_P0_1.2', ...]
    
    
    
#%% Testing code with Claude  ####


import numpy as np
import matplotlib.pyplot as plt




























def compute_boundary_layer_thickness(x_quad, y_quad, omega_z_quad, rho_quad,
                                     x_wall, tau_x_wall,
                                     x_min=0.0, x_max=0.1,
                                     num_stations=100,
                                     x_tolerance=0.001,
                                     vorticity_threshold=100.0,
                                     density_gradient_threshold=1000.0):
    """
    Compute boundary layer thickness based on vorticity criterion.
    
    Excludes:
    - Locations where flow is separated (tau_x < 0)
    - Locations where shocks are present (large density gradient)
    
    Parameters:
    -----------
    x_quad, y_quad, omega_z_quad, rho_quad : dict
        Quadrant (full domain) data
    x_wall, tau_x_wall : dict
        Wall data (x-locations and wall shear stress)
    x_min, x_max : float
        Range of x to analyze
    num_stations : int
        Number of x-stations to evaluate
    x_tolerance : float
        Tolerance for finding points near x-station
    vorticity_threshold : float
        Absolute vorticity threshold for BL edge
    density_gradient_threshold : float
        If |d(rho)/dy| exceeds this, consider it a shock
        
    Returns:
    --------
    delta : dict
        Boundary layer thickness (NaN where separated or shock)
    x_stations : ndarray
        x-locations where delta was computed
    separation_mask : dict
        Boolean array - True where flow is separated
    shock_mask : dict
        Boolean array - True where shock is detected
    """
    
    # Define x-stations
    x_stations = np.linspace(x_min, x_max, num_stations)
    
    # Output dictionaries
    delta = {}
    separation_mask = {}
    shock_mask = {}
    
    # Loop through each case
    for key in x_quad.keys():
        
        # Extract quad arrays
        x_arr = x_quad[key]
        y_arr = y_quad[key]
        omega_arr = omega_z_quad[key]
        rho_arr = rho_quad[key]
        
        # Extract wall arrays
        x_w = x_wall[key]
        tau_x_w = tau_x_wall[key]
        
        # Storage for this case
        delta_values = np.zeros(num_stations)
        is_separated = np.zeros(num_stations, dtype=bool)
        is_shock = np.zeros(num_stations, dtype=bool)
        
        # Loop through each x-station
        for i, x_loc in enumerate(x_stations):
            
            # ---------------------------
            # Step 1: Check for separation
            # ---------------------------
            # Find closest wall point to this x-station
            wall_idx = np.argmin(np.abs(x_w - x_loc))
            tau_x_local = tau_x_w[wall_idx]
            
            if tau_x_local < 0:
                # Flow is separated here
                delta_values[i] = np.nan
                is_separated[i] = True
                continue
            
            # ---------------------------
            # Step 2: Get points near x-station
            # ---------------------------
            mask = np.abs(x_arr - x_loc) < x_tolerance
            
            if np.sum(mask) < 5:
                delta_values[i] = np.nan
                continue
            
            # Sort by y
            y_local = y_arr[mask]
            omega_local = omega_arr[mask]
            rho_local = rho_arr[mask]
            
            sort_idx = np.argsort(y_local)
            y_sorted = y_local[sort_idx]
            omega_sorted = omega_local[sort_idx]
            rho_sorted = rho_local[sort_idx]
            
            # Wall location
            y_wall_local = y_sorted[0]
            
            # ---------------------------
            # Step 3: Check for shock
            # ---------------------------
            # Compute density gradient (d_rho/dy)
            dy = np.diff(y_sorted)
            d_rho = np.diff(rho_sorted)
            
            # Avoid division by zero
            dy[dy == 0] = 1e-10
            
            rho_gradient = np.abs(d_rho / dy)
            
            if np.max(rho_gradient) > density_gradient_threshold:
                # Shock detected at this x-station
                delta_values[i] = np.nan
                is_shock[i] = True
                continue
            
            # ---------------------------
            # Step 4: Find BL edge
            # ---------------------------
            abs_omega = np.abs(omega_sorted)
            below_threshold = abs_omega < vorticity_threshold
            
            if not np.any(below_threshold):
                y_edge = y_sorted[-1]
            else:
                edge_idx = np.argmax(below_threshold)
                y_edge = y_sorted[edge_idx]
            
            # ---------------------------
            # Step 5: Compute BL thickness
            # ---------------------------
            delta_values[i] = y_edge - y_wall_local
        
        # Store results
        delta[key] = delta_values
        separation_mask[key] = is_separated
        shock_mask[key] = is_shock
    
    return delta, x_stations, separation_mask, shock_mask







def plot_bl_edge_validation(x_quad, y_quad, omega_z_quad, rho_quad,
                            x_wall, y_wall, tau_x_wall,
                            x_min=0.0, x_max=0.1,
                            num_stations=50,
                            x_tolerance=0.001,
                            vorticity_threshold=100.0,
                            density_gradient_threshold=1000.0):
    """
    Plot boundary layer edge detection with separation and shock regions marked.
    """
    
    # Compute BL thickness
    delta, x_stations, separation_mask, shock_mask = compute_boundary_layer_thickness(
        x_quad, y_quad, omega_z_quad, rho_quad,
        x_wall, tau_x_wall,
        x_min=x_min, x_max=x_max,
        num_stations=num_stations,
        x_tolerance=x_tolerance,
        vorticity_threshold=vorticity_threshold,
        density_gradient_threshold=density_gradient_threshold
    )
    
    # Loop through each case
    for key in x_quad.keys():
        
        # Get wall geometry
        x_w = x_wall[key]
        y_w = y_wall[key]
        
        # Get results for this case
        delta_vals = delta[key]
        is_sep = separation_mask[key]
        is_shock = shock_mask[key]
        
        # Compute y_edge = y_wall + delta at each station
        # First, get wall y at each x-station
        y_wall_at_stations = np.zeros(len(x_stations))
        for i, x_loc in enumerate(x_stations):
            wall_idx = np.argmin(np.abs(x_w - x_loc))
            y_wall_at_stations[i] = y_w[wall_idx]
        
        y_edge = y_wall_at_stations + delta_vals
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot wall geometry
        ax.plot(x_w, y_w, 'k-', linewidth=2, label='Wall')
        
        # Plot valid BL edge points (not separated, not shock)
        valid = ~is_sep & ~is_shock & ~np.isnan(delta_vals)
        ax.plot(x_stations[valid], y_edge[valid], 'go', markersize=5, 
                label='BL Edge (valid)')
        
        # Plot separation regions
        if np.any(is_sep):
            ax.axvspan(x_stations[is_sep].min(), x_stations[is_sep].max(),
                       alpha=0.2, color='red', label='Separated region')
            # Also mark individual points
            ax.plot(x_stations[is_sep], y_wall_at_stations[is_sep], 'rx', 
                    markersize=8, label='Separated points')
        
        # Plot shock regions
        if np.any(is_shock):
            ax.plot(x_stations[is_shock], y_wall_at_stations[is_shock], 'b^',
                    markersize=8, label='Shock detected')
        
        # Formatting
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Boundary Layer Edge Detection - {key}\n'
                     f'(ω threshold = {vorticity_threshold}, '
                     f'dρ/dy threshold = {density_gradient_threshold})')
        ax.legend(loc='upper left')
        ax.set_xlim([x_min - 0.005, x_max + 0.005])
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{key}:")
        print(f"  Valid points:     {np.sum(valid)} / {num_stations}")
        print(f"  Separated points: {np.sum(is_sep)}")
        print(f"  Shock points:     {np.sum(is_shock)}")


#%%
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

# Run validation
plot_bl_edge_validation(
    x_quad, y_quad, omega_z_quad, rho_quad,
    x, y, tau_x,  # Wall data
    x_min=0.0,
    x_max=0.1,
    num_stations=50,
    x_tolerance=0.001,
    vorticity_threshold=100.0,
    density_gradient_threshold=1000.0  # Adjust based on your results
)





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






########### NEEEDS WORK ######################


    
#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                        Getting all Boundary Layer profiles using Total enthalpy method 
#------------------------------------------------------------------------------------------------------------------------------------#
""" 


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




temp_key = 'h_l_0.02_Mach_1.5'
mask_thresh_temp = 0.006
mask_thresh_temp_x = 0.01
mask_thresh_tempMin_x = 0

 
mask_temp_x =(x_quad[temp_key] > 0 ) & ( x_quad[temp_key] < mask_thresh_temp_x)
mask_temp = y_quad[temp_key] < mask_thresh_temp
mask_idx = np.where(mask_temp)[0]



# Apply mask to BOTH arrays
x_masked_temp = x_quad[temp_key][mask_temp]
y_masked_temp = y_quad[temp_key][mask_temp]
omega_z_masked_temp = omega_z_quad[temp_key][mask_idx]



# Finding the index when vorticity becomes zero # 
boolean_mask =  (0 < omega_z_masked_temp) & (omega_z_masked_temp < 45)
zero_idx = np.where(boolean_mask)[0]


print(zero_idx) 
plt.scatter(x_masked_temp[zero_idx], y_masked_temp[zero_idx])
plt.plot(x[temp_key],y[temp_key])
plt.xlim([0,0.1])
plt.grid()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

mask_thresh_temp = 0.006

for key in x.keys():
    # Mask and filter
    mask_temp = y_quad[key] < mask_thresh_temp
    mask_idx = np.where(mask_temp)[0]
    
    x_masked_temp = x_quad[key][mask_temp]
    y_masked_temp = y_quad[key][mask_temp]
    omega_z_masked_temp = omega_z_quad[key][mask_idx]
    
    boolean_mask = (0 < omega_z_masked_temp) & (omega_z_masked_temp < 100)
    zero_idx = np.where(boolean_mask)[0]
    
    x_scatter = x_masked_temp[zero_idx]
    y_scatter = y_masked_temp[zero_idx]
    
    # Bin and get minimum y per bin
    n_bins = 100
    x_bins = np.linspace(x_scatter.min(), x_scatter.max(), n_bins)
    bin_idx = np.digitize(x_scatter, x_bins)
    
    lower_curve = {}
    for i in np.unique(bin_idx):
        mask = bin_idx == i
        min_pos = np.argmin(y_scatter[mask])
        lower_curve[i] = {'x': x_scatter[mask][min_pos], 'y': y_scatter[mask][min_pos]}
    
    x_lower = np.array([lower_curve[k]['x'] for k in sorted(lower_curve)])
    y_lower = np.array([lower_curve[k]['y'] for k in sorted(lower_curve)])
    
    # Plot
    plt.scatter(x_lower, y_lower, s=10, c='red')
    plt.plot(x[key], y[key], 'k-')
    plt.xlim([0, 0.1])
    plt.title(key)
    plt.grid()
    plt.show()




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

#%% Plotting separation length versus Mach number only!
# Plot Lsep/Lwidth vs Mach Number with lines for each h/L

import numpy as np














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
"""
TEACHING POINT - Why organize into results_by_mach?
---------------------------------------------------
Your raw data is keyed by case name (e.g., "h_l_0.02_Mach_1.5"), but your plots
need data organized by Mach number (each Mach = one line on the plot).

This step pivots the data from:
    sep_length["h_l_0.02_Mach_1.5"] = 0.012
    sep_length["h_l_0.02_Mach_2.0"] = 0.008

To:
    results_by_mach[1.5] = {'h_l': [0.02, 0.03, ...], 'sep_len': [0.012, 0.015, ...]}
    results_by_mach[2.0] = {'h_l': [0.02, 0.03, ...], 'sep_len': [0.008, 0.010, ...]}

This is the same pattern you used before, just now pulling from find_sepLength outputs.
"""

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
"""
TEACHING POINT - Dimensional vs Non-Dimensional:
-------------------------------------------------
You have two options here:
    1. sep_len     → dimensional [m], good for seeing absolute bubble sizes
    2. sep_len_nonDim → L_sep / L_domain, good for comparing across geometries

I'm plotting BOTH so you can decide which tells the better story for your paper.
The non-dimensional version is typically preferred because it removes domain-length
dependence and makes the physics clearer.
"""

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
"""
TEACHING POINT - Why the heatmap still works the same:
------------------------------------------------------
The heatmap only cares about YES/NO separation, which is the same whether you
detect it from x_sep arrays or from sep_length > 0. The difference is that
find_sepLength is more robust because it:
    1. Handles boundary negatives (tau_x < 0 at domain edges)
    2. Drops pairs that touch the domain boundary (not real separation)
    3. Uses interpolated zero-crossings instead of raw indices
"""

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
"""
TEACHING POINT - Why keep this plot?
------------------------------------
Separation LENGTH tells you HOW BIG the bubble is.
First separation LOCATION tells you WHERE on the geometry it starts.
Both are important but answer different questions:
    - Length → "How severe is separation?"  (affects drag, heat transfer)
    - Location → "Where does separation start?" (for geometry optimization)
"""

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
    for i, key in enumerate(ds_by_case):
        color = cmap_mach(i)
        print(pstatic_list[i])
        print(mach)
        y_vals = [pivot_hl.loc[h_l, mach] for h_l in h_l_values]
        ax1.plot(h_l_values, y_vals / first_shock_pressures[key], 'o-', color=color, linewidth=2, 
                 markersize=8, label=f'M = {mach_numbers[i]}') # Divide by y_values to be able to non-dim the solution CHANGE HERE 
        
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

#%% TEST WITH CLAUDE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =============================================================================
# Helper function to create DataFrame from your dictionaries
# =============================================================================
def create_axial_force_dataframe(tau_x_dict, x_dict, mach_numbers=None, h_l_values=None):
    """
    Create a DataFrame from your dictionary structure for axial force analysis.
    
    Parameters
    ----------
    tau_x_dict : dict
        Dictionary of tau_x arrays {case_name: tau_x_array}
    x_dict : dict
        Dictionary of x arrays {case_name: x_array}
    mach_numbers : list, optional
        List of Mach numbers to extract (if None, extracts all)
    h_l_values : list, optional
        List of h/l values to extract (if None, extracts all)
        
    Returns
    -------
    df : pandas DataFrame
        DataFrame with columns: 'Mach', 'h/l', 'tau_h_l [N·m/m²]', 'tau_h_l_x [N·m/m²]', 'case_key'
    """
    import re
    
    data_rows = []
    
    # Get optimal tau_x (h/l = 0.05) for each Mach number
    tau_x_optimal = {}
    
    for key in tau_x_dict.keys():
        # Extract Mach and h/l from key
        match_mach = re.search(r'(?:mach|M|Mach)_?([\d.]+)', key)
        match_hl = re.search(r'h_l_([\d.]+)', key)
        
        if match_mach and match_hl:
            mach = float(match_mach.group(1))
            h_l = float(match_hl.group(1))
            
            # Calculate integrated axial force (trapezoidal integration)
            # Use np.trapezoid for NumPy >= 2.0, or scipy alternative
            try:
                tau_integral = np.trapezoid(tau_x_dict[key], x_dict[key])
            except AttributeError:
                # Fallback for older NumPy versions
                from scipy.integrate import trapezoid
                tau_integral = trapezoid(tau_x_dict[key], x_dict[key])
            
            # Store optimal value (h/l = 0.05)
            if abs(h_l - 0.05) < 1e-6:
                tau_x_optimal[mach] = tau_integral
            
            data_rows.append({
                'Mach': mach,
                'h/l': h_l,
                'tau_h_l [N·m/m²]': tau_integral,
                'case_key': key  # Store the key for normalization lookup
            })
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Add optimal column
    df['tau_h_l_x [N·m/m²]'] = df['Mach'].map(tau_x_optimal)
    
    return df


# =============================================================================
# Plot generation function (Mach number version)
# =============================================================================
def generate_axial_force_plot_mach(df, output_path='axial_force_plot_mach.png', 
                                    title=None, ylabel="Axial Force [N/m]", 
                                    show_optimal=True, show_plot=True):
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
    show_plot : bool
        Whether to display the plot in terminal (default: True)
    
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
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Plot saved to: {output_path}")


def generate_axial_force_plot_dual_mach(df, first_shock_pressures,
                                         output_path='axial_force_dual_plot_mach.png', 
                                         title=None, ylabel="Axial Force [N/m]",
                                         normalize_by_shock=True, show_plot=True):
    """
    Generate a dual-panel plot:
    - Left: Axial force vs h/l (lines for each Mach number)
    - Right: Axial force vs Mach number (lines for each h/l)
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with axial force data (must include 'case_key' column)
    first_shock_pressures : dict
        Dictionary of first shock pressures {case_name: pressure}
    output_path : str
        Path to save the output image
    title : str, optional
        Custom title
    ylabel : str
        Label for y-axis
    normalize_by_shock : bool
        Whether to normalize by first shock pressure (default: True)
    show_plot : bool
        Whether to display the plot in terminal (default: True)
    """
    import re
    
    # Extract unique values
    mach_numbers = sorted(df['Mach'].unique())
    h_l_values = sorted(df['h/l'].unique())
    
    # Get optimal values
    tau_x_by_mach = df.groupby('Mach')['tau_h_l_x [N·m/m²]'].first().to_dict()
    
    # Create normalized DataFrame if requested
    if normalize_by_shock:
        df_plot = df.copy()
        df_plot['tau_h_l_norm'] = df_plot.apply(
            lambda row: row['tau_h_l [N·m/m²]'] / first_shock_pressures[row['case_key']] 
            if row['case_key'] in first_shock_pressures else row['tau_h_l [N·m/m²]'], 
            axis=1
        )
        df_plot['tau_h_l_x_norm'] = df_plot.apply(
            lambda row: row['tau_h_l_x [N·m/m²]'] / first_shock_pressures[row['case_key']] 
            if row['case_key'] in first_shock_pressures else row['tau_h_l_x [N·m/m²]'], 
            axis=1
        )
        
        # Update optimal values dictionary with normalized values
        tau_x_by_mach_norm = {}
        for mach in mach_numbers:
            # Find the case_key for h/l = 0.05 at this Mach
            mask = (df_plot['Mach'] == mach) & (np.abs(df_plot['h/l'] - 0.05) < 1e-6)
            if mask.any():
                tau_x_by_mach_norm[mach] = df_plot.loc[mask, 'tau_h_l_norm'].iloc[0]
        
        # Pivot tables with normalized values
        pivot_hl = df_plot.pivot(index='h/l', columns='Mach', values='tau_h_l_norm')
        pivot_mach = df_plot.pivot(index='Mach', columns='h/l', values='tau_h_l_norm')
        tau_x_by_mach_plot = tau_x_by_mach_norm
    else:
        # Pivot tables without normalization
        pivot_hl = df.pivot(index='h/l', columns='Mach', values='tau_h_l [N·m/m²]')
        pivot_mach = df.pivot(index='Mach', columns='h/l', values='tau_h_l [N·m/m²]')
        tau_x_by_mach_plot = tau_x_by_mach
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Colormaps
    cmap_mach = cm.get_cmap('viridis', len(mach_numbers))
    cmap_hl = cm.get_cmap('plasma', len(h_l_values))
    
    # =========================================================================
    # Left plot: Force vs h/l (one line per Mach number)
    # =========================================================================
    for i, mach in enumerate(mach_numbers):
        color = cmap_mach(i)
        y_vals = [pivot_hl.loc[h_l, mach] for h_l in h_l_values]
        
        ax1.plot(h_l_values, y_vals, 'o-', color=color, linewidth=2, 
                 markersize=8, label=f'M = {mach}')
        
        # Optimal reference (h/l = 0.05)
        if mach in tau_x_by_mach_plot:
            ax1.axhline(y=tau_x_by_mach_plot[mach], color=color, linestyle='--', 
                        alpha=0.4, linewidth=1.5)
    
    title_left = "Axial Force / P_shock vs h/l" if normalize_by_shock else "Axial Force vs h/l"
    ax1.set_title(title_left, fontsize=28, fontweight='bold')
    ax1.set_xlabel("h/l", fontsize=21)
    ax1.set_ylabel(ylabel, fontsize=18)
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.05, color='black', linestyle='--', linewidth=2, label='Optimal')
    ax1.legend(title="Mach", title_fontsize=16, fontsize=14, loc='best', ncol=2)
    
    # =========================================================================
    # Right plot: Force vs Mach number (one line per h/l)
    # =========================================================================
    # First plot optimal
    optimal_vals = [tau_x_by_mach_plot[mach] for mach in mach_numbers if mach in tau_x_by_mach_plot]
    if optimal_vals:
        ax2.plot(mach_numbers, optimal_vals, 'k-', linewidth=3, markersize=10, 
                 marker='*', label='Optimal (h/l=0.05)', zorder=10)
    
    # Then plot each h/l
    for i, h_l in enumerate(h_l_values):
        color = cmap_hl(i)
        y_vals = [pivot_mach.loc[mach, h_l] for mach in mach_numbers]
        ax2.plot(mach_numbers, y_vals, 'o-', color=color, linewidth=2, 
                 markersize=6, label=f'h/l = {h_l:.2f}')
    
    title_right = "Axial Force / P_shock vs Mach" if normalize_by_shock else "Axial Force vs Mach Number"
    ax2.set_title(title_right, fontsize=28, fontweight='bold')
    ax2.set_xlabel("Mach Number", fontsize=21)
    ax2.set_ylabel(ylabel, fontsize=18)
    ax2.tick_params(labelsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="h/l", title_fontsize=16, fontsize=14, loc='best', ncol=2)
    
    # Main title
    if title is None:
        title = "Axial Force Trends: h/l and Mach Number Effects"
        if normalize_by_shock:
            title += " (Normalized by First Shock Pressure)"
    fig.suptitle(title, fontsize=34, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Dual plot saved to: {output_path}")




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



   