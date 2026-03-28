
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
