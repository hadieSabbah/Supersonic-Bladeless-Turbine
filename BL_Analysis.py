#%%
## EXTRACTING THE BOUDNARY LAYER CHARACTERSTICS AUTOMATICALLY USING PYTECPLOT ###


#### Importing the required libraries ####
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import tecplot as tp
import scipy
import seaborn as sns 
from pathlib import Path
import os 
import glob
import re 
import sympy as sp 
from scipy.integrate import simps
from matplotlib import cm
import time


# Start Timer Command #
start_time = time.time()


# Directory #
dirc = "D:/Downloads/3_Research/Tecplot_Mach_data/12_Tecplot Wavy Data" #Computer Directory 
#dirc = "D:/1_Research Work/1_Bladeless Turbine Project/4_Code/2_Tecplot Automation Code/Tecplot Data" #Lab Directory 
high_amp = "High Amplitude"
low_amp = "Low Amplitude"


# Getting Dirctories of high amp and low amp #
high_amp_dirc = Path(dirc) / high_amp
low_amp_dirc = Path(dirc) / low_amp


# Getting files from each respective directory #
files_highAmp = list(high_amp_dirc.rglob("*.plt*"))
files_lowAmp = list(low_amp_dirc.rglob("*.plt*"))


# Converting to string variable #
files_highAmp_str = []
files_lowAmp_str = []


# Converting to string for high amp files # 
for file_highAmp in files_highAmp:
    file_highAmp = str(file_highAmp)
    file_highAmp = file_highAmp.replace("\\","*")
    file_highAmp = file_highAmp.replace("*","/")
    files_highAmp_str.append(str(file_highAmp))
    


# Converting to string for low amp files #
for file_lowAmp in files_lowAmp:
    file_lowAmp = str(file_lowAmp)
    file_lowAmp = file_lowAmp.replace("\\","*")
    file_lowAmp = file_lowAmp.replace("*","/")                                  
    files_lowAmp_str.append(str(file_lowAmp))


### Separting the surface and the volume tecplot files (Because ansys is weird) ###

# Pre-locating high amp variables #
files_highAmp_comb = []
files_highAmp_surf = []


# Pre-locating low amp variables #
files_lowAmp_comb = []
files_lowAmp_surf = []


# Separting High amp surf and comb .plt file directories (important to note that surf==> surface; comb ==> volume#
for n in range(len(files_highAmp_str)):
    if n % 2 == 0:
        files_highAmp_comb.append(files_highAmp_str[n])
    else:
        files_highAmp_surf.append(files_highAmp_str[n])
        
highAmp_dirc_zipped = list(zip(files_highAmp_comb,files_highAmp_surf))      


# Separting Low amp surf and comb .plt file directories (important to note that surf==> surface; comb ==> volume#
for n in range(len(files_lowAmp_str)):
    if n % 2 == 0:
        files_lowAmp_comb.append(files_lowAmp_str[n])
    else:
        files_lowAmp_surf.append(files_lowAmp_str[n])
        
lowAmp_dirc_zipped = list(zip(files_lowAmp_comb,files_lowAmp_surf))      
 




#% Low Amp post-processing #%%%

plt.close('all')  # Closing all graphs automatically 

# Pre-locating Variables #
mach_labels = []
line_data = []

for n in range(len(lowAmp_dirc_zipped)):
    
    # Pre-locating variables #
    
    all_H_all = []
    all_X_np = []
    
    ### Getting which Mach Number we are looping through ###
    pattern = r'Mach \d+(\.\d+)?'
    match = re.search(pattern,lowAmp_dirc_zipped[n][0])
    title_name = match.group(0)
    mach_labels.append(title_name)
    
    
    # Step 2: Delete the active frame (to remove the dataset and start fresh)
    page = tp.add_page()
    tp.delete_page(page)
    
    
    # Step 3: Add a new frame to reset the workspace
    f = tp.active_page().add_frame()  # Adds a new blank frame
    tp.active_frame().dataset.zone("curve Step 1 Incr 0")
    
    
    p = tp.active_page()
    
    
    # Load the Tecplot .dat file
    
    # Use this when using your personal computer #
    dataset = tp.data.load_tecplot(lowAmp_dirc_zipped[n][0])
    dataset_2 =  tp.data.load_tecplot(lowAmp_dirc_zipped[n][1])
    
    # Access the curve zone, get variables, and data #
    zone_curve = dataset.zone("curve Step 1 Incr 0")
 
    
    
    
    
    ### Temporary If else statement for Mach 0.9 until the rest of the simulations are complete
    if title_name == 'Mach 0.9':
        # Left Side #
        zone_left_wedge = dataset.zone("left_wedge Step 1 Incr 0")
        zone_left_flat = dataset.zone("left_flate Step 1 Incr 0")
        
        # Right Side #
        zone_right_wedge = dataset.zone("right_wedge Step 1 Incr 0")
        zone_right_flat = dataset.zone("right_flate Step 1 Incr 0")
 


##########  DATA ALLOCATION BEINGS    ##########


    # Pre-locating variables #
    x_np_all = []
    perp_cords_all = []
    y_np_all = []
    
    #### !!!!!!! IF ELSE statement here is temporary.... once all the simulations are done, we shall implement this onto all conditions not only Mach 0.9! !!!!!!!!!!####
    if title_name == 'Mach 0.9':
        # X and Y Coordinates #
        x_values_09 = np.concatenate((zone_left_flat.values('CoordinateX').as_numpy_array(), zone_curve.values('CoordinateX').as_numpy_array()))
        y_values_09 = np.concatenate((zone_left_flat.values('CoordinateY').as_numpy_array(), zone_curve.values('CoordinateY').as_numpy_array()))
        
        # For debugging purposes #
        x_values = x_values_09
        y_values = y_values_09
        
        
        # Naming variables as np for numpy #
        x_np = x_values
        y_np = y_values 
        
        # Sorting indicies and saving x and y variables #
        sorted_indices = np.argsort(x_np)
        x_np_all.append(x_np)
        y_np_all.append(y_np)
        
    else:
        #X and Y Coordinates # 
        x_values = zone_curve.values('CoordinateX')
        y_values =  zone_curve.values('CoordinateY')
        
        # Changing the fieldtype to a numpy array # 
        x_np = x_values.as_numpy_array()
        y_np = y_values.as_numpy_array()
        
        # Sorting indicies and saving x and y variables #
        sorted_indices = np.argsort(x_np)
        x_np_all.append(x_np[sorted_indices])
        y_np_all.append(x_np[sorted_indices])
 




     
########## Where data processing Begins ############


    # Sorting the indices #
    sorted_indices = np.argsort(x_np)
    x_sorted = x_np[sorted_indices]
    y_sorted = y_np[sorted_indices]
    
    
    # Compute the gradient (dy/dx) #
    dy_dx = np.gradient(y_sorted, x_sorted)
    
    dy_dx = np.nan_to_num(dy_dx , nan = 0.0) # !!!! Substituting NaN values with zero assuming that the cause is due to no change !!!!
   
    ### Computing Unit Vector ###
    
    # Defining Variables #
    dy = np.gradient(y_sorted)
    dx = np.gradient(x_sorted)
    
    # Computing Unit Vectors #
    unit_mag = np.sqrt(dx**2 + dy**2)
    unit_vector_x = dx/unit_mag
    unit_vector_y = dy/unit_mag
    
    # Computing Tan Vector and pre-locating variable #
    tan_vector = []
    tan_vector.append(np.array([unit_vector_x, unit_vector_y]))
    
    ### Desired line length in millimeters (1mm) ###
    line_length_mm = 10 / 1000  # 1mm in the same data units
    
    ### Create the plot ###
    fig, ax = plt.subplots()
    plt.plot(x_sorted, y_sorted, label='Curve')
    
    # Loop through each point and plot the tangent and perpendicular lines
    
    
    ### Pre-locating Variables ###
    perp_cords = []
    
    

  
    
    ###################### FINDING THE DISCRETE PERPENDICULAR LINES ######################
    for i in range(0, len(x_sorted), 10):  # Plot every 25th point for visibility
        
        
        
        # Tangent line slope and intercept #
        slope_tangent = dy_dx[i]
        intercept_tangent = y_sorted[i] - slope_tangent * x_sorted[i]
        
        
        # Compute the delta_x required to achieve the desired tangent line length #
        delta_x = line_length_mm / np.sqrt(1 + slope_tangent**2)
        
        
        # Create a limited x range for tangent and perpendicular lines #
        x_local_tangent = np.linspace(x_sorted[i] - delta_x, x_sorted[i] + delta_x, 10)
        y_tangent = slope_tangent * x_local_tangent + intercept_tangent
        
        
        # Plot tangent line #
        plt.plot(x_local_tangent, y_tangent, '--', color='red', alpha=1, label='Tangent' if i == 0 else "")
        
        
        # Perpendicular line: slope = -1 / tangent_slope #
        slope_perpendicular = -1 / slope_tangent if slope_tangent != 0 else 0
        intercept_perpendicular = y_sorted[i] - slope_perpendicular * x_sorted[i]
        
        
        # Compute the delta_x for the perpendicular line (it should have the same length) #
        delta_x_perpendicular = line_length_mm / np.sqrt(1 + slope_perpendicular**2)
        delta_y_perpendicular = slope_perpendicular * delta_x_perpendicular
        
        # Determine direction for the perpendicular line #
        if y_sorted[i] > y_sorted[i] - delta_y_perpendicular:
            multiplier = -1
        else:
            multiplier = 1
        
        
        # Generate the two endpoints of the perpendicular line #
        x_perpendicular = [x_sorted[i], x_sorted[i] - multiplier * delta_x_perpendicular]
        y_perpendicular = [y_sorted[i], y_sorted[i] - multiplier * delta_y_perpendicular]
        
        
        #Printing for sanity check
        #print(x_perpendicular)
        #print(y_perpendicular)
        
        # Start and end points of the #
        start_point = [x_sorted[i],y_sorted[i]]
        end_point = [x_sorted[i] - multiplier * delta_x_perpendicular,y_sorted[i] - multiplier * delta_y_perpendicular]
        
        
        # Getting Line data #
        perp_cords.append(zip(x_perpendicular,y_perpendicular))
        

        
        # Plot perpendicular line #
        ax.plot(x_perpendicular, y_perpendicular, '--', color='green', alpha=1, label='Perpendicular' if i == 0 else "")
            
      
        # Mark the point on the curve #
        ax.scatter(x_sorted[i], y_sorted[i], color='red', zorder=5)
        
        # Adding titles and labels #
        ax.set_title("Discrete method used", fontsize = 18)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m] ")
         
        # Set equal aspect ratio to make the lines visually perpendicular #
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
    num_of_points = 3500
    for cord_pair in perp_cords:
        line_data.append(tp.data.extract.extract_line(cord_pair, num_points= num_of_points))  # Adjust num_points if needed  

# Splitting the Line data into multiple arrays #
line_data = np.array(line_data)        
line_data_split = np.array_split(line_data, len(mach_labels))

# End Timer Command # 
end_time = time.time()  # End the timer
elapsed_time = end_time - start_time

print(f"Execution time: {elapsed_time:.4f} seconds")

#%% SAVING THE DATA #####

import numpy as np

# Save numerical and string data
np.savez_compressed(
    "combined_data.npz",
    mach_labels=np.array(mach_labels, dtype=object),  # Strings
    line_data=line_data,  # Numerical
    line_data_split=np.array(line_data_split, dtype=object),  # Numerical (list of arrays)
    files_highAmp_str=np.array(files_highAmp_str, dtype=object),  # Strings
    files_lowAmp_str=np.array(files_lowAmp_str, dtype=object),  # Strings
    highAmp_dirc_zipped=np.array(highAmp_dirc_zipped, dtype=object),  # Mixed (strings, tuples)
    lowAmp_dirc_zipped=np.array(lowAmp_dirc_zipped, dtype=object)  # Mixed (strings, tuples)
)

print("Numerical and string data saved successfully!")


#%% LOADING THE DATA #####
import numpy as np

# Load the combined data
data = np.load("combined_data.npz", allow_pickle=True)

# Access individual variables
mach_labels = data["mach_labels"]
line_data = data["line_data"]
line_data_split = data["line_data_split"]
files_highAmp_str = data["files_highAmp_str"]
files_lowAmp_str = data["files_lowAmp_str"]
highAmp_dirc_zipped = data["highAmp_dirc_zipped"]
lowAmp_dirc_zipped = data["lowAmp_dirc_zipped"]

print("Numerical and string data loaded successfully!")


#%%       

from scipy.integrate import quad
import numpy as np 
import math

### Getting boundary layer characteristics ###

def shape_factor_calc(y, u, U):
    # Normalize the velocity #
    f = u / U
    
    # Calculate displacement thickness (delta_star) using Simpson's rule #
    delta_star = simps(1 - f, y)
    
    # Calculate momentum thickness (theta) using Simpson's rule #
    theta = simps(f * (1 - f), y)
    
    # Shape factor #
    H = delta_star / theta
    
    return H, delta_star, theta




    
# Access variables by name #
desired_variables = ['CoordinateY', 'CoordinateX','X Component Velocity', 'Y Component Velocity','dx-velocity-dy','dy-velocity-dx','X Component Wall Shear']  # replace with your variable names
selected_vars = [dataset.variable(var_name) for var_name in desired_variables]

# Pre-locating all variables #
X_np_all = []
Y_np_all = []
H_all = []
theta_all = []
delta_star_all =[]
omega_all = []
U_x_all = []   
U_np_all = []
omega = []
wall_shear = []
wall_shear_all = []

for k in range(len(line_data_split)):
    # Assuming line_data is your list of Zone objects #
    #zone_uids = [zone.uid for zone in line_data_all[n]]
    
    
    # Ensure you're operating within an active frame #
    frame = tp.active_frame()
    
    # Get the dataset from the active frame #
    dataset = frame.dataset
    
 
    # Pre-locating Variables #
    H_add = []
    delta_star_add = []
    theta_add = []
    X_np = []
  


    for zone in line_data_split[k]:   
                       
        # Extract variable values for this zone #
        Y = selected_vars[0].values(zone)
        X = selected_vars[1].values(zone)
        U_x = selected_vars[2].values(zone)
        U_y = selected_vars[3].values(zone)
        du_dy = selected_vars[4].values(zone)
        dv_dx = selected_vars[5].values(zone)
        #!!!! wall_shear= selected_vars[6].values(zone)  Issue with extracting wall shear stress for some reason??? !!!!
 
        #Converting to Numpy array #
        Y_np = Y.as_numpy_array()
        X_pre = X.as_numpy_array()
        du_dy_np = du_dy.as_numpy_array()
        dv_dx_np = dv_dx.as_numpy_array()
       #!!!! wall_shear = wall_shear.as_numpy_array() !!!!
        
        
        X_np.append(X_pre[0])
        U_x_np = U_x.as_numpy_array()
        U_y_np = U_y.as_numpy_array()
        U_np = np.sqrt(U_x_np**2 + U_y_np**2)
        
        
        # Getting All Values for Y_np and X_np #
        Y_np_all.append(Y_np)
        X_np_all.append(X_np)
        # Storing Data Values in intended data structure #
        U_np_all.append(U_np)
        omega.append(du_dy_np - dv_dx_np)
        wall_shear_all.append(wall_shear)
        
        #Calculating the shape factor 
        [H,delta_star,theta] = shape_factor_calc(sorted(Y_np),sorted(U_np),sorted(U_np)[-1])
        H_add.append(H)
        delta_star_add.append(delta_star)
        theta_add.append(theta)
            

        
    
   # Y_np_all.append(Y_np)   
    #X_np_all.append(X_np)
    H_all.append(H_add)
    delta_star_all.append(delta_star_add)
    theta_all.append(theta_add)
    omega_all.append(omega)
    U_x_all.append(U_x_np)
   # U_np_all.append(U_np)
    
    delta_e = []
    omega_mean = []
    U_e = []
    #U_e_list = []
    U_e_list = [ [None for _ in range(len(line_data_split[1]))] for _ in range(len(mach_labels)) ]
    

    
 #%%   

### Splitting arrays ###
U_np_all = np.array_split(U_np_all, len(mach_labels))    
omega_z = np.array_split(omega, len(mach_labels)) #Splitting omega_z 


# !!!! for m in range(len(wall_shear_all)): !!!!
    # !!!! wall_shear_all[m] = np.delete(wall_shear_all[m], [np.arange(1, len(wall_shear_all[m])  ) ] ) !!!!!


# Splitting Wall shear in numpy #
# !!!! wall_shear_all = np.array_split(wall_shear_all,len(mach_labels)) !!!!


# Splitting coordinate arrays X_np and Y_np #
Y_np_all = np.array_split(Y_np_all,len(mach_labels))

X_np_all = np.array(X_np_all, dtype=object)  # Allow lists of different sizes
X_np_all = np.array_split(X_np_all, len(mach_labels))

 
#%%
from scipy.integrate import cumtrapz  # For cumulative numerical integration

# Creating a dictionary to store data for each Mach number
Y_np_datum_dict = {}

# Iterate through each Mach number
for i, mach_key in enumerate(mach_labels):
    # Initialize lists for Y_np_datum, omega_z, and omega_values
    mach_data = []
    omega_data = []
    omega_values = []  # To store cumulative integrals of Omega_z

    for k in range(len(Y_np_all[0])):
        # Process Y_np_datum (subtract first value from each row)
        y_row = Y_np_all[0][k] - Y_np_all[0][k][0]
        mach_data.append(y_row)

        # Process omega_z values (structured like Y_np_all)
        omega_row = omega_all[0][k]  # Assuming omega_all matches the structure of Y_np_all
        omega_data.append(omega_row)

        # Compute the cumulative integral of -Omega_z with respect to y_row
        omega_integral = cumtrapz(-omega_row, y_row, initial=0)  # Cumulative integration
        omega_values.append(omega_integral)  # Append the cumulative integral for this row

    # Store the processed data in the dictionary
    Y_np_datum_dict[mach_key] = {
        "Y_np_datum": mach_data,
        "omega_z": omega_data,
        "omega_values": omega_values  # Add the cumulative integrals
    }

# Validate by printing the structure of the dictionary
print(f"Data for {mach_labels[0]}: {Y_np_datum_dict[mach_labels[0]]}")




#%% Allocating results using dictionary #


# Creating a datum to accurately capture the boundary layer thickness delta_e #
y_datum = []

# Creating a range for the indices using numpy #
all_points = np.arange(0,len(X_np)) # All the points across the discretized section 

for point in all_points:
   
    # Extract the x and y coordinates for all the points 
    x_points = X_np[point]
    x_point_indicies = np.where(x_sorted == x_points)
    y_datum.append(abs(y_sorted[x_point_indicies]))
    
   
    
# Multiple omega thresholds
omega_thresholds = np.linspace(10, 4.25e4, 30)  # Thresholds for different Mach numbers
extra_thresholds = np.array([1e3 , 5e3 , 1e4])
omega_thresholds =  np.concatenate((extra_thresholds, omega_thresholds))
tolerance = 1000


# Initialize the dictionary to store results
results = {
    mach: {omega: {"delta_e": [], "U_e": [], "U_e_list": []} for omega in omega_thresholds}
    for mach in mach_labels
}

# Main computation loop for multiple thresholds
for k, mach in enumerate(mach_labels):  # Iterate over Mach labels
    for omega_thresh in omega_thresholds:  # Iterate over thresholds
        for n in range(len(y_datum)):
            for z in range(len(U_np)):
                if abs(omega_thresh - math.floor(omega_z[k][n][z])) <= tolerance:
                    delta_value = ( Y_np_all[k][n][z] - y_datum[n] * np.sign(Y_np_all[k][n][z]) ) * 1000
                    U_e_value = U_np_all[k][n][z]
                    U_e_list_value = U_np_all[k][n][0:z]

                    # Append to the dictionary under the current threshold
                    results[mach][omega_thresh]["delta_e"].append(delta_value)
                    results[mach][omega_thresh]["U_e"].append(U_e_value)
                    results[mach][omega_thresh]["U_e_list"].append(U_e_list_value)

                    break  # Stop searching after finding the first match



### Adding Wall_shear_all variable to the dictionary ###

for mach_index, mach_key in enumerate(mach_labels):  # Loop through each Mach number
    # Extract wall_shear_all values corresponding to the Mach number
    wall_shear_values = wall_shear_all[mach_index]  # Shape: 77 x 3500

    # Add wall_shear_all to the results dictionary at the Mach key level
    if mach_key not in results:
        results[mach_key] = {}  # Initialize if not present

    # Add wall_shear_all under the Mach key
    results[mach_key]["wall_shear_all"] = wall_shear_values







#%% Plotting delta_e vs X[mm] at multiple thresholds ###

############################# NEED TO DO FOR EACH CASE INSTEAD! USING FOR ONLY ONE CASE IS NOT IDEAL. CREATE A FOR LO

# Assuming `x_values_for_plot` and `results` are already defined #
# Define omega thresholds as integers for consistency #

mach_keys = mach_labels  # Mach number for which we are plotting

timesFont = {"fontname": "Times New Roman"}
cmap = cm.get_cmap('viridis', len(mach_labels))  # Get 'viridis' colormap with as many colors as mach_labels

# Loop through each Mach number and create a separate plot
for mach_key in mach_keys:
    # Create a new figure and axis for each Mach number
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counter = 0  # Reset counter for colormap
    
    # Loop through each omega threshold and add a plot for each
    for omega in omega_thresholds:
        # Extract delta_e for the current omega threshold
        delta_e = np.array(results[mach_key][omega]["delta_e"]).flatten()
        counter += 1
        
        # Define the X values for the plot
        x_values_for_plot = np.array(X_np[:len(delta_e)]).flatten() * 1000
        
        # Plot the current threshold
        ax.plot(
            x_values_for_plot,
            delta_e,
            label=f"$\\omega_{{z}} = {omega}$",
            linewidth=2,
            color=cmap(counter)
        )
    
    # Plot the wavy section curve
    ax.plot(x_sorted * 1e3, y_sorted * 1e3, color="red", label="Wavy Section")
    
    # Customize the plot
    ax.set_title("$\\delta [mm]$ Vs $X[mm]$ for Different $\\omega_{z}$ Thresholds \n" + mach_key, **timesFont, fontsize=25)
    ax.set_xlabel("$X \\: [mm]$", fontsize=18)
    ax.set_ylabel("$\\delta [mm]$", fontsize=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure for the current Mach number
    save_path = f"D:/Downloads/3_Research/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/Delta VS Omega_z/delta_Vs_X_combined_{mach_key}.png"
    plt.savefig(save_path)
    
    #Showing Plot
    plt.show()
    # Close the current figure to prevent overlap
    plt.close(fig)




#%% delta_e vs X[mm] where, the separation tool is being implemented  ######


import matplotlib.pyplot as plt
import numpy as np

# Assuming `x_values_for_plot`, `results`, `wall_shear_all`, `mach_labels`, `omega_thresholds`, `X_np`, `x_sorted`, and `y_sorted` are already defined

timesFont = {"fontname": "Times New Roman"}

# Loop through each Mach number
for mach_key in mach_labels:
    # Create a new figure and axis for each Mach number
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get wall_shear_all for the current Mach number
    wall_shear = np.array(results[mach_key]["wall_shear_all"])  # Shape: 77 x 3500

    cmap = cm.get_cmap('viridis', len(omega_thresholds))  # Get 'viridis' colormap
    counter = 0

    # Loop through each omega threshold and add a plot for each
    for omega in omega_thresholds:
        # Extract delta_e for the current omega threshold
        delta_e = np.array(results[mach_key][omega]["delta_e"]).flatten()

        # Apply the criterion: Set delta_e to NaN where wall_shear_all <= 0
        delta_e[wall_shear.flatten()[:len(delta_e)] <= 0] = np.nan

        counter += 1

        # Define the X values for the plot
        x_values_for_plot = np.array(X_np[:len(delta_e)]).flatten() * 1000

        # Plot the current threshold
        ax.plot(
            x_values_for_plot,
            delta_e,
            label=f"$\\omega_{{z}} = {omega}$",
            linewidth=2,
            color=cmap(counter)
        )

    # Plot the curve
    ax.plot(x_sorted * 1e3, y_sorted * 1e3, color="red", label="Wavy Section")

    # Customize the plot
    ax.set_title("$\\delta [mm]$ Vs $X[mm]$ for Different $\\omega_{z}$ Thresholds \n" +  mach_key, **timesFont, fontsize=25)
    ax.set_xlabel("$X \\: [mm]$", fontsize=18)
    ax.set_ylabel("$\\delta [mm]$", fontsize=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure for the current Mach number
    save_path = f"D:/Downloads/3_Research/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/Delta Vs Omega_z (Separated)/delta_Vs_X_combined_separation_{mach_key}.png"
    plt.savefig(save_path)
    
    # Showing the grpah #
    plt.show(fig)
    
    # Close the current figure to prevent overlap
    plt.close(fig)


    
    




#%% Visualizing where the point of interest is found across the curve for illustartion purposes. 
    
# Define a list of indices for points of interest
point_indices = [10 , 36 , 50]  # Replace with desired indices

# Color map #D
cmap = cm.get_cmap('viridis', len(point_indices))  # Get 'viridis' colormap with as many colors as mach_labels


# Create a figure for the curve and perpendicular lines
fig1, ax1 = plt.subplots(figsize = (9,2.5))
timesFont = {'fontname':'Times New Roman'}


ax1.plot(
    x_sorted,
    y_sorted,
    color="black",
    label="Curve",
    linewidth=2.5
)

# Starting with counter for color map indexing #
counter = 0 


# Loop over each index to compute and plot perpendicular lines
for point_index in point_indices:
    
    
    # Extract the x and y coordinates for the current point
    x_point = X_np[point_index]
    x_point_index = np.where(x_sorted == x_point)
    y_point = y_sorted[x_point_index]
    
    x_point_index_float = x_point_index[0].item()

  
    
    # Compute the slope of the tangent at the current point
    slope_tangent = dy_dx[x_point_index_float]
    intercept_tangent = y_point - slope_tangent * x_point

    # Compute the slope and intercept for the perpendicular line
    slope_perpendicular = -1 / slope_tangent if slope_tangent != 0 else 0
    intercept_perpendicular = y_point - slope_perpendicular * x_point

    # Compute the delta_x for the perpendicular line
    delta_x_perpendicular = line_length_mm / np.sqrt(1 + slope_perpendicular**2)
    delta_y_perpendicular = slope_perpendicular * delta_x_perpendicular
    
  


    # Ensure the perpendicular line is only above the curve
    if slope_perpendicular >= 0:  # Line going upwards (positive slope)
        x_perpendicular = [x_point, x_point + delta_x_perpendicular]
        y_perpendicular = [y_point, y_point + delta_y_perpendicular]
    else:  # Line going downwards (negative slope), adjust to stay above
        x_perpendicular = [x_point, x_point - delta_x_perpendicular]
        y_perpendicular = [y_point, y_point - delta_y_perpendicular]

       # Plot the specific point of interest
    ax1.scatter(x_point, y_point, color="red", zorder=5, label="Point" if point_index == point_indices[0] else "")

    # Plot the perpendicular line (restricted to top part only)
    ax1.plot(
        x_perpendicular,
        y_perpendicular,
        "--",
        color= cmap(counter),
        label= f"Point {counter +1 }" ,
        #if point_index == point_indices[0] else "",
        linewidth=1.5,
    )
    counter += 1

# Customize and finalize the plot for curve
ax1.set_title("Points of Interest", fontsize=18)
ax1.set_xlabel("X [m]", fontsize=16)
ax1.set_ylabel("Y [m]", fontsize=16)
ax1.set_aspect("equal")
ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
ax1.grid(True)


# Improving DPI #
plt.rcParams["figure.dpi"] = 350
plt.rcParams["savefig.dpi"] = 350

# Changing Font size of xlabel and ylabel #
plt.rc('xtick', labelsize = 15)
plt.rc('ytick',labelsize = 15)

# Saving the figure  #
plt.savefig('D:/Downloads/3_Research/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/Velocity Profiles/Visualization/curve_viz_points.png')


#%%
# Create a figure for velocity profiles
fig2, ax2 = plt.subplots(figsize = (6,4))
timesFont = {'fontname':'Times New Roman'}

mach_key = "Mach 2.5"
# Extract velocity profiles from the dictionary and plot them
for idx, point_index in enumerate(point_indices):
    # Extract the velocity profile (U_e_list) and corresponding Y values from the dictionary
    U_e_profile = results[mach_key][5e3]["U_e_list"][point_index]  # Replace with the appropriate Mach key and index
    Y_values = np.array(Y_np[:len(U_e_profile)]) * 1000  # Convert to mm

    # Plot velocity profile
    ax2.plot(
        Y_values,
        U_e_profile,
        label=f"Point {idx + 1}",
        linewidth=2,
        color = cmap(idx)
    )
    


# Customize and finalize the plot for velocity profiles
ax2.set_title("Velocity Profiles at Points of Interest", fontsize=18,)
ax2.set_ylabel("$U_{e} [m/s]$", fontsize=14)
ax2.set_xlabel("Y [mm]", fontsize=14)
ax2.legend(loc="upper left", bbox_to_anchor = (1,1) , fontsize=12)
ax2.grid(True)

# Adjust layout 
plt.tight_layout()

# Improving DPI #
plt.rcParams["figure.dpi"] = 350
plt.rcParams["savefig.dpi"] = 350

# Changing Font size of xlabel and ylabel #
plt.rc('xtick', labelsize = 15)
plt.rc('ytick',labelsize = 15)

# Saving the figure  #
plt.savefig('D:/Downloads/3_Research/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/Velocity Profiles/U(y) Profiles/velocity_profile.png')

# Display the plots
plt.show()




plt.show()


#%% U_infinity Vs Y[mm] across different omega values ###

# Assuming `results`, `Y_np`, `U_e_list`, `omega_thresholds`, and `mach_labels` are already defined

timesFont = {"fontname": "Times New Roman"}

# Loop through each Mach number
for mach_key in mach_labels:
    # Create a new figure and axis for each Mach number
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set up a colormap for distinct colors
    cmap = cm.get_cmap("viridis", len(omega_thresholds))
    
    # Loop through each omega threshold and plot U_e_list
    for idx, omega in enumerate(omega_thresholds):
        # Extract U_e_list for the current threshold
        U_e_list_values = results[mach_key][omega]["U_e_list"]
        
        # Loop through each velocity profile in U_e_list
        for n, U_e_profile in enumerate(U_e_list_values):
            if len(U_e_profile) > 0:  # Check if the profile is non-empty
                # Dynamically extract the corresponding Y values for the profile
                Y_values = np.array(Y_np[:len(U_e_profile)]) * 1000  # Convert to mm
                
                # Plot the current velocity profile
                ax.plot(
                    Y_values,
                    U_e_profile,
                    label=f"$\\omega_{{z}} = {omega}$" if n == 0 else None,
                    linewidth=1.5,
                    color=cmap(idx)  # Use colormap for color
                )
    
    # Customize the plot
    ax.set_title("$U_{\\infty} [m/s]$ Vs $Y[mm]$ for Different $\\omega_{z}$ Thresholds \n" + mach_key, **timesFont, fontsize=25)
    ax.set_xlabel("$Y \\: [mm]$", fontsize=14)
    ax.set_ylabel("$U_{\\infty} \\: [m/s]$", fontsize=14)
    
    # Adjust the legend with a larger font size
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    
    # Add a grid
    ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure for the current Mach number
    save_path = f"D:/Downloads/3_Research/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/U_Vs_Omega(Profile)/Ue_List_Vs_Y_multiple_thresholds_{mach_key}.png"
    plt.savefig(save_path)
    
    # Close the current figure to prevent overlap
    plt.close(fig)
#%%

########## Plotting Wall Shear Stress Vs Y[mm] ###########


fig, ax = plt.subplots(figsize = (6,4))
timesFont = {'fontname':'Times New Roman'}

cmap = cm.get_cmap('viridis', len(mach_labels))  # Get 'viridis' colormap with as many colors as mach_labels



# Plotting Wall Shear Vs X #
for k in range(len(mach_labels)):
        ax.plot(
            np.array(X_np) * 1000 , 
            results[mach_labels[k]]["wall_shear_all"],
            color = cmap(k) ,
            label = mach_labels[k],
            linewidth = 2
            ) 


# Plotting Wavy Section #
ax.plot(x_sorted * 1e3 , y_sorted * 1e4 - 1e2 , color = "red", label = "Wavy Section")    

# Plotting the line of separation #
ax.axhline(y = 0, color = "black", linestyle = "--", linewidth = 2, label = "Separation")    

# Customizing graph #
ax.set_title("$\\tau_{x}$ Vs $X[mm]$",**timesFont, fontsize = 25)
ax.set_xlabel("$X \\: [mm]$", fontsize = 14)
ax.set_ylabel("$\\tau_{x} [N/m^{2}]$", fontsize = 14)
ax.legend(loc = "upper left", bbox_to_anchor = (1,1))

# Improving DPI #
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.dpi"] = 450

# Changing Font size of xlabel and ylabel #
plt.rc('xtick', labelsize = 15)
plt.rc('ytick',labelsize = 15)

# Changing Graph look #
ax.grid(True)
plt.tight_layout()

# Saving the figure  #
plt.savefig('D:/Downloads/3_Research/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/Wall Shear Stress/shear_VS_X.png')


#Showing Plot
plt.show()




#%% Determing when the mean if equal to zero using numerical integration ##

# finding the mean omega_z #
mean_omega = simps(Y, omega_all)



#%% #################################### NEEEEDS MORE WORK############################################################################################################################
#%%%%%%%%%%%%%%%%% PLOTTING THE SHAPE FACTOR VS X-COORDINATE[M] %%%%%%%%%%%%%%%#
from matplotlib import cm

### Customizing Figure

fig, ax = plt.subplots(figsize = (6,4))
timesFont = {'fontname':'Times New Roman'}


cmap = cm.get_cmap('viridis', len(mach_labels))  # Get 'viridis' colormap with as many colors as mach_labels

for k in range(len(mach_labels)):
    
    # Plotting Shape Factor #
    ax.plot(X_np_all[k],H_all[k], color = cmap(k), label = mach_labels[k] )
    
      
   

# Plotting Curve #
ax.plot(x_sorted , y_sorted*100, color = "red", label = "Wavy Section")

    
# Name Labeling
ax.set_title( "Shape Factor Vs X[m]",**timesFont, fontsize = 25)
ax.set_xlabel("$X \\: [m]$", fontsize = 14)
ax.set_ylabel("$H$", fontsize = 14)


ax.grid(True)
ax.legend(loc = "upper left", bbox_to_anchor = (1,1) ) 

# Changing Font size of xlabel and ylabel
plt.rc('xtick', labelsize = 15)
plt.rc('ytick',labelsize = 15)

 
# Improving DPI 
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.dpi"] = 450
 
   
#Showing Plot
plt.tight_layout()
plt.show()
        
    
# Saving the figure 
fig_save_var_1 = 'D:/Downloads/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/ShapeFactorVsX_all.png'
plt.savefig(fig_save_var_1)
    
    


    
#%%%%%%%% PLOTTING THETA VS X-COORDINATE #%%%%%%%%%%%%%%%%#
    
    
### Customizing Figure
fig, ax = plt.subplots(figsize = (6,4))
timesFont = {'fontname':'Times New Roman'}

cmap = cm.get_cmap('viridis', len(mach_labels))  # Get 'viridis' colormap with as many colors as mach_labels

#### Plotting theta #####    
for n in range(len(mach_labels)): 
    ax.plot(X_np_all[n],np.array(theta_all[n]) * 1000, color = cmap(n), label = mach_labels[n] )
    

# Plotting Wavy Section #
ax.plot(x_sorted , y_sorted*100, color = "red", label = "Wavy Section")    
    
# Customizing graph #
ax.set_title("$\\theta$ Vs $X[m]$",**timesFont, fontsize = 25)
ax.set_xlabel("$X \\: [m]$", fontsize = 14)
ax.set_ylabel("$\\theta$ [mm]", fontsize = 14)
ax.legend(loc = "upper left", bbox_to_anchor = (1,1))


# Improving DPI #
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.dpi"] = 450

# Changing Font size of xlabel and ylabel #
plt.rc('xtick', labelsize = 15)
plt.rc('ytick',labelsize = 15)

# Changing Graph look #
ax.grid(True)
plt.tight_layout()

# Saving the figure  #
fig_save_var_2 = 'D:/Downloads/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/thetaVsX.png'
plt.savefig(fig_save_var_2)


#Showing Plot
plt.show()


#%%########################### PLOTTING DELTA STAR VS X[M]  #%%###########################


### Customizing Figure ###
fig, ax = plt.subplots(figsize = (6,4))
timesFont = {'fontname':'Times New Roman'}

cmap = cm.get_cmap('viridis', len(mach_labels))  # Get 'viridis' colormap with as many colors as mach_labels

    
    
for n in range(len(mach_labels)):
    plt.plot(X_np_all[n],np.array(delta_star_all[n])*1000, color = cmap(n), label = "$\delta$ graph")


# Plotting Wavy Section #
ax.plot(x_sorted , y_sorted*100, color = "red", label = "Wavy Section")   

# Name Labeling

ax.set_title("$\delta^{*}$ Vs $X[m]$",**timesFont, fontsize = 25)
ax.set_xlabel("$X \\: [m]$", fontsize = 14)
ax.set_ylabel("$\delta^{*}$ [mm]", fontsize = 14)


# Changing Font size of xlabel and ylabel
plt.rc('xtick', labelsize = 15)
plt.rc('ytick',labelsize = 15)


# Improving DPI 
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.dpi"] = 450


# Changing Graph look 
plt.tight_layout()
ax.grid(True)
    
    
# Saving the figure 
fig_save_var_3 = 'D:/Downloads/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/delta_starVsX.png'
plt.savefig(fig_save_var_3)


#Showing Plot
plt.show()



#%%%%%%%% PLOTTING Omega #%%%%%%%%%%%%%%%%#


### Customizing Figure ###
fig, ax = plt.subplots(figsize = (6,4))
timesFont = {'fontname':'Times New Roman'}

cmap = cm.get_cmap('viridis', len(mach_labels))  # Get 'viridis' colormap with as many colors as mach_labels

for k in range(len(Y_np_all[0])):
    for n in range(len(mach_labels)):
        ax.plot(np.array(x_values[n])*1000,omega_all[n][k], color = cmap(n))   

# Plotting Wavy Section #
ax.plot(x_sorted *55, y_sorted*1e9, color = "red", label = "Wavy Section")    
    
# Customizing graph #
ax.set_title("$\\omega_{z}$ Vs $Y[m]$",**timesFont, fontsize = 25)
ax.set_xlabel("$Y \\: [mm]$", fontsize = 14)
ax.set_ylabel("$\\omega_{z}$ [1/sec]", fontsize = 14)
ax.legend(loc = "upper left", bbox_to_anchor = (1,1))


# Improving DPI #
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.dpi"] = 450

# Changing Font size of xlabel and ylabel #
plt.rc('xtick', labelsize = 15)
plt.rc('ytick',labelsize = 15)

# Changing Graph look #
ax.grid(True)
plt.tight_layout()

# Saving the figure  #
fig_save_var_4 = 'D:/Downloads/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/omega.png'
plt.savefig(fig_save_var_4)


#Showing Plot
plt.show()

#%% Single Omega for a single spot #%%
fig = plt.plot(figsize = (6,4))
timesFont = {'fontname':'Times New Roman'}


plt.plot(np.array(Y_np_all[1]) * 1000, omega_all[1][1])
plt.title("$\\omega_{z}$ Vs $Y[mm]$",**timesFont, fontsize = 25)
plt.xlabel("$Y \\: [mm]$", fontsize = 14)
plt.ylabel("$\\omega_{z}$ [1/sec]", fontsize = 14)

plt.legend(loc = "upper left", bbox_to_anchor = (1,1) )

plt.grid(True)

# Improving DPI #
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.dpi"] = 450

# Saving Figure # 
fig_save_var_5 = 'D:/Downloads/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/omega_single.png'
plt.savefig(fig_save_var_5)

# Showing Figure #
plt.show()


#%% Omega and Velocity for a single Point #%%
fig, ax1 = plt.subplots(figsize = (6,4))
timesFont = {'fontname':'Times New Roman'}


ax1.plot(np.array(Y_np_all[1]) * 1000, omega_all[1][1],color = 'blue')
ax1.set_xlabel("$Y \\: [mm]$", fontsize = 14)
ax1.set_ylabel("$\\omega_{z}$ [1/sec]", fontsize = 14)
ax1.tick_params(axis = 'y', labelcolor = 'blue')


# Creating a second axes for velocity #
ax2 = ax1.twinx()
ax2.plot(np.array(Y_np_all[1])*1000, U_x_all[1],color = 'red')
ax2.set_ylabel("Velocity [m/s]", fontsize = 14)
ax2.set_xlabel("$Y[mm]",fontsize = 14)
ax2.tick_params(axis = 'y' , labelcolor = 'red')


# Adding title #
plt.title(" $\\omega_{z}$ and Velocity Vs Y[mm]", fontsize = 16)


# Customizing Graph #
plt.grid(True)


# Improving DPI #
plt.rcParams["figure.dpi"] = 450
plt.rcParams["savefig.dpi"] = 450


# Saving Figure # 
fig_save_var_6 = 'D:/Downloads/Tecplot_Mach_data/12_Tecplot Wavy Data/Graphs/Low Amplitude/omega_velocity_single.png'
plt.savefig(fig_save_var_6)


# Showing Figure #
plt.show()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%% HIGH AMPLITDUE GEOMETRY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#### 
    
    for n in range(len(highAmp_dirc_zipped)):
        
        ### Getting which Mach Number we are looping through ###
        pattern = r'Mach \d+(\.\d+)?'
        match = re.search(pattern,lowAmp_dirc_zipped[n][0])
        title_name = match.group(0)
        
        # Step 2: Delete the active frame (to remove the dataset and start fresh)
        page = tp.add_page()
        tp.delete_page(page)
        
        
        # Step 3: Add a new frame to reset the workspace
        f = tp.active_page().add_frame()  # Adds a new blank frame
        tp.active_frame().dataset.zone("curve Step 1 Incr 0")
        
        
        p = tp.active_page()
        
        
        # Load the Tecplot .dat file
        
        #This is using the lab computer
        #dataset = tp.data.load_tecplot('C:/Users/hhsabbah/Documents/01_Bladeless_Proj/01_CFD Simulations/01_2D Simulations/02_Variable Change Simulations/01_Mach Change Cases/01_High Amplitude Simulations/Mach 0.9/Tecplot Files/mach_09_comb.plt')
        #dataset_2 = tp.data.load_tecplot('C:/Users/hhsabbah/Documents/01_Bladeless_Proj/01_CFD Simulations/01_2D Simulations/02_Variable Change Simulations/01_Mach Change Cases/01_High Amplitude Simulations/Mach 0.9/Tecplot Files/mach_09_surf.plt')
        
        
        #Use this when using your personal computer 
        dataset = tp.data.load_tecplot(highAmp_dirc_zipped[n][0])
        dataset_2 =  tp.data.load_tecplot(highAmp_dirc_zipped[n][1])
        
        
        # Access the curve zone, get variables, and data
        zone_curve = dataset.zone("curve Step 1 Incr 0")
        
        
        for var in dataset.variables():
                print(f"Variable: {var.name}")
                data = zone_curve.values(var)
                print(data)  # This will print the variable's values



    #### Plotting and gathering values ####


        # X and Y coordinates 
        x_values = zone_curve.values('CoordinateX')
        y_values = zone_curve.values('CoordinateY')
        
        
        # Changing the fieldtype to a numpy array 
        x_np = x_values.as_numpy_array()
        y_np = y_values.as_numpy_array()
        
        
        # Sorting the indices
        sorted_indices = np.argsort(x_np)
        x_sorted = x_np[sorted_indices]
        y_sorted = y_np[sorted_indices]
        
        
        # Compute the gradient (dy/dx)
        dy_dx = np.gradient(y_sorted, x_sorted)
        
        
        # Desired line length in millimeters (1mm)
        line_length_mm = 3.5 / 1000  # 1mm in the same data units
        
        
        # Create the plot
        fig, ax = plt.subplots()
        plt.plot(x_sorted, y_sorted, label='Curve')
        
        # Loop through each point and plot the tangent and perpendicular lines
        
        
        #Pre-locating Variables
        perp_cords = []
        
        for i in range(0, len(x_sorted), 10):  # Plot every 25th point for visibility
            
        
            # Tangent line slope and intercept
            slope_tangent = dy_dx[i]
            intercept_tangent = y_sorted[i] - slope_tangent * x_sorted[i]
            
            
            # Compute the delta_x required to achieve the desired tangent line length
            delta_x = line_length_mm / np.sqrt(1 + slope_tangent**2)
            
            
            # Create a limited x range for tangent and perpendicular lines
            x_local_tangent = np.linspace(x_sorted[i] - delta_x, x_sorted[i] + delta_x, 10)
            y_tangent = slope_tangent * x_local_tangent + intercept_tangent
            
            
            # Plot tangent line
            plt.plot(x_local_tangent, y_tangent, '--', color='red', alpha=1, label='Tangent' if i == 0 else "")
            
            
            # Perpendicular line: slope = -1 / tangent_slope
            slope_perpendicular = -1 / slope_tangent if slope_tangent != 0 else 0
            intercept_perpendicular = y_sorted[i] - slope_perpendicular * x_sorted[i]
            
            
            # Compute the delta_x for the perpendicular line (it should have the same length)
            delta_x_perpendicular = line_length_mm / np.sqrt(1 + slope_perpendicular**2)
            delta_y_perpendicular = slope_perpendicular * delta_x_perpendicular
            
            # Determine direction for the perpendicular line
            if y_sorted[i] > y_sorted[i] - delta_y_perpendicular:
                multiplier = -1
            else:
                multiplier = 1
            
            
            # Generate the two endpoints of the perpendicular line
            x_perpendicular = [x_sorted[i], x_sorted[i] - multiplier * delta_x_perpendicular]
            y_perpendicular = [y_sorted[i], y_sorted[i] - multiplier * delta_y_perpendicular]
            
            
            #Printing for sanity check
            #print(x_perpendicular)
            #print(y_perpendicular)
            
            # Start and end points of the 
            start_point = [x_sorted[i],y_sorted[i]]
            end_point = [x_sorted[i] - multiplier * delta_x_perpendicular,y_sorted[i] - multiplier * delta_y_perpendicular]
            
            
            # Getting Line data
            perp_cords.append(zip(x_perpendicular,y_perpendicular))
            
           
            
            
        # Plot perpendicular line
        plt.plot(x_perpendicular, y_perpendicular, '--', color='green', alpha=1, label='Perpendicular' if i == 0 else "")
            
            
        # Mark the point on the curve
        plt.scatter(x_sorted[i], y_sorted[i], color='red', zorder=5)
        
        
        # Set equal aspect ratio to make the lines visually perpendicular
        ax.set_aspect('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
            
            
            

    ### PRELOCATING VARIABLES FOR LOOP ###
            # Pre-Locating Variable
        line_data = []
        for idx in range(0,len(perp_cords)):
            line_data.append(tp.data.extract.extract_line(list(perp_cords[idx]),num_points = len(perp_cords)))
        




    ### Getting boundary layer characteristics ###

        #Getting zone by its uid 
        def get_zone_by_uid(uid, dataset):
            for zone in dataset.zones():
                if zone.uid == uid:
                    return zone
            return None  # Return None if no matching zone is found
        
        
        def shape_factor_calc(y, u, U):
            # Normalize the velocity #
            f = u / U
            
            
            # Calculate displacement thickness (delta_star) using Simpson's rule #
            delta_star = simps(1 - f, y)
            
            
            # Calculate momentum thickness (theta) using Simpson's rule #
            theta = simps(f * (1 - f), y)
            
            
            # Shape factor #
            H = delta_star / theta
            
            
            return H, delta_star, theta
        
        
        
        
        # Access variables by name #
        desired_variables = ['CoordinateY', 'CoordinateX','X Component Velocity', 'Y Component Velocity','dx-velocity-dy','dy-velocity-dx']  # replace with your variable names
        selected_vars = [dataset.variable(var_name) for var_name in desired_variables]
        
        
        # Assuming line_data is your list of Zone objects #
        zone_uids = [zone.uid for zone in line_data]
        
        
        # Ensure you're operating within an active frame #
        frame = tp.active_frame()
        
        # Get the dataset from the active frame #
        dataset = frame.dataset
        
        # Pre-locating variables #
        H_all = []
        X_np = []
        theta_all = []
        delta_star_all =[]
        omega = []
        
        
        all_H_all = [] 
        
        for  zone_uid in zone_uids:
            
           zone = get_zone_by_uid(zone_uid, dataset)
           if zone:
               
               # Extract variable values for this zone #
               Y = selected_vars[0].values(zone)
               X = selected_vars[1].values(zone)
               U_x = selected_vars[2].values(zone)
               U_y = selected_vars[3].values(zone)
               du_dy = selected_vars[4].values(zone)
               dv_dx = selected_vars[5].values(zone)
               
               #Converting to Numpy array #
               Y_np = Y.as_numpy_array()
               X_pre = X.as_numpy_array()
               X_np.append(X_pre[0])
               U_x_np = U_x.as_numpy_array()
               U_y_np = U_y.as_numpy_array()
               
               U_np = np.sqrt(U_x_np**2 + U_y_np**2)
               
               du_dy_np =du_dy.as_numpy_array()
               dv_dx_np = dv_dx.as_numpy_array()
               
               
               #Calculating the shape factor 
               [H,delta_star,theta] = shape_factor_calc(sorted(Y_np),sorted(U_np),sorted(U_np)[-1])
               H_all.append(H)
               delta_star_all.append(delta_star)
               theta_all.append(theta)
               all_H_all.append(H_all)
               
               #Calculating Vorticity 
               omega.append(du_dy_np - dv_dx_np) 
       
           else:
              
            # Print the extracted list of zone uids
            print(zone_uids)
        
        
        # Print the variables
        for var in selected_vars:
            print(var.name)
        
                    
         
    
        
#%% ### PLOTTING THE SHAPE FACTOR VS X-COORDINATE[M] ####
        
        
        ### Customizing Figure
        fig = plt.figure(figsize = (6,4))
        timesFont = {'fontname':'Times New Roman'}
        
        
        # Changing Font size of xlabel and ylabel
        plt.rc('xtick', labelsize = 15)
        plt.rc('ytick',labelsize = 15)
        
        
        # Plotting Shape factor # 
        plt.plot(X_np,H_all)
        
        
        # Plotting Curve #
        plt.plot(x_sorted,y_sorted*100)
        
        # Name Labeling
        title_full_name_1 = "Shape Factor Vs X[m] " + title_name 
        plt.title(title_full_name_1,**timesFont, fontsize = 25)
        plt.xlabel("$X \\: [m]$", fontsize = 14)
        plt.ylabel("$H$", fontsize = 14)
        
        
        # Improving DPI 
        plt.rcParams["figure.dpi"] = 450
        plt.rcParams["savefig.dpi"] = 450
        
        
        # Changing Graph look 
        ax.set_aspect('equal')
        plt.grid(True)
        
        
        # Saving the figure 
        fig_save_var_1 = 'E:/1_Research Work/1_Bladeless Turbine Project/4_Code/2_Tecplot Automation Code/Graphs/High Amplitude/ShapeFactorVsX_' + title_name +".png"
        plt.savefig(fig_save_var_1)
        
        
        #Showing Plot
        plt.show()
        
        
        #### PLOTTING THETA VS X-COORDINATE ####
        
        
        ### Customizing Figure
        fig = plt.figure(figsize = (6,4))
        timesFont = {'fontname':'Times New Roman'}
        
        
        # Changing Font size of xlabel and ylabel
        plt.rc('xtick', labelsize = 15)
        plt.rc('ytick',labelsize = 15)
        
        
        # Plotting momentum thickness figure #
        m_to_mm = 1000 # conversion from meters to milimeters 
        plt.plot(X_np,np.array(theta_all)*m_to_mm)
        
    
        
        
        # Name Labeling
        title_full_name_2 = "Momentum Thickness Vs X [m] " + title_name 
        plt.title(title_full_name_2,**timesFont, fontsize = 25)
        plt.xlabel("$X \\: [m]$", fontsize = 14)
        plt.ylabel("$\\theta$", fontsize = 14)
        
        
        # Improving DPI 
        plt.rcParams["figure.dpi"] = 450
        plt.rcParams["savefig.dpi"] = 450
        
        
        # Changing Graph look 
        ax.set_aspect('equal')
        plt.grid(True)
        
        
        # Saving the figure 
        fig_save_var_2 = 'E:/1_Research Work/1_Bladeless Turbine Project/4_Code/2_Tecplot Automation Code/Graphs/High Amplitude/thetaVsX_' + title_name +".png"
        plt.savefig(fig_save_var_2)
        
        
        #Showing Plot
        plt.show()
        
        
        ### PLOTTING DELTA STAR VS X[M]  ###
        
        
        ### Customizing Figure
        fig = plt.figure(figsize = (6,4))
        timesFont = {'fontname':'Times New Roman'}
        
        
        # Changing Font size of xlabel and ylabel
        plt.rc('xtick', labelsize = 15)
        plt.rc('ytick',labelsize = 15)
        
        
        # Plotting figure 
        plt.plot(X_np,np.array(delta_star_all)*m_to_mm)
        
        
        # Name Labeling
        title_full_name_3 = "Delta Star Vs X [m] for " + title_name 
        plt.title(title_full_name_3,**timesFont, fontsize = 25)
        plt.xlabel("$X \\: [m]$", fontsize = 14)
        plt.ylabel("$\delta^{*}$", fontsize = 14)
        
        
        # Improving DPI 
        plt.rcParams["figure.dpi"] = 450
        plt.rcParams["savefig.dpi"] = 450
        
        
        # Changing Graph look 
        ax.set_aspect('equal')
        plt.grid(True)
        
        
        # Saving the figure 
        fig_save_var_3 = 'E:/1_Research Work/1_Bladeless Turbine Project/4_Code/2_Tecplot Automation Code/Graphs/High Amplitude/delta_starVsX_' + title_name +".png"
        plt.savefig(fig_save_var_3)
        
        
        #Showing Plot
        plt.show()
        
        #%% Plotting vorticity against the X-cooridnate ###
        
        # Voriticity Plotting #
        plt.plot(Y_np ,omega[1], label = "$\\omega_{x}$ graph")
        plt.title("$\\omega_{z}$ Vs X-Coordinate", fontsize = 25)
        plt.xlabel("Y - Coordinate", fontsize = 14)
        plt.ylabel("$\\omega_{z}$", fontsize = 14)
        plt.grid()
        
        #plt.plot(x_sorted , y_sorted  * 1e8, label = "Wavy Section")
        
        plt.legend()
        
        # Saving the figure #
        fig_save_var_4 = 'E:/1_Research Work/1_Bladeless Turbine Project/4_Code/2_Tecplot Automation Code/Graphs/High Amplitude/Omega Vs X/omega_' + title_name +".png"
        plt.savefig(fig_save_var_4)
        
        
        plt.show()
