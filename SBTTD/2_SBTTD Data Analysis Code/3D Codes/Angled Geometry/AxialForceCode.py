# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:41:42 2026

@author: hhsabbah
"""

### Computing torque from the 3D geometry by creating a function!# 
import xarray as xr
import numpy as np
from pathlib import Path
import sympy as sp
import tecplot as tp
import os 
import matplotlib.pyplot as plt


#%%
### Connecting to the session # 
tp.session.connect()


#%%
### Importing the case onto Tecplot ###
soln_dirc = Path(r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\01_CFD Simulations\18_3D Angled Wavy Project\7_Solutions\2_7_2026\angledWavy.bin")


"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                                Data importing
#------------------------------------------------------------------------------------------------------------------------------------#

"""

data = tp.data.load_tecplot(soln_dirc.as_posix())


# Extracting Values from Certain Zones # 
wavy_zone = data.zone("Wavy")


# All variable names in the dataset
var_names = [v.name for v in data.variables()]


#### Extracting Variables from Section Zone ####

# Build dict: {var_name: numpy_array}
data_dict = {}
for var in var_names:
    try:
        data_dict[var] = wavy_zone.values(data.variable(var)).as_numpy_array()
    except Exception as e:
        print(f"Skipping {var}: not found")


#%%
"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                    Using name space to access data quicker
#------------------------------------------------------------------------------------------------------------------------------------#

"""
from types import SimpleNamespace
data = SimpleNamespace(**data_dict)
    

#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                    Understanding the shape of the code
#------------------------------------------------------------------------------------------------------------------------------------#

"""
print("X shape:", data.X.shape)
print("Y shape:", data.Y.shape)
print("Z shape:", data.Z.shape)


print("\nX range:", data.X.min(), "to", data.X.max())
print("Y range:", data.Y.min(), "to", data.Y.max())
print("Z range:", data.Z.min(), "to", data.Z.max())


print("\nFirst 500 X values:", data.X[:500])
print("First 500 Y values:", data.Y[:500])
print("First 500 Z values:", data.Z[:500])


#%%

##### RE-ORGANIZING DATA STRUCTURE ####
# Find how many unique values along each axis define the grid
# Since X is the fastest-varying, find where X wraps around
diff_x = np.diff(data.X)
# X resets when diff goes negative (or when it jumps back to the start)
nx = np.argmax(diff_x < 0) + 1  # number of points before first reset
n_remaining = len(data.X) // nx  # total points in the other two dimensions

print(f"nx = {nx}, remaining = {n_remaining}, total = {nx * n_remaining}")

# Reshape to separate X from the rest, then find ny
temp = data.Y.reshape(-1, nx)  # each row is one X-sweep
# All values in a row should be the same Y if Y is the next-slowest index
# Find where Y changes
y_column = temp[:, 0]  # Y values at fixed X
diff_y = np.diff(y_column)
ny = np.argmax(diff_y < 0) + 1 if np.any(diff_y < 0) else np.argmax(np.abs(diff_y) > 1e-10 * np.abs(diff_y[diff_y != 0]).min()) + 1

nz = len(data.X) // (nx * ny)
print(f"Grid dimensions: nx={nx}, ny={ny}, nz={nz}")
print(f"Verify: {nx * ny * nz} == {len(data.X)}")
#%%

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                        Integrating the viscous and inviscid contribution
#------------------------------------------------------------------------------------------------------------------------------------#

"""

def plotterFunction2D(x,y,x_str,y_str,unit_x,unit_y):
    plt.title(f"{y_str} Vs {x_str}",fontsize = 24)
    plt.xlabel(f"{x_str} {unit_x} ",fontsize = 14)
    plt.ylabel(f"{y_str} {unit_y}",fontsize = 14)
    plt.plot(x, y)
    plt.grid()
    return

plotterFunction2D(data.X,data.P, "X","P","[m]","[N/m]")


ax = plt.axes(projection = '3d')
ax.plot_surface(data_dict["X"],data_dict["Y"],data_dict["Z"])
plt.show()
        