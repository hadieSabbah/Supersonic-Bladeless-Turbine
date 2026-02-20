


### Code to generate the 2D geometry for wind tunnel testing ###
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path

### ================ DEFINING GEOMETRY PARAMETERS ================ ###
n_points = 1500          # Number of points 

h_l = 0.02               # h/l ratio
l_2D = 0.1  # Length of 2D section in meters


# For the 3D cross-section, define wavelength based on circumference
n_waves_circumference = 5  # Number of complete waves around the circle
amplitude = h_l * l_2D
h = amplitude
nozzle_dim = 50
avg_radius =  (50/2 - amplitude) / 1000            # Average radius in m  (This shouuld be the radius of the inner nozzle)

print(f"Average radius: {avg_radius:.2f} m")
print(f"Amplitude (h): {amplitude:.4f} m")
print(f"h/l ratio: {h_l}")
print(f"Number of waves: {n_waves_circumference}")



### ================ 2D GEOMETRY (Flat plate) ================ ###
n_waves_2D = 2.5  # Number of waves on flat plate
N = 1 # Number of 2D recirculation bubbles
lamb_2D = l_2D / (2*N+1) * 2  # Wavelength for 2D case
amplitude_2D = h_l * l_2D  # Amplitude for 2D case

x = np.linspace(0, l_2D, n_points)
y = amplitude_2D * np.sin(2 * np.pi * x / lamb_2D)




### ================ 3D GEOMETRY (Wavy cylinder cross-section) ================ ###
theta = np.linspace(0, 2 * np.pi, n_points)
s = avg_radius * theta # Arc Length

N_3D = n_waves_circumference 
lamb_3D = l_2D / (2*N_3D +1 ) *2
# THIS IS THE KEY FIX:
# n_waves_circumference controls how many peaks appear around the circle
R = avg_radius + amplitude * np.sin( (2 * np.pi * s / lamb_3D)) 

#np.sin(n_waves_circumference * theta)

# Convert to Cartesian coordinates
X = R * np.cos(theta)
Y = R * np.sin(theta)
Z = np.zeros(np.size(theta))

# Exporting the data points # 
points3D = np.column_stack([X,Z,Y])
np.savetxt(rf'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\02_CAD\9_Bladeless Turbine\V2\cyl_{h_l}.txt' , points3D, fmt = '%.6f')


### ================ PLOTTING RESULTS ================ ###
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2D sine wave
axes[0].plot(x , y , 'b', linewidth=1.5)
axes[0].set_ylabel("Y [m]")
axes[0].set_xlabel("X [m]")
axes[0].set_title("2D Wavy Cross-Section Profile")
axes[0].grid(True)
#axes[0].set_aspect('equal')

# 3D cross-section
axes[1].plot(X, Y , 'b', linewidth=1.5)
axes[1].set_xlabel("X [m]")
axes[1].set_ylabel("Y [m]")
axes[1].set_title(f"Wavy Cylinder Cross-Section\n({n_waves_circumference} waves, h/l = {h_l})")
axes[1].grid(True)
#axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()



#%%

### Code to generate 2D and 3D wavy geometries ###
import numpy as np 
import matplotlib.pyplot as plt

### ================ YOU SET THESE ================ ###
h_l = 0.02               # h/l ratio
l = 0.1                  # Wavelength (lambda) in meters
avg_radius = 0.023       # Average radius in meters (set by nozzle)
n_waves_3D = 10           # Number of waves around cylinder (integer)
n_points = 1500          # Number of points

### ================ CALCULATED FROM YOUR INPUTS ================ ###
amplitude = h_l * l      # Amplitude (same for 2D and 3D)

### ================ 2D GEOMETRY ================ ###
x = np.linspace(0, l, n_points)
y = amplitude * np.sin(2 * np.pi * x / l)

### ================ 3D GEOMETRY ================ ###
theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
R = avg_radius + amplitude * np.sin(n_waves_3D * theta)

X = R * np.cos(theta)
Y = R * np.sin(theta)
Z = np.zeros(np.size(theta))

### ================ PRINT SUMMARY ================ ###
print(f"h/l ratio:       {h_l}")
print(f"Wavelength (l):  {l} m")
print(f"Amplitude (h):   {amplitude} m")
print(f"Average radius:  {avg_radius} m")
print(f"Number of waves: {n_waves_3D}")

### ================ EXPORT FOR SOLIDWORKS ================ ###
points3D = np.column_stack([X, Z, Y])
np.savetxt(r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\02_CAD\9_Bladeless Turbine\V2\wavy_cylinder.txt', points3D, fmt='%.6f')

### ================ PLOTTING ================ ###
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x, y, 'b', linewidth=1.5)
axes[0].set_xlabel("X [m]")
axes[0].set_ylabel("Y [m]")
axes[0].set_title("2D Profile")
axes[0].grid(True)

axes[1].plot(X, Y, 'b', linewidth=1.5)
axes[1].set_xlabel("X [m]")
axes[1].set_ylabel("Y [m]")
axes[1].set_title(f"3D Cross-Section ({n_waves_3D} waves)")
axes[1].grid(True)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()

