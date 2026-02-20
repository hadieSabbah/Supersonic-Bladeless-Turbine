### Code to generate the 2D geomtery to have it tested in the wind tunnel ####

import numpy as np 
import math 
import matplotlib.pyplot as plt
import os
from pathlib import Path



### ================ DEFINING 3D GEOMETRY PARAMETERS ================ ###
n_points = 1500 # Number of points 
avg_radius = 15 /1000 # Average radius in m
h_l = 0.02 # h/l value; change it if needed
l = 0.1
amplitude = h_l * l  # Amplitude in m
h = amplitude 
period_num = 10 # Number of periods you want in one... if 10 ---> 20 peaks / 10periods 
period = amplitude / period_num # Why 10? 


# Defining the current directory # 
current_dir = Path(__file__).resolve()
current_dir = current_dir.parent
current_dir = current_dir.as_posix()


# ================ EXPORTING THE X AND Y VALUES FOR THE 2D GEOMETRY ================ # 



# Defining the variables #
 
x = np.linspace(0,l,n_points)
B = 2 * np.pi
N_2D = 1 #number of reciruclation bubbles 
lamb = l / (2*N_2D +1 )*2


# Determining the length needed # 
period2D = (2*np.pi)/(B)
n_waves = 2.5 # CHANGE THAT BASED ON YOUR LIKING 
#l = n_waves * period2D
#print(f"The Length should be the following: {l} meters\n")


# Setting up the equation # 
y = h * np.sin(2*np.pi*x/lamb)
z = np.zeros(np.size(y))
#points2D = np.column_stack([x,y,z])
#np.savetxt(rf'{current_dir}\1_Geometry Coordiantes\2D_curve.txt',points2D, fmt = '%.6f')






# ================ EXPORTING THE X AND Y VALUES FOR THE 3D GEOMETRY  ================ # 

# Generate Angles from 0 to 2*pi # 
theta = np.linspace(0,2*np.pi, n_points)


# Calculate Radius w/ Sinusoidal variation # 
R = avg_radius + amplitude * np.sin(theta / period) 


# Convert to cartesion coordinates # 
X = R * np.cos(theta)
Y = R * np.sin(theta)
Z = np.zeros(np.size(theta))


# Writing matrix to import this into solidworks # 
points3D = np.column_stack([X,Y,Z])
#np.savetxt(rf'{current_dir}\1_Geometry Coordiantes\3D_h_l_{h_l}_amp_{amplitude}_period_{period}.txt' , points3D, fmt = '%.6f')




# ================ EXECUTING MACROS IN SOLIDWORKS ================ # 









# ================ PLOTTING RESULTS ================ # 

# Plotting 2D sine Wave #
plt.plot(x,y)
plt.ylabel("Y[mm]")
plt.xlabel("X[mm]")
plt.grid()
plt.show()

# Plotting Wavy Circle # 
plt.plot(X,Y)
plt.grid()
plt.title("X[mm] Vs Y[mm]")
plt.xlabel("X[mm]")
plt.ylabel("Y[mm]")
plt.show()

#%% New Code ##
import numpy as np 
import math 
import matplotlib.pyplot as plt


# Creating the geometry # 
num_points = 500
theta = np.linspace(0,2*np.pi,num_points)

# Defining the Radius # 
A = 0.5 #amplitude 
n_wave = 20
avg_radius = 15 # Average radius in mm


# Radius #
R = avg_radius + A * np.sin(theta*n_wave)
x = R * np.cos(theta)
y = R * np.sin(theta)
z = np.zeros(np.size(theta))


# Writing matrix to import this into solidworks # 
points3D = np.column_stack([X,Y,Z])
#np.savetxt(r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\07_Python Codes\01_Python Automation Codes\03_Post-Processing Code\8_Geometry Generator\1_Goemetry Coordiantes\3D_curve_points.txt',points3D, fmt = '%.6f')


# Plotting results # 
plt.plot(x,y)
plt.grid()
plt.title("X Vs Y")
plt.xlabel("X[mm]")
plt.ylabel("Y[mm]")
plt.show()
