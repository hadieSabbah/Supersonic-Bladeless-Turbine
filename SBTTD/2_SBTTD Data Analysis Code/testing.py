# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 19:56:13 2026

@author: hhsabbah
"""
import numpy as np
import matplotlib.pyplot as plt

#%% Example 1 # 
# Finding the maximum value # 
x = np.linspace(-5,5,1000)
y = -x**2 # Simple quadratic equation 


# finding the discrete values #
dy_dx = np.gradient(y,x)
dy = np.gradient(y)
dx = np.gradient(x)


# finding the second derivative # 
dy_dx_2 = np.gradient(dy_dx,x)


# Finding the index of the max value #
dy_dx_abs = np.abs(dy_dx)
max_idx = np.argmin(dy_dx_abs)

# Finding whether this is the max or the min value # 
if dy_dx_2[max_idx +1] < 0:
    print("This is the max value")

else:
    print("This is the min value")

# Plotting results # 
plt.plot(x,y)
plt.scatter(x[max_idx],y[max_idx], color = "red")
plt.grid()
plt.title("X vs Y", fontsize = 24)
plt.xlabel("X")
plt.ylabel("y")
plt.show()


#%% Example 2: Multi-variable optimization # 


# Defining the variables #
# Create a TRUE 2D grid
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Define the surface
Z = Y * X**2 + 5 * Y * X
### Getting the max values ### 


# Finding the first derivative # 
fy = np.gradient(Z,y, axis = 0)
fx = np.gradient(Z,x, axis = 1)

# Finding the second derivative #
fyy = np.gradient(fy,y, axis = 0)
fxx = np.gradient(fx,x, axis  = 1)

# Computing the partial x partial y derivative # 
fxy = fy * fx


gradient_magnitude = np.sqrt(fx**2 + fy**2)
critical_idx = np.unravel_index(np.argmin(gradient_magnitude), Z.shape)

# Determining if we have a max or a min #
if fxx[critical_idx] < 0 and (fxx[critical_idx]*fyy[critical_idx] - fxy[critical_idx]**2) > 0:
    print("local maxima")
elif fxx[critical_idx] > 0 and (fxx[critical_idx]*fyy[critical_idx] - fxy[critical_idx]**2) > 0: 
    print("local minima")
elif (fxx[critical_idx]*fyy[critical_idx] - fxy[critical_idx]**2) < 0:
    print("The Function has a local saddle")
elif (fxx[critical_idx]*fyy[critical_idx] - fxy[critical_idx]**2) == 0:
    print("Inconclusive")






# Plotting results # 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X,Y,Z)
#ax.scatter(x[min_idx_y], y[min_idx_y], color = "red")
#ax.scatter(x[min_idx_x], y[min_idx_x], color = "blue")
plt.show()