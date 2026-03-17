# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:01:54 2026

@author: hhsabbah
"""

### Creating the planar nozzle ###

##### Functions ######
import numpy as np
import math
# ===================
# GAS PROPERTIES
# ===================
gamma = 1.4

# ===================
# FUNDAMENTAL FUNCTIONS
# ===================

def prandtl_meyer(M):
    """Compute ν (in radians) from Mach number."""
    # Your job: implement the P-M equation
    nu = np.sqrt((gamma+1)/(gamma-1)) * math.atan( np.sqrt(  (gamma-1)/(gamma+1)  * (M**2 -1))   ) - math.atan(np.sqrt(M**2 -1))
    return nu

def inverse_prandtl_meyer(nu):
    """
    Compute M from ν (in radians) using bisection.
    
    Since ν increases monotonically with M, bisection is reliable.
    """
    # Bounds: M must be > 1 (supersonic)
    # Upper bound: pick something large enough for your application
    M_low = 1.0001
    M_high = 10.0
    
    # Defining the tolerance # 
    tol = 1e-6
    
    while (M_high - M_low) > tol:
        M_mid = 0.5 * (M_low + M_high)
        nu_mid = prandtl_meyer(M_mid)
        
        if nu_mid < nu:
            M_low = M_mid
        else:
            M_high = M_mid
    
    return 0.5 * (M_low + M_high)




def mach_angle(M):
    """Compute μ (in radians) from Mach number."""
    return np.arcsin(1/M)

# ===================
# UNIT PROCESSES
# ===================

def interior_point(point1, point2):
    """
    Compute new point from two known points.
    point1 is on the C+ characteristic (coming from below)
    point2 is on the C- characteristic (coming from above)
    
    Each point is a dict: {'x': , 'y': , 'theta': , 'nu': , 'M': , 'mu': }
    
    Returns new point dict.
    """
    # Step 1: Use Riemann invariants to find θ and ν at new point
    # C+ (from point1): ν - θ = constant = ν1 - θ1
    # C- (from point2): ν + θ = constant = ν2 + θ2
    
    # Solve:
    # ν - θ = ν1 - θ1  --> call this K_minus
    # ν + θ = ν2 + θ2  --> call this K_plus
    # Adding: 2ν = K_minus + K_plus
    # Subtracting: 2θ = K_plus - K_minus
    
    # Step 2: Get M and μ from ν
    
    # Step 3: Find (x, y) where characteristics intersect
    # Use average angles for characteristic slopes
    
    pass










def wall_point(point1, wall_func, wall_slope_func):
    """
    Compute wall point from one interior point and wall geometry.
    C- characteristic from point1 hits the wall.
    
    Boundary condition: θ_wall = wall slope at that x location.
    """
    # Step 1: θ at wall = wall slope (flow tangent to wall)
    theta_wall = np.deg2rad(4) # degrees 
    
    # Step 2: Use C- invariant: ν + θ = ν1 + θ1
    # So: ν_wall = (ν1 + θ1) - θ_wall
    
    nu1 = 
    nu_wall = nu1 + theta1 - theta_wall
    # Step 3: Get M, μ from ν
    
    # Step 4: Find intersection of C- characteristic with wall curve
    
    pass

def centerline_point(point1):
    """
    Compute centerline point from one interior point.
    C+ characteristic from point1 hits the centerline (y = 0).
    
    Boundary condition: θ = 0 (symmetry).
    """
    # Step 1: θ = 0 at centerline
    
    # Step 2: Use C+ invariant: ν - θ = ν1 - θ1
    # So: ν_centerline = (ν1 - θ1) + 0 = ν1 - θ1
    
    # Step 3: Get M, μ from ν
    
    # Step 4: Find where C+ characteristic crosses y = 0
    
    pass

# ===================
# INITIAL DATA LINE
# ===================

def generate_initial_line(y_throat, theta_max, n_points):
    """
    Generate the initial data line for the expansion region.
    
    For a gradual expansion nozzle, this could be:
    - Points along a prescribed wall curve
    - Points along a straight characteristic from the throat
    
    Returns list of point dicts.
    """
    pass

# ===================
# MAIN MOC SOLVER
# ===================

def solve_nozzle(M_exit, theta_max, n_char, y_throat=1.0):
    """
    Main function to compute the nozzle flow field and wall shape.
    
    n_char: number of characteristic lines to use
    """
    
    # Step 1: Generate initial data
    
    # Step 2: March through kernel region (compute interior points)
    
    # Step 3: Compute straightening wall points
    
    # Step 4: Return wall coordinates and flow field
    
    pass















