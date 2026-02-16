
# Libraries Needed # 
import numpy as np
import pprint



"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                        Computing ALL Variables and putting them in a dictionary 
#------------------------------------------------------------------------------------------------------------------------------------#

"""



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



# This function gets all the quantities at the wall for analysis #

def variableImporterMasked(ds_by_case, min_l, max_l, mask_input = True):
    
    # Non-dimensional quantities #
    y_plus = {}
    
    # Wall shear stress #
    tau_x = {}
    tau_y = {}
    tau_separation = {}
    tau_separation_idx = {}
    
    # Geometry # 
    x = {}
    y = {}
    
    # Temperature #
    T = {}

    
    # Pressure #
    P = {}
    Px = {}
    Py = {}
    P0 = {}
    
    # Density #
    rho = {}

    
    # Enthalpy # 
    h = {}
    h0 = {}
    
    # Non-dim quantities # 
    mach = {}
    
    
    # Vorticity # 
    omega_z = {}
    
    # Velocity # 
    u = {}
    v = {}
    
    # Thermal Quantities #
    q_dot = {}
    
    # Turbulent Quantities # 
    mu_tur = {}
    
    # Loop through cases and extract data #
    for key in ds_by_case:
        # Non-dimensional quantities #
        y_plus[key] = ds_by_case[key]["Y_plus"].data
        
        # Wall shear stress #
        tau_x[key] = ds_by_case[key]["Tau_x"].data
        tau_y[key] = ds_by_case[key]["Tau_y"].data
        
        # Geometry #
        x[key] = ds_by_case[key]["X"].data
        y[key] = ds_by_case[key]["Y"].data
        
        # Temperature #
        T[key] = ds_by_case[key]["T"].data

        
        # Pressure #
        P[key] = ds_by_case[key]["P"].data
        P0[key] = ds_by_case[key]["P_total"].data
        Px[key] = ds_by_case[key]["P_x"].data
        Py[key] = ds_by_case[key]["P_y"].data
        
        # Density #
        rho[key] = ds_by_case[key]["R"].data

        
        # Non-dim quantities #
        mach[key] = ds_by_case[key]["M"].data
        
        
        # Vorticity #
        omega_z[key] = ds_by_case[key]["Vort_z"].data
        
        # Velocity #
        u[key] = ds_by_case[key]["U"].data
        v[key] = ds_by_case[key]["V"].data
        
        # Thermal Quantities #
        q_dot[key] = ds_by_case[key]["Qdot"].data
        
        # Turbulent Quantities #
        #mu_tur[key] = ds_by_case[key]["Mutur"].data
        
        
        if mask_input == True:
            ##### Apply mask #######
            mask = (x[key] >= min_l) & (x[key] <= max_l)
            
            # Geometry #
            x[key] = x[key][mask]
            y[key] = y[key][mask]
            
            # Wall shear stress #
            tau_x[key] = tau_x[key][mask]
            tau_y[key] = tau_y[key][mask]
            
            # Non-dimensional quantities #
            y_plus[key] = y_plus[key][mask]
            
            # Temperature #
            T[key] = T[key][mask]
 
            
            # Pressure #
            P[key] = P[key][mask]
            P0[key] = P0[key][mask]
            Px[key] = Px[key][mask]
            Py[key] = Py[key][mask]
            
            # Density #
            rho[key] = rho[key][mask]
    
            
            # Non-dim quantities #
            mach[key] = mach[key][mask]
            
            # Vorticity #
            omega_z[key] = omega_z[key][mask]
            
            # Velocity #
            u[key] = u[key][mask]
            v[key] = v[key][mask]
            
            # Thermal Quantities #
            q_dot[key] = q_dot[key][mask]
            
            # Turbulent Quantities #
            #mu_tur[key] = mu_tur[key][mask]
        
        # Compute Separation #
        first_index_tau = np.argmax(tau_x[key] < 0)
        tau_separation[key] = tau_x[key][first_index_tau]
        tau_separation_idx[key] = first_index_tau
    
    return (y_plus, tau_x, tau_y, tau_separation, tau_separation_idx, 
            x, y, T, P, Px, Py, P0, rho, mach, omega_z, u, v, q_dot)
            #, mu_tur)
            
    
    
    






"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                                Computing Reynolds Number
#------------------------------------------------------------------------------------------------------------------------------------#

"""


def ReCompute(T, U, rho, S =120, mu_ref = 1.78e-5, X_max = 0.1):
    
    # Pre-Allocating Reynolds number variable as a dictionary  # 
    Re = {}
    
    # Getting All Variables to compute Reynolds number # 
    for key in T:
        mu_ref = 1.78e-5 #Pa*s
        T_ind = T[key]
        T_ref = 300 # kelvin
        S = 120 #kelvin
        
        mu = (mu_ref * (T_ind/T_ref)**(1.5)) * ((T_ref + S)/ (T_ind + S))
        U_ind = U[key]
        rho_ind = rho[key]
        X = X_max
        Re[key] = (np.mean(rho_ind) * (np.mean(U_ind) ) * X ) / np.mean(mu) # this is just a test. Re wall is not a thing. Well, it is, but you use boundary layer thickness to compute that stuff not the length of the entire thing...
       
        
    ReMAX = global_max_from_dict(Re)
    return Re, ReMAX



"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                            Finding which cases have a higher Y-plus value 
#------------------------------------------------------------------------------------------------------------------------------------#

"""


def yplusThreshold(y_plus):
    for y_plus_val, y_plus_key,  in y_plus.items():
        y_plusMax = np.max(y_plus_val)
        if y_plusMax > 1:
            print("=="*35)
            pprint(fr"WARNING! $y^{{+}}$ is greater than 1 ({y_plusMax:.2f}) at Key ---> {y_plus_key}")
            print("=="*35)
            print("\n \n")
    return


