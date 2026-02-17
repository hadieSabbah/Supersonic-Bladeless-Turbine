import tecplot as tp
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from sympy import symbols, init_printing, Integral, sqrt, pprint, simplify
from pygasflow import isentropic_solver
from scipy.optimize import fsolve
from utils.plotting import plotter, plotter_multi_all, plotter_multiPerCase, subplotter, plot_scaled_axialForce_vs_hl

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                               Boundary layer capturing algorithm 
#------------------------------------------------------------------------------------------------------------------------------------#
""" 


# Helper function to get the offset of the wave # 
def offsetGeomPoints(x, y, Bl_height_est, num_points=100):
    """
    Inputs:
        x: dict of x-coordinates for each Mach case
        y: dict of y-coordinates for each Mach case  
        Bl_height_est: estimated boundary layer height (offset distance)
        num_points: number of discretization points along the normal
    
    Outputs:
        x_offset: dict of 2D arrays (n_surface_pts, num_points)
        y_offset: dict of 2D arrays (n_surface_pts, num_points)
    """
    x_offset = {}
    y_offset = {}
    
    for key in x.keys():  # Note: keys() is a method, needs parentheses on dict, not here
        x_value = x[key]
        y_value = y[key]
        #n_pts = len(x_value)
        
        # Computing derivatives (note: gradient(y, x) for dy/dx)
        dy_dx = np.gradient(y_value, x_value)
        
        # Calculating angle of the tangent
        angles = np.arctan(dy_dx)
        
        # Getting the normal vectors (pointing "up" from surface)
        nx = -np.sin(angles)
        ny = np.cos(angles)
        
        # Create parameter t from 0 to 1 for interpolation
        t = np.linspace(0, 1, num_points)  # Shape: (num_points,)
        
        # Vectorized discretization using broadcasting
        x_offset[key] = x_value[:, None] + nx[:, None] * Bl_height_est * t[None, :]
        y_offset[key] = y_value[:, None] + ny[:, None] * Bl_height_est * t[None, :]
    
    return x_offset, y_offset
        
        
def BLVortZero(x_offset,y_offset):
    '''
    This function assumes that the BL edge is found when the vorticity has reached zero!
    INPUTS-----------------------------------
    input 1: x_offset (the dictionary with 2D x numpy arrays that are offset from the discretized wall surface)
    input 2: y_offset (the dictionary with 2D y numpy arrays that are offset from the discretized wall surface)
    
    outlet 1: Bl height (A dictionary with all the boundary layer heights at different discretized points across the wall surface)
    '''
    
    
    










#% calude code needs to change do one yourself:
    
"""
Shock-Expansion Theory for Sinusoidal Wall Geometries
======================================================

This module implements shock-expansion theory to predict pressure distributions
over wavy surfaces in supersonic flow. The theory treats the surface as a series
of infinitesimal compressions (oblique shocks) and expansions (Prandtl-Meyer fans).

Key Physics:
- Compressions occur where the wall turns INTO the flow (dθ/dx > 0)
- Expansions occur where the wall turns AWAY from the flow (dθ/dx < 0)
- For weak disturbances, changes are approximately isentropic except at shocks

Author: HS Research
Application: Supersonic flow separation over sinusoidal geometries
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Physical Constants and Data Classes
# =============================================================================

@dataclass
class FlowState:
    """Represents the thermodynamic state at a point in the flow."""
    Mach: float
    pressure: float      # Static pressure
    temperature: float   # Static temperature (optional, for completeness)
    
    def __repr__(self):
        return f"FlowState(M={self.Mach:.4f}, p={self.pressure:.4f})"


@dataclass 
class ShockExpansionResult:
    """Contains all results from shock-expansion analysis."""
    x: np.ndarray                    # Streamwise coordinates
    y: np.ndarray                    # Wall coordinates
    theta: np.ndarray                # Local wall angle (degrees)
    Mach: np.ndarray                 # Local Mach number distribution
    pressure_ratio: np.ndarray       # p/p_inf along the surface
    pressure: np.ndarray             # Absolute pressure if p_inf provided
    first_shock_pressure: float      # Pressure immediately after first shock
    first_shock_location: float      # x-location of first shock
    first_shock_Mach: float          # Mach number after first shock


# =============================================================================
# Fundamental Compressible Flow Functions
# =============================================================================

def prandtl_meyer(M: float, gamma: float = 1.4) -> float:
    """
    Compute the Prandtl-Meyer function ν(M).
    
    The Prandtl-Meyer function gives the angle (in radians) through which
    a sonic flow must expand isentropically to reach Mach number M.
    
    Parameters
    ----------
    M : float
        Mach number (must be >= 1)
    gamma : float
        Ratio of specific heats (default: 1.4 for air)
    
    Returns
    -------
    float
        Prandtl-Meyer angle in RADIANS
    
    Physical Interpretation:
        ν(M) represents the maximum turning angle for an isentropic expansion
        from M=1 to the given Mach number. For M→∞, ν_max ≈ 130.5° for air.
    """
    if M < 1.0:
        raise ValueError(f"Mach number must be >= 1 for supersonic flow, got M={M}")
    
    # Convenient groupings
    gp1 = gamma + 1
    gm1 = gamma - 1
    
    term1 = np.sqrt(gp1 / gm1)
    term2 = np.arctan(np.sqrt(gm1 / gp1 * (M**2 - 1)))
    term3 = np.arctan(np.sqrt(M**2 - 1))
    
    return term1 * term2 - term3


def inverse_prandtl_meyer(nu: float, gamma: float = 1.4, 
                          M_guess: float = 2.0) -> float:
    """
    Find Mach number given Prandtl-Meyer angle (inverse problem).
    
    This solves ν(M) = nu for M using numerical root finding.
    
    Parameters
    ----------
    nu : float
        Target Prandtl-Meyer angle in RADIANS
    gamma : float
        Ratio of specific heats
    M_guess : float
        Initial guess for Mach number
    
    Returns
    -------
    float
        Mach number corresponding to the given P-M angle
    """
    # Maximum possible P-M angle
    nu_max = (np.pi / 2) * (np.sqrt((gamma + 1) / (gamma - 1)) - 1)
    
    if nu < 0:
        raise ValueError("Prandtl-Meyer angle cannot be negative")
    if nu > nu_max:
        raise ValueError(f"Prandtl-Meyer angle {np.degrees(nu):.1f}° exceeds maximum {np.degrees(nu_max):.1f}°")
    
    # Solve ν(M) - nu = 0
    def residual(M):
        return prandtl_meyer(M, gamma) - nu
    
    # Mach number bounds: 1 (sonic) to ~50 (practical upper limit)
    M = brentq(residual, 1.0001, 50.0)
    return M


def isentropic_pressure_ratio(M: float, gamma: float = 1.4) -> float:
    """
    Compute isentropic pressure ratio p/p0.
    
    Parameters
    ----------
    M : float
        Local Mach number
    gamma : float
        Ratio of specific heats
    
    Returns
    -------
    float
        Static-to-total pressure ratio p/p0
    """
    return (1 + (gamma - 1) / 2 * M**2) ** (-gamma / (gamma - 1))


def max_deflection_angle(M1: float, gamma: float = 1.4) -> float:
    """
    Compute the maximum deflection angle for an attached oblique shock.
    
    Beyond this angle, the shock detaches and becomes a bow shock.
    
    Parameters
    ----------
    M1 : float
        Upstream Mach number
    gamma : float
        Ratio of specific heats
    
    Returns
    -------
    float
        Maximum deflection angle in RADIANS
    """
    # This is an approximation; the exact value requires solving a transcendental equation
    # For most purposes, this approximation works well
    from scipy.optimize import minimize_scalar
    
    def neg_theta(beta):
        """Negative of deflection angle (for maximization)."""
        sin_b = np.sin(beta)
        tan_b = np.tan(beta)
        if tan_b == 0:
            return 0
        numerator = 2 * (M1**2 * sin_b**2 - 1) / tan_b
        denominator = M1**2 * (gamma + np.cos(2 * beta)) + 2
        if denominator == 0:
            return 0
        return -np.arctan(numerator / denominator)
    
    mu = np.arcsin(1 / M1)  # Mach angle
    result = minimize_scalar(neg_theta, bounds=(mu, np.pi/2), method='bounded')
    return -result.fun


def oblique_shock_beta(M1: float, theta: float, gamma: float = 1.4,
                       weak: bool = True) -> float:
    """
    Solve for oblique shock wave angle β given upstream Mach and deflection angle.
    
    The θ-β-M relation is implicit and requires numerical solution:
    tan(θ) = 2 cot(β) * [M1²sin²(β) - 1] / [M1²(γ + cos(2β)) + 2]
    
    Parameters
    ----------
    M1 : float
        Upstream Mach number
    theta : float
        Flow deflection angle in RADIANS (must be positive for compression)
    gamma : float
        Ratio of specific heats
    weak : bool
        If True, return weak shock solution; if False, return strong shock
    
    Returns
    -------
    float
        Shock wave angle β in RADIANS
    
    Raises
    ------
    ValueError
        If theta exceeds the maximum deflection angle for the given Mach number
    """
    if theta <= 0:
        raise ValueError("Deflection angle must be positive for oblique shock")
    
    # Check if deflection angle exceeds maximum
    theta_max = max_deflection_angle(M1, gamma)
    if theta > theta_max:
        raise ValueError(f"No oblique shock solution exists for M1={M1:.2f}, θ={np.degrees(theta):.2f}°. "
                        f"Maximum is θ_max={np.degrees(theta_max):.2f}°")
    
    def theta_beta_M(beta):
        """θ-β-M relation residual."""
        sin_b = np.sin(beta)
        tan_b = np.tan(beta)
        
        numerator = 2 * (M1**2 * sin_b**2 - 1) / tan_b
        denominator = M1**2 * (gamma + np.cos(2 * beta)) + 2
        
        return np.arctan(numerator / denominator) - theta
    
    # Bounds for β: Mach angle (minimum) to 90° (maximum/normal shock)
    mu = np.arcsin(1 / M1)  # Mach angle
    
    try:
        if weak:
            # Weak shock: β between Mach angle and ~60-70°
            beta = brentq(theta_beta_M, mu + 1e-6, np.pi/2 - 1e-6)
        else:
            # Strong shock: would need different bounds (not typically used)
            beta = brentq(theta_beta_M, mu + 1e-6, np.pi/2 - 1e-6)
    except ValueError:
        raise ValueError(f"No oblique shock solution exists for M1={M1:.2f}, θ={np.degrees(theta):.2f}°. "
                        f"Deflection angle may exceed maximum.")
    
    return beta


def oblique_shock_relations(M1: float, beta: float, gamma: float = 1.4) -> Tuple[float, float, float]:
    """
    Compute flow properties downstream of an oblique shock.
    
    Parameters
    ----------
    M1 : float
        Upstream Mach number
    beta : float
        Shock wave angle in RADIANS
    gamma : float
        Ratio of specific heats
    
    Returns
    -------
    M2 : float
        Downstream Mach number
    p2_p1 : float
        Static pressure ratio p2/p1
    T2_T1 : float
        Static temperature ratio T2/T1
    """
    # Normal component of upstream Mach number
    M1n = M1 * np.sin(beta)
    
    # Deflection angle
    theta = np.arctan(2 * (M1n**2 - 1) / (np.tan(beta) * (M1**2 * (gamma + np.cos(2*beta)) + 2)))
    
    # Normal shock relations for the normal component
    M2n_sq = (M1n**2 + 2/(gamma-1)) / (2*gamma/(gamma-1) * M1n**2 - 1)
    M2n = np.sqrt(M2n_sq)
    
    # Downstream Mach number
    M2 = M2n / np.sin(beta - theta)
    
    # Pressure ratio across shock
    p2_p1 = 1 + 2*gamma/(gamma+1) * (M1n**2 - 1)
    
    # Temperature ratio across shock
    T2_T1 = p2_p1 * (2 + (gamma-1)*M1n**2) / ((gamma+1)*M1n**2)
    
    return M2, p2_p1, T2_T1


# =============================================================================
# Main Shock-Expansion Analysis Function
# =============================================================================

def shock_expansion_analysis(x: np.ndarray, y: np.ndarray,
                             M_inf: float, p_inf: float = 1.0,
                             gamma: float = 1.4,
                             theta_threshold: float = 0.01,
                             verbose: bool = False) -> ShockExpansionResult:
    """
    Perform shock-expansion theory analysis on a wall geometry.
    
    This function marches along the wall surface, applying:
    - Oblique shock relations where the flow compresses (wall turns into flow)
    - Prandtl-Meyer expansion relations where the flow expands (wall turns away)
    
    Parameters
    ----------
    x : np.ndarray
        Streamwise coordinates of wall (should be monotonically increasing)
    y : np.ndarray
        Normal coordinates of wall (defines the geometry)
    M_inf : float
        Freestream Mach number
    p_inf : float
        Freestream static pressure (default: 1.0 for normalized results)
    gamma : float
        Ratio of specific heats (default: 1.4 for air)
    theta_threshold : float
        Minimum angle change (degrees) to process; smaller changes are accumulated
        This prevents numerical issues with very small turning angles
    verbose : bool
        If True, print warnings about failed shock solutions
    
    Returns
    -------
    ShockExpansionResult
        Dataclass containing all analysis results
    
    Notes
    -----
    The analysis assumes the incoming flow is HORIZONTAL (parallel to x-axis).
    At the first point, if the wall has a non-zero slope, an initial shock or
    expansion is applied to turn the flow to match the wall.
    
    Sign Convention:
    - θ > 0: Wall slopes upward → initial compression (oblique shock)
    - θ < 0: Wall slopes downward → initial expansion (P-M fan)
    - dθ > 0: Flow turning INTO the wall (compression/shock)
    - dθ < 0: Flow turning AWAY from wall (expansion/P-M fan)
    """
    n_points = len(x)
    
    # Initialize output arrays
    Mach = np.zeros(n_points)
    pressure_ratio = np.zeros(n_points)  # p/p_inf
    
    # Compute wall geometry
    dy_dx = np.gradient(y, x)
    theta = np.arctan(dy_dx)  # Wall angle in radians
    theta_deg = np.degrees(theta)
    
    # Change in wall angle between adjacent points
    dtheta = np.diff(theta)  # Length n_points - 1
    
    # ==========================================================================
    # CRITICAL: Handle initial wall angle
    # The freestream flow is horizontal. If the wall has a non-zero initial
    # angle, we need an initial shock (θ₀ > 0) or expansion (θ₀ < 0).
    # ==========================================================================
    theta_initial = theta[0]  # Initial wall angle
    
    M_local = M_inf
    p_local = p_inf
    p0_local = p_inf / isentropic_pressure_ratio(M_inf, gamma)  # Total pressure
    nu_local = prandtl_meyer(M_inf, gamma)
    
    # Track first shock
    first_shock_found = False
    first_shock_pressure = None
    first_shock_location = None
    first_shock_Mach = None
    
    # Apply initial turning (flow must turn from horizontal to match wall)
    if theta_initial > 1e-6:  # Wall slopes upward → initial oblique shock
        try:
            beta = oblique_shock_beta(M_local, theta_initial, gamma)
            M_local, p_ratio_shock, _ = oblique_shock_relations(M_inf, beta, gamma)
            p_local = p_inf * p_ratio_shock
            nu_local = prandtl_meyer(M_local, gamma)
            p0_local = p_local / isentropic_pressure_ratio(M_local, gamma)
            
            # This IS the first shock
            first_shock_found = True
            first_shock_pressure = p_local
            first_shock_location = x[0]
            first_shock_Mach = M_local
            
        except ValueError as e:
            if verbose:
                print(f"Warning at initial point: {e}")
            # Use linearized theory for initial compression
            dp_p = gamma * M_inf**2 * theta_initial / np.sqrt(M_inf**2 - 1)
            p_local = p_inf * (1 + dp_p)
            dM = -M_inf * (1 + (gamma-1)/2 * M_inf**2) * theta_initial / np.sqrt(M_inf**2 - 1)
            M_local = max(1.01, M_inf + dM)
            nu_local = prandtl_meyer(M_local, gamma)
            p0_local = p_local / isentropic_pressure_ratio(M_local, gamma)
            
            first_shock_found = True
            first_shock_pressure = p_local
            first_shock_location = x[0]
            first_shock_Mach = M_local
            
    elif theta_initial < -1e-6:  # Wall slopes downward → initial P-M expansion
        nu_new = nu_local + abs(theta_initial)
        try:
            M_local = inverse_prandtl_meyer(nu_new, gamma)
            nu_local = nu_new
            p_local = p0_local * isentropic_pressure_ratio(M_local, gamma)
        except ValueError as e:
            if verbose:
                print(f"Warning at initial point: {e}")
    
    # Store first point
    Mach[0] = M_local
    pressure_ratio[0] = p_local / p_inf
    
    # Convert threshold to radians
    threshold_rad = np.radians(theta_threshold)
    
    # March along the wall (now using CHANGES in wall angle)
    accumulated_dtheta = 0.0
    
    for i in range(1, n_points):
        # Accumulate the angle change from previous point
        accumulated_dtheta += dtheta[i-1]
        
        # Only process if accumulated angle change is significant
        if abs(accumulated_dtheta) < threshold_rad and i < n_points - 1:
            Mach[i] = M_local
            pressure_ratio[i] = p_local / p_inf
            continue
        
        delta_theta = accumulated_dtheta
        accumulated_dtheta = 0.0
        
        if delta_theta > 1e-6:  # Compression (shock)
            try:
                beta = oblique_shock_beta(M_local, delta_theta, gamma)
                M_new, p_ratio_shock, _ = oblique_shock_relations(M_local, beta, gamma)
                
                p_local = p_local * p_ratio_shock
                M_local = M_new
                nu_local = prandtl_meyer(M_local, gamma)
                p0_local = p_local / isentropic_pressure_ratio(M_local, gamma)
                
                # Record first shock if not already found
                if not first_shock_found and p_ratio_shock > 1.01:
                    first_shock_found = True
                    first_shock_pressure = p_local
                    first_shock_location = x[i]
                    first_shock_Mach = M_local
                    
            except ValueError as e:
                if verbose:
                    print(f"Warning at x={x[i]:.4f}: {e}")
                # Use linearized theory
                dp_p = gamma * M_local**2 * delta_theta / np.sqrt(M_local**2 - 1)
                p_local = p_local * (1 + dp_p)
                dM = -M_local * (1 + (gamma-1)/2 * M_local**2) * delta_theta / np.sqrt(M_local**2 - 1)
                M_local = max(1.01, M_local + dM)
                nu_local = prandtl_meyer(M_local, gamma)
                
        elif delta_theta < -1e-6:  # Expansion (P-M fan)
            nu_new = nu_local + abs(delta_theta)
            try:
                M_local = inverse_prandtl_meyer(nu_new, gamma)
                nu_local = nu_new
                p_local = p0_local * isentropic_pressure_ratio(M_local, gamma)
            except ValueError as e:
                if verbose:
                    print(f"Warning at x={x[i]:.4f}: {e}")
        
        Mach[i] = M_local
        pressure_ratio[i] = p_local / p_inf
    
    # If no shock was found, use freestream values
    if not first_shock_found:
        first_shock_pressure = p_inf
        first_shock_location = x[0]
        first_shock_Mach = M_inf
    
    return ShockExpansionResult(
        x=x,
        y=y,
        theta=theta_deg,
        Mach=Mach,
        pressure_ratio=pressure_ratio,
        pressure=pressure_ratio * p_inf,
        first_shock_pressure=first_shock_pressure,
        first_shock_location=first_shock_location,
        first_shock_Mach=first_shock_Mach
    )


# =============================================================================
# Batch Processing Function for Multiple Geometries
# =============================================================================

def analyze_geometries(x_dict: Dict[str, np.ndarray], 
                       y_dict: Dict[str, np.ndarray],
                       M_inf_dict: Optional[Dict[str, float]] = None,
                       p_inf: float = 1.0,
                       gamma: float = 1.4) -> Dict[str, Dict]:
    """
    Perform shock-expansion analysis on multiple geometries.
    
    Parameters
    ----------
    x_dict : Dict[str, np.ndarray]
        Dictionary of x-coordinates, keyed by geometry identifier (e.g., "h_l_0.02_Mach_1.5")
    y_dict : Dict[str, np.ndarray]
        Dictionary of y-coordinates, keyed by same identifiers
    M_inf_dict : Dict[str, float], optional
        Dictionary of freestream Mach numbers for each case.
        If None, Mach number is extracted from the key (expects format "..._Mach_X.X")
    p_inf : float
        Freestream pressure (same for all cases)
    gamma : float
        Ratio of specific heats
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary containing results for each geometry, with structure:
        {
            "h_l_0.02_Mach_1.5": {
                "first_shock_pressure": float,
                "first_shock_location": float,
                "first_shock_Mach": float,
                "full_result": ShockExpansionResult
            },
            ...
        }
    """
    results = {}
    
    for key in x_dict.keys():
        if key not in y_dict:
            print(f"Warning: Key '{key}' found in x_dict but not y_dict. Skipping.")
            continue
        
        # Extract Mach number from key or dictionary
        if M_inf_dict is not None and key in M_inf_dict:
            M_inf = M_inf_dict[key]
        else:
            # Try to parse from key (format: "h_l_X.XX_Mach_Y.Y")
            try:
                parts = key.split("_")
                mach_idx = parts.index("Mach")
                M_inf = float(parts[mach_idx + 1])
            except (ValueError, IndexError):
                print(f"Warning: Could not extract Mach number from key '{key}'. Using M=2.0")
                M_inf = 2.0
        
        # Extract h/l value for reference
        try:
            parts = key.split("_")
            h_l_idx = parts.index("h") if "h" in parts else parts.index("l") - 1
            h_l_value = float(parts[h_l_idx + 2])  # Assumes format "h_l_X.XX"
        except (ValueError, IndexError):
            h_l_value = None
        
        # Run analysis
        x = x_dict[key]
        y = y_dict[key]
        
        try:
            result = shock_expansion_analysis(x, y, M_inf, p_inf, gamma)
            
            results[key] = {
                "h_l": h_l_value,
                "M_inf": M_inf,
                "first_shock_pressure": result.first_shock_pressure,
                "first_shock_pressure_ratio": result.first_shock_pressure / p_inf,
                "first_shock_location": result.first_shock_location,
                "first_shock_Mach": result.first_shock_Mach,
                "full_result": result
            }
            
        except Exception as e:
            print(f"Error processing '{key}': {e}")
            results[key] = {"error": str(e)}
    
    return results


def get_first_shock_pressures(results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Extract just the first shock pressures from batch results.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        Output from analyze_geometries()
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping geometry key to first shock pressure
    """
    return {
        key: data["first_shock_pressure"] 
        for key, data in results.items() 
        if "first_shock_pressure" in data
    }


# =============================================================================
# Example Usage / Testing
# =============================================================================

if __name__ == "__main__":
    # Create a test sinusoidal geometry
    L = 1.0  # Wavelength
    h = 0.02  # Amplitude (h/L = 0.02)
    n_points = 200
    
    x_test = np.linspace(0, 2*L, n_points)  # Two wavelengths
    y_test = h * np.sin(2 * np.pi * x_test / L)
    
    # Test single geometry analysis
    print("=" * 60)
    print("SHOCK-EXPANSION THEORY TEST")
    print("=" * 60)
    print(f"\nGeometry: Sinusoidal wall, h/L = {h/L:.3f}")
    print(f"Freestream Mach: 2.0")
    print("-" * 60)
    
    result = shock_expansion_analysis(x_test, y_test, M_inf=2.0, p_inf=1.0)
    
    print(f"\nFirst Shock Results:")
    print(f"  Location: x = {result.first_shock_location:.4f}")
    print(f"  Downstream pressure: p = {result.first_shock_pressure:.4f} (p/p_inf = {result.first_shock_pressure:.4f})")
    print(f"  Downstream Mach: M = {result.first_shock_Mach:.4f}")
    
    print(f"\nPressure Statistics:")
    print(f"  Min p/p_inf: {result.pressure_ratio.min():.4f}")
    print(f"  Max p/p_inf: {result.pressure_ratio.max():.4f}")
    
    print(f"\nMach Statistics:")
    print(f"  Min Mach: {result.Mach.min():.4f}")
    print(f"  Max Mach: {result.Mach.max():.4f}")
    
    # Test batch processing with multiple h/L values
    print("\n" + "=" * 60)
    print("BATCH PROCESSING TEST - Multiple h/L values")
    print("=" * 60)
    
    x_dict = {}
    y_dict = {}
    
    for h_l in [0.01, 0.02, 0.03, 0.04]:
        for Mach in [1.5, 2.0, 2.5]:
            key = f"h_l_{h_l}_Mach_{Mach}"
            h_amp = h_l * L
            x_dict[key] = x_test.copy()
            y_dict[key] = h_amp * np.sin(2 * np.pi * x_test / L)
    
    batch_results = analyze_geometries(x_dict, y_dict)
    first_shock_pressures = get_first_shock_pressures(batch_results)
    
    print("\nFirst Shock Pressures by Case:")
    print("-" * 40)
    for key, pressure in sorted(first_shock_pressures.items()):
        print(f"  {key}: p = {pressure:.4f}")
        
        
        
        
        
        
        
 
    
# Small perturbation theory  # 

"""
COMPLETE PHASE-CORRECTED SMALL PERTURBATION THEORY CODE
========================================================
This script combines symbolic mathematics and numerical computation to analyze
supersonic flow over wavy walls using linearized potential flow theory.
"""



# ================================================================
# SECTION 1: SYMBOLIC SETUP AND FUNCTION CREATION
# ================================================================



# Functions #
def find_wall_source_point(x, y, beta, y_wall_func, l):
    """Find the wall source point for a given point (x,y) in the flow field"""
    if y <= y_wall_func(np.clip(x, 0, l)):
        return x
    
    def equation(x_wall):
        y_wall_val = y_wall_func(x_wall)
        return x_wall - (x - beta * (y - y_wall_val))
    
    x_wall_guess = x - beta * y
    x_wall_guess = np.clip(x_wall_guess, 0, l)
    
    try:
        x_wall_solution = fsolve(equation, x_wall_guess)[0]
        return np.clip(x_wall_solution, 0, l)
    except:
        return x_wall_guess
    
    
    

def compute_phi_corrected(x, y, f_indefinite_func, y_wall_func, beta, l, h, V_infty, B, C):
    """
    Compute velocity potential at points (x,y) using zone of influence method.
    
    Parameters:
    -----------
    x, y : array-like
        Coordinates where to compute phi
    f_indefinite_func : callable
        Function requiring (x, h, l, V_infty, B)
    y_wall_func : callable
        Wall shape function
    beta : float
        √(M²-1)
    l, h : float
        Geometry parameters
    V_infty : float
        Freestream velocity
    B : float
        M²-1
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    if x.ndim == 2:
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_wall = find_wall_source_point(x[i,j], y[i,j], beta, y_wall_func, l)
                # Pass all 5 required parameters
                result[i,j] = f_indefinite_func(x_wall, h, l, V_infty, B, C)
        return result
    else:
        result = np.zeros_like(x)
        for i in range(len(x)):
            x_wall = find_wall_source_point(x[i], y[i], beta, y_wall_func, l)
            # Pass all 5 required parameters
            result[i] = f_indefinite_func(x_wall, h, l, V_infty, B)
        return result





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








# ====== CHARACTERISTIC SOURCE POINTS ====== #

def get_wall_critical_points(l, n):
    """Generate critical points along the wall for characteristic lines"""
    wavelength = l / n
    points = []
    
    for i in range(n):
        points.append(i * wavelength)
        points.append(i * wavelength + wavelength/4)
        points.append(i * wavelength + wavelength/2)
        points.append(i * wavelength + 3*wavelength/4)
    points.append(n * wavelength)
    
    all_points = []
    for i in range(len(points)-1):
        all_points.append(points[i])
        all_points.append(points[i] + (points[i+1] - points[i])/3)
        all_points.append(points[i] + 2*(points[i+1] - points[i])/3)
    all_points.append(points[-1])
    
    return np.array([p for p in all_points if 0.01 <= p <= l-0.01])









'''
This functions solves for the pressure on the wall using small perturbation theory. 

The geometry equation is the same equation that is used to describe the simulated geometry

This current solver is an analytical solver. However, I would like to have a numerical solver that creaters...
an equation that describes the geometry from data points


Input 1: h_l_values ---> list of h/l values you would like to analyze 
Input 2: 
'''

def smallPertSolver(h_l_values, ds_by_case, plotting = False):

    # Pre-Allocating variable for results # 
    results_list = []
    axialForceScaled = {}
    

    
    for k, h_l in enumerate(h_l_values):
        # Pre-defined variables # 
        N = 1 
        l = 0.1
        h = h_l * l 
        num_of_points = 1000
        
        
        # Defining the geometry # 
        lam = l / (2 * N + 1)*2 
        x_wave = np.linspace(0, l, num_of_points)
        y_wave = h * np.sin(2 * np.pi * x_wave / lam)
        #y_wave = h*np.cos((2*np.pi*x_wave) / l ) # Textbook Problem 
        
        
        # Defining variables and equations # 
        x_variable = sp.Symbol('x_variable')
        y_variable = sp.Symbol('y_variable')
        y_wall = 0  
        h_variable = sp.Symbol('h_variable')
        l_variable = sp.Symbol("l_variable")
        
        
        # The equation fo the wall is defined here # 
        y_equation = h_variable*sp.sin((2 * sp.pi * x_variable)/lam) # Re-define if you have changed y_wave
        
        
        # Range of Mach numbers # 
        M_infty_range = np.arange(1.5,4.5,0.5)
        
        
     
        
        # For loop to to evaluate the symbolic functions # 
        
        
        for M_infty in M_infty_range:
            
            # Construct key from loop variables
            case_key = f"h_l_{h_l:.2f}_Mach_{M_infty:.1f}"
            
            if case_key not in ds_by_case:
                print(f"Skipping: {case_key} not found")
                continue
            
            
            
            # Defining the flow domain # 
            B = M_infty**2 - 1
            
            # Flow properties (replace with your actual data)
            gamma = 1.4  # Standard air
            R = 287
            
            # Flow results # 
            flow_results = isentropic_solver("m",M_infty)
            P_P0 = flow_results[1]
            #rho_rho0 = flow_results[2]
            T_T0 = flow_results[3]
            
            # Ambient Conditions # 
            T0 = 300 #kelvin 
            P0 = 1e6 #Pa
            
            # Getting static conditions # 
            T_infty = T_T0 * T0  # K
            p_infty = P_P0 * P0  # Pa
            rho_infty = p_infty / (R* T_infty)
            
            #### End of initial conditions ####
            
            # Derived properties
            a_infty = np.sqrt(gamma * R * T_infty)
            V_infty = a_infty * M_infty
            

            # Plotting the geometry # 
            plt.plot(x_wave,y_wave)
            plt.xlabel("X[m]", fontsize = 16)
            plt.ylabel("Y[m]", fontsize = 16)
            plt.title("Wavy Section Geometry",fontsize = 24)
            plt.grid()
            plt.show()
            
            
            
            # ====== SYMBOLIC MATHEMATICS ====== #
            
            dy_dx = sp.diff(y_equation, x_variable)
            V_infty_variable = sp.Symbol("V_infty_variable")
            B_variable = sp.Symbol("B_variable")
            
            dphi_dy_wall = dy_dx * V_infty_variable
            df_dx = dphi_dy_wall / -sp.sqrt(B_variable)
            
            # Integration with phase correction
            C = sp.Symbol('C')
            y_variable = sp.Symbol('y_variable')
            f_indefinite = sp.integrate(df_dx, x_variable) + C
            phi_xy = sp.simplify(f_indefinite).subs(x_variable, x_variable - B_variable * y_variable)
            
            # Defining general equation with constant # 
            phi_xy_general = phi_xy
            phi_xy_wall = phi_xy_general.subs(y_variable, 0)
            
            dphi_dx = sp.diff(phi_xy_general, x_variable)
            dphi_dy = sp.diff(phi_xy_general, y_variable)
            
            Cp = ((-2/V_infty_variable) * dphi_dx)
            Cp_wall = Cp.subs(y_variable, 0)
            
            # Finding Values from the velocity potential equations # 
            u_prime = dphi_dx
            v_prime = dphi_dy
            V_x = V_infty_variable + u_prime
            V_y = v_prime
            
            # Pritning symbolic results # 
            print("=="*20)
            pprint(f"{case_key}")
            pprint(Cp_wall)
            print("=="*20)
            print("\n")
            
            # ====== CREATE NUMERICAL FUNCTIONS ====== #

            
            # CRITICAL: Wall shape function
            y_wall_func = sp.lambdify(x_variable, y_equation.subs([(h_variable, h), (l_variable, l)]), 'numpy')
            
            # f_indefinite function (5 parameters)
            f_indefinite_func = sp.lambdify((x_variable, h_variable, l_variable, V_infty_variable, B_variable, C), 
                                             f_indefinite, 'numpy')
            
            # Velocity potential functions (7 parameters)
            phi_xy_general_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable, C), 
                                               phi_xy_general, 'numpy')
            
            phi_xy_wall_func = sp.lambdify((x_variable, h_variable, l_variable, V_infty_variable, B_variable, C),
                                            phi_xy_wall, 'numpy')
            
            # Partial derivative functions (7 parameters)
            dphi_dx_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                        dphi_dx, 'numpy')
            
            dphi_dy_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                        dphi_dy, 'numpy')
            
            # Pressure coefficient functions
            Cp_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                   Cp, 'numpy')
            
            Cp_wall_func = sp.lambdify((x_variable, h_variable, l_variable, V_infty_variable, B_variable),
                                        Cp_wall, 'numpy')
            
            # Velocity perturbation functions
            u_prime_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                        u_prime, 'numpy')
            
            v_prime_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                        v_prime, 'numpy')
            
            # Total velocity component functions
            V_x_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                    V_x, 'numpy')
            
            V_y_func = sp.lambdify((x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable), 
                                    V_y, 'numpy')
            

            
            # ================================================================
            # SECTION 2: NUMERICAL COMPUTATION AND VISUALIZATION
            # ================================================================
     
        
            
            
            # ====== ZONE OF INFLUENCE ====== #
            
            
            
               
            # ====== DOMAIN ====== #
            
            x_min, x_max = 0, l
            y_min = -h - 0.01
            y_max = 0.3
            
            x_grid = np.linspace(x_min, x_max, 150)
            y_grid = np.linspace(y_min, y_max, 150)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            

            phi_corrected = compute_phi_corrected(X_grid, Y_grid, f_indefinite_func, y_wall_func, 
                                                  np.sqrt(B), l, h, V_infty, B, 5)
            
            # Velocity components
            dx = x_grid[1] - x_grid[0]
            dy = y_grid[1] - y_grid[0]
            
            u_prime = np.gradient(phi_corrected, dx, axis=1)
            v_prime = np.gradient(phi_corrected, dy, axis=0)
            
            u_total = V_infty + u_prime
            v_total = v_prime
            V_total = np.sqrt(u_total**2 + v_total**2)
            
            # Pressure coefficient
            Cp_grid = (-2/V_infty) * u_prime
            
            # Pressure
            pressure_ratio = 1 + (gamma * M_infty**2 / 2) * Cp_grid
            P_local = p_infty * pressure_ratio
            
            # Temperature (linearized)
            temperature_ratio = 1 - (gamma - 1) * M_infty**2 * Cp_grid / 2
            temperature_ratio = np.clip(temperature_ratio, 0.5, 2.0)
            T_local = T_infty * temperature_ratio
            
            # Mach number
            M_local = V_total / a_infty
            
            # Mask below wall
            mask = np.zeros_like(X_grid, dtype=bool)
            for i, x_val in enumerate(x_grid):
                y_wall_at_x = y_wall_func(x_val)
                mask[:, i] = Y_grid[:, i] >= y_wall_at_x
            
            Cp_masked = np.where(mask, Cp_grid, np.nan)
            P_masked = np.where(mask, P_local, np.nan)
            M_masked = np.where(mask, M_local, np.nan)
            

            
 
            n = 500 # number of points 
            wall_x_sources = get_wall_critical_points(l, n)
            wall_y_sources = y_wall_func(wall_x_sources)
            


        
            """
            #------------------------------------------------------------------------------------------------------------------------------------#
                                                Comparing small perturbations theory with RANS Simulation
            #------------------------------------------------------------------------------------------------------------------------------------#
            """
            ### %Comparing Small perturbation theory with RANS viscous simulations amoung all cases ###
        
        
        
            # Masking to the desired points [0,0.1]
            x_wall_RANS = ds_by_case[case_key]["X"].data 
            x_min = 0 
            x_max = l
            mask = (x_min < x_wall_RANS) & (x_wall_RANS < x_max)
        
            # Defining pressure and y at the wall based on the mask
            P_wall_RANS = ds_by_case[case_key]["P"].data[mask]
            y_wall_RANS = ds_by_case[case_key]["Y"].data[mask] 
            x_wall_RANS = x_wall_RANS[mask]
        
        
            # Define points along the wall
            x_wall = np.linspace(0, l, len(P_wall_RANS))
            y_wall = y_wall_func(x_wall)
        
            Cp_wall_results = Cp_wall_func(x_wall, h, l, V_infty, B)
            P_wall = Cp_wall_results*0.5*rho_infty*V_infty**2 + p_infty
            P_diff = (np.abs(P_wall_RANS - P_wall) / P_wall_RANS) * 100
        
        
            # Computing torque for both cases #
            R = 0 # No torque is actually be extraced here. 
            hl_RANS = compute_torque_2D_norm(x_wall_RANS, y_wall_RANS, P_wall_RANS, R)
            hl_smallPert = compute_torque_2D_norm(x_wall, y_wall, P_wall, R)
            
            # Computing the axial force 
            axialForce_RANS = hl_RANS['F_theta']
            axialForce_smallPert = hl_smallPert['F_theta']
            
            
            if plotting == True:
                ##### Creating Plots ###### 
                
                # Imposing figure axes # 
                fig, (ax1,ax2) = plt.subplots(1,2,figsize = (10, 6))
                
                
                # Plot on ax1 (P vs X) #
                ax1.plot(x_wall_RANS, P_wall_RANS, label="RANS", linewidth = 3)
                ax1.plot(x_wall, P_wall, label="Small Perturbation", linestyle='--', linewidth = 3)
                ax1.set_title(f"{case_key}: $P_{{wall}}$ Vs X", fontsize=14)
                ax1.set_xlabel("X [m]", fontsize=12)
                ax1.set_ylabel(r"$P_{wall}$ [Pa]", fontsize=12)
                ax1.grid(True, which="both", alpha=0.3)
                ax1.legend()
                
            
            
                # P_difference Vs X Plot ax2 # 
                ax2.plot(x_wall, P_diff)
                ax2.set_xlim([0,l])
                ax2.set_title(r"$P_{difference}$ Vs X", fontsize = 24)
            
                ax2.set_xlabel("X [m]", fontsize = 18)
                ax2.set_ylabel(r"$P_{difference}$[%]", fontsize = 18) 
            
                ax2.grid(True, which = "both")
                ax2.legend()
                
            
      
            
            # Scaling the axial force from RANS with the axial force from small perturbation theory # 
            axialForceScaled[case_key] = axialForce_RANS / axialForce_smallPert 
            

            # Computing the axial force from small perturbation theory # 
            axialForce_diff =  (1 - (axialForce_RANS / axialForce_smallPert)) * 100
            results_list.append({
                'h/l': h_l,
                'M_infty': M_infty,
                'F_axial_RANS [N/m]': axialForce_RANS,
                'F_axial_SmallPert [N/m]': axialForce_smallPert,
                'Difference [%]': axialForce_diff,
                'Case_Key': case_key
            })
            
            

            
            
            # Convert list of dictionaries to DataFrame
            df_results = pd.DataFrame(results_list)

            pivot_table = df_results.pivot_table(
                values='Difference [%]',
                index='h/l',
                columns='M_infty',
                aggfunc='mean'  # In case of duplicates
            )
            
            # Exporting Table into excel that compares M\infty Axial force, percentage difference, and the case. # 
            df_results.to_csv(r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Graphs\axial_force_comparison.csv', index=False)
            pivot_table.to_csv(r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\Graphs\axial_force_pivot.csv') 

                    
            plt.show()
            
    return axialForceScaled






#%% Small perturbation theory with shock - expansion theory ###



"""
#------------------------------------------------------------------------------------------------------------------------------------#
             Enhanced Small Perturbation Solver with Shock-Expansion Theory
#------------------------------------------------------------------------------------------------------------------------------------#

PURPOSE:
========
This is your existing smallPertSolver with shock-expansion theory ADDED as a second
theoretical comparison. The original small perturbation theory is UNCHANGED — shock-expansion
is purely additive.

WHY BOTH THEORIES?
==================
Small Perturbation Theory:
    - Assumes small disturbances (h/l << 1)
    - Linearizes the governing equations → Cp = -2*dy/dx / sqrt(M²-1)
    - Gives a sinusoidal pressure distribution (same shape as the wall)
    - Breaks down at higher h/l and higher Mach (nonlinear effects)

Shock-Expansion Theory:
    - Handles finite-amplitude disturbances
    - Uses exact oblique shock relations for compressions
    - Uses exact Prandtl-Meyer function for expansions
    - Captures nonlinear wave steepening, pressure asymmetry
    - More accurate at higher h/l and Mach, but still inviscid (no separation)

Together they bracket the RANS solution: small perturbation from the linear side,
shock-expansion from the nonlinear inviscid side.

WHAT CHANGED FROM YOUR ORIGINAL:
=================================
1. Added: shock_expansion_wall_pressure() function call inside the Mach loop
2. Added: P_wall_SE to the comparison plots (third line)
3. Added: axialForce_SE and its difference to the results DataFrame
4. Added: axialForceScaled_SE output dictionary
5. Everything else is IDENTICAL to your original code

USAGE:
======
    # Same call signature as before, with one new output
    axialForceScaled, axialForceScaled_SE = smallPertSolver_with_SE(
        h_l_values, ds_by_case, plotting=True
    )
"""

import numpy as np
import sympy as sp
from sympy import pprint
import matplotlib.pyplot as plt
import pandas as pd
#from shock_expansion_theory import shock_expansion_analysis
# You need these from your existing codebase:
# from your_module import isentropic_solver, compute_torque_2D_norm, compute_phi_corrected
# from your_module import get_wall_critical_points
# from shock_expansion_theory import shock_expansion_analysis


# =============================================================================
# NEW FUNCTION: Get wall pressure from shock-expansion theory
# =============================================================================
"""
TEACHING POINT - Why this wrapper function?
--------------------------------------------
Your shock_expansion_analysis() returns a ShockExpansionResult object with 
pressure_ratio (p/p_inf) at the x-coordinates of the INPUT geometry.

But in your solver, the RANS wall points (x_wall_RANS) and the theory wall 
points (x_wall from linspace) may have DIFFERENT x-coordinates and DIFFERENT 
numbers of points.

So this wrapper:
    1. Runs shock-expansion on the analytical geometry (x_wave, y_wave)
    2. Interpolates the result onto whatever x-grid you need
    3. Returns dimensional pressure [Pa] for direct comparison with RANS

This keeps the shock-expansion call clean and reusable.
"""

def shock_expansion_wall_pressure(x_wall, y_wall, M_inf, p_inf, gamma=1.4):
    """
    Compute wall pressure distribution using shock-expansion theory.
    
    Parameters
    ----------
    x_wall : np.ndarray
        x-coordinates along the wall where you want P
    y_wall : np.ndarray
        y-coordinates of the wall at those x locations
    M_inf : float
        Freestream Mach number
    p_inf : float
        Freestream static pressure [Pa]
    gamma : float
        Ratio of specific heats (default 1.4)
    
    Returns
    -------
    P_wall_SE : np.ndarray
        Wall pressure [Pa] at each x_wall location
    result : ShockExpansionResult
        Full result object (in case you want Mach distribution, etc.)
    """
    # Run shock-expansion analysis
    # This marches along the wall, applying oblique shocks for compressions
    # and Prandtl-Meyer expansions for expansions
    result = shock_expansion_analysis(x_wall, y_wall, M_inf, p_inf=p_inf, gamma=gamma)
    
    # result.pressure is already in [Pa] since we passed p_inf in dimensional form
    P_wall_SE = result.pressure
    
    return P_wall_SE, result


# =============================================================================
# ENHANCED SOLVER: Original + Shock-Expansion
# =============================================================================

def smallPertSolver_with_SE(h_l_values, ds_by_case, plotting=False):
    """
    Enhanced version of smallPertSolver that includes shock-expansion theory.
    
    Parameters
    ----------
    h_l_values : list
        List of h/l amplitude ratios
    ds_by_case : dict
        Dictionary of case datasets
    plotting : bool
        Whether to show comparison plots
    
    Returns
    -------
    axialForceScaled : dict
        F_RANS / F_smallPert for each case (same as original)
    axialForceScaled_SE : dict
        F_RANS / F_shockExpansion for each case (NEW)
    """
    
    # Pre-Allocating variables for results
    results_list = []
    axialForceScaled = {}
    axialForceScaled_SE = {}  # NEW: shock-expansion scaling
    
    for k, h_l in enumerate(h_l_values):
        # Pre-defined variables
        N = 1
        l = 0.1
        h = h_l * l
        num_of_points = 1000
        
        # Defining the geometry
        lam = l / (2 * N + 1) * 2
        x_wave = np.linspace(0, l, num_of_points)
        y_wave = h * np.sin(2 * np.pi * x_wave / lam)
        
        # Defining symbolic variables and equations
        x_variable = sp.Symbol('x_variable')
        y_variable = sp.Symbol('y_variable')
        y_wall = 0
        h_variable = sp.Symbol('h_variable')
        l_variable = sp.Symbol("l_variable")
        
        # The equation of the wall
        y_equation = h_variable * sp.sin((2 * sp.pi * x_variable) / lam)
        
        # Range of Mach numbers
        M_infty_range = np.arange(1.5, 4.5, 0.5)
        
        for M_infty in M_infty_range:
            
            # Construct key from loop variables
            case_key = f"h_l_{h_l:.2f}_Mach_{M_infty:.1f}"
            
            if case_key not in ds_by_case:
                print(f"Skipping: {case_key} not found")
                continue
            
            # ================================================================
            # FLOW CONDITIONS (unchanged from your original)
            # ================================================================
            B = M_infty**2 - 1
            gamma = 1.4
            R = 287
            
            flow_results = isentropic_solver("m", M_infty)
            P_P0 = flow_results[1]
            T_T0 = flow_results[3]
            
            T0 = 300   # K
            P0 = 1e6   # Pa
            
            T_infty = T_T0 * T0
            p_infty = P_P0 * P0
            rho_infty = p_infty / (R * T_infty)
            
            a_infty = np.sqrt(gamma * R * T_infty)
            V_infty = a_infty * M_infty
            
            # ================================================================
            # SYMBOLIC MATH FOR SMALL PERTURBATION THEORY (unchanged)
            # ================================================================
            
            dy_dx = sp.diff(y_equation, x_variable)
            V_infty_variable = sp.Symbol("V_infty_variable")
            B_variable = sp.Symbol("B_variable")
            
            dphi_dy_wall = dy_dx * V_infty_variable
            df_dx = dphi_dy_wall / -sp.sqrt(B_variable)
            
            C = sp.Symbol('C')
            y_variable = sp.Symbol('y_variable')
            f_indefinite = sp.integrate(df_dx, x_variable) + C
            phi_xy = sp.simplify(f_indefinite).subs(x_variable, x_variable - B_variable * y_variable)
            
            phi_xy_general = phi_xy
            phi_xy_wall = phi_xy_general.subs(y_variable, 0)
            
            dphi_dx = sp.diff(phi_xy_general, x_variable)
            dphi_dy = sp.diff(phi_xy_general, y_variable)
            
            Cp = ((-2 / V_infty_variable) * dphi_dx)
            Cp_wall = Cp.subs(y_variable, 0)
            
            u_prime = dphi_dx
            v_prime = dphi_dy
            V_x = V_infty_variable + u_prime
            V_y = v_prime
            
            print("==" * 20)
            pprint(f"{case_key}")
            pprint(Cp_wall)
            print("==" * 20)
            print("\n")
            
            # ================================================================
            # CREATE NUMERICAL FUNCTIONS (unchanged)
            # ================================================================
            
            y_wall_func = sp.lambdify(x_variable,
                                       y_equation.subs([(h_variable, h), (l_variable, l)]), 'numpy')
            
            f_indefinite_func = sp.lambdify(
                (x_variable, h_variable, l_variable, V_infty_variable, B_variable, C),
                f_indefinite, 'numpy')
            
            phi_xy_general_func = sp.lambdify(
                (x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable, C),
                phi_xy_general, 'numpy')
            
            phi_xy_wall_func = sp.lambdify(
                (x_variable, h_variable, l_variable, V_infty_variable, B_variable, C),
                phi_xy_wall, 'numpy')
            
            dphi_dx_func = sp.lambdify(
                (x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable),
                dphi_dx, 'numpy')
            
            dphi_dy_func = sp.lambdify(
                (x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable),
                dphi_dy, 'numpy')
            
            Cp_func = sp.lambdify(
                (x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable),
                Cp, 'numpy')
            
            Cp_wall_func = sp.lambdify(
                (x_variable, h_variable, l_variable, V_infty_variable, B_variable),
                Cp_wall, 'numpy')
            
            u_prime_func = sp.lambdify(
                (x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable),
                u_prime, 'numpy')
            
            v_prime_func = sp.lambdify(
                (x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable),
                v_prime, 'numpy')
            
            V_x_func = sp.lambdify(
                (x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable),
                V_x, 'numpy')
            
            V_y_func = sp.lambdify(
                (x_variable, y_variable, h_variable, l_variable, V_infty_variable, B_variable),
                V_y, 'numpy')
            
            # ================================================================
            # DOMAIN AND FIELD COMPUTATION (unchanged)
            # ================================================================
            
            x_min, x_max = 0, l
            y_min = -h - 0.01
            y_max = 0.3
            
            x_grid = np.linspace(x_min, x_max, 150)
            y_grid = np.linspace(y_min, y_max, 150)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            
            phi_corrected = compute_phi_corrected(X_grid, Y_grid, f_indefinite_func,
                                                   y_wall_func, np.sqrt(B), l, h,
                                                   V_infty, B, 5)
            
            dx = x_grid[1] - x_grid[0]
            dy = y_grid[1] - y_grid[0]
            
            u_prime_grid = np.gradient(phi_corrected, dx, axis=1)
            v_prime_grid = np.gradient(phi_corrected, dy, axis=0)
            
            u_total = V_infty + u_prime_grid
            v_total = v_prime_grid
            V_total = np.sqrt(u_total**2 + v_total**2)
            
            Cp_grid = (-2 / V_infty) * u_prime_grid
            
            pressure_ratio = 1 + (gamma * M_infty**2 / 2) * Cp_grid
            P_local = p_infty * pressure_ratio
            
            temperature_ratio = 1 - (gamma - 1) * M_infty**2 * Cp_grid / 2
            temperature_ratio = np.clip(temperature_ratio, 0.5, 2.0)
            T_local = T_infty * temperature_ratio
            
            M_local = V_total / a_infty
            
            mask = np.zeros_like(X_grid, dtype=bool)
            for i, x_val in enumerate(x_grid):
                y_wall_at_x = y_wall_func(x_val)
                mask[:, i] = Y_grid[:, i] >= y_wall_at_x
            
            Cp_masked = np.where(mask, Cp_grid, np.nan)
            P_masked = np.where(mask, P_local, np.nan)
            M_masked = np.where(mask, M_local, np.nan)
            
            n = 500
            wall_x_sources = get_wall_critical_points(l, n)
            wall_y_sources = y_wall_func(wall_x_sources)
            
            # ================================================================
            # RANS DATA EXTRACTION (unchanged)
            # ================================================================
            
            x_wall_RANS = ds_by_case[case_key]["X"].data
            x_min_mask = 0
            x_max_mask = l
            mask_rans = (x_min_mask < x_wall_RANS) & (x_wall_RANS < x_max_mask)
            
            P_wall_RANS = ds_by_case[case_key]["P"].data[mask_rans]
            y_wall_RANS = ds_by_case[case_key]["Y"].data[mask_rans]
            x_wall_RANS = x_wall_RANS[mask_rans]
            
            # ================================================================
            # SMALL PERTURBATION WALL PRESSURE (unchanged)
            # ================================================================
            
            x_wall_sp = np.linspace(0, l, len(P_wall_RANS))
            y_wall_sp = y_wall_func(x_wall_sp)
            
            Cp_wall_results = Cp_wall_func(x_wall_sp, h, l, V_infty, B)
            P_wall_smallPert = Cp_wall_results * 0.5 * rho_infty * V_infty**2 + p_infty
            P_diff_smallPert = (np.abs(P_wall_RANS - P_wall_smallPert) / P_wall_RANS) * 100
            
            # ================================================================
            # NEW: SHOCK-EXPANSION WALL PRESSURE
            # ================================================================
            """
            TEACHING POINT - What's happening here:
            ----------------------------------------
            We evaluate shock-expansion theory on the SAME x-grid as the small 
            perturbation theory (x_wall_sp) so that all three curves (RANS, 
            small pert, shock-expansion) are on the same x-coordinates.
            
            The shock_expansion_wall_pressure() function:
                1. Computes dy/dx → local wall angle θ at each point
                2. Computes dθ between successive points
                3. If dθ > 0 (compression): applies oblique shock relations
                   → solves for shock angle β, gets post-shock M and p
                4. If dθ < 0 (expansion): applies Prandtl-Meyer function
                   → uses ν(M) + |dθ| to get new M, then isentropic p/p0
                5. Returns the absolute pressure at each wall point
            
            KEY DIFFERENCE from small perturbation:
            - Small pert: Cp = -2*(dy/dx) / sqrt(M²-1) → always sinusoidal
            - Shock-expansion: each compression/expansion changes the LOCAL 
              Mach number, so subsequent waves see different M → asymmetric
              pressure distribution, stronger peaks, physical nonlinearity
            """
            
            P_wall_SE, SE_result = shock_expansion_wall_pressure(
                x_wall_sp, y_wall_sp, M_infty, p_infty, gamma
            )
            P_diff_SE = (np.abs(P_wall_RANS - P_wall_SE) / P_wall_RANS) * 100
            
            # ================================================================
            # FORCE COMPUTATION (original + new)
            # ================================================================
            
            R_torque = 0  # No torque extraction
            hl_RANS = compute_torque_2D_norm(x_wall_RANS, y_wall_RANS, P_wall_RANS, R_torque)
            hl_smallPert = compute_torque_2D_norm(x_wall_sp, y_wall_sp, P_wall_smallPert, R_torque)
            hl_SE = compute_torque_2D_norm(x_wall_sp, y_wall_sp, P_wall_SE, R_torque)  # NEW
            
            axialForce_RANS = hl_RANS['F_theta']
            axialForce_smallPert = hl_smallPert['F_theta']
            axialForce_SE = hl_SE['F_theta']  # NEW
            
            # ================================================================
            # PLOTTING (enhanced with shock-expansion)
            # ================================================================
            
            if plotting:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # --- Panel 1: P_wall comparison (3 curves now) ---
                ax1 = axes[0]
                ax1.plot(x_wall_RANS, P_wall_RANS, 'k-', linewidth=3, label="RANS")
                ax1.plot(x_wall_sp, P_wall_smallPert, 'b--', linewidth=2.5, label="Small Perturbation")
                ax1.plot(x_wall_sp, P_wall_SE, 'r-.', linewidth=2.5, label="Shock-Expansion")
                ax1.set_title(f"{case_key}: $P_{{wall}}$ Comparison", fontsize=14)
                ax1.set_xlabel("X [m]", fontsize=12)
                ax1.set_ylabel(r"$P_{wall}$ [Pa]", fontsize=12)
                ax1.grid(True, which="both", alpha=0.3)
                ax1.legend(fontsize=10)
                
                # --- Panel 2: % Difference for both theories ---
                ax2 = axes[1]
                ax2.plot(x_wall_sp, P_diff_smallPert, 'b-', linewidth=2, label="Small Pert Error")
                ax2.plot(x_wall_sp, P_diff_SE, 'r-', linewidth=2, label="Shock-Exp Error")
                ax2.set_xlim([0, l])
                ax2.set_title(r"$P_{difference}$ [%] vs X", fontsize=14)
                ax2.set_xlabel("X [m]", fontsize=12)
                ax2.set_ylabel(r"$P_{difference}$ [%]", fontsize=12)
                ax2.grid(True, which="both", alpha=0.3)
                ax2.legend(fontsize=10)
                
                # --- Panel 3: Mach distribution from shock-expansion ---
                """
                TEACHING POINT - Why plot the local Mach?
                ------------------------------------------
                This is something small perturbation can't give you directly.
                Shock-expansion tracks the LOCAL Mach number as it changes 
                through each compression/expansion. This is important because:
                    - If M_local drops below 1.0, the theory breaks (subsonic pocket)
                    - If M_local gets very high, you may get unrealistic expansions
                    - The Mach distribution shows WHERE the flow is most stressed
                """
                ax3 = axes[2]
                ax3.plot(SE_result.x, SE_result.Mach, 'r-', linewidth=2)
                ax3.axhline(y=M_infty, linestyle='--', color='gray', alpha=0.5, label=f'$M_\\infty$ = {M_infty}')
                ax3.axhline(y=1.0, linestyle=':', color='black', alpha=0.3, label='M = 1')
                ax3.set_title(f"Local Mach (Shock-Expansion)", fontsize=14)
                ax3.set_xlabel("X [m]", fontsize=12)
                ax3.set_ylabel("Mach Number", fontsize=12)
                ax3.grid(True, which="both", alpha=0.3)
                ax3.legend(fontsize=10)
                
                plt.tight_layout()
                plt.show()
            
            # ================================================================
            # STORE RESULTS (original + new)
            # ================================================================
            
            # Original scaling
            axialForceScaled[case_key] = axialForce_RANS / axialForce_smallPert
            
            # NEW: shock-expansion scaling
            if axialForce_SE != 0:
                axialForceScaled_SE[case_key] = axialForce_RANS / axialForce_SE
            else:
                axialForceScaled_SE[case_key] = np.nan
            
            # Compute differences
            axialForce_diff_sp = (1 - (axialForce_RANS / axialForce_smallPert)) * 100
            
            if axialForce_SE != 0:
                axialForce_diff_SE = (1 - (axialForce_RANS / axialForce_SE)) * 100
            else:
                axialForce_diff_SE = np.nan
            
            results_list.append({
                'h/l': h_l,
                'M_infty': M_infty,
                'F_axial_RANS [N/m]': axialForce_RANS,
                'F_axial_SmallPert [N/m]': axialForce_smallPert,
                'F_axial_ShockExp [N/m]': axialForce_SE,           # NEW
                'Diff_SmallPert [%]': axialForce_diff_sp,
                'Diff_ShockExp [%]': axialForce_diff_SE,            # NEW
                'Case_Key': case_key
            })
            
            # ================================================================
            # EXPORT (enhanced with shock-expansion columns)
            # ================================================================
            
            df_results = pd.DataFrame(results_list)
            
            # Pivot table for small perturbation (original)
            pivot_sp = df_results.pivot_table(
                values='Diff_SmallPert [%]',
                index='h/l',
                columns='M_infty',
                aggfunc='mean'
            )
            
            # NEW: Pivot table for shock-expansion
            pivot_SE = df_results.pivot_table(
                values='Diff_ShockExp [%]',
                index='h/l',
                columns='M_infty',
                aggfunc='mean'
            )
            
            # Save to CSV
            df_results.to_csv(
                r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\axial_force_comparison.csv',
                index=False
            )
            pivot_sp.to_csv(
                r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\axial_force_pivot_smallPert.csv'
            )
            pivot_SE.to_csv(
                r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study\axial_force_pivot_shockExp.csv'
            )
            
            
            
            plt.show()
    
    return axialForceScaled, axialForceScaled_SE




#%% Small perturbation theory + SE ###


"""
#------------------------------------------------------------------------------------------------------------------------------------#
         Combined Small Perturbation + Shock-Expansion Theory
#------------------------------------------------------------------------------------------------------------------------------------#

PURPOSE:
========
This module provides a COMBINED theoretical prediction that leverages the strengths
of both small perturbation theory and shock-expansion theory.

THE PHYSICS OF WHY THIS WORKS:
==============================

Small Perturbation Theory (SPT):
    Strengths:
        - Solves the linearized potential equation over the FULL domain
        - Naturally captures wave reflections, interference, superposition
        - Gives smooth, continuous pressure/velocity fields
        - Handles the global flow structure well
    Weaknesses:
        - Assumes small disturbances → underestimates pressure at shocks
        - Cp = -2*(dy/dx)/sqrt(M²-1) is symmetric (sinusoidal in, sinusoidal out)
        - Cannot capture nonlinear wave steepening

Shock-Expansion Theory (SE):
    Strengths:
        - Uses EXACT oblique shock and Prandtl-Meyer relations
        - Captures nonlinear pressure amplification at compressions
        - Tracks local Mach number changes → each wave sees correct M
        - Gets the pressure asymmetry right (sharp compression, gentle expansion)
    Weaknesses:
        - Marching method → only knows about upstream conditions
        - No wave reflections or interference from boundaries
        - Point-by-point, no global flow awareness

COMBINATION STRATEGY:
=====================
We use shock-expansion as a NONLINEAR CORRECTION to the small perturbation baseline.

    P_combined(x) = P_smallPert(x) * [ P_SE(x) / P_SE_linearized(x) ]

Where P_SE_linearized is what shock-expansion theory WOULD give if the disturbances
were truly small (i.e., if SE agreed with small perturbation). The ratio captures
ONLY the nonlinear amplification factor.

In practice, for weak disturbances (low h/l), the ratio → 1 and you recover SPT.
For strong disturbances (high h/l, high Mach), the ratio deviates from 1 and
corrects the pressure peaks.

Think of it like this:
    - SPT provides the "map" of the flow (correct topology, wave patterns)
    - SE provides the "intensity correction" (how much stronger shocks really are)
    - Combined = correct map × correct intensity

USAGE:
======
    # After running both theories independently:
    P_combined = combined_wall_pressure(x_wall, P_wall_smallPert, P_wall_SE, p_infty)

    # Or use the full solver that does everything:
    axialForceScaled, axialForceScaled_SE, axialForceScaled_combined = \
        smallPertSolver_combined(h_l_values, ds_by_case, plotting=True)
"""

import numpy as np
import sympy as sp
from sympy import pprint
import matplotlib.pyplot as plt
import pandas as pd

# You need these from your existing codebase:
# from your_module import isentropic_solver, compute_torque_2D_norm
# from your_module import compute_phi_corrected, get_wall_critical_points
# from shock_expansion_theory import shock_expansion_analysis


# =============================================================================
# CORE COMBINATION FUNCTION
# =============================================================================

def combined_wall_pressure(x_wall, P_wall_smallPert, P_wall_SE, p_infty,
                           blend_sharpness=10.0):
    """
    Combine small perturbation and shock-expansion wall pressures.
    
    The method computes a nonlinear correction ratio from shock-expansion
    and applies it to the small perturbation baseline.
    
    Parameters
    ----------
    x_wall : np.ndarray
        x-coordinates along the wall
    P_wall_smallPert : np.ndarray
        Wall pressure from small perturbation theory [Pa]
    P_wall_SE : np.ndarray
        Wall pressure from shock-expansion theory [Pa]
    p_infty : float
        Freestream static pressure [Pa]
    blend_sharpness : float
        Controls how aggressively the nonlinear correction is applied.
        Higher = trusts SE more at large deviations. Default 10.0 works
        well for h/l in [0.02, 0.09] range.
    
    Returns
    -------
    P_combined : np.ndarray
        Combined wall pressure [Pa]
    correction_ratio : np.ndarray
        The nonlinear correction factor applied (for diagnostics)
    
    TEACHING POINT - The Math:
    --------------------------
    Define pressure perturbations relative to freestream:
        delta_SP(x) = P_smallPert(x) - p_infty    (linear perturbation)
        delta_SE(x) = P_SE(x) - p_infty            (nonlinear perturbation)
    
    The correction ratio is:
        R(x) = delta_SE(x) / delta_SP(x)    where delta_SP ≠ 0
    
    This ratio captures HOW MUCH the nonlinear physics amplifies (or reduces)
    the pressure perturbation compared to linear theory:
        R > 1 → nonlinear amplification (compressions are stronger than linear)
        R < 1 → nonlinear weakening (expansions are gentler than linear)
        R ≈ 1 → linear and nonlinear agree (weak disturbance region)
    
    The combined pressure is then:
        P_combined(x) = p_infty + delta_SP(x) * R(x)
                      = p_infty + delta_SE(x)     ... wait, that's just P_SE!
    
    NOT QUITE — because we apply a WEIGHTED blend. Near the freestream pressure
    (where both theories agree), we trust SPT. Where there are large perturbations
    (where nonlinearity matters), we shift toward SE. This is implemented via
    a smooth weighting function based on the magnitude of the perturbation.
    
    The blending weight w(x) is:
        w(x) = tanh(blend_sharpness * |delta_SP(x) / p_infty|)
    
    Where:
        - Small perturbation (|delta_SP| << p_infty) → w ≈ 0 → trust SPT
        - Large perturbation (|delta_SP| ~ p_infty)  → w ≈ 1 → trust SE
    
    Final combined pressure:
        P_combined(x) = (1 - w(x)) * P_smallPert(x) + w(x) * P_SE(x)
    """
    
    # Pressure perturbations relative to freestream
    delta_SP = P_wall_smallPert - p_infty
    delta_SE = P_wall_SE - p_infty
    
    # Normalized perturbation magnitude (how "large" is the disturbance)
    perturbation_magnitude = np.abs(delta_SP) / p_infty
    
    # Blending weight: smooth transition from SPT (w=0) to SE (w=1)
    # tanh gives: ~0 for small perturbations, ~1 for large perturbations
    w = np.tanh(blend_sharpness * perturbation_magnitude)
    
    # Combined pressure: weighted blend
    P_combined = (1 - w) * P_wall_smallPert + w * P_wall_SE
    
    # Correction ratio for diagnostics (how much did we shift from SPT?)
    correction_ratio = np.where(
        np.abs(P_wall_smallPert) > 1e-10,
        P_combined / P_wall_smallPert,
        1.0
    )
    
    return P_combined, correction_ratio


# =============================================================================
# WRAPPER: Get SE wall pressure (same as previous file)
# =============================================================================

def shock_expansion_wall_pressure(x_wall, y_wall, M_inf, p_inf, gamma=1.4):
    """
    Compute wall pressure from shock-expansion theory.
    """
    result = shock_expansion_analysis(x_wall, y_wall, M_inf, p_inf=p_inf, gamma=gamma)
    return result.pressure, result


# =============================================================================
# FULL COMBINED SOLVER
# =============================================================================

def smallPertSolver_combined(h_l_values, ds_by_case, plotting=False):
    """
    Solver that computes and compares ALL THREE predictions:
        1. Small Perturbation Theory (SPT)
        2. Shock-Expansion Theory (SE)
        3. Combined SPT + SE
    
    Parameters
    ----------
    h_l_values : list
        List of h/l amplitude ratios
    ds_by_case : dict
        Dictionary of case datasets
    plotting : bool
        Whether to show comparison plots
    
    Returns
    -------
    axialForceScaled : dict
        F_RANS / F_smallPert
    axialForceScaled_SE : dict
        F_RANS / F_shockExpansion
    axialForceScaled_combined : dict
        F_RANS / F_combined
    """
    
    results_list = []
    axialForceScaled = {}
    axialForceScaled_SE = {}
    axialForceScaled_combined = {}  # NEW
    
    for k, h_l in enumerate(h_l_values):
        N = 1
        l = 0.1
        h = h_l * l
        num_of_points = 1000
        
        lam = l / (2 * N + 1) * 2
        x_wave = np.linspace(0, l, num_of_points)
        y_wave = h * np.sin(2 * np.pi * x_wave / lam)
        
        # Symbolic setup
        x_variable = sp.Symbol('x_variable')
        y_variable = sp.Symbol('y_variable')
        h_variable = sp.Symbol('h_variable')
        l_variable = sp.Symbol("l_variable")
        
        y_equation = h_variable * sp.sin((2 * sp.pi * x_variable) / lam)
        
        M_infty_range = np.arange(1.5, 4.5, 0.5)
        
        for M_infty in M_infty_range:
            
            case_key = f"h_l_{h_l:.2f}_Mach_{M_infty:.1f}"
            
            if case_key not in ds_by_case:
                print(f"Skipping: {case_key} not found")
                continue
            
            # ============================================================
            # FLOW CONDITIONS
            # ============================================================
            B = M_infty**2 - 1
            gamma = 1.4
            R = 287
            
            flow_results = isentropic_solver("m", M_infty)
            P_P0 = flow_results[1]
            T_T0 = flow_results[3]
            
            T0 = 300
            P0 = 1e6
            
            T_infty = T_T0 * T0
            p_infty = P_P0 * P0
            rho_infty = p_infty / (R * T_infty)
            
            a_infty = np.sqrt(gamma * R * T_infty)
            V_infty = a_infty * M_infty
            
            # ============================================================
            # SMALL PERTURBATION THEORY (symbolic, unchanged)
            # ============================================================
            
            dy_dx = sp.diff(y_equation, x_variable)
            V_infty_variable = sp.Symbol("V_infty_variable")
            B_variable = sp.Symbol("B_variable")
            
            dphi_dy_wall = dy_dx * V_infty_variable
            df_dx = dphi_dy_wall / -sp.sqrt(B_variable)
            
            C = sp.Symbol('C')
            y_variable = sp.Symbol('y_variable')
            f_indefinite = sp.integrate(df_dx, x_variable) + C
            phi_xy = sp.simplify(f_indefinite).subs(
                x_variable, x_variable - B_variable * y_variable
            )
            
            phi_xy_general = phi_xy
            dphi_dx = sp.diff(phi_xy_general, x_variable)
            
            Cp_wall = ((-2 / V_infty_variable) * dphi_dx).subs(y_variable, 0)
            
            print("==" * 20)
            pprint(f"{case_key}")
            pprint(Cp_wall)
            print("==" * 20 + "\n")
            
            # Lambdify
            y_wall_func = sp.lambdify(
                x_variable,
                y_equation.subs([(h_variable, h), (l_variable, l)]),
                'numpy'
            )
            
            Cp_wall_func = sp.lambdify(
                (x_variable, h_variable, l_variable, V_infty_variable, B_variable),
                Cp_wall, 'numpy'
            )
            
            # ============================================================
            # RANS DATA
            # ============================================================
            
            x_wall_RANS = ds_by_case[case_key]["X"].data
            mask_rans = (0 < x_wall_RANS) & (x_wall_RANS < l)
            
            P_wall_RANS = ds_by_case[case_key]["P"].data[mask_rans]
            y_wall_RANS = ds_by_case[case_key]["Y"].data[mask_rans]
            x_wall_RANS = x_wall_RANS[mask_rans]
            
            # ============================================================
            # THEORY 1: SMALL PERTURBATION
            # ============================================================
            
            x_wall_th = np.linspace(0, l, len(P_wall_RANS))
            y_wall_th = y_wall_func(x_wall_th)
            
            Cp_wall_results = Cp_wall_func(x_wall_th, h, l, V_infty, B)
            P_wall_SP = Cp_wall_results * 0.5 * rho_infty * V_infty**2 + p_infty
            
            # ============================================================
            # THEORY 2: SHOCK-EXPANSION
            # ============================================================
            
            P_wall_SE, SE_result = shock_expansion_wall_pressure(
                x_wall_th, y_wall_th, M_infty, p_infty, gamma
            )
            
            # ============================================================
            # THEORY 3: COMBINED (the new part)
            # ============================================================
            """
            TEACHING POINT - The blend_sharpness parameter:
            ------------------------------------------------
            This controls the transition sensitivity. A value of 10 means:
                - At |delta_P/p_inf| = 0.05 (5% perturbation):  w ≈ 0.46
                - At |delta_P/p_inf| = 0.10 (10% perturbation): w ≈ 0.76
                - At |delta_P/p_inf| = 0.20 (20% perturbation): w ≈ 0.96
            
            For your range of h/l = 0.02 to 0.09:
                - Low h/l (0.02): pressure perturbations are ~5-10% → mostly SPT
                - High h/l (0.09): perturbations are ~30-50% → mostly SE
            
            This naturally transitions from linear to nonlinear theory 
            exactly where it should. You can tune this if needed.
            """
            
            P_wall_combined, correction_ratio = combined_wall_pressure(
                x_wall_th, P_wall_SP, P_wall_SE, p_infty,
                blend_sharpness=10.0
            )
            
            # ============================================================
            # % ERRORS
            # ============================================================
            
            P_diff_SP = (np.abs(P_wall_RANS - P_wall_SP) / P_wall_RANS) * 100
            P_diff_SE = (np.abs(P_wall_RANS - P_wall_SE) / P_wall_RANS) * 100
            P_diff_combined = (np.abs(P_wall_RANS - P_wall_combined) / P_wall_RANS) * 100
            
            # Print average errors for quick comparison
            print(f"  Avg Error → SPT: {np.nanmean(P_diff_SP):.2f}%  |  "
                  f"SE: {np.nanmean(P_diff_SE):.2f}%  |  "
                  f"Combined: {np.nanmean(P_diff_combined):.2f}%")
            
            # ============================================================
            # FORCES
            # ============================================================
            
            R_torque = 0
            hl_RANS = compute_torque_2D_norm(x_wall_RANS, y_wall_RANS, P_wall_RANS, R_torque)
            hl_SP = compute_torque_2D_norm(x_wall_th, y_wall_th, P_wall_SP, R_torque)
            hl_SE = compute_torque_2D_norm(x_wall_th, y_wall_th, P_wall_SE, R_torque)
            hl_combined = compute_torque_2D_norm(x_wall_th, y_wall_th, P_wall_combined, R_torque)
            
            F_RANS = hl_RANS['F_theta']
            F_SP = hl_SP['F_theta']
            F_SE = hl_SE['F_theta']
            F_combined = hl_combined['F_theta']
            
            # ============================================================
            # PLOTTING
            # ============================================================
            
            if plotting:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # --- Panel 1: P_wall comparison (4 curves) ---
                ax1 = axes[0, 0]
                ax1.plot(x_wall_RANS, P_wall_RANS, 'k-', linewidth=3, label="RANS")
                ax1.plot(x_wall_th, P_wall_SP, 'b--', linewidth=2, label="Small Perturbation")
                ax1.plot(x_wall_th, P_wall_SE, 'r-.', linewidth=2, label="Shock-Expansion")
                ax1.plot(x_wall_th, P_wall_combined, 'g-', linewidth=2.5,
                         label="Combined", alpha=0.85)
                ax1.set_title(f"{case_key}: $P_{{wall}}$ Comparison", fontsize=14)
                ax1.set_xlabel("X [m]", fontsize=12)
                ax1.set_ylabel(r"$P_{wall}$ [Pa]", fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=10)
                
                # --- Panel 2: % Error comparison ---
                ax2 = axes[0, 1]
                ax2.plot(x_wall_th, P_diff_SP, 'b-', linewidth=1.5, label="Small Pert", alpha=0.7)
                ax2.plot(x_wall_th, P_diff_SE, 'r-', linewidth=1.5, label="Shock-Exp", alpha=0.7)
                ax2.plot(x_wall_th, P_diff_combined, 'g-', linewidth=2.5, label="Combined")
                ax2.set_title(r"$|P_{theory} - P_{RANS}| / P_{RANS}$ [%]", fontsize=14)
                ax2.set_xlabel("X [m]", fontsize=12)
                ax2.set_ylabel("Error [%]", fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=10)
                ax2.set_xlim([0, l])
                
                # --- Panel 3: Blending weight ---
                """
                TEACHING POINT - Why plot the blending weight?
                -----------------------------------------------
                This shows you WHERE the combined method is trusting SPT 
                vs SE. At compression peaks (high pressure), w → 1 (trust SE).
                Near freestream pressure, w → 0 (trust SPT).
                
                If the combined method isn't improving accuracy, look at this 
                plot: it tells you whether the blending is activating in the 
                right places.
                """
                ax3 = axes[1, 0]
                delta_SP = P_wall_SP - p_infty
                w = np.tanh(10.0 * np.abs(delta_SP) / p_infty)
                
                ax3.plot(x_wall_th, w, 'g-', linewidth=2)
                ax3.fill_between(x_wall_th, 0, w, alpha=0.15, color='green')
                ax3.axhline(y=0.5, linestyle='--', color='gray', alpha=0.5)
                ax3.set_title("Blending Weight: w(x)", fontsize=14)
                ax3.set_xlabel("X [m]", fontsize=12)
                ax3.set_ylabel("w  (0=SPT, 1=SE)", fontsize=12)
                ax3.set_ylim([-0.05, 1.05])
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim([0, l])
                
                # --- Panel 4: Local Mach from SE ---
                ax4 = axes[1, 1]
                ax4.plot(SE_result.x, SE_result.Mach, 'r-', linewidth=2)
                ax4.axhline(y=M_infty, linestyle='--', color='gray', alpha=0.5,
                            label=f'$M_\\infty$ = {M_infty}')
                ax4.axhline(y=1.0, linestyle=':', color='black', alpha=0.3, label='M = 1')
                ax4.set_title("Local Mach (Shock-Expansion)", fontsize=14)
                ax4.set_xlabel("X [m]", fontsize=12)
                ax4.set_ylabel("Mach Number", fontsize=12)
                ax4.grid(True, alpha=0.3)
                ax4.legend(fontsize=10)
                
                plt.suptitle(f"{case_key}", fontsize=18, fontweight='bold', y=1.01)
                plt.tight_layout()
                plt.show()
            
            # ============================================================
            # STORE RESULTS
            # ============================================================
            
            axialForceScaled[case_key] = F_RANS / F_SP if F_SP != 0 else np.nan
            axialForceScaled_SE[case_key] = F_RANS / F_SE if F_SE != 0 else np.nan
            axialForceScaled_combined[case_key] = F_RANS / F_combined if F_combined != 0 else np.nan
            
            # % differences
            diff_SP = (1 - F_RANS / F_SP) * 100 if F_SP != 0 else np.nan
            diff_SE = (1 - F_RANS / F_SE) * 100 if F_SE != 0 else np.nan
            diff_combined = (1 - F_RANS / F_combined) * 100 if F_combined != 0 else np.nan
            
            results_list.append({
                'h/l': h_l,
                'M_infty': M_infty,
                'F_axial_RANS [N/m]': F_RANS,
                'F_axial_SmallPert [N/m]': F_SP,
                'F_axial_ShockExp [N/m]': F_SE,
                'F_axial_Combined [N/m]': F_combined,
                'Diff_SmallPert [%]': diff_SP,
                'Diff_ShockExp [%]': diff_SE,
                'Diff_Combined [%]': diff_combined,
                'Avg_P_Error_SP [%]': np.nanmean(P_diff_SP),
                'Avg_P_Error_SE [%]': np.nanmean(P_diff_SE),
                'Avg_P_Error_Combined [%]': np.nanmean(P_diff_combined),
                'Case_Key': case_key
            })
            
            # ============================================================
            # EXPORT
            # ============================================================
            
            df_results = pd.DataFrame(results_list)
            
            pivot_SP = df_results.pivot_table(
                values='Diff_SmallPert [%]', index='h/l', columns='M_infty', aggfunc='mean')
            pivot_SE = df_results.pivot_table(
                values='Diff_ShockExp [%]', index='h/l', columns='M_infty', aggfunc='mean')
            pivot_combined = df_results.pivot_table(
                values='Diff_Combined [%]', index='h/l', columns='M_infty', aggfunc='mean')
            
            save_dir = r'C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\reports\figures\Mach Study'
            df_results.to_csv(f'{save_dir}\\axial_force_comparison_3theories.csv', index=False)
            pivot_SP.to_csv(f'{save_dir}\\pivot_smallPert.csv')
            pivot_SE.to_csv(f'{save_dir}\\pivot_shockExp.csv')
            pivot_combined.to_csv(f'{save_dir}\\pivot_combined.csv')
            
            plt.show()
    
    return axialForceScaled, axialForceScaled_SE, axialForceScaled_combined






#%%



# === Separation/Attachment from sign of Tau_x (no splines), ignoring edge pairs ===
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- helpers ---
def get_hl(section_key: str):
    # First try numeric pattern: h_l_0.180
    m = re.search(r'h_l_([0-9.]+)', section_key)
    if m:
        return float(m.group(1))
    
    # Then check for h_l_x pattern (return a placeholder value or string)
    if 'h_l_x' in section_key:
        return 'x'  # or return a specific numeric value if you know what h/L this represents
    
    return None



def x_at_zero(i, x, y):# What is this ????
    """Linear interpolation of zero between samples i and i+1."""
    x0, x1 = x[i], x[i+1]
    y0, y1 = y[i], y[i+1]
    if y1 == y0:
        return x0  # plateau fallback
    t = y0 / (y0 - y1)
    return x0 + t * (x1 - x0)

# --- Pre-Allocating Variables  --- #
idx_separation = {}
x_sep_points = {}
sep_location = {}
tau_w_zeros = {}

sep_length = {}
sep_length_nonDim = {}

x_attach = {}
y_attach = {}


x_sep = {}
y_sep = {}


def find_sepLength(ds_by_case,x,tau_x, plot_results = False):
    # --- main loop ---
    for section_key in ds_by_case:
        if get_hl(section_key) is None:
            continue
    
    
        # sign of tau: negative segments define separated regions
        
        neg = tau_x[section_key] < 0
        edge = np.diff(neg.astype(np.int8))
    
        # +1: +→- (enter negative) = SEP,  -1: -→+ (exit negative) = ATTACH
        i_sep    = np.where(edge == +1)[0]
        i_attach = np.where(edge == -1)[0]
    
        # interpolate exact zero x-positions
        x_sep_i    = np.array([x_at_zero(i, x[section_key], tau_x[section_key]) for i in i_sep], dtype=float)
        x_attach_i = np.array([x_at_zero(i, x[section_key], tau_x[section_key]) for i in i_attach], dtype=float)
    
        # handle boundary negatives (start/end inside a negative interval)
        if neg[0]:
            if x_sep_i.size == 0 or (x_attach_i.size and x_attach_i[0] < x_sep_i[0]):
                x_sep_i = np.r_[x[section_key][0], x_sep_i]
        if neg[-1]:
            if x_attach_i.size == 0 or (x_sep_i.size and x_sep_i[-1] > x_attach_i[-1]):
                x_attach_i = np.r_[x_attach_i, x[section_key][-1]]
    
        # pair in order (guaranteed alternating by construction)
        n = min(x_sep_i.size, x_attach_i.size)
        x_sep_i, x_attach_i = x_sep_i[:n], x_attach_i[:n]
    
        # --------- NEW: drop first/last pairs if they touch the domain edges ---------
        if n > 0:
            xmin, xmax = x[section_key][0], x[section_key][-1]
            xrng = max(xmax - xmin, 1.0)
            tol = 1e-6 * xrng  # tolerance for "at the boundary"
    
            # drop leading boundary pair
            if abs(x_sep_i[0] - xmin) <= tol or abs(x_attach_i[0] - xmin) <= tol:
                x_sep_i    = x_sep_i[1:]
                x_attach_i = x_attach_i[1:]
                n = min(x_sep_i.size, x_attach_i.size)
    
            # drop trailing boundary pair
            if n > 0 and (abs(x_sep_i[-1] - xmax) <= tol or abs(x_attach_i[-1] - xmax) <= tol):
                x_sep_i    = x_sep_i[:-1]
                x_attach_i = x_attach_i[:-1]
                n = min(x_sep_i.size, x_attach_i.size)
        # ---------------------------------------------------------------------------
    
        # store sets
        x_sep[section_key]    = x_sep_i
        y_sep[section_key]    = np.zeros_like(x_sep_i)
        x_attach[section_key] = x_attach_i
        y_attach[section_key] = np.zeros_like(x_attach_i)
    
        # all zero-locations (for plotting like before)
        if n > 0:
            x0_all = np.sort(np.concatenate([x_sep_i, x_attach_i]))
        else:
            x0_all = np.array([], dtype=float)
        sep_location[section_key] = x0_all
        tau_w_zeros[section_key]  = np.zeros_like(x0_all)
    
        # separation length (sum of SEP→ATTACH spans)
        sep_len = float(np.sum(np.abs(x_sep_i - x_attach_i))) if n > 0 else 0.0
        sep_length[section_key] = sep_len
        sep_length_nonDim[section_key] = (sep_len / x[section_key][-1]) if x[section_key].size else np.nan
    
        # optional: index-based separation mask in cleaned data
        idx_separation[section_key] = np.where(tau_x[section_key] <= 0)[0]
    
        # keep alias if you use it elsewhere
        x_sep_points[section_key] = x_sep_i
    
        # ---- Plotting results  ----
        
        if plot_results:
            fig , ax = plotter(x[section_key],tau_x[section_key],"X",r"$\tau_x$",'[m]','[Pa]', return_axes = True)
            ax.axhline(y = 0, linestyle = '--', color = 'Black')
            
            # plotting the scatter plot # 
            if x_sep_i.size:
                ax.scatter(x_sep_i, np.zeros_like(x_sep_i), color='red', s=36, label='SEP', zorder=3)
            if x_attach_i.size:
                ax.scatter(x_attach_i , np.zeros_like(x_attach_i), color='green', s=36, label='ATTACH', zorder=3)
            
            
          
        #plt.axhline(0, linestyle='--', color='black', label='Separation Line')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
        #plt.grid(True, which="both")
        #plt.title(rf"$\tau_x$ vs X [in]: {section_key}")
        #plt.ylabel(r"$\tau_x$ [Pa]")
        #plt.xlabel("X [m]")
        #plt.tight_layout()
        #plt.show()
    return sep_length, sep_length_nonDim, x_sep, y_sep, x_attach, y_attach
















"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                Finding MAXIMAS AND MINIMAS OF THE GEOMETRY 
#------------------------------------------------------------------------------------------------------------------------------------#
"""
# Prior to plotting results, I am going to find the maximas and minimas to accurately represent the separation length
# The definition of the separation length will be the separation length of the second wave
# To do so, the maximas and minimas of each respective geometry will be found and evaluated.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks  # <-- simple peak/valley finder

# --- Pre-allocating Dictionaries ---

# In mm
y_max, x_max = {}, {}
y_min, x_min = {}, {}

# In inches
x_max_in, y_max_in = {}, {}
x_min_in, y_min_in = {}, {}

# Unit conversion



def max_min_finder(ds_by_case,x,y):
# Using a for loop to find the y_max and x_max of each respective geometry
    for section_key in ds_by_case:
  
        # SIMPLE peak picking
        i_max, _ = find_peaks(y[section_key])       # local maxima
        i_min, _ = find_peaks(-y[section_key])      # local minima
    
        # (If you need a tiny bit more robustness, uncomment one of these one-liners)
        # i_max, _ = find_peaks(y_all, distance=20)                       # enforce min spacing
        # i_max, _ = find_peaks(y_all, prominence=0.05*np.ptp(y_all))     # ignore tiny ripples
        # i_max, _ = find_peaks(y_all, plateau_size=1)                    # detect flat tops
        # i_min, _ = find_peaks(-y_all, distance=20)
        # i_min, _ = find_peaks(-y_all, prominence=0.05*np.ptp(y_all))
        # i_min, _ = find_peaks(-y_all, plateau_size=1)
    
    
        # Storing Results # 
        y_max[section_key] = y[section_key][i_max]
        x_max[section_key] = x[section_key][i_max] 
        #x_max[pressure_key] = 0.015
        y_min[section_key] = y[section_key][i_min]
        x_min[section_key] = x[section_key][i_min]
    return x_max, x_min, y_max, y_min














'''

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                Plotting Separation Length Vs Re_L
#------------------------------------------------------------------------------------------------------------------------------------#
"""







# Plot Re vs Lsep/Lwidth with markers colored by Mach and h/L in the legend
# Plot Re vs Lsep/Lwidth with markers colored by Mach (1.5–4.5).
# h/L groups are shown as separate lines with a legend.

# Re vs Lsep/Lwidth with markers colored by Mach (1.5…4.5) and lines per h/L.

# This version avoids regex brittleness by matching known Mach tags directly.


import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable



# ---------------- helpers ---------------- #
def get_hl_Mach(key: str):
    # Handles negative numbers, decimals, scientific notation
    m = re.search(r'h[_-]?l[_-]?(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', key, flags=re.I)
    return float(m.group(1)) if m else None

# Simple & robust Mach extractor:
# checks for 'mach_1.5', 'mach_2.0', … and also 'mach_1_5' style.
MACH_LEVELS = np.arange(1.5, 4.5 , 0.5)  # 1.5, 2.0, …, 4.5



def extract_mach_from_filename(key: str, mach_levels: list) -> float:
    """
    Extract Mach number from filename.
    
    Args:
        key: String containing Mach number (e.g., 'h_l_0.030_Mach_1.0' or 'h_l_0.030_Mach_1_5')
        mach_levels: List of possible Mach values to search for
    
    Returns:
        float: Extracted Mach number or np.nan if not found
    
    Examples:
        >>> extract_mach_from_filename("h_l_0.030_Mach_1.0", [1.0, 1.5, 2.0])
        1.0
        >>> extract_mach_from_filename("h_l_0.030_Mach_2_5", [1.0, 1.5, 2.0, 2.5])
        2.5
    """
    s = key.lower()
    
    for mv in mach_levels:
        # Create pattern variations
        # For mv=2.5: creates "mach_2.5" and "mach_2_5"
        # For mv=1.0: creates "mach_1.0" and "mach_1_0"
        tag_dot = f"mach_{mv:.1f}"        # e.g., "mach_2.5"
        tag_us = tag_dot.replace(".", "_") # e.g., "mach_2_5"
        
        # Check if either pattern exists in the filename
        if tag_dot in s or tag_us in s:
            return float(f"{mv:.1f}")
    
    return np.nan







def Re_sepLength(ds_by_case, x,y,Re):

    # ---------------- filter + group by h/L ----------------
    min_hl, max_hl = 0.02, 0.09
    
    keys = ds_by_case.keys()
    filtered_keys = [
        k for k in keys
        if (get_hl(k) is not None and min_hl <= get_hl(k) <= max_hl and "Mach_0.5" not in k)
    ]
    
    groups = defaultdict(list)
    for k in filtered_keys:
        groups[get_hl(k)].append(k)
    
        # ---------------- discrete Mach mapping for colorbar ----------------
        # Create discrete bins with boundaries half-way between the levels.
        M0_bounds = np.linspace(1.5, 4.0, len(MACH_LEVELS) + 1)
        cmap_mach = cm.get_cmap("viridis", len(MACH_LEVELS))   # 7 distinct colors
        norm = BoundaryNorm(M0_bounds, cmap_mach.N)
        
        
        
        # line colors per h/L (legend)
        cmap_lines = cm.get_cmap("plasma", len(groups))
        
        # ---------------- plotting ----------------
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_ref = None
        unmatched = 0
        
    for idx, section_key in enumerate(ds_by_case):
        
        # Getting x run # 
        x_run = x[section_key]
        y_run = y[section_key]

        #Plotting # 
        ax.plot(x, y , color=cmap_lines(i), lw=4, label=f"h/L = {hl:.2f}")
    
        # markers: color by Mach  #
        mask_col = mask_xy & np.isfinite(ms)
        if np.any(mask_col):
            sc = ax.scatter(xs[mask_col], ys[mask_col], c=ms[mask_col],
                            cmap=cmap_mach, norm=norm,
                            s=70, marker='o', edgecolor='k', linewidths= 1.35,
                            zorder=5, alpha=0.98)
            scatter_ref = sc
            

            
            
            
        # (Optional) show unmatched as neutral markers so you can spot them
        mask_neu = mask_xy & ~np.isfinite(ms)
        if np.any(mask_neu):
            ax.scatter(xs[mask_neu], ys[mask_neu],
                       color='white', edgecolor='k', linewidths=0.35,
                       s=46, marker='o', zorder=4, alpha=0.9)
    
    # ---------------- colorbar with exact Mach ticks ----------------
    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=ax, pad=0.02, ticks=MACH_LEVELS)
        cbar.set_label("Mach Number", fontsize = 18)
        cbar.ax.tick_params(labelsize = 18)
    else:
        # fallback colorbar so layout stays stable; also tell you why
        sm = ScalarMappable(norm=norm, cmap=cmap_mach); sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02, ticks=MACH_LEVELS).set_label("Mach")
        print("Warning: no markers received a Mach color. Check key naming.")
    
    
    
    
    # ---------------- cosmetics ----------------
    ax.set_title("Normalized Separation Length vs Reynolds Number", fontsize = 24, pad = 15)
    ax.set_xlabel("Reynolds Number", fontsize = 21)
    ax.set_ylabel(r"$L_{separation}/L_{Length}$", fontsize = 21)
    ax.tick_params(labelsize = 14)
    
    ax.grid(True, which="both")
    ax.legend(title="Cases", loc = 'best', bbox_to_anchor = (0.4, 0.3), fontsize = 12)
    fig.tight_layout()
    plt.show()
    
    # Optional: see how many keys didn’t contain a Mach tag
    if unmatched:
        print(f"{unmatched} case(s) had no recognizable Mach tag (e.g., 'mach_2.5').")
'''        
    




"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                                Plotting Separation Length Vs Mach Number
#------------------------------------------------------------------------------------------------------------------------------------#
"""

import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl





# ---------------- helpers ---------------- #
def get_hl(key: str):
    m = re.search(r'h[_-]?l[_-]?(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', key, flags=re.I)
    return float(m.group(1)) if m else None

MACH_LEVELS = np.arange(1.5, 4.5, 0.5)  # 1.5, 2.0, …, 4.0

def extract_mach_from_filename(key: str, mach_levels: list) -> float:
    s = key.lower()
    for mv in mach_levels:
        tag_dot = f"mach_{mv:.1f}"
        tag_us = tag_dot.replace(".", "_")
        if tag_dot in s or tag_us in s:
            return float(f"{mv:.1f}")
    return np.nan


def mach_vs_sepLength(ds_by_case, x, y, sep_length_nonDim):

    # ---------------- publication style ----------------
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 21
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.titlesize'] = 21
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['grid.linewidth'] = 0.5
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 600

    # ---------------- filter + group by h/L ----------------
    min_hl, max_hl = 0.02, 0.09

    keys = ds_by_case.keys()
    filtered_keys = [
        k for k in keys
        if (get_hl(k) is not None and min_hl <= get_hl(k) <= max_hl and "Mach_0.5" not in k)
    ]

    groups = defaultdict(list)
    for k in filtered_keys:
        groups[get_hl(k)].append(k)

    # ---------------- style definitions ----------------
    hl_values = sorted(groups.keys())
    cmap_lines = cm.get_cmap("viridis", len(hl_values))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    # ---------------- plotting ----------------
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, hl_val in enumerate(hl_values):
        group_keys = groups[hl_val]

        machs = []
        seps = []
        for k in group_keys:
            m = extract_mach_from_filename(k, MACH_LEVELS)
            if np.isnan(m):
                continue
            if k not in sep_length_nonDim:
                continue
            machs.append(m)
            seps.append(sep_length_nonDim[k])

        if len(machs) == 0:
            continue

        order = np.argsort(machs)
        machs = np.array(machs)[order]
        seps = np.array(seps)[order]

        ax.plot(machs, seps,
                color=cmap_lines(i),
                marker=markers[i % len(markers)],
                markersize=6,
                markeredgecolor='black',
                markeredgewidth=0.6,
                label=f"h/l = {hl_val:.2f}")

    # ---------------- formatting (no hardcoded fontsizes) ----------------
    ax.set_xlabel("Mach Number")
    ax.set_ylabel(r"$L_{\mathrm{sep}} / L_{\mathrm{width}}$")
    ax.set_title("Normalized Separation Length vs Mach Number", fontweight='bold')
    ax.set_xticks(MACH_LEVELS)
    ax.set_xlim([MACH_LEVELS[0] - 0.2, MACH_LEVELS[-1] + 0.2])
    ax.grid(True, alpha=0.3)
    ax.legend(title="h/l", loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0)

    fig.tight_layout()
    plt.savefig('sep_length_vs_mach.png', bbox_inches='tight')
    plt.show()

    return machs