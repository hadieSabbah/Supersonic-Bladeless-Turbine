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