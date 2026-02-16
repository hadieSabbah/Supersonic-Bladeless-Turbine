"""
#=====================================================================================================#
     URANS Wall Pressure Evolution Extractor (PyTecplot Version)
     ------------------------------------------------------------
     Reads time-step .bin files from iCFD++ URANS simulations using PyTecplot,
     extracts wall pressure as a function of streamwise position (x) and time step,
     and produces a space-time carpet plot of wall pressure evolution.

     Author: HS  |  Date: Feb 2026

     WHAT THIS SCRIPT DOES:
     ----------------------
     1. Discovers all .bin time-step files in a specified directory
     2. Loads each file with PyTecplot (which handles the binary format automatically)
     3. Identifies the wall boundary nodes (bottom of the domain, i.e., lowest-y nodes)
     4. Extracts wall pressure at each time step
     5. Produces multiple visualizations:
        a) Space-time carpet plot (x vs time step, colored by pressure)
        b) Animated-style line plots of wall pressure vs x at selected times
        c) Pressure history at selected x-locations

     HOW TO USE THIS SCRIPT:
     -----------------------
     This script runs in CONNECTED MODE:
       1. Open Tecplot 360 2022 R2
       2. Go to Scripting → PyTecplot Connections...
       3. Check "Accept connections" (default port 7600)
       4. Run this script:  python carpetcode.py
          (Do NOT use python -O — that's for batch mode which we're not using)

     EDUCATIONAL NOTE — Connected vs Batch Mode:
     ---------------------------------------------
     CONNECTED MODE (what we use here):
       - Sends commands to a running Tecplot 360 instance over a socket (port 7600)
       - You can watch the data load in the GUI — great for debugging
       - No need to find/configure Tecplot library paths
       - Slightly slower due to socket overhead, but perfectly fine for local work

     BATCH MODE (python -O script.py):
       - Loads Tecplot's engine DLLs directly into the Python process
       - Faster, no GUI needed, good for HPC/cluster automation
       - Requires TEC360HOME environment variable or correct library paths
       - This is what caused your FileNotFoundError (missing mingw-w64 path)

     DEPENDENCIES: pytecplot, numpy, matplotlib
#=====================================================================================================#
"""

import tecplot as tp

# ── CONNECT TO RUNNING TECPLOT 360 ──
# This MUST come before any tp.data or tp.new_layout() calls.
# Make sure Tecplot 360 is open with PyTecplot Connections enabled (port 7600).
#
# TEACHING POINT — Why connect() first?
#   In connected mode, every tp.* call is forwarded to the running Tecplot instance.
#   If you call tp.new_layout() before connect(), PyTecplot tries to start its own
#   engine (batch mode), which is what caused your FileNotFoundError.
#   By connecting first, all subsequent calls go through the socket instead.
tp.session.connect(port=7600)

from tecplot.constant import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import os
import glob
import re
from pathlib import Path
import sys


#=====================================================================================================#
#                                    USER CONFIGURATION
#=====================================================================================================#
# ── Modify these settings to match YOUR data ──

# Path to the directory containing your .bin files
DATA_DIR = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\data\raw\Unsteady Data\tec_files3"

# File pattern for the time-step files
FILE_PATTERN = "*.bin"

# Output directory for saved figures
OUTPUT_DIR = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\2_SBTTD Data Analysis Code\UnsteadyCode\2_Results"

# ── Variable names (as they appear in Tecplot) ──
# These MUST match the variable names in your .bin files exactly.
# If you're unsure, run the DIAGNOSTIC section first (DIAGNOSTIC_ONLY = True).
X_VAR = "X"
Y_VAR = "Y"
PRESSURE_VAR = "P"

# ── Wall identification settings ──
# Method to find wall nodes: 'min_y' or 'zone_name'
#   'min_y':      Identifies wall as the set of nodes with the smallest y-values
#   'zone_name':  Looks for a specific zone name that corresponds to the wall boundary
#
# FROM DIAGNOSTICS: Zone [3] 'Section' is the wall boundary (U=V=0, no-slip).
#   It has 1000 nodes along the wavy wall surface — perfect for direct extraction.
WALL_METHOD = "zone_name"

# If WALL_METHOD = 'zone_name', specify the wall zone name (or part of it):
WALL_ZONE_NAME = "Section"  # Case-insensitive partial match

# Tolerance for identifying wall nodes when using 'min_y' method.
# Nodes within Y_TOLERANCE of the minimum y-coordinate are considered "on the wall."
# For your wavy wall, this should be small enough to capture only the bottom boundary.
Y_TOLERANCE = 1e-6  # [meters]

# ── Region of interest ──
# Your domain spans x = [-0.01, 0.2] m, but the wavy wall is only from 0 to 0.1 m.
# Set these to crop the extracted data to just the region you care about.
# Set both to None to keep the full domain.
X_MIN = 0.0    # [meters] — start of region of interest
X_MAX = 0.1    # [meters] — end of region of interest
NORMALIZE_PRESSURE = False
P_INF = 101325.0  # Freestream pressure [Pa] — only used if NORMALIZE_PRESSURE = True

# ── Time step settings ──
# iCFD++ does not embed solution time in the Tecplot files (all show 0.0).
# We reconstruct physical time from the file numbering and your simulation dt.
# Set DT_PHYSICAL = None to just use step index on the time axis.
#
# TEACHING POINT — Reconstructing physical time:
#   Your files are named mcfd_tec_10.bin, mcfd_tec_20.bin, ..., mcfd_tec_3500.bin
#   The number in the filename is the ITERATION number, not the time-step count.
#   If your simulation writes output every N_OUTPUT iterations with time step dt:
#     physical_time = iteration_number * dt
#   Set these values from your iCFD++ input file.
DT_PHYSICAL = None      # Physical time step [s] per iteration (e.g., 1e-7). Set None to skip.
N_OUTPUT = 10           # Output frequency (iterations between file writes)

# ── Plotting settings ──
DPI = 200
COLORMAP = "RdBu_r"
FIGSIZE_CARPET = (12, 6)
FIGSIZE_LINES = (10, 6)


#=====================================================================================================#
#                              PART 1: FILE DISCOVERY & SORTING
#=====================================================================================================#

def discover_files(data_dir, pattern):
    """
    Find all time-step files and sort them in proper numerical order.

    TEACHING POINT — Natural Sorting:
    ---------------------------------
    Simple alphabetical sort gives: file_1, file_10, file_100, file_2, file_20, ...
    We need NATURAL sort:           file_1, file_2, ..., file_10, ..., file_20, ..., file_100

    The trick is to extract the numeric part and sort by that integer value.
    """
    search_path = os.path.join(data_dir, pattern)
    files = glob.glob(search_path)

    if len(files) == 0:
        print(f"ERROR: No files found matching '{search_path}'")
        print(f"  Check that DATA_DIR and FILE_PATTERN are correct.")
        print(f"  Current DATA_DIR: {data_dir}")
        sys.exit(1)

    def natural_sort_key(filepath):
        """Extract numbers from filename for natural sorting."""
        basename = os.path.basename(filepath)
        numbers = re.findall(r'\d+', basename)
        if numbers:
            return int(numbers[-1])  # sort by the LAST number (usually the time step)
        return 0

    files.sort(key=natural_sort_key)

    print(f"Found {len(files)} files:")
    print(f"  First: {os.path.basename(files[0])}")
    print(f"  Last:  {os.path.basename(files[-1])}")

    return files


#=====================================================================================================#
#                              PART 2: LOAD DATA & EXTRACT WALL PRESSURE
#=====================================================================================================#

def get_wall_pressure_from_file(filepath, is_first_file=False):
    """
    Load a single .bin file and extract wall pressure vs x-position.

    TEACHING POINT — PyTecplot Data Loading:
    -----------------------------------------
    tp.data.load_tecplot() handles ALL Tecplot formats (.plt, .bin, .dat, .szplt)
    automatically. It detects binary vs ASCII, byte order, and version internally.

    TEACHING POINT — Why clear data between files?
    ------------------------------------------------
    PyTecplot keeps data in memory. If we don't clear between files, data accumulates
    and (a) uses tons of memory, (b) makes zone indexing confusing. We use
    tp.new_layout() to start fresh for each file.

    Parameters:
    -----------
    filepath : str
        Path to the .bin file
    is_first_file : bool
        If True, print diagnostic info about the dataset

    Returns:
    --------
    x_wall : np.ndarray — Streamwise coordinates of wall nodes, sorted by x
    p_wall : np.ndarray — Pressure values at wall nodes, sorted by x
    solution_time : float — Solution time associated with this file (if available)
    """

    # Clear any existing data
    tp.new_layout()

    # Load the file — PyTecplot auto-detects the format
    dataset = tp.data.load_tecplot(filepath)

    # ── Diagnostic output for the first file ──
    if is_first_file:
        print("\n" + "="*70)
        print("DATASET DIAGNOSTICS (from first file)")
        print("="*70)
        print(f"  File: {os.path.basename(filepath)}")
        print(f"  Number of variables: {dataset.num_variables}")
        print(f"  Variable names:")
        for i, var in enumerate(dataset.variables()):
            print(f"    [{i}] {var.name}")
        print(f"  Number of zones: {dataset.num_zones}")
        for i, zone in enumerate(dataset.zones()):
            ztype = zone.zone_type
            print(f"    [{i}] '{zone.name}' | Type: {ztype} | ", end="")
            if ztype == ZoneType.Ordered:
                dims = zone.dimensions
                print(f"Dims: {dims}")
            else:
                print(f"Nodes: {zone.num_points}, Elements: {zone.num_elements}")
        print("="*70 + "\n")

    # ── Find the correct zone ──
    target_zone = None
    wall_method_local = WALL_METHOD

    if WALL_METHOD == "zone_name":
        for zone in dataset.zones():
            if WALL_ZONE_NAME.lower() in zone.name.lower():
                target_zone = zone
                if is_first_file:
                    print(f"  Found wall zone: '{zone.name}'")
                break
        if target_zone is None:
            print(f"  WARNING: No zone matching '{WALL_ZONE_NAME}' found.")
            print(f"  Available zones: {[z.name for z in dataset.zones()]}")
            print(f"  Falling back to 'min_y' method...")
            wall_method_local = "min_y"

    # ── Extract wall data ──

    if wall_method_local == "zone_name" and target_zone is not None:
        # Direct extraction from boundary zone — all nodes in this zone are on the wall
        x_wall = np.array(target_zone.values(X_VAR)[:])
        p_wall = np.array(target_zone.values(PRESSURE_VAR)[:])

    else:
        # Use 'min_y' method: find the bottom-most nodes in the domain
        zone = dataset.zone(0)

        x_all = np.array(zone.values(X_VAR)[:])
        y_all = np.array(zone.values(Y_VAR)[:])
        p_all = np.array(zone.values(PRESSURE_VAR)[:])

        if zone.zone_type == ZoneType.Ordered:
            # ── STRUCTURED GRID (IJ-ORDERED) ──
            #
            # TEACHING POINT — Structured Grid Indexing:
            # For an IJ-ordered zone, data is stored as a 1D array: index = i + j * imax
            # The j=0 row (indices 0..imax-1) is typically the wall boundary.
            # j increases away from the wall into the freestream.

            imax = zone.dimensions[0]
            jmax = zone.dimensions[1]

            if is_first_file:
                print(f"  Structured grid: imax={imax}, jmax={jmax}")
                print(f"  Extracting j=0 row (wall) — {imax} points")

            # j=0 row
            wall_indices = np.arange(imax)

            # Verify j=0 is actually the wall (lowest y) not the freestream
            y_j0 = y_all[wall_indices]
            y_j_last = y_all[(jmax-1)*imax : jmax*imax]

            if np.mean(y_j0) > np.mean(y_j_last):
                if is_first_file:
                    print(f"  NOTE: j=0 is the top boundary. Using j={jmax-1} as wall.")
                wall_indices = np.arange((jmax-1)*imax, jmax*imax)

            x_wall = x_all[wall_indices]
            p_wall = p_all[wall_indices]

        else:
            # ── UNSTRUCTURED GRID ──
            y_min = np.min(y_all)

            # Simple approach first
            wall_mask = y_all <= (y_min + Y_TOLERANCE)

            if np.sum(wall_mask) < 10:
                # Too few points — wall has significant curvature, use binning
                if is_first_file:
                    print(f"  Using adaptive wall detection (binning approach)...")

                n_bins = 500
                x_min_global, x_max_global = np.min(x_all), np.max(x_all)
                x_edges = np.linspace(x_min_global, x_max_global, n_bins + 1)

                wall_node_indices = []
                for ib in range(n_bins):
                    in_bin = (x_all >= x_edges[ib]) & (x_all < x_edges[ib + 1])
                    if np.any(in_bin):
                        bin_indices = np.where(in_bin)[0]
                        y_in_bin = y_all[bin_indices]
                        local_min_idx = bin_indices[np.argmin(y_in_bin)]
                        wall_node_indices.append(local_min_idx)

                wall_node_indices = np.array(wall_node_indices)
                x_wall = x_all[wall_node_indices]
                p_wall = p_all[wall_node_indices]
            else:
                x_wall = x_all[wall_mask]
                p_wall = p_all[wall_mask]

            if is_first_file:
                print(f"  Unstructured grid: {zone.num_points} total nodes")
                print(f"  Wall nodes identified: {len(x_wall)}")

    # ── Sort by x-coordinate ──
    sort_idx = np.argsort(x_wall)
    x_wall = x_wall[sort_idx]
    p_wall = p_wall[sort_idx]

    # ── Crop to region of interest ──
    if X_MIN is not None or X_MAX is not None:
        mask = np.ones(len(x_wall), dtype=bool)
        if X_MIN is not None:
            mask &= x_wall >= X_MIN
        if X_MAX is not None:
            mask &= x_wall <= X_MAX
        x_wall = x_wall[mask]
        p_wall = p_wall[mask]

        if is_first_file:
            print(f"  Cropped to x ∈ [{X_MIN}, {X_MAX}] m → {len(x_wall)} wall points")

    # ── Get solution time ──
    # iCFD++ stores 0.0 for solution_time, so we derive it from the filename
    zone = dataset.zone(0)
    try:
        solution_time = zone.solution_time
    except:
        solution_time = 0.0

    # If solution_time is 0, try to reconstruct from the filename
    if solution_time == 0.0 and DT_PHYSICAL is not None:
        # Extract iteration number from filename (e.g., "mcfd_tec_3500.bin" → 3500)
        numbers = re.findall(r'\d+', os.path.basename(filepath))
        if numbers:
            iteration = int(numbers[-1])
            solution_time = iteration * DT_PHYSICAL

    # ── Normalize if requested ──
    if NORMALIZE_PRESSURE:
        p_wall = p_wall / P_INF

    return x_wall, p_wall, solution_time


def extract_all_timesteps(files):
    """
    Loop through all time-step files and collect wall pressure data.

    Returns:
    --------
    x_common : np.ndarray  (N_wall_points,)
    pressure_matrix : np.ndarray  (N_timesteps, N_wall_points)
    solution_times : np.ndarray  (N_timesteps,)
    """

    all_x = []
    all_p = []
    solution_times = []

    n_files = len(files)

    for i, fpath in enumerate(files):
        if (i % max(1, n_files // 20)) == 0 or i == n_files - 1:
            pct = 100 * (i + 1) / n_files
            print(f"  Processing file {i+1}/{n_files} ({pct:.0f}%): {os.path.basename(fpath)}")

        x_w, p_w, sol_t = get_wall_pressure_from_file(fpath, is_first_file=(i == 0))
        all_x.append(x_w)
        all_p.append(p_w)
        solution_times.append(sol_t)

    x_common = all_x[0]
    n_x = len(x_common)

    all_same = all(len(x) == n_x for x in all_x)

    if all_same:
        pressure_matrix = np.array(all_p)
        print(f"\n  All files have consistent wall point count: {n_x}")
    else:
        print(f"\n  WARNING: Wall point count varies across files — interpolating to common grid.")
        pressure_matrix = np.zeros((n_files, n_x))
        pressure_matrix[0, :] = all_p[0]
        for i in range(1, n_files):
            pressure_matrix[i, :] = np.interp(x_common, all_x[i], all_p[i])

    solution_times = np.array(solution_times)

    return x_common, pressure_matrix, solution_times


#=====================================================================================================#
#                              PART 3: VISUALIZATION
#=====================================================================================================#

def plot_carpet(x_wall, pressure_matrix, solution_times, output_dir):
    """
    Space-time carpet plot: x on horizontal axis, time step on vertical axis,
    color = wall pressure.

    TEACHING POINT — Why a carpet plot?
    ------------------------------------
    A carpet plot lets you see how the entire wall pressure distribution evolves
    simultaneously. Key features to look for:
      - Vertical bands → standing features (shocks anchored to wave crests)
      - Diagonal streaks → traveling features (propagating pressure waves)
      - Convergence → the pattern stops changing once the flow reaches steady state
      - Oscillations → unsteadiness that persists (characteristic of URANS)
    """

    fig, ax = plt.subplots(figsize=FIGSIZE_CARPET)

    n_t, n_x = pressure_matrix.shape

    if np.all(solution_times == 0):
        y_axis = np.arange(n_t)
        y_label = "Time Step Index"
    else:
        y_axis = solution_times
        y_label = "Solution Time [s]"

    # pcolormesh expects cell edges, not centers
    dx = np.diff(x_wall)
    x_edges = np.concatenate([[x_wall[0] - dx[0]/2],
                               x_wall[:-1] + dx/2,
                               [x_wall[-1] + dx[-1]/2]])

    if len(y_axis) > 1:
        dy = np.diff(y_axis)
        y_edges = np.concatenate([[y_axis[0] - dy[0]/2],
                                   y_axis[:-1] + dy/2,
                                   [y_axis[-1] + dy[-1]/2]])
    else:
        y_edges = np.array([y_axis[0] - 0.5, y_axis[0] + 0.5])

    pcm = ax.pcolormesh(x_edges * 1000, y_edges, pressure_matrix,
                        cmap=COLORMAP, shading='flat')

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    p_label = "P / P∞" if NORMALIZE_PRESSURE else "Pressure [Pa]"
    cbar.set_label(p_label, fontsize=12)

    ax.set_xlabel("x [mm]", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title("Wall Pressure Evolution (URANS)", fontsize=14)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "carpet_plot.png"), dpi=DPI, bbox_inches='tight')
    print(f"  Saved: carpet_plot.png")
    plt.close(fig)


def plot_selected_timesteps(x_wall, pressure_matrix, solution_times, output_dir, n_lines=8):
    """
    Overlay wall pressure profiles at selected time steps.
    Shows how the pressure distribution converges (or oscillates) over time.
    """

    fig, ax = plt.subplots(figsize=FIGSIZE_LINES)

    n_t = pressure_matrix.shape[0]
    indices = np.linspace(0, n_t - 1, n_lines, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, n_lines))

    for k, idx in enumerate(indices):
        if np.all(solution_times == 0):
            label = f"Step {idx}"
        else:
            label = f"t = {solution_times[idx]:.4g} s"

        ax.plot(x_wall * 1000, pressure_matrix[idx, :],
                color=colors[k], linewidth=1.2, label=label, alpha=0.85)

    p_label = "P / P∞" if NORMALIZE_PRESSURE else "Pressure [Pa]"
    ax.set_xlabel("x [mm]", fontsize=12)
    ax.set_ylabel(p_label, fontsize=12)
    ax.set_title("Wall Pressure at Selected Time Steps", fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "pressure_profiles.png"), dpi=DPI, bbox_inches='tight')
    print(f"  Saved: pressure_profiles.png")
    plt.close(fig)


def plot_pressure_history(x_wall, pressure_matrix, solution_times, output_dir, n_probes=5):
    """
    Pressure vs time at selected x-locations along the wall.
    Shows whether the pressure has reached a periodic steady state.
    """

    fig, ax = plt.subplots(figsize=FIGSIZE_LINES)

    n_t = pressure_matrix.shape[0]
    x_min, x_max = x_wall[0], x_wall[-1]
    x_probes = np.linspace(x_min + 0.05 * (x_max - x_min),
                            x_max - 0.05 * (x_max - x_min), n_probes)

    if np.all(solution_times == 0):
        t_axis = np.arange(n_t)
        t_label = "Time Step Index"
    else:
        t_axis = solution_times
        t_label = "Solution Time [s]"

    colors = plt.cm.tab10(np.linspace(0, 1, n_probes))

    for k, xp in enumerate(x_probes):
        idx = np.argmin(np.abs(x_wall - xp))
        p_history = pressure_matrix[:, idx]

        ax.plot(t_axis, p_history, color=colors[k], linewidth=1.0,
                label=f"x = {x_wall[idx]*1000:.1f} mm", alpha=0.85)

    p_label = "P / P∞" if NORMALIZE_PRESSURE else "Pressure [Pa]"
    ax.set_xlabel(t_label, fontsize=12)
    ax.set_ylabel(p_label, fontsize=12)
    ax.set_title("Pressure History at Selected Wall Locations", fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "pressure_history.png"), dpi=DPI, bbox_inches='tight')
    print(f"  Saved: pressure_history.png")
    plt.close(fig)


def plot_convergence_check(pressure_matrix, solution_times, output_dir):
    """
    Plot the RMS pressure change between consecutive time steps.

    TEACHING POINT — Convergence Check:
    For URANS, "convergence" doesn't mean the solution stops changing (it oscillates).
    Instead, the AMPLITUDE of changes should stabilize. If the RMS settles to a
    roughly constant value, the flow has reached periodic behavior. If it's still
    decreasing monotonically, the simulation may not yet be fully developed.
    """

    n_t = pressure_matrix.shape[0]

    if n_t < 2:
        print("  Skipping convergence plot (need at least 2 time steps)")
        return

    dp_norm = np.zeros(n_t - 1)
    for i in range(n_t - 1):
        dp = pressure_matrix[i+1, :] - pressure_matrix[i, :]
        dp_norm[i] = np.sqrt(np.mean(dp**2))

    fig, ax = plt.subplots(figsize=(10, 4))

    if np.all(solution_times == 0):
        t_axis = np.arange(1, n_t)
        t_label = "Time Step Index"
    else:
        t_axis = solution_times[1:]
        t_label = "Solution Time [s]"

    ax.semilogy(t_axis, dp_norm, 'k-', linewidth=1.0)
    ax.set_xlabel(t_label, fontsize=12)
    ax.set_ylabel("RMS Pressure Change [Pa]", fontsize=12)
    ax.set_title("Wall Pressure Convergence Check", fontsize=14)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence_check.png"), dpi=DPI, bbox_inches='tight')
    print(f"  Saved: convergence_check.png")
    plt.close(fig)


#=====================================================================================================#
#                              PART 4: DIAGNOSTIC MODE
#=====================================================================================================#

def run_diagnostic(data_dir, pattern):
    """
    Load the FIRST file only and print all available information.
    Use this to confirm variable names, zone names, and grid structure
    before running the full extraction.
    """
    files = discover_files(data_dir, pattern)
    filepath = files[0]

    print("\n" + "="*70)
    print("DIAGNOSTIC MODE — Inspecting first file")
    print("="*70)
    print(f"File: {filepath}\n")

    tp.new_layout()
    dataset = tp.data.load_tecplot(filepath)

    print(f"Number of variables: {dataset.num_variables}")
    print(f"Variable names:")
    for i, var in enumerate(dataset.variables()):
        print(f"  [{i}] '{var.name}'")

    print(f"\nNumber of zones: {dataset.num_zones}")
    for i, zone in enumerate(dataset.zones()):
        ztype = zone.zone_type
        print(f"\n  Zone [{i}]: '{zone.name}'")
        print(f"    Type: {ztype}")

        if ztype == ZoneType.Ordered:
            imax, jmax, kmax = zone.dimensions
            print(f"    Dimensions: I={imax}, J={jmax}, K={kmax}")
            print(f"    Total nodes: {imax * jmax * kmax}")
        else:
            print(f"    Nodes: {zone.num_points}")
            print(f"    Elements: {zone.num_elements}")

        try:
            print(f"    Solution time: {zone.solution_time}")
            print(f"    Strand ID: {zone.strand}")
        except:
            print(f"    Solution time: N/A")

        # Print min/max of first few variables
        print(f"    Variable ranges:")
        for j, var in enumerate(dataset.variables()):
            vals = zone.values(var.name)[:]
            if len(vals) > 0:
                print(f"      {var.name}: [{np.min(vals):.6g}, {np.max(vals):.6g}]")
            if j >= 8:
                print(f"      ... ({dataset.num_variables - 9} more variables)")
                break

    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("  Copy the exact variable names above into the USER CONFIGURATION section.")
    print("  Look for variables like 'X', 'Y', 'Pressure' or 'P' or 'Static_Pressure'.")
    print("  If you see a zone named 'wall' or 'boundary', set WALL_METHOD = 'zone_name'.")
    print("="*70)


#=====================================================================================================#
#                              PART 5: MAIN EXECUTION
#=====================================================================================================#

def main():
    """
    Main execution flow.
    Set DIAGNOSTIC_ONLY = True to inspect one file first.
    Set DIAGNOSTIC_ONLY = False for full processing.
    """

    # ┌─────────────────────────────────────────────────┐
    # │  Set this to True to inspect the first file     │
    # │  before running the full extraction              │
    # └─────────────────────────────────────────────────┘
    DIAGNOSTIC_ONLY = True  # ← Change to False after configuring variables

    print("\n" + "#"*70)
    print("#  URANS Wall Pressure Evolution Extractor")
    print("#"*70)

    if DIAGNOSTIC_ONLY:
        run_diagnostic(DATA_DIR, FILE_PATTERN)
        return

    # ── Step 1: Find files ──
    print("\n[Step 1] Discovering files...")
    files = discover_files(DATA_DIR, FILE_PATTERN)

    # ── Step 2: Extract wall pressure from all time steps ──
    print("\n[Step 2] Extracting wall pressure from each time step...")
    x_wall, pressure_matrix, solution_times = extract_all_timesteps(files)

    print(f"\n  Result: {pressure_matrix.shape[0]} time steps × {pressure_matrix.shape[1]} wall points")
    print(f"  x range: [{x_wall[0]*1000:.2f}, {x_wall[-1]*1000:.2f}] mm")
    print(f"  P range: [{np.min(pressure_matrix):.1f}, {np.max(pressure_matrix):.1f}]")

    # ── Step 3: Save raw data for later use ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.savez(os.path.join(OUTPUT_DIR, "wall_pressure_data.npz"),
             x_wall=x_wall,
             pressure_matrix=pressure_matrix,
             solution_times=solution_times)
    print(f"\n  Saved raw data: wall_pressure_data.npz")
    print(f"  (You can reload this with: data = np.load('wall_pressure_data.npz'))")

    # ── Step 4: Generate plots ──
    print("\n[Step 3] Generating visualizations...")
    plot_carpet(x_wall, pressure_matrix, solution_times, OUTPUT_DIR)
    plot_selected_timesteps(x_wall, pressure_matrix, solution_times, OUTPUT_DIR)
    plot_pressure_history(x_wall, pressure_matrix, solution_times, OUTPUT_DIR)
    plot_convergence_check(pressure_matrix, solution_times, OUTPUT_DIR)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("\n  DONE!")


if __name__ == "__main__":
    main()