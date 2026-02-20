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
COLORMAP = "RdBu_r"     # Diverging colormap for pressure (red = high, blue = low)


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

    TEACHING POINT — PyTecplot Connected Mode Gotcha:
    ---------------------------------------------------
    In connected mode, PyTecplot communicates with Tecplot 360 over a socket.
    After tp.new_layout(), the old dataset/zone objects become STALE — they still
    point to memory that Tecplot has freed. Using dataset.zones() (a generator)
    triggers lazy evaluation that hits the stale reference → SystemError.
    
    The fix: use INDEX-BASED access (dataset.zone(i)) instead of generators,
    and always get a fresh dataset reference from tp.active_frame().dataset
    after loading data. This forces PyTecplot to query the live Tecplot state.

    Parameters:
    -----------
    filepath : str — Path to the .bin file
    is_first_file : bool — If True, print diagnostic info about the dataset

    Returns:
    --------
    x_wall : np.ndarray — Streamwise coordinates of wall nodes, sorted by x
    p_wall : np.ndarray — Pressure values at wall nodes, sorted by x
    solution_time : float — Solution time associated with this file (if available)
    """

    # Clear any existing data
    tp.new_layout()

    # Load the file — PyTecplot auto-detects the format
    tp.data.load_tecplot(filepath)
    
    # CRITICAL: Get a FRESH dataset reference from the active frame.
    # Do NOT use the return value of load_tecplot() after new_layout() in connected mode.
    dataset = tp.active_frame().dataset

    # ── Diagnostic output for the first file ──
    if is_first_file:
        print("\n" + "="*70)
        print("DATASET DIAGNOSTICS (from first file)")
        print("="*70)
        print(f"  File: {os.path.basename(filepath)}")
        print(f"  Number of variables: {dataset.num_variables}")
        print(f"  Variable names:")
        for i in range(dataset.num_variables):
            var = dataset.variable(i)
            print(f"    [{i}] {var.name}")
        print(f"  Number of zones: {dataset.num_zones}")
        for i in range(dataset.num_zones):
            zone = dataset.zone(i)
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
        # Use index-based access — NOT the .zones() generator
        for i in range(dataset.num_zones):
            zone = dataset.zone(i)
            if WALL_ZONE_NAME.lower() in zone.name.lower():
                target_zone = zone
                if is_first_file:
                    print(f"  Found wall zone: '{zone.name}'")
                break
        if target_zone is None:
            zone_names = [dataset.zone(i).name for i in range(dataset.num_zones)]
            print(f"  WARNING: No zone matching '{WALL_ZONE_NAME}' found.")
            print(f"  Available zones: {zone_names}")
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
            imax = zone.dimensions[0]
            jmax = zone.dimensions[1]

            if is_first_file:
                print(f"  Structured grid: imax={imax}, jmax={jmax}")
                print(f"  Extracting j=0 row (wall) — {imax} points")

            wall_indices = np.arange(imax)

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

            wall_mask = y_all <= (y_min + Y_TOLERANCE)

            if np.sum(wall_mask) < 10:
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
    zone = dataset.zone(0)
    try:
        solution_time = zone.solution_time
    except:
        solution_time = 0.0

    if solution_time == 0.0 and DT_PHYSICAL is not None:
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
    iteration_numbers : np.ndarray  (N_timesteps,) — iteration from filenames
    """

    all_x = []
    all_p = []
    solution_times = []
    iteration_numbers = []

    n_files = len(files)

    for i, fpath in enumerate(files):
        if (i % max(1, n_files // 20)) == 0 or i == n_files - 1:
            pct = 100 * (i + 1) / n_files
            print(f"  Processing file {i+1}/{n_files} ({pct:.0f}%): {os.path.basename(fpath)}")

        x_w, p_w, sol_t = get_wall_pressure_from_file(fpath, is_first_file=(i == 0))
        all_x.append(x_w)
        all_p.append(p_w)
        solution_times.append(sol_t)
        
        # Extract iteration number from filename (e.g., "mcfd_tec_3500.bin" → 3500)
        numbers = re.findall(r'\d+', os.path.basename(fpath))
        iter_num = int(numbers[-1]) if numbers else i
        iteration_numbers.append(iter_num)

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
    iteration_numbers = np.array(iteration_numbers, dtype=int)
    
    print(f"  Iteration range: {iteration_numbers[0]} → {iteration_numbers[-1]}")

    return x_common, pressure_matrix, solution_times, iteration_numbers


#=====================================================================================================#
#                              PART 3: VISUALIZATION (Publication Quality)
#=====================================================================================================#
#
# TEACHING POINT — Publication-Quality Figures in matplotlib:
# -----------------------------------------------------------
# Academic journals (AIAA, JFM, Physics of Fluids, etc.) have specific expectations:
#
#   1. FONT: Use a serif font (Times, Computer Modern) or clean sans-serif (Helvetica).
#      matplotlib's default font looks amateurish in papers. We use 'STIXGeneral'
#      which matches LaTeX Computer Modern and is built into matplotlib.
#
#   2. FONT SIZE: Axis labels and tick labels must be readable when the figure is
#      scaled to column width (~3.5 inches single-column, ~7 inches double-column).
#      Rule of thumb: 10-12 pt labels at final print size.
#
#   3. LINE WEIGHTS: Thin lines disappear in print. Use ≥1.0 pt for data lines,
#      ≥0.6 pt for axes/ticks. Avoid hairline gridlines.
#
#   4. COLORS: Must be distinguishable in grayscale (many readers print B&W).
#      Use colorblind-friendly palettes. Avoid red-green only distinctions.
#
#   5. TICKS: Inward-facing ticks on all four sides is the standard in physics/
#      engineering journals. Major + minor ticks show scale clearly.
#
#   6. NO TITLES: Journal figures use captions, not titles. The title wastes space.
#      We include them here as optional since this is for your own analysis too.
#
#   7. SAVE AS PDF + PNG: PDF is vector (scalable, small file), PNG is raster
#      (for presentations). Always save both.

def set_pub_style():
    """
    Configure matplotlib for publication-quality output.
    
    TEACHING POINT — Font sizing for different contexts:
    -----------------------------------------------------
    The font sizes here are tuned to be readable BOTH in journal figures (printed
    at ~3.5-7 inch width) AND in PowerPoint slides (projected at ~10 inches).
    The key is: labels should be 14-16 pt at figure size, ticks 12-13 pt.
    If you find them too large for a journal, scale down by ~2 pt.
    """
    plt.rcParams.update({
        # ── Font ──
        'font.family':        'serif',
        'font.serif':         ['STIXGeneral'],
        'mathtext.fontset':   'stix',
        'font.size':          14,

        # ── Axes ──
        'axes.linewidth':     1.0,
        'axes.labelsize':     16,
        'axes.titlesize':     17,
        'axes.labelpad':      8,

        # ── Ticks (inward, all four sides) ──
        'xtick.direction':    'in',
        'ytick.direction':    'in',
        'xtick.major.size':   6,
        'ytick.major.size':   6,
        'xtick.minor.size':   3,
        'ytick.minor.size':   3,
        'xtick.major.width':  1.0,
        'ytick.major.width':  1.0,
        'xtick.minor.width':  0.6,
        'ytick.minor.width':  0.6,
        'xtick.labelsize':    13,
        'ytick.labelsize':    13,
        'xtick.top':          True,
        'ytick.right':        True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,

        # ── Legend ──
        'legend.fontsize':    12,
        'legend.framealpha':  0.9,
        'legend.edgecolor':   '0.6',
        'legend.fancybox':    False,

        # ── Lines ──
        'lines.linewidth':    1.8,

        # ── Figure ──
        'figure.dpi':         150,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.05,
    })


def save_figure(fig, output_dir, name):
    """Save figure as both PNG (300 dpi) and PDF (vector)."""
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches='tight')
    print(f"  Saved: {name}.png  +  {name}.pdf")


def plot_carpet(x_wall, pressure_matrix, solution_times, output_dir, iteration_numbers=None):
    """
    Space-time carpet plot: x on horizontal axis, iteration on vertical axis,
    color = wall pressure.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    n_t, n_x = pressure_matrix.shape

    # Use iteration numbers for y-axis (extracted from filenames)
    if iteration_numbers is not None:
        y_axis = iteration_numbers
        y_label = r"Iteration"
    elif not np.all(solution_times == 0):
        y_axis = solution_times
        y_label = r"Solution Time $[\mathrm{s}]$"
    else:
        y_axis = np.arange(n_t)
        y_label = "Time Step Index"

    # pcolormesh expects cell edges, not centers
    dx = np.diff(x_wall)
    x_edges = np.concatenate([[x_wall[0] - dx[0]/2],
                               x_wall[:-1] + dx/2,
                               [x_wall[-1] + dx[-1]/2]])

    y_axis_float = y_axis.astype(float)
    if len(y_axis_float) > 1:
        dy = np.diff(y_axis_float)
        y_edges = np.concatenate([[y_axis_float[0] - dy[0]/2],
                                   y_axis_float[:-1] + dy/2,
                                   [y_axis_float[-1] + dy[-1]/2]])
    else:
        y_edges = np.array([y_axis_float[0] - 0.5, y_axis_float[0] + 0.5])

    pcm = ax.pcolormesh(x_edges, y_edges, pressure_matrix,
                        cmap=COLORMAP, shading='flat', rasterized=True)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.03, aspect=25)
    p_label = r"$P_w\,/\,P_\infty$" if NORMALIZE_PRESSURE else r"$P_w$ $[\mathrm{Pa}]$"
    cbar.set_label(p_label, fontsize=16)
    cbar.ax.tick_params(labelsize=13, direction='in')

    ax.set_xlabel(r"$x$ $[\mathrm{m}]$", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    plt.tight_layout()
    save_figure(fig, output_dir, "carpet_plot")
    plt.close(fig)


def plot_selected_timesteps(x_wall, pressure_matrix, solution_times, output_dir,
                             iteration_numbers=None, n_lines=8):
    """
    Overlay wall pressure profiles at selected time steps.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    n_t = pressure_matrix.shape[0]
    indices = np.linspace(0, n_t - 1, n_lines, dtype=int)

    cmap = plt.cm.coolwarm
    colors = [cmap(v) for v in np.linspace(0.05, 0.95, n_lines)]

    for k, idx in enumerate(indices):
        if iteration_numbers is not None:
            label = f"Iter {iteration_numbers[idx]}"
        elif not np.all(solution_times == 0):
            label = f"$t = {solution_times[idx]:.4g}$ s"
        else:
            label = f"Step {idx}"

        ax.plot(x_wall, pressure_matrix[idx, :],
                color=colors[k], label=label)

    p_label = r"$P_w\,/\,P_\infty$" if NORMALIZE_PRESSURE else r"$P_w$ $[\mathrm{Pa}]$"
    ax.set_xlabel(r"$x$ $[\mathrm{m}]$", fontsize=16)
    ax.set_ylabel(p_label, fontsize=16)
    ax.legend(fontsize=10, ncol=2, loc='best', handlelength=1.5)

    plt.tight_layout()
    save_figure(fig, output_dir, "pressure_profiles")
    plt.close(fig)


def plot_pressure_history(x_wall, pressure_matrix, solution_times, output_dir,
                           iteration_numbers=None, n_probes=5):
    """
    Wall pressure vs time at selected x-locations along the wall.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    n_t = pressure_matrix.shape[0]
    x_min, x_max = x_wall[0], x_wall[-1]
    x_probes = np.linspace(x_min + 0.05 * (x_max - x_min),
                            x_max - 0.05 * (x_max - x_min), n_probes)

    # Use iteration numbers for x-axis
    if iteration_numbers is not None:
        t_axis = iteration_numbers
        t_label = r"Iteration"
    elif not np.all(solution_times == 0):
        t_axis = solution_times
        t_label = r"Solution Time $[\mathrm{s}]$"
    else:
        t_axis = np.arange(n_t)
        t_label = "Time Step Index"

    prop_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

    for k, xp in enumerate(x_probes):
        idx = np.argmin(np.abs(x_wall - xp))
        p_history = pressure_matrix[:, idx]

        ax.plot(t_axis, p_history, color=prop_colors[k % len(prop_colors)],
                label=f"$x = {x_wall[idx]:.4f}$ m")

    p_label = r"$P_w\,/\,P_\infty$" if NORMALIZE_PRESSURE else r"$P_w$ $[\mathrm{Pa}]$"
    ax.set_xlabel(t_label, fontsize=16)
    ax.set_ylabel(p_label, fontsize=16)
    ax.legend(fontsize=11, loc='best', handlelength=1.5)

    plt.tight_layout()
    save_figure(fig, output_dir, "pressure_history")
    plt.close(fig)


def plot_convergence_check(pressure_matrix, solution_times, output_dir,
                            iteration_numbers=None):
    """
    Plot the RMS pressure change between consecutive time steps.
    """

    n_t = pressure_matrix.shape[0]

    if n_t < 2:
        print("  Skipping convergence plot (need at least 2 time steps)")
        return

    dp_norm = np.zeros(n_t - 1)
    for i in range(n_t - 1):
        dp = pressure_matrix[i+1, :] - pressure_matrix[i, :]
        dp_norm[i] = np.sqrt(np.mean(dp**2))

    fig, ax = plt.subplots(figsize=(8, 4))

    if iteration_numbers is not None:
        t_axis = iteration_numbers[1:]
        t_label = r"Iteration"
    elif not np.all(solution_times == 0):
        t_axis = solution_times[1:]
        t_label = r"Solution Time $[\mathrm{s}]$"
    else:
        t_axis = np.arange(1, n_t)
        t_label = "Time Step Index"

    ax.semilogy(t_axis, dp_norm, color='#333333')
    ax.set_xlabel(t_label, fontsize=16)
    ax.set_ylabel(r"RMS $\Delta P_w$ $[\mathrm{Pa}]$", fontsize=16)

    plt.tight_layout()
    save_figure(fig, output_dir, "convergence_check")
    plt.close(fig)


#=====================================================================================================#
#                    PART 3b: SHOCK DETECTION & PRESSURE INTEGRAL ANALYSIS
#=====================================================================================================#
#
# TEACHING POINT — Robust Shock Detection with a Spatial Search Window:
# ----------------------------------------------------------------------
# The naive approach (find the largest |dP/dx| globally) fails because:
#   1. Early time steps have no clear shock — the flow is still developing
#   2. Multiple shocks exist on a multi-bump wavy wall
#   3. The inlet region can have large gradients that aren't shocks
#
# The robust approach: CONSTRAIN the search to a physical window where
# you KNOW the shock must be. For the shock after the first bump:
#   - It must be DOWNSTREAM of the first crest (x > x_crest1)
#   - It must be UPSTREAM of the second crest (x < x_crest2)
#   - It's a COMPRESSION shock, so dP/dx > 0 (pressure rises)
#
# Within that window, the maximum positive dP/dx is the shock.
# This is simple, physical, and doesn't rely on peak-finding tuning.

# ── Geometry parameters for your wavy wall ──
# These define where to search for the shock.
X_CREST1 = 0.015   # [m] — x-location of the first bump crest
X_CREST2 = 0.08    # [m] — x-location of the second bump crest

# Search window: from just past the first crest to just before the second
# The shock forms on the downstream face of the first bump.
SHOCK_SEARCH_X_MIN = X_CREST1       # [m] — start searching here
SHOCK_SEARCH_X_MAX = X_CREST2       # [m] — stop searching here


def detect_shock_after_first_bump(x_wall, p_wall):
    """
    Detect the shock that forms downstream of the first bump crest.
    
    Strategy:
      1. Restrict to the search window [SHOCK_SEARCH_X_MIN, SHOCK_SEARCH_X_MAX]
      2. Compute dP/dx within that window
      3. The shock is where dP/dx is MAXIMUM (largest positive gradient = compression)
    
    TEACHING POINT — Why max(dP/dx) instead of max(|dP/dx|)?
    ----------------------------------------------------------
    We use the POSITIVE gradient specifically because a shock is a compression:
    pressure jumps UP across it (dP/dx > 0). Expansion fans have dP/dx < 0.
    By looking for max(dP/dx) instead of max(|dP/dx|), we ensure we find
    the compression shock and not an expansion fan, which is more physically
    meaningful and more robust.
    
    Returns:
    --------
    shock_idx : int — index into the FULL x_wall/p_wall arrays
    shock_x : float — x-coordinate of the shock [m]
    dpdx_full : np.ndarray — pressure gradient over the full domain (for diagnostics)
    """
    # Compute gradient over the full domain
    dx = np.diff(x_wall)
    dp = np.diff(p_wall)
    dpdx_full = dp / dx
    x_mid = 0.5 * (x_wall[:-1] + x_wall[1:])
    
    # Restrict to search window
    window_mask = (x_mid >= SHOCK_SEARCH_X_MIN) & (x_mid <= SHOCK_SEARCH_X_MAX)
    
    if not np.any(window_mask):
        # Fallback: search the whole domain
        print(f"  WARNING: No points in shock search window [{SHOCK_SEARCH_X_MIN}, {SHOCK_SEARCH_X_MAX}]")
        print(f"           Falling back to global max |dP/dx|")
        shock_mid_idx = np.argmax(np.abs(dpdx_full))
    else:
        # Find the maximum POSITIVE gradient in the window (compression shock)
        dpdx_window = dpdx_full[window_mask]
        x_mid_window = x_mid[window_mask]
        
        # Get indices within the window
        window_indices = np.where(window_mask)[0]
        
        # Find max positive gradient (compression shock)
        if np.any(dpdx_window > 0):
            local_idx = np.argmax(dpdx_window)
        else:
            # No positive gradient — flow hasn't developed yet, use max |dP/dx|
            local_idx = np.argmax(np.abs(dpdx_window))
        
        shock_mid_idx = window_indices[local_idx]
    
    # Map midpoint index to nearest node index
    shock_x = x_mid[shock_mid_idx]
    shock_idx = np.argmin(np.abs(x_wall - shock_x))
    
    return shock_idx, x_wall[shock_idx], dpdx_full


def compute_pressure_integrals(x_wall, p_wall, shock_idx, P_ref=None):
    """
    Compute pressure integrals on the left and right of the shock.
    
    Returns both absolute ∫P dx and gauge ∫(P - P_ref) dx.
    
    TEACHING POINT — Numerical Integration:
    -----------------------------------------
    We use the trapezoidal rule (np.trapz), which approximates:
        ∫P dx ≈ Σ 0.5 * (P[i] + P[i+1]) * (x[i+1] - x[i])
    Second-order accurate and perfectly adequate for this mesh resolution.
    """
    if P_ref is None:
        P_ref = p_wall[0]
    
    # Split at shock
    x_left = x_wall[:shock_idx + 1]
    p_left = p_wall[:shock_idx + 1]
    x_right = x_wall[shock_idx:]
    p_right = p_wall[shock_idx:]
    
    # Absolute integrals: ∫P dx
    abs_left = np.trapzoid(p_left, x_left)
    abs_right = np.trapzoid(p_right, x_right)
    
    # Gauge integrals: ∫(P - P_ref) dx
    gauge_left = np.trapzoid(p_left - P_ref, x_left)
    gauge_right = np.trapzoid(p_right - P_ref, x_right)
    
    return {
        'abs_left': abs_left,
        'abs_right': abs_right,
        'gauge_left': gauge_left,
        'gauge_right': gauge_right,
        'P_ref': P_ref,
        'x_left_range': (x_left[0], x_left[-1]),
        'x_right_range': (x_right[0], x_right[-1]),
    }


def analyze_shock_pressure_integrals(x_wall, pressure_matrix, solution_times, output_dir,
                                      iteration_numbers=None):
    """
    For each time step:
      1. Detect the shock after the first bump (windowed search)
      2. Compute pressure integrals on both sides
      3. Plot the evolution + shock location + carpet overlay
    """
    
    n_t = pressure_matrix.shape[0]
    
    shock_x_history = np.zeros(n_t)
    abs_left_history = np.zeros(n_t)
    abs_right_history = np.zeros(n_t)
    gauge_left_history = np.zeros(n_t)
    gauge_right_history = np.zeros(n_t)
    pref_history = np.zeros(n_t)
    
    print(f"  Analyzing shock after first bump (search window: "
          f"x ∈ [{SHOCK_SEARCH_X_MIN}, {SHOCK_SEARCH_X_MAX}] m)...")
    
    for i in range(n_t):
        p_wall = pressure_matrix[i, :]
        
        shock_idx, shock_x, _ = detect_shock_after_first_bump(x_wall, p_wall)
        results = compute_pressure_integrals(x_wall, p_wall, shock_idx)
        
        shock_x_history[i] = shock_x
        abs_left_history[i] = results['abs_left']
        abs_right_history[i] = results['abs_right']
        gauge_left_history[i] = results['gauge_left']
        gauge_right_history[i] = results['gauge_right']
        pref_history[i] = results['P_ref']
    
    # ── Time axis ──
    if iteration_numbers is not None:
        t_axis = iteration_numbers
        t_label = r"Iteration"
    elif not np.all(solution_times == 0):
        t_axis = solution_times
        t_label = r"Solution Time $[\mathrm{s}]$"
    else:
        t_axis = np.arange(n_t)
        t_label = "Time Step Index"
    
    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 1: Representative pressure profile with shock and regions
    # ══════════════════════════════════════════════════════════════════════
    last_idx = n_t - 1
    p_last = pressure_matrix[last_idx, :]
    shock_idx_last, shock_x_last, _ = detect_shock_after_first_bump(x_wall, p_last)
    results_last = compute_pressure_integrals(x_wall, p_last, shock_idx_last)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Shade integration regions
    ax.fill_between(x_wall[:shock_idx_last + 1], p_last[:shock_idx_last + 1],
                    results_last['P_ref'],
                    color='#2166ac', alpha=0.15, label='Left of shock')
    ax.fill_between(x_wall[shock_idx_last:], p_last[shock_idx_last:],
                    results_last['P_ref'],
                    color='#b2182b', alpha=0.15, label='Right of shock')
    
    # Pressure profile
    ax.plot(x_wall, p_last, color='#333333')
    
    # Shock location
    ax.axvline(x=shock_x_last, color='#e41a1c', linewidth=1.2, linestyle='--',
               label=f'Shock at $x = {shock_x_last:.4f}$ m')
    
    # Search window shading
    ax.axvspan(SHOCK_SEARCH_X_MIN, SHOCK_SEARCH_X_MAX, alpha=0.06, color='orange',
               label='Search window')
    
    # Reference pressure line
    ax.axhline(y=results_last['P_ref'], color='#666666', linewidth=0.8, linestyle=':',
               label=f'$P_{{\\mathrm{{ref}}}} = {results_last["P_ref"]:.0f}$ Pa')
    
    p_label = r"$P_w\,/\,P_\infty$" if NORMALIZE_PRESSURE else r"$P_w$ $[\mathrm{Pa}]$"
    ax.set_xlabel(r"$x$ $[\mathrm{m}]$", fontsize=16)
    ax.set_ylabel(p_label, fontsize=16)
    ax.legend(fontsize=10, loc='best', handlelength=1.8)
    
    plt.tight_layout()
    save_figure(fig, output_dir, "shock_pressure_regions")
    plt.close(fig)
    
    # Print summary
    print(f"\n  ── Pressure Integral Summary (last time step, iter {iteration_numbers[-1] if iteration_numbers is not None else last_idx}) ──")
    print(f"     Shock location:      x = {shock_x_last:.5f} m")
    print(f"     Reference pressure:  P_ref = {results_last['P_ref']:.1f} Pa")
    print(f"     Left region:         x ∈ [{results_last['x_left_range'][0]:.4f}, {results_last['x_left_range'][1]:.4f}] m")
    print(f"     Right region:        x ∈ [{results_last['x_right_range'][0]:.4f}, {results_last['x_right_range'][1]:.4f}] m")
    print(f"")
    print(f"     ABSOLUTE integrals (∫Pw dx):")
    print(f"       Left:   {results_last['abs_left']:.4f} Pa·m")
    print(f"       Right:  {results_last['abs_right']:.4f} Pa·m")
    abs_ratio = results_last['abs_left'] / results_last['abs_right'] if results_last['abs_right'] != 0 else float('inf')
    print(f"       Ratio (L/R):  {abs_ratio:.4f}")
    if abs(abs_ratio - 1.0) < 0.01:
        print(f"       → Nearly equal")
    elif abs_ratio > 1.0:
        print(f"       → Left is larger by {(abs_ratio - 1)*100:.1f}%")
    else:
        print(f"       → Right is larger by {(1/abs_ratio - 1)*100:.1f}%")
    print(f"")
    print(f"     GAUGE integrals (∫(Pw - Pref) dx):")
    print(f"       Left:   {results_last['gauge_left']:.4f} Pa·m")
    print(f"       Right:  {results_last['gauge_right']:.4f} Pa·m")
    if results_last['gauge_right'] != 0:
        gauge_ratio = results_last['gauge_left'] / results_last['gauge_right']
        print(f"       Ratio (L/R):  {gauge_ratio:.4f}")
    print(f"  {'─' * 50}")
    
    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 2: Pressure integrals vs time
    # ══════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.08})
    
    ax1.plot(t_axis, abs_left_history, color='#2166ac',
             label=r'Left of shock: $\int P_w\,dx$')
    ax1.plot(t_axis, abs_right_history, color='#b2182b',
             label=r'Right of shock: $\int P_w\,dx$')
    ax1.set_ylabel(r"$\int P_w\,dx$ $[\mathrm{Pa \cdot m}]$", fontsize=16)
    ax1.legend(fontsize=11, loc='best', handlelength=1.5)
    
    ax2.plot(t_axis, gauge_left_history, color='#2166ac',
             label=r'Left of shock: $\int (P_w - P_{\mathrm{ref}})\,dx$')
    ax2.plot(t_axis, gauge_right_history, color='#b2182b',
             label=r'Right of shock: $\int (P_w - P_{\mathrm{ref}})\,dx$')
    ax2.axhline(y=0, color='#999999', linewidth=0.5, linestyle=':')
    ax2.set_xlabel(t_label, fontsize=16)
    ax2.set_ylabel(r"$\int (P_w - P_{\mathrm{ref}})\,dx$ $[\mathrm{Pa \cdot m}]$", fontsize=16)
    ax2.legend(fontsize=11, loc='best', handlelength=1.5)
    
    plt.tight_layout()
    save_figure(fig, output_dir, "pressure_integrals_vs_time")
    plt.close(fig)
    
    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 3: Shock location vs time
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(t_axis, shock_x_history, color='#e41a1c')
    ax.axhline(y=X_CREST1, color='#999999', linewidth=0.6, linestyle=':',
               label=f'1st crest ($x = {X_CREST1}$ m)')
    ax.axhline(y=X_CREST2, color='#999999', linewidth=0.6, linestyle='--',
               label=f'2nd crest ($x = {X_CREST2}$ m)')
    ax.set_xlabel(t_label, fontsize=16)
    ax.set_ylabel(r"Shock location $x_s$ $[\mathrm{m}]$", fontsize=16)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    save_figure(fig, output_dir, "shock_location_vs_time")
    plt.close(fig)
    
    # ══════════════════════════════════════════════════════════════════════
    # FIGURE 4: Carpet plot with shock location overlay
    # ══════════════════════════════════════════════════════════════════════
    #
    # TEACHING POINT — Overlaying data on a carpet plot:
    # ---------------------------------------------------
    # This is one of the most informative plots you can make for unsteady
    # shock analysis. The carpet plot shows the full P_w(x, t) field,
    # and the overlaid white line shows where the shock is at each time step.
    # If the shock is stationary, the line is vertical.
    # If it oscillates, the line wiggles — and you can see the frequency
    # and amplitude of the shock motion directly.
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    n_t, n_x = pressure_matrix.shape
    
    if iteration_numbers is not None:
        y_axis = iteration_numbers.astype(float)
        y_label = r"Iteration"
    elif not np.all(solution_times == 0):
        y_axis = solution_times
        y_label = r"Solution Time $[\mathrm{s}]$"
    else:
        y_axis = np.arange(n_t, dtype=float)
        y_label = "Time Step Index"
    
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
    
    pcm = ax.pcolormesh(x_edges, y_edges, pressure_matrix,
                        cmap=COLORMAP, shading='flat', rasterized=True)
    
    # Overlay shock trajectory
    ax.plot(shock_x_history, y_axis, color='white', linewidth=2.0,
            label='Shock location')
    ax.plot(shock_x_history, y_axis, color='#e41a1c', linewidth=0.8,
            linestyle='--')
    
    cbar = fig.colorbar(pcm, ax=ax, pad=0.03, aspect=25)
    p_label = r"$P_w\,/\,P_\infty$" if NORMALIZE_PRESSURE else r"$P_w$ $[\mathrm{Pa}]$"
    cbar.set_label(p_label, fontsize=16)
    cbar.ax.tick_params(labelsize=13, direction='in')
    
    ax.set_xlabel(r"$x$ $[\mathrm{m}]$", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, output_dir, "carpet_plot_with_shock")
    plt.close(fig)
    
    # ══════════════════════════════════════════════════════════════════════
    # Save data
    # ══════════════════════════════════════════════════════════════════════
    np.savez(os.path.join(output_dir, "shock_integral_data.npz"),
             t_axis=t_axis,
             shock_x_history=shock_x_history,
             abs_left=abs_left_history,
             abs_right=abs_right_history,
             gauge_left=gauge_left_history,
             gauge_right=gauge_right_history,
             pref=pref_history)
    print(f"  Saved: shock_integral_data.npz")


#=====================================================================================================#
#                              PART 4: ANIMATED WALL PRESSURE
#=====================================================================================================#

def create_pressure_animation(x_wall, pressure_matrix, solution_times, output_dir,
                               iteration_numbers=None, skip=1):
    """
    Create TWO MP4 animations of wall pressure P_w(x) evolving through the URANS:
      1. wall_pressure_animation_60fps.mp4  — 60 fps, no frame duplication
      2. wall_pressure_animation_120fps.mp4 — 120 fps, smooth extended playback
    
    TEACHING POINT — Frame rate and perceived smoothness:
    ------------------------------------------------------
    At 60 fps with 350 frames, the video is ~5.8 seconds — very fast.
    At 120 fps it's even shorter (~2.9 s). To make a longer, smoother video
    at 120 fps, we write each frame TWICE (effectively 60 fps of unique data
    but encoded at 120 fps), giving the same ~5.8 s duration but with the
    smoothness benefits of a higher frame-rate container. This matches the
    approach in your Tecplot animation code.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    
    n_t, n_x = pressure_matrix.shape
    
    # Subsample time steps if requested, but ALWAYS include the last frame
    frame_indices = np.arange(0, n_t, skip)
    if frame_indices[-1] != n_t - 1:
        frame_indices = np.append(frame_indices, n_t - 1)
    n_frames = len(frame_indices)
    
    # Fixed y-axis limits
    p_min = np.min(pressure_matrix)
    p_max = np.max(pressure_matrix)
    p_pad = 0.05 * (p_max - p_min)
    
    # Build iteration labels for each frame
    if iteration_numbers is not None:
        frame_labels = [f"Iteration {iteration_numbers[frame_indices[k]]}"
                        for k in range(n_frames)]
    elif not np.all(solution_times == 0):
        frame_labels = [f"$t = {solution_times[frame_indices[k]]:.4g}$ s"
                        for k in range(n_frames)]
    else:
        frame_labels = [f"Step {frame_indices[k]} / {n_t - 1}"
                        for k in range(n_frames)]
    
    def _make_animation(fps, filename, repeat_frames=1):
        """Internal helper to render and save one animation variant."""
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        line, = ax.plot(x_wall, pressure_matrix[0, :], color='#2166ac')
        
        p_label = r"$P_w\,/\,P_\infty$" if NORMALIZE_PRESSURE else r"$P_w$ $[\mathrm{Pa}]$"
        ax.set_xlabel(r"$x$ $[\mathrm{m}]$", fontsize=16)
        ax.set_ylabel(p_label, fontsize=16)
        ax.set_xlim(x_wall[0], x_wall[-1])
        ax.set_ylim(p_min - p_pad, p_max + p_pad)
        
        # Iteration label box
        time_text = ax.text(0.03, 0.94, "", transform=ax.transAxes,
                            fontsize=14, verticalalignment='top',
                            fontfamily='serif',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      edgecolor='0.6', alpha=0.9))
        
        plt.tight_layout()
        
        # For repeated frames, we build a flat list of frame indices
        # where each unique frame appears `repeat_frames` times
        if repeat_frames > 1:
            expanded_indices = []
            expanded_labels = []
            for k in range(n_frames):
                for _ in range(repeat_frames):
                    expanded_indices.append(k)
                    expanded_labels.append(frame_labels[k])
            total_render_frames = len(expanded_indices)
        else:
            expanded_indices = list(range(n_frames))
            expanded_labels = frame_labels
            total_render_frames = n_frames
        
        def update(render_frame):
            k = expanded_indices[render_frame]
            idx = frame_indices[k]
            line.set_ydata(pressure_matrix[idx, :])
            time_text.set_text(expanded_labels[render_frame])
            return line, time_text
        
        print(f"  Rendering {filename}: {total_render_frames} frames at {fps} fps "
              f"({total_render_frames/fps:.1f}s video)...")
        
        anim = FuncAnimation(fig, update, frames=total_render_frames,
                              interval=1000/fps, blit=True)
        
        mp4_path = os.path.join(output_dir, filename)
        gif_name = filename.replace('.mp4', '.gif')
        gif_path = os.path.join(output_dir, gif_name)
        
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(title='Wall Pressure URANS'),
                                  bitrate=3000)
            anim.save(mp4_path, writer=writer, dpi=200)
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  MP4 save failed ({e})")
            print(f"  Trying GIF instead (install ffmpeg for MP4 support)...")
            try:
                writer = PillowWriter(fps=min(fps, 50))  # GIF maxes out around 50 fps
                anim.save(gif_path, writer=writer, dpi=150)
                print(f"  Saved: {gif_name}")
            except Exception as e2:
                print(f"  GIF save also failed ({e2})")
                print(f"  Install ffmpeg: conda install -c conda-forge ffmpeg")
        
        plt.close(fig)
    
    # ── Generate both versions ──
    _make_animation(fps=60,  filename="wall_pressure_animation_60fps.mp4",  repeat_frames=1)
    _make_animation(fps=120, filename="wall_pressure_animation_120fps.mp4", repeat_frames=2)


#=====================================================================================================#
#                              PART 5: DIAGNOSTIC MODE
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
    tp.data.load_tecplot(filepath)
    dataset = tp.active_frame().dataset

    print(f"Number of variables: {dataset.num_variables}")
    print(f"Variable names:")
    for i in range(dataset.num_variables):
        var = dataset.variable(i)
        print(f"  [{i}] '{var.name}'")

    print(f"\nNumber of zones: {dataset.num_zones}")
    for i in range(dataset.num_zones):
        zone = dataset.zone(i)
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
        for j in range(min(dataset.num_variables, 9)):
            var = dataset.variable(j)
            vals = zone.values(var.name)[:]
            if len(vals) > 0:
                print(f"      {var.name}: [{np.min(vals):.6g}, {np.max(vals):.6g}]")
        if dataset.num_variables > 9:
            print(f"      ... ({dataset.num_variables - 9} more variables)")

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
    DIAGNOSTIC_ONLY = False  # ← Change to False after configuring variables

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
    x_wall, pressure_matrix, solution_times, iteration_numbers = extract_all_timesteps(files)

    print(f"\n  Result: {pressure_matrix.shape[0]} time steps × {pressure_matrix.shape[1]} wall points")
    print(f"  x range: [{x_wall[0]:.4f}, {x_wall[-1]:.4f}] m")
    print(f"  P range: [{np.min(pressure_matrix):.1f}, {np.max(pressure_matrix):.1f}]")

    # ── Step 3: Save raw data for later use ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.savez(os.path.join(OUTPUT_DIR, "wall_pressure_data.npz"),
             x_wall=x_wall,
             pressure_matrix=pressure_matrix,
             solution_times=solution_times,
             iteration_numbers=iteration_numbers)
    print(f"\n  Saved raw data: wall_pressure_data.npz")
    print(f"  (You can reload this with: data = np.load('wall_pressure_data.npz'))")

    # ── Step 4: Generate plots ──
    print("\n[Step 3] Generating visualizations...")
    set_pub_style()  # Apply publication-quality formatting to all figures
    plot_carpet(x_wall, pressure_matrix, solution_times, OUTPUT_DIR, iteration_numbers)
    plot_selected_timesteps(x_wall, pressure_matrix, solution_times, OUTPUT_DIR, iteration_numbers)
    plot_pressure_history(x_wall, pressure_matrix, solution_times, OUTPUT_DIR, iteration_numbers)
    plot_convergence_check(pressure_matrix, solution_times, OUTPUT_DIR, iteration_numbers)
    
    # ── Step 5: Shock detection & pressure integral analysis ──
    print("\n[Step 4] Analyzing shock location and pressure integrals...")
    analyze_shock_pressure_integrals(x_wall, pressure_matrix, solution_times, OUTPUT_DIR, iteration_numbers)
    
    # ── Step 6: Create animations ──
    print("\n[Step 5] Creating animations...")
    create_pressure_animation(x_wall, pressure_matrix, solution_times, OUTPUT_DIR,
                               iteration_numbers=iteration_numbers, skip=1)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("\n  DONE!")


if __name__ == "__main__":
    main()