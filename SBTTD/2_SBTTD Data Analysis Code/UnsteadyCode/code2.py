"""
Fast dP/dx GIF Generator — Duration-Matched to MP4
====================================================
Standalone script that loads wall_pressure_data.npz and generates
a GIF of dP/dx(x) evolution matching your 15.11s video duration exactly.

No Tecplot needed — runs purely from saved .npz data.

TEACHING POINT — Matching GIF duration to MP4:
-----------------------------------------------
GIF frame timing is in centiseconds (10ms increments). With 350 unique
frames, a single uniform delay can't hit 15.11s exactly:
  - 40ms/frame -> 14.0s (too short)
  - 50ms/frame -> 17.5s (too long)

Solution: use VARIABLE per-frame durations. 239 frames at 40ms + 111 at
50ms = 9560 + 5550 = 15110ms = 15.11s exactly. The slower frames go at
the end where the shock has stabilized — natural pacing.

Author: HS  |  Date: Feb 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no GUI overhead
import matplotlib.pyplot as plt
from PIL import Image
import os
import time


# =============================================================================
# CONFIGURATION — edit these to match your setup
# =============================================================================
DATA_FILE = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\2_SBTTD Data Analysis Code\UnsteadyCode\2_Results\wall_pressure_data.npz"
OUTPUT_DIR = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\35_Git\Supersonic-Bladeless-Turbine\SBTTD\2_SBTTD Data Analysis Code\UnsteadyCode\2_Results"

TARGET_DURATION_MS = 15110   # Match your MP4: 15.11 seconds
DPI = 150                    # Figure resolution


# =============================================================================
# PUBLICATION STYLE
# =============================================================================
def set_pub_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 14,
        'axes.linewidth': 1.0, 'axes.labelsize': 16, 'axes.labelpad': 8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'lines.linewidth': 1.8,
    })


# =============================================================================
# COMPUTE PER-FRAME DURATIONS TO HIT TARGET EXACTLY
# =============================================================================
def compute_frame_durations(n_frames, target_ms):
    """
    Build a list of per-frame delays (multiples of 10ms) that sum to target_ms.

    Solves: n_fast * 40 + n_slow * 50 = target_ms
    where   n_fast + n_slow = n_frames
    ->      n_fast = (50 * n_frames - target_ms) / 10

    TEACHING POINT — Variable GIF Frame Timing:
    The GIF spec allows a DIFFERENT delay for each frame. Pillow supports
    this by passing a list to the `duration` parameter instead of an int.
    Most tools only use a uniform delay, but variable timing lets us nail
    an exact total duration even with GIF's coarse 10ms granularity.
    """
    n_fast = int(round((50 * n_frames - target_ms) / 10))
    n_fast = max(0, min(n_frames, n_fast))
    n_slow = n_frames - n_fast

    actual_ms = n_fast * 40 + n_slow * 50
    durations = [40] * n_fast + [50] * n_slow

    print(f"  Frame timing: {n_fast} x 40ms + {n_slow} x 50ms = {actual_ms}ms ({actual_ms/1000:.2f}s)")
    print(f"  Target: {target_ms}ms ({target_ms/1000:.2f}s) | Error: {abs(actual_ms - target_ms)}ms")

    return durations


# =============================================================================
# RENDER GIF
# =============================================================================
def render_dpdx_gif(x_wall, dpdx_matrix, iteration_numbers, output_path):
    """
    Render all frames directly to PIL Images and assemble GIF.

    TEACHING POINT — Why direct rendering beats FuncAnimation:
    FuncAnimation goes through: update() -> canvas.draw() -> writer.grab_frame()
    with per-frame file I/O and animation framework overhead.

    Direct rendering: update line data -> canvas.draw() -> buffer_rgba() -> PIL
    All in-memory, no intermediate files, no framework overhead. 3-5x faster.
    """
    n_t = dpdx_matrix.shape[0]

    # Fixed axis limits from full dataset
    dpdx_min, dpdx_max = np.min(dpdx_matrix), np.max(dpdx_matrix)
    pad = 0.05 * (dpdx_max - dpdx_min)

    # Create figure once
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    line, = ax.plot(x_wall, dpdx_matrix[0, :], color='#d95f02')
    ax.axhline(y=0, color='#999999', linewidth=0.5, linestyle=':')
    ax.set_xlabel(r"$x$ $[\mathrm{m}]$")
    ax.set_ylabel(r"$dP/dx$ $[\mathrm{Pa/m}]$")
    ax.set_xlim(x_wall[0], x_wall[-1])
    ax.set_ylim(dpdx_min - pad, dpdx_max + pad)

    time_text = ax.text(
        0.03, 0.94, "", transform=ax.transAxes,
        fontsize=14, verticalalignment='top', fontfamily='serif',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='0.6', alpha=0.9))

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # Compute per-frame durations to match MP4
    durations = compute_frame_durations(n_t, TARGET_DURATION_MS)

    print(f"  Rendering {n_t} frames at {w}x{h} px...")
    t0 = time.time()

    pil_frames = []
    for i in range(n_t):
        # Update only what changes
        line.set_ydata(dpdx_matrix[i, :])
        label = f"Iteration {iteration_numbers[i]}" if iteration_numbers is not None else f"Step {i}"
        time_text.set_text(label)

        # Render to in-memory buffer
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1).copy()
        pil_frames.append(img.convert('P', palette=Image.Palette.ADAPTIVE, colors=256))

        if (i + 1) % 50 == 0 or i == n_t - 1:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_t} frames ({(i+1)/elapsed:.1f} render fps)")

    plt.close(fig)

    # Assemble GIF with variable per-frame durations
    print(f"  Assembling GIF...")
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=False
    )

    elapsed_total = time.time() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done: {elapsed_total:.1f}s total, {size_mb:.1f} MB")


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_pub_style()

    print("Loading data...")
    data = np.load(DATA_FILE)
    x_wall = data['x_wall']
    dpdx_matrix = data['dpdx_matrix']
    iter_nums = data.get('iteration_numbers', None)

    print(f"  {dpdx_matrix.shape[0]} timesteps x {dpdx_matrix.shape[1]} wall points")
    print(f"  dP/dx range: [{np.min(dpdx_matrix):.0f}, {np.max(dpdx_matrix):.0f}] Pa/m")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "wall_dpdx_animation_120fps.gif")

    render_dpdx_gif(x_wall, dpdx_matrix, iter_nums, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()