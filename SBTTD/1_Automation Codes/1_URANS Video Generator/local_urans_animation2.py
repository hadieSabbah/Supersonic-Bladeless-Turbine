#!/usr/bin/env python3
"""
Local URANS Animation Generator
================================
This script is designed for running on your local machine with Tecplot 360 installed.

Two modes of operation:l
1. Connected Mode: PyTecplot connects to running Tecplot 360 (see live updates)
2. Batch Mode: Runs without GUI (faster, no display needed)

Requirements:
- Tecplot 360 installed and licensed
- PyTecplot: pip install pytecplot

Author: For HS's bladeless turbine URANS post-processing
"""

import os
import sys
import glob
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

# =============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR SETUP
# =============================================================================
@dataclass
class Config:
    """Configuration settings - modify these for your specific case."""
    
    # Input/Output paths
    input_dir: str = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\tecfiles"
    output_dir: str = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\frames"
    layout_file: Optional[str] = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\10_PowerPoints\6_SciTech Paper\1_Scitech PPT\2_Tecplot Layout\machLayout.lay"
    
    # File pattern
    file_pattern: str = "mcfd_tec_*.bin"
    
    # Image settings
    width: int = 4096
    # Note: height is determined by the frame aspect ratio in Tecplot
    supersample: int = 3  # Anti-aliasing: 1-4 (higher = smoother, slower)
    
    # Processing options
    start_frame: int = 0      # Index of first file to process
    end_frame: Optional[int] = None  # None = process all
    skip: int = 1             # Process every Nth frame (use 10 for quick preview)
    
    # Annotation
    add_timestamp: bool = False
    timestamp_position: tuple = (5, 95)  # (x%, y%) from bottom-left
    timestamp_size: int = 14
    
    # Video settings (for automatic video creation)
    create_video: bool = True
    framerate: int = 60
    video_filename: str = "urans_animation.mp4"


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def natural_sort_key(filepath: str) -> int:
    """
    Extract timestep number for natural sorting.
    
    Why this matters:
    - String sort: 100, 1000, 10000, 10050, 10100, ...
    - Natural sort: 100, 1000, 10000, 10050, 10100, ... (correct temporal order)
    
    The function finds all numbers in the filename and returns the last one,
    which is typically the timestep identifier.
    """
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0


def get_sorted_files(input_dir: str, pattern: str) -> List[str]:
    """Get all matching files sorted by timestep."""
    search_path = os.path.join(input_dir, pattern)
    files = glob.glob(search_path)
    
    if not files:
        raise FileNotFoundError(f"No files found matching: {search_path}")
    
    return sorted(files, key=natural_sort_key)


def setup_tecplot_connection(use_batch_mode: bool = False):
    """
    Initialize PyTecplot connection.
    
    Parameters
    ----------
    use_batch_mode : bool
        If True, run without GUI (faster, no visual feedback)
        If False, connect to running Tecplot 360 instance (see live updates)
    
    Returns
    -------
    tp : module
        The tecplot module, configured for the selected mode
    """
    if use_batch_mode:
        # Batch mode - no GUI, faster processing
        # Must be set BEFORE importing tecplot
        os.environ['PYTECPLOT_BATCH_MODE'] = '1'
    
    import tecplot as tp
    
    if not use_batch_mode:
        # Connected mode - connect to running Tecplot 360
        # You must have Tecplot 360 open first!
        try:
            tp.session.connect()
            print("✓ Connected to Tecplot 360")
        except Exception as e:
            print(f"Could not connect to Tecplot 360: {e}")
            print("\n" + "="*60)
            print("TROUBLESHOOTING:")
            print("  1. Make sure Tecplot 360 is OPEN and running")
            print("  2. In Tecplot: Scripting → PyTecplot Connections...")
            print("     → Check 'Accept connections'")
            print("="*60)
            sys.exit(1)
    else:
        print("✓ Running in batch mode (no GUI)")
    
    return tp


def export_frame(tp, frame_num: int, output_dir: str, config: Config) -> str:
    """
    Export current visualization as PNG.
    
    Note: PyTecplot's save_png() only takes 'width' parameter.
    The height is determined automatically by the frame's aspect ratio.
    To control aspect ratio, adjust the frame size in your layout file.
    """
    output_path = os.path.join(output_dir, f"frame_{frame_num:05d}.png")
    
    # PyTecplot save_png signature (works with most versions):
    #   save_png(filename, width=, supersample=)
    # Height is determined by frame aspect ratio
    tp.export.save_png(
        output_path,
        width=config.width,
        supersample=config.supersample
    )
    
    return output_path


def add_timestamp_annotation(tp, timestep: int, config: Config):
    """Add timestep text to the current frame."""
    frame = tp.active_frame()
    
    # Remove previous timestamp annotations (if any)
    # This prevents text accumulation when processing multiple files
    for text in list(frame.texts()):
        if hasattr(text, 'text_string') and 'Timestep:' in text.text_string:
            frame.delete_text(text)
    
    # Add new timestamp
    frame.add_text(
        f"Timestep: {timestep}",
        position=config.timestamp_position,
        size=config.timestamp_size,
        color=tp.constant.Color.Black
    )


def process_files(tp, files: List[str], config: Config):
    """
    Main processing loop - load each file, apply layout, export frame.
    """
    from tecplot.constant import ReadDataOption
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    total_files = len(files)
    start_time = time.time()
    
    print(f"\nProcessing {total_files} files...")
    print(f"Output: {os.path.abspath(config.output_dir)}")
    print(f"Resolution: {config.width} pixels wide")
    print("-" * 50)
    
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        timestep = natural_sort_key(filepath)
        
        # Progress indicator
        progress = (i + 1) / total_files * 100
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total_files - i - 1) if i > 0 else 0
        
        print(f"[{i+1:4d}/{total_files}] {filename:25s} | "
              f"{progress:5.1f}% | ETA: {eta/60:.1f}min", end="")
        
        try:
            # Clear layout
            tp.new_layout()
            
            # Load layout FIRST (brings visualization settings)
            if config.layout_file and os.path.exists(config.layout_file):
                tp.load_layout(config.layout_file)
            
            # Load data with REPLACE option - keeps styling, replaces data
            tp.data.load_tecplot(filepath, 
                                 read_data_option=ReadDataOption.Replace, 
                                 reset_style=False)
            
            # Add timestamp annotation if requested
            if config.add_timestamp:
                add_timestamp_annotation(tp, timestep, config)
            
            # Export frame
            output_path = export_frame(tp, i, config.output_dir, config)
            print(f" ✓")
            
        except Exception as e:
            print(f" ✗ Error: {e}")
            continue
    
    total_time = time.time() - start_time
    print("-" * 50)
    print(f"Completed in {total_time/60:.1f} minutes")
    print(f"Frames saved to: {os.path.abspath(config.output_dir)}")

def create_video_ffmpeg(config: Config):
    """
    Create video from frames using ffmpeg.
    
    This requires ffmpeg to be installed and in your PATH.
    - Windows: Download from ffmpeg.org, add to PATH
    - Mac: brew install ffmpeg
    - Linux: sudo apt install ffmpeg
    """
    import subprocess
    
    frames_pattern = os.path.join(config.output_dir, "frame_%05d.png")
    output_path = os.path.join(os.path.dirname(config.output_dir), config.video_filename)
    
    cmd = [
        "ffmpeg", "-y",  # -y to overwrite without asking
        "-framerate", str(config.framerate),
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",  # Quality: 18 is visually lossless
        output_path
    ]
    
    print(f"\nCreating video: {output_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Video created: {output_path}")
    except FileNotFoundError:
        print("✗ ffmpeg not found. Install it or create video manually:")
        print(f"  ffmpeg -framerate {config.framerate} -i \"{frames_pattern}\" "
              f"-c:v libx264 -pix_fmt yuv420p -crf 18 \"{output_path}\"")
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg error: {e}")


# =============================================================================
# ALTERNATIVE: Using imageio (no ffmpeg required)
# =============================================================================

def create_video_imageio(config: Config):
    """
    Create video using imageio (pure Python, no ffmpeg needed).
    
    Install with: pip install imageio imageio-ffmpeg
    
    This is easier to set up but may be slower for large frame counts.
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return
    
    frames_dir = config.output_dir
    output_path = os.path.join(os.path.dirname(frames_dir), config.video_filename)
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    
    if not frame_files:
        print("No frames found!")
        return
    
    print(f"\nCreating video with imageio: {output_path}")
    print(f"Processing {len(frame_files)} frames...")
    
    # Create video writer
    writer = imageio.get_writer(output_path, fps=config.framerate, codec='libx264')
    
    for i, frame_path in enumerate(frame_files):
        if i % 50 == 0:
            print(f"  Adding frame {i+1}/{len(frame_files)}...")
        writer.append_data(imageio.imread(frame_path))
    
    writer.close()
    print(f"✓ Video created: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main entry point.
    
    To use this script:
    1. Edit the Config class above with your paths
    2. Run: python local_urans_animation.py
    
    Or import and use programmatically:
        from local_urans_animation import Config, run_animation
        config = Config(input_dir="/path/to/files", ...)
        run_animation(config, use_batch_mode=True)
    """
    # Create configuration - uses the values from the Config class above
    config = Config()
    
    # =========================================================================
    # CHOOSE YOUR MODE:
    # 
    # use_batch_mode=False (RECOMMENDED FOR WINDOWS)
    #   - Requires Tecplot 360 to be OPEN first
    #   - You can watch the frames being created live
    #   - More reliable on Windows
    #
    # use_batch_mode=True
    #   - Runs without GUI (faster)
    #   - Requires TEC360HOME environment variable to be set
    # =========================================================================
    
    run_animation(config, use_batch_mode=False)  # <-- Using connected mode


def run_animation(config: Config, use_batch_mode: bool = True):
    """
    Execute the full animation workflow.
    
    Parameters
    ----------
    config : Config
        Configuration settings
    use_batch_mode : bool
        True = faster, no GUI (requires TEC360HOME on Windows)
        False = see live updates in Tecplot 360 (must be running)
    """
    print("=" * 60)
    print("URANS Animation Generator - Local Version")
    print("=" * 60)
    
    # Get sorted file list
    try:
        files = get_sorted_files(config.input_dir, config.file_pattern)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"Found {len(files)} files")
    print(f"  First: {os.path.basename(files[0])} (timestep {natural_sort_key(files[0])})")
    print(f"  Last:  {os.path.basename(files[-1])} (timestep {natural_sort_key(files[-1])})")
    
    # Apply frame selection
    end = config.end_frame if config.end_frame else len(files)
    files = files[config.start_frame:end:config.skip]
    print(f"Processing {len(files)} files (skip={config.skip})")
    
    # Setup Tecplot
    tp = setup_tecplot_connection(use_batch_mode)
    
    # Process all files
    process_files(tp, files, config)
    
    # Create video if requested
    if config.create_video:
        create_video_ffmpeg(config)
        # Alternative: create_video_imageio(config)
    
    print("\n" + "=" * 60)
    print("Animation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
    
    
    
#%% Script to create the video from the frames ###



"""
#------------------------------------------------------------------------------------------------------------------------------------#
                                    A script to create a video from the frames
#------------------------------------------------------------------------------------------------------------------------------------#
"""
    
import imageio
import glob
import os

frames_dir = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\frames"
output_path = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\urans_animation.mp4"

frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
print(f"Found {len(frame_files)} frames")

writer = imageio.get_writer(output_path, fps=60, codec='libx264')
for i, f in enumerate(frame_files):
    if i % 50 == 0:
        print(f"Processing frame {i+1}/{len(frame_files)}...")
    writer.append_data(imageio.imread(f))
writer.close()

print(f"✓ Video saved to: {output_path}")

#%% Optimized Code 



#!/usr/bin/env python3
"""
OPTIMIZED URANS Animation Generator
"""

import os
import sys
import glob
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import tecplot as tp

tp.session.connect()

@dataclass
class Config:
    """Configuration settings."""
    
    # Input/Output paths
    input_dir: str = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\tecfiles"
    output_dir: str = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\frames"
    layout_file: Optional[str] = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\10_PowerPoints\6_SciTech Paper\1_Scitech PPT\2_Tecplot Layout\machLayout.lay"
    
    # File pattern
    file_pattern: str = "mcfd_tec_*.bin"
    
    # Image settings
    width: int = 1920
    supersample: int = 1
    
    # Processing options
    start_frame: int = 0
    end_frame: Optional[int] = None
    skip: int = 1  # Set to 1 to process ALL files
    
    # Annotation
    add_timestamp: bool = False
    timestamp_position: tuple = (5, 95)
    timestamp_size: int = 14
    
    # Video settings
    create_video: bool = True
    framerate: int = 60
    video_filename: str = "urans_animation.mp4"
    
    # Debugging
    show_all_files: bool = False  # Set True to print all found files


def natural_sort_key(filepath: str) -> int:
    """Extract timestep number for natural sorting."""
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0


def get_sorted_files(input_dir: str, pattern: str) -> List[str]:
    """Get all matching files sorted by timestep."""
    search_path = os.path.join(input_dir, pattern)
    files = glob.glob(search_path)
    
    if not files:
        raise FileNotFoundError(f"No files found matching: {search_path}")
    
    return sorted(files, key=natural_sort_key)


def setup_tecplot_batch():
    """Initialize PyTecplot in BATCH MODE."""
    os.environ['PYTECPLOT_BATCH_MODE'] = '1'
    import tecplot as tp
    print("✓ Running in batch mode (optimized)")
    return tp


def export_frame(tp, frame_num: int, output_dir: str, config: Config) -> str:
    """Export current visualization as PNG."""
    output_path = os.path.join(output_dir, f"frame_{frame_num:05d}.png")
    tp.export.save_png(
        output_path,
        width=config.width,
        supersample=config.supersample
    )
    return output_path


def process_files_optimized(tp, files: List[str], config: Config):
    """OPTIMIZED processing loop."""
    from tecplot.constant import ReadDataOption
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    total_files = len(files)
    start_time = time.time()
    
    print(f"\nProcessing {total_files} files...")
    print(f"Output: {os.path.abspath(config.output_dir)}")
    print(f"Resolution: {config.width} pixels wide (supersample={config.supersample})")
    print("-" * 60)
    
    # Load layout and first data file
    print("Loading layout and initial data...")
    
    if config.layout_file and os.path.exists(config.layout_file):
        tp.load_layout(config.layout_file)
        print(f"  ✓ Layout loaded: {os.path.basename(config.layout_file)}")
    
    tp.data.load_tecplot(files[0], read_data_option=ReadDataOption.Replace, reset_style=False)
    print(f"  ✓ Initial data loaded: {os.path.basename(files[0])}")
    
    export_frame(tp, 0, config.output_dir, config)
    print(f"  ✓ Frame 0 exported")
    print("-" * 60)
    
    # Process remaining files
    for i, filepath in enumerate(files[1:], start=1):
        filename = os.path.basename(filepath)
        
        progress = (i + 1) / total_files * 100
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total_files - i - 1) / rate if rate > 0 else 0
        
        print(f"[{i+1:4d}/{total_files}] {filename:30s} | "
              f"{progress:5.1f}% | {rate:.1f} fps | ETA: {eta:.0f}s", end="")
        
        try:
            tp.data.load_tecplot(
                filepath, 
                read_data_option=ReadDataOption.Replace,
                reset_style=False
            )
            export_frame(tp, i, config.output_dir, config)
            print(f" ✓")
        except Exception as e:
            print(f" ✗ Error: {e}")
            continue
    
    total_time = time.time() - start_time
    avg_rate = total_files / total_time
    
    print("-" * 60)
    print(f"Completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average rate: {avg_rate:.2f} frames/second")
    print(f"Frames saved to: {os.path.abspath(config.output_dir)}")


def create_video_ffmpeg(config: Config):
    """Create video from frames using ffmpeg."""
    import subprocess
    
    frames_pattern = os.path.join(config.output_dir, "frame_%05d.png")
    output_path = os.path.join(os.path.dirname(config.output_dir), config.video_filename)
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(config.framerate),
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path
    ]
    
    print(f"\nCreating video: {output_path}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Video created: {output_path}")
    except FileNotFoundError:
        print("✗ ffmpeg not found.")
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg error: {e}")


def run_animation_optimized(config: Config):
    """Execute the optimized animation workflow."""
    
    print("=" * 60)
    print("URANS Animation Generator - OPTIMIZED VERSION")
    print("=" * 60)
    
    # Get sorted file list
    try:
        files = get_sorted_files(config.input_dir, config.file_pattern)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"Found {len(files)} files")
    print(f"  First: {os.path.basename(files[0])}")
    print(f"  Last:  {os.path.basename(files[-1])}")
    
    # Show all files if debugging
    if config.show_all_files:
        print("\nAll files found:")
        for f in files:
            print(f"  {os.path.basename(f)}")
        print()
    
    # Apply frame selection
    end = config.end_frame if config.end_frame else len(files)
    files = files[config.start_frame:end:config.skip]
    print(f"Processing {len(files)} files (skip={config.skip})")
    
    # Setup Tecplot
    tp = setup_tecplot_batch()
    
    # Process all files
    process_files_optimized(tp, files, config)
    
    # Create video if requested
    if config.create_video:
        create_video_ffmpeg(config)
    
    print("\n" + "=" * 60)
    print("Animation complete!")
    print("=" * 60)


if __name__ == "__main__":
    config = Config()
    run_animation_optimized(config)
#%% Code 2 

"""
#------------------------------------------------------------------------------------------------------------------------------------#
                               A script to create a video that is LONGER by duplicating frames as needed 
#------------------------------------------------------------------------------------------------------------------------------------#
"""


import imageio
import glob
import os

frames_dir = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\frames_2"
output_path = r"C:\Users\hhsabbah\Documents\01_Bladeless_Proj\32_Geometry Code\urans_animation.mp4"

frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
print(f"Found {len(frame_files)} frames")

# Settings
fps = 120
repeats = 0  # Repeat each frame N times (adjust for desired duration)

# Duration calculation
total_frames = len(frame_files) * repeats
duration = total_frames / fps
print(f"Output: {total_frames} frames at {fps} fps = {duration:.1f} seconds")

writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

for i, f in enumerate(frame_files):
    print(f"Processing frame {i+1}/{len(frame_files)}...")
    img = imageio.imread(f)
    for _ in range(repeats):  # Write each frame multiple times
        writer.append_data(img)

writer.close()
print(f"✓ Video saved to: {output_path}")