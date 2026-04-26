"""
Schlieren Image Sequence → Video Converter
==========================================
Converts a folder of high-speed camera schlieren images into an MP4 video.

Usage:
    python schlieren_to_video.py --input_dir ./Optimized_Schlieren --fps 1000 --output schlieren_video.mp4

Key concepts for you to know:
-----------------------------
1. OpenCV's VideoWriter uses a FourCC codec code to compress frames.
   'mp4v' is widely compatible. 'avc1' (H.264) gives smaller files but
   needs the codec installed. We try H.264 first, fall back to mp4v.

2. Frame ordering matters — we sort filenames naturally so frame_2 comes
   before frame_10 (lexicographic sort would put 10 before 2).

3. All frames must have the same resolution for VideoWriter. The script
   reads the first image to lock in the size, then resizes any mismatches.

4. FPS should match your camera's recording rate for real-time playback,
   or you can slow it down (e.g., recorded at 100kfps, play at 30fps
   for ~3333x slow motion).

Dependencies:
    pip install opencv-python natsort
"""

import cv2
import argparse
import sys
from pathlib import Path

# natsort gives us "natural" sorting: frame_1, frame_2, ..., frame_10
# instead of the lexicographic: frame_1, frame_10, frame_2, ...
try:
    from natsort import natsorted
except ImportError:
    print("natsort not found. Install it with: pip install natsort")
    print("Falling back to basic sorted() — order may be wrong if filenames lack zero-padding.")
    natsorted = sorted


# ── Supported image extensions ──────────────────────────────────────
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def collect_images(input_dir: Path) -> list[Path]:
    """
    Gather and naturally sort all image files from the input directory.
    Natural sorting ensures frame_2 < frame_10, which plain sort() does NOT.
    """
    images = [f for f in input_dir.iterdir()
              if f.suffix.lower() in IMAGE_EXTENSIONS]

    if not images:
        print(f"ERROR: No image files found in {input_dir}")
        print(f"  Looked for extensions: {IMAGE_EXTENSIONS}")
        sys.exit(1)

    # natsorted uses the key= to sort by filename (not full path)
    images = natsorted(images, key=lambda p: p.name)
    print(f"Found {len(images)} images (first: {images[0].name}, last: {images[-1].name})")
    return images


def get_codec_and_extension(output_path: str) -> tuple:
    """
    Try H.264 first (smaller files, better quality), fall back to mp4v.

    FourCC is a 4-character code identifying the video codec.
    Common ones:
      - 'avc1' or 'H264' → H.264, great compression, needs system codec
      - 'mp4v'            → MPEG-4, universally supported, larger files
      - 'XVID'            → good for .avi containers
    """
    # H.264 attempt
    fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1')
    # mp4v fallback
    fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')

    # We'll try H.264 first in the main function and fall back if it fails
    return fourcc_h264, fourcc_mp4v


def create_video(images: list[Path], output_path: str, fps: float,
                 scale: float = 1.0, add_frame_number: bool = False):
    """
    Write sorted image frames into a video file.

    Parameters
    ----------
    images : list of Paths to image files (already sorted)
    output_path : where to save the .mp4
    fps : frames per second for the output video
    scale : resize factor (0.5 = half resolution, useful for large images)
    add_frame_number : burn frame index into each frame (handy for debugging)
    """
    # ── Read first frame to determine video dimensions ──────────────
    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        print(f"ERROR: Could not read {images[0]}")
        sys.exit(1)

    h, w = first_frame.shape[:2]

    # Apply scaling if requested
    if scale != 1.0:
        w = int(w * scale)
        h = int(h * scale)
        print(f"Scaling frames to {w}x{h} (factor={scale})")

    print(f"Video dimensions: {w}x{h}, FPS: {fps}")

    # ── Initialize VideoWriter ──────────────────────────────────────
    fourcc_h264, fourcc_mp4v = get_codec_and_extension(output_path)

    writer = cv2.VideoWriter(output_path, fourcc_h264, fps, (w, h))

    # Check if H.264 writer opened successfully
    if not writer.isOpened():
        print("H.264 codec not available, falling back to mp4v...")
        writer = cv2.VideoWriter(output_path, fourcc_mp4v, fps, (w, h))

    if not writer.isOpened():
        print("ERROR: Could not open VideoWriter. Check codec availability.")
        sys.exit(1)

    # ── Write frames ────────────────────────────────────────────────
    total = len(images)
    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"  WARNING: Skipping unreadable frame {img_path.name}")
            continue

        # Resize if needed (either from scale or mismatched frame sizes)
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        # Optional: burn frame number into the image
        if add_frame_number:
            cv2.putText(frame, f"Frame {i}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        writer.write(frame)

        # Progress indicator every 10%
        if (i + 1) % max(1, total // 10) == 0:
            print(f"  Progress: {i+1}/{total} ({100*(i+1)//total}%)")

    writer.release()

    # ── Verify output ───────────────────────────────────────────────
    out = Path(output_path)
    if out.exists() and out.stat().st_size > 0:
        size_mb = out.stat().st_size / (1024 * 1024)
        duration = total / fps
        print(f"\nVideo saved: {output_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Duration: {duration:.2f}s at {fps} fps ({total} frames)")
    else:
        print("ERROR: Output file was not created or is empty.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of schlieren images into a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time playback at camera frame rate
  python schlieren_to_video.py -i ./frames -fps 100000 -o realtime.mp4

  # Slow motion: recorded at 100kfps, play at 30fps
  python schlieren_to_video.py -i ./frames -fps 30 -o slowmo.mp4

  # Half resolution with frame numbers burned in
  python schlieren_to_video.py -i ./frames -fps 1000 -o debug.mp4 --scale 0.5 --frame_numbers
        """
    )

    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Path to folder containing image frames')
    parser.add_argument('-fps', '--fps', type=float, default=30.0,
                        help='Output video frame rate (default: 30)')
    parser.add_argument('-o', '--output', type=str, default='schlieren_video.mp4',
                        help='Output video filename (default: schlieren_video.mp4)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Resize factor, e.g. 0.5 for half res (default: 1.0)')
    parser.add_argument('--frame_numbers', action='store_true',
                        help='Burn frame index onto each frame')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} is not a valid directory")
        sys.exit(1)

    images = collect_images(input_dir)
    create_video(images, args.output, args.fps, args.scale, args.frame_numbers)


if __name__ == '__main__':
    main()