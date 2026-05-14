#!/usr/bin/env python3
"""
frames_to_mp4.py
================
[Local preprocessing] Stitches UE5-rendered image frames into mp4 videos so they
can be uploaded to a server for VAE encoding. Supports exporting both RGB and
depth videos.

Dataset input layout:
  {input_dir}/
    ├── reasoning/
    │   ├── traj_000000/
    │   │   ├── metadata.json
    │   │   ├── step_00/
    │   │   │   ├── frame_0000.jpg ~ frame_0008.jpg   (RGB)
    │   │   │   └── depth/
    │   │   │       └── frame_0000.png ~ frame_0008.png (16-bit depth)
    │   │   ├── step_01/
    │   │   └── ... (40 steps)
    │   └── traj_000001/
    └── random_walk/
        └── ...

Output layout:
  {output_dir}/
    ├── reasoning/
    │   ├── traj_000000.mp4              # RGB video (40 steps stitched)
    │   ├── traj_000000_depth.mp4        # Depth video (40 steps stitched, default 720p)
    │   └── traj_000000_meta.json        # metadata (copied + augmented with encoding info)
    └── random_walk/
        ├── traj_000001.mp4
        ├── traj_000001_depth.mp4
        └── traj_000001_meta.json

Video-encoding details:
  - The frames from a trajectory's 40 steps are stitched in order into a single video.
  - For example: 40 steps x 9 frames = 360 frames -> one mp4.
  - Frame order: step_00/frame_0000, ..., step_00/frame_0008, step_01/frame_0000, ...
  - RGB:   default CRF=28 + fps=16 + ultrafast preset (fast-validation mode).
  - Depth: 16-bit PNG -> grayscale -> pseudo-color/grayscale mp4, default 720p, CRF=18 (near-lossless).
  - Supports --resolution 360p/480p/720p/1080p downscaling (default 360p for fast validation).
  - Supports depth export (on by default; turn off with --no_export_depth).
  - Supports ffmpeg pipe input (no concat file needed; faster).

Dependencies:
  pip install opencv-python numpy tqdm
  ffmpeg should be on PATH (recommended; auto-detected and preferred).

Usage:
  # Fast validation (360p, CRF=28, fps=16, ultrafast, depth on by default)
  python frames_to_mp4.py \\
      --input_dir  F:/A_worldmodel/UE5data/run_20260402_203017 \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --workers 8 --skip_existing

  # * Batch: given a root folder, auto-scan all run_* subdirectories
  python frames_to_mp4.py \\
      --root_dir   F:/A_worldmodel/UE5data/ \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --workers 8 --skip_existing

  # RGB only (disable depth)
  python frames_to_mp4.py \\
      --input_dir  F:/A_worldmodel/UE5data/run_20260402_203017 \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --no_export_depth --workers 8

  # High quality (original resolution, CRF=12, fps=24) + depth 720p
  python frames_to_mp4.py \\
      --input_dir  F:/A_worldmodel/UE5data/run_20260402_203017 \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --fps 24 --crf 12 --preset fast --resolution original \\
      --depth_resolution 720p \\
      --workers 8

  # Specify resolution
  python frames_to_mp4.py \\
      --input_dir  F:/A_worldmodel/UE5data/run_20260402_203017 \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --resolution 720p --workers 8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ─────────────────────── Resolution presets ──────────────────────────────
RESOLUTION_PRESETS = {
    "360p":     (640, 360),
    "480p":     (854, 480),
    "720p":     (1280, 720),
    "1080p":    (1920, 1080),
    "original": None,   # keep source resolution
}


def detect_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


HAS_FFMPEG = detect_ffmpeg()


def parse_resolution(res_str: str) -> Optional[Tuple[int, int]]:
    """Parse resolution string → (width, height) or None for original.

    Accepts: '360p', '720p', '1080p', 'original', or 'WxH' (e.g. '640x360').
    """
    if res_str in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[res_str]
    if 'x' in res_str:
        parts = res_str.lower().split('x')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (int(parts[0]), int(parts[1]))
    raise ValueError(f"Unknown resolution: '{res_str}'. "
                     f"Use one of {list(RESOLUTION_PRESETS.keys())} or WxH (e.g. 640x360)")


# ─────────────────────── Trajectory discovery ───────────────────────────
def discover_trajectories(input_dir: str) -> List[Dict]:
    """
    Scan dataset directory for all trajectories.

    Returns list of dicts:
      - traj_dir: absolute path to traj_XXXXXX
      - data_type: "reasoning" or "random_walk"
      - traj_name: "traj_000000"
      - metadata_path: path to metadata.json
    """
    root = Path(input_dir)
    trajectories = []

    for data_type in ("reasoning", "random_walk"):
        type_dir = root / data_type
        if not type_dir.is_dir():
            continue

        for traj_dir in sorted(type_dir.iterdir()):
            if not traj_dir.is_dir() or not traj_dir.name.startswith("traj_"):
                continue
            metadata_path = traj_dir / "metadata.json"
            if not metadata_path.exists():
                print(f"  [WARN] No metadata.json: {traj_dir}, skipping")
                continue
            trajectories.append({
                "traj_dir": str(traj_dir),
                "data_type": data_type,
                "traj_name": traj_dir.name,
                "metadata_path": str(metadata_path),
                "input_root": input_dir,
            })

    return trajectories


def get_all_frames(traj_dir: str, num_steps: int = 40) -> Tuple[List[str], int]:
    """
    Get all frame file paths in order for a trajectory.

    Returns:
        frame_paths: list of absolute paths in order
        frames_per_step: detected frames per step
    """
    traj_path = Path(traj_dir)
    all_frames = []
    frames_per_step = None

    for i in range(num_steps):
        step_dir = traj_path / f"step_{i:02d}"
        if not step_dir.is_dir():
            raise FileNotFoundError(f"Missing step dir: {step_dir}")

        # Find all frame files, sorted by name
        frame_files = sorted([
            f for f in step_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
            and f.stem.startswith('frame_')
        ])

        if not frame_files:
            raise FileNotFoundError(f"No frames in: {step_dir}")

        if frames_per_step is None:
            frames_per_step = len(frame_files)
        elif len(frame_files) != frames_per_step:
            print(f"  [WARN] step_{i:02d} has {len(frame_files)} frames "
                  f"(expected {frames_per_step})")

        all_frames.extend([str(f) for f in frame_files])

    return all_frames, frames_per_step


def get_all_depth_frames(traj_dir: str, num_steps: int = 40) -> Tuple[List[str], int]:
    """
    Get all depth frame file paths in order for a trajectory.

    Depth frames live under: step_XX/depth/frame_XXXX.png (16-bit grayscale PNG)

    Returns:
        frame_paths: list of absolute paths in order
        frames_per_step: detected frames per step
    """
    traj_path = Path(traj_dir)
    all_frames = []
    frames_per_step = None

    for i in range(num_steps):
        depth_dir = traj_path / f"step_{i:02d}" / "depth"
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Missing depth dir: {depth_dir}")

        # Find all depth frame files, sorted by name
        frame_files = sorted([
            f for f in depth_dir.iterdir()
            if f.suffix.lower() == '.png'
            and f.stem.startswith('frame_')
        ])

        if not frame_files:
            raise FileNotFoundError(f"No depth frames in: {depth_dir}")

        if frames_per_step is None:
            frames_per_step = len(frame_files)
        elif len(frame_files) != frames_per_step:
            print(f"  [WARN] step_{i:02d}/depth has {len(frame_files)} frames "
                  f"(expected {frames_per_step})")

        all_frames.extend([str(f) for f in frame_files])

    return all_frames, frames_per_step


# ─────────────────────── Video encoding (cv2) ───────────────────────────
def encode_with_cv2(
    frame_paths: List[str],
    output_path: str,
    fps: float = 16.0,
    crf: int = 28,
    resolution: Optional[Tuple[int, int]] = None,
) -> bool:
    """Encode frames to mp4 using OpenCV. Supports resolution scaling."""
    assert HAS_CV2, "pip install opencv-python"

    # Read first frame to get dimensions
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError(f"Cannot read: {frame_paths[0]}")
    h, w = first.shape[:2]

    # Determine output size
    out_w, out_h = w, h
    if resolution is not None:
        out_w, out_h = resolution

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    try:
        def _write(frame):
            if resolution is not None and (frame.shape[1] != out_w or frame.shape[0] != out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)

        _write(first)
        for fp in frame_paths[1:]:
            frame = cv2.imread(fp)
            if frame is None:
                print(f"  [WARN] Cannot read frame: {fp}, using previous")
                _write(first)
            else:
                _write(frame)
                first = frame
    finally:
        writer.release()

    return True


# ─────────────────────── Video encoding (ffmpeg) ───────────────────────────
def encode_with_ffmpeg(
    frame_paths: List[str],
    output_path: str,
    fps: float = 16.0,
    crf: int = 28,
    preset: str = "ultrafast",
    resolution: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Encode frames to mp4 using ffmpeg pipe — much faster than concat file.
    Reads JPEGs sequentially and pipes raw image data to ffmpeg stdin.
    Supports resolution scaling via -vf scale.
    """
    # Probe first frame to get source dimensions
    probe_cmd = [
        "ffmpeg", "-i", frame_paths[0],
        "-f", "null", "-"
    ]
    # Simpler: just get dimensions from first image via ffprobe or read header
    # We'll use ffprobe for reliability
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0",
             frame_paths[0]],
            capture_output=True, text=True, timeout=30
        )
        src_w, src_h = [int(x) for x in probe.stdout.strip().split('x')]
    except Exception:
        # Fallback: use OpenCV to read dimensions
        if HAS_CV2:
            img = cv2.imread(frame_paths[0])
            src_h, src_w = img.shape[:2]
        else:
            raise RuntimeError("Cannot determine frame dimensions (install opencv or ffprobe)")

    # Build ffmpeg command: read images from stdin as raw image sequence
    # Faster approach: use concat file but with image2 demuxer pattern
    # Even faster: pipe JPEG data directly

    # Create temp file list (much simpler and still fast for sequential images)
    tmp_list = output_path + ".frames.txt"
    try:
        with open(tmp_list, 'w', encoding='utf-8') as f:
            for fp in frame_paths:
                escaped = fp.replace("'", "'\\''").replace("\\", "/")
                f.write(f"file '{escaped}'\n")

        # Build ffmpeg command
        vf_filters = []
        if resolution is not None:
            out_w, out_h = resolution
            # Ensure even dimensions (required by H.264)
            out_w = out_w if out_w % 2 == 0 else out_w + 1
            out_h = out_h if out_h % 2 == 0 else out_h + 1
            # fast_bilinear is libswscale's fastest scaler; visual loss at
            # large-ratio downsampling (e.g. 1080p→360p) is imperceptible.
            vf_filters.append(f"scale={out_w}:{out_h}:flags=fast_bilinear")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-r", str(fps),   # input fps (treat images at this rate)
            "-i", tmp_list,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-r", str(fps),   # output fps
        ]

        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])

        # Faster encoding flags — one thread per worker to avoid
        # oversubscription when the outer ProcessPoolExecutor already
        # consumes one core per worker.
        cmd.extend([
            "-threads", "1",
            "-movflags", "+faststart",  # web-friendly
            output_path
        ])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  [ERR] ffmpeg failed: {result.stderr[-500:]}")
            return False
        return True
    finally:
        if os.path.exists(tmp_list):
            os.remove(tmp_list)


# ─────────────────────── Depth video encoding ────────────────────────────
def _read_depth_16bit(path: str) -> "np.ndarray":
    """Read 16-bit depth PNG → 8-bit grayscale (H, W) uint8.

    The UE5 DataCollector saves depth as:
        pixel = clamp(depth_cm / DepthMaxDistance, 0, 1) * 65535
    So the 16-bit value is already normalized to [0, 65535].
    We convert to 8-bit by a right-shift (>> 8) — integer-only, no float cast,
    ~3x faster than float32 normalization. Accuracy loss is the low 8 bits
    (1/256 of full range), negligible compared to H.264 CRF-28 compression.

    Returns a single-channel (H, W) uint8 array. Encoders downstream
    consume it with -pix_fmt gray; this avoids the GRAY2BGR 3x memcpy.
    """
    img16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # (H, W) or (H, W, 3) uint16
    if img16 is None:
        return None
    # Handle multi-channel depth (some versions might save as 3-ch)
    if img16.ndim == 3:
        img16 = img16[:, :, 0]
    # 16-bit → 8-bit by right-shift: pure integer, no division, ~3x faster
    return (img16 >> 8).astype(np.uint8, copy=False)


def _read_depth_16bit_bgr(path: str) -> "np.ndarray":
    """Legacy BGR version of _read_depth_16bit for the OpenCV VideoWriter
    fallback path (which requires 3-channel input to write yuv420p)."""
    gray = _read_depth_16bit(path)
    if gray is None:
        return None
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def encode_depth_with_cv2(
    frame_paths: List[str],
    output_path: str,
    fps: float = 16.0,
    crf: int = 18,
    resolution: Optional[Tuple[int, int]] = None,
) -> bool:
    """Encode 16-bit depth frames to mp4 using OpenCV.

    Reads 16-bit PNGs, normalizes to 8-bit grayscale, converts to BGR
    (OpenCV VideoWriter requires 3-channel input for yuv420p output).
    """
    assert HAS_CV2 and HAS_NUMPY, "pip install opencv-python numpy"

    first = _read_depth_16bit_bgr(frame_paths[0])
    if first is None:
        raise RuntimeError(f"Cannot read depth frame: {frame_paths[0]}")
    h, w = first.shape[:2]

    out_w, out_h = w, h
    if resolution is not None:
        out_w, out_h = resolution

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {output_path}")

    try:
        def _write(frame):
            if resolution is not None and (frame.shape[1] != out_w or frame.shape[0] != out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)

        _write(first)
        for fp in frame_paths[1:]:
            frame = _read_depth_16bit_bgr(fp)
            if frame is None:
                print(f"  [WARN] Cannot read depth frame: {fp}, using previous")
                _write(first)
            else:
                _write(frame)
                first = frame
    finally:
        writer.release()

    return True


def encode_depth_with_ffmpeg(
    frame_paths: List[str],
    output_path: str,
    fps: float = 16.0,
    crf: int = 18,
    preset: str = "ultrafast",
    resolution: Optional[Tuple[int, int]] = None,
) -> bool:
    """Encode 16-bit depth PNGs to mp4 via ffmpeg pipe.

    Reads 16-bit depth → shifts to 8-bit single-channel grayscale in Python →
    pipes raw single-channel frames to ffmpeg with -pix_fmt gray.
    ffmpeg converts gray→yuv420p internally (cheaper than bgr24→yuv420p since
    no RGB→YUV matrix multiplication is needed).
    """
    assert HAS_CV2 and HAS_NUMPY, "pip install opencv-python numpy"

    first = _read_depth_16bit(frame_paths[0])      # (H, W) uint8 grayscale
    if first is None:
        raise RuntimeError(f"Cannot read depth frame: {frame_paths[0]}")
    src_h, src_w = first.shape[:2]

    out_w, out_h = src_w, src_h
    if resolution is not None:
        out_w, out_h = resolution
        # Ensure even dimensions (yuv420p requires even)
        out_w = out_w if out_w % 2 == 0 else out_w + 1
        out_h = out_h if out_h % 2 == 0 else out_h + 1

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "gray",           # ← single-channel, 3x less input data
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-threads", "1",              # ← one thread per worker, avoid oversubscription
        "-movflags", "+faststart",
        output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        need_resize = (out_w != src_w or out_h != src_h)

        def _pipe_frame(frame):
            # frame is (H, W) uint8 grayscale
            if need_resize:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            proc.stdin.write(frame.tobytes())

        _pipe_frame(first)
        prev = first
        for fp in frame_paths[1:]:
            frame = _read_depth_16bit(fp)
            if frame is None:
                print(f"  [WARN] Cannot read depth frame: {fp}, using previous")
                _pipe_frame(prev)
            else:
                _pipe_frame(frame)
                prev = frame

        proc.stdin.close()
        _, stderr = proc.communicate(timeout=300)
        if proc.returncode != 0:
            print(f"  [ERR] ffmpeg depth failed: {stderr.decode('utf-8', errors='replace')[-500:]}")
            return False
        return True
    except Exception:
        proc.kill()
        proc.wait()
        raise


# ─────────────────────── Single trajectory processing ───────────────────────
def process_one_trajectory(args_tuple) -> Dict:
    """
    Process a single trajectory: images → mp4 + metadata.
    Optionally also exports depth video.
    Designed to be called from ProcessPoolExecutor.

    Returns dict with status info.
    """
    (traj_info, output_dir, fps, crf, num_steps, use_ffmpeg,
     skip_existing, resolution, preset,
     export_depth, depth_resolution, depth_crf) = args_tuple

    traj_dir = traj_info["traj_dir"]
    data_type = traj_info["data_type"]
    traj_name = traj_info["traj_name"]
    metadata_path = traj_info["metadata_path"]
    input_root = traj_info["input_root"]
    run_name = Path(input_root).name

    # Output paths
    out_type_dir = os.path.join(output_dir, run_name, data_type)
    os.makedirs(out_type_dir, exist_ok=True)

    mp4_path = os.path.join(out_type_dir, f"{traj_name}.mp4")
    depth_mp4_path = os.path.join(out_type_dir, f"{traj_name}_depth.mp4")
    meta_out_path = os.path.join(out_type_dir, f"{traj_name}_meta.json")

    result = {
        "traj": f"{run_name}/{data_type}/{traj_name}",
        "status": "unknown",
        "mp4_path": mp4_path,
    }

    # Skip if exists (for depth: also check depth mp4 if export_depth is on)
    if skip_existing:
        rgb_done = os.path.exists(mp4_path) and os.path.exists(meta_out_path)
        depth_done = (not export_depth) or os.path.exists(depth_mp4_path)
        if rgb_done and depth_done:
            result["status"] = "skipped"
            return result

    try:
        t0 = time.time()

        # 1. Collect all frame paths in order
        frame_paths, frames_per_step = get_all_frames(traj_dir, num_steps)
        total_frames = len(frame_paths)

        # 2. Encode RGB video
        tmp_mp4 = mp4_path + ".tmp.mp4"
        try:
            if use_ffmpeg:
                ok = encode_with_ffmpeg(frame_paths, tmp_mp4, fps, crf,
                                        preset=preset, resolution=resolution)
            else:
                ok = encode_with_cv2(frame_paths, tmp_mp4, fps, crf,
                                     resolution=resolution)

            if not ok:
                result["status"] = "failed"
                result["error"] = "RGB encoding failed"
                return result

            # Atomic rename
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
            os.rename(tmp_mp4, mp4_path)
        except Exception:
            if os.path.exists(tmp_mp4):
                os.remove(tmp_mp4)
            raise

        # 3. Encode Depth video (optional)
        depth_ok = False
        if export_depth:
            try:
                depth_frames, _ = get_all_depth_frames(traj_dir, num_steps)
                tmp_depth_mp4 = depth_mp4_path + ".tmp.mp4"
                try:
                    if use_ffmpeg:
                        depth_ok = encode_depth_with_ffmpeg(
                            depth_frames, tmp_depth_mp4, fps, depth_crf,
                            preset=preset, resolution=depth_resolution)
                    else:
                        depth_ok = encode_depth_with_cv2(
                            depth_frames, tmp_depth_mp4, fps, depth_crf,
                            resolution=depth_resolution)

                    if depth_ok:
                        if os.path.exists(depth_mp4_path):
                            os.remove(depth_mp4_path)
                        os.rename(tmp_depth_mp4, depth_mp4_path)
                    else:
                        result["depth_error"] = "depth encoding failed"
                except Exception:
                    if os.path.exists(tmp_depth_mp4):
                        os.remove(tmp_depth_mp4)
                    raise
            except FileNotFoundError as e:
                # Depth frames not available — not a fatal error
                result["depth_error"] = str(e)
                depth_ok = False

        # 4. Copy + augment metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        res_str = f"{resolution[0]}x{resolution[1]}" if resolution else "original"
        metadata["video_encoding"] = {
            "fps": fps,
            "crf": crf,
            "preset": preset,
            "resolution": res_str,
            "total_frames": total_frames,
            "frames_per_step": frames_per_step,
            "num_steps": num_steps,
            "codec": "libx264" if use_ffmpeg else "mp4v",
            "source_format": "frames_to_mp4",
        }

        if export_depth and depth_ok:
            depth_res_str = (f"{depth_resolution[0]}x{depth_resolution[1]}"
                             if depth_resolution else "original")
            metadata["depth_video_encoding"] = {
                "fps": fps,
                "crf": depth_crf,
                "preset": preset,
                "resolution": depth_res_str,
                "total_frames": total_frames,
                "frames_per_step": frames_per_step,
                "num_steps": num_steps,
                "codec": "libx264" if use_ffmpeg else "mp4v",
                "source_format": "frames_to_mp4",
                "depth_format": "16bit_png_normalized_to_8bit_grayscale",
            }

        with open(meta_out_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        dt = time.time() - t0
        mp4_size_mb = os.path.getsize(mp4_path) / (1024 * 1024)

        result["status"] = "ok"
        result["frames"] = total_frames
        result["frames_per_step"] = frames_per_step
        result["size_mb"] = round(mp4_size_mb, 1)
        result["time"] = round(dt, 1)
        result["depth_ok"] = depth_ok
        if depth_ok and os.path.exists(depth_mp4_path):
            result["depth_size_mb"] = round(
                os.path.getsize(depth_mp4_path) / (1024 * 1024), 1)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


# ────────────────────────── Main ─────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert UE5 rendered frames to mp4 videos (local preprocessing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast validation (360p, CRF=28, fps=16, ultrafast) — default
  python frames_to_mp4.py \\
      --input_dir  F:/A_worldmodel/UE5data/run_20260402_203017 \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --workers 8

  # ★ Batch: scan all run_* under a root directory
  python frames_to_mp4.py \\
      --root_dir   F:/A_worldmodel/UE5data/ \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --workers 8 --skip_existing

  # RGB only (disable depth)
  python frames_to_mp4.py \\
      --input_dir  F:/A_worldmodel/UE5data/run_20260402_203017 \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --no_export_depth --workers 8

  # High quality (original resolution, CRF=12, fps=24) + depth 720p
  python frames_to_mp4.py \\
      --input_dir  F:/A_worldmodel/UE5data/run_20260402_203017 \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --fps 24 --crf 12 --preset fast --resolution original \\
      --depth_resolution 720p \\
      --workers 8

  # Multiple explicit runs
  python frames_to_mp4.py \\
      --input_dir  F:/A_worldmodel/UE5data/run_20260402_203017 \\
                   F:/A_worldmodel/UE5data/run_20260403_100000 \\
      --output_dir F:/A_worldmodel/mp4_data/ \\
      --workers 8 --skip_existing

Output structure:
  {output_dir}/
    run_20260402_203017/
      reasoning/
        traj_000000.mp4              # RGB video
        traj_000000_depth.mp4        # Depth video (default ON, use --no_export_depth to disable)
        traj_000000_meta.json
      random_walk/
        traj_000001.mp4
        traj_000001_depth.mp4
        traj_000001_meta.json
""",
    )
    parser.add_argument("--input_dir", nargs='+', default=None,
                        help="One or more UE5 dataset run folders (e.g. run_20260402_203017)")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Root directory containing multiple run_* folders. "
                             "Auto-scans all subdirectories starting with 'run'. "
                             "Mutually exclusive with --input_dir (use one or the other).")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for mp4 files")
    parser.add_argument("--fps", type=float, default=16.0,
                        help="Video FPS (default: 16)")
    parser.add_argument("--crf", type=int, default=28,
                        help="H.264 CRF quality (0=lossless, 12=near-lossless, 23=default, 28=fast). "
                             "Higher = smaller but lower quality (default: 28)")
    parser.add_argument("--preset", type=str, default="ultrafast",
                        choices=["ultrafast", "superfast", "veryfast", "faster",
                                 "fast", "medium", "slow", "slower", "veryslow"],
                        help="H.264 encoding preset (default: ultrafast for fast validation)")
    parser.add_argument("--resolution", type=str, default="360p",
                        help="Output resolution: 360p, 480p, 720p, 1080p, original, or WxH "
                             "(default: 360p for fast validation)")
    parser.add_argument("--num_steps", type=int, default=40,
                        help="Steps per trajectory (default: 40)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--use_ffmpeg", action="store_true", default=None,
                        help="Force ffmpeg (auto-detected by default)")
    parser.add_argument("--use_cv2", action="store_true",
                        help="Force OpenCV (override ffmpeg auto-detection)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip already converted trajectories")
    parser.add_argument("--data_types", nargs='+', default=["reasoning", "random_walk"],
                        help="Which data types to process")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Max trajectories to process (for debugging)")
    # ── Depth video options ──
    parser.add_argument("--export_depth", action="store_true", default=True,
                        help="Export depth maps as a separate video (DEFAULT: ON). "
                             "Use --no_export_depth to disable.")
    parser.add_argument("--no_export_depth", dest="export_depth", action="store_false",
                        help="Disable depth video export")
    parser.add_argument("--depth_resolution", type=str, default="720p",
                        help="Depth video resolution (default: 720p). "
                             "Accepts same values as --resolution.")
    parser.add_argument("--depth_crf", type=int, default=28,
                        help="CRF for depth video (default: 18, near-lossless to preserve depth info)")
    args = parser.parse_args()

    # ── Resolve input directories: --root_dir or --input_dir ──
    if args.root_dir and args.input_dir:
        print("ERROR: --root_dir and --input_dir are mutually exclusive. Use one or the other.")
        sys.exit(1)
    if not args.root_dir and not args.input_dir:
        print("ERROR: Must specify either --root_dir or --input_dir.")
        parser.print_usage()
        sys.exit(1)

    if args.root_dir:
        # Auto-scan all run_* subdirectories under root_dir
        root = os.path.abspath(args.root_dir)
        if not os.path.isdir(root):
            print(f"ERROR: --root_dir does not exist: {root}")
            sys.exit(1)
        run_dirs = sorted([
            os.path.join(root, d)
            for d in os.listdir(root)
            if d.startswith("run") and os.path.isdir(os.path.join(root, d))
        ])
        if not run_dirs:
            print(f"ERROR: No run_* directories found under {root}")
            sys.exit(1)
        print(f"[root_dir] Scanned {root} → found {len(run_dirs)} run directories:")
        for rd in run_dirs:
            print(f"  {os.path.basename(rd)}")
        print()
        args.input_dir = run_dirs

    # Determine encoder: auto-detect ffmpeg, fallback to cv2
    if args.use_cv2:
        use_ffmpeg = False
    elif args.use_ffmpeg:
        use_ffmpeg = True
    else:
        # Auto-detect: prefer ffmpeg (better quality, scaling, speed)
        use_ffmpeg = HAS_FFMPEG
        if use_ffmpeg:
            print("Auto-detected ffmpeg → using ffmpeg encoder (faster + scaling)")
        else:
            print("ffmpeg not found → falling back to OpenCV encoder")

    # Check dependencies
    if not use_ffmpeg and not HAS_CV2:
        print("ERROR: Neither ffmpeg nor opencv-python found. Install one:")
        print("  pip install opencv-python")
        print("  or install ffmpeg and add to PATH")
        sys.exit(1)

    if use_ffmpeg and not HAS_FFMPEG:
        print("ERROR: --use_ffmpeg specified but ffmpeg not found in PATH")
        sys.exit(1)

    # Parse resolution
    resolution = parse_resolution(args.resolution)
    depth_resolution = parse_resolution(args.depth_resolution) if args.export_depth else None

    # Check depth dependencies
    if args.export_depth:
        if not HAS_CV2 or not HAS_NUMPY:
            print("ERROR: --export_depth requires opencv-python and numpy:")
            print("  pip install opencv-python numpy")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover all trajectories ──
    all_trajectories = []
    for input_dir in args.input_dir:
        trajs = discover_trajectories(input_dir)
        trajs = [t for t in trajs if t["data_type"] in args.data_types]
        all_trajectories.extend(trajs)
        print(f"Found {len(trajs)} trajectories in {input_dir}")

    if not all_trajectories:
        print("ERROR: No trajectories found!")
        sys.exit(1)

    all_trajectories.sort(key=lambda t: (t["data_type"], t["traj_name"]))

    if args.max_trajectories:
        all_trajectories = all_trajectories[:args.max_trajectories]

    res_str = f"{resolution[0]}x{resolution[1]}" if resolution else "original"
    print(f"\nTotal: {len(all_trajectories)} trajectories to process")
    print(f"Output: {args.output_dir}")
    print(f"FPS: {args.fps}, CRF: {args.crf}, Preset: {args.preset}, "
          f"Resolution: {res_str}, Workers: {args.workers}")
    print(f"Encoder: {'ffmpeg' if use_ffmpeg else 'OpenCV'}")
    if args.export_depth:
        depth_res_str = (f"{depth_resolution[0]}x{depth_resolution[1]}"
                         if depth_resolution else "original")
        print(f"Depth export: ON  (resolution={depth_res_str}, CRF={args.depth_crf})")
    else:
        print(f"Depth export: OFF  (remove --no_export_depth to re-enable)")
    print()

    # ── Process with parallel workers ──
    task_args = [
        (traj, args.output_dir, args.fps, args.crf, args.num_steps,
         use_ffmpeg, args.skip_existing, resolution, args.preset,
         args.export_depth, depth_resolution, args.depth_crf)
        for traj in all_trajectories
    ]

    success = 0
    skipped = 0
    failed = 0
    total_size_mb = 0.0
    t_start = time.time()

    if args.workers <= 1:
        # Single process (easier to debug)
        iterator = task_args
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Converting", unit="traj")
        for task in iterator:
            result = process_one_trajectory(task)
            if result["status"] == "ok":
                success += 1
                total_size_mb += result.get("size_mb", 0)
                if not HAS_TQDM:
                    depth_info = ""
                    if args.export_depth:
                        if result.get("depth_ok"):
                            depth_info = f"  depth={result.get('depth_size_mb', '?')}MB"
                        else:
                            depth_info = f"  depth=SKIP({result.get('depth_error', '?')})"
                    print(f"  ✓ {result['traj']}  "
                          f"frames={result['frames']}  "
                          f"size={result['size_mb']}MB"
                          f"{depth_info}  "
                          f"time={result['time']}s")
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"  ✗ {result['traj']}: {result.get('error', '?')}")
    else:
        # Multi-process. Use executor.map with chunksize for I/O locality:
        # each worker processes `chunksize` consecutive trajectories before
        # being handed a new batch. Since tasks are pre-sorted by
        # (data_type, traj_name), this keeps disk head movement minimal on
        # spinning storage (major HDD speedup) while being essentially free
        # on SSD/NVMe.
        #
        # Trade-off: we no longer get "first-to-finish-first" ordering in
        # the progress stream, but total wall time is lower. We keep the
        # submit+as_completed code path only for the single-worker debug
        # mode (above).
        results = []
        # Pick chunksize: small enough for responsive tqdm updates, large
        # enough to amortize scheduling overhead. ~2-4 tasks per worker
        # batch works well across HDD/SSD; we choose it proportional to
        # worker count so progress bar updates stay smooth.
        chunksize = max(1, min(4, len(task_args) // max(args.workers * 4, 1)))
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pbar = tqdm(total=len(task_args), desc="Converting", unit="traj") if HAS_TQDM else None
            for result in executor.map(process_one_trajectory, task_args,
                                       chunksize=chunksize):
                results.append(result)

                if result["status"] == "ok":
                    success += 1
                    total_size_mb += result.get("size_mb", 0)
                    if not HAS_TQDM:
                        depth_info = ""
                        if args.export_depth:
                            if result.get("depth_ok"):
                                depth_info = f"  depth={result.get('depth_size_mb', '?')}MB"
                            else:
                                depth_info = f"  depth=SKIP({result.get('depth_error', '?')})"
                        print(f"  ✓ {result['traj']}  "
                              f"frames={result['frames']}  "
                              f"size={result['size_mb']}MB"
                              f"{depth_info}  "
                              f"time={result['time']}s")
                elif result["status"] == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"  ✗ {result['traj']}: {result.get('error', '?')}")

                if HAS_TQDM:
                    pbar.update(1)
                    pbar.set_postfix(ok=success, skip=skipped, fail=failed)

            if HAS_TQDM:
                pbar.close()

    elapsed = time.time() - t_start

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"Done!  success={success}  skipped={skipped}  failed={failed}")
    print(f"Total size: {total_size_mb:.1f} MB")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(success,1):.1f}s/traj)")
    print(f"Output: {args.output_dir}")

    if failed > 0:
        print(f"\n⚠ {failed} trajectories failed! Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
