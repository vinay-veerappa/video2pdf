import cv2
import imutils
import time
import os
import subprocess
import shutil
import glob
import sys
import re
from config import (
    VAR_THRESHOLD, DETECT_SHADOWS, MIN_TIME_BETWEEN_CAPTURES, 
    MAX_SIMILARITY_COMPARISONS
)
from similarity import calculate_similarity, dhash, calculate_hamming_distance

def get_frames(video_path, frame_rate):
    """Generator function to return frames from video at specified frame rate"""
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        raise Exception(f'Unable to open file {video_path}')

    total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vs.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {total_duration/60:.2f} minutes")
    print(f"Processing at {frame_rate} frames per second")

    frame_time = 0
    frame_count = 0

    while True:
        vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
        frame_time += 1 / frame_rate

        (_, frame) = vs.read()
        if frame is None:
            break

        frame_count += 1
        yield frame_count, frame_time, frame

    vs.release()


def get_keyframe_timestamps(video_path):
    """Get exact PTS timestamps for all keyframes using ffprobe (Fast and Reliable)"""
    try:
        cmd = [
            "ffprobe", "-v", "error", 
            "-skip_frame", "nokey", 
            "-select_streams", "v:0", 
            "-show_entries", "frame=pts_time", 
            "-of", "csv=p=0", 
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        timestamps = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                try:
                    timestamps.append(float(line))
                except ValueError:
                    continue
        return timestamps
    except Exception as e:
        print(f"Warning: ffprobe failed to get timestamps: {e}")
        return []

def detect_unique_screenshots(video_path, output_folder_screenshot_path, 
                              frame_rate, min_percent, max_percent,
                              use_similarity=True, similarity_threshold=0.8,
                              min_time_interval=MIN_TIME_BETWEEN_CAPTURES,
                              save_duplicates_path=None,
                              similarity_method='grid',
                              progress_callback=None):
    """
    Detect and save unique screenshots using FFmpeg for fast extraction 
    followed by Python-based deduplication.
    """
    start_time = time.time()
    
    # Check if images already exist
    if os.path.exists(output_folder_screenshot_path):
        existing_images = glob.glob(os.path.join(output_folder_screenshot_path, "*.png"))
        if len(existing_images) > 0:
            print(f"Images already exist. Skipping extraction.")
            return len(existing_images)

    # 0. Pre-scan for Master List of Timestamps (Reliable Source)
    print("Pre-scanning video for keyframe timestamps...")
    timestamps = get_keyframe_timestamps(video_path)
    if timestamps:
        print(f"Found {len(timestamps)} keyframe candidates via ffprobe.")

    # 1. Extract ALL frames to a temp directory
    temp_extract_dir = os.path.join(os.path.dirname(output_folder_screenshot_path), "temp_extracted")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    # Get total duration for tertiary fallback
    duration = 0
    try:
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        total_frames_count = vs.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0:
            duration = total_frames_count / fps
        vs.release()
    except:
        pass

    print(f"Extracting I-frames using FFmpeg...")
    if progress_callback:
        progress_callback({
            'status': 'extracting',
            'percent': 0,
            'message': "Extracting frames from video..."
        })
    
    # Secondary timestamp source: showinfo parsing (in case ffprobe failed)
    showinfo_timestamps = []
    
    try:
        cmd = [
            "ffmpeg",
            "-skip_frame", "nokey", 
            "-hwaccel", "auto",
            "-i", video_path,
            "-vf", "scale='min(1920,iw)':-2,showinfo", 
            "-vsync", "vfr", 
            "-q:v", "2", 
            os.path.join(temp_extract_dir, "%05d.png")
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)
        for line in process.stderr:
            if "pts_time:" in line:
                match = re.search(r"pts_time:([0-9\.]+)", line)
                if match:
                    showinfo_timestamps.append(float(match.group(1)))
        
        process.wait()
        if process.returncode != 0:
            print(f"FFmpeg extraction code: {process.returncode}")
            
    except Exception as e:
        print(f"Error: FFmpeg extraction failed: {e}")

    # 2. Process extracted frames
    extracted_files = sorted(glob.glob(os.path.join(temp_extract_dir, "*.png")))
    num_extracted = len(extracted_files)
    
    if num_extracted == 0:
        print("Error: No frames were extracted.")
        return 0

    # Finalize timestamps list using the fallback chain
    final_timestamps = []
    if len(timestamps) == num_extracted:
        final_timestamps = timestamps
        print("Using Primary (ffprobe) timestamps.")
    elif len(showinfo_timestamps) == num_extracted:
        final_timestamps = showinfo_timestamps
        print("Using Secondary (showinfo) timestamps.")
    else:
        print(f"Warning: Master list mismatch ({len(timestamps)} vs {num_extracted}). Using distribution.")
        if duration > 0:
            for i in range(num_extracted):
                final_timestamps.append((i / max(1, num_extracted - 1)) * duration)
        else:
            final_timestamps = [i for i in range(num_extracted)] # Last resort: seconds = index
    
    print(f"Moving {num_extracted} images to output folder...")
    for i, file_path in enumerate(extracted_files):
        if progress_callback and i % 50 == 0:
            percent = int((i / num_extracted) * 100)
            progress_callback({
                'status': 'analyzing',
                'percent': percent,
                'message': f"Processing frame {i}/{num_extracted}"
            })
            
        frame_time = final_timestamps[i]
        filename = f"{i:03}_{round(frame_time, 2)}.png"
        dst = os.path.join(output_folder_screenshot_path, filename)
        shutil.move(file_path, dst)
            
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    
    elapsed_time = time.time() - start_time
    print(f'\n{num_extracted} frames extracted in {elapsed_time:.2f}s!')
    return num_extracted
