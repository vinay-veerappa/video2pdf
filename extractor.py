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
    
    # Check if images already exist in output folder
    if os.path.exists(output_folder_screenshot_path):
        existing_images = glob.glob(os.path.join(output_folder_screenshot_path, "*.png"))
        if len(existing_images) > 0:
            print(f"Images already exist in {output_folder_screenshot_path}. Skipping extraction.")
            if progress_callback:
                progress_callback({
                    'status': 'finished',
                    'percent': 100,
                    'message': "Images already extracted."
                })
            return output_folder_screenshot_path

    # 1. Extract ALL frames to a temp directory using FFmpeg (Fast!)
    temp_extract_dir = os.path.join(os.path.dirname(output_folder_screenshot_path), "temp_extracted")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    print(f"Extracting frames using FFmpeg (rate: {frame_rate} fps)...")
    if progress_callback:
        progress_callback({
            'status': 'extracting',
            'percent': 0,
            'message': "Extracting frames from video..."
        })
    
    timestamps = []
    
    try:
        # FFmpeg command: Optimized I-Frame extraction using -skip_frame nokey
        # We add 'showinfo' filter to get exact timestamps of extracted frames
        
        cmd = [
            "ffmpeg",
            "-skip_frame", "nokey", # Skip non-keyframes (must be before -i)
            "-hwaccel", "cuda", # Explicitly use NVIDIA CUDA
            "-i", video_path,
            "-vf", "scale='min(1920,iw)':-2,showinfo", # Resize and show info
            "-vsync", "vfr", # Variable frame rate for I-frames
            "-q:v", "2", # High quality JPEG/PNG
            "-compression_level", "1", # Low compression for speed
            os.path.join(temp_extract_dir, "%05d.png")
        ]
        
        # Run and capture stderr for timestamps
        result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        # Parse timestamps
        # [Parsed_showinfo_0 @ ...] n:   0 pts:      0 pts_time:0       pos:      0 ...
        regex = re.compile(r"pts_time:([0-9\.]+)")
        for line in result.stderr.splitlines():
            if "pts_time:" in line and "showinfo" in line:
                match = regex.search(line)
                if match:
                    timestamps.append(float(match.group(1)))
                    
    except subprocess.CalledProcessError:
        print("Warning: CUDA acceleration failed. Falling back to CPU extraction.")
        # Remove -hwaccel cuda and retry
        cmd = [c for c in cmd if c not in ["-hwaccel", "cuda"]]
        result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        # Parse timestamps again
        regex = re.compile(r"pts_time:([0-9\.]+)")
        for line in result.stderr.splitlines():
            if "pts_time:" in line and "showinfo" in line:
                match = regex.search(line)
                if match:
                    timestamps.append(float(match.group(1)))

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: FFmpeg extraction failed: {e}")
        raise Exception("FFmpeg extraction failed. Please ensure FFmpeg is installed and working.")

    # 2. Process extracted frames (Simple Rename & Move)
    extracted_files = sorted(glob.glob(os.path.join(temp_extract_dir, "*.png")))
    print(f"Extracted {len(extracted_files)} frames. Moving to output folder...")
    
    # Verify timestamp count
    if len(timestamps) != len(extracted_files):
        print(f"Warning: Timestamp count ({len(timestamps)}) does not match file count ({len(extracted_files)}). Falling back to estimated timestamps.")
        timestamps = [] # Fallback
    
    screenshots_count = 0
    total_frames = len(extracted_files)
    
    for i, file_path in enumerate(extracted_files):
        # Update progress
        if progress_callback and i % 50 == 0:
            percent = int((i / total_frames) * 100)
            progress_callback({
                'status': 'analyzing',
                'percent': percent,
                'message': f"Processing frame {i}/{total_frames}"
            })
            
        # Calculate timestamp
        if timestamps:
            frame_time = timestamps[i]
        else:
            # Fallback (incorrect for I-frames but better than crashing)
            frame_time = i / frame_rate
        
        # Rename and move
        # We trust image_dedup.py to do the actual deduplication later
        filename = f"{i:03}_{round(frame_time/60, 2)}.png"
        dst = os.path.join(output_folder_screenshot_path, filename)
        shutil.move(file_path, dst)
        screenshots_count += 1
            
    # Cleanup temp extraction
    shutil.rmtree(temp_extract_dir)
    
    elapsed_time = time.time() - start_time
    print(f'\n{screenshots_count} frames extracted for analysis!')
    print(f'Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')
    return screenshots_count
