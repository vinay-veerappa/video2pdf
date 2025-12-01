import cv2
import imutils
import time
import os
import subprocess
import shutil
import glob
import sys
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
    
    try:
        # FFmpeg command: extract at frame_rate, resize to max 1920 width
        # -vf "fps=1,scale='min(1920,iw)':-1"
        # We use min(1920,iw) to avoid upscaling if video is smaller
        
        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda", # Explicitly use NVIDIA CUDA
            "-i", video_path,
            "-vf", f"fps={frame_rate},scale='min(1920,iw)':-2", 
            "-vsync", "0",
            "-q:v", "2", # High quality JPEG for temp (faster) or use PNG
            # Let's use PNG for accuracy as requested, but maybe compression level 1 for speed
            "-compression_level", "1",
            os.path.join(temp_extract_dir, "%05d.png")
        ]
        
        # Check if ffmpeg is in path, otherwise try to find it or fail gracefully
        # We assume it is since we checked earlier.
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("Warning: CUDA acceleration failed. Falling back to CPU extraction.")
            # Remove -hwaccel cuda and retry
            cmd = [c for c in cmd if c not in ["-hwaccel", "cuda"]]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: FFmpeg extraction failed: {e}")
        raise Exception("FFmpeg extraction failed. Please ensure FFmpeg is installed and working.")

    # 2. Process extracted frames (Simple Rename & Move)
    extracted_files = sorted(glob.glob(os.path.join(temp_extract_dir, "*.png")))
    print(f"Extracted {len(extracted_files)} frames. Moving to output folder...")
    
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
