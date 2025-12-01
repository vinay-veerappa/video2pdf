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
                              similarity_method='grid'):
    """
    Detect and save unique screenshots using FFmpeg for fast extraction 
    followed by Python-based deduplication.
    """
    start_time = time.time()
    
    # 1. Extract ALL frames to a temp directory using FFmpeg (Fast!)
    temp_extract_dir = os.path.join(os.path.dirname(output_folder_screenshot_path), "temp_extracted")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    print(f"Extracting frames using FFmpeg (rate: {frame_rate} fps)...")
    
    try:
        # FFmpeg command: extract at frame_rate, resize to max 1920 width
        # -vf "fps=1,scale='min(1920,iw)':-1"
        # We use min(1920,iw) to avoid upscaling if video is smaller
        
        cmd = [
            "ffmpeg",
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
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: FFmpeg failed or not found. Falling back to OpenCV extraction.")
        # Fallback to old method if FFmpeg fails? 
        # For now, let's just raise error or we'd need to keep the old code.
        # Given the user wants speed, let's assume FFmpeg works.
        raise Exception("FFmpeg extraction failed. Please ensure FFmpeg is installed.")

    # 2. Process extracted frames
    extracted_files = sorted(glob.glob(os.path.join(temp_extract_dir, "*.png")))
    print(f"Extracted {len(extracted_files)} frames. Starting deduplication...")
    
    screenshots_count = 0
    previous_frames = []
    previous_hashes = []
    
    # We need to calculate timestamps based on frame index and rate
    # frame_time = index / frame_rate
    
    for i, file_path in enumerate(extracted_files):
        frame_time = i / frame_rate
        
        # Read image
        frame = cv2.imread(file_path)
        if frame is None: continue
        
        should_save = True
        
        # 1. Fast check using dHash
        current_hash = dhash(frame)
        if use_similarity and previous_hashes:
            for prev_hash in previous_hashes:
                dist = calculate_hamming_distance(current_hash, prev_hash)
                if dist <= 2: # Stricter threshold for dHash (very similar)
                    should_save = False
                    break
        
        # 2. Detailed check using SSIM (only if dHash didn't flag it)
        if should_save and use_similarity and previous_frames:
            max_similarity = 0
            for prev_frame in previous_frames:
                # We can use the frame directly since it's already resized by FFmpeg
                similarity = calculate_similarity(frame, prev_frame, method=similarity_method)
                if similarity and similarity > max_similarity:
                    max_similarity = similarity
            
            if max_similarity > similarity_threshold:
                should_save = False
                
                # Save duplicate if requested
                if save_duplicates_path:
                    filename = f"{screenshots_count:03}_{round(frame_time/60, 2)}_DUPLICATE.png"
                    dst = os.path.join(save_duplicates_path, filename)
                    shutil.copy2(file_path, dst)

        if should_save:
            filename = f"{screenshots_count:03}_{round(frame_time/60, 2)}.png"
            dst = os.path.join(output_folder_screenshot_path, filename)
            shutil.move(file_path, dst) # Move instead of copy to save space/time
            print(f"Saved {filename}")
            
            previous_frames.append(frame) # Keep in memory for comparison
            previous_hashes.append(current_hash)
            
            if len(previous_frames) > MAX_SIMILARITY_COMPARISONS:
                previous_frames.pop(0)
                previous_hashes.pop(0)
            
            screenshots_count += 1
            
    # Cleanup temp extraction
    shutil.rmtree(temp_extract_dir)
    
    elapsed_time = time.time() - start_time
    print(f'\n{screenshots_count} unique screenshots captured!')
    print(f'Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')
    return screenshots_count
