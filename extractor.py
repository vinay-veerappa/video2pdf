import cv2
import imutils
import time
import os
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
                              use_similarity=True, similarity_threshold=0.95,
                              min_time_interval=MIN_TIME_BETWEEN_CAPTURES,
                              save_duplicates_path=None,
                              similarity_method='ssim'):
    """Detect and save unique screenshots from video with improved duplicate detection"""
    # Calculate derived parameters
    warmup = frame_rate
    fgbg_history = frame_rate * 15
    
    # Initialize background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=fgbg_history, 
        varThreshold=VAR_THRESHOLD,
        detectShadows=DETECT_SHADOWS
    )

    captured = False
    start_time = time.time()
    (W, H) = (None, None)
    screenshots_count = 0
    previous_frames = []  # Store last N frames for comparison
    previous_hashes = []  # Store hashes of last N frames
    last_capture_time = None  # Track time of last capture

    for frame_count, frame_time, frame in get_frames(video_path, frame_rate):
        orig = frame.copy()
        frame_resized = imutils.resize(frame, width=600)
        mask = fgbg.apply(frame_resized)

        if W is None or H is None:
            (H, W) = mask.shape[:2]

        # Compute percentage of foreground
        p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100

        # Motion detection logic
        if p_diff < min_percent and not captured and frame_count > warmup:
            # Check minimum time interval (if enabled)
            if min_time_interval > 0 and last_capture_time is not None:
                time_since_last = (frame_time - last_capture_time) * 60  # Convert to seconds
                if time_since_last < min_time_interval:
                    continue  # Skip if too soon after last capture
            
            # Additional similarity check to avoid duplicates
            should_save = True
            
            # 1. Fast check using dHash
            current_hash = dhash(orig)
            if use_similarity and previous_hashes:
                for prev_hash in previous_hashes:
                    dist = calculate_hamming_distance(current_hash, prev_hash)
                    if dist <= 5:  # Threshold for "very similar" (0-5 bits different out of 64)
                        should_save = False
                        # print(f"Frame {frame_count} skipped (dHash distance: {dist})")
                        break
            
            # 2. Detailed check using SSIM (only if dHash didn't flag it)
            if should_save and use_similarity and previous_frames:
                # Compare with last N frames
                max_similarity = 0
                for prev_frame in previous_frames:
                    similarity = calculate_similarity(orig, prev_frame, method=similarity_method)
                    if similarity and similarity > max_similarity:
                        max_similarity = similarity
                
                if max_similarity > similarity_threshold:
                    should_save = False
                    print(f"Frame {frame_count} skipped (similarity: {max_similarity:.3f}, "
                          f"time: {frame_time/60:.2f} min)")
                    
                    # Save duplicate if requested
                    if save_duplicates_path:
                        filename = f"{screenshots_count:03}_{round(frame_time/60, 2)}_DUPLICATE.png"
                        path = os.path.join(save_duplicates_path, filename)
                        cv2.imwrite(path, orig)

            if should_save:
                captured = True
                filename = f"{screenshots_count:03}_{round(frame_time/60, 2)}.png"
                path = os.path.join(output_folder_screenshot_path, filename)
                print(f"Saving {path} (frame {frame_count}, time: {frame_time/60:.2f} min)")
                cv2.imwrite(path, orig)
                
                # Update tracking
                last_capture_time = frame_time
                previous_frames.append(orig.copy())
                previous_hashes.append(current_hash)
                
                # Keep only last N frames for comparison
                if len(previous_frames) > MAX_SIMILARITY_COMPARISONS:
                    previous_frames.pop(0)
                    previous_hashes.pop(0)
                
                screenshots_count += 1

        elif captured and p_diff >= max_percent:
            captured = False

    elapsed_time = time.time() - start_time
    print(f'\n{screenshots_count} unique screenshots captured!')
    print(f'Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')
    return screenshots_count
