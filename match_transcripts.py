import os
import sys
import glob
import json
import logging
from pathlib import Path
from rapidfuzz import process, fuzz
from downloader import get_playlist_videos
from transcript import download_youtube_transcript
from utils import sanitize_filename

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def match_and_download(local_dir, channel_url, threshold=70):
    """
    1. Scan local directory for videos without transcripts.
    2. Fetch all videos from the YouTube channel.
    3. Fuzzy match local filenames to YouTube titles.
    4. Download transcripts for matches.
    """
    
    # 1. Scan Local Directory
    logger.info(f"Scanning local directory: {local_dir}")
    local_videos = []
    # Using the same extensions as bulk_transcribe.py
    VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.wmv', '.m4v'}
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if Path(file).suffix.lower() in VIDEO_EXTENSIONS:
                video_path = Path(root) / file
                # Check if transcript already exists
                # Assuming transcript is in 'transcripts' subfolder or same folder
                # Logic from bulk_transcribe: target_folder/transcripts/filename.txt
                
                # Check standard location
                transcript_path = Path(root) / "transcripts" / f"{sanitize_filename(video_path.stem)}.txt"
                
                if not transcript_path.exists():
                     local_videos.append({
                         'path': video_path,
                         'name': video_path.stem,
                         'clean_name': _clean_name(video_path.stem)
                     })

    if not local_videos:
        logger.info("No local videos found missing transcripts.")
        return

    logger.info(f"Found {len(local_videos)} local videos missing transcripts.")

    # 2. Fetch YouTube Channel Videos
    logger.info(f"Fetching video list from channel: {channel_url}")
    try:
        yt_videos, channel_title = get_playlist_videos(channel_url)
    except Exception as e:
        logger.error(f"Failed to fetch channel videos: {e}")
        return

    logger.info(f"Found {len(yt_videos)} videos on channel '{channel_title}'.")
    
    # Prepare YouTube titles for matching
    yt_titles = {v['title']: v for v in yt_videos}
    yt_clean_titles = {_clean_name(t): t for t in yt_titles.keys()}
    
    # 3. Fuzzy Match and Download
    matches_found = 0
    
    # We use rapidfuzz to find the best match for each local video
    # optimization: build a list of choices
    choices = list(yt_clean_titles.keys())
    
    for local_vid in local_videos:
        # Find best match
        # process.extractOne returns (match, score, index)
        result = process.extractOne(local_vid['clean_name'], choices, scorer=fuzz.token_sort_ratio)
        
        if result:
            match_clean_title, score, _ = result
            
            if score >= threshold:
                original_yt_title = yt_clean_titles[match_clean_title]
                yt_video_data = yt_titles[original_yt_title]
                
                logger.info(f"MATCH FOUND ({score:.1f}%):")
                logger.info(f"  Local:   {local_vid['name']}")
                logger.info(f"  YouTube: {original_yt_title}")
                
                # 4. Download Transcript
                try:
                    # Determine output folder (same structure as bulk_transcribe)
                    output_folder = local_vid['path'].parent
                    filename = sanitize_filename(local_vid['path'].stem) # Use local name for the file
                    
                    logger.info(f"  Downloading transcript to {output_folder}...")
                    
                    cookies_path = None
                    if os.path.exists("cookies.txt"):
                        cookies_path = "cookies.txt"
                        
                    res, txt_path = download_youtube_transcript(
                        yt_video_data['url'],
                        str(output_folder),
                        output_filename=filename,
                        cookies_path=cookies_path
                    )
                    
                    if txt_path:
                        logger.info(f"  SUCCESS: Saved to {txt_path}\n")
                        matches_found += 1
                        
                except Exception as e:
                     logger.error(f"  FAILED to download: {e}\n")
            else:
                logger.info(f"  No match for: {local_vid['name']} (Best: {match_clean_title} @ {score:.1f}%)")
                pass
                
    logger.info(f"Processing complete. Downloaded {matches_found} new transcripts.")

def _clean_name(name):
    """Normalize name for better matching: lowercase, remove dates/extensions if possible."""
    # Simple normalization
    import re
    name = name.lower()
    # Remove file extensions just in case
    name = os.path.splitext(name)[0]
    # Replace separators with spaces
    name = re.sub(r'[_\-\.]', ' ', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Match local videos to YouTube channel and download transcripts.")
    parser.add_argument("local_dir", help="Path to local video directory")
    parser.add_argument("channel_url", help="YouTube Channel or Playlist URL")
    parser.add_argument("--threshold", type=float, default=70, help="Fuzzy match threshold (0-100)")
    
    args = parser.parse_args()
    
    match_and_download(args.local_dir, args.channel_url, args.threshold)
