import os
import sys
import subprocess
import glob
from utils import get_youtube_cookies

def download_youtube_video(url, output_dir, cookies_path=None):
    """Download YouTube video using yt-dlp"""
    print(f"Downloading video from YouTube: {url}")
    
    # Check if yt-dlp is available
    try:
        subprocess.run([sys.executable, "-m", "yt_dlp", "--version"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise Exception("yt-dlp is not installed. Please install it with: pip install yt-dlp")
    
    # Create temp directory for downloads
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Use yt-dlp to download video
        # Download best quality video in mp4 format
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "-o", os.path.join(temp_dir, "%(title)s.%(ext)s"),
        ]
        
        # Add cookies if available
        if cookies_path and os.path.exists(cookies_path):
            print(f"Using cookies from: {cookies_path}")
            cmd.extend(["--cookies", cookies_path])
        else:
            # Try automatic extraction if no specific file provided
            auto_cookies = get_youtube_cookies()
            if auto_cookies:
                print(f"Using extracted browser cookies: {auto_cookies}")
                cmd.extend(["--cookies", auto_cookies])
            
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find the downloaded file (prefer mp4, but accept any video format)
        video_extensions = ['*.mp4', '*.mkv', '*.webm', '*.avi', '*.mov']
        downloaded_files = []
        for ext in video_extensions:
            downloaded_files.extend(glob.glob(os.path.join(temp_dir, ext)))
        
        if not downloaded_files:
            raise Exception("Video file not found after download")
        
        video_path = downloaded_files[0]
        print(f"Video downloaded successfully: {video_path}")
        
        # Create metadata file
        try:
            # We need to know where the final folder will be. 
            # This function returns a temp path, but the caller (main.py) moves/uses it.
            # However, we can save metadata in the same temp dir for now, or return it.
            # Better approach: main.py calls initialize_output_folder which creates the final folder.
            # Let's write metadata to the temp dir, and let main.py or utils.py handle moving it?
            # Or better yet, just write it here if we can.
            
            # Actually, let's write it to the output_dir/video_title/metadata.txt
            # But we don't know the video title folder name yet easily here without re-sanitizing.
            # Let's write it alongside the video file in the temp dir, and update main.py to move it.
            
            metadata_path = os.path.join(os.path.dirname(video_path), "metadata.txt")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"url: {url}\n")
                # We can also add title if we want
                # title = os.path.splitext(os.path.basename(video_path))[0]
                # f.write(f"title: {title}\n")
            print(f"Metadata file created: {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not create metadata file: {e}")
            
        return video_path
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout
        print(f"Error downloading video: {error_msg}")
        raise Exception(f"Failed to download video. Make sure the URL is valid and accessible.")
    except Exception as e:
        print(f"Error: {e}")
        raise


def get_video_title(url):
    """Extract video title from YouTube URL"""
    try:
        # Extract video ID from URL to get single video
        video_id = None
        if 'watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0].split('?')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0].split('&')[0]
        
        # Use video ID directly if we can extract it
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--get-title",
            "--no-playlist",
            "--no-warnings",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        title = result.stdout.strip().split('\n')[0]  # Get only first line
        return title if title else "youtube_video"
    except Exception as e:
        print(f"Warning: Could not get video title: {e}")
        return "youtube_video"
