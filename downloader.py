import os
import sys
import subprocess
import glob
from utils import get_youtube_cookies

import yt_dlp

def download_youtube_video(url, output_dir, cookies_path=None, progress_callback=None):
    """
    Download YouTube video using yt-dlp library with progress tracking.
    progress_callback(data): function to receive progress updates.
    """
    print(f"Downloading video from YouTube: {url}")
    
    # Check if video already exists in the persistent 'video' subfolder
    video_subfolder = os.path.join(output_dir, 'video')
    if os.path.exists(video_subfolder):
        existing_files = [f for f in os.listdir(video_subfolder) if f.endswith(('.mp4', '.mkv', '.webm'))]
        if existing_files:
            print(f"Video already exists: {existing_files[0]}")
            if progress_callback:
                progress_callback({'status': 'downloaded', 'percent': 100, 'message': 'Video already downloaded.'})
            return os.path.join(video_subfolder, existing_files[0])
    
    # Create temp directory for downloads
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Progress hook
    def progress_hook(d):
        if d['status'] == 'downloading':
            if progress_callback:
                # Calculate percentage
                p = d.get('_percent_str', '0%').replace('%','')
                try:
                    percent = float(p)
                except:
                    percent = 0
                
                # Get ETA
                eta = d.get('_eta_str', 'Unknown')
                
                progress_callback({
                    'status': 'downloading',
                    'percent': percent,
                    'message': f"Downloading: {d.get('_percent_str')} (ETA: {eta})"
                })
        elif d['status'] == 'finished':
            if progress_callback:
                progress_callback({
                    'status': 'downloading',
                    'percent': 100,
                    'message': "Download complete. Processing..."
                })

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'progress_hooks': [progress_hook],
        'quiet': True,
        'no_warnings': True
    }

    # Add cookies
    if cookies_path and os.path.exists(cookies_path):
        print(f"Using cookies from: {cookies_path}")
        ydl_opts['cookiefile'] = cookies_path
    else:
        auto_cookies = get_youtube_cookies()
        if auto_cookies:
            print(f"Using extracted browser cookies: {auto_cookies}")
            ydl_opts['cookiefile'] = auto_cookies

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            
            # If merged, the filename might be different (mp4)
            if 'merge_output_format' in ydl_opts:
                base, _ = os.path.splitext(filename)
                filename = base + '.mp4'
            
            if not os.path.exists(filename):
                # Fallback search if filename logic fails
                files = glob.glob(os.path.join(temp_dir, "*.mp4"))
                if files:
                    filename = files[0]
                else:
                    raise Exception("Video file not found after download")

            print(f"Video downloaded successfully: {filename}")
            
            # Create metadata file
            metadata_path = os.path.join(os.path.dirname(filename), "metadata.txt")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(f"url: {url}\n")
                f.write(f"title: {info.get('title', 'Unknown')}\n")
                
            return filename
            
    except Exception as e:
        print(f"Error downloading video: {e}")
        raise Exception(f"Failed to download video: {str(e)}")


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
