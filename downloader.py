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
        # Relaxed format selection: download best available and merge to mp4
        'format': 'bestvideo+bestaudio/best',
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
        else:
            # Fallback to yt-dlp native browser cookies
            # We try Chrome, then Edge, then Firefox.
            print("Warning: browser_cookie3 failed. Attempting to use yt-dlp native browser cookies...")
            # We can't easily loop inside the dict, so we'll just set it to 'chrome' + 'edge' + 'firefox' 
            # actually yt-dlp takes a string like "chrome" or "chrome+firefox" (no, it takes one).
            # But we can try to set it to 'chrome' by default. 
            # If the user gets the error, they should close the browser.
            # Let's try to be smarter: check if we can access the file? No.
            
            # Let's just default to 'chrome'. If it fails, the user sees the error.
            # But maybe we can try 'edge' if on Windows?
            ydl_opts['cookiesfrombrowser'] = ('chrome',) 
            # Note: The user must close the browser for this to work reliably.

    # Attempt 1: Preferred method (Cookies.txt or Chrome)
    try:
        print("Attempt 1: Downloading with preferred cookies...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return _process_download_result(ydl, info, ydl_opts)
            
    except Exception as e:
        print(f"Attempt 1 failed: {e}")
        
        # Check if it was a cookie lock issue
        if "cookie database" in str(e):
            print("Chrome cookie database locked. Waiting 5s...")
            import time
            time.sleep(5)
        
        # Attempt 2: Edge
        print("Attempt 2: Trying Edge cookies...")
        try:
            # Clear previous cookie settings
            ydl_opts.pop('cookiefile', None)
            ydl_opts['cookiesfrombrowser'] = ('edge',)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return _process_download_result(ydl, info, ydl_opts)
                
        except Exception as e2:
            print(f"Attempt 2 (Edge) failed: {e2}")
            
            # Attempt 3: No Cookies (Public)
            print("Attempt 3: Trying WITHOUT cookies...")
            try:
                ydl_opts.pop('cookiefile', None)
                ydl_opts.pop('cookiesfrombrowser', None)
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return _process_download_result(ydl, info, ydl_opts)
                    
            except Exception as e3:
                print(f"Attempt 3 (No Cookies) failed: {e3}")
                raise Exception(f"All download attempts failed. Last error: {e3}")

def _process_download_result(ydl, info, opts):
    filename = ydl.prepare_filename(info)
    if 'merge_output_format' in opts:
        base, _ = os.path.splitext(filename)
        filename = base + '.mp4'
    
    if not os.path.exists(filename):
         # Fallback search
         temp_dir = os.path.dirname(filename)
         files = glob.glob(os.path.join(temp_dir, "*.mp4"))
         if files: return files[0]
         raise Exception("Video file not found after download")
         
    print(f"Video downloaded successfully: {filename}")
    
    # Create metadata file
    metadata_path = os.path.join(os.path.dirname(filename), "metadata.txt")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(f"url: {info.get('webpage_url', '')}\n")
        f.write(f"title: {info.get('title', 'Unknown')}\n")
        
    return filename


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
