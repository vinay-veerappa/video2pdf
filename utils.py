import os
import re
import shutil
import tempfile
try:
    import browser_cookie3
except ImportError:
    browser_cookie3 = None

def is_youtube_url(url):
    """Check if the input is a YouTube URL"""
    if not url:
        return False
    youtube_patterns = [
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?(www\.)?youtu\.be/',
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def get_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'watch?v=' in url:
        return url.split('watch?v=')[1].split('&')[0].split('?')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0].split('&')[0]
    return None


def sanitize_filename(filename):
    """Sanitize filename for filesystem"""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove extra spaces
    filename = re.sub(r'\s+', ' ', filename).strip()
    # Limit length for Windows (max 255 chars for filename, but we'll use 100 to be safe)
    if len(filename) > 100:
        filename = filename[:100]
    filename = filename.rstrip('. ')
    return filename


def get_youtube_cookies():
    """Extract YouTube cookies from browser"""
    print("Attempting to extract cookies from browser...")
    if browser_cookie3 is None:
        print("Warning: browser_cookie3 module not installed. Skipping cookie extraction.")
        return None
        
    cookies_file = os.path.join(tempfile.gettempdir(), 'youtube_cookies.txt')
    
    try:
        # Try Chrome first, then Edge, then Firefox
        cj = None
        try:
            cj = browser_cookie3.chrome(domain_name='.youtube.com')
            print("Found cookies in Chrome")
        except:
            try:
                cj = browser_cookie3.edge(domain_name='.youtube.com')
                print("Found cookies in Edge")
            except:
                try:
                    cj = browser_cookie3.firefox(domain_name='.youtube.com')
                    print("Found cookies in Firefox")
                except:
                    pass
        
        if cj:
            # Write Netscape format cookies
            with open(cookies_file, 'w') as f:
                f.write("# Netscape HTTP Cookie File\n")
                for cookie in cj:
                    # Convert to Netscape format
                    # domain, flag, path, secure, expiration, name, value
                    domain = cookie.domain
                    flag = "TRUE" if domain.startswith('.') else "FALSE"
                    path = cookie.path
                    secure = "TRUE" if cookie.secure else "FALSE"
                    expiration = str(int(cookie.expires)) if cookie.expires else "0"
                    name = cookie.name
                    value = cookie.value
                    f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiration}\t{name}\t{value}\n")
            return cookies_file
    except Exception as e:
        print(f"Warning: Could not extract cookies: {e}")
    
    return None


def parse_image_timestamp(filename, return_minutes=False):
    """Parse timestamp from image filename (format: 042_34.16.png -> 34.16 minutes or HH:MM:SS)"""
    try:
        # Extract the timestamp part (e.g., "34.16" from "042_34.16.png")
        match = re.search(r'_(\d+\.\d+)\.png', filename)
        if match:
            minutes = float(match.group(1))
            if return_minutes:
                return minutes
            # Convert to seconds
            total_seconds = int(minutes * 60)
            # Convert to HH:MM:SS format
            hours = total_seconds // 3600
            mins = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
    except:
        pass
    return None


def initialize_output_folder(video_name, output_dir):
    """Create output folder structure"""
    # Sanitize video name for folder
    safe_name = sanitize_filename(video_name)
    if not safe_name:
        safe_name = "youtube_video"
    
    output_folder = os.path.join(output_dir, safe_name)
    
    # Create main output folder (create parent directories if needed)
    os.makedirs(output_folder, exist_ok=True)
    
    # Create images subfolder
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    print(f'Output folder initialized: {output_folder}')
    return output_folder, images_folder


def cleanup_temp_files(output_dir):
    """Clean up temporary files"""
    temp_dir = os.path.join(output_dir, "temp")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print("Temporary files cleaned up")
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")
