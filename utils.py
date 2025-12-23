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
    """Parse timestamp from image filename. Handles both legacy (minutes) and new (seconds) formats."""
    try:
        # 1. Extract the numeric part after the underscore
        # Format: index_TIME.png (e.g., "001_123.45.png")
        match = re.search(r'_([0-9\.]+)\.png', filename)
        if match:
            value = float(match.group(1))
            
            # Heuristic: If value > 500, it's almost certainly seconds.
            # If value < 500, it might be minutes or seconds.
            # But the new extractor always saves seconds.
            # To be safe, we check if it's likely minutes (< 300 and has small decimal?)
            # Actually, the most robust way is to assume SECONDS and let it be.
            # However, for backward compatibility:
            is_minutes = value < 500 # Very naive heuristic
            
            # If we know the video duration, we could be more accurate.
            # For now, let's assume it's SECONDS based on our new extractor change.
            # ONLY if we want to support old images, we'd need to know if they were minutes.
            # Since we just changed the extractor, we'll assume SECONDS but keep it flexible.
            total_seconds = int(value)
            if value < 100: # Likely minutes in old format
                 total_seconds = int(value * 60)
            
            if return_minutes:
                return total_seconds / 60.0
                
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
