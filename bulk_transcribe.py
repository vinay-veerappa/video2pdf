
import os
import sys
import argparse
import glob
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transcript import download_youtube_transcript, transcribe_video_local
from downloader import get_playlist_videos
from utils import sanitize_filename

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.wmv', '.m4v'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
ALL_EXTENSIONS = VIDEO_EXTENSIONS.union(AUDIO_EXTENSIONS)

def process_youtube_playlist(url, output_dir, lang='en', prefer_auto=False):
    """Process all videos in a YouTube playlist."""
    print(f"\n=== Processing YouTube Playlist: {url} ===")
    
    videos, playlist_title = get_playlist_videos(url)
    
    if not videos:
        print("No videos found in playlist.")
        return

    print(f"Found {len(videos)} videos in playlist '{playlist_title}'")
    
    # Create playlist folder
    playlist_folder = os.path.join(output_dir, sanitize_filename(playlist_title))
    os.makedirs(playlist_folder, exist_ok=True)
    
    success_count = 0
    
    for i, video in enumerate(videos):
        print(f"\n--- Processing {i+1}/{len(videos)}: {video['title']} ---")
        try:
            # Use video title for filename
            filename = sanitize_filename(video['title'])
            
            # Download transcript
            # We pass output_folder as the playlist folder. 
            # download_youtube_transcript puts it in a 'transcripts' subfolder by default.
            # We might want to flatten this or respect the structure.
            # Let's let it use its default behavior but pointing to our playlist folder.
            
            result_file, txt_path = download_youtube_transcript(
                video['url'], 
                playlist_folder, 
                lang=lang, 
                prefer_auto=prefer_auto,
                output_filename=filename
            )
            
            if txt_path:
                success_count += 1
                
        except Exception as e:
            print(f"Error processing video {video['id']}: {e}")
            
    print(f"\nPlaylist processing complete. {success_count}/{len(videos)} transcripts downloaded.")
    print(f"Output directory: {playlist_folder}")


def process_local_directory(input_path, output_dir, model_size='base', recursive=True, lang=None):
    """Process local directory recursively."""
    input_path = Path(input_path)
    print(f"\n=== Processing Local Directory: {input_path} ===")
    
    # Find all media files
    media_files = []
    if recursive:
        for ext in ALL_EXTENSIONS:
            media_files.extend(input_path.rglob(f"*{ext}"))
            media_files.extend(input_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in ALL_EXTENSIONS:
            media_files.extend(input_path.glob(f"*{ext}"))
            media_files.extend(input_path.glob(f"*{ext.upper()}"))
            
    # Remove duplicates
    media_files = sorted(list(set(media_files)))
    
    # Exclude 2023 directory (already processed with different naming)
    media_files = [f for f in media_files if '2023' not in str(f.parent)]
    
    if not media_files:
        print("No media files found.")
        return

    print(f"Found {len(media_files)} media files.")
    
    success_count = 0
    
    for i, media_file in enumerate(media_files):
        print(f"\n--- Processing {i+1}/{len(media_files)}: {media_file.name} ---")
        
        try:
            # Determine output location
            # If we want to mirror structure:
            # relative_path = media_file.relative_to(input_path).parent
            # target_folder = os.path.join(output_dir, relative_path)
            
            # For now, let's keep it simple: put everything in one 'transcripts' folder in output_dir
            # OR create a structure. Mirroring is better for "bulk" operations on deep trees.
            
            rel_path = media_file.relative_to(input_path).parent
            target_folder = os.path.join(output_dir, rel_path)
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder, exist_ok=True)
            
            # Checks for existing transcript?
            # transcribe_video_local does check, but it appends 'transcripts' subfolder.
            # We want to precise control?
            # Actually, transcribe_video_local enforces 'transcripts' subfolder.
            # Let's use it as is, it's safer.
            
            sanitize_name = sanitize_filename(media_file.stem)
            
            res_path = transcribe_video_local(
                str(media_file),
                target_folder,
                method='whisper',
                model_size=model_size,
                output_filename=sanitize_name,
                language=lang
            )
            
            if res_path:
                success_count += 1
                
        except Exception as e:
            print(f"Error producing transcript for {media_file.name}: {e}")
            
    print(f"\nDirectory processing complete. {success_count}/{len(media_files)} files processed.")


def main():
    parser = argparse.ArgumentParser(description="Bulk Transcript Extractor (YouTube & Local)")
    
    parser.add_argument("input", help="YouTube Playlist URL or Local Directory Path")
    parser.add_argument("-o", "--output", default="bulk_transcripts", help="Output directory")
    parser.add_argument("--lang", default="en", help="Language code (e.g. 'en')")
    parser.add_argument("--auto", action="store_true", help="Prefer auto-generated subtitles (YouTube only)")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subdirectories (Local only)")
    
    args = parser.parse_args()
    
    # Check if input is URL
    if "youtube.com" in args.input or "youtu.be" in args.input:
        process_youtube_playlist(args.input, args.output, args.lang, args.auto)
    elif os.path.exists(args.input):
        if os.path.isdir(args.input):
            process_local_directory(args.input, args.output, args.model, not args.no_recursive, lang=args.lang)
        elif os.path.isfile(args.input):
             # Single file
             try:
                 print(f"Processing single file: {args.input}")
                 transcribe_video_local(
                    args.input, 
                    args.output,
                    method='whisper',
                    model_size=args.model,
                    output_filename=sanitize_filename(Path(args.input).stem),
                    language=args.lang
                 )
             except Exception as e:
                 print(f"Error: {e}")
    else:
        print("Error: Input must be a valid YouTube URL or local path.")
        sys.exit(1)

if __name__ == "__main__":
    main()
