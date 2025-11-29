import os
import glob
import re
import time
import base64
from utils import sanitize_filename, parse_image_timestamp
from transcript import clean_transcript_text

def create_markdown_report(images_folder, transcript_file, output_folder, video_name, embed_images=False):
    """Create a structured Markdown report for NotebookLM"""
    print("\nCreating Markdown report for NotebookLM...")
    
    try:
        # Parse transcript
        transcript_entries = []
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.+)', line)
                if match:
                    timestamp = match.group(1)
                    text = match.group(2)
                    transcript_entries.append((timestamp, text))
        
        # Get all images
        image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
        image_data = []
        for img_file in image_files:
            timestamp = parse_image_timestamp(os.path.basename(img_file))
            if timestamp:
                image_data.append((timestamp, img_file))
        
        if not image_data:
            print("No images found for Markdown report")
            return None
            
        # Create Markdown file
        safe_name = sanitize_filename(video_name)
        output_md = os.path.join(output_folder, f"{safe_name}_notebooklm.md")
        
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(f"# {video_name}\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d')}\n")
            f.write(f"**Source:** Video Presentation\n\n")
            f.write("---\n\n")
            
            image_idx = 0
            
            # Helper to convert timestamp to seconds
            def timestamp_to_seconds(ts):
                parts = ts.split(':')
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            
            while image_idx < len(image_data):
                img_timestamp, img_path = image_data[image_idx]
                img_filename = os.path.basename(img_path)
                
                # Determine time range for this slide
                current_seconds = timestamp_to_seconds(img_timestamp)
                
                # Find next slide timestamp to define the range
                if image_idx + 1 < len(image_data):
                    next_timestamp, _ = image_data[image_idx + 1]
                    next_seconds = timestamp_to_seconds(next_timestamp)
                else:
                    next_seconds = current_seconds + 300 # Assume last slide lasts 5 mins max
                
                # Collect transcript for this time range
                slide_transcript = []
                for ts, text in transcript_entries:
                    ts_seconds = timestamp_to_seconds(ts)
                    # Include text that starts after this slide appears, up until the next slide
                    if current_seconds <= ts_seconds < next_seconds:
                        slide_transcript.append(text)
                
                # Write Slide Section
                f.write(f"## Slide {image_idx + 1} ({img_timestamp})\n\n")
                
                if embed_images:
                    # Embed image as base64
                    try:
                        with open(img_path, "rb") as img_f:
                            encoded_string = base64.b64encode(img_f.read()).decode('utf-8')
                        f.write(f"![Slide {image_idx + 1}](data:image/png;base64,{encoded_string})\n\n")
                    except Exception as e:
                        print(f"Warning: Could not embed image {img_filename}: {e}")
                        rel_path = os.path.join("images", img_filename).replace("\\", "/")
                        f.write(f"![Slide {image_idx + 1}]({rel_path})\n\n")
                else:
                    # Note: NotebookLM might not render local images, but having the reference helps context
                    # We use a relative path
                    rel_path = os.path.join("images", img_filename).replace("\\", "/")
                    f.write(f"![Slide {image_idx + 1}]({rel_path})\n\n")
                
                # Write Transcript
                if slide_transcript:
                    # Pass list of lines directly to the new cleaning function
                    cleaned_text = clean_transcript_text(slide_transcript)
                    f.write("### Transcript\n\n")
                    f.write(f"{cleaned_text}\n\n")
                else:
                    f.write("*No speech detected for this slide.*\n\n")
                
                f.write("---\n\n")
                image_idx += 1
                
        print(f"Markdown report created: {output_md}")
        return output_md
        
    except Exception as e:
        print(f"Error creating Markdown report: {e}")
        import traceback
        traceback.print_exc()
        return None
