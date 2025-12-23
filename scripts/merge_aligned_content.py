
import os
import sys
import re
import math
import glob
import json
import argparse
from datetime import datetime

# Add parent directory to path to allow imports from scripts.ollama_utils if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.ollama_utils import OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    print("Warning: scripts.ollama_utils not found. LLM generation functions will fail.")
    OLLAMA_AVAILABLE = False

# =============================================================================
# 1. LLM GENERATION (Refined Prompt)
# =============================================================================

def generate_structured_notes(input_file, output_file, model="gemma3"):
    """
    Generates structured notes from a raw transcript using a refined prompt 
    that enforces specific topic headers and strict data preservation.
    """
    if not OLLAMA_AVAILABLE:
        print("Error: Ollama client not available.")
        return False

    client = OllamaClient()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Generating structured notes using {model}...")
        
        # Chunking (simplified)
        chunk_size = 100000 
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        cleaned_content = []

        prompt_template = """
Role: You are an expert Technical Editor and Market Analyst. 
Goal: Lossless Condensation of a trading transcript into a high-fidelity descriptive study guide.

STRUCTURE REQUIREMENTS:
Group the content into the following specific sections based on the speaker's topic. 
Use EXACTLY these markdown headers (##):

## HTF Context
(Monthly, Weekly, Daily charts analysis)

## Session Probabilities
(Asia, London, Pre-market analysis)

## Live Execution
(Trade setups, live price action 09:30 onwards)

## Candle Science
(Projections for next day, technical concepts)

STRICT INSTRUCTIONS:
1. Group consecutive lines into logical paragraphs.
2. Start each paragraph with the [HH:MM:SS] timestamp of the FIRST line in that group.
3. Insert Topic Headers (## Topic) when the discussion shifts to a new phase.
   Allowed Topics: 
   - ## HTF Context  (Higher Timeframe analysis)
   - ## Session Probabilities (Pre-market/Overnight stats)
   - ## Live Execution (Price action, trade management)
   - ## Candle Science (Closing thoughts, next day projections)
4. DO NOT summarize. Keep the original text mostly verbatim, just grouped.
5. PRESERVE all prices, numbers, and "If/Then" logic.

EXAMPLE INPUT:
[00:15:01] Okay let's look at the open.
[00:15:05] We are opening at 4500.
[00:15:08] If we drop below, I'm looking for 4490.

EXAMPLE OUTPUT:
## Live Execution

[00:15:01] Okay let's look at the open. We are opening at 4500. If we drop below, I'm looking for 4490.

INPUT TEXT:
{chunk}

OUTPUT ONLY THE MARKDOWN CONTENT.
"""
        
        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
            prompt = prompt_template.format(chunk=chunk)
            res = client.generate(model, prompt)
            response_text = res.get("response", "").strip()
            cleaned_content.append(response_text)
            
        final_text = "\n\n".join(cleaned_content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)
            
        print(f"Structured notes saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Error generating notes: {e}")
        return False

# =============================================================================
# 2. PARSING & MERGING
# =============================================================================

def parse_transcript_timestamps(file_path):
    """
    Parses a transcript file (raw or cleaned) to extract blocks with timestamps.
    Returns: List of {'timestamp': 'HH:MM:SS', 'seconds': int, 'text': str, 'section': str}
    """
    data = []
    current_section = "General"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for Section Headers
        if line.startswith('## '):
            current_section = line.strip('# ').strip()
            continue
            
        # Check for [HH:MM:SS]
        match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
        if match:
            ts_str = match.group(1)
            h, m, s = map(int, ts_str.split(':'))
            seconds = h * 3600 + m * 60 + s
            
            # Remove timestamp from text for cleaner display
            text_content = line.replace(f"[{ts_str}]", "").strip()
            
            data.append({
                'timestamp': ts_str,
                'seconds': seconds,
                'text': text_content,
                'section': current_section
            })
            
    return data

def load_images(video_id, base_dir):
    """
    Loads images from the unique folder and parses their timestamps.
    Expects format: XXX_seconds.png or XXX_MM.SS.png
    """
    images_dir = os.path.join(base_dir, video_id, "images", "organized_moderate", "unique")
    if not os.path.exists(images_dir):
        # Fallback to main images
        images_dir = os.path.join(base_dir, video_id, "images")
        
    image_list = []
    if not os.path.exists(images_dir):
        print(f"Warning: No images found at {images_dir}")
        return image_list

    for img_path in glob.glob(os.path.join(images_dir, "*.png")):
        filename = os.path.basename(img_path)
        try:
            # Parse timestamp. Common formats:
            # 1. frame_123.45.png
            # 2. 001_120.5.png
            parts = filename.rsplit('.', 1)[0] # remove extension
            
            # Try getting the last part after underscore
            if '_' in parts:
                ts_part = parts.split('_')[-1]
            else:
                ts_part = parts
                
            # Filenames are in Decimal Minutes (e.g. 48.84 = 48m 50s)
            # Script expects Seconds.
            seconds = float(ts_part) * 60.0
            
            image_list.append({
                'path': img_path,
                'filename': filename,
                'seconds': seconds
            })
        except ValueError:
            pass
            
    # Sort by timestamp
    image_list.sort(key=lambda x: x['seconds'])
    return image_list

def select_image(target_seconds, images, window=2.0, force_nearest=False):
    """
    Finds the active image for the given timestamp.
    Default logic: Find the most recent image that appeared BEFORE (or slightly after) the target time.
    this effectively finds the 'slide currently on screen'.
    
    window: Tolerance for 'slightly after' (e.g. 2.0s). 
            Logic: img['seconds'] <= target_seconds + window.
    """
    if not images:
        return None
        
    if force_nearest:
        # Absolute closest match (ignoring direction)
        return min(images, key=lambda x: abs(x['seconds'] - target_seconds))

    # Active Slide Logic:
    # Filter for images that appeared at or before target_time (+ tolerance)
    candidates = [img for img in images if img['seconds'] <= target_seconds + window]
    
    if not candidates:
        # If no image appeared yet, maybe return the very first one if we are close?
        # Or return None.
        # If we are at 00:00:05 and first image is 00:00:10, candidates empty.
        # Maybe check if the first image is close enough?
        first_img = images[0]
        if abs(first_img['seconds'] - target_seconds) <= 30.0: # If within 30s of start, use first image
             return first_img
        return None
        
    # Valid candidates found. Pick the one with the largest timestamp (most recent)
    # This represents the slide that was 'triggered' most recently.
    best_img = max(candidates, key=lambda x: x['seconds'])
    
    return best_img

def find_keyword_override(text):
    """
    Returns True if high-value keywords are present.
    """
    keywords = [
        r'basis points', 
        r'MAE', 
        r'MFE', 
        r'0-5 box', 
        r'line in the sand', 
        r'reversal',
        r'if.*then', # Logic chain
        r'fair value gap',
        r'order block'
    ]
    
    for pat in keywords:
        if re.search(pat, text, re.IGNORECASE):
            return True
            
    return False

def format_text(text):
    """
    Bolds numbers, percentages, and timestamps.
    """
    # Bold percentages
    text = re.sub(r'(\d+(?:\.\d+)?%)', r'**\1**', text)
    # Bold prices (simple heuristic: digits with decimals or large ints?)
    # Let's bold specific patterns like 4500.25
    text = re.sub(r'(\b\d{4}\.\d{2}\b)', r'**\1**', text)
    # Bold simple integers if they look like levels (3 digits+)
    # text = re.sub(r'(\b\d{3,}\b)', r'**\1**', text) 
    return text

def assemble_content(video_id, structured_notes, images):
    """
    Combines structured notes with images using Time-Interval Grouping.
    1. Sorts all unique images.
    2. Creates time buckets based on image timestamps.
    3. Assigns text blocks to the bucket of the Active Slide.
    
    Returns: markdown string and HTML data structure.
    """
    output_md = []
    html_data = [] 
    
    # 1. Prepare Images (Timeline)
    # Ensure they are sorted
    timeline_images = sorted(images, key=lambda x: x['seconds'])
    
    # 2. Bucket Text
    # content_map: { image_index: [text_items] }
    # image_index -1 for text before first image
    content_map = {}
    for i in range(-1, len(timeline_images)):
        content_map[i] = []
        
    for note in structured_notes:
        ts = note['seconds']
        
        # Find which image interval this text belongs to
        # It belongs to the most recent image that appeared BEFORE or AT ts
        active_img_idx = -1
        
        # Optimization: Could be binary search, but list is small (<200)
        for i, img in enumerate(timeline_images):
            if img['seconds'] <= ts + 2.0: # 2s tolerance for "slightly before"
                active_img_idx = i
            else:
                break
        
        content_map[active_img_idx].append(note)

    # 3. Assemble Output
    # Iterate through images (and the pre-image block)
    
    current_section = None
    
    # Handle pre-image text (if any)
    if content_map[-1]:
        output_md.append("## Introduction / Context\n")
        html_data.append({'section': "Introduction", 'type': 'header'})
        
        for item in content_map[-1]:
            # Check if section changed (though usually intro)
            if item['section'] != current_section and item['section'] != "General":
                current_section = item['section']
                output_md.append(f"\n## {current_section}\n")
                html_data.append({'section': current_section, 'type': 'header'})
                
            formatted = format_text(item['text'])
            output_md.append(f"[{item['timestamp']}] {formatted}\n")
            html_data.append({
                'section': current_section,
                'timestamp': item['timestamp'],
                'text': formatted,
                'image': None,
                'is_key': False
            })
        output_md.append("\n---\n")

    # Handle Images -> Text
    for i, img in enumerate(timeline_images):
        text_blocks = content_map[i]
        
        # If no text associated with this image, skip it? 
        # User said "avoid repeated images" - implies showing unique images.
        # But if an image appears for 5 seconds and nothing is said, maybe we skip it?
        # OR we show it for visual completeness. 
        # Let's show it, unless it's a "duplicate" (which load_images should have handled, 
        # specifically "unique" folder).
        
        # Start Block
        img_rel_path = img['path']
        output_md.append(f"![{img['filename']}]({img_rel_path})\n")
        
        # Add all text blocks
        if not text_blocks:
            # Empty block
            output_md.append(f"*(Visual Context: {img['filename']})*\n\n")
            html_data.append({
                'section': current_section,
                'timestamp': f"{int(img['seconds'] // 3600):02d}:{int((img['seconds'] % 3600) // 60):02d}:{int(img['seconds'] % 60):02d}", 
                'text': "(Visual Transition)",
                'image': img_rel_path,
                'is_key': False
            })
        else:
            # We have text
            block_html_entry = {
                'section': current_section,
                'image': img_rel_path,
                'text_blocks': []
            }
            
            for item in text_blocks:
                # Update Section Header if needed
                if item['section'] != current_section:
                    current_section = item['section']
                    output_md.append(f"\n## {current_section}\n")
                    # For HTML, we might need a separate header entry or handle in rendering
                
                formatted = format_text(item['text'])
                output_md.append(f"[{item['timestamp']}] {formatted}\n")
                
                block_html_entry['text_blocks'].append({
                    'timestamp': item['timestamp'],
                    'text': formatted,
                    'section': item['section']
                })
            
            # For HTML generation simplicity, we'll flatten properly in generate_review_html
            # But here let's pass a structure that keeps them together
            html_data.append(block_html_entry)
            
            output_md.append("\n") # Spacing after block

    return "\n".join(output_md), html_data

# =============================================================================
# 3. OUTPUT GENERATION
# =============================================================================

def generate_review_html(video_id, html_data, output_path):
    """
    Generates a standalone HTML file for review.
    Handles both header entries and image+text blocks.
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Summary Review: {video_id}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
            .container {{ display: flex; flex-direction: column; gap: 30px; }}
            .block {{ display: flex; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); align-items: flex-start; }}
            .image-col {{ flex: 1.2; max-width: 60%; padding-right: 30px; position: sticky; top: 20px; }}
            .img-wrapper {{ width: 100%; border-radius: 4px; overflow: hidden; border: 1px solid #eee; }}
            .image-col img {{ width: 100%; display: block; }}
            .text-col {{ flex: 0.8; display: flex; flex-direction: column; gap: 15px; }}
            .text-item {{ padding: 10px; border-left: 3px solid #eee; }}
            .text-item:hover {{ border-left-color: #3498db; background: #fafafa; }}
            .timestamp {{ color: #e74c3c; font-family: monospace; font-weight: bold; font-size: 0.9em; }}
            .section-header {{ background: #2c3e50; color: white; padding: 15px; margin-top: 40px; border-radius: 6px; }}
            .section-label {{ font-size: 0.8em; text-transform: uppercase; color: #95a5a6; margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <h1>Content Review: {video_id}</h1>
        <div class="container">
    """
    
    last_section = None
    
    for item in html_data:
        # Check for Header Entry
        if 'type' in item and item['type'] == 'header':
             html += f"<div class='section-header'><h2>{item['section']}</h2></div>"
             last_section = item['section']
             continue
             
        # Normal Block (Image + Text List)
        html += "<div class='block'>"
        
        # Image Column
        html += "<div class='image-col'>"
        if item.get('image'):
            # Convert to file URI
            src = f"file:///{item['image'].replace(os.path.sep, '/')}"
            html += f"<div class='img-wrapper'><img src='{src}'></div>"
            html += f"<div style='margin-top:5px; color:#888; font-size:12px;'>{os.path.basename(item['image'])}</div>"
        else:
             html += "<div style='padding:40px; text-align:center; background:#eee;'>No Image</div>"
        html += "</div>"
        
        # Text Column
        html += "<div class='text-col'>"
        
        # Handle single text entry (legacy/simple) vs text_blocks list
        blocks = item.get('text_blocks', [])
        if not blocks and 'text' in item:
             # Convert single entry to list
             blocks = [{'timestamp': item['timestamp'], 'text': item['text'], 'section': item.get('section')}]
             
        for txt in blocks:
            # Check for section change inside a block (edge case)
            if txt.get('section') and txt['section'] != last_section:
                html += f"<div class='section-label'>Section: {txt['section']}</div>"
                last_section = txt['section']
                
            display_text = txt['text'].replace('**', '<b>').replace('**', '</b>')
            html += f"<div class='text-item'>"
            html += f"<span class='timestamp'>[{txt['timestamp']}]</span>"
            html += f"<p>{display_text}</p>"
            html += "</div>"

        html += "</div>" # End text-col
        html += "</div>" # End block
        
    html += "</div></body></html>"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Review HTML generated: {output_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Merge transcript and images intelligently.")
    parser.add_argument("video_id", help="Video ID or folder name in output directory")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM generation, use existing cleaned file")
    
    args = parser.parse_args()
    
    # Paths
    base_dir = r"C:\Users\vinay\video2pdf\output"
    video_dir = os.path.join(base_dir, args.video_id)
    
    if not os.path.exists(video_dir):
        print(f"Error: Directory not found: {video_dir}")
        return
        
    # Paths
    base_dir = r"C:\Users\vinay\video2pdf\output"
    video_dir = os.path.join(base_dir, args.video_id)
    
    if not os.path.exists(video_dir):
        print(f"Error: Directory not found: {video_dir}")
        return
        
    transcript_path = os.path.join(video_dir, "transcripts", f"{args.video_id}.txt")
    if not os.path.exists(transcript_path):
        # Fallback names search
        possible_names = [
            "transcript.txt",
            "transcript_whisper.txt",
            "transcript_gemini.txt",
            "transcript_cleaned.txt",
            "transcript_backup.txt"
        ]
        found = False
        for name in possible_names:
            p = os.path.join(video_dir, "transcripts", name)
            if os.path.exists(p):
                transcript_path = p
                found = True
                print(f"Found transcript: {name}")
                break
        
        if not found:
             # Try any txt file in transcripts
             txts = glob.glob(os.path.join(video_dir, "transcripts", "*.txt"))
             if txts:
                 transcript_path = txts[0]
                 print(f"Found transcript (fallback): {os.path.basename(transcript_path)}")
             else:
                 print("Error: No transcript found to process.")
                 return
        
    structured_notes_path = os.path.join(video_dir, "transcripts", f"{args.video_id}_structured.md")
    
    # 1. Run LLM logic
    if not args.skip_llm:
        if not os.path.exists(transcript_path):
            print("Error: No transcript found to process.")
            return
        success = generate_structured_notes(transcript_path, structured_notes_path)
        if not success:
            print("Aborting due to LLM failure.")
            return
    else:
        if not os.path.exists(structured_notes_path):
            print(f"Error: {structured_notes_path} not found. Capture transcript first or remove --skip-llm")
            return
            
    # 2. Parse Notes and Images
    print("Parsing structured notes...")
    notes_data = parse_transcript_timestamps(structured_notes_path)
    print(f"Loaded {len(notes_data)} content blocks.")
    
    print("Loading images...")
    images_data = load_images(args.video_id, base_dir)
    print(f"Loaded {len(images_data)} images.")
    
    # 3. Assemble
    print("Assembling content...")
    md_output, html_data = assemble_content(args.video_id, notes_data, images_data)
    
    # 4. Save Outputs
    output_md_path = os.path.join(video_dir, f"{args.video_id}_smart_summary.md")
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(md_output)
    print(f"Markdown summary saved: {output_md_path}")
    
    output_html_path = os.path.join(video_dir, f"{args.video_id}_review.html")
    generate_review_html(args.video_id, html_data, output_html_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
