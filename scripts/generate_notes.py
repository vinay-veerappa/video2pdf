import os
import re
import glob
import google.generativeai as genai
from PIL import Image
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file or environment.")

MODEL_NAME = "gemini-2.5-flash"
INPUT_DIR = r"C:\Users\vinay\video2pdf\output\Bootcamp Classroom - Week 7 Day 1 - 9_30 trade"
OUTPUT_FILE = "generated_notes.md"

def setup_gemini():
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(MODEL_NAME)

def parse_transcript(transcript_path):
    """
    Parses transcript file with format: [HH:MM:SS] Text
    Returns a list of dicts: {'time': seconds, 'text': text}
    """
    entries = []
    if not os.path.exists(transcript_path):
        print(f"Transcript not found: {transcript_path}")
        return entries

    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)', line)
            if match:
                h, m, s, text = match.groups()
                seconds = int(h) * 3600 + int(m) * 60 + int(s)
                entries.append({'time': seconds, 'text': text.strip()})
    return entries

def get_slides(images_dir):
    """
    Gets unique slides and their timestamps from filenames.
    Filename format: XXX_MM.SS.png (e.g., 000_0.0.png, 056_5.47.png)
    Returns list of dicts: {'path': path, 'time': seconds}
    """
    slides = []
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return slides

    # Look in the 'unique' folder first, as that's where curated slides are
    unique_dir = os.path.join(images_dir, "unique")
    if os.path.exists(unique_dir):
        search_dir = unique_dir
    else:
        search_dir = images_dir

    files = glob.glob(os.path.join(search_dir, "*.png"))
    for f in files:
        filename = os.path.basename(f)
        # Match format XXX_MM.SS.png
        match = re.search(r'_(\d+\.\d+)\.png', filename)
        if match:
            minutes = float(match.group(1))
            seconds = int(minutes * 60)
            slides.append({'path': f, 'time': seconds})
    
    # Sort by time
    slides.sort(key=lambda x: x['time'])
    return slides

def correlate_content(slides, transcript_entries):
    """
    Correlates slides with transcript segments.
    Each slide gets the text from its timestamp until the next slide's timestamp.
    """
    correlated = []
    for i, slide in enumerate(slides):
        start_time = slide['time']
        end_time = slides[i+1]['time'] if i < len(slides) - 1 else float('inf')
        
        # Get text segments that fall within this slide's duration
        # We include text that starts at or after this slide, but before the next slide
        # For the first slide, we also include anything before it (intro)
        
        segment_text = []
        for entry in transcript_entries:
            if i == 0 and entry['time'] < start_time:
                 segment_text.append(entry['text'])
            elif start_time <= entry['time'] < end_time:
                segment_text.append(entry['text'])
        
        correlated.append({
            'slide_path': slide['path'],
            'timestamp': start_time,
            'text': " ".join(segment_text)
        })
    return correlated

def generate_notes(model, content_list, output_file):
    """
    Generates notes using Gemini API for each slide/text pair.
    Appends to output_file and skips already processed slides.
    """
    
    # Check for existing processed slides
    processed_images = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Find all [[IMAGE_PATH: ...]] tags
            matches = re.findall(r'\[\[IMAGE_PATH: (.*?)\]\]', content)
            processed_images.update(matches)
            
    print(f"Found {len(processed_images)} already processed slides.")
    
    # Filter out processed slides
    remaining_content = [item for item in content_list if item['slide_path'] not in processed_images]
    
    print(f"Processing {len(remaining_content)} remaining slides...")
    
    for i, item in enumerate(remaining_content):
        print(f"Processing slide {i+1}/{len(remaining_content)}...")
        
        image_path = item['slide_path']
        text_context = item['text']
        
        try:
            img = Image.open(image_path)
            
            prompt = f"""
            You are an expert note-taker. 
            Analyze this presentation slide and the accompanying transcript text.
            
            1. **Evaluate Image Relevance**: Is this image useful for the notes? (e.g., contains text, diagrams, charts vs. just a talking head or blank screen).
            2. **Create Notes**: Create clear, concise, and structured notes that capture the key concepts, definitions, and insights. If the image is relevant, explicitly reference its content.
            
            Transcript Context:
            {text_context}
            
            Output Format:
            ## [Slide Title/Topic]
            [[IMAGE_RELEVANCE: High/Low - Reason]]
            
            *   **Key Point 1**: Description
            *   **Key Point 2**: Description
            ...
            """
            
            response = model.generate_content([prompt, img])
            
            # Append to file immediately
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"[[IMAGE_PATH: {image_path}]]\n\n")
                f.write(response.text + "\n\n---\n\n")
            
            # Rate limiting safety
            time.sleep(2) 
            
        except Exception as e:
            print(f"Error processing slide {i+1}: {e}")
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n[Error processing slide {i+1}]\n\n")

def main():
    print("Starting note generation...")
    model = setup_gemini()
    
    transcript_path = os.path.join(INPUT_DIR, "transcripts", "transcript.txt")
    images_dir = os.path.join(INPUT_DIR, "images", "organized_moderate")
    
    print("Loading data...")
    transcript_entries = parse_transcript(transcript_path)
    slides = get_slides(images_dir)
    
    if not slides:
        print("No slides found!")
        return

    print(f"Found {len(slides)} slides and {len(transcript_entries)} transcript entries.")
    
    correlated_content = correlate_content(slides, transcript_entries)
    
    output_path = os.path.join(INPUT_DIR, OUTPUT_FILE)
    
    # Create file if it doesn't exist
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Video Notes\n\n")
            
    generate_notes(model, correlated_content, output_path)
        
    print(f"Notes generated/updated successfully: {output_path}")

if __name__ == "__main__":
    main()
