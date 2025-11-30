import os
import sys
import glob
import argparse
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import inch
import re
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import parse_image_timestamp
from transcript import clean_transcript_text

def create_optimized_pdf_with_transcript(images_dir, transcript_file, output_pdf, jpeg_quality=75):
    """Create a compressed PDF with images and synchronized transcript"""
    print(f"Creating optimized PDF with transcript...")
    
    # Get all images (WebP or PNG)
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.webp")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    
    if not image_files:
        print("No images found")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Parse transcript
    transcript_entries = []
    if transcript_file and os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Extract timestamp and text: [00:00:01] text
                match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.+)', line)
                if match:
                    timestamp = match.group(1)
                    text = match.group(2)
                    transcript_entries.append((timestamp, text))
        print(f"Loaded {len(transcript_entries)} transcript entries")
    
    # Create PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    timestamp_style = ParagraphStyle(
        'Timestamp',
        parent=styles['Normal'],
        fontSize=9,
        textColor='#666666',
        spaceAfter=4,
        alignment=TA_LEFT
    )
    text_style = ParagraphStyle(
        'TranscriptText',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        spaceAfter=10,
        alignment=TA_LEFT
    )
    
    # Helper to convert timestamp to seconds
    def timestamp_to_seconds(ts):
        parts = ts.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    
    # Create temporary directory for JPEG conversion
    temp_dir = tempfile.mkdtemp()
    
    try:
        for i, img_path in enumerate(image_files):
            print(f"Processing {os.path.basename(img_path)} ({i+1}/{len(image_files)})...", end='\r')
            
            # Get timestamp from filename
            filename = os.path.basename(img_path)
            img_timestamp = parse_image_timestamp(filename)
            
            # Convert to JPEG for compression
            img = Image.open(img_path)
            # Convert to RGB (JPEG doesn't support transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img
            
            # Save as JPEG
            jpeg_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}.jpg")
            img.save(jpeg_path, "JPEG", quality=jpeg_quality, optimize=True)
            
            # Add image to PDF
            try:
                rl_img = RLImage(jpeg_path, width=6.5*inch, height=4.875*inch)
                story.append(rl_img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"\nWarning: Could not add image {filename}: {e}")
            
            # Find matching transcript
            if img_timestamp and transcript_entries:
                img_seconds = timestamp_to_seconds(img_timestamp)
                
                # Find next slide timestamp to define range
                if i + 1 < len(image_files):
                    next_filename = os.path.basename(image_files[i+1])
                    next_timestamp = parse_image_timestamp(next_filename)
                    if next_timestamp:
                        next_seconds = timestamp_to_seconds(next_timestamp)
                    else:
                        next_seconds = img_seconds + 300
                else:
                    next_seconds = img_seconds + 300
                
                # Collect transcript for this time range
                slide_transcript = []
                for ts, text in transcript_entries:
                    ts_seconds = timestamp_to_seconds(ts)
                    if img_seconds <= ts_seconds < next_seconds:
                        slide_transcript.append(text)
                
                if slide_transcript:
                    # Clean and format
                    cleaned_text = clean_transcript_text(slide_transcript)
                    if cleaned_text:
                        story.append(Paragraph(f"<b>Transcript ({img_timestamp}):</b>", timestamp_style))
                        story.append(Spacer(1, 0.05*inch))
                        # Include full transcript for document generation
                        story.append(Paragraph(cleaned_text.replace('\n', '<br/>'), text_style))
                        story.append(Spacer(1, 0.2*inch))
            
            story.append(PageBreak())
        
        print(f"\nBuilding PDF...")
        doc.build(story)
        
        # Get file sizes
        pdf_size = os.path.getsize(output_pdf)
        print(f"\nPDF created: {output_pdf}")
        print(f"PDF size: {pdf_size / 1024 / 1024:.2f} MB")
        
    finally:
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", help="Directory containing images")
    parser.add_argument("--transcript", help="Path to transcript file")
    parser.add_argument("--output", help="Output PDF path")
    parser.add_argument("--quality", type=int, default=75, help="JPEG quality (1-100, default 75)")
    args = parser.parse_args()
    
    if args.output:
        output_pdf = args.output
    else:
        output_pdf = os.path.join(os.path.dirname(args.images_dir), "slides_with_transcript.pdf")
    
    create_optimized_pdf_with_transcript(args.images_dir, args.transcript, output_pdf, args.quality)
