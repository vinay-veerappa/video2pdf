import os
import glob
import img2pdf
import re
from utils import sanitize_filename, parse_image_timestamp
from transcript import clean_transcript_text

def convert_screenshots_to_pdf(images_folder, output_folder, video_name):
    """Convert screenshots to PDF"""
    safe_name = sanitize_filename(video_name)
    output_pdf_path = os.path.join(output_folder, f"{safe_name}.pdf")
    
    # Get all PNG files sorted
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    
    if not image_files:
        print("No images found to convert to PDF")
        return None
    
    print(f'Converting {len(image_files)} images to PDF...')
    
    try:
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_files))
        print(f'PDF created successfully: {output_pdf_path}')
        return output_pdf_path
    except Exception as e:
        print(f'Error creating PDF: {e}')
        raise


def sync_images_with_transcript(images_folder, transcript_file, output_folder):
    """Sync images with transcript and create a combined document"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        
        print("\nSyncing images with transcript and creating combined document...")
        
        # Parse transcript
        transcript_entries = []
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
        
        # Get all images with their timestamps
        image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
        image_data = []
        for img_file in image_files:
            timestamp = parse_image_timestamp(os.path.basename(img_file))
            if timestamp:
                image_data.append((timestamp, img_file))
        
        if not image_data:
            print("No images found to sync")
            return None
        
        # Create PDF
        output_pdf = os.path.join(output_folder, "combined_slides_with_transcript.pdf")
        doc = SimpleDocTemplate(output_pdf, pagesize=letter)
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='#333333',
            spaceAfter=12,
            alignment=TA_CENTER
        )
        timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#666666',
            spaceAfter=6,
            alignment=TA_LEFT
        )
        text_style = ParagraphStyle(
            'TranscriptText',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=12,
            alignment=TA_LEFT
        )
        
        # Match images with transcript
        image_idx = 0
        transcript_idx = 0
        
        while image_idx < len(image_data):
            img_timestamp, img_path = image_data[image_idx]
            
            # Find transcript entries that match this image's timestamp
            # Look for transcript entries within 30 seconds of the image
            matching_transcript = []
            
            # Convert timestamps to seconds for comparison
            def timestamp_to_seconds(ts):
                parts = ts.split(':')
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            
            img_seconds = timestamp_to_seconds(img_timestamp)
            
            # Find transcript entries near this timestamp
            for ts, text in transcript_entries:
                ts_seconds = timestamp_to_seconds(ts)
                if abs(ts_seconds - img_seconds) <= 30:  # Within 30 seconds
                    matching_transcript.append((ts, text))
            
            # Add image
            try:
                img = Image(img_path, width=6*inch, height=4.5*inch)
                img.drawHeight = 4.5*inch
                img.drawWidth = 6*inch
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"Warning: Could not add image {img_path}: {e}")
            
            # Add matching transcript text
            if matching_transcript:
                # Clean and format transcript
                transcript_texts = [text for _, text in matching_transcript]
                combined_text = ' '.join(transcript_texts)
                cleaned_text = clean_transcript_text(combined_text)
                
                if cleaned_text:
                    story.append(Paragraph(f"<b>Transcript ({img_timestamp}):</b>", timestamp_style))
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph(cleaned_text.replace('\n', '<br/>'), text_style))
                    story.append(Spacer(1, 0.3*inch))
            
            story.append(PageBreak())
            image_idx += 1
        
        # Build PDF
        doc.build(story)
        print(f"Combined PDF created: {output_pdf}")
        return output_pdf
        
    except ImportError as e:
        print(f"Error: Missing required library. Please install: pip install reportlab pyspellchecker")
        print(f"Error details: {e}")
        return None
    except Exception as e:
        print(f"Error creating combined document: {e}")
        import traceback
        traceback.print_exc()
        return None
