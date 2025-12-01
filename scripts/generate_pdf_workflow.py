#!/usr/bin/env python3
"""
Complete PDF Generation Workflow
Processes existing extracted images to create optimized PDF with transcript
"""

import os
import sys
import argparse
import shutil
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_complete_workflow(video_folder, youtube_url=None, quality=70):
    """
    Run complete workflow from existing images to final PDF
    
    Folder structure:
    video_folder/
        images/           - Original extracted frames (input)
        images_optimized/ - Cropped and optimized images (temp)
        transcripts/      - All transcript files (temp)
        reports/          - Analysis reports (temp)
        final.pdf         - Final output PDF
    """
    
    print("="*70)
    print("COMPLETE PDF GENERATION WORKFLOW")
    print("="*70)
    
    # Validate input
    images_folder = os.path.join(video_folder, "images")
    if not os.path.exists(images_folder):
        print(f"Error: Images folder not found: {images_folder}")
        return
    
    image_files = glob.glob(os.path.join(images_folder, "*.png"))
    if not image_files:
        print(f"Error: No images found in {images_folder}")
        return
    
    print(f"\nFound {len(image_files)} images in {images_folder}")
    
    # Check for metadata file to get YouTube URL
    # Check for metadata file to get YouTube URL
    metadata_file = os.path.join(video_folder, "metadata.txt")
    if youtube_url is None and os.path.exists(metadata_file):
        print(f"\nReading metadata from {metadata_file}")
        
        encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp1252']
        metadata_content = None
        
        for encoding in encodings:
            try:
                with open(metadata_file, 'r', encoding=encoding) as f:
                    metadata_content = f.read()
                print(f"Successfully read metadata with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading metadata with {encoding}: {e}")
                continue
        
        if metadata_content:
            for line in metadata_content.splitlines():
                if line.startswith('url:'):
                    youtube_url = line.split('url:', 1)[1].strip()
                    print(f"Found YouTube URL: {youtube_url}")
                    break
        else:
            print("Failed to read metadata file with any encoding.")

    
    # Create organized folder structure
    images_optimized = os.path.join(video_folder, "images_optimized")
    transcripts_folder = os.path.join(video_folder, "transcripts")
    reports_folder = os.path.join(video_folder, "reports")
    
    os.makedirs(images_optimized, exist_ok=True)
    os.makedirs(transcripts_folder, exist_ok=True)
    os.makedirs(reports_folder, exist_ok=True)
    
    # Step 1: Download transcript (if YouTube URL provided)
    transcript_path = None
    if youtube_url:
        print("\n" + "="*70)
        print("STEP 1: Downloading Transcript")
        print("="*70)
        from transcript import download_youtube_transcript
        
        print(f"Downloading transcript for URL: {youtube_url}")
        print(f"Target folder: {transcripts_folder}")
        vtt_path, transcript_path = download_youtube_transcript(
            youtube_url,
            transcripts_folder,
            lang='en',
            prefer_auto=False
        )
        print(f"Download result - VTT: {vtt_path}, TXT: {transcript_path}")
    else:
        # Look for existing transcript
        print("\n" + "="*70)
        print("STEP 1: Looking for Existing Transcript")
        print("="*70)
        possible_transcripts = glob.glob(os.path.join(video_folder, "*.txt"))
        for t in possible_transcripts:
            filename = os.path.basename(t)
            if "cleaned" not in filename and "report" not in filename and "metadata" not in filename:
                # Move to transcripts folder
                transcript_path = os.path.join(transcripts_folder, "transcript.txt")
                shutil.copy(t, transcript_path)
                print(f"Found and copied transcript: {transcript_path}")
                break
    
    if not transcript_path or not os.path.exists(transcript_path):
        print("Warning: No transcript found. PDF will be created without transcript.")
    else:
        print(f"Using transcript file: {transcript_path}")
        print(f"Transcript file size: {os.path.getsize(transcript_path)} bytes")
        # Print first few lines for debug
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                print(f"Transcript head: {f.read(200)}...")
        except Exception as e:
            print(f"Could not read transcript head: {e}")
    
    # Step 2: Optimize images (crop borders)
    print("\n" + "="*70)
    print("STEP 2: Optimizing Images (Removing Borders)")
    print("="*70)
    from scripts.optimize_images import process_images
    process_images(images_folder, images_optimized, crop=True, compress=False, format='png')
    
    # Step 3: Run analysis (Deduplication)
    print("\n" + "="*70)
    print("STEP 3: Analyzing Images for Duplicates (Curator Grid)")
    print("="*70)
    
    # Define paths for new workflow
    # We assume 'moderate' mode is the default for the new script
    curated_folder = os.path.join(video_folder, "images_optimized", "organized_moderate", "unique")
    report_path = os.path.join(images_optimized, "duplicates_report_combined.html")
    
    # Check if curation has already been done (i.e., unique folder exists)
    if os.path.exists(curated_folder) and os.listdir(curated_folder):
        print(f"Found curated images in: {curated_folder}")
        print("Proceeding with PDF generation using curated images.")
        # Update images_optimized to point to the curated folder for the next steps
        images_to_use = curated_folder
    else:
        print("Running deduplication analysis...")
        # Import the new deduplication script
        from scripts import image_dedup
        
        # Run deduplication (Hash 12, Sequential)
        # We can call the main processing function directly or via subprocess
        # Calling via subprocess is safer to avoid global state issues with argparse
        import subprocess
        
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "image_dedup.py"),
            images_optimized,
            "--mode", "compare-all",
            "--sequential",
            "--crop-method", "content_aware",
            "--crop-margin", "0.20"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print("\n" + "!"*70)
        print("ACTION REQUIRED: Curation Needed")
        print("!"*70)
        print(f"1. Open the report: {report_path}")
        print("2. Review images, select 'Keep' or 'Discard'.")
        print("3. Click 'Download Move Script' in the report.")
        print("4. Run the downloaded .bat file (e.g., move_files_moderate.bat) in the images folder.")
        print("5. RERUN this workflow script to generate the final PDF.")
        print("!"*70)
        return

    # Step 4: Generate final PDF with transcript
    print("\n" + "="*70)
    print("STEP 4: Generating Final PDF")
    print("="*70)
    
    final_pdf = os.path.join(video_folder, "final.pdf")
    
    if transcript_path and os.path.exists(transcript_path):
        # Use optimized PDF generator with transcript
        from scripts.create_optimized_pdf import create_optimized_pdf_with_transcript
        create_optimized_pdf_with_transcript(
            images_to_use,
            transcript_path,
            final_pdf,
            jpeg_quality=quality
        )
    else:
        # Generate PDF without transcript
        print("Generating PDF without transcript...")
        from pdf_generator import convert_screenshots_to_pdf
        video_name = os.path.basename(video_folder)
        convert_screenshots_to_pdf(images_to_use, video_folder, video_name)
        # Rename to final.pdf
        generated_pdf = os.path.join(video_folder, f"{video_name}.pdf")
        if os.path.exists(generated_pdf):
            shutil.move(generated_pdf, final_pdf)
    
    # Step 5: Generate DOCX with transcript
    print("\n" + "="*70)
    print("STEP 5: Generating DOCX")
    print("="*70)
    
    final_docx = os.path.join(video_folder, "final.docx")
    
    if transcript_path and os.path.exists(transcript_path):
        from pdf_generator import sync_images_with_transcript_docx
        # Use optimized images for DOCX as well
        sync_images_with_transcript_docx(
            images_to_use,
            transcript_path,
            video_folder
        )
        
        # Rename to final.docx if needed (the function creates combined_slides_with_transcript.docx)
        generated_docx = os.path.join(video_folder, "combined_slides_with_transcript.docx")
        if os.path.exists(generated_docx):
            if os.path.exists(final_docx):
                os.remove(final_docx)
            os.rename(generated_docx, final_docx)
            print(f"DOCX created: {final_docx}")
    else:
        print("Skipping DOCX generation (no transcript available)")
    
    # Summary
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - Optimized images: {images_optimized}/")
    if transcript_path:
        print(f"  - Transcripts: {transcripts_folder}/")
    print(f"  - Analysis reports: {reports_folder}/")
    print(f"  - Final PDF: {final_pdf}")
    if os.path.exists(final_docx):
        print(f"  - Final DOCX: {final_docx}")
    
    # Show file size
    if os.path.exists(final_pdf):
        pdf_size = os.path.getsize(final_pdf) / 1024 / 1024
        print(f"\nFinal PDF size: {pdf_size:.2f} MB")
    
    if os.path.exists(final_docx):
        docx_size = os.path.getsize(final_docx) / 1024 / 1024
        print(f"Final DOCX size: {docx_size:.2f} MB")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete workflow to generate PDF from existing images"
    )
    parser.add_argument(
        "video_folder",
        help="Path to video folder containing images/ subdirectory"
    )
    parser.add_argument(
        "--youtube-url",
        help="YouTube URL to download transcript (optional)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=70,
        help="JPEG quality for PDF (1-100, default: 70)"
    )
    
    args = parser.parse_args()
    
    run_complete_workflow(args.video_folder, args.youtube_url, args.quality)
