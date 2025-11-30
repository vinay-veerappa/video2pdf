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
    metadata_file = os.path.join(video_folder, "metadata.txt")
    if youtube_url is None and os.path.exists(metadata_file):
        print(f"\nReading metadata from {metadata_file}")
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('url:'):
                        youtube_url = line.split('url:', 1)[1].strip()
                        print(f"Found YouTube URL: {youtube_url}")
                        break
        except Exception as e:
            print(f"Warning: Could not read metadata file: {e}")

    
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
        
        _, transcript_path = download_youtube_transcript(
            youtube_url,
            transcripts_folder,
            lang='en',
            prefer_auto=False
        )
    else:
        # Look for existing transcript
        print("\n" + "="*70)
        print("STEP 1: Looking for Existing Transcript")
        print("="*70)
        possible_transcripts = glob.glob(os.path.join(video_folder, "*.txt"))
        for t in possible_transcripts:
            if "cleaned" not in os.path.basename(t) and "report" not in os.path.basename(t):
                # Move to transcripts folder
                transcript_path = os.path.join(transcripts_folder, "transcript.txt")
                shutil.copy(t, transcript_path)
                print(f"Found and copied transcript: {transcript_path}")
                break
    
    if not transcript_path or not os.path.exists(transcript_path):
        print("Warning: No transcript found. PDF will be created without transcript.")
    
    # Step 2: Optimize images (crop borders)
    print("\n" + "="*70)
    print("STEP 2: Optimizing Images (Removing Borders)")
    print("="*70)
    from scripts.optimize_images import process_images
    process_images(images_folder, images_optimized, crop=True, compress=False, format='png')
    
    # Step 3: Run analysis
    print("\n" + "="*70)
    print("STEP 3: Analyzing Images for Duplicates")
    print("="*70)
    from analyzer import analyze_images_comprehensive
    
    analysis_result = analyze_images_comprehensive(
        images_optimized,
        similarity_threshold=0.8,
        output_folder=reports_folder,
        move_duplicates=False,
        similarity_method='grid'
    )
    
    if analysis_result:
        print(f"\nAnalysis Summary:")
        print(f"  - {len(analysis_result['irrelevant'])} potentially irrelevant images")
        print(f"  - {len(analysis_result['duplicates'])} duplicate groups")
        
        if analysis_result['irrelevant']:
            print("\nFlagged irrelevant images:")
            for img in analysis_result['irrelevant']:
                print(f"  - {img['filename']} ({img['reason']})")
    
    # Step 4: Generate final PDF with transcript
    print("\n" + "="*70)
    print("STEP 4: Generating Final PDF")
    print("="*70)
    
    final_pdf = os.path.join(video_folder, "final.pdf")
    
    if transcript_path and os.path.exists(transcript_path):
        # Use optimized PDF generator with transcript
        from scripts.create_optimized_pdf import create_optimized_pdf_with_transcript
        create_optimized_pdf_with_transcript(
            images_optimized,
            transcript_path,
            final_pdf,
            jpeg_quality=quality
        )
    else:
        # Generate PDF without transcript
        print("Generating PDF without transcript...")
        from pdf_generator import convert_screenshots_to_pdf
        video_name = os.path.basename(video_folder)
        convert_screenshots_to_pdf(images_optimized, video_folder, video_name)
        # Rename to final.pdf
        generated_pdf = os.path.join(video_folder, f"{video_name}.pdf")
        if os.path.exists(generated_pdf):
            shutil.move(generated_pdf, final_pdf)
    
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
    
    # Show file size
    if os.path.exists(final_pdf):
        pdf_size = os.path.getsize(final_pdf) / 1024 / 1024
        print(f"\nFinal PDF size: {pdf_size:.2f} MB")
    
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
