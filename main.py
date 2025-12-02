#!/usr/bin/env python3
"""
YouTube Video to PDF Slides Converter
Based on miavisc approach with YouTube URL support
"""

import os
import sys
import argparse
import glob
import re
import webbrowser
from pathlib import Path

# Import modules
from config import (
    OUTPUT_DIR, FRAME_RATE, SIMILARITY_THRESHOLD, MIN_PERCENT, MAX_PERCENT,
    MIN_TIME_BETWEEN_CAPTURES
)
from utils import (
    is_youtube_url, sanitize_filename, initialize_output_folder, 
    cleanup_temp_files
)
from downloader import download_youtube_video, get_video_title
from transcript import download_youtube_transcript, clean_transcript_text
from extractor import detect_unique_screenshots
from analyzer import analyze_images_comprehensive
from pdf_generator import convert_screenshots_to_pdf, sync_images_with_transcript, sync_images_with_transcript_docx
from report_generator import create_markdown_report


def main():
    parser = argparse.ArgumentParser(
        description="Convert YouTube video or local video file to PDF slides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python main.py "./input/video.mp4"
  python main.py "https://youtu.be/VIDEO_ID" --frame-rate 10 --similarity-threshold 0.98
  python main.py "https://youtu.be/VIDEO_ID" --download-transcript
  python main.py "https://youtu.be/VIDEO_ID" --download-transcript --create-combined
  python main.py "https://youtu.be/VIDEO_ID" --download-transcript --clean-transcript --create-combined
  python main.py "https://youtu.be/VIDEO_ID" --download-transcript --create-docx
        """
    )
    
    parser.add_argument(
        "input",
        help="YouTube URL or path to local video file"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "-fr", "--frame-rate",
        type=int,
        default=FRAME_RATE,
        help=f"Frames per second to process (default: {FRAME_RATE})"
    )
    
    parser.add_argument(
        "-st", "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Similarity threshold for duplicate detection 0-1 (default: {SIMILARITY_THRESHOLD})"
    )
    
    parser.add_argument(
        "--no-similarity",
        action="store_true",
        help="Disable similarity-based duplicate detection"
    )
    
    parser.add_argument(
        "--min-percent",
        type=float,
        default=MIN_PERCENT,
        help=f"Min %% diff to detect motion stopped (default: {MIN_PERCENT})"
    )
    
    parser.add_argument(
        "--max-percent",
        type=float,
        default=MAX_PERCENT,
        help=f"Max %% diff to detect motion (default: {MAX_PERCENT})"
    )
    
    parser.add_argument(
        "--download-transcript",
        action="store_true",
        help="Download transcript/subtitles from YouTube video (YouTube URLs only)"
    )
    
    parser.add_argument(
        "--transcript-lang",
        type=str,
        default="en",
        help="Language code for transcript (default: 'en'). Use 'auto' for auto-generated subtitles"
    )
    
    parser.add_argument(
        "--prefer-auto-subs",
        action="store_true",
        help="Prefer auto-generated subtitles over manual subtitles"
    )
    
    parser.add_argument(
        "--create-combined",
        action="store_true",
        help="Create a combined PDF with images and synchronized transcript"
    )

    parser.add_argument(
        "--create-docx",
        action="store_true",
        help="Create a combined DOCX with images and synchronized transcript"
    )
    
    parser.add_argument(
        "--create-markdown",
        action="store_true",
        help="Create a Markdown report suitable for NotebookLM"
    )
    
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed images as Base64 in the Markdown report (WARNING: Creates very large files)"
    )
    
    parser.add_argument(
        "--clean-transcript",
        action="store_true",
        help="Create a cleaned transcript file with paragraphs and spell checking"
    )
    
    parser.add_argument(
        "--min-time-interval",
        type=int,
        default=MIN_TIME_BETWEEN_CAPTURES,
        help=f"Minimum seconds between image captures (default: {MIN_TIME_BETWEEN_CAPTURES}, 0 = disabled)"
    )
    
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="Analyze images and generate report for duplicates and irrelevant content (does not auto-delete)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause for visual curation (Curator Grid) before generating final PDF"
    )
    
    parser.add_argument(
        "--save-duplicates",
        action="store_true",
        help="Save detected duplicates to a 'duplicates' folder instead of discarding them"
    )
    
    parser.add_argument(
        "--cookies",
        type=str,
        help="Path to cookies file (Netscape format) for YouTube authentication"
    )
    
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip video download and image extraction, only run post-processing on existing images"
    )

    parser.add_argument(
        "--optimize-images",
        action="store_true",
        help="Optimize images (crop borders) before generating PDF/DOCX"
    )

    parser.add_argument(
        "--similarity-method", 
        choices=['ssim', 'phash', 'cnn', 'grid'], 
        default='grid', 
        help="Method for duplicate detection (default: grid - 8x8 Grid SSIM)"
    )
    
    parser.add_argument("--transcribe-method", choices=['whisper', 'gemini'], default='whisper', help="Method for local video transcription (default: whisper)")
    parser.add_argument("--whisper-model", default='base', help="Whisper model size (tiny, base, small, medium, large)")
    

    args = parser.parse_args()
    
    # Get parameters from arguments
    frame_rate = args.frame_rate
    min_percent = args.min_percent
    max_percent = args.max_percent
    similarity_threshold = args.similarity_threshold
    
    # Ensure output directory exists
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    input_source = args.input
    video_path = None
    video_name = None
    
    try:
        # Check if input is YouTube URL or local file
        is_youtube = is_youtube_url(input_source)
        if is_youtube:
            print("Detected YouTube URL")
            video_name = get_video_title(input_source)
            if not args.skip_extraction:
                video_path = download_youtube_video(input_source, output_dir, cookies_path=args.cookies)
            else:
                print("Skipping video download (--skip-extraction)")
        else:
            print("Detected local video file")
            video_name = Path(input_source).stem
            if not args.skip_extraction:
                if not os.path.exists(input_source):
                    raise FileNotFoundError(f"Video file not found: {input_source}")
                video_path = input_source
            else:
                print("Skipping video check (--skip-extraction)")
        
        print(f"\nProcessing video: {video_name}")
        if video_path:
            print(f"Video path: {video_path}\n")
        
        # Initialize output folders
        output_folder, images_folder = initialize_output_folder(video_name, output_dir)
        
        # Move metadata file if it was created in the temp/video folder during download
        # The downloader creates it next to the video file
        if video_path and os.path.exists(video_path):
            temp_metadata = os.path.join(os.path.dirname(video_path), "metadata.txt")
            if os.path.exists(temp_metadata):
                import shutil
                final_metadata = os.path.join(output_folder, "metadata.txt")
                shutil.copy2(temp_metadata, final_metadata)
                print(f"Metadata file moved to: {final_metadata}")
        
        # Create duplicates folder if requested
        duplicates_folder = None
        if args.save_duplicates:
            duplicates_folder = os.path.join(output_folder, "duplicates")
            os.makedirs(duplicates_folder, exist_ok=True)
            print(f"Duplicates will be saved to: {duplicates_folder}")
        
        # Download transcript if requested and it's a YouTube URL
        transcript_vtt = None
        transcript_txt = None
        if args.download_transcript:
            if is_youtube:
                transcript_vtt, transcript_txt = download_youtube_transcript(
                    input_source,
                    output_folder,
                    lang=args.transcript_lang,
                    prefer_auto=args.prefer_auto_subs,
                    cookies_path=args.cookies
                )
            else:
                # Local video transcription
                from transcript import transcribe_video_local
                # Get API key if needed
                api_key = None
                if args.transcribe_method == 'gemini':
                    api_key = os.environ.get("GEMINI_API_KEY")
                    if not api_key:
                        try:
                            with open('generate_notes.py', 'r') as f:
                                content = f.read()
                                match = re.search(r'API_KEY\s*=\s*["\']([^"\']+)["\']', content)
                                if match:
                                    api_key = match.group(1)
                        except:
                            pass
                            
                transcript_txt = transcribe_video_local(
                    video_path, 
                    output_folder, 
                    method=args.transcribe_method,
                    model_size=args.whisper_model,
                    api_key=api_key
                )
        
        # If we skipped download or didn't request it, try to find existing transcript
        if not transcript_txt:
            # Check transcripts subfolder first
            transcripts_folder = os.path.join(output_folder, "transcripts")
            possible_transcripts = glob.glob(os.path.join(transcripts_folder, "*.txt"))
            
            # Also check main folder for backward compatibility
            possible_transcripts.extend(glob.glob(os.path.join(output_folder, "*.txt")))
            
            for t in possible_transcripts:
                filename = os.path.basename(t)
                if "cleaned" not in filename and "report" not in filename and "info" not in filename and "metadata" not in filename:
                    transcript_txt = t
                    print(f"Found existing transcript: {transcript_txt}")
                    break

        # Detect unique screenshots
        if not args.skip_extraction:
            screenshots_count = detect_unique_screenshots(
                video_path,
                images_folder,
                frame_rate=frame_rate,
                min_percent=min_percent,
                max_percent=max_percent,
                use_similarity=not args.no_similarity,
                similarity_threshold=similarity_threshold,
                min_time_interval=args.min_time_interval,
                save_duplicates_path=duplicates_folder,
                similarity_method=args.similarity_method
            )
        else:
            # Count existing images
            existing_images = glob.glob(os.path.join(images_folder, "*.png"))
            screenshots_count = len(existing_images)
            print(f"Skipping extraction. Found {screenshots_count} existing images.")
        
        if screenshots_count == 0:
            print("No screenshots were captured (or found). Exiting.")
            return
        
        # Analyze images if requested
        if args.post_process and not args.interactive:
            analysis_result = analyze_images_comprehensive(
                images_folder,
                similarity_threshold=similarity_threshold,
                output_folder=output_folder,
                move_duplicates=args.save_duplicates,
                similarity_method=args.similarity_method
            )
            if analysis_result:
                print(f"\nAnalysis Summary:")
                print(f"  - {analysis_result['total_flagged']} images flagged for review")
                print(f"  - {len(analysis_result['irrelevant'])} potentially irrelevant")
                print(f"  - {len(analysis_result['duplicates'])} duplicate groups")
                print(f"\nPlease review the report: {analysis_result['report_path']}")
        
        # Interactive Curation Mode
        if args.interactive:
            print("\n" + "="*60)
            print("INTERACTIVE CURATION MODE")
            print("="*60)
            print("Running deduplication analysis...")
            
            # Run image_dedup.py via subprocess to avoid state issues
            import subprocess
            
            # We assume 'moderate' mode for interactive workflow
            # Ensure we are using the optimized images if they exist
            target_images = images_folder
            
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "scripts", "image_dedup.py"),
                target_images,
                "--mode", "compare-all",
                "--sequential",
                "--crop-method", "content_aware",
                "--crop-margin", "0.20"
            ]
            
            try:
                subprocess.run(cmd, check=True)
                
                report_path = os.path.join(target_images, "duplicates_report_combined.html")
                if os.path.exists(report_path):
                    print(f"\nOpening report: {report_path}")
                    webbrowser.open(f"file://{os.path.abspath(report_path)}")
                    
                    print("\n" + "!"*60)
                    print("ACTION REQUIRED:")
                    print("1. Review images in the browser.")
                    print("2. Click 'Download Move Script'.")
                    print("3. Run the downloaded .bat file in the images folder.")
                    print("!"*60)
                    
                    input("Press Enter AFTER you have run the batch script to continue...")
                    
                    # Check for curated folder
                    curated_folder = os.path.join(target_images, "organized_moderate", "unique")
                    if os.path.exists(curated_folder) and os.listdir(curated_folder):
                        print(f"Found curated images in: {curated_folder}")
                        images_folder = curated_folder # Update pointer for PDF generation
                    else:
                        print("Warning: Curated 'unique' folder not found. Using original images.")
                else:
                    print("Error: Report not generated.")
            except Exception as e:
                print(f"Error running interactive curation: {e}")
        
        # Optimize images (crop) if requested
        if args.optimize_images:
            print("\nOptimizing images (cropping borders)...")
            from scripts.optimize_images import process_images
            # We process in-place or to a new folder? 
            # The workflow creates 'images_optimized'. 
            # To keep main.py simple, let's process in-place or update images_folder to point to optimized ones.
            
            images_optimized_folder = os.path.join(output_folder, "images_optimized")
            process_images(images_folder, images_optimized_folder, crop=True, compress=False, format='png')
            
            # Update images_folder to point to the optimized images for subsequent steps
            images_folder = images_optimized_folder
            print(f"Using optimized images from: {images_folder}")

        # Convert to PDF
        # We append '_slides_only' to avoid conflict with the combined PDF if both are generated
        # Or if the user only wants slides, this is still clear.
        pdf_path = convert_screenshots_to_pdf(images_folder, output_folder, f"{video_name}_slides_only")
        
        # Create cleaned transcript if requested
        cleaned_transcript_path = None
        if args.clean_transcript and transcript_txt:
            print("\nCreating cleaned transcript...")
            try:
                cleaned_transcript_path = os.path.join(output_folder, "transcript_cleaned.txt")
                
                # Try different encodings
                transcript_content = None
                for encoding in ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp1252']:
                    try:
                        with open(transcript_txt, 'r', encoding=encoding) as f:
                            transcript_content = f.read()
                        print(f"Read transcript with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if transcript_content is None:
                    raise ValueError("Could not read transcript file with any encoding")
                
                # Extract all text (remove timestamps for cleaning)
                text_only = re.sub(r'\[\d{2}:\d{2}:\d{2}\]\s*', '', transcript_content)
                cleaned_text = clean_transcript_text(text_only)
                
                with open(cleaned_transcript_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                print(f"Cleaned transcript saved to: {cleaned_transcript_path}")
            except Exception as e:
                print(f"Warning: Could not create cleaned transcript: {e}")
        
        # Create combined PDF if requested
        combined_pdf_path = None
        if args.create_combined and transcript_txt:
            combined_pdf_path = sync_images_with_transcript(
                images_folder, 
                transcript_txt, 
                output_folder,
                video_name
            )

        # Create combined DOCX if requested
        combined_docx_path = None
        if args.create_docx and transcript_txt:
            combined_docx_path = sync_images_with_transcript_docx(
                images_folder, 
                transcript_txt, 
                output_folder,
                video_name
            )
            
        # Create Markdown report if requested
        markdown_path = None
        if args.create_markdown and transcript_txt:
            markdown_path = create_markdown_report(
                images_folder,
                transcript_txt,
                output_folder,
                video_name,
                embed_images=args.embed_images
            )
        elif args.create_markdown and not transcript_txt:
            print("Warning: Cannot create Markdown report without transcript.")
        
        # Cleanup temp files
        cleanup_temp_files(output_dir)
        
        print("\n" + "="*60)
        print("Conversion completed successfully!")
        print(f"Images saved to: {images_folder}")
        print(f"PDF saved to: {pdf_path}")
        if transcript_txt:
            print(f"Transcript (text) saved to: {transcript_txt}")
        if transcript_vtt:
            print(f"Transcript (VTT) saved to: {transcript_vtt}")
        if cleaned_transcript_path:
            print(f"Cleaned transcript saved to: {cleaned_transcript_path}")
        if combined_pdf_path:
            print(f"Combined PDF (images + transcript) saved to: {combined_pdf_path}")
        if combined_docx_path:
            print(f"Combined DOCX (images + transcript) saved to: {combined_docx_path}")
        if markdown_path:
            print(f"NotebookLM Markdown Report saved to: {markdown_path}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def process_video_workflow(input_source, output_dir=OUTPUT_DIR, progress_callback=None, **kwargs):
    """
    Programmatic entry point for the video processing workflow.
    Returns: Dictionary with results (output_folder, images_folder, etc.)
    """
    # Default options
    options = {
        'frame_rate': FRAME_RATE,
        'min_percent': MIN_PERCENT,
        'max_percent': MAX_PERCENT,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'no_similarity': False,
        'min_time_interval': MIN_TIME_BETWEEN_CAPTURES,
        'download_transcript': True,
        'transcript_lang': 'en',
        'prefer_auto_subs': False,
        'skip_extraction': False,
        'optimize_images': False,
        'similarity_method': 'grid',
        'cookies': None,
        'save_duplicates': False
    }
    options.update(kwargs)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = None
    video_name = None
    
    # Check if input is YouTube URL or local file
    is_youtube = is_youtube_url(input_source)
    if is_youtube:
        print("Detected YouTube URL")
        video_name = get_video_title(input_source)
        if not options['skip_extraction']:
            video_path = download_youtube_video(
                input_source, 
                output_dir, 
                cookies_path=options['cookies'],
                progress_callback=progress_callback
            )
        else:
            print("Skipping video download (--skip-extraction)")
    else:
        print("Detected local video file")
        video_name = Path(input_source).stem
        if not options['skip_extraction']:
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"Video file not found: {input_source}")
            video_path = input_source
        else:
            print("Skipping video check (--skip-extraction)")
    
    print(f"\nProcessing video: {video_name}")
    
    # Initialize output folders
    output_folder, images_folder = initialize_output_folder(video_name, output_dir)
    
    # Create video folder
    video_output_folder = os.path.join(output_folder, "video")
    os.makedirs(video_output_folder, exist_ok=True)
    
    # Move metadata file
    if video_path and os.path.exists(video_path):
        temp_metadata = os.path.join(os.path.dirname(video_path), "metadata.txt")
        if os.path.exists(temp_metadata):
            import shutil
            final_metadata = os.path.join(output_folder, "metadata.txt")
            shutil.copy2(temp_metadata, final_metadata)
    
    # Create duplicates folder
    duplicates_folder = None
    if options['save_duplicates']:
        duplicates_folder = os.path.join(output_folder, "duplicates")
        os.makedirs(duplicates_folder, exist_ok=True)
    
    # Download transcript
    transcript_vtt = None
    transcript_txt = None
    if options['download_transcript'] and is_youtube:
        transcript_vtt, transcript_txt = download_youtube_transcript(
            input_source,
            output_folder,
            lang=options['transcript_lang'],
            prefer_auto=options['prefer_auto_subs'],
            cookies_path=options['cookies']
        )
    elif options['download_transcript'] and not is_youtube:
        # Local video transcription
        if video_path:
            from transcript import transcribe_video_local
            
            # Get API key if needed (reuse logic from main or simple env check)
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                try:
                    # Try to find it in generate_notes.py as fallback
                    with open('generate_notes.py', 'r') as f:
                        content = f.read()
                        match = re.search(r'API_KEY\s*=\s*["\']([^"\']+)["\']', content)
                        if match:
                            api_key = match.group(1)
                except:
                    pass
                    
            # Determine method (default to whisper if not specified in options, but options usually has defaults)
            method = options.get('transcribe_method', 'whisper')
            model_size = options.get('whisper_model', 'base')
            
            transcript_txt = transcribe_video_local(
                video_path, 
                output_folder, 
                method=method,
                model_size=model_size,
                api_key=api_key
            )
        else:
            print("Skipping local transcription (no video path available)")
    
    # Find existing transcript
    if not transcript_txt:
        transcripts_folder = os.path.join(output_folder, "transcripts")
        possible_transcripts = []
        for ext in ['*.txt', '*.vtt', '*.srt', '*.json', '*.xml']:
            possible_transcripts.extend(glob.glob(os.path.join(transcripts_folder, ext)))
            possible_transcripts.extend(glob.glob(os.path.join(output_folder, ext)))
        for t in possible_transcripts:
            filename = os.path.basename(t)
            if "cleaned" not in filename and "report" not in filename and "info" not in filename and "metadata" not in filename:
                transcript_txt = t
                break

    # Detect unique screenshots
    screenshots_count = 0
    if not options['skip_extraction']:
        # If user explicitly wants to extract (unchecked skip_extraction),
        # remove existing images to force re-extraction
        if os.path.exists(images_folder):
            existing_images = glob.glob(os.path.join(images_folder, "*.png"))
            if existing_images:
                print(f"Removing {len(existing_images)} existing images to force re-extraction...")
                for img in existing_images:
                    try:
                        os.remove(img)
                    except Exception as e:
                        print(f"Warning: Could not remove {img}: {e}")
        
        screenshots_count = detect_unique_screenshots(
            video_path,
            images_folder,
            frame_rate=options['frame_rate'],
            min_percent=options['min_percent'],
            max_percent=options['max_percent'],
            use_similarity=not options['no_similarity'],
            similarity_threshold=options['similarity_threshold'],
            min_time_interval=options['min_time_interval'],
            save_duplicates_path=duplicates_folder,
            similarity_method=options['similarity_method'],
            progress_callback=progress_callback
        )
    else:
        existing_images = glob.glob(os.path.join(images_folder, "*.png"))
        screenshots_count = len(existing_images)
    


    # Save video file if it exists in temp or was downloaded
    if video_path and os.path.exists(video_path):
        # Check if it's in the temp folder (meaning we downloaded it)
        # If it's a local file provided by user, we don't move it unless requested
        # But here we assume if we downloaded it (is_youtube), we want to save it.
        if is_youtube:
            try:
                video_filename = os.path.basename(video_path)
                final_video_path = os.path.join(video_output_folder, video_filename)
                if not os.path.exists(final_video_path):
                    shutil.copy2(video_path, final_video_path)
                    print(f"Video saved to: {final_video_path}")
            except Exception as e:
                print(f"Warning: Could not save video file: {e}")

    # Cleanup temp files
    cleanup_temp_files(output_dir)

    return {
        'video_name': video_name,
        'output_folder': output_folder,
        'images_folder': images_folder,
        'transcript_txt': transcript_txt,
        'screenshots_count': screenshots_count
    }



if __name__ == "__main__":
    main()
