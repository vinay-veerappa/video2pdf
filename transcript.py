import os
import sys
import subprocess
import glob
import re
import html

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
    YOUTUBE_TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_API_AVAILABLE = False


def download_youtube_transcript(url, output_folder, lang='en', prefer_auto=False, cookies_path=None, output_filename=None):
    """Download transcript/subtitles from YouTube video using youtube-transcript-api (fast) or yt-dlp (fallback)"""
    print(f"\nDownloading transcript from YouTube video...")
    
    # Create transcripts subfolder
    transcripts_folder = os.path.join(output_folder, "transcripts")
    os.makedirs(transcripts_folder, exist_ok=True)
    
    # Extract video ID
    video_id = None
    if 'watch?v=' in url:
        video_id = url.split('watch?v=')[1].split('&')[0].split('?')[0]
    elif 'youtu.be/' in url:
        video_id = url.split('youtu.be/')[1].split('?')[0].split('&')[0]
    
    if not output_filename:
        output_filename = "transcript"
    
    txt_path = os.path.join(transcripts_folder, f"{output_filename}.txt")
    
    # Method 1: Try youtube-transcript-api (Faster, cleaner)
    if YOUTUBE_TRANSCRIPT_API_AVAILABLE and video_id:
        try:
            print("Attempting download with youtube-transcript-api...")
            # Instantiate API
            ytt_api = YouTubeTranscriptApi()
            
            # Get transcript list
            transcript_list = ytt_api.list(video_id)
            
            # Find transcript (prefer manual 'en', then auto 'en')
            transcript = None
            try:
                # Try manual English first
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except:
                try:
                    # Try generated English
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    # Try any English
                    transcript = transcript_list.find_transcript(['en'])
            
            if transcript:
                # Fetch transcript data
                transcript_data = transcript.fetch()
                
                # Manually format with timestamps instead of using TextFormatter
                # transcript_data is a FetchedTranscript containing FetchedTranscriptSnippet objects
                formatted_lines = []
                for entry in transcript_data:
                    # Access attributes (not dictionary keys)
                    # Convert start time (in seconds) to HH:MM:SS format
                    start_seconds = int(entry.start)
                    hours = start_seconds // 3600
                    minutes = (start_seconds % 3600) // 60
                    seconds = start_seconds % 60
                    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    # Get text and clean it
                    text = entry.text.strip()
                    if text:
                        formatted_lines.append(f"[{timestamp}] {text}")
                
                # Save with timestamps
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(formatted_lines))
                
                print(f"Transcript downloaded successfully: {txt_path}")
                print(f"  - {len(formatted_lines)} entries with timestamps")
                return None, txt_path
                
        except Exception as e:
            print(f"youtube-transcript-api failed: {e}")
            print("Falling back to yt-dlp...")

    # Method 2: Fallback to yt-dlp
    try:
        # Use video ID directly if we can extract it
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Build command for downloading subtitles
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--no-playlist",
            "--no-warnings",
            "--skip-download",  # Don't download video, just subtitles
            "--sub-lang", lang,
            "--sub-format", "vtt",  # WebVTT format (can also use 'srt', 'json3')
            "-o", os.path.join(transcripts_folder, "%(title)s.%(ext)s"),
        ]
        
        # Add cookies if available
        if cookies_path and os.path.exists(cookies_path):
            cmd.extend(["--cookies", cookies_path])
        
        # Prefer auto-generated subtitles if requested
        if prefer_auto:
            cmd.append("--write-auto-subs")
        else:
            cmd.append("--write-subs")
            # Also try auto-subs as fallback
            cmd.append("--write-auto-subs")
        
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find downloaded subtitle files
        subtitle_files = []
        for ext in ['*.vtt', '*.srt', '*.json3']:
            subtitle_files.extend(glob.glob(os.path.join(transcripts_folder, ext)))
        
        if subtitle_files:
            # Get the most recent subtitle file
            subtitle_file = max(subtitle_files, key=os.path.getmtime)
            print(f"Transcript downloaded: {subtitle_file}")
            
            # Also create a plain text version for easier reading (with timestamps)
            transcript_txt_path = os.path.join(transcripts_folder, f"{output_filename}.txt")
            convert_vtt_to_txt(subtitle_file, transcript_txt_path, keep_timestamps=True)
            
            return subtitle_file, transcript_txt_path
        else:
            print("Warning: No transcript file found. The video may not have subtitles available.")
            return None, None
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout
        print(f"Warning: Could not download transcript: {error_msg}")
        # Try with auto-generated subtitles as fallback
        if not prefer_auto:
            print("Trying auto-generated subtitles...")
            return download_youtube_transcript(url, output_folder, lang, prefer_auto=True, cookies_path=cookies_path)
        return None, None
    except Exception as e:
        print(f"Warning: Error downloading transcript: {e}")
        return None, None


def convert_vtt_to_txt(vtt_file, txt_file, keep_timestamps=True):
    """Convert VTT subtitle file to plain text format with optional timestamps"""
    try:
        with open(vtt_file, 'r', encoding='utf-8') as f:
            vtt_content = f.read()
        
        lines = vtt_content.split('\n')
        text_lines = []
        current_timestamp = None
        current_text_block = []  # Collect all text for current timestamp
        seen_blocks = set()  # To avoid duplicate blocks
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines, headers, and metadata
            if not line or line.startswith('WEBVTT') or \
               line.startswith('Kind:') or line.startswith('Language:') or \
               line.startswith('NOTE') or line.startswith('align:') or \
               line.startswith('position:'):
                i += 1
                continue
            
            # Extract timestamp from lines like "00:00:01.189 --> 00:00:01.199"
            timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})', line)
            if timestamp_match:
                # Process previous block if exists
                if current_timestamp and current_text_block:
                    # Combine all text lines for this timestamp
                    combined_text = ' '.join(current_text_block)
                    combined_text = ' '.join(combined_text.split())  # Normalize whitespace
                    
                    if combined_text:
                        # Check if we already have text for this exact timestamp
                        # If so, keep the longer/more complete version
                        existing_line = None
                        existing_idx = None
                        for idx, existing in enumerate(text_lines):
                            if existing.startswith(f"[{current_timestamp}]"):
                                existing_line = existing
                                existing_idx = idx
                                break
                        
                        if existing_line:
                            # Compare lengths - keep the longer one
                            existing_text = existing_line.replace(f"[{current_timestamp}] ", "")
                            if len(combined_text) > len(existing_text):
                                # Replace with longer version
                                if keep_timestamps:
                                    text_lines[existing_idx] = f"[{current_timestamp}] {combined_text}"
                                else:
                                    text_lines[existing_idx] = combined_text
                        else:
                            # New timestamp, add it
                            # Create unique key to avoid exact duplicates
                            unique_key = f"{current_timestamp}:{combined_text.lower()}"
                            if unique_key not in seen_blocks:
                                seen_blocks.add(unique_key)
                                
                                # Add timestamp if requested
                                if keep_timestamps:
                                    formatted_line = f"[{current_timestamp}] {combined_text}"
                                else:
                                    formatted_line = combined_text
                                
                                text_lines.append(formatted_line)
                                
                                # Clear seen_blocks periodically
                                if len(seen_blocks) > 200:
                                    seen_blocks = set(list(seen_blocks)[-100:])
                
                # Start new block with new timestamp
                current_timestamp = timestamp_match.group(1)
                # Convert to readable format: 00:00:01.189 -> 00:00:01
                if current_timestamp:
                    time_parts = current_timestamp.split('.')
                    readable_time = time_parts[0]  # Keep HH:MM:SS format
                    current_timestamp = readable_time
                current_text_block = []
                i += 1
                continue
            
            # Process text lines
            if line and not line.startswith('00:') and not line.isdigit():
                # Remove VTT formatting tags like <00:00:01.439><c>text</c>
                cleaned = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}><c>', '', line)
                cleaned = re.sub(r'</c>', '', cleaned)
                cleaned = re.sub(r'<c>', '', cleaned)
                
                # Decode HTML entities
                cleaned = html.unescape(cleaned)
                
                # Remove extra whitespace
                cleaned = ' '.join(cleaned.split())
                
                # Add to current text block if not empty
                if cleaned:
                    current_text_block.append(cleaned)
            
            i += 1
        
        # Process final block
        if current_timestamp and current_text_block:
            combined_text = ' '.join(current_text_block)
            combined_text = ' '.join(combined_text.split())
            if combined_text:
                unique_key = f"{current_timestamp}:{combined_text.lower()}"
                if unique_key not in seen_blocks:
                    if keep_timestamps:
                        formatted_line = f"[{current_timestamp}] {combined_text}"
                    else:
                        formatted_line = combined_text
                    text_lines.append(formatted_line)
        
        # Write to text file
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_lines))
        
        print(f"Plain text transcript created: {txt_file} ({len(text_lines)} lines)")
        return txt_file
    except Exception as e:
        print(f"Warning: Could not convert VTT to text: {e}")
        import traceback
        traceback.print_exc()
        return None


def clean_transcript_text(text_or_lines):
    """Clean and improve transcript text with advanced duplicate removal for overlapping captions"""
    try:
        # If input is a single string, split it, otherwise assume it's a list of lines
        if isinstance(text_or_lines, str):
            lines = text_or_lines.split('\n')
        else:
            lines = text_or_lines

        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove timestamp if present at start of line
            line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', line)
            
            # Basic cleanup
            line = re.sub(r'\s+', ' ', line)
            line = line.replace('>>', '')
            line = line.replace('[laughter]', '')
            line = line.replace('[music]', '')
            line = line.replace('[applause]', '')
            line = line.strip()
            
            if not line:
                continue

            # Check for overlap with previous line
            # Many VTT/SRT files repeat the end of the previous line in the new line
            
            # Check for exact subset
            if line in prev_line:
                continue # Skip completely redundant lines
                
            if prev_line in line:
                # If previous line is fully contained in current line, prefer current line
                if len(cleaned_lines) > 0:
                    cleaned_lines.pop()
                cleaned_lines.append(line)
                prev_line = line
                continue

            # Check for partial overlap (suffix of prev == prefix of curr)
            min_overlap = 10 # Minimum characters to consider an overlap
            max_check = min(len(prev_line), len(line))
            
            best_overlap = 0
            for i in range(max_check, min_overlap - 1, -1):
                suffix = prev_line[-i:]
                prefix = line[:i]
                if suffix == prefix:
                    best_overlap = i
                    break
            
            if best_overlap > 0:
                # Append only the new part
                new_part = line[best_overlap:].strip()
                if new_part:
                    cleaned_lines.append(new_part)
                    prev_line = line # Update prev_line to the FULL current line for next comparison
            else:
                # No overlap, just append
                cleaned_lines.append(line)
                prev_line = line

        # Join all parts
        full_text = ' '.join(cleaned_lines)
        
        # Now perform sentence splitting and paragraphing on the cleaner text
        sentences = re.split(r'([.!?]+)\s+', full_text)
        
        # Reconstruct sentences (split keeps delimiters)
        final_sentences = []
        current_sent = ""
        for part in sentences:
            if re.match(r'[.!?]+', part):
                current_sent += part
                final_sentences.append(current_sent.strip())
                current_sent = ""
            else:
                current_sent += part
        if current_sent:
            final_sentences.append(current_sent.strip())

        # Post-processing sentences
        processed_sentences = []
        seen_sentences = set()
        
        # Common filler phrases to remove
        junk_patterns = [
            r'^(um|uh|ah|like|so|you know|i mean)\W*$', # Just filler
            r'^can you (hear|see) me\??$',
            r'^type \d in the chat',
            r'^let me know if',
            r'^sound check',
            r'^mic check',
            r'^alright guys',
            r'^welcome back',
            r'^hit the like button',
            r'^subscribe',
            r'^check the link',
        ]
        
        for sentence in final_sentences:
            if not sentence or len(sentence) < 3: # Remove very short noise
                continue
                
            # Remove duplicates (case-insensitive)
            sentence_lower = sentence.lower()
            if sentence_lower in seen_sentences:
                continue
            
            # Check for junk patterns
            is_junk = False
            for pattern in junk_patterns:
                if re.search(pattern, sentence_lower):
                    is_junk = True
                    break
            if is_junk:
                continue
                
            seen_sentences.add(sentence_lower)
            
            # Basic fixes
            sentence = sentence.replace(' cuz ', ' because ')
            sentence = sentence.replace(' u ', ' you ')
            sentence = sentence.replace(' ur ', ' your ')
            sentence = sentence.replace(' im ', ' I\'m ')
            sentence = sentence.replace(' dont ', ' don\'t ')
            sentence = sentence.replace(' cant ', ' can\'t ')
            sentence = sentence.replace(' wont ', ' won\'t ')
            sentence = sentence.replace(' its ', ' it\'s ')
            sentence = sentence.replace(' thats ', ' that\'s ')
            
            # Remove filler words at start of sentence
            sentence = re.sub(r'^(Um|Uh|Ah|Like|So|You know),?\s+', '', sentence, flags=re.IGNORECASE)
            
            # Capitalize
            if sentence and not sentence[0].isupper():
                sentence = sentence[0].upper() + sentence[1:]
                
            processed_sentences.append(sentence)
            
        # Group into paragraphs
        paragraphs = []
        current_paragraph = []
        
        for sentence in processed_sentences:
            current_paragraph.append(sentence)
            if len(current_paragraph) >= 4 or len(sentence) > 200:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            
        return '\n\n'.join(paragraphs)

    except Exception as e:
        print(f"Warning: Error cleaning transcript: {e}")
        if isinstance(text_or_lines, list):
            return ' '.join(text_or_lines)
        return str(text_or_lines)

def clean_transcript_llm(input_file, output_file, model="gemma3", style="condensed"):
    """
    Advanced transcript cleaning using Ollama/LLM.
    style: 'condensed' (technical notes) or 'clean' (textbook style)
    """
    try:
        from scripts.ollama_utils import OllamaClient
        client = OllamaClient()
    except ImportError:
        print("Error: scripts.ollama_utils or OllamaClient not found. Skipping LLM cleaning.")
        return None

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        model = "gemma3"
        print(f"Advanced cleaning  VxV ({style}) using {model}...")
        
        # Chunking for LLM context limits
        chunk_size = 128000
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        cleaned_content = []
        
        # Select prompt based on style
        prompt_style = "CONDENSED TECHNICAL NOTES" if style == "condensed" else "CONCISE TEXTBOOK-STYLE NOTES"
        style_rules = """1. FORMAT: Use clear paragraphs. Start each paragraph with a timestamp [HH:MM:SS].
2. STYLE: Third-person technical documentation (condensed) or Direct professional (clean).
3. CONTENT: PRESERVE ALL TECHNICAL DETAILS."""

        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}...")
            prompt = f"""
Role: You are an expert Technical transcript Editor and Market Analyst who uses statistics and math to analyze market data. Your goal is Lossless Condensation of a trading transcript into a high-fidelity descriptive trascript.

Objective: Rewrite the transcript while preserving every specific numerical value, "If/Then" logic chain, and statistical correlation. You must not simplify technical jargon or omit specific price levels or live price reading, as these are critical for the student's data-set entry.

STRICT RULES:
NO INFORMATION LOSS: Do not summarize or generalize. If a mentor describes a specific sequence (e.g., "rejected the 09:00 mid, breached 10 basis points, and failed the 3-hour line"), every one of those specific anchors must remain in the final text. 
NUMERICAL FIDELITY: Retain all percentages (e.g., 73% Asia probability),  basis points, risk-to-reward ratios, and specific time-of-day references (e.g., 09:30, 13:00, 15:00).
PRESERVE LOGIC CHAINS: Every "If/Then" statement must be preserved in full. If the mentor says, "If price breaches X, then Y becomes the target," do not shorten this to "Y is the target."
CAPTURE Bar-by-Bar Sentiment: Retain every mention of candle behavior (e.g., "rejected hardcore," "sucked back into the range," "wiped both sides," or "footprint rejection").
RETAIN Positional Anchors: Do not omit the specific relationship between price and key levels like the "line in the sand," the "pink flag," or "advertising vs. acceptance".
INCLUDE Time-Price Fractals: Keep logic that ties specific time frames to price levels, such as the relationship between Monday/Tuesday as "Q1 of the weekly candle".
MAINTAIN "Advertising vs. Acceptance" Logic: Preserve the mentor's distinction between price just "advertising" (moving past a level briefly) and "accepting" (closing or sustaining price beyond it).
PRESERVE REFERENCS: Asia, London, NY1, N2 , C1 C2,C3, quarters, 05 boxes, 0930, instat, outofstat, VVIX
REMOVE all filler words (um, uh, you know) and off-topic banter/jokes, unnecessary pauses and irrelevant conversations. 
CONSOLIDATE related text into a single paragraph preferably in 5 min or longer segments. Break logically into multiple paragraphs if needed.
CAPTURE PSYCHOLOGICAL & STRATEGIC ADVICE: Ensure all advice regarding business plans, data collection, surviving drawdowns, and "becoming obsessed with one thing" is retained with the same weight as the numerical data.

STRICT TEXT REPLACEMENTS:
"VIVIX" -> "VVIX"
"MAE" -> "MAE"
"MFE" -> "MFE"
"FVG" -> "Fair Value Gap (FVG)"
"MA" -> "MAE"
"MF" -> "MFE"
"Dogee" -> "Doji"
"braker" -> "Breaker"
"DMP" -> "DNP" (Directional No Pullbacks)
"DWP" -> "DWP" (Directional With Pullbacks)
"MAMF" -> "MAE MFE"
FORMATTING: * Use timestamps [HH:MM:SS] to start paragraphs. 
NO METALANGUAGE: Do NOT include "Here are the notes", "Glossary:", "Summary:", or introductory/concluding remarks.
OUTPUT ONLY THE CLEANED PARAGRAPHS.
{chunk}

NOTES:
"""
            res = client.generate("gemma3", prompt)
            cleaned_chunk = res.get("response", "").strip()
            
            # Basic model response cleaning
            lines = cleaned_chunk.split('\n')
            filtered_lines = [l for l in lines if not l.strip().startswith(("Here is", "Sure", "These are", "**", "Glossary"))]
            cleaned_content.append("\n".join(filtered_lines))

        final_text = "\n\n".join(cleaned_content)
        
        # Final cleanup/normalization
        final_text = re.sub(r'\[(\d{2}:\d{2}:\d{2}):\d{2}\]', r'[\1]', final_text) # Fix HH:MM:SS:ms
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)
            
        print(f"LLM-cleaned transcript saved to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error during LLM transcript cleaning: {e}")
        return None


def transcribe_video_local(video_path, output_folder, method='whisper', model_size='base', api_key=None, output_filename=None):
    """
    Transcribe a local video file using either OpenAI Whisper (local) or Google Gemini (cloud).
    
    Args:
        video_path (str): Path to the video file.
        output_folder (str): Directory to save the transcript.
        method (str): 'whisper' or 'gemini'.
        model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
        api_key (str): Gemini API key (required if method='gemini').
        output_filename (str): Base name for the output file (default: 'transcript').
        
    Returns:
        str: Path to the generated transcript file.
    """
    print(f"\nTranscribing local video: {video_path}")
    print(f"Method: {method}")
    
    transcripts_folder = os.path.join(output_folder, "transcripts")
    os.makedirs(transcripts_folder, exist_ok=True)
    
    if not output_filename:
        output_filename = "transcript"
    
    output_path = os.path.join(transcripts_folder, f"{output_filename}.txt")
    
    if os.path.exists(output_path):
        print(f"Transcript already exists at: {output_path}")
        return output_path

    try:
        if method == 'whisper':
            return _transcribe_with_whisper(video_path, output_path, model_size)
        elif method == 'gemini':
            return _transcribe_with_gemini(video_path, output_path, api_key)
        else:
            raise ValueError(f"Unknown transcription method: {method}")
            
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return None


def _transcribe_with_whisper(video_path, output_path, model_size):
    """Transcribe using OpenAI Whisper locally."""
    try:
        import whisper
        import torch
    except ImportError:
        print("Error: 'openai-whisper' not installed. Please run: pip install openai-whisper")
        return None

    print(f"Loading Whisper model '{model_size}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = whisper.load_model(model_size, device=device)
    
    print("Transcribing audio (this may take a while)...")
    # Whisper handles audio extraction from video automatically via ffmpeg
    result = model.transcribe(video_path, verbose=False)
    
    # Format segments with timestamps
    formatted_lines = []
    for segment in result['segments']:
        start = segment['start']
        text = segment['text'].strip()
        
        # Format timestamp HH:MM:SS
        hours = int(start // 3600)
        minutes = int((start % 3600) // 60)
        seconds = int(start % 60)
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        formatted_lines.append(f"[{timestamp}] {text}")
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(formatted_lines))
        
    print(f"Transcription complete: {output_path}")
    return output_path


def _transcribe_with_gemini(video_path, output_path, api_key):
    """Transcribe using Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("Error: 'google-generativeai' not installed.")
        return None
        
    if not api_key:
        print("Error: Gemini API key is required for Gemini transcription.")
        return None
        
    genai.configure(api_key=api_key)
    
    print("Extracting audio for Gemini upload...")
    # Extract audio to a temp file first (Gemini supports video, but audio is smaller/faster)
    audio_path = os.path.join(os.path.dirname(output_path), "temp_audio.mp3")
    
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn", # No video
            "-acodec", "libmp3lame",
            "-q:a", "4",
            audio_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Uploading audio to Gemini...")
        myfile = genai.upload_file(audio_path)
        
        print("Waiting for file processing...")
        import time
        while myfile.state.name == "PROCESSING":
            time.sleep(2)
            myfile = genai.get_file(myfile.name)
            
        if myfile.state.name == "FAILED":
            raise Exception("Gemini file processing failed.")
            
        print("Generating transcript...")
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        prompt = """
        Generate a detailed transcript for this audio. 
        Format each line with a timestamp at the beginning in [HH:MM:SS] format.
        Example:
        [00:00:00] Hello and welcome to this video.
        [00:00:05] Today we will discuss...
        """
        
        # Configure generation to handle long transcripts
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=65536,  # Increase token limit to max supported
            temperature=0.2,         # Lower temperature for more accurate transcription
        )
        
        response = model.generate_content(
            [myfile, prompt],
            generation_config=generation_config
        )
        
        # Clean up response text (remove markdown code blocks if present)
        text = response.text
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        print(f"Transcription complete: {output_path}")
        
        # Cleanup
        try:
            os.remove(audio_path)
            myfile.delete()
        except:
            pass
            
        return output_path
        
    except Exception as e:
        print(f"Gemini transcription failed: {e}")
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None

