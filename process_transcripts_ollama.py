"""
Enhanced ICT Trading Transcript Processor using Ollama
Features:
- Intelligent chunking with saved chunks
- Real-time progress visibility
- Resume capability
- Comprehensive extraction without information loss
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
import requests
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class EnhancedTranscriptProcessor:
    def __init__(self, input_dir, output_dir, model="qwen3:latest", 
                 ollama_url="http://localhost:11434", chunk_size=10000, skip_existing=True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model = model
        self.ollama_url = ollama_url
        self.chunk_size = chunk_size
        self.skip_existing = skip_existing
        
        # Create all output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "processed_notes").mkdir(exist_ok=True)
        (self.output_dir / "full_transcripts").mkdir(exist_ok=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
        (self.output_dir / "chunk_extractions").mkdir(exist_ok=True)
    
    def is_already_processed(self, filepath):
        """Check if this transcript has already been processed"""
        output_file = self.output_dir / "processed_notes" / f"{filepath.stem}_processed.md"
        
        if not output_file.exists():
            return False
        
        try:
            file_size = output_file.stat().st_size
            if file_size < 1000:
                return False
            
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read(500)
                if "ERROR" in content[:200]:
                    return False
            
            return True
        except Exception:
            return False
    
    def call_ollama(self, prompt, system_prompt="", timeout=600):
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 8192,
                "num_ctx": 16384  # Good for qwen3
            }
        }
        
        for attempt in range(3):
            try:
                print(f"    ⏳ Calling {self.model}...", end='', flush=True)
                start = time.time()
                response = requests.post(url, json=payload, timeout=timeout)
                elapsed = time.time() - start
                response.raise_for_status()
                print(f" ✓ ({elapsed:.1f}s)")
                return response.json()["response"]
            except Exception as e:
                print(f" ✗")
                if attempt == 2:
                    raise
                print(f"    ⚠️  Retry {attempt + 1}/3: {str(e)[:100]}")
                time.sleep(5)
    
    def split_transcript_by_timestamps(self, transcript):
        """Split transcript into chunks based on timestamps"""
        timestamp_pattern = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]'
        matches = list(re.finditer(timestamp_pattern, transcript))
        
        if not matches or len(transcript) < self.chunk_size:
            return [(transcript, "full", "00:00", "end")]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start_time = matches[0].group(1) if matches else "00:00"
        last_timestamp = chunk_start_time
        
        lines = transcript.split('\n')
        for line in lines:
            line_size = len(line) + 1
            ts_match = re.search(timestamp_pattern, line)
            if ts_match:
                last_timestamp = ts_match.group(1)
            
            if current_size + line_size > self.chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                time_range = f"{chunk_start_time} - {last_timestamp}"
                chunks.append((chunk_text, time_range, chunk_start_time, last_timestamp))
                
                overlap = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
                chunk_start_time = last_timestamp
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            time_range = f"{chunk_start_time} - {last_timestamp}"
            chunks.append((chunk_text, time_range, chunk_start_time, last_timestamp))
        
        return chunks if len(chunks) > 1 else [(transcript, "full", "00:00", "end")]
    
    def save_chunk(self, filepath_stem, chunk_num, chunk_text, start_time, end_time):
        """Save individual chunk to disk"""
        chunk_dir = self.output_dir / "chunks" / filepath_stem
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize timestamps for filename
        start_clean = start_time.replace(':', '-')
        end_clean = end_time.replace(':', '-')
        
        chunk_file = chunk_dir / f"chunk_{chunk_num:02d}_{start_clean}_to_{end_clean}.txt"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(f"# Chunk {chunk_num}\n")
            f.write(f"# Time Range: {start_time} to {end_time}\n")
            f.write(f"# Length: {len(chunk_text):,} characters\n")
            f.write("#" + "="*70 + "\n\n")
            f.write(chunk_text)
        
        return chunk_file
    
    def save_chunk_extraction(self, filepath_stem, chunk_num, extraction, start_time, end_time):
        """Save extraction from a chunk"""
        extract_dir = self.output_dir / "chunk_extractions" / filepath_stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        extract_file = extract_dir / f"chunk_{chunk_num:02d}_extraction.md"
        with open(extract_file, 'w', encoding='utf-8') as f:
            f.write(f"# Chunk {chunk_num} Extraction\n\n")
            f.write(f"**Time Range:** {start_time} to {end_time}\n\n")
            f.write(f"**Extracted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(extraction)
        
        return extract_file
    
    def extract_from_chunk(self, chunk, time_range, chunk_num, total_chunks):
        system_prompt = f"""Expert ICT analyst. Extract ALL trading information from chunk {chunk_num}/{total_chunks} ({time_range})."""
        
        prompt = f"""Extract EVERYTHING from this ICT transcript chunk {chunk_num}/{total_chunks} ({time_range}):

{chunk}

Extract:
- Session context, bias, conditions
- Price levels (exact numbers), FVGs, order blocks, liquidity
- Entry/exit criteria, stops, targets
- Timeframes, macros, session timing
- ICT terminology (PD arrays, imbalances, etc.)
- Casual comments, tidbits, pro tips
- Rules, conditions, decision frameworks
- News events, volatility, market context

Be comprehensive. Include timestamps. Capture every detail."""

        try:
            return self.call_ollama(prompt, system_prompt, timeout=900)
        except Exception as e:
            return f"ERROR chunk {chunk_num}: {e}"
    
    def extract_chart_setup(self, transcript):
        prompt = f"""Extract ALL chart details:

{transcript[:15000]}

List:
- ALL price levels (exact numbers)
- FVG boundaries, order blocks
- Liquidity levels, support/resistance
- Session times, macros
- Visual elements (boxes, lines, colors)
- Market structure (highs, lows, breaks)"""

        try:
            return self.call_ollama(prompt, "Extract chart setup for TradingView", timeout=900)
        except Exception as e:
            return f"ERROR: {e}"
    
    def extract_quotes(self, transcript):
        if len(transcript) > 20000:
            sections = [transcript[:7000], transcript[len(transcript)//2-3500:len(transcript)//2+3500], transcript[-7000:]]
            transcript = "\n\n[...]\n\n".join(sections)
        
        prompt = f"""Extract 10-15 most important quotes with timestamps:

{transcript}

Focus on: trading wisdom, rules, tidbits, risk management, decision frameworks.
Format: [HH:MM:SS] "quote" - context if needed"""

        try:
            return self.call_ollama(prompt, "Extract key quotes", timeout=600)
        except Exception as e:
            return f"ERROR: {e}"
    
    def process_single_transcript(self, filepath):
        if self.skip_existing and self.is_already_processed(filepath):
            print(f"\n⏭️  SKIPPED (already processed): {filepath.name}")
            output_file = self.output_dir / "processed_notes" / f"{filepath.stem}_processed.md"
            return {
                "filename": filepath.name,
                "output_file": str(output_file),
                "success": True,
                "skipped": True,
                "processing_time": "0s"
            }
        
        print(f"\n{'='*70}")
        print(f"📄 PROCESSING: {filepath.name}")
        print(f"{'='*70}")
        start = time.time()
        
        try:
            # Read transcript
            print(f"  📖 Reading transcript...")
            with open(filepath, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            length = len(transcript)
            print(f"  📊 Length: {length:,} characters")
            
            if length < 100:
                print(f"  ⊘ Too short, skipping")
                return None
            
            # Save full transcript
            print(f"  💾 Saving full transcript...")
            full_file = self.output_dir / "full_transcripts" / f"{filepath.stem}.txt"
            with open(full_file, 'w', encoding='utf-8') as f:
                f.write(transcript)
            print(f"     ✓ Saved to: {full_file.relative_to(self.output_dir)}")
            
            # Split into chunks
            print(f"\n  ✂️  Splitting into chunks...")
            chunks = self.split_transcript_by_timestamps(transcript)
            print(f"     ✓ Created {len(chunks)} chunk(s)")
            
            # Save chunks and extract
            all_extractions = []
            for i, (chunk, time_range, start_time, end_time) in enumerate(chunks, 1):
                print(f"\n  {'─'*68}")
                print(f"  📑 CHUNK {i}/{len(chunks)}: {time_range}")
                print(f"     Size: {len(chunk):,} characters")
                
                # Save chunk
                print(f"     💾 Saving chunk...")
                chunk_file = self.save_chunk(filepath.stem, i, chunk, start_time, end_time)
                print(f"        ✓ {chunk_file.relative_to(self.output_dir)}")
                
                # Extract from chunk
                print(f"     🔍 Extracting information...")
                extraction = self.extract_from_chunk(chunk, time_range, i, len(chunks))
                
                # Save extraction
                print(f"     💾 Saving extraction...")
                extract_file = self.save_chunk_extraction(filepath.stem, i, extraction, start_time, end_time)
                print(f"        ✓ {extract_file.relative_to(self.output_dir)}")
                
                all_extractions.append({
                    "chunk_num": i,
                    "time_range": time_range,
                    "extraction": extraction,
                    "chunk_file": str(chunk_file),
                    "extract_file": str(extract_file)
                })
            
            # Extract chart setup
            print(f"\n  {'─'*68}")
            print(f"  📊 Extracting chart setup from full transcript...")
            chart_desc = self.extract_chart_setup(transcript)
            
            # Extract quotes
            print(f"\n  💬 Extracting key quotes...")
            quotes = self.extract_quotes(transcript)
            
            # Build comprehensive notes
            print(f"\n  📝 Building comprehensive notes...")
            notes = f"""# Trading Review: {filepath.stem}

## Processing Info
- **Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model:** {self.model}
- **Length:** {length:,} characters
- **Chunks:** {len(chunks)}

## Chunk Files
"""
            
            for ext in all_extractions:
                notes += f"- Chunk {ext['chunk_num']}: `{Path(ext['chunk_file']).relative_to(self.output_dir)}`\n"
                notes += f"  - Extraction: `{Path(ext['extract_file']).relative_to(self.output_dir)}`\n"
            
            notes += "\n---\n\n"
            
            if len(chunks) > 1:
                notes += "## Comprehensive Extraction (All Chunks Combined)\n\n"
                for ext in all_extractions:
                    notes += f"### Chunk {ext['chunk_num']}: {ext['time_range']}\n\n"
                    notes += f"*See detailed extraction: `{Path(ext['extract_file']).relative_to(self.output_dir)}`*\n\n"
                    notes += f"{ext['extraction']}\n\n---\n\n"
            else:
                notes += f"## Comprehensive Extraction\n\n{all_extractions[0]['extraction']}\n\n---\n\n"
            
            notes += f"""## Chart Setup Guide\n\n{chart_desc}\n\n---\n\n"""
            notes += f"""## Key Quotes & Trading Wisdom\n\n{quotes}\n\n---\n\n"""
            notes += f"""## Original Transcript\n\n**Full:** `full_transcripts/{filepath.stem}.txt`\n\n**Preview:**\n```\n{transcript[:3000]}{'...' if len(transcript) > 3000 else ''}\n```\n"""
            
            output_file = self.output_dir / "processed_notes" / f"{filepath.stem}_processed.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(notes)
            
            elapsed = time.time() - start
            print(f"\n  {'─'*68}")
            print(f"  ✅ COMPLETED in {elapsed:.1f}s")
            print(f"  📄 Final notes: {output_file.relative_to(self.output_dir)}")
            print(f"{'='*70}\n")
            
            return {
                "filename": filepath.name,
                "output_file": str(output_file),
                "success": True,
                "skipped": False,
                "processing_time": f"{elapsed:.1f}s",
                "transcript_length": length,
                "chunks_processed": len(chunks)
            }
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"\n  ❌ ERROR after {elapsed:.1f}s: {e}")
            print(f"{'='*70}\n")
            return {"filename": filepath.name, "success": False, "skipped": False, "error": str(e)}
    
    def process_all(self, max_workers=1):
        files = sorted(self.input_dir.glob("*.txt"))
        if not files:
            print(f"❌ No files in {self.input_dir}")
            return {}
        
        already_done = sum(1 for f in files if self.is_already_processed(f)) if self.skip_existing else 0
        to_process = len(files) - already_done
        
        print("="*70)
        print("ENHANCED ICT TRANSCRIPT PROCESSOR")
        print("="*70)
        print(f"📁 Input:  {self.input_dir}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🤖 Model:  {self.model}")
        print(f"📄 Total:  {len(files)} transcripts")
        if self.skip_existing and already_done > 0:
            print(f"✓  Done:   {already_done} (will skip)")
            print(f"⏳ Todo:   {to_process}")
        print(f"📏 Chunk:  {self.chunk_size:,} chars")
        print("="*70)
        
        results = []
        start = time.time()
        
        for idx, fp in enumerate(files, 1):
            print(f"\n[{idx}/{len(files)}] ", end='')
            result = self.process_single_transcript(fp)
            if result:
                results.append(result)
        
        elapsed = time.time() - start
        successful = sum(1 for r in results if r.get("success"))
        skipped = sum(1 for r in results if r.get("skipped"))
        processed = successful - skipped
        
        summary = {
            "total_files": len(files),
            "successful": successful,
            "newly_processed": processed,
            "skipped_existing": skipped,
            "failed": len(results) - successful,
            "elapsed_time": f"{elapsed:.1f}s",
            "model_used": self.model,
            "skip_existing": self.skip_existing
        }
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Total files:    {len(files)}")
        print(f"✓  Successful:     {successful}")
        print(f"⏭️  Skipped:        {skipped}")
        print(f"🆕 Newly processed: {processed}")
        print(f"✗  Failed:         {summary['failed']}")
        print(f"⏱️  Time:           {summary['elapsed_time']}")
        print(f"📁 Output:         {self.output_dir}")
        print(f"\n📂 Output Structure:")
        print(f"   ├── processed_notes/     (combined final notes)")
        print(f"   ├── full_transcripts/    (original transcripts)")
        print(f"   ├── chunks/              (individual chunks)")
        print(f"   └── chunk_extractions/   (extractions per chunk)")
        print("="*70)
        
        return summary


def main():
    INPUT_DIR = r"C:\ICT_Videos\TCM\2023\transcripts"
    OUTPUT_DIR = r"C:\ICT_Videos\TCM\2023\processed_output"
    MODEL = "llama:latest"
    CHUNK_SIZE = 8000
    SKIP_EXISTING = True
    
    processor = EnhancedTranscriptProcessor(
        INPUT_DIR, 
        OUTPUT_DIR, 
        MODEL, 
        chunk_size=CHUNK_SIZE,
        skip_existing=SKIP_EXISTING
    )
    
    print("\n🔌 Testing Ollama...")
    try:
        processor.call_ollama("Test", "You are helpful", timeout=60)
        print("✅ Connected!\n")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return
    
    processor.process_all(max_workers=1)


if __name__ == "__main__":
    main()
