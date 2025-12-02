# Component Design: CLI Entry Point (`main.py`)

## 1. Overview
`main.py` serves as the command-line interface (CLI) and the central orchestrator for the Video2PDF pipeline. It parses user arguments, configures the execution environment, and sequentially calls the core processing modules.

## 2. Responsibilities
- **Argument Parsing**: Handles a wide range of CLI flags for customization (frame rate, thresholds, output formats, etc.).
- **Input Validation**: Checks if the input is a YouTube URL or a local file.
- **Workflow Orchestration**: Calls `downloader`, `extractor`, `analyzer`, and `pdf_generator` in the correct order.
- **Error Handling**: Catches global exceptions and ensures proper exit codes.
- **Programmatic API**: Exposes `process_video_workflow` for use by the Web App (`app.py`).

## 3. Internal Logic Flow

```mermaid
graph TD
    Start([Start]) --> Args[Parse Arguments]
    Args --> InputCheck{Input Type?}
    
    InputCheck -->|YouTube URL| Download[Downloader: download_youtube_video]
    InputCheck -->|Local File| Validate[Validate File Path]
    
    Download --> Init[Initialize Output Folders]
    Validate --> Init
    
    Init --> Transcript{Download Transcript?}
    Transcript -->|Yes| DLTrans[Transcript: download_youtube_transcript]
    Transcript -->|No| CheckTrans[Check Existing Transcript]
    
    DLTrans --> Extract{Skip Extraction?}
    CheckTrans --> Extract
    
    Extract -->|No| Extraction[Extractor: detect_unique_screenshots]
    Extract -->|Yes| Count[Count Existing Images]
    
    Extraction --> PostProcess{Post-Process?}
    Count --> PostProcess
    
    PostProcess -->|Yes| Analyze[Analyzer: analyze_images_comprehensive]
    PostProcess -->|No| Interactive{Interactive Mode?}
    
    Analyze --> Interactive
    
    Interactive -->|Yes| Curate[Launch Interactive Curation]
    Interactive -->|No| Optimize{Optimize Images?}
    
    Curate --> Optimize
    
    Optimize -->|Yes| OptImg[Optimize Images (Crop/Compress)]
    Optimize -->|No| GenPDF[PDF Generator: convert_screenshots_to_pdf]
    
    OptImg --> GenPDF
    
    GenPDF --> GenCombined{Create Combined PDF/DOCX?}
    
    GenCombined -->|Yes| Sync[PDF Generator: sync_images_with_transcript]
    GenCombined -->|No| End([End])
    
    Sync --> End
```

## 4. Key Functions

### `main()`
The entry point for CLI execution.
- **Inputs**: Command line arguments (via `argparse`).
- **Outputs**: None (Side effects: files created).
- **Logic**: 
    1.  Sets up `argparse`.
    2.  Resolves input source.
    3.  Calls individual modules based on flags.
    4.  Prints summary.

### `process_video_workflow(input_source, ...)`
A reusable function designed for integration with `app.py`.
- **Inputs**: `input_source` (str), `output_dir` (str), `progress_callback` (func), and kwargs for options.
- **Outputs**: `dict` containing paths to generated folders and files.
- **Logic**: Similar to `main()` but returns structured data instead of printing to stdout, and supports a progress callback.

## 5. Dependencies
- **Internal**:
    - `config.py`: Global constants.
    - `utils.py`: Helper functions.
    - `downloader.py`: Video downloading.
    - `transcript.py`: Transcript handling.
    - `extractor.py`: Frame extraction.
    - `analyzer.py`: Image analysis.
    - `pdf_generator.py`: Document creation.
    - `report_generator.py`: Markdown reports.
- **External**:
    - `argparse`: CLI parsing.
    - `os`, `sys`, `glob`, `shutil`: File operations.

## 6. Usage Examples
```bash
# Basic usage
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"

# With transcript and combined PDF
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --download-transcript --create-combined

# Local file
python main.py "./my_video.mp4"
```
