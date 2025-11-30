# Video to Document Converter (Video2PDF)

A powerful tool to convert video presentations (YouTube or local) into high-quality PDF and DOCX documents. It captures unique slides, synchronizes them with the video transcript, and optimizes the output for sharing.

## Features

*   **Video Processing**: Downloads videos and transcripts directly from YouTube.
*   **Smart Slide Extraction**: Uses Structural Similarity Index (SSIM) to detect and extract unique slides, minimizing duplicates.
*   **Document Generation**:
    *   **PDF**: Creates a visual slide deck.
    *   **DOCX**: Generates a Word document with slides and their corresponding transcript text.
    *   **Combined PDF**: Produces a PDF with both slides and synchronized transcripts.
*   **Image Optimization**: Automatically resizes and compresses images to keep document file sizes manageable without losing legibility.
*   **Smart Deduplication**: Includes a specialized tool (`scripts/image_dedup.py`) to detect and remove duplicate slides, specifically designed to ignore video conference UI elements (Zoom/Teams bars, etc.).

## Installation

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need `ffmpeg` installed on your system for video processing.*

## Usage

### 1. Main Conversion Tool

The main script handles the entire pipeline: downloading, extracting, and generating documents.

```bash
python main.py <VIDEO_URL_OR_PATH> [OPTIONS]
```

**Common Options:**
*   `--download-transcript`: Download subtitles/transcript from YouTube.
*   `--create-docx`: Generate a DOCX file with synchronized transcript.
*   `--create-combined`: Generate a PDF file with synchronized transcript.
*   `--optimize-images`: Resize and compress images to reduce output file size.
*   `--save-duplicates`: Keep duplicate frames (useful for debugging).

**Example:**
```bash
python main.py "https://www.youtube.com/watch?v=example" --download-transcript --create-docx --create-combined --optimize-images
```

### 2. Smart Image Deduplication

If the main extraction process leaves too many similar slides (common in screen recordings with static UI bars), use the dedicated deduplication tool.

```bash
python scripts/image_dedup.py <IMAGES_DIRECTORY> [OPTIONS]
```

**Modes:**
*   `--mode strict`: Only removes near-identical images (Threshold: 8).
*   `--mode moderate`: Balanced approach (Threshold: 12). **Default**.
*   `--mode lenient`: Aggressive removal of visually similar images (Threshold: 15). Recommended for video screenshots with compression artifacts.

**Other Options:**
*   `--compare-modes`: Runs all 3 modes and generates HTML reports for comparison.
*   `--report`: Generates an HTML report showing side-by-side comparisons of detected duplicates.
*   `--visualize`: Saves images showing exactly what parts of the image are being cropped/analyzed.

**Example:**
```bash
python scripts/image_dedup.py "./output/My Video/images" --mode lenient --report
```

## Output Structure

The tool organizes outputs into a structured folder:

```
output/
└── Video Title/
    ├── images/                 # Extracted screenshots
    ├── transcripts/            # Raw transcript files
    ├── Video Title.pdf         # Combined PDF (if requested)
    ├── Video Title.docx        # Combined DOCX (if requested)
    ├── Video Title_slides.pdf  # Slides-only PDF
    └── metadata.txt            # Video metadata
```

## Credits

Original concept based on `video2pdfslides`. Enhanced with advanced transcript synchronization, document generation, and smart deduplication logic.
