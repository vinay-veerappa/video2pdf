# Video to Document Converter (Video2PDF)
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need `ffmpeg` installed on your system for video processing.*

3.  **GPU Support (Recommended)**:
    For faster processing and better accuracy (OCR), install PyTorch with CUDA support:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

## Usage

### Web Interface (Recommended)

Start the web application:
```bash
python app.py
```
Open your browser to `http://localhost:5000`.

**Workflow:**
1.  **Start**: Enter a YouTube URL or select an existing project.
2.  **Process**: The system downloads the video, extracts frames, and detects duplicates.
    *   **Options**:
        *   **Skip Video Download**: Use existing video file if available.
        *   **Skip Extraction**: Use existing extracted images.
        *   **Skip Deduplication**: Use existing deduplication results.
        *   **Download Transcript**: Download subtitles/transcript from YouTube.
3.  **Curate**: Review the extracted slides in the "Curate Slides" interface.
    *   **Green**: Kept slides.
    *   **Red**: Detected duplicates (discarded).
    *   **Yellow**: Detected blanks (discarded).
    *   *Click any slide to toggle its status.*
4.  **Edit**: Review the synchronized transcript for each slide.
5.  **Generate**: Download your final PDF or DOCX.

### CLI Usage (Advanced)

You can also run the pipeline from the command line:

```bash
python main.py <VIDEO_URL_OR_PATH> [OPTIONS]
```

**Common Options:**
*   `--download-transcript`: Download subtitles/transcript from YouTube.
*   `--create-docx`: Generate a DOCX file with synchronized transcript.
*   `--create-combined`: Generate a PDF file with synchronized transcript.
*   `--optimize-images`: Resize and compress images to reduce output file size.

## Output Structure

The tool organizes outputs into a structured folder:

```
output/
└── Video Title/
    ├── video/                  # Downloaded video file
    ├── images/                 # Extracted screenshots
    │   └── organized_moderate/ # Curated unique/duplicate folders
    ├── transcripts/            # Raw transcript files
    ├── Video Title.pdf         # Final PDF
    ├── Video Title.docx        # Final DOCX
    └── metadata.txt            # Video metadata
```
