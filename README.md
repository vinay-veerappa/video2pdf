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

3.  **Configuration**:
    *   Duplicate the example configuration file:
        ```bash
        cp .env.example .env
        ```
    *   Open `.env` and add your **Gemini API Key** (get one from [Google AI Studio](https://aistudio.google.com/app/apikey)).
    *   *This is required for advanced transcript cleaning features.*

    *   *This is required for advanced transcript cleaning features.*

4.  **Ollama (Optional - For AI Transcript Cleaning)**:
    *   Install [Ollama](https://ollama.com/).
    *   Pull the required model (default is `gemma3`):
        ```bash
        ollama pull gemma3
        ```
    *   Ensure Ollama is running (`ollama serve`).

## Advanced Setup

### YouTube Authentication (Cookies)
If you encounter "Sign in to confirm you're not a bot" errors:
1.  **Automatic**: Log in to YouTube in Chrome, Edge, or Firefox. The tool attempts to extract cookies automatically.
2.  **Manual**:
    *   Export your YouTube cookies to Netscape format (using a browser extension like "Get cookies.txt LOCALLY").
    *   Save as `cookies.txt`.
    *   Use the CLI: `python main.py URL --cookies cookies.txt`.

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

## Project Structure

This repository is organized as follows:
- **`app.py`**: Main Flask web application.
- **`main.py`**: CLI entry point.
- **`scripts/`**: Utility scripts (e.g., `analyze_images.py`, `convert_md_to_docx.py`) and experimental tools.
- **`docs/`**: Documentation files including testing guides and optimization reports.
- **`experiments/`**: Source code for previous experiments (deduplication, compression).
- **`archive/`**: Deprecated code and temporary files.

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

For more detailed documentation, please check the `docs/` folder.
