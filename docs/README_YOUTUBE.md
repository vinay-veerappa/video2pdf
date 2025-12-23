# YouTube Video to PDF Slides Converter

A Python tool that converts YouTube videos or local video files into PDF slides by extracting unique frames. Based on the [miavisc](https://github.com/pannxe/miavisc) approach with enhanced features including YouTube URL support and similarity-based duplicate detection.

## Features

- 🎥 **YouTube URL Support** - Directly download and process videos from YouTube
- 📁 **Local Video Support** - Process local video files (mp4, avi, etc.)
- 🚀 **Fast Processing** - Efficient frame extraction with configurable frame rates
- 🎯 **Smart Duplicate Detection** - Uses structural similarity (SSIM) to avoid duplicate slides
- 📊 **Motion Detection** - Background subtraction algorithm to detect slide changes
- 📦 **Organized Output** - All artifacts (images, PDF) stored in organized output folder
- ⚙️ **Configurable** - Tunable parameters for different video types

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

**YouTube URL:**
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Local Video File:**
```bash
python main.py "./input/video.mp4"
```

### Advanced Options

```bash
python main.py "YOUTUBE_URL_OR_PATH" [OPTIONS]
```

**Options:**
- `-o, --output DIR` - Output directory (default: `./output`)
- `-fr, --frame-rate N` - Frames per second to process (default: 6)
- `-st, --similarity-threshold FLOAT` - Similarity threshold 0-1 for duplicate detection (default: 0.95)
- `--no-similarity` - Disable similarity-based duplicate detection
- `--min-percent FLOAT` - Min % diff to detect motion stopped (default: 0.1)
- `--max-percent FLOAT` - Max % diff to detect motion (default: 0.3)

### Examples

**High quality extraction (slower but more accurate):**
```bash
python main.py "https://youtu.be/VIDEO_ID" --frame-rate 10 --similarity-threshold 0.98
```

**Fast extraction (faster but may miss some slides):**
```bash
python main.py "https://youtu.be/VIDEO_ID" --frame-rate 3 --similarity-threshold 0.90
```

**Disable similarity check (use only motion detection):**
```bash
python main.py "./video.mp4" --no-similarity
```

**Download transcript/subtitles (YouTube URLs only):**
```bash
python main.py "https://youtu.be/VIDEO_ID" --download-transcript
```

**Download transcript in specific language:**
```bash
python main.py "https://youtu.be/VIDEO_ID" --download-transcript --transcript-lang es
```

**Prefer auto-generated subtitles:**
```bash
python main.py "https://youtu.be/VIDEO_ID" --download-transcript --prefer-auto-subs
```

## Output Structure

The tool creates the following structure in the output folder:

```
output/
├── video_name/
│   ├── images/
│   │   ├── 000_0.00.png
│   │   ├── 001_1.25.png
│   │   └── ...
│   ├── video_name.pdf
│   ├── transcript.txt (if --download-transcript used)
│   └── video_name.en.vtt (if --download-transcript used)
└── temp/  (temporary files, auto-cleaned)
```

- **images/** - Contains all extracted slide images (raw images)
- **video_name.pdf** - Final PDF with all slides
- **transcript.txt** - Plain text transcript (if transcript download enabled)
- **video_name.en.vtt** - VTT format transcript (if transcript download enabled)
- **temp/** - Temporary download files (automatically cleaned up)

## How It Works

1. **Video Download** (if YouTube URL):
   - Uses `yt-dlp` to download the video in best available quality
   - Saves to temporary folder

2. **Frame Extraction**:
   - Processes video at configurable frame rate (default: 6 fps)
   - Uses MOG2 background subtraction to detect motion
   - Captures frames when motion stops (slide is stable)

3. **Duplicate Detection**:
   - Compares new frames with previous frames using SSIM (Structural Similarity Index)
   - Skips frames that are too similar (configurable threshold)
   - Prevents duplicate slides from minor variations (mouse movements, etc.)

4. **PDF Generation**:
   - Converts all unique slide images to a single PDF
   - Images are sorted by timestamp
   - PDF is saved alongside raw images

5. **Transcript Download** (optional, YouTube URLs only):
   - Downloads subtitles/transcripts using yt-dlp
   - Supports both manual and auto-generated subtitles
   - Creates both VTT format (with timestamps) and plain text format
   - Saves to output folder alongside images and PDF

## Parameters Explained

- **Frame Rate (`--frame-rate`)**: How many frames per second to process. Higher = more accurate but slower. Lower = faster but may miss slides.
- **Similarity Threshold (`--similarity-threshold`)**: 0-1 value. Higher = stricter (fewer duplicates). Lower = more lenient (may include similar slides).
- **Min Percent (`--min-percent`)**: Minimum percentage of foreground pixels to consider motion stopped. Lower = more sensitive.
- **Max Percent (`--max-percent`)**: Maximum percentage to consider frame as "in motion". Higher = less sensitive to small movements.
- **Download Transcript (`--download-transcript`)**: Download subtitles/transcript from YouTube video (YouTube URLs only).
- **Transcript Language (`--transcript-lang`)**: Language code for transcript (default: 'en'). Examples: 'en', 'es', 'fr', 'de', etc.
- **Prefer Auto Subs (`--prefer-auto-subs`)**: Prefer auto-generated subtitles over manual subtitles.

## Tips for Best Results

1. **For lecture videos with clear slide transitions**: Use default settings
2. **For videos with animations**: Increase `--frame-rate` to 10-15 and lower `--similarity-threshold` to 0.90
3. **For videos with camera overlays**: Consider cropping the video first or adjust the ignored area (future feature)
4. **For fast processing**: Use `--frame-rate 3` and `--similarity-threshold 0.90`

## Troubleshooting

**Error: "Video file not found"**
- Check that the YouTube URL is valid and accessible
- Ensure you have internet connection for YouTube downloads
- For local files, verify the path is correct

**Too many duplicate slides:**
- Increase `--similarity-threshold` (e.g., 0.98)
- Enable similarity detection (remove `--no-similarity` if used)

**Missing slides:**
- Increase `--frame-rate` (e.g., 10-15)
- Decrease `--min-percent` (e.g., 0.05)
- Decrease `--similarity-threshold` (e.g., 0.90)

**Download fails:**
- Update `yt-dlp`: `pip install --upgrade yt-dlp`
- Check internet connection
- Some videos may be region-restricted or require login

**Transcript not available:**
- Not all videos have subtitles/transcripts
- Try using `--prefer-auto-subs` to get auto-generated subtitles
- Check if the video has subtitles available on YouTube
- Some videos may not have subtitles in the requested language

## Dependencies

- `opencv-python` - Video processing and computer vision
- `imutils` - Image utilities
- `img2pdf` - PDF generation
- `yt-dlp` - YouTube video downloading
- `scikit-image` - Image similarity calculation (SSIM)
- `numpy` - Numerical operations

## Credits

- Inspired by [miavisc](https://github.com/pannxe/miavisc) - Video to Slide Converter
- Uses background subtraction approach similar to [LearnOpenCV tutorial](https://learnopencv.com/)

## License

MIT License

