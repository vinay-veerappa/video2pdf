# Component Design: Frame Extraction (`extractor.py`)

## 1. Overview
`extractor.py` is responsible for converting the video file into a sequence of individual image frames (screenshots). It balances speed and coverage by using FFmpeg for the heavy lifting and Python for file management.

## 2. Responsibilities
- **High-Speed Extraction**: Uses FFmpeg with hardware acceleration (CUDA) where available.
- **I-Frame Extraction**: Optimized to extract Keyframes (I-Frames) which are typically the highest quality and most distinct frames in a video.
- **Timestamp Management**: Extracts exact presentation timestamps (PTS) for accurate synchronization with transcripts.

## 3. Internal Logic Flow

```mermaid
graph TD
    Start([Start Extraction]) --> CheckExist{Images Exist?}
    CheckExist -->|Yes| Return([Return Path])
    CheckExist -->|No| SetupDir[Setup Temp Directory]
    
    SetupDir --> FFmpeg[Run FFmpeg Command]
    
    FFmpeg -->|Try| CUDA[Hardware Accel (CUDA)]
    CUDA --> Success{Success?}
    
    Success -->|No| CPU[Fallback to CPU]
    Success -->|Yes| ParseLogs[Parse Stderr for Timestamps]
    CPU --> ParseLogs
    
    ParseLogs --> MoveFiles[Move & Rename Files]
    
    MoveFiles --> Loop{For each file}
    Loop --> Rename[Rename: 001_12.5.png]
    Rename --> Loop
    
    Loop --> Done([Cleanup & Return])
```

## 4. Key Functions

### `detect_unique_screenshots(...)`
The main function called by `main.py`.
- **Inputs**: `video_path`, `output_folder`, `frame_rate`.
- **Logic**:
    1.  **FFmpeg Command Construction**:
        ```bash
        ffmpeg -skip_frame nokey -hwaccel cuda -i video.mp4 -vf scale=... -vsync vfr ...
        ```
        - `-skip_frame nokey`: Only process I-frames (huge speedup).
        - `-vsync vfr`: Variable frame rate output.
    2.  **Timestamp Parsing**: Reads FFmpeg's `showinfo` filter output from stderr to get the exact `pts_time` for each extracted frame.
    3.  **File Organization**: Renames the raw numbered output (e.g., `00001.png`) to a timestamp-based format (e.g., `001_5.23.png` where 5.23 is minutes).

### `get_frames(...)`
A generator function (legacy/alternative) that uses OpenCV to yield frames one by one.
- *Note*: Currently less used in favor of the FFmpeg batch approach due to performance.

## 5. Data Models
- **Output Filename Format**: `{index:03d}_{timestamp_minutes:.2f}.png`
    - Example: `042_12.50.png` -> 42nd extracted frame, occurs at 12 minutes 30 seconds.

## 6. Dependencies
- **External**:
    - `ffmpeg`: Must be installed on the system path.
    - `cv2` (OpenCV): Used for the generator method.
    - `subprocess`: To invoke FFmpeg.
