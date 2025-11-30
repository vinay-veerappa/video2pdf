# Constants
OUTPUT_DIR = "./output"
FRAME_RATE = 6  # frames per second to process
WARMUP = FRAME_RATE  # initial frames to skip
FGBG_HISTORY = FRAME_RATE * 15  # frames in background model
VAR_THRESHOLD = 16  # Mahalanobis distance threshold
DETECT_SHADOWS = False
MIN_PERCENT = 0.1  # min % diff to detect motion stopped
MAX_PERCENT = 0.3  # max % diff to detect motion
SIMILARITY_THRESHOLD = 0.8  # Grid SSIM threshold (80% matching cells)
MIN_TIME_BETWEEN_CAPTURES = 1  # Minimum seconds between captures (0 = disabled by default)
MAX_SIMILARITY_COMPARISONS = 5  # Compare with last N images
