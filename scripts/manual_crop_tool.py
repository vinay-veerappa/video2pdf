import cv2
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

@dataclass
class CropConfig:
    """Stores learned crop parameters"""
    top_ratio: float  # Toolbar as % of height
    bottom_ratio: float  # Status bar as % of height
    left_ratio: float  # Left panel as % of width
    right_ratio: float  # Right panel as % of width
    platform: str  # 'tradingview' or 'tradovate'
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))


class ImageCropCalibrator:
    """Interactive calibration tool"""
    
    def __init__(self):
        self.crop_regions = []
        self.current_img = None
        self.current_path = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_img = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_img = self.current_img.copy()
                cv2.rectangle(self.temp_img, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Calibration', self.temp_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # Normalize coordinates
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            
            # Ensure top-left to bottom-right
            top_left = (min(x1, x2), min(y1, y2))
            bottom_right = (max(x1, x2), max(y1, y2))
            
            h, w = self.current_img.shape[:2]
            
            # Store as ratios
            crop_region = {
                'path': str(self.current_path),
                'top_ratio': top_left[1] / h,
                'bottom_ratio': bottom_right[1] / h,
                'left_ratio': top_left[0] / w,
                'right_ratio': bottom_right[0] / w,
                'dimensions': (w, h)
            }
            
            self.crop_regions.append(crop_region)
            print(f"✓ Marked content region for {self.current_path.name}")
            print(f"  Top: {crop_region['top_ratio']:.3f}, Bottom: {crop_region['bottom_ratio']:.3f}")
            print(f"  Left: {crop_region['left_ratio']:.3f}, Right: {crop_region['right_ratio']:.3f}")
            
    def calibrate_image(self, img_path: Path):
        """Show image and let user mark content region"""
        self.current_path = img_path
        self.current_img = cv2.imread(str(img_path))
        
        if self.current_img is None:
            print(f"❌ Could not load {img_path}")
            return False
        
        # Remove black borders first to make marking easier
        cleaned = self.remove_uniform_borders(self.current_img)
        self.current_img = cleaned
        
        # Resize if too large
        h, w = self.current_img.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            self.current_img = cv2.resize(self.current_img, (new_w, new_h))
        
        self.temp_img = self.current_img.copy()
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', self.mouse_callback)
        
        print(f"\n📷 {img_path.name}")
        print("Draw a rectangle around the CONTENT area (exclude toolbars, menus, status bars)")
        print("Press SPACE to confirm, ESC to skip")
        
        while True:
            cv2.imshow('Calibration', self.temp_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("⏭️  Skipped")
                return False
            elif key == 32:  # SPACE
                if self.end_point:
                    cv2.destroyAllWindows()
                    return True
                    
        return False
    
    @staticmethod
    def remove_uniform_borders(img: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Remove black bars and uniform borders"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Find top boundary
        top = 0
        for i in range(h):
            if gray[i].std() > threshold:
                top = i
                break
        
        # Find bottom boundary
        bottom = h
        for i in range(h-1, -1, -1):
            if gray[i].std() > threshold:
                bottom = i + 1
                break
        
        # Find left boundary
        left = 0
        for i in range(w):
            if gray[:, i].std() > threshold:
                left = i
                break
        
        # Find right boundary
        right = w
        for i in range(w-1, -1, -1):
            if gray[:, i].std() > threshold:
                right = i + 1
                break
        
        return img[top:bottom, left:right]
    
    def compute_config(self, platform: str) -> CropConfig:
        """Calculate average crop ratios from marked regions"""
        if not self.crop_regions:
            raise ValueError("No calibration data available")
        
        top_ratios = [r['top_ratio'] for r in self.crop_regions]
        bottom_ratios = [r['bottom_ratio'] for r in self.crop_regions]
        left_ratios = [r['left_ratio'] for r in self.crop_regions]
        right_ratios = [r['right_ratio'] for r in self.crop_regions]
        
        config = CropConfig(
            top_ratio=np.mean(top_ratios),
            bottom_ratio=np.mean(bottom_ratios),
            left_ratio=np.mean(left_ratios),
            right_ratio=np.mean(right_ratios),
            platform=platform
        )
        
        # Print statistics
        print("\n" + "="*60)
        print(f"📊 Calibration Results ({len(self.crop_regions)} images)")
        print("="*60)
        print(f"Top margin:    {config.top_ratio:.3f} (±{np.std(top_ratios):.3f})")
        print(f"Bottom margin: {1-config.bottom_ratio:.3f} (±{np.std(bottom_ratios):.3f})")
        print(f"Left margin:   {config.left_ratio:.3f} (±{np.std(left_ratios):.3f})")
        print(f"Right margin:  {1-config.right_ratio:.3f} (±{np.std(right_ratios):.3f})")
        
        return config


class SmartCropper:
    """Production cropper using calibrated config"""
    
    def __init__(self, config: CropConfig):
        self.config = config
    
    def crop(self, img: np.ndarray, remove_borders: bool = True) -> np.ndarray:
        """Apply smart crop to image"""
        # First remove black bars if requested
        if remove_borders:
            img = self.remove_uniform_borders(img)
        
        h, w = img.shape[:2]
        
        # Apply learned ratios
        top = int(h * self.config.top_ratio)
        bottom = int(h * self.config.bottom_ratio)
        left = int(w * self.config.left_ratio)
        right = int(w * self.config.right_ratio)
        
        # Sanity checks
        if bottom <= top or right <= left:
            print("⚠️  Invalid crop region, returning original")
            return img
        
        return img[top:bottom, left:right]
    
    def crop_file(self, input_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """Crop an image file"""
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        
        cropped = self.crop(img)
        
        if output_path:
            cv2.imwrite(output_path, cropped)
            
        return cropped
    
    @staticmethod
    def remove_uniform_borders(img: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Remove black bars and uniform borders"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        top = 0
        for i in range(h):
            if gray[i].std() > threshold:
                top = i
                break
        
        bottom = h
        for i in range(h-1, -1, -1):
            if gray[i].std() > threshold:
                bottom = i + 1
                break
        
        left = 0
        for i in range(w):
            if gray[:, i].std() > threshold:
                left = i
                break
        
        right = w
        for i in range(w-1, -1, -1):
            if gray[:, i].std() > threshold:
                right = i + 1
                break
        
        return img[top:bottom, left:right]


def run_calibration(image_dir: str, platform: str = 'tradingview', num_samples: int = 5):
    """
    Run interactive calibration
    
    Args:
        image_dir: Directory containing sample screenshots
        platform: 'tradingview' or 'tradovate'
        num_samples: Number of images to calibrate with
    """
    image_dir = Path(image_dir)
    
    # Find images
    image_extensions = ['.png', '.jpg', '.jpeg']
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f'*{ext}'))
    
    if not images:
        print(f"❌ No images found in {image_dir}")
        return None
    
    print(f"Found {len(images)} images")
    
    # Sample random images if too many
    if len(images) > num_samples:
        images = np.random.choice(images, num_samples, replace=False)
        print(f"Using {num_samples} random samples for calibration")
    
    # Run calibration
    calibrator = ImageCropCalibrator()
    
    for img_path in images:
        calibrator.calibrate_image(img_path)
    
    if not calibrator.crop_regions:
        print("❌ No calibration data collected")
        return None
    
    # Compute config
    config = calibrator.compute_config(platform)
    
    # Save config
    config_path = f'crop_config_{platform}.json'
    config.save(config_path)
    print(f"\n💾 Saved configuration to {config_path}")
    
    return config


def batch_crop(input_dir: str, output_dir: str, config_path: str):
    """
    Apply cropping to all images in a directory
    
    Args:
        input_dir: Source directory
        output_dir: Destination directory
        config_path: Path to calibration config JSON
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load config
    config = CropConfig.load(config_path)
    cropper = SmartCropper(config)
    
    # Find images
    image_extensions = ['.png', '.jpg', '.jpeg']
    images = []
    for ext in image_extensions:
        images.extend(input_dir.glob(f'*{ext}'))
    
    print(f"Processing {len(images)} images...")
    
    for img_path in images:
        try:
            output_path = output_dir / img_path.name
            cropper.crop_file(str(img_path), str(output_path))
            print(f"✓ {img_path.name}")
        except Exception as e:
            print(f"❌ {img_path.name}: {e}")
    
    print(f"\n✅ Done! Cropped images saved to {output_dir}")


# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Calibration: python script.py calibrate <image_dir> [platform] [num_samples]")
        print("  Batch crop:  python script.py crop <input_dir> <output_dir> <config.json>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'calibrate':
        image_dir = sys.argv[2]
        platform = sys.argv[3] if len(sys.argv) > 3 else 'tradingview'
        num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        
        run_calibration(image_dir, platform, num_samples)
        
    elif command == 'crop':
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        config_path = sys.argv[4]
        
        batch_crop(input_dir, output_dir, config_path)
    
    else:
        print(f"Unknown command: {command}")
