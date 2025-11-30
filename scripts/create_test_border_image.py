import cv2
import numpy as np
import os

def create_test_image():
    # Create a 400x400 white square (content)
    content = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw some "content"
    cv2.circle(content, (200, 200), 100, (0, 0, 255), -1)
    
    # Create a 600x600 black background
    background = np.zeros((600, 600, 3), dtype=np.uint8)
    
    # Place content in the center (100px border on all sides)
    background[100:500, 100:500] = content
    
    output_path = "scripts/test_border.png"
    cv2.imwrite(output_path, background)
    print(f"Created test image with borders: {output_path} (600x600)")

if __name__ == "__main__":
    create_test_image()
