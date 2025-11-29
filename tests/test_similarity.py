import sys
import os
import cv2
import numpy as np
import unittest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similarity import calculate_similarity, dhash, calculate_hamming_distance

class TestSimilarity(unittest.TestCase):
    def setUp(self):
        # Create dummy images
        self.img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Add some noise to img2
        cv2.randn(self.img2, 0, 10)
        self.img2 = cv2.add(self.img1, self.img2)
        
        # Create a different image
        self.img3 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.img3, (20, 20), (80, 80), (255, 255, 255), -1)

    def test_ssim_identical(self):
        sim = calculate_similarity(self.img1, self.img1, method='ssim')
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_ssim_different(self):
        sim = calculate_similarity(self.img1, self.img3, method='ssim')
        self.assertLess(sim, 0.9)

    def test_dhash_identical(self):
        h1 = dhash(self.img1)
        h2 = dhash(self.img1)
        dist = calculate_hamming_distance(h1, h2)
        self.assertEqual(dist, 0)

    def test_dhash_different(self):
        h1 = dhash(self.img1)
        h3 = dhash(self.img3)
        dist = calculate_hamming_distance(h1, h3)
        self.assertGreater(dist, 0)

if __name__ == '__main__':
    unittest.main()
