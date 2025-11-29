from imagededup.methods import PHash, CNN
import sys
import os
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_imagededup_directory(image_dir):
    print(f"Analyzing directory: {image_dir}")
    
    # Get all png images
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images. Comparing consecutive pairs...\n")
    
    # Initialize methods
    phasher = PHash()
    try:
        cnn = CNN()
        cnn_available = True
    except Exception as e:
        print(f"CNN initialization failed: {e}")
        cnn_available = False

    print(f"{'Pair':<30} | {'PHash Dist':<12} | {'CNN Sim':<10} | {'Verdict'}")
    print("-" * 70)

    for i in range(len(images) - 1):
        img1_path = images[i]
        img2_path = images[i+1]
        name1 = os.path.basename(img1_path)
        name2 = os.path.basename(img2_path)
        pair_name = f"{name1[:10]}.. vs {name2[:10]}.."
        
        # PHash
        try:
            hash1 = phasher.encode_image(image_file=img1_path)
            hash2 = phasher.encode_image(image_file=img2_path)
            dist_phash = phasher.hamming_distance(hash1, hash2)
        except Exception as e:
            dist_phash = "Err"

        # CNN
        sim_cnn = "N/A"
        if cnn_available:
            try:
                emb1 = cnn.encode_image(image_file=img1_path)
                emb2 = cnn.encode_image(image_file=img2_path)
                
                if isinstance(emb1, np.ndarray):
                    emb1 = emb1.reshape(1, -1)
                    emb2 = emb2.reshape(1, -1)
                
                sim = cosine_similarity(emb1, emb2)[0][0]
                sim_cnn = f"{sim:.4f}"
            except Exception as e:
                sim_cnn = "Err"
        
        # Verdict logic (just for display)
        verdict = ""
        if isinstance(dist_phash, int) and dist_phash <= 5:
            verdict += "PHash-Dup "
        if sim_cnn != "N/A" and sim_cnn != "Err" and float(sim_cnn) > 0.95:
            verdict += "CNN-Dup"
            
        print(f"{pair_name:<30} | {str(dist_phash):<12} | {sim_cnn:<10} | {verdict}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_imagededup.py <image_directory>")
    else:
        test_imagededup_directory(sys.argv[1])
