import os
import time
import subprocess
import shutil
import glob

def run_benchmark(video_path, output_base="benchmark_output"):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    print(f"Benchmarking extraction on: {video_path}")
    print("-" * 60)
    
    methods = [
        {
            "name": "Baseline (Fixed FPS=1)",
            "cmd": [
                "ffmpeg", "-hwaccel", "cuda", "-i", video_path, 
                "-vf", "fps=1,scale='min(1920,iw)':-2", "-vsync", "0", 
                "-q:v", "2", "-compression_level", "1",
                os.path.join(output_base, "baseline", "%05d.png")
            ]
        },
        {
            "name": "I-Frame Select (pict_type==I)",
            "cmd": [
                "ffmpeg", "-hwaccel", "cuda", "-i", video_path, 
                "-vf", "select='eq(pict_type,I)',scale='min(1920,iw)':-2", "-vsync", "vfr", 
                "-q:v", "2", "-compression_level", "1",
                os.path.join(output_base, "iframe", "%05d.png")
            ]
        },
        {
            "name": "Skip Non-Key (Fastest Decoding)",
            "cmd": [
                "ffmpeg", "-skip_frame", "nokey", "-hwaccel", "cuda", "-i", video_path, 
                "-vsync", "vfr", "-q:v", "2", "-compression_level", "1",
                os.path.join(output_base, "skipkey", "%05d.png")
            ]
        }
    ]
    
    results = []
    
    for method in methods:
        name = method["name"]
        cmd = method["cmd"]
        out_dir = os.path.dirname(cmd[-1])
        
        # Cleanup
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Running: {name}...")
        start_time = time.time()
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            duration = time.time() - start_time
            count = len(glob.glob(os.path.join(out_dir, "*.png")))
            
            print(f"  -> Done in {duration:.2f}s. Extracted {count} frames.")
            results.append({"name": name, "time": duration, "count": count})
            
        except subprocess.CalledProcessError as e:
            print(f"  -> Failed: {e}")
            results.append({"name": name, "time": -1, "count": 0})
            
    print("-" * 60)
    print(f"{'Method':<30} | {'Time (s)':<10} | {'Frames':<10} | {'FPS (Processing)':<15}")
    print("-" * 60)
    
    for r in results:
        if r["time"] > 0:
            fps = r["count"] / r["time"]
            print(f"{r['name']:<30} | {r['time']:<10.2f} | {r['count']:<10} | {fps:<15.2f}")
        else:
            print(f"{r['name']:<30} | {'FAILED':<10} | {'0':<10} | {'-':<15}")
            
    # Cleanup all
    # shutil.rmtree(output_base)

if __name__ == "__main__":
    # Default to Test Video 1 if available
    default_video = os.path.join("input", "Test Video 1.mp4")
    if os.path.exists(default_video):
        run_benchmark(default_video)
    else:
        print("Please provide a video path.")
