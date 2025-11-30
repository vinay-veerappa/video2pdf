import os
import argparse
from analyzer import analyze_images_comprehensive

def run_analysis(images_dir, output_dir):
    print(f"Running comprehensive analysis on {images_dir}...")
    
    # 1. Run Grid SSIM Analysis
    print("\n--- GRID SSIM ANALYSIS ---")
    grid_result = analyze_images_comprehensive(
        images_dir,
        similarity_threshold=0.8,
        output_folder=output_dir,
        move_duplicates=False,
        similarity_method='grid'
    )
    
    # 2. Run CNN Analysis
    print("\n--- CNN ANALYSIS ---")
    cnn_result = analyze_images_comprehensive(
        images_dir,
        similarity_threshold=0.95, # CNN needs higher threshold
        output_folder=output_dir,
        move_duplicates=False,
        similarity_method='cnn'
    )
    
    # Print Irrelevant Images (from Grid run, should be same for both)
    if grid_result and grid_result['irrelevant']:
        print("\n" + "="*60)
        print("FLAGGED IRRELEVANT IMAGES:")
        print("="*60)
        for img in grid_result['irrelevant']:
            print(f"- {img['filename']} (Reason: {img['reason']})")
    
    # Compare Results
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)
    print(f"Grid SSIM found {len(grid_result['duplicates'])} duplicate groups")
    print(f"CNN found {len(cnn_result['duplicates'])} duplicate groups")
    
    # Save consolidated report
    report_path = os.path.join(output_dir, "consolidated_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("CONSOLIDATED ANALYSIS REPORT\n")
        f.write("============================\n\n")
        
        f.write("IRRELEVANT IMAGES:\n")
        if grid_result['irrelevant']:
            for img in grid_result['irrelevant']:
                f.write(f"- {img['filename']} ({img['reason']})\n")
        else:
            f.write("None found.\n")
            
        f.write("\nDUPLICATES (GRID SSIM):\n")
        f.write(f"Found {len(grid_result['duplicates'])} groups.\n")
        
        f.write("\nDUPLICATES (CNN):\n")
        f.write(f"Found {len(cnn_result['duplicates'])} groups.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir")
    parser.add_argument("--output", default=None)
    # Method arg removed as we run both
    args = parser.parse_args()
    
    out = args.output if args.output else os.path.dirname(args.images_dir)
    run_analysis(args.images_dir, out)
