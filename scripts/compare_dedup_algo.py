import json
import os
import sys
from pathlib import Path

def compare_results(old_json_path, new_json_path, image_dir, output_html):
    # Load Old Results
    with open(old_json_path, 'r') as f:
        old_data = json.load(f)
    
    # Identify Old Kept Images
    old_kept = set()
    if isinstance(old_data, list):
        old_kept = set(old_data)
    elif isinstance(old_data, dict):
        # Try common keys
        if 'duplicates' in old_data and 'all_files' in old_data:
            duplicates_list = old_data['duplicates']
            duplicates = set()
            for d in duplicates_list:
                if isinstance(d, dict):
                    duplicates.add(d.get('image2'))
                else:
                    duplicates.add(d)
                    
            blanks = set(old_data.get('blanks', []))
            all_files = set(old_data['all_files'])
            old_kept = all_files - duplicates - blanks
        else:
            for key in ['unique_images', 'images', 'kept_images']:
                if key in old_data:
                    old_kept = set(old_data[key])
                    break
    
    # Load New Results
    with open(new_json_path, 'r') as f:
        new_data = json.load(f)
    
    # New Logic: Kept = All Images - AutoDiscarded
    # Or Kept = Unique + Ambiguous
    # We need to know all images to reconstruct "Unique". 
    # But wait, checking the 'ambiguous' and 'auto_discarded' lists might be enough to classify.
    
    new_auto_discarded = set([x['image'] for x in new_data.get('auto_discarded', [])])
    new_ambiguous = set([x['image'] for x in new_data.get('ambiguous', [])])
    
    # Scan directory to get full list
    image_dir = Path(image_dir)
    all_images = sorted([p.name for p in image_dir.glob("*.png")])
    
    new_kept = set()
    new_unique = set()
    
    for img in all_images:
        if img in new_auto_discarded:
            continue # Discarded
        new_kept.add(img)
        if img not in new_ambiguous:
            new_unique.add(img)

    # Calculate Deltas
    # 1. Images OLD kept but NEW discarded (Efficiency Gain)
    old_only = old_kept - new_kept
    
    # 2. Images NEW kept but OLD discarded (Safety Net / Potential missed content in old)
    new_only = new_kept - old_kept
    
    # 3. Intersection
    both = old_kept.intersection(new_kept)

    print(f"Total Images: {len(all_images)}")
    print(f"Old Algorithm Kept: {len(old_kept)}")
    print(f"New Algorithm Kept: {len(new_kept)} (Unique: {len(new_unique)}, Ambiguous: {len(new_ambiguous)})")
    
    print(f"Efficiency Gain (Old kept, New discarded): {len(old_only)}")
    print(f"Safety Catch (New kept, Old discarded): {len(new_only)}")
    
    # Generate HTML Report
    generate_html(old_only, new_only, new_data, image_dir, output_html)

def generate_html(old_only, new_only, new_data, image_dir, output_path):
    html = """
    <html>
    <head>
        <style>
            body { font-family: sans-serif; padding: 20px; background: #eef; }
            .section { margin-bottom: 40px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h2 { border-bottom: 2px solid #ddd; padding-bottom: 10px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
            .card { border: 1px solid #ccc; padding: 10px; border-radius: 4px; background: #fff; }
            img { max-width: 100%; border: 1px solid #ddd; }
            .meta { font-size: 0.85em; color: #555; margin-top: 5px; }
            .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; font-weight: bold; }
            .badge-gain { background: #d4edda; color: #155724; }
            .badge-safety { background: #fff3cd; color: #856404; }
        </style>
    </head>
    <body>
        <h1>Deduplication Comparison</h1>
        
        <div class="section">
            <h2>Efficiency Gain: Images OLD kept but NEW discarded ({len(old_only)})</h2>
            <p>These are images the old algorithm thought were unique, but the New Algorithm flagged as duplicates (and auto-discarded).</p>
            <div class="grid">
    """
    
    # Helper to find match reason from new data
    discard_map = {x['image']: x for x in new_data.get('auto_discarded', [])}
    
    count = 0
    for img_name in sorted(list(old_only)):
        if count > 50: break # Limit
        reason = discard_map.get(img_name, {})
        match_name = reason.get('matched_with', 'Unknown')
        
        html += f"""
            <div class="card">
                <div class="badge badge-gain">Efficiency Gain</div>
                <div><strong>{img_name}</strong></div>
                <img src="../../../../output/Bootcamp Classroom - Week 7 Day 4 - Cody Market Structure - Copy/images/{img_name}" loading="lazy">
                <div class="meta">
                    Detected as duplicate of: <strong>{match_name}</strong><br>
                    Dist: {reason.get('dist')}, Hist: {reason.get('hist_sim'):.4f}
                </div>
            </div>
        """
        count += 1
        
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2>Safety Net: Images NEW kept but OLD discarded ({len(new_only)})</h2>
            <p>These are images the old algorithm discarded, but the New Algorithm flagged as Ambiguous (Needs Review) or Unique.</p>
            <div class="grid">
    """
    
    count = 0
    for img_name in sorted(list(new_only)):
        if count > 50: break
        
        html += f"""
            <div class="card">
                <div class="badge badge-safety">Safety Catch</div>
                <div><strong>{img_name}</strong></div>
                <img src="../../../../output/Bootcamp Classroom - Week 7 Day 4 - Cody Market Structure - Copy/images/{img_name}" loading="lazy">
            </div>
        """
        count += 1
        
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html.replace("{len(old_only)}", str(len(old_only))).replace("{len(new_only)}", str(len(new_only))))
    
    print(f"Comparison report saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python compare_dedup_algo.py <old_json> <new_json> <image_dir> <output_html>")
        sys.exit(1)
        
    compare_results(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
