import os
import sys
import json
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def img_to_base64(path, max_w=400):
    try:
        img = Image.open(path)
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        return None

def generate_tiered_report(json_path, image_dir, output_html):
    with open(json_path, 'r') as f:
        data = json.load(f)

    json_path = Path(json_path)
    image_dir = Path(image_dir)
    
    html = ["""
<!DOCTYPE html>
<html>
<head>
    <title>Deduplication Tier Review</title>
    <style>
        body { font-family: sans-serif; background: #f0f0f0; margin: 20px; }
        .section { background: white; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .card { border: 1px solid #ddd; padding: 10px; border-radius: 4px; background: #fff; }
        .card img { max-width: 100%; height: auto; border: 1px solid #eee; }
        .tier1 { border-left: 5px solid #ff5252; }
        .tier2 { border-left: 5px solid #ffd740; }
        .info { font-size: 0.85em; margin-top: 8px; color: #666; }
        .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 0.8em; color: white; }
        .badge-1 { background: #ff5252; }
        .badge-2 { background: #ffd740; color: #333; }
        .stats { margin-bottom: 20px; padding: 15px; background: #e3f2fd; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Deduplication Experimental Verification</h1>
    """]

    html.append(f"""
    <div class="stats">
        <strong>Total Images:</strong> {data['total_images']} | 
        <strong>Auto-Discard (Tier 1):</strong> {data['auto_discarded_count']} | 
        <strong>Ambiguous (Tier 2):</strong> {data['ambiguous_count']} | 
        <strong>Unique:</strong> {data['unique_count']}
    </div>
    """)

    # Tier 1 Sections
    html.append('<div class="section"><h2>Tier 1: Auto-Discard (High Confidence Duplicates)</h2><div class="grid">')
    # Limit to 50 for sample if too many, but user wants verification. 
    # Let's show first 100 Tier 1 to save space but give confidence.
    for item in data['auto_discarded'][:100]:
        img_path = image_dir / item['image']
        match_path = image_dir / item['matched_with']
        b64 = img_to_base64(img_path)
        if b64:
            html.append(f"""
            <div class="card tier1">
                <img src="{b64}">
                <div class="info">
                    <strong>{item['image']}</strong><br>
                    Matched with: {item['matched_with']}<br>
                    Dist: {item['dist']} | Hist Sim: {item['hist_sim']:.4f}
                </div>
            </div>
            """)
    html.append('</div></div>')

    # Tier 2 Section
    html.append('<div class="section"><h2>Tier 2: Ambiguous (Remaining for Review)</h2><div class="grid">')
    for item in data['ambiguous'][:100]: # Show first 100
        img_path = image_dir / item['image']
        b64 = img_to_base64(img_path)
        if b64:
            html.append(f"""
            <div class="card tier2">
                <img src="{b64}">
                <div class="info">
                    <strong>{item['image']}</strong><br>
                    Matched with: {item['matched_with']}<br>
                    Dist: {item['dist']} | Hist Sim: {item['hist_sim']:.4f}
                </div>
            </div>
            """)
    html.append('</div></div>')

    html.append("</body></html>")

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write("".join(html))
    print(f"Tiered report generated: {output_html}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python experiment_visual_verification.py <json_path> <image_dir> <output_html>")
        sys.exit(1)
    
    generate_tiered_report(sys.argv[1], sys.argv[2], sys.argv[3])
