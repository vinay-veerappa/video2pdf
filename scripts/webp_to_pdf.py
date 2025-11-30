import os
import glob
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import argparse

def webp_to_pdf(input_dir, output_pdf):
    """Convert WebP images to PDF"""
    # Get all WebP files
    webp_files = sorted(glob.glob(os.path.join(input_dir, "*.webp")))
    
    if not webp_files:
        print("No WebP files found")
        return
    
    print(f"Found {len(webp_files)} WebP images")
    
    # Get dimensions from first image to set page size
    first_img = Image.open(webp_files[0])
    img_width, img_height = first_img.size
    
    # Create PDF with appropriate page size
    # Scale to fit letter size while maintaining aspect ratio
    page_width, page_height = letter
    scale = min(page_width / img_width, page_height / img_height)
    
    c = canvas.Canvas(output_pdf, pagesize=(img_width * scale, img_height * scale))
    
    for i, webp_file in enumerate(webp_files):
        print(f"Processing {os.path.basename(webp_file)} ({i+1}/{len(webp_files)})...", end='\r')
        
        img = Image.open(webp_file)
        w, h = img.size
        
        # Draw image on canvas
        c.drawInlineImage(img, 0, 0, width=w*scale, height=h*scale)
        c.showPage()
    
    c.save()
    print(f"\nPDF created: {output_pdf}")
    
    # Get file sizes
    total_webp_size = sum(os.path.getsize(f) for f in webp_files)
    pdf_size = os.path.getsize(output_pdf)
    
    print(f"Total WebP size: {total_webp_size / 1024 / 1024:.2f} MB")
    print(f"PDF size: {pdf_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing WebP images")
    parser.add_argument("--output", help="Output PDF path")
    args = parser.parse_args()
    
    if args.output:
        output_pdf = args.output
    else:
        output_pdf = os.path.join(os.path.dirname(args.input_dir), "slides.pdf")
    
    webp_to_pdf(args.input_dir, output_pdf)
