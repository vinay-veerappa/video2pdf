"""
MinerU PDF extraction integration.

MinerU (https://github.com/opendatalab/MinerU) extracts text, images, and
structured content from PDFs. This module provides a clean interface to:

1. Run MinerU on a PDF → get markdown + extracted images
2. Route extracted content through the knowledge_ingest pipeline
3. Combine with existing chart + transcript knowledge bases

Usage:
  python -m knowledge_ingest.mineru_integration --pdf "C:\\path\\to\\file.pdf"
  python -m knowledge_ingest.mineru_integration --pdf-dir "C:\\path\\to\\pdfs" --batch
"""

import os, sys, subprocess, argparse, json, glob, shutil, time
from pathlib import Path

MINERU_VENV = r"C:\Users\vinay\mineru_venv"
MINERU_PYTHON = os.path.join(MINERU_VENV, "Scripts", "python.exe")

# Output base for MinerU extractions
MINERU_OUTPUT_BASE = r"C:\ICT_Videos\Testing\_mineru_output"

# Where to put extracted images for v4 chart extraction
MINERU_IMAGE_DIR = r"C:\ICT_Videos\Testing\_mineru_images"

# Where to put extracted markdown for text ingestion
MINERU_TEXT_DIR = r"C:\ICT_Videos\Testing\_mineru_text"


def check_mineru():
    """Verify MinerU is installed in the venv."""
    if not os.path.exists(MINERU_PYTHON):
        print(f"ERROR: MinerU venv not found at {MINERU_VENV}")
        print("Install with: python -m venv C:\\Users\\vinay\\mineru_venv")
        print("Then: C:\\Users\\vinay\\mineru_venv\\Scripts\\pip install magic-pdf[full]")
        return False

    # Check if magic_pdf is installed
    result = subprocess.run(
        [MINERU_PYTHON, "-c", "import magic_pdf; print(magic_pdf.__version__)"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        print(f"ERROR: magic_pdf not installed in {MINERU_VENV}")
        print("Install with: C:\\Users\\vinay\\mineru_venv\\Scripts\\pip install magic-pdf[full]")
        return False

    print(f"MinerU ready: magic_pdf v{result.stdout.strip()}")
    return True


def run_mineru(pdf_path, output_dir=None, image_dpi=200):
    """
    Run MinerU on a single PDF.
    Returns (markdown_path, images_dir) or (None, None) on failure.
    """
    pdf_path = str(pdf_path)
    pdf_name = Path(pdf_path).stem

    if output_dir is None:
        output_dir = os.path.join(MINERU_OUTPUT_BASE, pdf_name)

    os.makedirs(output_dir, exist_ok=True)

    # MinerU CLI: magic-pdf -p <pdf> -o <output> -m auto
    cmd = [
        MINERU_PYTHON, "-m", "magic_pdf.cli",
        "-p", pdf_path,
        "-o", output_dir,
        "-m", "auto",  # auto mode (OCR + text)
    ]

    print(f"Running MinerU on: {pdf_path}")
    print(f"  Output: {output_dir}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        return None, None

    print(f"  Done ({elapsed:.1f}s)")

    # Find the output markdown and images
    # MinerU creates: <output>/<name>/auto/<name>.md and images in <output>/<name>/auto/images/
    auto_dir = os.path.join(output_dir, pdf_name, "auto")
    md_path = os.path.join(auto_dir, f"{pdf_name}.md")
    images_dir = os.path.join(auto_dir, "images")

    if not os.path.exists(md_path):
        # Try alternative layout
        md_files = glob.glob(os.path.join(output_dir, "**", "*.md"), recursive=True)
        if md_files:
            md_path = md_files[0]
            images_dir = os.path.join(os.path.dirname(md_path), "images")
        else:
            print(f"  WARNING: no markdown found in {output_dir}")
            return None, None

    return md_path, images_dir


def collect_images_for_v4(images_dir, dest_dir, pdf_name):
    """Copy extracted images to a flat dir for v4 chart extraction."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for img in glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")):
        dest = os.path.join(dest_dir, f"{pdf_name}_{Path(img).stem}{Path(img).suffix}")
        shutil.copy2(img, dest)
        copied += 1
    return copied


def collect_text_for_ingest(md_path, dest_dir, pdf_name):
    """Copy markdown to a flat dir for text ingestion."""
    os.makedirs(dest_dir, exist_ok=True)
    # Convert .md to .txt for the ingest pipeline (it expects .txt)
    dest = os.path.join(dest_dir, f"{pdf_name}.txt")
    shutil.copy2(md_path, dest)
    return dest


def process_pdf(pdf_path, run_v4=False, run_text=False):
    """Full pipeline: MinerU extract → route images/text."""
    md_path, images_dir = run_mineru(pdf_path)
    if not md_path:
        return

    pdf_name = Path(pdf_path).stem
    n_images = 0
    text_path = None

    if images_dir and os.path.exists(images_dir):
        n_images = collect_images_for_v4(images_dir, MINERU_IMAGE_DIR, pdf_name)
        print(f"  Collected {n_images} images → {MINERU_IMAGE_DIR}")

    text_path = collect_text_for_ingest(md_path, MINERU_TEXT_DIR, pdf_name)
    print(f"  Text → {text_path}")

    if run_v4 and n_images > 0:
        print(f"\nTo extract chart knowledge from {n_images} images, run:")
        print(f'  python -m knowledge_ingest.tests.run_v4_full '
              f'--out-dir "C:\\ICT_Videos\\Testing\\_mineru_v4" '
              f'--filter "{pdf_name}_*"')

    if run_text and text_path:
        print(f"\nTo ingest text knowledge, run:")
        print(f'  python -m knowledge_ingest.run '
              f'--input "{MINERU_TEXT_DIR}" '
              f'--output "C:\\ICT_Videos\\Testing\\_mineru_text_ingest" '
              f'--file-filter "{pdf_name}.txt" --ict-aware --no-skip')

    return {"pdf": pdf_path, "markdown": md_path, "images": n_images, "text": text_path}


def main():
    ap = argparse.ArgumentParser(description="MinerU PDF → knowledge pipeline")
    ap.add_argument("--pdf", help="single PDF to process")
    ap.add_argument("--pdf-dir", help="directory of PDFs to batch process")
    ap.add_argument("--batch", action="store_true", help="batch mode for --pdf-dir")
    ap.add_argument("--run-v4", action="store_true", help="print v4 chart extraction command")
    manual = ap.add_argument("--run-text", action="store_true", help="print text ingest command")
    ap.add_argument("--check", action="store_true", help="just check MinerU installation")
    args = ap.parse_args()

    if args.check:
        check_mineru()
        return

    if not check_mineru():
        return

    results = []

    if args.pdf:
        r = process_pdf(args.pdf, run_v4=args.run_v4, run_text=args.run_text)
        if r:
            results.append(r)

    if args.pdf_dir and args.batch:
        pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
        print(f"\nFound {len(pdfs)} PDFs in {args.pdf_dir}")
        for pdf in pdfs:
            r = process_pdf(pdf, run_v4=args.run_v4, run_text=args.run_text)
            if r:
                results.append(r)
            print()

    if results:
        report_path = os.path.join(MINERU_OUTPUT_BASE, "mineru_report.json")
        os.makedirs(MINERU_OUTPUT_BASE, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n{'='*60}")
        print(f"MinerU extraction complete: {len(results)} PDFs")
        print(f"Report: {report_path}")
        print(f"Images: {MINERU_IMAGE_DIR}")
        print(f"Text:   {MINERU_TEXT_DIR}")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"  1. Run v4 chart extraction on images:")
        print(f'     python -m knowledge_ingest.tests.run_v4_full --out-dir "C:\\ICT_Videos\\Testing\\_mineru_v4" --filter "*_page*"')
        print(f"  2. Run ICT-aware text ingestion:")
        print(f'     python -m knowledge_ingest.run --input "{MINERU_TEXT_DIR}" --output "C:\\ICT_Videos\\Testing\\_mineru_text_ingest" --ict-aware --no-skip')
        print(f"  3. Merge all knowledge bases:")
        print(f'     python -m knowledge_ingest.merge_knowledge_base --transcript-dir "C:\\ICT_Videos\\Testing\\_mineru_text_ingest\\units"')
        print(f"\n{'='*60}")
    else:
        print("No PDFs processed.")
        print(f"\nUsage:")
        print(f"  python -m knowledge_ingest.mineru_integration --pdf <path>")
        print(f"  python -m knowledge_ingest.mineru_integration --pdf-dir <dir> --batch")
        print(f"  python -m knowledge_ingest.mineru_integration --check")


if __name__ == "__main__":
    main()