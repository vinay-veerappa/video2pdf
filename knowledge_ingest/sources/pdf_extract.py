"""
PDF -> text front stage.

Your collection is a MIX of text-based and scanned PDFs, so this does NOT assume
— it inspects each PDF, extracts text where a text layer exists, and FLAGS scanned
PDFs (no font/text layer) for the image/OCR pipeline instead of silently emitting
garbage. Text PDFs become .md files in an input dir, ready for the normal pipeline
(which now segments prose sources on headers).

Per the pdf-reading skill:
- pdffonts empty  -> scanned/raster -> route to OCR/image pipeline (flagged here)
- pdffonts present -> pdftotext -layout (best for multi-column) or pdfplumber

Requires: poppler-utils (pdftotext, pdffonts, pdfinfo) and/or pdfplumber.
    pip install pdfplumber
    (poppler-utils: apt-get install poppler-utils  /  brew install poppler)

Usage:
    python -m knowledge_ingest.sources.pdf_extract \
        --in  "C:\\path\\to\\pdfs" \
        --out "C:\\path\\to\\text_input" \
        --scanned-list scanned_pdfs.txt
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess


def has_text_layer(pdf_path: str) -> bool:
    """True if pdffonts reports any fonts (i.e., a text layer exists)."""
    try:
        out = subprocess.run(["pdffonts", pdf_path], capture_output=True,
                             text=True, timeout=60).stdout
        # header is 2 lines; any line beyond = a font present
        return len(out.strip().splitlines()) > 2
    except (FileNotFoundError, subprocess.SubprocessError):
        return _pdfplumber_has_text(pdf_path)


def _pdfplumber_has_text(pdf_path: str) -> bool:
    """Fallback text-layer check if poppler isn't installed."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:
                if (page.extract_text() or "").strip():
                    return True
        return False
    except Exception:
        return False


def extract_text(pdf_path: str) -> str:
    """Extract text, preferring pdftotext -layout, falling back to pdfplumber."""
    try:
        out = subprocess.run(["pdftotext", "-layout", pdf_path, "-"],
                             capture_output=True, text=True, timeout=300)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    # fallback
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n\n".join(parts)
    except Exception as e:
        return ""


def page_has_images(pdf_path: str, page_num: int) -> bool:
    """True if a page has raster images or is image-dominant (mixed-PDF routing)."""
    try:
        out = subprocess.run(["pdfimages", "-list", "-f", str(page_num), "-l",
                              str(page_num), pdf_path],
                             capture_output=True, text=True, timeout=60).stdout
        # header is 2 lines; any image row beyond = images present
        return len(out.strip().splitlines()) > 2
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def page_text(pdf_path: str, page_num: int) -> str:
    try:
        out = subprocess.run(["pdftotext", "-layout", "-f", str(page_num),
                              "-l", str(page_num), pdf_path, "-"],
                             capture_output=True, text=True, timeout=120)
        return out.stdout if out.returncode == 0 else ""
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


def rasterize_page(pdf_path: str, page_num: int, out_dir: str, stem: str) -> str:
    """Render one page to an image for the chart-extract path. Returns image path."""
    prefix = os.path.join(out_dir, f"{stem}_p{page_num}")
    try:
        subprocess.run(["pdftoppm", "-jpeg", "-r", "150", "-f", str(page_num),
                        "-l", str(page_num), pdf_path, prefix],
                       capture_output=True, timeout=120)
        # pdftoppm zero-pads; find the actual file
        matches = glob.glob(prefix + "*.jpg")
        return matches[0] if matches else ""
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


def page_count(pdf_path: str) -> int:
    try:
        out = subprocess.run(["pdfinfo", pdf_path], capture_output=True,
                             text=True, timeout=60).stdout
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def extract_mixed(pdf_path, text_out_dir, image_out_dir):
    """Per-page router for mixed PDFs (and Slides-exported-as-PDF).
    Text pages -> appended to one .md; image-bearing pages -> rasterized to the
    chart images folder for the chart-extract path."""
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    n = page_count(pdf_path)
    if n == 0:
        return 0, 0
    text_parts, img_pages = [], 0
    for pg in range(1, n + 1):
        txt = page_text(pdf_path, pg)
        if txt.strip():
            text_parts.append(f"\n\n<!-- page {pg} -->\n{txt}")
        # a page with meaningful images -> also rasterize for chart-extract
        if page_has_images(pdf_path, pg):
            img = rasterize_page(pdf_path, pg, image_out_dir, stem)
            if img:
                img_pages += 1
    if text_parts:
        with open(os.path.join(text_out_dir, f"{stem}.md"), "w", encoding="utf-8") as f:
            f.write(f"<!-- source_type: pdf | source_file: {os.path.basename(pdf_path)} -->\n")
            f.write("".join(text_parts))
    return len(text_parts), img_pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--scanned-list", default="scanned_pdfs.txt",
                    help="where to write the list of scanned PDFs needing OCR")
    ap.add_argument("--mixed", action="store_true",
                    help="per-page routing: text pages -> text, image pages -> chart images folder")
    ap.add_argument("--image-out", default=None,
                    help="folder for rasterized image-pages (default: <out>/_chart_images)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(args.in_dir, "*.pdf")))
    print(f"Found {len(pdfs)} PDFs in {args.in_dir}")

    if args.mixed:
        img_dir = args.image_out or os.path.join(args.out_dir, "_chart_images")
        os.makedirs(img_dir, exist_ok=True)
        tot_text, tot_img = 0, 0
        for p in pdfs:
            stem = os.path.splitext(os.path.basename(p))[0]
            tp, ip = extract_mixed(p, args.out_dir, img_dir)
            tot_text += tp; tot_img += ip
            print(f"  [mixed] {stem}: {tp} text pages, {ip} image pages rasterized")
        print(f"\nDone (mixed). text pages: {tot_text} | image pages: {tot_img} -> {img_dir}")
        print(f"Next: run pipeline --source-type pdf on {args.out_dir} for the text;")
        print(f"      run chart_extract propose --images {img_dir} for the image pages.")
        return

    _run_text_only(args, pdfs)


def _run_text_only(args, pdfs):
    text_ok, scanned, empty = 0, [], 0
    for p in pdfs:
        stem = os.path.splitext(os.path.basename(p))[0]
        if not has_text_layer(p):
            scanned.append(p)
            print(f"  [scanned]  {stem}  -> flagged for OCR/image pipeline")
            continue
        text = extract_text(p)
        if not text.strip():
            empty += 1
            print(f"  [empty!]   {stem}  -> text layer claimed but extraction empty; check manually")
            continue
        out_fp = os.path.join(args.out_dir, f"{stem}.md")
        with open(out_fp, "w", encoding="utf-8") as f:
            f.write(f"<!-- source_type: pdf | source_file: {os.path.basename(p)} -->\n\n")
            f.write(text)
        text_ok += 1
        print(f"  [text ok]  {stem}  -> {os.path.basename(out_fp)}")

    if scanned:
        with open(args.scanned_list, "w", encoding="utf-8") as f:
            f.write("\n".join(scanned) + "\n")

    print(f"\nDone. text-extracted: {text_ok} | scanned (need OCR): {len(scanned)}"
          f" | empty/problem: {empty}")
    if scanned:
        print(f"Scanned PDFs listed in {args.scanned_list} — these go to the "
              f"chart/OCR pipeline (use --mixed or chart_extract), NOT the text pipeline.")
    print(f"\nNext: run the pipeline with --source-type pdf pointed at {args.out_dir}")


if __name__ == "__main__":
    main()
