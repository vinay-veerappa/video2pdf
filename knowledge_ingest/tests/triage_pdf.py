"""
PDF triage tool — Phase 0 of the image/PDF plan (HANDOVER §17a).

Structural categorization of every page in one or more PDFs WITHOUT calling any
VLM/LLM. For each page it reports:
  - text_char_count   : chars of extractable text
  - image_count       : number of embedded raster images
  - image_area_ratio  : fraction of page area covered by images
  - page_dominance    : text_dominant | image_dominant | co_dependent | near_empty
                        (the 3-way classifier from §11d/§12b, applied by cheap heuristic)
  - has_likely_table  : heuristic — many short text lines aligned in columns
  - render_path       : path to a rendered PNG of the page (for the bake-off / VLM)

The output feeds the calibration set growth (§11c/§13g): pages tagged
image_dominant / co_dependent become candidates for the VLM bake-off and the
page-dominance-classifier calibration.

Why no VLM here: the bulk of real PDFs are text-dominant (§12c found Flux was
~60% text). A structural pass routes those to the existing text pipeline for free
and surfaces only the pages that actually need vision. Vision is the expensive
tool — use it only where structure can't decide.

Usage:
    python knowledge_ingest/tests/triage_pdf.py --pdfs "C:\\ICT_Videos\\Testing\\*.pdf"
    python knowledge_ingest/tests/triage_pdf.py --pdfs <pdf1> <pdf2> --render
    python knowledge_ingest/tests/triage_pdf.py --pdfs <pdfs> --render --summary triage.json
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import fitz  # PyMuPDF


def classify_page(text_chars, image_area_ratio, image_count, line_count, short_line_ratio):
    """Cheap 3-way page-dominance heuristic. Tunable; not magic.

    Thresholds are deliberately conservative — when unsure, return co_dependent
    (the conservative kick-to-human choice from §11b, applied at page level).

    Rules refined after the first real run (Flux + Lumitrader): when a page is a
    full-page rendered image with text BAKED INTO the image, get_text() returns
    the OCR'd text layer (non-zero chars) but the page is still image_dominant —
    that text can't be extracted by the text pipeline, it has to be read by a VLM.
    So img_area near 1.0 -> image_dominant regardless of text_chars. Conversely,
    img_count==0 means no image at all -> text_dominant regardless of text length
    (there's nothing to be co-dependent WITH).
    """
    if text_chars < 80 and image_area_ratio < 0.05:
        return "near_empty"
    # full-page image: text (if any) is baked into the image -> VLM, not text pipeline
    if image_area_ratio >= 0.85:
        return "image_dominant"
    # no images at all: pure text, regardless of length
    if image_count == 0 or image_area_ratio < 0.05:
        return "text_dominant"
    # image-dominant: little text AND substantial image coverage
    if text_chars < 200 and image_area_ratio > 0.30:
        return "image_dominant"
    # text-dominant: lots of text AND little image coverage
    if text_chars > 1500 and image_area_ratio < 0.15:
        return "text_dominant"
    # everything else is the hard middle — co_dependent or truly mixed
    return "co_dependent"


def page_stats(page):
    """Extract per-page structural stats via PyMuPDF."""
    # text
    text = page.get_text("text") or ""
    text_chars = len(text.strip())
    lines = [ln for ln in text.splitlines() if ln.strip()]
    line_count = len(lines)
    short_lines = sum(1 for ln in lines if len(ln.strip()) < 40)
    short_line_ratio = (short_lines / line_count) if line_count else 0.0

    # images
    imgs = page.get_images(full=True)
    image_count = len(imgs)
    image_area_ratio = 0.0
    page_area = page.rect.width * page.rect.height
    if page_area > 0 and image_count:
        total_img_area = 0.0
        for im in imgs:
            # im: (xref, smask, w, h, bpc, colorspace, alt, name, filter)
            try:
                w, h = float(im[2]), float(im[3])
                # crude: assume image drawn at native size capped to page;
                # real coverage comes from get_image_bbox per placement
                total_img_area += w * h
            except (IndexError, ValueError):
                pass
        # use placement bboxes if available (more accurate)
        try:
            bboxes = page.get_image_info()
            placed = 0.0
            for b in bboxes:
                bb = b.get("bbox")
                if bb:
                    placed += abs((bb[2] - bb[0]) * (bb[3] - bb[1]))
            if placed > 0:
                image_area_ratio = min(1.0, placed / page_area)
            else:
                image_area_ratio = min(1.0, total_img_area / page_area)
        except Exception:
            image_area_ratio = min(1.0, total_img_area / page_area)

    dom = classify_page(text_chars, image_area_ratio, image_count, line_count, short_line_ratio)

    return {
        "text_chars": text_chars,
        "line_count": line_count,
        "short_line_ratio": round(short_line_ratio, 2),
        "image_count": image_count,
        "image_area_ratio": round(image_area_ratio, 3),
        "page_dominance": dom,
    }


def triage_pdf(pdf_path, render_dir=None, render_dpi=72):
    """Triage all pages of one PDF. Returns (summary, per_page_list)."""
    doc = fitz.open(pdf_path)
    per_page = []
    dom_counts = {"text_dominant": 0, "image_dominant": 0, "co_dependent": 0, "near_empty": 0}
    render_paths = []

    stem = Path(pdf_path).stem
    for pno in range(len(doc)):
        page = doc[pno]
        s = page_stats(page)
        s["page"] = pno + 1
        dom_counts[s["page_dominance"]] += 1

        if render_dir and s["page_dominance"] in ("image_dominant", "co_dependent"):
            # only render the pages we can't decide structurally — saves disk
            out_png = os.path.join(render_dir, f"{stem}_p{pno + 1:03d}.png")
            pix = page.get_pixmap(dpi=render_dpi)
            pix.save(out_png)
            s["render_path"] = out_png
            render_paths.append(out_png)

        per_page.append(s)
    doc.close()

    summary = {
        "pdf": pdf_path,
        "pages": len(per_page),
        "dominance_counts": dom_counts,
    }
    return summary, per_page, render_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True, nargs="+",
                    help="PDF files or glob pattern(s)")
    ap.add_argument("--render", action="store_true",
                    help="render image_dominant + co_dependent pages to PNG for the bake-off")
    ap.add_argument("--render-dir", default=None,
                    help="where to put rendered PNGs (default: <pdf_dir>/_triage_renders)")
    ap.add_argument("--render-dpi", type=int, default=100)
    ap.add_argument("--summary", default=None,
                    help="path to write a JSON summary of all PDFs")
    ap.add_argument("--max-pages-per-pdf", type=int, default=None,
                    help="cap pages processed per PDF (for quick scans of huge books)")
    args = ap.parse_args()

    # expand globs
    pdfs = []
    for pat in args.pdfs:
        if any(c in pat for c in "*?[]"):
            pdfs.extend(sorted(glob.glob(pat)))
        else:
            pdfs.append(pat)
    pdfs = [p for p in pdfs if os.path.exists(p)]
    if not pdfs:
        print("no PDFs found")
        return 1

    all_summary = []
    all_rendered = []

    for pdf in pdfs:
        if args.render:
            rdir = args.render_dir or os.path.join(os.path.dirname(pdf), "_triage_renders")
            os.makedirs(rdir, exist_ok=True)
        else:
            rdir = None
        print(f"\n=== {os.path.basename(pdf)} ===")
        summary, per_page, rendered = triage_pdf(pdf, rdir, args.render_dpi)
        all_summary.append(summary)
        all_rendered.extend(rendered)

        # print a compact per-page table — only the non-text-dominant pages
        # (text-dominant is the boring majority; the others are the actionable set)
        print(f"  pages: {summary['pages']}")
        dc = summary["dominance_counts"]
        total = summary["pages"] or 1
        print(f"  text_dominant   : {dc['text_dominant']:4d}  ({dc['text_dominant']*100//total}%)")
        print(f"  image_dominant  : {dc['image_dominant']:4d}  ({dc['image_dominant']*100//total}%)")
        print(f"  co_dependent    : {dc['co_dependent']:4d}  ({dc['co_dependent']*100//total}%)")
        print(f"  near_empty      : {dc['near_empty']:4d}  ({dc['near_empty']*100//total}%)")
        interesting = [p for p in per_page if p["page_dominance"] in
                       ("image_dominant", "co_dependent", "near_empty")]
        if interesting:
            print(f"  --- non-text-dominant pages (the actionable set) ---")
            for p in interesting[:60]:
                rp = p.get("render_path", "")
                rp_s = f" [rendered]" if rp else ""
                print(f"    p{p['page']:4d}  {p['page_dominance']:14s}  "
                      f"txt={p['text_chars']:5d}  img={p['image_count']:2d}  "
                      f"img_area={p['image_area_ratio']:.2f}{rp_s}")
            if len(interesting) > 60:
                print(f"    ... and {len(interesting)-60} more")

    if args.summary:
        out = {
            "pdfs": all_summary,
            "rendered_pages": all_rendered,
        }
        Path(args.summary).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nsummary -> {args.summary}")
    print(f"\nrendered {len(all_rendered)} pages for the bake-off")
    return 0


if __name__ == "__main__":
    sys.exit(main())