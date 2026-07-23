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

from knowledge_ingest.paths import (
    mineru_text_staged_dir, mineru_text_ingest_dir, kb_data_dir,
)

MINERU_VENV = r"C:\Users\vinay\mineru_venv"
MINERU_PYTHON = os.path.join(MINERU_VENV, "Scripts", "python.exe")

# MinerU raw extraction output. These are transient build artifacts; they live
# under KB_DATA_DIR (consumer-owned data tree — see knowledge_ingest/paths.py).
MINERU_OUTPUT_BASE = os.path.join(kb_data_dir(), "ingest", "mineru_output")

# Where to put extracted images for v4 chart extraction
MINERU_IMAGE_DIR = os.path.join(kb_data_dir(), "ingest", "mineru_images")

# Where to put extracted markdown for text ingestion (staged as .txt)
MINERU_TEXT_DIR = mineru_text_staged_dir()

# Where the ingest pipeline writes its output (units/, notes/, etc.)
MINERU_TEXT_INGEST_DIR = mineru_text_ingest_dir()


def check_mineru():
    """Verify MinerU is installed in the venv.

    The package was renamed from `magic_pdf` to `mineru` (v3.x). We check
    for the `mineru` package and the `mineru.exe` CLI entry point.
    """
    if not os.path.exists(MINERU_PYTHON):
        print(f"ERROR: MinerU venv not found at {MINERU_VENV}")
        print("Install with: python -m venv C:\\Users\\vinay\\mineru_venv")
        print("Then: C:\\Users\\vinay\\mineru_venv\\Scripts\\pip install mineru[full]")
        return False

    # Check if the `mineru` package is installed (was `magic_pdf` in older versions)
    result = subprocess.run(
        [MINERU_PYTHON, "-c", "import mineru; print(getattr(mineru, '__version__', 'unknown'))"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        print(f"ERROR: `mineru` package not installed in {MINERU_VENV}")
        print("Install with: C:\\Users\\vinay\\mineru_venv\\Scripts\\pip install mineru[full]")
        return False

    # Also verify the CLI entry point exists
    mineru_cli = os.path.join(MINERU_VENV, "Scripts", "mineru.exe")
    if not os.path.exists(mineru_cli):
        print(f"ERROR: mineru.exe CLI not found at {mineru_cli}")
        print("Reinstall with: C:\\Users\\vinay\\mineru_venv\\Scripts\\pip install --force-reinstall mineru[full]")
        return False

    print(f"MinerU ready: mineru v{result.stdout.strip()} ({mineru_cli})")
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

    # MinerU CLI (v3.x): mineru.exe -p <pdf> -o <output> -m auto -b hybrid-engine
    # The package was renamed from magic_pdf to mineru; the CLI entry point is mineru.exe
    mineru_cli = os.path.join(MINERU_VENV, "Scripts", "mineru.exe")
    cmd = [
        mineru_cli,
        "-p", pdf_path,
        "-o", output_dir,
        "-m", "auto",        # auto mode (OCR + text) -- handles mixed PDFs
        "-b", "hybrid-engine", # backend that produces hybrid_auto/ output
    ]

    print(f"Running MinerU on: {pdf_path}")
    print(f"  Output: {output_dir}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        if result.stdout:
            print(f"  stdout: {result.stdout[:500]}")
        return None, None

    print(f"  Done ({elapsed:.1f}s)")

    # Find the output markdown and images.
    # hybrid-engine backend creates: <output>/<name>/hybrid_auto/<name>.md
    #                         images in <output>/<name>/hybrid_auto/images/
    # (Older pipeline backend created <output>/<name>/auto/ -- we glob as fallback.)
    auto_dir = os.path.join(output_dir, pdf_name, "hybrid_auto")
    md_path = os.path.join(auto_dir, f"{pdf_name}.md")
    images_dir = os.path.join(auto_dir, "images")

    if not os.path.exists(md_path):
        # Try older "auto" layout, then glob recursively for any .md
        auto_dir_old = os.path.join(output_dir, pdf_name, "auto")
        md_path_old = os.path.join(auto_dir_old, f"{pdf_name}.md")
        if os.path.exists(md_path_old):
            md_path = md_path_old
            images_dir = os.path.join(auto_dir_old, "images")
        else:
            # Last resort: glob recursively
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
    """Copy markdown to a flat dir for text ingestion.

    The ingest pipeline expects .txt files; MinerU produces .md.
    We copy the .md to a .txt so the pipeline picks it up.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{pdf_name}.txt")
    shutil.copy2(md_path, dest)
    return dest


def _run_text_ingest(text_dir, output_dir, file_filter=None, ict_aware=True, profile=None):
    """Actually execute the ICT-aware text ingest via IngestPipeline.

    Called from process_pdf() when run_text=True. Uses the same IngestPipeline
    that run.py uses, but invoked directly (no subprocess) to avoid venv issues.

    Args:
        ict_aware: if True (default), use ICT-aware prompts (embeds ICT domain
            knowledge into classify/extract). If False, use generic prompts.
            DEPRECATED in favor of `profile`; kept for back-compat. If `profile`
            is given it takes precedence.
        profile: prompt profile name (DESIGN.md §9 Phase 2). A single name or a
            "+"-joined combination ("ict+gex"). Overrides `ict_aware`.
    """
    from knowledge_ingest.config.config import PipelineConfig
    from knowledge_ingest.pipeline.ingest import IngestPipeline

    cfg = PipelineConfig()
    cfg.input_dir = text_dir
    cfg.output_dir = output_dir
    cfg.source_type = "pdf"
    cfg.ict_aware = ict_aware
    if profile:
        cfg.profile = profile
    cfg.skip_existing = False
    if file_filter:
        cfg.file_filter = file_filter

    mode = cfg.profile if cfg.profile else ("ict-aware" if ict_aware else "generic")
    print(f"  Ingesting text from {text_dir} -> {output_dir} ({mode})")
    pipeline = IngestPipeline(cfg)
    pipeline.run()
    units_dir = os.path.join(output_dir, "units")
    n_units = len(glob.glob(os.path.join(units_dir, "*.jsonl")))
    print(f"  Text ingest complete: {n_units} unit files in {units_dir}")
    return units_dir


def process_pdf(pdf_path, run_v4=False, run_text=False,
                text_output_dir=None, text_input_dir=None, ict_aware=True,
                profile=None):
    """Full pipeline: MinerU extract -> route images/text.

    Args:
        pdf_path: path to the PDF file.
        run_v4: if True, print the v4 chart extraction command for images.
        run_text: if True, actually execute the text ingest (not just print it).
        text_output_dir: where to write ingest output (default: MINERU_TEXT_INGEST_DIR).
        text_input_dir: where to stage .txt files for ingest (default: MINERU_TEXT_DIR).
        ict_aware: if True (default), use ICT-aware prompts for text ingest.
            DEPRECATED; use `profile` instead.
        profile: prompt profile name (DESIGN.md §9 Phase 2). Overrides ict_aware.
    """
    md_path, images_dir = run_mineru(pdf_path)
    if not md_path:
        return None

    pdf_name = Path(pdf_path).stem
    n_images = 0
    text_path = None

    if images_dir and os.path.exists(images_dir):
        n_images = collect_images_for_v4(images_dir, MINERU_IMAGE_DIR, pdf_name)
        print(f"  Collected {n_images} images → {MINERU_IMAGE_DIR}")

    stage_dir = text_input_dir or MINERU_TEXT_DIR
    text_path = collect_text_for_ingest(md_path, stage_dir, pdf_name)
    print(f"  Text staged → {text_path}")

    if run_v4 and n_images > 0:
        print(f"\n  To extract chart knowledge from {n_images} images, run:")
        print(f'    python -m knowledge_ingest.tests.run_v4_full '
              f'--out-dir "C:\\ICT_Videos\\Testing\\_mineru_v4" '
              f'--filter "{pdf_name}_*"')

    if run_text and text_path:
        out_dir = text_output_dir or MINERU_TEXT_INGEST_DIR
        units_dir = _run_text_ingest(stage_dir, out_dir,
                                    file_filter=f"{pdf_name}.txt",
                                    ict_aware=ict_aware,
                                    profile=profile)
        return {"pdf": pdf_path, "markdown": md_path, "images": n_images,
                "text": text_path, "units_dir": units_dir}

    return {"pdf": pdf_path, "markdown": md_path, "images": n_images,
            "text": text_path}


def main():
    ap = argparse.ArgumentParser(description="MinerU PDF → knowledge pipeline")
    ap.add_argument("--pdf", help="single PDF to process")
    ap.add_argument("--pdf-dir", help="directory of PDFs to batch process")
    ap.add_argument("--batch", action="store_true", help="batch mode for --pdf-dir")
    ap.add_argument("--run-v4", action="store_true", help="print v4 chart extraction command")
    ap.add_argument("--run-text", action="store_true", default=True,
                    help="execute ICT-aware text ingest after MinerU extraction (default: True)")
    ap.add_argument("--no-text", action="store_true",
                    help="skip text ingest (only do MinerU extraction + image routing)")
    ap.add_argument("--text-output", default=None,
                    help="override text ingest output dir (default: _mineru_text_ingest)")
    ap.add_argument("--ict-aware", action="store_true", default=True,
                    help="(deprecated, use --profile) use ICT-aware prompts (default: True)")
    ap.add_argument("--no-ict-aware", action="store_false", dest="ict_aware",
                    help="use generic (domain-agnostic) prompts instead of ICT-aware")
    ap.add_argument("--profile", default=None,
                    help="prompt profile (DESIGN.md §9 Phase 2): single name "
                         "(ict, generic, gex) or '+'-joined (ict+gex). Use "
                         "'list' to print and exit. Overrides --ict-aware.")
    ap.add_argument("--check", action="store_true", help="just check MinerU installation")
    args = ap.parse_args()

    # `--profile list` prints registered profiles and exits.
    if args.profile and args.profile.lower() in ("list", "?", "help"):
        from knowledge_ingest.domains import list_profiles
        print("Registered prompt profiles:")
        for p in list_profiles():
            print(f"  {p.name:10s} domains={p.domains}  — {p.description.splitlines()[0]}")
        print("\nCombine with '+': e.g. --profile ict+gex")
        return

    if args.check:
        check_mineru()
        return

    if not check_mineru():
        return

    run_text = args.run_text and not args.no_text
    text_output = args.text_output or MINERU_TEXT_INGEST_DIR
    ict_aware = args.ict_aware
    profile = args.profile

    results = []

    if args.pdf:
        r = process_pdf(args.pdf, run_v4=args.run_v4, run_text=run_text,
                        text_output_dir=text_output, ict_aware=ict_aware,
                        profile=profile)
        if r:
            results.append(r)

    if args.pdf_dir and args.batch:
        pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
        print(f"\nFound {len(pdfs)} PDFs in {args.pdf_dir}")
        for pdf in pdfs:
            r = process_pdf(pdf, run_v4=args.run_v4, run_text=run_text,
                            text_output_dir=text_output, ict_aware=ict_aware,
                            profile=profile)
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