"""
Full production run — ICT-aware chart extraction (HANDOVER §17, v4 prompt).

Processes ALL images in C:\\ICT_Videos\\Testing:
  - 7 standalone labeled images
  - 811 rendered PDF pages (from 11 PDFs)

Uses the v4 classification-free prompt with gemma4:31b-cloud (proven 7/7 on the
labeled set). Single model — no consensus needed since v4's path_is_method
judgment eliminates the classification failure mode.

Output: one JSON per image in C:\\ICT_Videos\\Testing\\_v4_full_run\\
  - Resume supported (skip images already processed)
  - Progress printed every 10 images
  - Final summary: count, errors, educator distribution

Usage:
    python knowledge_ingest/tests/run_v4_full.py
    python knowledge_ingest/tests/run_v4_full.py --model gemma4:31b-cloud
    python knowledge_ingest/tests/run_v4_full.py --resume
"""

import os
import sys
import json
import base64
import argparse
import time
import glob

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from knowledge_ingest.sources.ict_chart_prompts import get_prompt

OLLAMA_URL = "http://localhost:11434"

# Input directories
STANDALONE_DIR = r"C:\ICT_Videos\Testing"
RENDERS_DIR = r"C:\ICT_Videos\Testing\_triage_renders"
OUTPUT_DIR = r"C:\ICT_Videos\Testing\_v4_full_run"

# Standalone image extensions
IMG_EXTS = ("png", "jpg", "jpeg", "webp", "jfif", "gif", "bmp")


def b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(model, image_b64, prompt, temperature=0.1, timeout=600):
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model, "prompt": prompt, "images": [image_b64],
        "stream": False, "format": "json",
        "options": {"temperature": temperature},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def parse_json_lenient(raw):
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


def collect_inputs():
    """Collect all standalone images + all rendered PDF pages."""
    inputs = []

    # Standalone images (not in _triage_renders, not .pdf, not .txt, not .json)
    skip_dirs = {"_triage_renders", "_v4_full_run", "_prompt_runs", "_mineru_out",
                 "_bakeoff_out", "bakeoff_results.jsonl"}
    for ext in IMG_EXTS:
        for f in glob.glob(os.path.join(STANDALONE_DIR, f"*.{ext}")):
            inputs.append({"path": f, "source": "standalone", "source_pdf": None, "page": None})

    # Rendered PDF pages
    for f in sorted(glob.glob(os.path.join(RENDERS_DIR, "*.png"))):
        fname = os.path.basename(f)
        # parse source PDF and page from filename: e.g. "Flux_NY_Guide_p001.png"
        parts = fname.rsplit("_p", 1)
        if len(parts) == 2:
            source_pdf = parts[0] + ".pdf"
            try:
                page = int(parts[1].replace(".png", ""))
            except ValueError:
                page = None
        else:
            source_pdf = fname
            page = None
        inputs.append({"path": f, "source": "pdf_page", "source_pdf": source_pdf, "page": page})

    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:31b-cloud")
    ap.add_argument("--prompt", default="ict_v4")
    ap.add_argument("--resume", action="store_true", help="skip images already in output dir")
    ap.add_argument("--limit", type=int, default=None, help="cap number of images (for testing)")
    ap.add_argument("--out-dir", default=None, help="override output directory (for parallel sessions)")
    ap.add_argument("--filter", default=None,
                    help="only process images whose stem/source_pdf contains this substring (for parallel sessions)")
    args = ap.parse_args()

    out_dir = args.out_dir or OUTPUT_DIR

    prompt, prompt_label = get_prompt(args.prompt)
    print(f"prompt: {args.prompt}  ({prompt_label})")
    print(f"model:  {args.model}")
    print(f"output: {out_dir}")
    if args.filter:
        print(f"filter: {args.filter}")
    print()

    os.makedirs(out_dir, exist_ok=True)

    inputs = collect_inputs()
    print(f"Total inputs: {len(inputs)} "
          f"({sum(1 for i in inputs if i['source']=='standalone')} standalone + "
          f"{sum(1 for i in inputs if i['source']=='pdf_page')} PDF pages)")

    if args.filter:
        filt = args.filter.lower()
        inputs = [i for i in inputs if filt in os.path.basename(i["path"]).lower()
                  or (i.get("source_pdf") and filt in i["source_pdf"].lower())]
        print(f"After filter '{args.filter}': {len(inputs)} inputs")

    if args.limit:
        inputs = inputs[:args.limit]
        print(f"Limited to {len(inputs)} inputs")

    # Resume: skip images already processed
    done = set()
    if args.resume:
        for f in os.listdir(out_dir):
            if f.endswith(".json"):
                done.add(os.path.splitext(f)[0])
        print(f"Resume: {len(done)} images already done")

    out_f = open(os.path.join(out_dir, "_run_log.jsonl"), "a", encoding="utf-8")
    results = []
    errors = 0
    t_start = time.time()

    for i, inp in enumerate(inputs, 1):
        stem = os.path.splitext(os.path.basename(inp["path"]))[0]
        if stem in done:
            continue

        try:
            img_b64 = b64_image(inp["path"])
            t0 = time.time()
            raw = call_vlm(args.model, img_b64, prompt)
            elapsed = time.time() - t0
            obj = parse_json_lenient(raw)

            if obj is None:
                print(f"  [{i}/{len(inputs)}] {stem} — PARSE FAIL ({elapsed:.1f}s)")
                errors += 1
                # save raw for debugging
                with open(os.path.join(OUTPUT_DIR, f"{stem}.json"), "w", encoding="utf-8") as f:
                    json.dump({"input": inp, "error": "parse_fail", "raw": raw[:500]}, f, indent=2)
                continue

            result = {
                "input": inp,
                "model": args.model,
                "elapsed_s": round(elapsed, 1),
                "image_type": obj.get("image_type"),
                "path_is_method": obj.get("path_is_method"),
                "name": obj.get("name"),
                "framework": obj.get("framework"),
                "educator_guess": obj.get("educator_guess"),
                "n_seq": len(obj.get("sequence") or []),
                "n_text": sum(len((obj.get("text_content") or {}).get(k) or [])
                              for k in ("conditions", "confluences", "notes", "other_text")),
                "entry_mechanics": obj.get("entry_mechanics") or [],
                "concepts_raw": obj.get("concepts_raw") or [],
                "inferred": obj.get("inferred") or [],
                "obj": obj,
            }

            with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

            out_f.write(json.dumps({"stem": stem, **{k: v for k, v in result.items()
                                                     if k != "obj"}}) + "\n")
            out_f.flush()
            results.append(result)

            if i % 10 == 0 or i <= 5:
                elapsed_total = time.time() - t_start
                eta = elapsed_total / i * (len(inputs) - i) if i > 0 else 0
                print(f"  [{i}/{len(inputs)}] {stem[:40]:40s}  "
                      f"educ={str(result['educator_guess']):12s}  "
                      f"fw={str(result['framework']):10s}  "
                      f"pim={str(result['path_is_method']):5s}  "
                      f"seq={result['n_seq']:2d}  "
                      f"{result['elapsed_s']:.1f}s  "
                      f"ETA {eta/60:.0f}min")

        except Exception as e:
            print(f"  [{i}/{len(inputs)}] {stem} — ERROR {str(e)[:100]}")
            errors += 1
            with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump({"input": inp, "error": str(e)[:300]}, f, indent=2)

    out_f.close()
    total_time = time.time() - t_start

    # Summary
    print(f"\n{'='*70}")
    print(f"DONE — {len(results)} processed, {errors} errors, {total_time/60:.1f} min total")
    print(f"{'='*70}")

    # Educator distribution
    from collections import Counter
    educ_dist = Counter(r.get("educator_guess") for r in results if r.get("educator_guess"))
    print("\nEducator distribution:")
    for educ, count in educ_dist.most_common():
        print(f"  {educ:20s} {count:4d}")

    # Framework distribution
    fw_dist = Counter(r.get("framework") for r in results if r.get("framework"))
    print("\nFramework distribution:")
    for fw, count in fw_dist.most_common():
        print(f"  {fw:20s} {count:4d}")

    # path_is_method distribution
    pim_dist = Counter(r.get("path_is_method") for r in results)
    print(f"\npath_is_method: true={pim_dist.get(True,0)}  false={pim_dist.get(False,0)}  "
          f"null={pim_dist.get(None,0)}")

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()