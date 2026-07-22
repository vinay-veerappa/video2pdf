"""
VLM consensus bake-off — Phase 1 of the image/PDF plan (HANDOVER §17b).

Tests the founding premise: does cross-model consensus predict correctness?
Also answers the user's known concern: where does `kind` get misclassified?

For each input image:
  - runs 3 VLMs (gemma4, qwen3.5, minimax-m3 — the §11a honest pair + reader)
  - extracts kind, sequence length, text_content counts, inferred
  - compares honest-pair (gemma+qwen) kind consensus to ground truth (where labeled)
  - compares pure-VLM vs OCR+VLM on the full-page-image-no-text-layer subset
  - reports per-image and aggregate: kind agreement, seq fabrication, text divergence

Also runs an OCR+VLM variant on the no-text-layer subset (per user decision
"test both, let data decide"): OCR first (glm-ocr / deepseek-ocr), feed the
OCR text into the VLM prompt, then run the same VLM read. Compare to pure-VLM.

Output:
  bakeoff_results.jsonl  - per-input, per-model results
  bakeoff_report.md       - human-readable aggregate + the decision-relevant tables

Usage:
    python knowledge_ingest/examples/vlm_calibration_run.py --inputs bakeoff_inputs.jsonl
    python knowledge_ingest/examples/vlm_calibration_run.py --inputs ... --models gemma4:cloud qwen3.5:cloud minimax-m3:cloud
    python knowledge_ingest/examples/vlm_calibration_run.py --inputs ... --skip-ocr  # pure VLM only
    python knowledge_ingest/examples/vlm_calibration_run.py --inputs ... --resume    # skip inputs already in results

Requires: Ollama running locally with the named VLMs pulled.
"""

import os
import sys
import json
import base64
import argparse
import time
from pathlib import Path

import requests

# reuse the chart_extract PROPOSE_PROMPT for consistency with the production path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from knowledge_ingest.sources.chart_extract import PROPOSE_PROMPT


OLLAMA_URL = "http://localhost:11434"


def b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(model, image_b64, prompt=PROPOSE_PROMPT, temperature=0.1, timeout=300):
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


def text_count(obj):
    """Total text_content items across all channels (the §11a BAD metric, kept for audit)."""
    tc = obj.get("text_content") or {}
    return sum(len(tc.get(k) or []) for k in ("conditions", "confluences", "notes", "other_text"))


def seq_len(obj):
    seq = obj.get("sequence")
    return len(seq) if isinstance(seq, list) else 0


def run_one(model, image_b64, ocr_text=None):
    """Run one VLM call. If ocr_text given, prepend it to the prompt (OCR+VLM merge)."""
    prompt = PROPOSE_PROMPT
    if ocr_text:
        prompt = ("An OCR pass on this image produced the following text. Use it to "
                  "ground your reading; cross-check against what you see in the image. "
                  "Do NOT trust the OCR blindly — if it conflicts with the image, the "
                  "image wins.\n\nOCR TEXT:\n" + ocr_text + "\n\n" + PROPOSE_PROMPT)
    t0 = time.time()
    raw = call_vlm(model, image_b64, prompt=prompt)
    elapsed = time.time() - t0
    obj = parse_json_lenient(raw)
    return {
        "model": model,
        "elapsed_s": round(elapsed, 1),
        "raw_ok": obj is not None,
        "kind": (obj or {}).get("kind"),
        "n_seq": seq_len(obj or {}),
        "n_text": text_count(obj or {}),
        "bias": (obj or {}).get("bias"),
        "n_inferred": len((obj or {}).get("inferred") or []),
        "name": (obj or {}).get("name"),
        "raw": raw if obj is None else None,  # keep raw only on parse failure (debug)
    }


def ocr_image(ocr_model, image_b64, timeout=300):
    """Run an OCR model on the image; return the extracted text or ''."""
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": ocr_model,
        "prompt": "Transcribe ALL text visible in this image verbatim. "
                  "Output only the transcription, no commentary.",
        "images": [image_b64], "stream": False,
        "options": {"temperature": 0.0},
    }, timeout=timeout)
    if not r.ok:
        return ""
    return r.json().get("response", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="bakeoff_inputs.jsonl from build_bakeoff_inputs.py")
    ap.add_argument("--models", nargs="+",
                    default=["gemma4:cloud", "qwen3.5:cloud", "minimax-m3:cloud"],
                    help="VLMs to run (the §11a honest pair + reader)")
    ap.add_argument("--ocr-model", default="glm-ocr:cloud",
                    help="OCR model for the OCR+VLM comparison on no-text-layer pages")
    ap.add_argument("--ocr-on-sources", nargs="+",
                    default=["Flux_NY_Guide.pdf", "lumitrader-ict-2022-book.pdf"],
                    help="run OCR+VLM on inputs from these source PDFs (full-page-image docs)")
    ap.add_argument("--out", default="bakeoff_results.jsonl")
    ap.add_argument("--report", default="bakeoff_report.md")
    ap.add_argument("--resume", action="store_true",
                    help="skip inputs whose path already has a result line in --out")
    ap.add_argument("--skip-ocr", action="store_true",
                    help="don't run the OCR+VLM comparison (pure VLM only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of inputs (for a quick smoke test)")
    args = ap.parse_args()

    inputs = [json.loads(l) for l in open(args.inputs, encoding="utf-8") if l.strip()]
    if args.limit:
        inputs = inputs[:args.limit]
    print(f"{len(inputs)} inputs; models: {args.models}; OCR on: {args.ocr_on_sources}")

    # resume
    done_paths = set()
    if args.resume and os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            if l.strip():
                done_paths.add(json.loads(l).get("path"))
        print(f"resume: {len(done_paths)} inputs already done")

    out_f = open(args.out, "a", encoding="utf-8")
    results = []
    for i, inp in enumerate(inputs, 1):
        if inp["path"] in done_paths:
            continue
        if not os.path.exists(inp["path"]):
            print(f"  ! missing: {inp['path']}")
            continue
        print(f"\n[{i}/{len(inputs)}] {os.path.basename(inp['path'])}  ({inp.get('source_pdf') or 'standalone'} p{inp.get('page')})")
        img = b64_image(inp["path"])
        per_model = {}
        for m in args.models:
            try:
                per_model[m] = run_one(m, img)
                r = per_model[m]
                print(f"    {m:20s} {r['elapsed_s']:5.1f}s  kind={str(r['kind']):20s} "
                      f"seq={r['n_seq']:2d}  text={r['n_text']:3d}")
            except Exception as e:
                print(f"    {m:20s} ERROR {str(e)[:80]}")
                per_model[m] = {"model": m, "error": str(e)[:200], "raw_ok": False}

        # OCR+VLM variant on the full-page-image subset
        ocr_variant = None
        if not args.skip_ocr and inp.get("source_pdf") in args.ocr_on_sources:
            try:
                print(f"    OCR ({args.ocr_model})...", end=" ", flush=True)
                ocr_txt = ocr_image(args.ocr_model, img)
                print(f"{len(ocr_txt)} chars")
                # run minimax with OCR text (minimax is the price_path reader; for full-page
                # images we want the best reader, and the user said "test both, let data decide")
                ocr_variant = {
                    "ocr_text_len": len(ocr_txt),
                    "ocr_text_preview": ocr_txt[:200],
                    "vlm_with_ocr": run_one("minimax-m3:cloud", img, ocr_text=ocr_txt),
                }
                r = ocr_variant["vlm_with_ocr"]
                print(f"    minimax+OCR         {r['elapsed_s']:5.1f}s  kind={str(r['kind']):20s} "
                      f"seq={r['n_seq']:2d}  text={r['n_text']:3d}")
            except Exception as e:
                print(f"    OCR ERROR {str(e)[:80]}")
                ocr_variant = {"error": str(e)[:200]}

        rec = {**inp, "models": per_model, "ocr_variant": ocr_variant}
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        results.append(rec)
    out_f.close()

    write_report(results, args.report)
    print(f"\nresults -> {args.out}")
    print(f"report  -> {args.report}")


def write_report(results, path):
    """Aggregate the decision-relevant tables for §17b."""
    lines = ["# VLM calibration bake-off report\n",
             f"_{len(results)} inputs run_\n"]

    # --- (a) honest-pair kind consensus vs ground truth (labeled subset only) ---
    labeled = [r for r in results if r.get("true_kind") and r["true_kind"] != "TODO"]
    lines.append("\n## (a) Honest-pair kind consensus vs ground truth\n")
    lines.append(f"_{len(labeled)} labeled inputs (the §13d standalone set)_\n")
    if labeled:
        correct = 0
        rows = ["| input | true_kind | gemma | qwen | minimax | honest-pair consensus | match? |",
                "|---|---|---|---|---|---|---|"]
        for r in labeled:
            g = r["models"].get("gemma4:cloud", {}).get("kind")
            q = r["models"].get("qwen3.5:cloud", {}).get("kind")
            m = r["models"].get("minimax-m3:cloud", {}).get("kind")
            consensus = g if g == q else f"DISAGREE({g}/{q})"
            match = "✓" if consensus == r["true_kind"] else "✗"
            if match == "✓":
                correct += 1
            name = os.path.basename(r["path"])[:30]
            rows.append(f"| {name} | {r['true_kind']} | {g} | {q} | {m} | {consensus} | {match} |")
        lines.extend(rows)
        lines.append(f"\n**Honest-pair consensus accuracy: {correct}/{len(labeled)} "
                     f"({correct*100//len(labeled)}%)**")
    else:
        lines.append("_No labeled inputs in this run — label the standalone set first._")

    # --- (b) where does kind get misclassified? ---
    lines.append("\n## (b) Kind misclassification patterns\n")
    kind_disagree = []
    minimax_price_path_bias = 0
    minimax_total = 0
    for r in results:
        g = r["models"].get("gemma4:cloud", {}).get("kind")
        q = r["models"].get("qwen3.5:cloud", {}).get("kind")
        m = r["models"].get("minimax-m3:cloud", {}).get("kind")
        if g and q and g != q:
            kind_disagree.append((os.path.basename(r["path"]), g, q, m))
        if m:
            minimax_total += 1
            if m == "price_path":
                minimax_price_path_bias += 1
    lines.append(f"\n**Honest-pair kind disagreements (gemma vs qwen): {len(kind_disagree)}**")
    if kind_disagree:
        lines.append("| input | gemma | qwen | minimax |")
        lines.append("|---|---|---|---|")
        for n, g, q, m in kind_disagree[:30]:
            lines.append(f"| {n[:30]} | {g} | {q} | {m} |")
    if minimax_total:
        lines.append(f"\n**Minimax price_path bias: {minimax_price_path_bias}/{minimax_total} "
                     f"({minimax_price_path_bias*100//minimax_total}%) of inputs labeled price_path** "
                     f"(§11a found 5/7 = 71%)")

    # --- (c) text_content: union vs single model ---
    lines.append("\n## (c) text_content — union vs single model\n")
    lines.append("_Does union-and-dedupe across the 3 models beat any single model?_")
    union_examples = []
    for r in results[:20]:
        per_model = r["models"]
        texts = {}
        for m in ("gemma4:cloud", "qwen3.5:cloud", "minimax-m3:cloud"):
            texts[m] = per_model.get(m, {}).get("n_text", 0)
        # union would need full text arrays; here we just show counts as a first look
        union_examples.append((os.path.basename(r["path"])[:25], texts))
    if union_examples:
        lines.append("| input | gemma | qwen | minimax |")
        lines.append("|---|---|---|---|")
        for n, t in union_examples:
            lines.append(f"| {n} | {t.get('gemma4:cloud',0)} | {t.get('qwen3.5:cloud',0)} | {t.get('minimax-m3:cloud',0)} |")

    # --- (d) sequence fabrication on case-A images ---
    lines.append("\n## (d) Sequence fabrication (case-A images)\n")
    lines.append("_Case A = framework-illustrated-on-a-path; correct output has NO sequence._")
    case_a = [r for r in labeled if r.get("true_case") == "A"]
    if case_a:
        lines.append(f"_{len(case_a)} case-A labeled inputs:_\n")
        lines.append("| input | true | gemma seq | qwen seq | minimax seq | minimax fabricates? |")
        lines.append("|---|---|---|---|---|---|")
        for r in case_a:
            gs = r["models"].get("gemma4:cloud", {}).get("n_seq", 0)
            qs = r["models"].get("qwen3.5:cloud", {}).get("n_seq", 0)
            ms = r["models"].get("minimax-m3:cloud", {}).get("n_seq", 0)
            fabricates = "✗ FABRICATES" if ms > 0 and gs == 0 and qs == 0 else ("" if ms == 0 else "?")
            lines.append(f"| {os.path.basename(r['path'])[:25]} | 0 | {gs} | {qs} | {ms} | {fabricates} |")
    else:
        lines.append("_No case-A labeled inputs in this run._")

    # --- OCR+VLM vs pure VLM on full-page-image subset ---
    ocr_runs = [r for r in results if r.get("ocr_variant") and not r["ocr_variant"].get("error")]
    if ocr_runs:
        lines.append("\n## (e) OCR+VLM vs pure-VLM on full-page-image (no-text-layer) pages\n")
        lines.append(f"_{len(ocr_runs)} pages ran both variants. Decision-relevant per user._\n")
        lines.append("| input | pure-minimax kind | pure-minimax text | +OCR kind | +OCR text | OCR chars |")
        lines.append("|---|---|---|---|---|---|")
        for r in ocr_runs:
            pure = r["models"].get("minimax-m3:cloud", {})
            ocr = r["ocr_variant"].get("vlm_with_ocr", {})
            lines.append(f"| {os.path.basename(r['path'])[:25]} "
                         f"| {pure.get('kind')} | {pure.get('n_text',0)} "
                         f"| {ocr.get('kind')} | {ocr.get('n_text',0)} "
                         f"| {r['ocr_variant'].get('ocr_text_len',0)} |")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()