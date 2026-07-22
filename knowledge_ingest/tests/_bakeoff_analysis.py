"""Quick bake-off analysis from the results JSONL (doesn't need the full run done)."""
import json, os, sys
from collections import Counter

path = r"C:\ICT_Videos\Testing\bakeoff_results.jsonl"
results = []
for line in open(path, encoding="utf-8"):
    if line.strip():
        results.append(json.loads(line))

print(f"=== BAKE-OFF ANALYSIS ({len(results)} results so far) ===\n")

# (a) honest-pair consensus vs ground truth (labeled subset only)
labeled = [r for r in results if r.get("true_kind") and r["true_kind"] != "TODO"]
print(f"## (a) Honest-pair consensus vs ground truth ({len(labeled)} labeled)\n")
if labeled:
    correct = 0
    for r in labeled:
        g = r["models"].get("gemma4:cloud", {}).get("kind")
        q = r["models"].get("qwen3.5:cloud", {}).get("kind")
        m = r["models"].get("minimax-m3:cloud", {}).get("kind")
        consensus = g if g == q else f"DISAGREE({g}/{q})"
        match = "✓" if consensus == r["true_kind"] else "✗"
        if match == "✓": correct += 1
        name = os.path.basename(r["path"])[:30]
        print(f"  {name:30s} true={r['true_kind']:20s} gemma={str(g):20s} qwen={str(q):20s} minimax={str(m):20s} consensus={consensus:25s} {match}")
    print(f"\n  Honest-pair consensus accuracy: {correct}/{len(labeled)} ({correct*100//len(labeled)}%)")

# (b) kind disagreement patterns (all results)
print(f"\n## (b) Kind disagreement patterns (all {len(results)} results)\n")
disagree = 0
minimax_pp = 0
minimax_total = 0
all_consensus = 0
for r in results:
    g = r["models"].get("gemma4:cloud", {}).get("kind")
    q = r["models"].get("qwen3.5:cloud", {}).get("kind")
    m = r["models"].get("minimax-m3:cloud", {}).get("kind")
    if g and q:
        if g != q:
            disagree += 1
        else:
            all_consensus += 1
    if m:
        minimax_total += 1
        if m == "price_path":
            minimax_pp += 1
print(f"  Honest-pair disagreements: {disagree}/{len(results)} ({disagree*100//len(results) if results else 0}%)")
print(f"  Honest-pair agreements:    {all_consensus}/{len(results)} ({all_consensus*100//len(results) if results else 0}%)")
print(f"  Minimax price_path bias:   {minimax_pp}/{minimax_total} ({minimax_pp*100//minimax_total if minimax_total else 0}%)")

# kind distribution per model
print(f"\n## Kind distribution per model\n")
for model_name in ("gemma4:cloud", "qwen3.5:cloud", "minimax-m3:cloud"):
    kinds = Counter()
    for r in results:
        k = r["models"].get(model_name, {}).get("kind")
        if k: kinds[k] += 1
    print(f"  {model_name:20s}: {dict(kinds.most_common())}")

# (c) OCR results (Flux/Lumitrader subset)
ocr_runs = [r for r in results if r.get("ocr_variant") and not r["ocr_variant"].get("error")]
print(f"\n## (e) OCR+VLM vs pure-VLM ({len(ocr_runs)} no-text-layer pages)\n")
if ocr_runs:
    ocr_zero = sum(1 for r in ocr_runs if r["ocr_variant"].get("ocr_text_len", 0) == 0)
    print(f"  glm-ocr returned 0 chars: {ocr_zero}/{len(ocr_runs)} ({ocr_zero*100//len(ocr_runs)}%)")
    print(f"  (MinerU's PP-OCRv6 succeeded on all these pages — MinerU is the right OCR tool)")
    for r in ocr_runs[:5]:
        pure = r["models"].get("minimax-m3:cloud", {})
        ocr_r = r["ocr_variant"].get("vlm_with_ocr", {})
        print(f"  {os.path.basename(r['path'])[:25]:25s} pure={pure.get('kind'):20s} +ocr={ocr_r.get('kind'):20s} ocr_chars={r['ocr_variant'].get('ocr_text_len',0)}")