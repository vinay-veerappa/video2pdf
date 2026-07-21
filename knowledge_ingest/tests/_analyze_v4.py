"""Analyze v4 full-run results — quality metrics and distributions."""
import json, os
from collections import Counter

RUN_DIR = r"C:\ICT_Videos\Testing\_v4_full_run"

results = []
for f in os.listdir(RUN_DIR):
    if not f.endswith(".json") or f.startswith("_"):
        continue
    d = json.load(open(os.path.join(RUN_DIR, f), encoding="utf-8"))
    if d.get("error") or d.get("raw"):
        continue
    results.append(d)

print(f"Total valid results: {len(results)}")
print()

# Educator distribution (normalized)
def norm_educ(e):
    if not e:
        return "unknown"
    e = e.lower().strip()
    if "trader diego" in e or "trader-diego" in e:
        return "Trader-Diego"
    if "mmxm" in e:
        return "MMxM-trader"
    return e

educ = Counter(norm_educ(r.get("educator_guess", "")) for r in results)
print("Educator (normalized):")
for k, v in educ.most_common():
    pct = v / len(results) * 100
    print(f"  {k:20s} {v:4d} ({pct:.1f}%)")

# path_is_method
pim_true = sum(1 for r in results if r.get("path_is_method") is True)
pim_false = sum(1 for r in results if r.get("path_is_method") is False)
print(f"\npath_is_method: True={pim_true} ({pim_true/len(results)*100:.1f}%)  False={pim_false} ({pim_false/len(results)*100:.1f}%)")

# Pages with sequences
has_seq = sum(1 for r in results if r.get("n_seq", 0) > 0)
print(f"Pages with sequence: {has_seq} ({has_seq/len(results)*100:.1f}%)")

# Pages with entry mechanics
has_em = sum(1 for r in results if r.get("entry_mechanics"))
print(f"Pages with entry_mechanics: {has_em} ({has_em/len(results)*100:.1f}%)")

# Pages with concepts
has_concepts = sum(1 for r in results if r.get("concepts_raw"))
print(f"Pages with concepts_raw: {has_concepts} ({has_concepts/len(results)*100:.1f}%)")

# Pages with inferred
has_inferred = sum(1 for r in results if r.get("inferred"))
print(f"Pages with inferred text: {has_inferred} ({has_inferred/len(results)*100:.1f}%)")

# Avg concepts per page
avg_concepts = sum(len(r.get("concepts_raw", [])) for r in results) / len(results)
print(f"Avg concepts per page: {avg_concepts:.1f}")

# By source type
standalone = [r for r in results if r.get("input", {}).get("source") == "standalone"]
pdf_pages = [r for r in results if r.get("input", {}).get("source") == "pdf_page"]
if standalone:
    avg_seq_s = sum(r.get("n_seq", 0) for r in standalone) / len(standalone)
    pim_s = sum(1 for r in standalone if r.get("path_is_method") is True)
    print(f"\nStandalone images: {len(standalone)} (avg seq={avg_seq_s:.1f}, pim_true={pim_s}/{len(standalone)})")
if pdf_pages:
    avg_seq_p = sum(r.get("n_seq", 0) for r in pdf_pages) / len(pdf_pages)
    pim_p = sum(1 for r in pdf_pages if r.get("path_is_method") is True)
    print(f"PDF pages: {len(pdf_pages)} (avg seq={avg_seq_p:.1f}, pim_true={pim_p}/{len(pdf_pages)})")

# Framework distribution
fw = Counter(str(r.get("framework")) for r in results)
print(f"\nFramework distribution (top 10):")
for k, v in fw.most_common(10):
    pct = v / len(results) * 100
    print(f"  {k:25s} {v:4d} ({pct:.1f}%)")