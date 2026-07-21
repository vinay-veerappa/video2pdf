"""Compare gemma4 vs qwen3.5 on the 435 lumitrader pages — dual-model agreement."""
import json, os

GEMMA_DIR = r"C:\ICT_Videos\Testing\_v4_full_run"
QWEN_DIR = r"C:\ICT_Videos\Testing\_v4_lumitrader"

# Load all lumitrader results from both runs
def load_run(d):
    results = {}
    for f in os.listdir(d):
        if not f.endswith(".json") or f.startswith("_"):
            continue
        if "lumitrader" not in f:
            continue
        data = json.load(open(os.path.join(d, f), encoding="utf-8"))
        if data.get("error") or data.get("raw"):
            continue
        results[f.replace(".json", "")] = data
    return results

gemma = load_run(GEMMA_DIR)
qwen = load_run(QWEN_DIR)

print(f"Gemma4 lumitrader pages: {len(gemma)}")
print(f"Qwen3.5 lumitrader pages: {len(qwen)}")

common = set(gemma.keys()) & set(qwen.keys())
print(f"Common pages: {len(common)}")

# Compare key fields
pim_agree = 0
pim_disagree = 0
fw_agree = 0
fw_disagree = 0
educ_agree = 0
educ_disagree = 0
seq_diffs = []

for stem in sorted(common):
    g = gemma[stem]
    q = qwen[stem]

    # path_is_method
    g_pim = g.get("path_is_method")
    q_pim = q.get("path_is_method")
    if g_pim == q_pim:
        pim_agree += 1
    else:
        pim_disagree += 1

    # framework (normalize None/null/empty)
    g_fw = str(g.get("framework") or "").lower().strip()
    q_fw = str(q.get("framework") or "").lower().strip()
    if g_fw == q_fw:
        fw_agree += 1
    else:
        fw_disagree += 1

    # educator
    g_ed = str(g.get("educator_guess") or "").lower().strip()
    q_ed = str(q.get("educator_guess") or "").lower().strip()
    if g_ed == q_ed:
        educ_agree += 1
    else:
        educ_disagree += 1

    # sequence count
    g_seq = g.get("n_seq", 0)
    q_seq = q.get("n_seq", 0)
    if g_seq != q_seq:
        seq_diffs.append((stem, g_seq, q_seq))

total = len(common)
print(f"\n=== Agreement on {total} common lumitrader pages ===")
print(f"path_is_method:  agree={pim_agree} ({pim_agree/total*100:.1f}%)  disagree={pim_disagree} ({pim_disagree/total*100:.1f}%)")
print(f"framework:        agree={fw_agree} ({fw_agree/total*100:.1f}%)  disagree={fw_disagree} ({fw_disagree/total*100:.1f}%)")
print(f"educator:         agree={educ_agree} ({educ_agree/total*100:.1f}%)  disagree={educ_disagree} ({educ_disagree/total*100:.1f}%)")
print(f"sequence count:   same={total-len(seq_diffs)} ({(total-len(seq_diffs))/total*100:.1f}%)  different={len(seq_diffs)} ({len(seq_diffs)/total*100:.1f}%)")

# Show some disagreements
if pim_disagree > 0:
    print(f"\n--- path_is_method disagreements (first 10) ---")
    for stem in sorted(common):
        g = gemma[stem]
        q = qwen[stem]
        if g.get("path_is_method") != q.get("path_is_method"):
            print(f"  {stem}: gemma={g.get('path_is_method')} qwen={q.get('path_is_method')}")

if fw_disagree > 0:
    print(f"\n--- framework disagreements (first 10) ---")
    count = 0
    for stem in sorted(common):
        g = gemma[stem]
        q = qwen[stem]
        g_fw = str(g.get("framework") or "").lower().strip()
        q_fw = str(q.get("framework") or "").lower().strip()
        if g_fw != q_fw and count < 10:
            print(f"  {stem}: gemma={g.get('framework')} qwen={q.get('framework')}")
            count += 1

if seq_diffs:
    print(f"\n--- sequence count differences (first 10) ---")
    for stem, gs, qs in sorted(seq_diffs)[:10]:
        print(f"  {stem}: gemma_seq={gs} qwen_seq={qs}")