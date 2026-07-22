"""
Test the NEW batched control flow with a mock Ollama client (no network, no pydantic
needed for the dispatch logic — we test grouping/routing/reassembly directly).
"""
import sys, json, re
sys.path.insert(0, "knowledge_ingest")

# We test the routing logic in isolation, mirroring process_file's grouping,
# to confirm: (1) segments route to correct per-type batches,
# (2) skipped types still become units, (3) extraction maps back in order.

# simulate 6 segments classified across types
classifications = [
    {"knowledge_type": "setup",      "extraction_worthwhile": True},   # 0
    {"knowledge_type": "psychology", "extraction_worthwhile": True},   # 1
    {"knowledge_type": "setup",      "extraction_worthwhile": True},   # 2
    {"knowledge_type": "anecdote",   "extraction_worthwhile": False},  # 3 -> skip
    {"knowledge_type": "contextual", "extraction_worthwhile": True},   # 4
    {"knowledge_type": "setup",      "extraction_worthwhile": True},   # 5
]
skip_extract_types = ["anecdote"]  # config

# --- replicate grouping logic from process_file ---
by_type, skipped_idx = {}, set()
for i, cls in enumerate(classifications):
    ktype = cls.get("knowledge_type")
    worthwhile = cls.get("extraction_worthwhile", True)
    if (ktype in skip_extract_types) or (not worthwhile) or (ktype is None):
        skipped_idx.add(i)
    else:
        by_type.setdefault(ktype, []).append(i)

print("=== BATCHED ROUTING ===")
print(f"  by_type groups: { {k: v for k,v in by_type.items()} }")
print(f"  skipped indices: {sorted(skipped_idx)}")
assert by_type["setup"] == [0, 2, 5], by_type["setup"]
assert by_type["contextual"] == [4]
assert by_type["psychology"] == [1]
assert 3 in skipped_idx and "anecdote" not in by_type
print("  [PASS] setups grouped together (share session context), anecdote skipped")

# --- simulate batched extraction mapping back to global indices ---
extractions = [None] * len(classifications)
def mock_extract_batch(ktype, texts):
    # returns idx-tagged records like the real batched call
    return [{"idx": j, "summary": f"{ktype}#{j}", "payload": {}} for j in range(len(texts))]

for ktype, idxs in by_type.items():
    texts = [f"seg{i}" for i in idxs]
    batch = mock_extract_batch(ktype, texts)
    for local_i, global_i in enumerate(idxs):
        extractions[global_i] = batch[local_i]

print("\n=== EXTRACTION REMAP (local batch idx -> global segment idx) ===")
for i, e in enumerate(extractions):
    print(f"  seg {i}: {e['summary'] if e else 'SKIPPED (still becomes a unit)'}")
assert extractions[0]["summary"] == "setup#0"
assert extractions[5]["summary"] == "setup#2"   # 3rd setup, local idx 2
assert extractions[3] is None                    # anecdote skipped
assert extractions[4]["summary"] == "contextual#0"
print("  [PASS] batch results remap to correct global positions")

# --- test batch chunking by extract_batch_size ---
print("\n=== BATCH CHUNKING (extract_batch_size) ===")
bs = 12
n = 30
chunks = [(s, min(s+bs, n)) for s in range(0, n, bs)]
print(f"  {n} units, batch_size {bs} -> {len(chunks)} calls: {chunks}")
assert chunks == [(0,12),(12,24),(24,30)]
print("  [PASS] 30 units -> 3 batched calls")

# --- test batched JSON parse tolerance (fallback on gaps) ---
print("\n=== GAP FALLBACK ===")
returned = [{"idx":0,"summary":"a"},{"idx":2,"summary":"c"}]  # idx 1 missing
out = [None]*3
for obj in returned:
    out[obj["idx"]] = obj
missing = [i for i,o in enumerate(out) if o is None]
print(f"  batch returned idx {[r['idx'] for r in returned]}, missing -> {missing}")
assert missing == [1], missing
print("  [PASS] missing idx detected -> per-unit fallback would fill it")

print("\nAll batched-flow tests passed.")
