# Dependency-free tests of the pure (non-pydantic, non-LLM) logic.
import sys, re
from datetime import date, datetime
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vocab.ict_vocabulary import map_to_canonical, canonical_label, CANONICAL

print("=== VOCAB MAPPING (dedupe raw -> canonical) ===")
cases = [
    ["rebalance macro", "the 11:00 macro", "11 am macro"],   # all -> one id
    ["Judas swing", "sell-side liquidity", "CSD after the sweep"],
    ["Options Expiry week", "silver bullet", "quad witching"],
    ["9:12 macro", "three-shot rule", "we need speed"],
    ["Profile 6", "efficient overnight session"],
]
for raw in cases:
    canon = map_to_canonical(raw)
    labels = [canonical_label(c) for c in canon]
    print(f"  raw={raw}")
    print(f"    -> {canon}")
    print(f"    -> {labels}")

# assertion: the three phrasings of the 11:00 macro collapse to exactly one id
c = map_to_canonical(["rebalance macro", "the 11:00 macro", "11 am macro"])
assert c == ["macro_rebalance_1100"], c
print("  [PASS] three phrasings -> single canonical id")

print(f"\n  vocabulary size: {len(CANONICAL)} canonical concepts")

print("\n=== DATE PARSING FROM FILENAMES ===")
def parse_date(stem):
    m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', stem)
    if m: return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r'(\d{2})_?([A-Za-z]{3})[_ ](\d{1,2})[_ ](\d{4})', stem)
    if m:
        try: return datetime.strptime(f"{m.group(2)} {m.group(3)} {m.group(4)}", "%b %d %Y").date()
        except ValueError: return None
    return None

names = ["2023-05-08", "2023-05-11_PM", "05_May_11_2023__Review_", "05_May 11 2023 (Review)"]
for n in names:
    print(f"  {n!r:40} -> {parse_date(n)}")
assert parse_date("2023-05-08") == date(2023,5,8)
assert parse_date("05_May_11_2023__Review_") == date(2023,5,11)
print("  [PASS] both filename conventions parse")

print("\n=== TIMESTAMP-AWARE PRECHUNK FALLBACK ===")
TS = re.compile(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]')
sample = open("/mnt/user-data/uploads/05_May_11_2023__Review_.txt").read()
tss = TS.findall(sample)
print(f"  transcript chars={len(sample)}, timestamps found={len(tss)}")
print(f"  first ts={tss[0]}  last ts={tss[-1]}")
assert len(tss) > 5
print("  [PASS] timestamps extractable for segmentation")

print("\nAll pure-logic tests passed.")
