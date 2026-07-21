"""Test the v4 LanceDB vector store with semantic searches."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import lancedb
from knowledge_ingest.pipeline.vector_store import search

DB = r"C:\ICT_Videos\Testing\_v4_lancedb"

# Verify table
db = lancedb.connect(DB)
tbl = db.open_table("knowledge")
print(f"LanceDB table: {tbl.count_rows()} rows")

# Run test searches
queries = [
    "sharp turn entry model FVG displacement",
    "power of three accumulation manipulation distribution",
    "MMXM market maker sell model liquidity sweep",
    "NY session profiling killzones 9:30 open",
    "order block fair value gap CSD entry",
]

for q in queries:
    print(f"\n{'='*70}")
    print(f"Search: \"{q}\"")
    print(f"{'='*70}")
    results = search(q, db_path=DB, k=3)
    for i, r in enumerate(results, 1):
        src = r.get("source_file", "")[:40]
        kt = r.get("knowledge_type", "?")
        conf = r.get("confidence", 0)
        rt = r.get("retrieval_text", "")[:150]
        concepts = r.get("concepts", "")
        print(f"  {i}. [{kt}] {src:40s} conf={conf:.1f}")
        print(f"     concepts: {concepts[:80]}")
        print(f"     text: {rt}")

print("\n\nDone!")