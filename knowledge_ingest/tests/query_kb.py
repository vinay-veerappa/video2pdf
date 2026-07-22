"""
Interactive knowledge base query tool.

Usage:
  python -m knowledge_ingest.tests.query_kb
  python -m knowledge_ingest.tests.query_kb --db "C:\\ICT_Videos\\Testing\\_v4_lancedb"
  python -m knowledge_ingest.tests.query_kb --text-dir "C:\\ICT_Videos\\Testing\\_text_ict_ingest\\units"

Features:
  - Semantic search across chart + text knowledge units
  - Filter by knowledge_type, session, testability, min_confidence
  - Show unit details (summary, payload, concepts, provenance)
  - Stats: unit counts, type distribution, confidence histogram
"""

import os, sys, json, argparse, glob
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CHART_DB = r"C:\ICT_Videos\Testing\_v4_lancedb"
TEXT_UNITS_DIR = r"C:\ICT_Videos\Testing\_text_ict_ingest\units"


def stats(db_path, table="knowledge"):
    """Print database statistics."""
    import lancedb
    db = lancedb.connect(db_path)
    tbl = db.open_table(table)
    count = tbl.count_rows()
    print(f"\n{'='*60}")
    print(f"Knowledge Base: {db_path}")
    print(f"Table: {table}")
    print(f"Total units: {count}")
    print(f"{'='*60}")

    data = tbl.to_pandas()
    print(f"\nBy knowledge_type:")
    for kt, cnt in data["knowledge_type"].value_counts().items():
        print(f"  {kt:15s} {cnt:4d}")

    if "confidence" in data.columns:
        print(f"\nConfidence distribution:")
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        for lo, hi in bins:
            n = ((data["confidence"] >= lo) & (data["confidence"] < hi)).sum()
            bar = "█" * int(n / max(count, 1) * 50)
            print(f"  {lo:.1f}-{hi:.1f}: {n:4d} {bar}")

    if "source_file" in data.columns:
        print(f"\nTop 10 sources:")
        for src, cnt in data["source_file"].value_counts().head(10).items():
            print(f"  {cnt:3d}  {src[:50]}")

    if "sessions" in data.columns:
        print(f"\nSessions covered:")
        for sess, cnt in data["sessions"].value_counts().head(10).items():
            if sess and sess != "[]":
                print(f"  {sess[:30]:30s} {cnt:4d}")


def do_search(query, db_path, table="knowledge", k=8, knowledge_type=None,
              session=None, testability=None, min_confidence=0.0, verbose=False):
    """Run a semantic search and display results."""
    from knowledge_ingest.pipeline.vector_store import search

    results = search(
        query, db_path=db_path, table=table, k=k,
        knowledge_type=knowledge_type, session=session,
        testability=testability, min_confidence=min_confidence,
    )

    print(f"\n{'='*60}")
    print(f"Query: \"{query}\"")
    filters = []
    if knowledge_type: filters.append(f"type={knowledge_type}")
    if session: filters.append(f"session={session}")
    if testability: filters.append(f"testability={testability}")
    if min_confidence: filters.append(f"conf>={min_confidence}")
    print(f"Filters: {', '.join(filters) or 'none'}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        score = r.get("_distance", r.get("distance", 0))
        ktype = r.get("knowledge_type", "?")
        src = r.get("source_file", "?")[:40]
        conf = r.get("confidence", 0)
        print(f"\n  [{i}] [{ktype:12s}] {src:40s} conf={conf:.1f} dist={score:.3f}")
        print(f"      {r.get('retrieval_text', '')[:150]}")
        if verbose:
            print(f"      concepts: {r.get('concepts', '')}")
            print(f"      sessions: {r.get('sessions', '')}")
            print(f"      speaker: {r.get('speaker', '')}")
            print(f"      chunk: {r.get('chunk_id', '')}")
    return results


def text_stats(text_dir):
    """Print stats for text units not yet in LanceDB."""
    files = glob.glob(str(Path(text_dir) / "*.jsonl"))
    if not files:
        print(f"\nNo text units found at {text_dir}")
        return

    total = 0
    by_type = {}
    by_speaker = {}
    confidences = []

    for fp in files:
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            u = json.loads(line)
            total += 1
            kt = u["metadata"]["knowledge_type"]
            by_type[kt] = by_type.get(kt, 0) + 1
            sp = u["provenance"].get("speaker", "unknown")
            by_speaker[sp] = by_speaker.get(sp, 0) + 1
            confidences.append(u["metadata"].get("extraction_confidence", 0))

    print(f"\n{'='*60}")
    print(f"Text units: {text_dir}")
    print(f"Total: {total} units from {len(files)} transcripts")
    print(f"{'='*60}")
    print(f"\nBy knowledge_type:")
    for kt, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {kt:15s} {cnt:4d}")
    print(f"\nBy speaker:")
    for sp, cnt in sorted(by_speaker.items(), key=lambda x: -x[1]):
        print(f"  {sp:15s} {cnt:4d}")
    if confidences:
        avg = sum(confidences) / len(confidences)
        print(f"\nAvg confidence: {avg:.2f}")
        below_06 = sum(1 for c in confidences if c < 0.6)
        print(f"  Below 0.6: {below_06}/{total} ({below_06/total*100:.0f}%)")


def interactive(db_path, table="knowledge"):
    """Interactive REPL for querying the knowledge base."""
    print(f"\nInteractive Knowledge Base Query")
    print(f"DB: {db_path}, Table: {table}")
    print(f"Commands: search <query>, stats, type <type>, conf <min>, verbose, quit")
    print()

    k = 8
    knowledge_type = None
    min_confidence = 0.0
    verbose = False

    while True:
        try:
            cmd = input("kb> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue
        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "stats":
            stats(db_path, table)
        elif cmd.startswith("type "):
            knowledge_type = cmd[5:].strip() or None
            print(f"  filter: knowledge_type={knowledge_type}")
        elif cmd == "type":
            knowledge_type = None
            print(f"  filter: knowledge_type=none")
        elif cmd.startswith("conf "):
            min_confidence = float(cmd[5:])
            print(f"  filter: min_confidence={min_confidence}")
        elif cmd == "conf":
            min_confidence = 0.0
            print(f"  filter: min_confidence=0.0")
        elif cmd == "verbose":
            verbose = not verbose
            print(f"  verbose: {verbose}")
        elif cmd.startswith("k "):
            k = int(cmd[2:])
            print(f"  k={k}")
        elif cmd.startswith("search "):
            query = cmd[7:]
            do_search(query, db_path, table, k=k,
                      knowledge_type=knowledge_type,
                      min_confidence=min_confidence, verbose=verbose)
        elif cmd == "help":
            print("Commands:")
            print("  search <query>   - semantic search")
            print("  stats            - database statistics")
            print("  type <type>      - filter by knowledge_type (setup, framework, etc.)")
            print("  type             - clear type filter")
            print("  conf <min>       - set min confidence filter")
            print("  conf             - clear confidence filter")
            print("  verbose          - toggle verbose output")
            print("  k <n>            - set number of results")
            print("  quit             - exit")
        else:
            # treat as direct search
            do_search(cmd, db_path, table, k=k,
                      knowledge_type=knowledge_type,
                      min_confidence=min_confidence, verbose=verbose)


def main():
    ap = argparse.ArgumentParser(description="Knowledge base query tool")
    ap.add_argument("--db", default=CHART_DB, help="LanceDB path")
    ap.add_argument("--table", default="knowledge")
    ap.add_argument("--text-dir", default=TEXT_UNITS_DIR, help="text units dir (for --text-stats)")
    ap.add_argument("--search", help="run a single search and exit")
    ap.add_argument("--type", help="filter by knowledge_type")
    ap.add_argument("--conf", type=float, default=0.0, help="min confidence")
    ap.add_argument("-k", type=int, default=8, help="number of results")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stats", action="store_true", help="print DB stats and exit")
    ap.add_argument("--text-stats", action="store_true", help="print text unit stats and exit")
    args = ap.parse_args()

    if args.text_stats:
        text_stats(args.text_dir)
        return

    if args.stats:
        stats(args.db, args.table)
        return

    if args.search:
        do_search(args.search, args.db, args.table, k=args.k,
                  knowledge_type=args.type, min_confidence=args.conf,
                  verbose=args.verbose)
        return

    # default: interactive
    stats(args.db, args.table)
    interactive(args.db, args.table)


if __name__ == "__main__":
    main()