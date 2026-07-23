"""
Merge chart-derived units (v4) + transcript-derived units into a unified LanceDB.

Reads from:
  - C:\\ICT_Videos\\Testing\\_v4_units\\v4_chart_units.jsonl  (818 chart units)
  - <transcript_output>/units/*.jsonl  (transcript units, if they exist)

Writes a unified LanceDB at C:\\ICT_Videos\\Testing\\unified_knowledge.lancedb

Usage:
  python -m knowledge_ingest.merge_knowledge_base
  python -m knowledge_ingest.merge_knowledge_base --transcript-dir C:\\path\\to\\transcript\\ingest_output
"""

import os, sys, json, argparse, glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knowledge_ingest.pipeline.vector_store import build_lancedb, embed_ollama
from knowledge_ingest.paths import unified_db_path, units_dir as kb_units_dir
import lancedb

# Default paths. Chart units (the v4 chart-extract output) remain external
# (raw input artifact); the unified DB and transcript units default to the
# consumer-owned KB data tree (see knowledge_ingest/paths.py).
CHART_UNITS = r"C:\ICT_Videos\Testing\_v4_units\v4_chart_units.jsonl"
DEFAULT_TRANSCRIPT_DIR = os.path.join(kb_units_dir(), "tcm_2023", "units")
UNIFIED_DB = unified_db_path()


def collect_unit_dirs(chart_path, transcript_dirs=None):
    """Collect all unit JSONL directories to merge.

    transcript_dirs may be None, a str, or a list of str.
    """
    dirs = []

    # Chart units (single JSONL file → treat as a dir)
    chart_dir = os.path.dirname(chart_path)
    if os.path.exists(chart_dir):
        dirs.append(chart_dir)
        print(f"  Chart units: {chart_dir} ({count_jsonl(chart_dir)} files)")

    # Transcript units (one or more dirs)
    if transcript_dirs is None:
        transcript_dirs = []
    elif isinstance(transcript_dirs, str):
        transcript_dirs = [transcript_dirs]

    for tdir in transcript_dirs:
        if tdir and os.path.exists(tdir):
            dirs.append(tdir)
            print(f"  Transcript units: {tdir} ({count_jsonl(tdir)} files)")
        elif tdir:
            print(f"  WARNING: transcript dir not found: {tdir}")

    return dirs


def count_jsonl(d):
    return len(glob.glob(os.path.join(d, "*.jsonl")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript-dir", nargs="+", default=[DEFAULT_TRANSCRIPT_DIR],
                    help="one or more directories containing transcript unit JSONL files")
    ap.add_argument("--chart-units", default=CHART_UNITS,
                    help="path to chart unit JSONL file")
    ap.add_argument("--db", default=UNIFIED_DB,
                    help="output LanceDB path")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    args = ap.parse_args()

    print("=" * 60)
    print("Unified Knowledge Base Builder")
    print("=" * 60)

    dirs = collect_unit_dirs(args.chart_units, args.transcript_dir)

    if not dirs:
        print("ERROR: No unit directories found!")
        return

    if len(dirs) > 1:
        from knowledge_ingest.multidir import assert_no_collisions
        try:
            assert_no_collisions(dirs)
        except Exception as e:
            print(f"WARNING: collision detected: {e}")
            print("Proceeding anyway (chart units use 'chart__' prefix)")

    print(f"\nBuilding unified LanceDB at {args.db}...")
    print(f"  Embedding model: {args.embed_model}")

    from knowledge_ingest.config.config import PipelineConfig
    cfg = PipelineConfig()

    build_lancedb(
        units_dirs=dirs,
        db_path=args.db,
        embed_model=args.embed_model,
    )

    # Verify
    db = lancedb.connect(args.db)
    tbl = db.open_table("knowledge")
    count = tbl.count_rows()
    print(f"\n{'='*60}")
    print(f"Unified knowledge base: {count} units")
    print(f"DB: {args.db}")
    print(f"{'='*60}")

    # Show distribution
    import pyarrow as pa
    data = tbl.to_pandas()
    print(f"\nBy knowledge_type:")
    for kt, cnt in data["knowledge_type"].value_counts().items():
        print(f"  {kt:15s} {cnt:4d}")
    print(f"\nBy source_type:")
    for st, cnt in data.get("source_type", pd.Series()).value_counts().items() if "source_type" in data.columns else []:
        print(f"  {st:15s} {cnt:4d}")


if __name__ == "__main__":
    import pandas as pd
    main()