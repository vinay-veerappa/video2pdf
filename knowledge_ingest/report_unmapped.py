"""
Unmapped-concepts report — a per-DOMAIN tool, reusable across every knowledge base.

For a given units directory and a given vocabulary domain, it finds every
`concepts_raw` phrase that FAILED to map to a canonical id, ranks them by
frequency, and prints paste-ready stubs to grow that domain's vocabulary.

Why per-domain: ICT liquidity terms and GEX/gamma terms barely overlap. Running
this against the ICT vocab tells you what ICT concepts you're missing; against
the GEX vocab, what GEX concepts you're missing. Same tool, different lens.

Usage:
    python -m knowledge_ingest.report_unmapped --units <dir> --vocab ict
    python -m knowledge_ingest.report_unmapped --units <dir> --vocab gex --min-count 3

Reads only existing output; never touches the pipeline. Run it AFTER a batch
completes (and, per your plan, after the concept-overview directory is ingested,
where vocabulary is richest).
"""

import os
import sys
import json
import glob
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge_ingest.vocab.registry import load_domain


def _canonical_id_stub(raw: str) -> str:
    """Suggest a snake_case canonical id from a raw phrase."""
    import re
    s = re.sub(r'[^a-z0-9]+', '_', raw.lower()).strip('_')
    return s[:40] or "unnamed_concept"


def collect(units_dirs):
    """Aggregate concepts_raw across ALL units dirs. Accepts a list of dirs."""
    if isinstance(units_dirs, str):
        units_dirs = [units_dirs]
    raw_counter = Counter()
    raw_to_files = defaultdict(set)
    raw_to_types = defaultdict(Counter)
    n_files = 0
    for units_dir in units_dirs:
        files = glob.glob(os.path.join(units_dir, "*.jsonl"))
        n_files += len(files)
        for fp in files:
            stem = os.path.basename(fp)
            for line in open(fp, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                meta = rec.get("metadata", {})
                for raw in meta.get("concepts_raw", []) or []:
                    raw_counter[raw] += 1
                    raw_to_files[raw].add(stem)
                    raw_to_types[raw][meta.get("knowledge_type", "?")] += 1
    return raw_counter, raw_to_files, raw_to_types, n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, nargs="+",
                    help="one or more units/ dirs (treated as one logical base)")
    ap.add_argument("--vocab", required=True, help="vocab domain name, e.g. 'ict'")
    ap.add_argument("--min-count", type=int, default=2,
                    help="only report concepts appearing at least this many times")
    ap.add_argument("--out", default=None, help="optional path to write a stub file")
    args = ap.parse_args()

    from knowledge_ingest.multidir import assert_no_collisions
    if len(args.units) > 1:
        assert_no_collisions(args.units)

    vocab = load_domain(args.vocab)
    raw_counter, raw_to_files, raw_to_types, n_files = collect(args.units)

    unmapped = Counter()
    for raw, count in raw_counter.items():
        # a raw concept is "mapped" if map_to_canonical returns anything for it
        if not vocab.map_to_canonical([raw]):
            unmapped[raw] = count

    total_raw = len(raw_counter)
    mapped = total_raw - len(unmapped)
    print(f"\n=== Unmapped-concepts report — domain '{args.vocab}' ===")
    print(f"files scanned: {n_files}")
    print(f"distinct raw concepts: {total_raw} | mapped: {mapped} "
          f"| unmapped: {len(unmapped)} ({100*len(unmapped)/max(1,total_raw):.0f}%)")
    print(f"showing unmapped with count >= {args.min_count}, by frequency:\n")

    stubs = []
    for raw, count in unmapped.most_common():
        if count < args.min_count:
            continue
        types = ", ".join(f"{k}:{v}" for k, v in raw_to_types[raw].most_common())
        print(f"  [{count:3}x] {raw!r}   (types: {types}; in {len(raw_to_files[raw])} files)")
        cid = _canonical_id_stub(raw)
        stubs.append(
            f'    "{cid}": {{\n'
            f'        "label": "{raw}",\n'
            f'        "category": "TODO",\n'
            f'        "aliases": ["{raw.lower()}"],\n'
            f'    }},'
        )

    print(f"\n--- paste-ready stubs for vocab/{vocab.__name__.split(".")[-1]}.py "
          f"(edit label/category/aliases, then merge aliases where they're the same concept) ---\n")
    stub_text = "\n".join(stubs)
    print(stub_text or "  (nothing above the count threshold)")

    if args.out and stubs:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# candidate canonical entries — review, merge duplicates, set category\n")
            f.write("CANDIDATES = {\n" + stub_text + "\n}\n")
        print(f"\nwrote stubs to {args.out}")

    print("\nNext: merge aliases for concepts that are the SAME idea phrased differently "
          "(that's the whole point of canonicalization), set categories, drop noise.")


if __name__ == "__main__":
    main()
