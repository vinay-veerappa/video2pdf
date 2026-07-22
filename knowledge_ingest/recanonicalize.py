"""
Re-canonicalize pass.

Units ingested against an OLDER (smaller) vocabulary have stale
`concepts_canonical`. This walks every unit across one or more dirs and re-runs
map_to_canonical(concepts_raw) against the CURRENT vocabulary, rewriting
concepts_canonical in place. Pure string matching — no LLM calls, fast, cheap.

Run this after growing the vocabulary (and before build_lancedb), so retrieval
filters see the full mapping. Multi-dir aware, with the shared collision guard.

Usage:
    # preview what would change, no writes:
    python -m knowledge_ingest.recanonicalize --units <dir1> <dir2> --vocab ict --dry-run

    # apply (writes .bak backups unless --no-backup):
    python -m knowledge_ingest.recanonicalize --units <dir1> <dir2> --vocab ict
"""

import os
import sys
import json
import glob
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge_ingest.vocab.registry import load_domain
from knowledge_ingest.multidir import assert_no_collisions


def process_dir(units_dir, vocab, dry_run, backup):
    files = glob.glob(os.path.join(units_dir, "*.jsonl"))
    changed_units = 0
    changed_files = 0
    newly_mapped = 0   # units that gained at least one canonical id
    lost_mapped = 0    # units that lost one (shouldn't happen when growing vocab)

    for fp in files:
        lines = [l for l in open(fp, encoding="utf-8").read().splitlines() if l.strip()]
        out_lines = []
        file_changed = False
        for line in lines:
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            raw = meta.get("concepts_raw", []) or []
            old = meta.get("concepts_canonical", []) or []
            new = vocab.map_to_canonical(raw)
            if new != old:
                changed_units += 1
                file_changed = True
                if len(new) > len(old):
                    newly_mapped += 1
                elif len(new) < len(old):
                    lost_mapped += 1
                meta["concepts_canonical"] = new
                rec["metadata"] = meta
            out_lines.append(json.dumps(rec))
        if file_changed and not dry_run:
            if backup:
                shutil.copy2(fp, fp + ".bak")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")
            changed_files += 1

    return changed_units, changed_files, newly_mapped, lost_mapped, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, nargs="+",
                    help="one or more units/ dirs")
    ap.add_argument("--vocab", required=True, help="vocab domain, e.g. 'ict'")
    ap.add_argument("--dry-run", action="store_true", help="report only, no writes")
    ap.add_argument("--no-backup", action="store_true", help="skip .bak backups")
    args = ap.parse_args()

    if len(args.units) > 1:
        assert_no_collisions(args.units)

    vocab = load_domain(args.vocab)

    tot_units = tot_files = tot_new = tot_lost = tot_scanned = 0
    for d in args.units:
        cu, cf, nm, lm, nf = process_dir(d, vocab, args.dry_run, not args.no_backup)
        tot_units += cu; tot_files += cf; tot_new += nm
        tot_lost += lm; tot_scanned += nf
        print(f"  {d}: {cu} units changed across {nf} files")

    mode = "DRY-RUN (no writes)" if args.dry_run else "APPLIED"
    print(f"\n=== Re-canonicalize {mode} — domain '{args.vocab}' ===")
    print(f"files scanned: {tot_scanned}")
    print(f"units changed: {tot_units}  (gained mapping: {tot_new}, "
          f"lost mapping: {tot_lost})")
    if tot_lost:
        print("  ! some units LOST canonical ids — unexpected when growing a "
              "vocabulary. Check that aliases weren't removed.")
    if not args.dry_run and tot_files:
        print(f"files rewritten: {tot_files} (.bak backups "
              f"{'skipped' if args.no_backup else 'written'})")
    if args.dry_run:
        print("\nRe-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
