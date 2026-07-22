"""
Verify a recanonicalize run was correct and safe.

Read-only: never touches units. Compares each *.jsonl against its *.jsonl.bak
backup to confirm:
  1. Every line in the new file still parses as valid JSON.
  2. No unit LOST any canonical id (only gains allowed when growing vocab /
     collecting multi-matches). A loss is a regression — flag it.
  3. Spot-checks units that gained the MOST new canonical ids, printing
     before/after so a human can catch over-matching (e.g. a setup unit suddenly
     tagged with `video` or `circuit_breaker`).
  4. Summarizes: total units, units changed, ids gained, ids lost, top new ids.

Usage:
    python knowledge_ingest/tests/verify_recanonicalize.py --units <dir1> <dir2> ...
    python knowledge_ingest/tests/verify_recanonicalize.py --units <dir> --sample 20
"""

import os
import sys
import json
import glob
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_units(path):
    """Load units from a JSONL file keyed by unit_id."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["unit_id"]] = rec
    return out


def canonical_set(rec):
    return set(rec.get("metadata", {}).get("concepts_canonical", []) or [])


def verify_dir(units_dir, sample=10):
    files = sorted(glob.glob(os.path.join(units_dir, "*.jsonl")))
    # exclude .bak from the glob — it doesn't match *.jsonl anyway
    stats = {
        "files": 0,
        "units": 0,
        "units_changed": 0,
        "ids_gained": Counter(),
        "ids_lost": Counter(),
        "units_with_losses": 0,
        "parse_errors": 0,
        "no_backup": 0,
        "biggest_gainers": [],   # (gain_count, unit_id, before, after, ktype)
    }

    for fp in files:
        bak = fp + ".bak"
        if not os.path.exists(bak):
            stats["no_backup"] += 1
            continue
        stats["files"] += 1

        # validate new file parses fully
        try:
            new_units = load_units(fp)
        except Exception as e:
            stats["parse_errors"] += 1
            print(f"  ! PARSE ERROR in {fp}: {e}")
            continue
        try:
            old_units = load_units(bak)
        except Exception as e:
            print(f"  ! could not parse backup {bak}: {e}")
            continue

        stats["units"] += len(new_units)

        for uid, new_rec in new_units.items():
            old_rec = old_units.get(uid)
            if old_rec is None:
                # unit exists in new but not old — shouldn't happen on recanon
                continue
            old_c = canonical_set(old_rec)
            new_c = canonical_set(new_rec)
            stats["units"] += 0
            if old_c == new_c:
                continue
            stats["units_changed"] += 1
            gained = new_c - old_c
            lost = old_c - new_c
            for g in gained:
                stats["ids_gained"][g] += 1
            for l in lost:
                stats["ids_lost"][l] += 1
            if lost:
                stats["units_with_losses"] += 1
            if gained and len(gained) >= 2:
                ktype = new_rec.get("metadata", {}).get("knowledge_type", "?")
                stats["biggest_gainers"].append(
                    (len(gained), uid, sorted(old_c), sorted(new_c), ktype)
                )

    stats["biggest_gainers"].sort(reverse=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, nargs="+",
                    help="one or more units/ dirs to verify against their .bak")
    ap.add_argument("--sample", type=int, default=10,
                    help="how many biggest-gainer units to print per dir")
    args = ap.parse_args()

    grand = dict(files=0, units=0, units_changed=0, units_with_losses=0,
                 parse_errors=0, no_backup=0,
                 ids_gained=Counter(), ids_lost=Counter(), biggest_gainers=[])
    for d in args.units:
        print(f"\n=== {d} ===")
        s = verify_dir(d, args.sample)
        print(f"  files with backups : {s['files']}")
        print(f"  files w/o backup   : {s['no_backup']}")
        print(f"  units              : {s['units']}")
        print(f"  units changed      : {s['units_changed']}")
        print(f"  units with LOSSES  : {s['units_with_losses']}  (regression — must be 0)")
        print(f"  parse errors       : {s['parse_errors']}")
        print(f"  top 10 ids GAINED:")
        for cid, n in s["ids_gained"].most_common(10):
            print(f"    +{n:5d}  {cid}")
        if s["ids_lost"]:
            print(f"  !! ids LOST (REGRESSION):")
            for cid, n in s["ids_lost"].most_common(10):
                print(f"    -{n:5d}  {cid}")
        print(f"  top {args.sample} biggest-gainer units (gain>=2) — eyeball for over-match:")
        for gain, uid, before, after, ktype in s["biggest_gainers"][:args.sample]:
            print(f"    [{ktype}] {uid}  +{gain}")
            print(f"      before: {before}")
            print(f"      after : {after}")
        # merge into grand
        for k in ("files", "units", "units_changed", "units_with_losses",
                  "parse_errors", "no_backup"):
            grand[k] += s[k]
        grand["ids_gained"].update(s["ids_gained"])
        grand["ids_lost"].update(s["ids_lost"])
        grand["biggest_gainers"].extend(s["biggest_gainers"])

    print("\n" + "=" * 60)
    print("GRAND TOTAL")
    print("=" * 60)
    print(f"  files with backups : {grand['files']}")
    print(f"  units              : {grand['units']}")
    print(f"  units changed      : {grand['units_changed']}")
    print(f"  units with LOSSES  : {grand['units_with_losses']}  (regression — must be 0)")
    print(f"  parse errors       : {grand['parse_errors']}")
    print(f"  files w/o backup   : {grand['no_backup']}")
    print(f"  total distinct ids gained: {len(grand['ids_gained'])}")
    print(f"  total distinct ids lost : {len(grand['ids_lost'])}")
    print(f"  top 20 ids GAINED across all dirs:")
    for cid, n in grand["ids_gained"].most_common(20):
        print(f"    +{n:5d}  {cid}")
    if grand["ids_lost"]:
        print(f"  !! IDS LOST (REGRESSION):")
        for cid, n in grand["ids_lost"].most_common(20):
            print(f"    -{n:5d}  {cid}")

    # exit code: 0 only if no losses and no parse errors
    ok = (grand["units_with_losses"] == 0 and grand["parse_errors"] == 0)
    print(f"\nVERIFICATION: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()