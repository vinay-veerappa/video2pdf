"""
Shared collision check for multi-dir tools.

The 5 (or 3) output folders from one source are treated as ONE logical knowledge
base. That only works if filename stems are unique across folders — otherwise two
different sessions could share a unit_id and clobber each other. Per the user's
choice: do NOT auto-namespace silently; FLAG collisions loudly so a file can be
renamed deliberately.

Every multi-dir tool (report, vector-store builder, re-canonicalize pass) calls
check_collisions() first and refuses to proceed on a colliding set.
"""

import os
import glob
from collections import defaultdict


def check_collisions(units_dirs):
    """Return dict{stem: [dirs...]} for stems appearing in >1 dir. Empty = clean."""
    stem_to_dirs = defaultdict(list)
    for d in units_dirs:
        for fp in glob.glob(os.path.join(d, "*.jsonl")):
            stem = os.path.basename(fp)[:-len(".jsonl")]
            stem_to_dirs[stem].append(d)
    return {s: ds for s, ds in stem_to_dirs.items() if len(ds) > 1}


def assert_no_collisions(units_dirs):
    """Halt with a clear report if any stem collides across dirs."""
    collisions = check_collisions(units_dirs)
    if collisions:
        print("\n!!! FILENAME COLLISION across units dirs — cannot treat as one base.")
        print("    Rename one of each colliding file and re-run:\n")
        for stem, ds in sorted(collisions.items()):
            print(f"    '{stem}' appears in:")
            for d in ds:
                print(f"        {d}")
        raise SystemExit(1)
    print(f"[ok] no filename collisions across {len(units_dirs)} dirs")
