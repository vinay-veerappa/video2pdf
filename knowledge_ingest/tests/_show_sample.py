"""Quick quality sampler — prints v4 outputs for spot-check pages."""
import json, os

RUN_DIR = r"C:\ICT_Videos\Testing\_v4_full_run"

# Pages to sample: standalone + representative PDF pages
SAMPLES = [
    "DailyPo3",
    "Arjo15mSTEntryModel",
    "BSL_DOL",
    "LRS",
    "lumitrader-ict-2022-book_p067",
    "lumitrader-ict-2022-book_p230",
    "Vinay_Models_p047",
    "Vinay_Models_p075",
    "ICTNotes_p022",
    "Flux_NY_Guide_p040",
    "MMXM_p017",
    "Lecture6-12_p007",
]

for stem in SAMPLES:
    f = os.path.join(RUN_DIR, f"{stem}.json")
    if not os.path.exists(f):
        print(f"\n{'='*60}")
        print(f"{stem}: MISSING")
        continue
    d = json.load(open(f, encoding="utf-8"))
    if d.get("error") or d.get("raw"):
        print(f"\n{'='*60}")
        print(f"{stem}: ERROR / PARSE FAIL")
        continue

    print(f"\n{'='*60}")
    print(f"{stem}")
    print(f"  educator: {d.get('educator_guess')}")
    print(f"  framework: {d.get('framework')}")
    print(f"  path_is_method: {d.get('path_is_method')}")
    print(f"  n_seq: {d.get('n_seq')}")
    print(f"  concepts: {d.get('concepts_raw', [])}")
    em = d.get("entry_mechanics", [])
    if em:
        for e in em:
            print(f"  entry_mech: {e.get('name')} — {e.get('description','')[:80]}")
    inferred = d.get("inferred", [])
    for i in inferred[:3]:
        print(f"  inferred: {i[:120]}")