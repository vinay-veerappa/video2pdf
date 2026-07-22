"""
Build the Phase 1 bake-off input set.

Selects a manageable, diverse set of images for the VLM consensus bake-off:
  - all 7 hand-labeled standalone images (already labeled in HANDOVER §13d)
  - 16 co-dependent pages from MMXM.pdf (the genuine mixed-doc stress case)
  - 6 co-dependent pages from PRE-MARKET PLAN.pdf
  - 10 sampled pages from Flux_NY_Guide.pdf (full-page-image, no text layer)
  - 10 sampled pages from lumitrader-ict-2022-book.pdf (full-page-image, no text layer)
  - all 3 pages of Delirium.pdf (small, has a co-dep page)

Writes:
  bakeoff_inputs.jsonl  - one line per input with: path, source, page, true_kind
                          (true_kind is "TODO" for PDF pages — user fills in
                          after eyeballing; the bake-off can run with TODO and
                          we compare only on the labeled subset first)

Usage:
    python knowledge_ingest/tests/build_bakeoff_inputs.py
"""

import os, json, glob
from pathlib import Path

TESTING = Path(r"C:\ICT_Videos\Testing")
RENDER_DIR = TESTING / "_triage_renders"
OUT = TESTING / "bakeoff_inputs.jsonl"

# --- 1. standalone images (hand-labeled in HANDOVER §13d) ---
STANDALONE = [
    # (filename, kind, case, seq_faithful, co_dependent, notes)
    ("Arjo15mSTEntryModel.png",  "annotated_chart",  "C", "weak", "yes", "§13d: real trade screenshot"),
    ("BSL_DOL.webp",             "price_path",        "B", "yes", "yes", "§13d: granularity 5/6/8"),
    ("DailyPo3.png",             "reference_diagram", "A", "no",  "no",  "§13d: mirrored bias checklists"),
    ("ICT_Month10IndexTradeSetups.jfif", "mixed",    "A", "no",  "yes", "§13d: labeled MMXM stages"),
    ("ict_mmxm_notes.jfif",      "mixed",             "A", "no",  "yes", "§13d: framework + notes panel"),
    ("LRS.jpeg",                "price_path",        "B", "yes", "image-led", "§13d: pure idealized model"),
    ("RTH ORG Repricing Model   Bias.jpeg", "mixed", "A", "no",  "yes", "§13d: rules/confluences text = payload"),
]

# --- 2. PDF pages — sample from the triage renders ---
# MMXM: all 16 co-dependent pages (the genuine mixed stress case)
MMXM_CODEP = [1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15, 17, 19, 22, 23, 24, 25, 27, 28, 29, 32]
# PRE-MARKET PLAN: 6 co-dependent pages
PREMARKET_CODEP = [2, 3, 6, 7, 11, 12, 13, 15, 17, 19, 22, 23, 24]
# Flux: sample 10 pages across the doc (every ~7th page) — full-page-image no-text-layer
FLUX_SAMPLE = [1, 7, 14, 21, 28, 35, 42, 49, 56, 63]
# Lumitrader: sample 10 pages — full-page-image no-text-layer, but skip first few (cover/TOC)
LUMI_SAMPLE = [10, 25, 50, 80, 120, 160, 200, 240, 280, 320]
# Delirium: 1 co-dependent page (small doc)
DELIRIUM_CODEP = [1]


def find_render(pdf_stem, page_num):
    # renders are named like "<stem>_p001.png"
    # stems have spaces/special chars preserved
    candidates = list(RENDER_DIR.glob(f"{pdf_stem}_p{page_num:03d}.png"))
    return candidates[0] if candidates else None


def main():
    inputs = []

    # 1. standalone images
    for fname, kind, case, seq, codep, notes in STANDALONE:
        p = TESTING / fname
        if not p.exists():
            print(f"  ! missing: {p}")
            continue
        inputs.append({
            "path": str(p),
            "source": "standalone",
            "source_pdf": None,
            "page": None,
            "true_kind": kind,
            "true_case": case,
            "seq_faithful": seq,
            "co_dependent": codep,
            "label_source": "HANDOVER §13d",
            "notes": notes,
        })

    # 2. MMXM co-dependent pages (the stress case) — true_kind = TODO (user labels)
    for pg in MMXM_CODEP:
        r = find_render("MMXM", pg)
        if r:
            inputs.append({
                "path": str(r), "source": "pdf_page", "source_pdf": "MMXM.pdf",
                "page": pg, "true_kind": "TODO", "true_case": "TODO",
                "seq_faithful": "TODO", "co_dependent": "TODO",
                "label_source": "user_pending",
                "notes": "MMXM co-dependent page — genuine mixed",
            })

    # 3. PRE-MARKET PLAN co-dependent
    for pg in PREMARKET_CODEP:
        r = find_render("PRE-MARKET PLAN", pg)
        if r:
            inputs.append({
                "path": str(r), "source": "pdf_page", "source_pdf": "PRE-MARKET PLAN.pdf",
                "page": pg, "true_kind": "TODO", "true_case": "TODO",
                "seq_faithful": "TODO", "co_dependent": "TODO",
                "label_source": "user_pending",
                "notes": "PRE-MARKET co-dependent page",
            })

    # 4. Flux sampled (full-page-image, no text layer)
    for pg in FLUX_SAMPLE:
        r = find_render("Flux_NY_Guide", pg)
        if r:
            inputs.append({
                "path": str(r), "source": "pdf_page", "source_pdf": "Flux_NY_Guide.pdf",
                "page": pg, "true_kind": "TODO", "true_case": "TODO",
                "seq_faithful": "TODO", "co_dependent": "TODO",
                "label_source": "user_pending",
                "notes": "Flux full-page-image, no text layer",
            })

    # 5. Lumitrader sampled (full-page-image, no text layer)
    for pg in LUMI_SAMPLE:
        r = find_render("lumitrader-ict-2022-book", pg)
        if r:
            inputs.append({
                "path": str(r), "source": "pdf_page", "source_pdf": "lumitrader-ict-2022-book.pdf",
                "page": pg, "true_kind": "TODO", "true_case": "TODO",
                "seq_faithful": "TODO", "co_dependent": "TODO",
                "label_source": "user_pending",
                "notes": "Lumitrader full-page-image, no text layer",
            })

    # 6. Delirium
    for pg in DELIRIUM_CODEP:
        r = find_render("Delirium", pg)
        if r:
            inputs.append({
                "path": str(r), "source": "pdf_page", "source_pdf": "Delirium.pdf",
                "page": pg, "true_kind": "TODO", "true_case": "TODO",
                "seq_faithful": "TODO", "co_dependent": "TODO",
                "label_source": "user_pending",
                "notes": "Delirium co-dependent page",
            })

    with open(OUT, "w", encoding="utf-8") as f:
        for inp in inputs:
            f.write(json.dumps(inp) + "\n")

    # summary
    by_src = {}
    for i in inputs:
        key = i["source_pdf"] or i["source"]
        by_src.setdefault(key, []).append(i)
    print(f"wrote {len(inputs)} inputs to {OUT}")
    print()
    for k, v in by_src.items():
        labeled = sum(1 for i in v if i["true_kind"] != "TODO")
        print(f"  {k:40s} {len(v):3d} inputs  ({labeled} labeled, {len(v)-labeled} TODO)")

    # missing renders?
    missing = [i for i in inputs if not os.path.exists(i["path"])]
    if missing:
        print(f"\n  ! {len(missing)} inputs have missing render files:")
        for m in missing[:10]:
            print(f"      {m['path']}")


if __name__ == "__main__":
    main()