"""
Chart / diagram extraction front stage — HUMAN-IN-THE-LOOP.

The problem this solves: teaching charts (like the LRS Smart Money Reversal or an
annotated EURUSD entry) encode a METHOD in their STRUCTURE — the ordered path of
price through zones/levels and each element's spatial relation (premium/discount,
ERL/IRL). Plain OCR gets labels but loses the path; a VLM can read the path but is
NOT reliable enough on spatial geometry to trust unverified (it misreads levels and
can flip relations). So: VLM PROPOSES a structured setup, YOU VERIFY against the
image, verified schema is stored. No prose (you have the source images).

Flow:
    image -> VLM -> proposed setup JSON in <output>/_review/  (status: needs_review)
    [you edit/approve the JSON, set status: approved]
    approve step -> writes a KnowledgeUnit into <output>/units/

Pluggable output target: 'setup' (knowledge base, default) or 'journal' (your
trade record — different schema). Same vision machinery, different destination.

Usage:
    # 1) propose: VLM reads images -> review JSONs
    python -m knowledge_ingest.sources.chart_extract propose \
        --images "C:\\charts" --output "C:\\out" --target setup

    # 2) (you edit the _review/*.json files: fix the sequence/relations, set
    #     "status": "approved")

    # 3) commit approved reviews into the units store
    python -m knowledge_ingest.sources.chart_extract commit --output "C:\\out"

Requires a vision model in Ollama (e.g. gemma4:cloud handled the test charts well).
"""

import os
import sys
import glob
import json
import base64
import hashlib
import argparse
from datetime import datetime

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --- VLM prompt: propose the STRUCTURED setup, stay grounded --------------- #
PROPOSE_PROMPT = """You are reading a trading teaching image. It may be a price-path
model, a reference/checklist diagram, an annotated trade chart, or a mix. Capture
EVERYTHING meaningful the image holds — BOTH the written text AND the visual structure.
Do not force it into a shape it isn't.

Return JSON:
{
 "kind": "<price_path | reference_diagram | annotated_chart | mixed>",
 "name": "<title/model name if shown>",
 "bias": "<bullish|bearish|reversal|null>",

 "text_content": {
   "conditions": ["<any listed conditions/requirements, verbatim>"],
   "confluences": ["<any listed confluences/desirables>"],
   "notes": ["<context notes, captions, caveats, definitions written on the image>"],
   "other_text": ["<any other meaningful text not captured above>"]
 },

 "reference_levels": ["<named levels/zones drawn, if any>"],

 "sequence": [
   {"order": 1, "action": "<what happens>", "position": "<premium|discount|equilibrium|null>",
    "range_liquidity": "<ERL|IRL|null>", "relative_to": "<level/zone or null>",
    "zone_label": "<label drawn for this step or null>"}
 ],

 "entry": "<entry rule if shown, else null>",
 "direction": "<long|short|null>",
 "target": "<target if shown, else null>",
 "concepts_raw": ["<ICT concepts present, as labeled>"],
 "inferred": ["<anything you concluded that is NOT explicitly written/drawn>"]
}

Guidance:
- Read ALL text on the image — labels, boxes, side-panel explanations, captions,
  small annotations — and put it in text_content even if it isn't part of a price path.
- sequence is ONLY for images that show an ordered price path. If the image is a
  reference/checklist diagram with no single path, leave sequence empty and put its
  content in text_content.
- Capture what is shown. If you infer something not explicitly written or drawn, put
  it in "inferred" rather than omitting it — do not silently drop information, and do
  not silently invent it either.
"""

def _client_call(model, image_b64, url):
    r = requests.post(f"{url}/api/generate", json={
        "model": model, "prompt": PROPOSE_PROMPT, "images": [image_b64],
        "stream": False, "format": "json", "options": {"temperature": 0.1},
    }, timeout=300)
    r.raise_for_status()
    return r.json()["response"]


def _parse(raw):
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    return json.loads(s[i:j + 1]) if 0 <= i < j else {}


def propose(args):
    review_dir = os.path.join(args.output, "_review")
    os.makedirs(review_dir, exist_ok=True)
    imgs = sorted(sum([glob.glob(os.path.join(args.images, f"*.{e}"))
                       for e in ("png", "jpg", "jpeg", "webp", "jfif", "gif", "bmp")], []))
    print(f"Proposing setups for {len(imgs)} images with {args.model}")
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        out_fp = os.path.join(review_dir, f"{stem}.json")
        if os.path.exists(out_fp) and not args.overwrite:
            print(f"  skip (review exists): {stem}")
            continue
        try:
            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            proposed = _parse(_client_call(args.model, b64, args.url))
        except Exception as e:
            print(f"  ERROR {stem}: {str(e)[:80]}")
            continue
        review = {
            "status": "needs_review",   # <- you change to "approved" after editing
            "target": args.target,       # setup | journal
            "source_image": img,
            "source_page": None,
            "source_type": args.source_type,
            "proposed": proposed,
        }
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2)
        print(f"  proposed: {stem} -> _review/{stem}.json  (VERIFY THIS)")
    print(f"\nNext: edit the _review/*.json files — fix sequence/relations, then set "
          f'"status": "approved". Then run: chart_extract commit --output {args.output}')


def _uid(stem):
    return f"chart__{stem}__{hashlib.sha1(stem.encode()).hexdigest()[:10]}"


def commit(args):
    """Turn approved reviews into KnowledgeUnit (or JournalEntry) records.

    FIX: previously the setup unit stored only name/bias/entry/target/
    reference_levels/sequence and SILENTLY DROPPED text_content — which for
    reference_diagram images (checklists, condition/confluence panels) is the
    entire payload. Now the full proposed structure is carried through:
    kind, text_content, and inferred are persisted, and inferred is also mirrored
    into metadata.inferred_fields so the grounding audit trail is preserved.
    """
    from knowledge_ingest.vocab.ict_vocabulary import map_to_canonical

    review_dir = os.path.join(args.output, "_review")
    units_dir = os.path.join(args.output, "units")
    journal_dir = os.path.join(args.output, "journal")
    os.makedirs(units_dir, exist_ok=True)
    os.makedirs(journal_dir, exist_ok=True)

    reviews = glob.glob(os.path.join(review_dir, "*.json"))
    committed = skipped = 0
    for fp in reviews:
        review = json.load(open(fp, encoding="utf-8"))
        if review.get("status") != "approved":
            skipped += 1
            continue
        stem = os.path.splitext(os.path.basename(fp))[0]
        p = review["proposed"]
        raw_concepts = p.get("concepts_raw", []) or []
        canon = map_to_canonical(raw_concepts)
        inferred = p.get("inferred", []) or []

        if review.get("target") == "journal":
            rec = {
                "entry_id": f"j__{stem}",
                "direction": p.get("direction"),
                "setup_used": p.get("name"),
                "what_i_saw": None, "what_i_missed": None,
                "concepts": canon,
                "chart_image_path": review.get("source_image"),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            with open(os.path.join(journal_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
        else:
            # knowledge-base setup unit
            unit = {
                "unit_id": _uid(stem),
                "summary": p.get("name") or f"chart setup: {stem}",
                "verbatim_anchor": None,
                "metadata": {
                    "knowledge_type": "setup",
                    "testability": "backtestable",
                    "epistemic_status": "unvalidated_concept",
                    "session_applicability": ["any"],
                    "instrument_applicability": ["any"],
                    "concepts_raw": raw_concepts,
                    "concepts_canonical": canon,
                    "linked_stat_ids": [],
                    # inferred items mirrored here so the grounding audit trail
                    # (what the VLM concluded but the image did not state) is not
                    # buried inside the setup payload only.
                    "inferred_fields": inferred,
                    "extraction_confidence": 0.6,  # human-verified, but one instance
                },
                "provenance": {
                    "source_file": stem,
                    "source_type": review.get("source_type", "chart"),
                    "source_credibility": "trusted_educator",
                    "chunk_id": f"{stem}:chart",
                    "image_path": review.get("source_image"),
                    "source_page": review.get("source_page"),
                    "extractor_model": args.model,
                    "extracted_at": datetime.now().isoformat(timespec="seconds"),
                },
                "setup": {
                    "name": p.get("name"),
                    "kind": p.get("kind"),                       # NEW: was dropped
                    "bias_source": p.get("bias"),
                    "entry": p.get("entry"),
                    "target_logic": p.get("target"),
                    "reference_levels": p.get("reference_levels"),
                    "sequence": p.get("sequence"),
                    "text_content": p.get("text_content"),       # NEW: was dropped
                    "chart_inferred": inferred,                  # NEW: was dropped (matches schema field name)
                },
            }
            with open(os.path.join(units_dir, f"{stem}.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps(unit) + "\n")
        committed += 1
        print(f"  committed: {stem} ({review.get('target')})")

    print(f"\nDone. committed: {committed} | still needs_review: {skipped}")
    if skipped:
        print("  (edit those reviews and set status:approved, then commit again)")
        
def compare(args):
    """Run several VLMs on the same charts; write proposals side-by-side so you can
    judge quality. Surfaces BOTH channels: the ordered price-path sequence AND the
    text_content block (conditions/confluences/notes). A reference_diagram will
    correctly have n_steps==0 — that is NOT a failure; judge it on text_content."""
    out_dir = os.path.join(args.output, "_compare")
    os.makedirs(out_dir, exist_ok=True)
    models = args.models
    imgs = sorted(sum([glob.glob(os.path.join(args.images, f"*.{e}"))
                       for e in ("png", "jpg", "jpeg", "webp", "jfif", "gif", "bmp")], []))
    print(f"Comparing {len(models)} models on {len(imgs)} charts")
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        result = {"chart": stem, "by_model": {}}
        for model in models:
            print(f"  {stem} <- {model}")
            try:
                proposed = _parse(_client_call(model, b64, args.url))
                seq = proposed.get("sequence", []) or []
                tc = proposed.get("text_content", {}) or {}
                # count how much text-channel content each model actually captured
                tc_counts = {k: len(tc.get(k, []) or [])
                             for k in ("conditions", "confluences", "notes", "other_text")}
                result["by_model"][model] = {
                    "name": proposed.get("name"),
                    "kind": proposed.get("kind"),
                    "bias": proposed.get("bias"),
                    "n_steps": len(seq),
                    "sequence": seq,
                    "text_content": tc,
                    "text_content_counts": tc_counts,
                    "text_items_total": sum(tc_counts.values()),
                    "reference_levels": proposed.get("reference_levels"),
                    "concepts_raw": proposed.get("concepts_raw"),
                    "inferred": proposed.get("inferred"),
                }
            except Exception as e:
                result["by_model"][model] = {"error": str(e)[:100]}
        with open(os.path.join(out_dir, f"{stem}.compare.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"    -> _compare/{stem}.compare.json")
    print(f"\nOpen each _compare/*.json next to the source image. Judge on:")
    print(f"  PRICE-PATH images:")
    print(f"    - is the SEQUENCE order correct?")
    print(f"    - are position (premium/discount) and range_liquidity (ERL/IRL) right?")
    print(f"  REFERENCE/CHECKLIST diagrams (n_steps==0 is fine):")
    print(f"    - did text_content capture the conditions / confluences / notes?")
    print(f"    - compare text_items_total; higher isn't always better — check for")
    print(f"      correctness and for verbatim conditions, not paraphrase/invention.")
    print(f"  Then pick the model whose draft needs the LEAST correction.")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("propose")
    pp.add_argument("--images", required=True)
    pp.add_argument("--output", required=True)
    pp.add_argument("--target", default="setup", choices=["setup", "journal"])
    pp.add_argument("--source-type", default="chart")
    pp.add_argument("--model", default="gemma4:cloud")
    pp.add_argument("--url", default="http://localhost:11434")
    pp.add_argument("--overwrite", action="store_true")
    pp.set_defaults(func=propose)

    cmp = sub.add_parser("compare")
    cmp.add_argument("--images", required=True)
    cmp.add_argument("--output", required=True)
    cmp.add_argument("--models", nargs="+",
                     default=["gemma4:cloud", "qwen3.5:cloud", "minimax-m3:cloud"],
                     help="vision models to compare on structured sequence quality")
    cmp.add_argument("--url", default="http://localhost:11434")
    cmp.set_defaults(func=compare)

    cp = sub.add_parser("commit")
    cp.add_argument("--output", required=True)
    cp.add_argument("--model", default="gemma4:cloud")
    cp.set_defaults(func=commit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
