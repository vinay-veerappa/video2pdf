"""
Convert v4 full-run JSON outputs → typed KnowledgeUnit JSONL.

Reads the 818 JSON files from C:\\ICT_Videos\\Testing\\_v4_full_run\\ and writes
KnowledgeUnit records (one per image) to a units/ directory, ready for
vocab mapping, recanonicalization, and LanceDB vector-store build.

Mapping logic:
  v4 path_is_method=True  → KnowledgeType.SETUP (with sequence, entry_mechanics, etc.)
  v4 path_is_method=False → KnowledgeType.FRAMEWORK (reference/text pages)

Educator names are normalized to canonical forms.
Concepts are mapped via ict_vocabulary.map_to_canonical().
"""

import os
import sys
import json
import glob
import hashlib
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knowledge_ingest.vocab.ict_vocabulary import map_to_canonical, canonical_label

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

V4_RUN_DIR = r"C:\ICT_Videos\Testing\_v4_full_run"
OUTPUT_DIR = r"C:\ICT_Videos\Testing\_v4_units"

# Educator normalization
EDUCATOR_NORMALIZE = {
    "trader diego": "Trader-Diego",
    "trader-diego": "Trader-Diego",
    "mmxm trader": "MMxM-trader",
    "mmxm-trader": "MMxM-trader",
    "mmxM trader": "MMxM-trader",
    "mmxM-trader": "MMxM-trader",
    "lumitrader": "LumiTrader",
    "flux": "Flux",
    "ict": "ICT",
    "arjo": "Arjo",
    "tinyvizla": "TinyVizla",
    "amtrades": "AMTrades",
    "hydra": "Hydra",
    "fx4living": "fx4living",
    "kish": "Kish",
    "stoicta": "StoicTA",
    "dexter": "Dexter",
    "ttrades": "TTrades",
    "afyz": "Afyz",
}

# Source credibility by educator
CREDIBILITY_MAP = {
    "ICT": "trusted_educator",
    "LumiTrader": "trusted_educator",
    "Flux": "trusted_educator",
    "Arjo": "trusted_educator",
    "Kish": "trusted_educator",
    "Trader-Diego": "trusted_educator",
    "MMxM-trader": "trusted_educator",
    "TinyVizla": "trusted_educator",
    "AMTrades": "community",
    "Hydra": "community",
    "fx4living": "community",
    "StoicTA": "community",
    "Dexter": "community",
    "TTrades": "community",
    "Afyz": "community",
}


def normalize_educator(name):
    """Normalize educator name to canonical form."""
    if not name:
        return "unknown"
    key = name.lower().strip()
    return EDUCATOR_NORMALIZE.get(key, name)


def make_unit_id(stem):
    """Stable unit ID — hash of stem."""
    h = hashlib.sha1(stem.encode()).hexdigest()[:10]
    return f"chart__{stem}__{h}"


def build_knowledge_unit(v4_result):
    """Convert a single v4 JSON result into a KnowledgeUnit dict."""
    stem = os.path.splitext(os.path.basename(v4_result["input"]["path"]))[0]
    inp = v4_result.get("input", {})
    obj = v4_result.get("obj", {})
    
    # Core fields from v4
    path_is_method = v4_result.get("path_is_method", False)
    name = v4_result.get("name") or obj.get("name")
    framework = v4_result.get("framework")
    educator = normalize_educator(v4_result.get("educator_guess"))
    concepts_raw = v4_result.get("concepts_raw", []) or []
    inferred = v4_result.get("inferred", []) or []
    entry_mechanics = v4_result.get("entry_mechanics", []) or []
    
    # Map concepts to canonical
    concepts_canonical = map_to_canonical(concepts_raw)
    
    # Provenance
    source_pdf = inp.get("source_pdf")
    source_type = "chart" if inp.get("source") == "standalone" else "diagram"
    image_path = inp.get("path")
    source_page = inp.get("page")
    extractor_model = v4_result.get("model", "gemma4:31b-cloud")
    
    credibility = CREDIBILITY_MAP.get(educator, "unrated")
    
    # Build provenance
    provenance = {
        "source_file": source_pdf or stem,
        "source_type": source_type,
        "source_credibility": credibility,
        "chunk_id": f"{stem}:chart",
        "image_path": image_path,
        "source_page": source_page,
        "source_url": None,
        "extractor_model": extractor_model,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "speaker": educator if educator != "unknown" else None,
    }
    
    # Determine knowledge type
    if path_is_method:
        knowledge_type = "setup"
        testability = "partially"  # chart setups are partially testable
        # Build SetupPayload
        sequence = obj.get("sequence", [])
        text_content = obj.get("text_content")
        reference_levels = obj.get("reference_levels")
        
        # Build entry mechanics as part of quality_notes
        em_notes = []
        for em in entry_mechanics:
            em_notes.append(
                f"{em.get('name','')}: {em.get('description','')} "
                f"(Risk: {em.get('risk_reward','')})"
            )
        
        setup_payload = {
            "name": name,
            "regime_precondition": None,
            "bias_source": obj.get("bias"),
            "timing_gate": None,
            "trigger": None,
            "entry": "; ".join(em_notes) if em_notes else None,
            "invalidation": None,
            "target_logic": None,
            "management": None,
            "stop_philosophy": None,
            "quality_notes": None,
            "sequence": sequence,
            "reference_levels": reference_levels,
            "kind": v4_result.get("image_type", "annotated_chart"),
            "text_content": text_content,
            "chart_inferred": inferred,
        }
        
        summary = name or f"Chart setup: {stem}"
        
        unit = {
            "unit_id": make_unit_id(stem),
            "summary": summary,
            "verbatim_anchor": None,
            "metadata": {
                "knowledge_type": knowledge_type,
                "testability": testability,
                "epistemic_status": "unvalidated_concept",
                "session_applicability": ["any"],
                "instrument_applicability": ["any"],
                "concepts_raw": concepts_raw,
                "concepts_canonical": concepts_canonical,
                "linked_stat_ids": [],
                "inferred_fields": inferred,
                "extraction_confidence": 0.7,  # VLM-extracted, not human-verified
            },
            "provenance": provenance,
            "setup": setup_payload,
            "contextual": None,
            "framework": None,
            "tip": None,
            "psychology": None,
            "anecdote": None,
        }
    else:
        # Non-methodology page → Framework (reference/conceptual)
        knowledge_type = "framework"
        testability = "not_testable"
        
        # Build FrameworkPayload
        fw_payload = {
            "method_name": name or framework or f"Reference: {stem}",
            "what_it_answers": "; ".join(inferred[:2]) if inferred else None,
            "steps": [],
            "inputs_required": [],
            "when_to_apply": None,
        }
        
        summary = name or f"Reference page: {stem}"
        if framework and framework != "other":
            summary = f"{framework} reference: {stem}"
        
        unit = {
            "unit_id": make_unit_id(stem),
            "summary": summary,
            "verbatim_anchor": None,
            "metadata": {
                "knowledge_type": knowledge_type,
                "testability": testability,
                "epistemic_status": "unvalidated_concept",
                "session_applicability": ["any"],
                "instrument_applicability": ["any"],
                "concepts_raw": concepts_raw,
                "concepts_canonical": concepts_canonical,
                "linked_stat_ids": [],
                "inferred_fields": inferred,
                "extraction_confidence": 0.6,  # lower confidence for non-method pages
            },
            "provenance": provenance,
            "setup": None,
            "contextual": None,
            "framework": fw_payload,
            "tip": None,
            "psychology": None,
            "anecdote": None,
        }
    
    return unit


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all v4 results
    json_files = sorted(glob.glob(os.path.join(V4_RUN_DIR, "*.json")))
    json_files = [f for f in json_files if not os.path.basename(f).startswith("_")]
    
    converted = 0
    skipped = 0
    errors = 0
    educ_dist = Counter()
    type_dist = Counter()
    
    # Write all units to a single JSONL file
    output_jsonl = os.path.join(OUTPUT_DIR, "v4_chart_units.jsonl")
    
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for fp in json_files:
            try:
                d = json.load(open(fp, encoding="utf-8"))
                if d.get("error") or d.get("raw"):
                    skipped += 1
                    continue
                
                unit = build_knowledge_unit(d)
                out_f.write(json.dumps(unit, default=str) + "\n")
                
                converted += 1
                educ = unit["provenance"].get("speaker") or "unknown"
                educ_dist[educ] += 1
                type_dist[unit["metadata"]["knowledge_type"]] += 1
                
            except Exception as e:
                errors += 1
                print(f"  ERROR converting {os.path.basename(fp)}: {e}")
    
    print(f"\n{'='*60}")
    print(f"v4 → KnowledgeUnit conversion complete")
    print(f"{'='*60}")
    print(f"Converted: {converted}")
    print(f"Skipped (errors/parse fails): {skipped}")
    print(f"Conversion errors: {errors}")
    print(f"Output: {output_jsonl}")
    
    print(f"\nKnowledge type distribution:")
    for k, v in type_dist.most_common():
        print(f"  {k:15s} {v:4d} ({v/converted*100:.1f}%)")
    
    print(f"\nEducator distribution:")
    for k, v in educ_dist.most_common():
        print(f"  {k:20s} {v:4d} ({v/converted*100:.1f}%)")
    
    # Verify a few units parse correctly
    print(f"\nVerifying sample units...")
    from knowledge_ingest.schema.models import KnowledgeUnit
    test_stems = ["Arjo15mSTEntryModel", "DailyPo3", "lumitrader-ict-2022-book_p067"]
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            unit = json.loads(line)
            stem = unit["unit_id"].split("__")[1]
            if stem in test_stems:
                try:
                    ku = KnowledgeUnit.model_validate(unit)
                    print(f"  ✅ {stem}: type={ku.metadata.knowledge_type} "
                          f"concepts={len(ku.metadata.concepts_canonical)} "
                          f"canonical={[canonical_label(c) for c in ku.metadata.concepts_canonical[:3]]}")
                except Exception as e:
                    print(f"  ❌ {stem}: {e}")


if __name__ == "__main__":
    main()