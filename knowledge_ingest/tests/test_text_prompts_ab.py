"""
A/B test: generic vs ICT-aware text pipeline on a small transcript subset.

Runs 3 small transcripts through BOTH the generic prompts and the ICT-aware
prompts (segment → classify → extract), then compares the results side-by-side.

Only uses the batched classify + extract stages (skips segmentation — we pre-split
manually to keep the test fast and focused on the prompt quality difference).
"""

import json, os, sys, re, time, requests
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from knowledge_ingest.pipeline.prompts import (
    CLASSIFY_BATCH_PROMPT, EXTRACT_BATCH_BASE, EXTRACT_SYSTEM,
    _CONCEPTS_RAW_SPEC, _PAYLOAD_SPECS, _TYPE_LABELS,
    classify_batch_prompt, extract_batch_prompt,
)
from knowledge_ingest.sources.ict_text_prompts import (
    ICT_CLASSIFY_BATCH_PROMPT, ICT_EXTRACT_SYSTEM, ICT_PAYLOAD_SPECS,
    ict_classify_batch_prompt, ict_extract_batch_prompt,
)
from knowledge_ingest.vocab.ict_vocabulary import map_to_canonical

OLLAMA = "http://localhost:11434"
MODEL = "gemma4:31b-cloud"

# Small transcripts for testing
TEST_FILES = [
    "8th May 2023.txt",       # 9.4KB — Kish review session
    "9th May 2023 (Review).txt",  # 8.6KB — Kish review
    "2023-05-11.txt",         # 3.8KB — very short
]

TRANSCRIPTS_DIR = r"C:\Users\vinay\video2pdf\transcripts_zip"
OUTPUT_DIR = r"C:\ICT_Videos\Testing\_text_ab_test"

# Pre-split a transcript into ~3-5 chunks for testing (simplified segmentation)
def simple_split(text, n_chunks=5):
    lines = text.strip().split("\n")
    chunk_size = max(1, len(lines) // n_chunks)
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def call_ollama(prompt, system=None, model=MODEL, temperature=0.0, timeout=120):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    r = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def parse_json_lenient(raw):
    s = (raw or "").strip()
    i, j = s.find("["), s.rfind("]")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j+1])
        except json.JSONDecodeError:
            pass
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j+1])
        except json.JSONDecodeError:
            pass
    return None


def run_pipeline_variant(transcript_text, variant_name, classify_fn, extract_fn,
                         classify_sys, extract_sys):
    """Run classify → extract on pre-split chunks. Returns list of units."""
    chunks = simple_split(transcript_text, n_chunks=4)
    all_units = []

    for chunk_idx, chunk in enumerate(chunks):
        # Classify batch
        numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate([chunk]))
        classify_prompt = classify_fn(numbered)
        raw = call_ollama(classify_prompt, system=classify_sys, temperature=0.0)
        classified = parse_json_lenient(raw)
        if not classified or not isinstance(classified, list):
            continue

        for item in classified:
            idx = item.get("idx", 0)
            ktype = item.get("knowledge_type", "framework")
            concepts_raw = item.get("concepts_raw", [])
            worthwhile = item.get("extraction_worthwhile", True)
            if not worthwhile:
                continue

            # Extract
            numbered_ext = f"[{idx}] {chunk}"
            ext_prompt = extract_fn(ktype, numbered_ext)
            raw2 = call_ollama(ext_prompt, system=extract_sys, temperature=0.15)
            extracted = parse_json_lenient(raw2)

            if extracted:
                if isinstance(extracted, list):
                    extracted = extracted[0] if extracted else {}
                unit = {
                    "chunk_idx": chunk_idx,
                    "knowledge_type": ktype,
                    "testability": item.get("testability"),
                    "session": item.get("session_applicability", []),
                    "concepts_raw": concepts_raw,
                    "concepts_canonical": map_to_canonical(concepts_raw),
                    "summary": extracted.get("summary"),
                    "confidence": extracted.get("extraction_confidence", 0),
                    "inferred_fields": extracted.get("inferred_fields", []),
                    "payload": extracted.get("payload", {}),
                }
                all_units.append(unit)

    return all_units


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in TEST_FILES:
        fpath = os.path.join(TRANSCRIPTS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"SKIP: {fname} not found")
            continue

        text = open(fpath, encoding="utf-8").read()
        print(f"\n{'='*70}")
        print(f"FILE: {fname} ({len(text)//1024:.1f}KB, {len(text.splitlines())} lines)")
        print(f"{'='*70}")

        # Run generic
        t0 = time.time()
        generic_units = run_pipeline_variant(
            text, "GENERIC",
            classify_batch_prompt, extract_batch_prompt,
            "You classify a single unit of trading-education content. Be decisive and literal. Return JSON only.",
            EXTRACT_SYSTEM,
        )
        t_generic = time.time() - t0
        print(f"\n  GENERIC: {len(generic_units)} units in {t_generic:.1f}s")

        # Run ICT-aware
        t0 = time.time()
        ict_units = run_pipeline_variant(
            text, "ICT-AWARE",
            ict_classify_batch_prompt, ict_extract_batch_prompt,
            "You are an expert in ICT / Smart Money Concepts (SMC) trading methodology. Be decisive and literal. Return JSON only.",
            ICT_EXTRACT_SYSTEM,
        )
        t_ict = time.time() - t0
        print(f"  ICT-AWARE: {len(ict_units)} units in {t_ict:.1f}s")

        # Compare
        print(f"\n  --- COMPARISON ---")
        print(f"  Units: generic={len(generic_units)}  ict={len(ict_units)}")

        # Type distribution
        g_types = Counter(u["knowledge_type"] for u in generic_units)
        i_types = Counter(u["knowledge_type"] for u in ict_units)
        print(f"  Types:  generic={dict(g_types)}")
        print(f"          ict={dict(i_types)}")

        # Avg confidence
        g_conf = sum(u["confidence"] for u in generic_units) / max(1, len(generic_units))
        i_conf = sum(u["confidence"] for u in ict_units) / max(1, len(ict_units))
        print(f"  Avg confidence: generic={g_conf:.2f}  ict={i_conf:.2f}")

        # Concept mapping rate
        g_mapped = sum(len(u["concepts_canonical"]) for u in generic_units)
        g_total = sum(len(u["concepts_raw"]) for u in generic_units)
        i_mapped = sum(len(u["concepts_canonical"]) for u in ict_units)
        i_total = sum(len(u["concepts_raw"]) for u in ict_units)
        g_rate = g_mapped / max(1, g_total) * 100
        i_rate = i_mapped / max(1, i_total) * 100
        print(f"  Concept mapping: generic={g_mapped}/{g_total} ({g_rate:.0f}%)  ict={i_mapped}/{i_total} ({i_rate:.0f}%)")

        # Setup name quality
        g_named = sum(1 for u in generic_units if u["knowledge_type"] == "setup" and u["payload"].get("name"))
        i_named = sum(1 for u in ict_units if u["knowledge_type"] == "setup" and u["payload"].get("name"))
        g_setups = sum(1 for u in generic_units if u["knowledge_type"] == "setup")
        i_setups = sum(1 for u in ict_units if u["knowledge_type"] == "setup")
        print(f"  Setup names: generic={g_named}/{g_setups}  ict={i_named}/{i_setups}")

        # Inferred fields
        g_inf = sum(len(u["inferred_fields"]) for u in generic_units)
        i_inf = sum(len(u["inferred_fields"]) for u in ict_units)
        print(f"  Inferred fields: generic={g_inf}  ict={i_inf}")

        # Show a few sample units side by side
        print(f"\n  --- SAMPLE UNITS (first 3 setups) ---")
        g_setups_list = [u for u in generic_units if u["knowledge_type"] == "setup"][:3]
        i_setups_list = [u for u in ict_units if u["knowledge_type"] == "setup"][:3]
        for idx in range(min(3, max(len(g_setups_list), len(i_setups_list)))):
            print(f"\n  Setup {idx+1}:")
            if idx < len(g_setups_list):
                u = g_setups_list[idx]
                print(f"    GENERIC: name={u['payload'].get('name')}  conf={u['confidence']:.1f}")
                print(f"      concepts: {u['concepts_raw'][:5]}")
                print(f"      summary: {u['summary'][:100] if u['summary'] else 'None'}")
            if idx < len(i_setups_list):
                u = i_setups_list[idx]
                print(f"    ICT:      name={u['payload'].get('name')}  conf={u['confidence']:.1f}")
                print(f"      concepts: {u['concepts_raw'][:5]}")
                print(f"      summary: {u['summary'][:100] if u['summary'] else 'None'}")

        # Save full results
        out_file = os.path.join(OUTPUT_DIR, fname.replace(".txt", "_comparison.json"))
        json.dump({
            "file": fname,
            "generic": {"units": generic_units, "time_s": t_generic},
            "ict_aware": {"units": ict_units, "time_s": t_ict},
        }, open(out_file, "w", encoding="utf-8"), indent=2, default=str)
        print(f"\n  Saved: {out_file}")


if __name__ == "__main__":
    main()