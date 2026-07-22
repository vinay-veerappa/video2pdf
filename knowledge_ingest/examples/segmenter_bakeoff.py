"""
Segmenter bake-off. Unlike the extractor bake-off (which scores grounding), this
scores what matters for SEGMENTATION:
  (1) no empty/failed output (reasoning models tend to return empty under json_mode)
  (2) VERBATIM fidelity — is the segment text actually in the source, or did the
      model paraphrase/invent? Paraphrasing here silently corrupts everything
      downstream, since classify/extract would run on words the speaker never said.
  (3) sane segment count + speed (runs on every file at scale).

Usage:
    python -m knowledge_ingest.examples.segmenter_bakeoff

IMPORTANT: the fidelity check strips timestamp markers from BOTH sides before
comparing (an earlier version scored 0.12 because the source had [HH:MM:SS]
markers breaking up otherwise-verbatim text — the models were fine, the check
was broken).
"""

import sys, os, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import OllamaConfig
from pipeline.ollama_client import OllamaClient
from pipeline import prompts

CANDIDATES = ["gemma4:31b-cloud", "deepseek-v4-flash:cloud", "gemma4:latest"]

# point this at one real transcript
SAMPLE_PATH = r"C:\ICT_Videos\TCM\2023\transcripts\2023-05-11.txt"


def norm(s: str) -> str:
    """Strip [HH:MM:SS] markers, collapse whitespace, lowercase."""
    s = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().lower()


def verbatim_score(source: str, segments: list):
    """Fraction of segments whose text is found in the timestamp-stripped source.
    Uses 3 probes per segment (start/middle/end) to tolerate boundary trimming."""
    src = norm(source)
    if not segments:
        return 0.0, 0
    hits = 0
    for seg in segments:
        t = norm(seg.get("text", ""))
        if not t:
            continue
        probes = ([t[10:50], t[len(t)//2: len(t)//2 + 40], t[-50:-10]]
                  if len(t) > 90 else [t])
        if any(p and p in src for p in probes):
            hits += 1
    return hits / len(segments), len(segments)


def main():
    client = OllamaClient(OllamaConfig())
    source = open(SAMPLE_PATH, encoding="utf-8").read()
    for model in CANDIDATES:
        print(f"\n--- {model} ---")
        raw = None
        try:
            raw = client.generate(
                prompts.SEGMENT_PROMPT.format(chunk=source),
                model=model, system=prompts.SEGMENT_SYSTEM,
                temperature=0.0, num_ctx=131072, num_predict=8192,
                json_mode=True,
            )
            if not (raw or "").strip():
                print("  ERROR: empty response — reject for segmentation")
                continue
            segs = client.parse_json(raw)
            if isinstance(segs, dict):
                # some models return {ts: text} instead of the array schema — reject
                print("  WARN: returned a dict, not the array schema — likely not following spec")
                segs = [{"text": v} for v in segs.values()]
            vscore, n = verbatim_score(source, segs)
            print(f"  segments={n}  verbatim_fidelity={vscore:.2f}")
            if segs:
                print(f"  first ts: {segs[0].get('start_ts')} -> {segs[0].get('end_ts')}")
                print(f"  sample: {segs[0].get('text','')[:110]!r}")
        except Exception as e:
            print(f"  ERROR: {str(e)[:120]}")
            if raw:
                print(f"  RAW: {raw[:200]!r}")
    print("\nPick: verbatim_fidelity ~1.00, sane segment count, NO empty response, fastest.")
    print("Low fidelity = paraphrasing = reject. Dict-not-array = not following schema = reject.")


if __name__ == "__main__":
    main()
