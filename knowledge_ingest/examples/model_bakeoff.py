"""
Extractor bake-off: compare candidate models on GROUNDING DISCIPLINE, not raw
capability. For a knowledge base that will eventually inform trades, the model
you want is the one that leaves fields NULL when the transcript didn't state them
— not the one that confidently invents a stop-loss or price level to look complete.

Usage:
    python -m knowledge_ingest.examples.model_bakeoff

Edit CANDIDATES and TEST_SEGMENTS below. It runs the real extraction prompt
against each model and scores each on:
  - null_discipline : fraction of not-stated fields correctly left null
  - hallucination   : count of invented specifics (price levels/rules) not in text
  - self_confidence : the model's own extraction_confidence (should be LOWER when
                      it correctly leaves things null — miscalibration is a red flag)

This is a HELPER, not an autograder — it prints outputs side by side for your
eyeball. Grounding is a judgment call; the script surfaces the evidence.
"""

import sys, os, re, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import OllamaConfig
from pipeline.ollama_client import OllamaClient
from pipeline import prompts

# --- Candidate extractor models (edit to taste) ---------------------------- #
CANDIDATES = [
    "qwen3.5:397b-cloud",
    "deepseek-v4-pro:cloud",
    "glm-5.2:cloud",
]

# --- Test segments: pick ones with DELIBERATE GAPS ------------------------- #
# The point is to include units where the transcript gives a trigger but NO stop,
# or a bias but NO explicit target. A well-grounded model leaves those null.
TEST_SEGMENTS = [
    {
        "type": "setup",
        # states trigger + entry, but NO stop and NO explicit target number:
        "text": ("So at 8:15 we check if the overnight session is efficient. It was "
                 "inefficient today, so we wait for the 11:00 rebalance macro to trade "
                 "back up into the imbalance. When the first up-candle low gets taken, "
                 "that's your CSD short. I'm more interested in the narrative than a "
                 "precise entry here."),
        "gap_fields": ["stop_philosophy", "target_logic"],  # should be null-ish
    },
    {
        "type": "contextual",
        # states the event + lean, but the '58%'-style number is NOT given:
        "text": ("Every options expiry week I'm looking for a major sell-off on NASDAQ, "
                 "usually Tuesday or Wednesday. There's almost always high-impact news "
                 "that propels the move."),
        "gap_fields": [],  # no fabricated stat should appear in expected_behavior
    },
]

# specifics that would be HALLUCINATED if they appear (not in the text)
HALLUCINATION_MARKERS = re.compile(r'\b\d{3,5}\b|\b\d+\s*(point|pt|pip|%)\b', re.I)


def score(seg, ext):
    payload = (ext or {}).get("payload", {}) or {}
    # null discipline on the gap fields
    nulls_ok = sum(1 for f in seg["gap_fields"] if not payload.get(f))
    null_disc = nulls_ok / len(seg["gap_fields"]) if seg["gap_fields"] else 1.0
    # hallucinated specifics anywhere in payload text
    blob = json.dumps(payload)
    halluc = len(HALLUCINATION_MARKERS.findall(blob))
    conf = (ext or {}).get("extraction_confidence", None)
    return null_disc, halluc, conf


def main():
    client = OllamaClient(OllamaConfig())
    for seg in TEST_SEGMENTS:
        print("\n" + "=" * 74)
        print(f"SEGMENT ({seg['type']}): {seg['text'][:80]}...")
        print(f"Expected-null fields: {seg['gap_fields'] or '(no fabricated numbers)'}")
        print("=" * 74)
        for model in CANDIDATES:
            print(f"\n--- {model} ---")
            try:
                raw = client.generate(
                    prompts.extract_prompt(seg["type"], seg["text"]),
                    model=model,
                    system=prompts.EXTRACT_SYSTEM,
                    temperature=0.15,
                    num_ctx=32768,
                    num_predict=2048,
                    json_mode=True,
                )
                ext = client.parse_json(raw)
                nd, hl, cf = score(seg, ext)
                print(f"  null_discipline={nd:.2f}  hallucinated_specifics={hl}  self_conf={cf}")
                print("  payload:", json.dumps(ext.get("payload", {}), indent=2)[:600])
            except Exception as e:
                print(f"  ERROR: {str(e)[:120]}")
    print("\nPick the model with high null_discipline AND low hallucinated_specifics.")
    print("Beware a model with high self_conf but LOW null_discipline — that's overconfident invention.")


if __name__ == "__main__":
    main()
