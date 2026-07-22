"""
VLM (vision model) bake-off — the GATE before any chart pipeline gets built.

Same discipline as the segmenter/extractor bake-offs, but for reading annotated
chart screenshots. The failure mode here is specific and dangerous: a vision model
will happily "read" a complete trading setup into a few boxes and arrows that don't
actually specify it — i.e. HALLUCINATE structure. For a knowledge base that may
inform trades, a model that invents annotations is disqualifying, even if it looks
articulate.

So this scores each candidate vision model on:
  (1) does it return output at all (some VLMs via Ollama are flaky)
  (2) GROUNDING — does its description mention ONLY elements you know are in the
      image, or does it invent labels/levels/setups that aren't there
  (3) label fidelity — does it read the actual text labels correctly

This is a HELPER, not an autograder. You provide a few charts whose contents you
KNOW (ground truth), and it prints each model's reading side by side so you can
judge grounding yourself. Vision grounding is a judgment call; the harness
surfaces the evidence and flags obvious invention.

Prereqs:
  - pull a vision model in Ollama, e.g.:
        ollama pull llama3.2-vision       (or qwen2-vl / your available VLM)
  - put 3-5 annotated chart screenshots in a folder
  - for each, write what's ACTUALLY in it in GROUND_TRUTH below

Usage:
    python -m knowledge_ingest.examples.vlm_bakeoff --charts "C:\\path\\to\\charts"
"""

import os
import sys
import glob
import json
import base64
import argparse

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- candidate vision models (edit to what you've pulled) ------------------ #
CANDIDATES = [
    "llama3.2-vision:latest",
    # "qwen2-vl:latest",
    # "minicpm-v:latest",
]

# --- GROUND TRUTH: for each chart filename, list what's ACTUALLY in it. ----- #
# The scorer checks whether the model's reading stays within these known
# elements (grounded) or invents things beyond them (hallucination risk).
GROUND_TRUTH = {
    # "chart1.png": {
    #     "instrument": "NQ", "timeframe": "5m",
    #     "drawn": ["blue box labeled FVG", "horizontal line at 13500", "down arrow"],
    #     "labels": ["FVG", "13500", "OB"],
    #     "known_absent": ["entry price", "stop loss", "target"],  # NOT annotated
    # },
}

# structured description prompt — asks for elements, not interpretation
VLM_PROMPT = """You are reading an annotated trading chart screenshot. Describe
ONLY what is literally drawn or written on the chart. Do NOT infer a trading
setup, entry, stop, or target unless it is explicitly labeled.

Return JSON:
{"instrument": "<ticker if visible or null>",
 "timeframe": "<if visible or null>",
 "drawn_elements": ["<each box/line/arrow/zone actually drawn>"],
 "text_labels": ["<each text label literally visible>"],
 "explicitly_stated_setup": "<only if the chart literally spells one out, else null>"}

If something is not visible, use null or omit it. Do not guess."""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(model, image_b64, url="http://localhost:11434"):
    r = requests.post(f"{url}/api/generate", json={
        "model": model, "prompt": VLM_PROMPT, "images": [image_b64],
        "stream": False, "format": "json",
        "options": {"temperature": 0.1},
    }, timeout=300)
    r.raise_for_status()
    return r.json()["response"]


def score(reading, truth):
    """Compare model reading against known ground truth for grounding."""
    if not truth:
        return None  # no ground truth provided for this chart
    labels_read = [str(x).lower() for x in reading.get("text_labels", []) or []]
    known_labels = [x.lower() for x in truth.get("labels", [])]
    known_absent = [x.lower() for x in truth.get("known_absent", [])]

    # label recall: how many real labels did it catch
    hit = sum(1 for k in known_labels if any(k in lr for lr in labels_read))
    recall = hit / len(known_labels) if known_labels else 1.0

    # hallucination: did it assert a setup / fields known to be ABSENT?
    setup = (reading.get("explicitly_stated_setup") or "")
    invented = [a for a in known_absent if a in setup.lower()]
    # also: labels it reported that aren't in the known set (possible invention)
    extra = [lr for lr in labels_read
             if not any(k in lr for k in known_labels) and len(lr) > 2]

    return {"label_recall": recall, "invented_absent_fields": invented,
            "extra_labels": extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", required=True, help="folder of chart screenshots")
    ap.add_argument("--url", default="http://localhost:11434")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.charts, "*.png")) +
                  glob.glob(os.path.join(args.charts, "*.jpg")) +
                  glob.glob(os.path.join(args.charts, "*.jpeg")))
    if not imgs:
        print(f"No images found in {args.charts}")
        return
    if not GROUND_TRUTH:
        print("WARNING: GROUND_TRUTH is empty. You'll see model readings but no "
              "grounding scores. Fill in GROUND_TRUTH for real evaluation.\n")

    for img in imgs:
        name = os.path.basename(img)
        print("\n" + "=" * 74)
        print(f"CHART: {name}")
        truth = GROUND_TRUTH.get(name)
        if truth:
            print(f"  known labels: {truth.get('labels')}")
            print(f"  known ABSENT (must NOT be invented): {truth.get('known_absent')}")
        print("=" * 74)
        b64 = encode_image(img)
        for model in CANDIDATES:
            print(f"\n--- {model} ---")
            try:
                raw = call_vlm(model, b64, args.url)
                if not (raw or "").strip():
                    print("  ERROR: empty response — reject for chart reading")
                    continue
                # tolerant parse
                s = raw.strip()
                i, j = s.find("{"), s.rfind("}")
                reading = json.loads(s[i:j+1]) if 0 <= i < j else {}
                print(f"  instrument={reading.get('instrument')} "
                      f"timeframe={reading.get('timeframe')}")
                print(f"  drawn: {reading.get('drawn_elements')}")
                print(f"  labels: {reading.get('text_labels')}")
                print(f"  setup: {reading.get('explicitly_stated_setup')}")
                sc = score(reading, truth)
                if sc:
                    flag = ""
                    if sc["invented_absent_fields"]:
                        flag = "  <-- INVENTED absent fields! grounding failure"
                    print(f"  SCORE: label_recall={sc['label_recall']:.2f} "
                          f"invented={sc['invented_absent_fields']} "
                          f"extra_labels={sc['extra_labels']}{flag}")
            except Exception as e:
                print(f"  ERROR: {str(e)[:120]}")

    print("\n" + "=" * 74)
    print("PICK: highest label_recall, ZERO invented-absent-fields, few extra labels.")
    print("A model that invents a setup/entry/stop not on the chart is DISQUALIFIED")
    print("for standalone chart extraction — even if its prose sounds authoritative.")
    print("If NONE pass, charts stay illustration-only (linked to text), never standalone.")


if __name__ == "__main__":
    main()
