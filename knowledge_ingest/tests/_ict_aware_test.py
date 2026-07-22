"""
ICT-aware chart extraction prompt + test against the 7 labeled images.

Tests two things:
1. An ICT-domain-aware prompt (with vocabulary, kind taxonomy, A/B/C rules,
   few-shot examples) vs the generic PROPOSE_PROMPT
2. MinerU OCR text as grounding (fed into the prompt alongside the image)

Architecture: MinerU extracts text (OCR) → text feeds into the ICT-aware VLM
prompt → gemma4 classifies kind + extracts sequence using domain knowledge.
"""

import os, sys, json, base64, time, requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OLLAMA_URL = "http://localhost:11434"
TESTING = r"C:\ICT_Videos\Testing"
MINERU_OUT = os.path.join(TESTING, "_mineru_img_test")

LABELED = [
    ("Arjo15mSTEntryModel.png",  "annotated_chart", "C", "weak"),
    ("BSL_DOL.webp",             "price_path",      "B", "yes"),
    ("DailyPo3.png",             "reference_diagram", "A", "no"),
    ("ICT_Month10IndexTradeSetups.jfif", "mixed",  "A", "no"),
    ("ict_mmxm_notes.jfif",      "mixed",           "A", "no"),
    ("LRS.jpeg",                "price_path",       "B", "yes"),
    ("RTH ORG Repricing Model   Bias.jpeg", "mixed", "A", "no"),
]


# ---- ICT-AWARE PROMPT ----
ICT_AWARE_PROMPT = """You are an expert in ICT (Inner Circle Trader) trading methodology, 
reading a teaching chart/diagram image. You understand concepts like FVG, order blocks, 
liquidity sweeps, CSD (change in state of delivery), MMXM, daily Po3, premium/discount, 
SMT divergence, and the 7-rule TCM execution framework.

CLASSIFY THE IMAGE INTO ONE OF THESE KINDS (based on ICT pedagogy, not just visual layout):

- "price_path": An IDEALIZED/SCHEMATIC drawing where the ordered price path IS the method.
  The drawn path through zones/levels is the content itself. Usually near-textless 
  (just zone labels on a curve). The SEQUENCE of moves is faithful — extracting it is correct.
  Examples: LRS (Smart Money Reversal), SBS wave models, fibonacci sequence diagrams.

- "reference_diagram": A checklist/framework with TEXT PANELS as the payload. May have a 
  price path drawn as illustration, but the TEXT (conditions, confluences, rules) is what 
  matters. The path merely illustrates. NO SEQUENCE should be extracted — the text is 
  the payload. Examples: DailyPo3 (bullish/bearish checklists), MMXM stages diagram.

- "annotated_chart": REAL market data (candlesticks, actual price bars) with markup 
  (zones, arrows, labels). This is a trade screenshot or annotated real chart, NOT an 
  idealized model. May have little text. Examples: trade entries, Arjo15m-style screenshots.

- "mixed": Combines framework text with chart imagery. The text panels and the visual 
  are co-dependent. Examples: ICT_Month10 (MMXM stages with chart), RTH_ORG (rules with 
  chart illustration), ict_mmxm_notes (framework + notes panel).

CRITICAL SEQUENCE RULE (the A/B/C distinction):
- If the image has SUBSTANTIAL EXPLANATORY TEXT (conditions, rules, confluence lists, 
  framework panels) → the TEXT is the payload. DO NOT extract a sequence. The drawn path 
  is just an illustration. (This is "case A" — most ICT teaching charts.)
- If the image is NEAR-TEXTLESS and SCHEMATIC (just zone labels on an idealized curve, 
  no conditions/rules prose) → the PATH is the method. Extract the sequence faithfully. 
  (This is "case B" — rare but clusters by source, e.g. SBS/wave model educators.)
- If the image has REAL MARKET DATA (candlesticks) → it's an annotated_chart, regardless 
  of text. (This is "case C".)

Return JSON:
{
 "kind": "<price_path | reference_diagram | annotated_chart | mixed>",
 "name": "<title/model name if shown>",
 "bias": "<bullish|bearish|reversal|null>",
 "text_content": {
   "conditions": ["<listed conditions/requirements, verbatim from image>"],
   "confluences": ["<listed confluences/desirables, verbatim>"],
   "notes": ["<context notes, captions, caveats, definitions>"],
   "other_text": ["<any other meaningful text>"]
 },
 "reference_levels": ["<named levels/zones drawn, if any>"],
 "sequence": [
   {"order": 1, "action": "<what happens>", "position": "<premium|discount|equilibrium|null>",
    "range_liquidity": "<ERL|IRL|null>", "relative_to": "<level/zone or null>",
    "zone_label": "<label drawn or null>"}
 ],
 "entry": "<entry rule if shown, else null>",
 "direction": "<long|short|null>",
 "target": "<target if shown, else null>",
 "concepts_raw": ["<ICT concepts present, as labeled — e.g. FVG, CSD, MMXM, SMT, order_block, liquidity_sweep>"],
 "inferred": ["<anything you concluded that is NOT explicitly written/drawn>"]
}

Guidance:
- Read ALL text on the image and put it in text_content.
- Only extract sequence if the image is near-textless AND schematic (case B). 
  If there's substantial text, leave sequence EMPTY — the text is the payload (case A).
- If you infer something not explicitly written/drawn, put it in "inferred", don't fabricate.
- Tag ICT concepts you recognize in concepts_raw.
"""

# Add OCR text to the prompt if available
def build_prompt_with_ocr(ocr_text):
    if not ocr_text or len(ocr_text.strip()) < 10:
        return ICT_AWARE_PROMPT
    return ("An OCR pass extracted the following text from this image. Use it as "
            "grounding — the text IS on the image. Cross-check what you see against it. "
            "Do NOT trust OCR blindly — if it conflicts with what you see, the image wins.\n\n"
            "OCR EXTRACTED TEXT:\n" + ocr_text + "\n\n" + ICT_AWARE_PROMPT)


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(model, img_b64, prompt, timeout=300):
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model, "prompt": prompt, "images": [img_b64],
        "stream": False, "format": "json", "options": {"temperature": 0.1},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def parse(raw):
    s = (raw or "").strip()
    if "```" in s:
        import re
        m = re.search(r'```(?:json)?\s*(.*?)```', s, flags=re.DOTALL)
        if m: s = m.group(1).strip()
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try: return json.loads(s[i:j+1])
        except: return None
    return None


def text_count(obj):
    tc = obj.get("text_content") or {}
    return sum(len(tc.get(k) or []) for k in ("conditions","confluences","notes","other_text"))


def seq_len(obj):
    seq = obj.get("sequence")
    return len(seq) if isinstance(seq, list) else 0


def get_mineru_ocr(img_name):
    """Get OCR text from MinerU output if available."""
    stem = os.path.splitext(img_name)[0]
    # MinerU output path pattern
    base = os.path.join(MINERU_OUT, stem, "hybrid_auto")
    md_path = os.path.join(base, f"{stem}.md")
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            return f.read()
    return None


MODEL = "gemma4:31b-cloud"

print(f"Testing ICT-aware prompt + MinerU OCR grounding on 7 labeled images")
print(f"Model: {MODEL}\n")

# Run MinerU on any images we haven't OCR'd yet
print("Step 1: Ensuring MinerU OCR for all 7 images...")
for fname, _, _, _ in LABELED:
    stem = os.path.splitext(fname)[0]
    md_path = os.path.join(MINERU_OUT, stem, "hybrid_auto", f"{stem}.md")
    if not os.path.exists(md_path):
        img_path = os.path.join(TESTING, fname)
        if os.path.exists(img_path):
            print(f"  Running MinerU on {fname}...")
            os.system(f'C:\\Users\\vinay\\mineru_venv\\Scripts\\mineru.exe -p "{img_path}" -o "{MINERU_OUT}" -b hybrid-engine --effort high 2>nul')
    else:
        print(f"  {fname}: OCR already available")

print(f"\nStep 2: Testing ICT-aware prompt (with OCR grounding) vs generic prompt\n")
print(f"{'image':35s} {'true':20s} | {'GENERIC kind':20s} {'seq':>4s} | {'ICT-AWARE kind':20s} {'seq':>4s} | {'match':>5s}")
print("-" * 115)

from knowledge_ingest.sources.chart_extract import PROPOSE_PROMPT as GENERIC_PROMPT

correct_generic = 0
correct_ict = 0

for fname, true_kind, case, seq_faithful in LABELED:
    path = os.path.join(TESTING, fname)
    if not os.path.exists(path):
        print(f"  ! missing: {fname}")
        continue
    img = b64(path)
    ocr_text = get_mineru_ocr(fname)
    ocr_len = len(ocr_text) if ocr_text else 0
    
    # Generic prompt (no OCR, no ICT knowledge)
    try:
        raw_g = call_vlm(MODEL, img, GENERIC_PROMPT)
        obj_g = parse(raw_g)
        kind_g = (obj_g or {}).get("kind", "?")
        seq_g = seq_len(obj_g or {})
    except Exception as e:
        kind_g, seq_g = f"ERR:{str(e)[:15]}", -1
    
    # ICT-aware prompt (with OCR grounding)
    ict_prompt = build_prompt_with_ocr(ocr_text)
    try:
        raw_i = call_vlm(MODEL, img, ict_prompt)
        obj_i = parse(raw_i)
        kind_i = (obj_i or {}).get("kind", "?")
        seq_i = seq_len(obj_i or {})
    except Exception as e:
        kind_i, seq_i = f"ERR:{str(e)[:15]}", -1
    
    match_g = "✓" if kind_g == true_kind else "✗"
    match_i = "✓" if kind_i == true_kind else "✗"
    if match_g == "✓": correct_generic += 1
    if match_i == "✓": correct_ict += 1
    
    print(f"  {fname[:32]:32s} {true_kind:20s} | {kind_g:20s} {seq_g:4d} | {kind_i:20s} {seq_i:4d} | {match_i:5s}  ocr={ocr_len}")

print(f"\n=== RESULTS ===")
print(f"Generic prompt:  {correct_generic}/7 ({correct_generic*100//7}%)")
print(f"ICT-aware prompt: {correct_ict}/7 ({correct_ict*100//7}%)")
print(f"\nImprovement: {correct_ict - correct_generic} images")