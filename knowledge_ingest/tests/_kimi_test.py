"""Quick test of kimi-k2.7-code:cloud on the 7 labeled standalone images.
Reuses the bake-off harness infrastructure but only the 7 labeled images."""
import os, sys, json, base64, time, requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from knowledge_ingest.sources.chart_extract import PROPOSE_PROMPT

OLLAMA_URL = "http://localhost:11434"
LABELED = [
    ("Arjo15mSTEntryModel.png",  "annotated_chart"),
    ("BSL_DOL.webp",             "price_path"),
    ("DailyPo3.png",             "reference_diagram"),
    ("ICT_Month10IndexTradeSetups.jfif", "mixed"),
    ("ict_mmxm_notes.jfif",      "mixed"),
    ("LRS.jpeg",                "price_path"),
    ("RTH ORG Repricing Model   Bias.jpeg", "mixed"),
]

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def call_vlm(model, img_b64, timeout=300):
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model, "prompt": PROPOSE_PROMPT, "images": [img_b64],
        "stream": False, "format": "json", "options": {"temperature": 0.1},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]

def parse(raw):
    s = (raw or "").strip()
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

MODEL = "kimi-k2.7-code:cloud"
TESTING = r"C:\ICT_Videos\Testing"

print(f"Testing {MODEL} on 7 labeled images\n")
print(f"{'image':40s} {'true_kind':20s} {'kimi_kind':20s} {'seq':>4s} {'text':>4s} {'match':>5s}")
print("-" * 95)

correct = 0
for fname, true_kind in LABELED:
    path = os.path.join(TESTING, fname)
    if not os.path.exists(path):
        print(f"  ! missing: {fname}")
        continue
    try:
        t0 = time.time()
        raw = call_vlm(MODEL, b64(path))
        elapsed = time.time() - t0
        obj = parse(raw)
        if obj is None:
            print(f"  {fname[:35]:35s} {true_kind:20s} PARSE_FAIL")
            continue
        kind = obj.get("kind")
        seq = seq_len(obj)
        text = text_count(obj)
        match = "✓" if kind == true_kind else "✗"
        if match == "✓": correct += 1
        print(f"  {fname[:35]:35s} {true_kind:20s} {str(kind):20s} {seq:4d} {text:4d} {match:5s}  ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  {fname[:35]:35s} {true_kind:20s} ERROR {str(e)[:60]}")

print(f"\nkimi-k2.7 kind accuracy: {correct}/7 ({correct*100//7}%)")
print(f"\nFor comparison (from full bake-off):")
print(f"  gemma4:cloud   5/7 correct on labeled (1 timeout, 1 BSL_DOL)")
print(f"  qwen3.5:cloud   5/7 correct on labeled (BSL_DOL wrong, RTH_ORG right)")
print(f"  minimax-m3:     4/7 correct (ICT_Month10 + ict_mmxm_notes + RTH_ORG wrong)")