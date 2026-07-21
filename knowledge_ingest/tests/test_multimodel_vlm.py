"""
Multi-model VLM comparison for ICT-aware prompts (HANDOVER §17i/§17h).

Runs one prompt across multiple cloud VLMs on a FOCUSED problem set:
  - the 2 v3-failure cases (BSL_DOL, ICT_Month10) — to see which model gets them right
  - a few PDF pages from different sources (Flux, MMXM, LumiTrader, Lecture)
    — to test educator identification on mixed-source material

Each model sees the SAME prompt + SAME image. We compare:
  - kind classification (vs ground truth where labeled)
  - educator_guess / framework identification
  - sequence presence (case A=0, B=>0)
  - concept naming quality

Usage:
    python knowledge_ingest/tests/test_multimodel_vlm.py --prompt ict_v3
    python knowledge_ingest/tests/test_multimodel_vlm.py --prompt ict_v3 --models gemma4:31b-cloud qwen3.5:cloud minimax-m3:cloud kimi-k2.7-code:cloud
    python knowledge_ingest/tests/test_multimodel_vlm.py --prompt ict_v3 --save C:\\ICT_Videos\\Testing\\_prompt_runs\\multimodel
"""

import os
import sys
import json
import base64
import argparse
import time

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from knowledge_ingest.sources.ict_chart_prompts import get_prompt

OLLAMA_URL = "http://localhost:11434"

# --- focused problem set ---
# the 2 v3 failures + a PDF page from each major source to test educator ID
RENDERS = r"C:\ICT_Videos\Testing\_triage_renders"
STANDALONE = r"C:\ICT_Videos\Testing"

PROBLEM_SET = [
    # the 2 v3-failure cases (stress the classifier)
    {"path": os.path.join(STANDALONE, "BSL_DOL.webp"),
     "source": "standalone", "true_kind": "price_path", "true_case": "B",
     "notes": "v3 FAIL: said mixed. near-textless schematic, path=method"},
    {"path": os.path.join(STANDALONE, "ICT_Month10IndexTradeSetups.jfif"),
     "source": "standalone", "true_kind": "mixed", "true_case": "A",
     "notes": "v3 FAIL: said reference_diagram. co-dependent text+structure"},
    # the 2 v3-success edge cases (guard against regression)
    {"path": os.path.join(STANDALONE, "LRS.jpeg"),
     "source": "standalone", "true_kind": "price_path", "true_case": "B",
     "notes": "v3 OK: pure schematic B. guard against regression"},
    {"path": os.path.join(STANDALONE, "ict_mmxm_notes.jfif"),
     "source": "standalone", "true_kind": "mixed", "true_case": "A",
     "notes": "v3 OK: mixed co-dependent. guard against regression"},
    # PDF pages — test educator ID on real doc pages (no ground truth kind)
    {"path": os.path.join(RENDERS, "Flux_NY_Guide_p001.png"),
     "source": "Flux_NY_Guide.pdf", "true_kind": None, "true_case": None,
     "notes": "Flux title page — educator should be Flux"},
    {"path": os.path.join(RENDERS, "Flux_NY_Guide_p013.png"),
     "source": "Flux_NY_Guide.pdf", "true_kind": None, "true_case": None,
     "notes": "Flux mid-doc — session profiling content"},
    {"path": os.path.join(RENDERS, "MMXM_p001.png"),
     "source": "MMXM.pdf", "true_kind": None, "true_case": None,
     "notes": "MMXM title page — educator should be MMXM trader"},
    {"path": os.path.join(RENDERS, "MMXM_p003.png"),
     "source": "MMXM.pdf", "true_kind": None, "true_case": None,
     "notes": "MMXM mixed page (co-dependent per triage)"},
    {"path": os.path.join(RENDERS, "lumitrader-ict-2022-book_p001.png"),
     "source": "lumitrader-ict-2022-book.pdf", "true_kind": None, "true_case": None,
     "notes": "LumiTrader title — educator should be LumiTrader"},
    {"path": os.path.join(RENDERS, "Lecture 1-5_p001.png"),
     "source": "Lecture 1-5.pdf", "true_kind": None, "true_case": None,
     "notes": "Lecture slide deck — test educator ID on slides"},
]


def b64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(model, image_b64, prompt, temperature=0.1, timeout=600):
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model, "prompt": prompt, "images": [image_b64],
        "stream": False, "format": "json",
        "options": {"temperature": temperature},
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def parse_json_lenient(raw):
    s = (raw or "").strip()
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


def text_count(obj):
    tc = obj.get("text_content") or {}
    return sum(len(tc.get(k) or []) for k in ("conditions", "confluences", "notes", "other_text"))


def seq_len(obj):
    seq = obj.get("sequence")
    return len(seq) if isinstance(seq, list) else 0


def run_one(model, image_b64, prompt):
    t0 = time.time()
    try:
        raw = call_vlm(model, image_b64, prompt)
    except Exception as e:
        return {"model": model, "error": str(e)[:200], "raw_ok": False}
    elapsed = time.time() - t0
    obj = parse_json_lenient(raw)
    return {
        "model": model,
        "elapsed_s": round(elapsed, 1),
        "raw_ok": obj is not None,
        "kind": (obj or {}).get("kind"),
        "n_seq": seq_len(obj or {}),
        "n_text": text_count(obj or {}),
        "name": (obj or {}).get("name"),
        "framework": (obj or {}).get("framework"),
        "educator_guess": (obj or {}).get("educator_guess"),
        "concepts_raw": (obj or {}).get("concepts_raw") or [],
        "inferred_count": len((obj or {}).get("inferred") or []),
        "obj": obj,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ict_v3")
    ap.add_argument("--models", nargs="+",
                    default=["gemma4:31b-cloud", "qwen3.5:cloud", "minimax-m3:cloud", "kimi-k2.7-code:cloud"])
    ap.add_argument("--save", default=None, help="dir to dump full JSON outputs")
    args = ap.parse_args()

    prompt, prompt_label = get_prompt(args.prompt)
    print(f"prompt: {args.prompt}  ({prompt_label})")
    print(f"models: {args.models}")
    print(f"problem set: {len(PROBLEM_SET)} images\n")

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    # filter to existing files
    inputs = [p for p in PROBLEM_SET if os.path.exists(p["path"])]
    print(f"{len(inputs)} images found\n")

    all_results = []
    for i, inp in enumerate(inputs, 1):
        fname = os.path.basename(inp["path"])
        gt = f"{inp['true_kind'] or '?'}/{inp['true_case'] or '?'}"
        print(f"[{i}/{len(inputs)}] {fname}  ({inp['source']})  true={gt}")
        print(f"    notes: {inp['notes']}")

        img = b64_image(inp["path"])
        per_model = {}
        for m in args.models:
            try:
                r = run_one(m, img, prompt)
                per_model[m] = r
                if "error" in r:
                    print(f"    {m:25s} ERROR {r['error'][:80]}")
                else:
                    kind = str(r.get("kind"))
                    match = ""
                    if inp["true_kind"]:
                        match = " ✓" if r.get("kind") == inp["true_kind"] else " ✗"
                    print(f"    {m:25s} {r.get('elapsed_s',0):5.1f}s  kind={kind:20s}{match}  "
                          f"seq={r.get('n_seq',0):2d}  text={r.get('n_text',0):3d}  "
                          f"educ={str(r.get('educator_guess')):12s}  fw={str(r.get('framework')):8s}  "
                          f"nm={str(r.get('name',''))[:25]}")
            except Exception as e:
                per_model[m] = {"model": m, "error": str(e)[:200]}
                print(f"    {m:25s} EXCEPTION {str(e)[:80]}")

        if args.save:
            with open(os.path.join(args.save, f"{os.path.splitext(fname)[0]}.json"), "w", encoding="utf-8") as f:
                json.dump({"input": inp, "results": per_model}, f, indent=2, default=str)

        all_results.append({"input": inp, "results": per_model})
        print()

    # --- summary table ---
    print("=" * 100)
    print("SUMMARY — kind accuracy on labeled images (standalone set)")
    print("=" * 100)
    labeled = [r for r in all_results if r["input"].get("true_kind")]
    if labeled:
        header = f"{'image':<40s} " + " ".join(f"{m.split(':')[0]:>16s}" for m in args.models)
        print(header)
        print("-" * len(header))
        for r in labeled:
            fname = os.path.basename(r["input"]["path"])[:38]
            true_case = r["input"]["true_case"]
            true_kind = r["input"]["true_kind"]
            cells = []
            for m in args.models:
                res = r["results"].get(m, {})
                k = res.get("kind")
                pim = res.get("path_is_method")
                # v4 mode: score path_is_method vs true_case
                if k is None and pim is not None:
                    if true_case == "A":
                        mark = "✓" if pim == False else "✗"
                        label = f"pim={pim}"
                    elif true_case == "B":
                        mark = "✓" if pim == True else "✗"
                        label = f"pim={pim}"
                    else:  # C
                        mark = "~"
                        label = f"pim={pim}"
                elif k is not None:
                    # v3 mode: score kind vs true_kind
                    mark = "✓" if k == true_kind else "✗"
                    label = str(k)[:14]
                else:
                    mark = "?"
                    label = "ERR"
                cells.append(f"{mark} {label:>14s}")
            print(f"{fname:<40s} " + " ".join(f"{c:>16s}" for c in cells))
        # accuracy per model
        print()
        for m in args.models:
            correct = 0
            for r in labeled:
                res = r["results"].get(m, {})
                k = res.get("kind")
                pim = res.get("path_is_method")
                if k is not None:
                    # v3 mode
                    if k == r["input"]["true_kind"]:
                        correct += 1
                elif pim is not None:
                    # v4 mode
                    tc = r["input"]["true_case"]
                    if tc == "A" and pim == False:
                        correct += 1
                    elif tc == "B" and pim == True:
                        correct += 1
                    elif tc == "C":
                        correct += 1  # weak — don't penalize
            print(f"  {m:25s}  {correct}/{len(labeled)}  ({correct*100//len(labeled)}%)")

    # --- educator identification on PDF pages ---
    print("\n" + "=" * 100)
    print("EDUCATOR GUESS on PDF pages (no ground truth — eyeball check)")
    print("=" * 100)
    pdf_pages = [r for r in all_results if r["input"]["source"] != "standalone"]
    if pdf_pages:
        header = f"{'page':<35s} " + " ".join(f"{m.split(':')[0]:>16s}" for m in args.models)
        print(header)
        print("-" * len(header))
        for r in pdf_pages:
            fname = os.path.basename(r["input"]["path"])[:33]
            cells = []
            for m in args.models:
                res = r["results"].get(m, {})
                edu = str(res.get("educator_guess", "?"))[:14]
                cells.append(f"{edu:>16s}")
            print(f"{fname:<35s} " + " ".join(cells))

    print(f"\nprompt: {args.prompt}")


if __name__ == "__main__":
    main()