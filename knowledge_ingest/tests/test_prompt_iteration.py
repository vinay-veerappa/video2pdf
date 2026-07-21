"""
Prompt iteration harness for ICT-aware chart extraction (HANDOVER §17i).

Runs ONE prompt across the 7 hand-labeled standalone images, scores kind accuracy,
and prints a per-image diagnostic so we can see WHERE a prompt over/under-corrects.
This is the fast iteration loop — change the prompt in ict_chart_prompts.py, re-run,
read the table, adjust. No 3-model consensus here; one model, one prompt, fast.

Usage:
    python knowledge_ingest/tests/test_prompt_iteration.py --prompt ict_v2
    python knowledge_ingest/tests/test_prompt_iteration.py --prompt ict_v2 --model gemma4:cloud
    python knowledge_ingest/tests/test_prompt_iteration.py --prompt ict_v2 --save C:\\ICT_Videos\\Testing\\_prompt_runs\\ict_v2

Ground truth is the §13d labeled set baked into bakeoff_inputs.jsonl.
"""

import os
import sys
import json
import base64
import argparse
import time
from pathlib import Path

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from knowledge_ingest.sources.ict_chart_prompts import get_prompt

OLLAMA_URL = "http://localhost:11434"

# the §13d hand-labeled standalone set (true_kind / true_case from HANDOVER)
GROUND_TRUTH = {
    "Arjo15mSTEntryModel.png":            {"true_kind": "annotated_chart",  "true_case": "C", "seq_faithful": "weak"},
    "BSL_DOL.webp":                       {"true_kind": "price_path",       "true_case": "B", "seq_faithful": "yes"},
    "DailyPo3.png":                       {"true_kind": "reference_diagram","true_case": "A", "seq_faithful": "no"},
    "ICT_Month10IndexTradeSetups.jfif":   {"true_kind": "mixed",           "true_case": "A", "seq_faithful": "no"},
    "ict_mmxm_notes.jfif":                {"true_kind": "mixed",           "true_case": "A", "seq_faithful": "no"},
    "LRS.jpeg":                           {"true_kind": "price_path",       "true_case": "B", "seq_faithful": "yes"},
    "RTH ORG Repricing Model   Bias.jpeg":{"true_kind": "mixed",           "true_case": "A", "seq_faithful": "no"},
}

IMG_DIR = r"C:\ICT_Videos\Testing"


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
    raw = call_vlm(model, image_b64, prompt)
    elapsed = time.time() - t0
    obj = parse_json_lenient(raw)
    return {
        "elapsed_s": round(elapsed, 1),
        "raw_ok": obj is not None,
        "kind": (obj or {}).get("kind"),
        "image_type": (obj or {}).get("image_type"),
        "path_is_method": (obj or {}).get("path_is_method"),
        "n_seq": seq_len(obj or {}),
        "n_text": text_count(obj or {}),
        "bias": (obj or {}).get("bias"),
        "name": (obj or {}).get("name"),
        "framework": (obj or {}).get("framework"),
        "educator_guess": (obj or {}).get("educator_guess"),
        "concepts_raw": (obj or {}).get("concepts_raw") or [],
        "inferred": (obj or {}).get("inferred") or [],
        "entry_mechanics": (obj or {}).get("entry_mechanics") or [],
        "obj": obj,
        "raw": raw if obj is None else None,
    }


def seq_correctness(result, gt):
    """Did the prompt get sequence presence right for the case?
    A: no sequence (seq_faithful=no => 0 steps correct).
    B: sequence present (seq_faithful=yes => >0 steps correct).
    C: weak/optional (don't penalize)."""
    n = result["n_seq"]
    case = gt["true_case"]
    if case == "A":
        return "OK" if n == 0 else f"OVER(n={n})"
    if case == "B":
        return "OK" if n > 0 else "UNDER(n=0)"
    # C
    return "weak"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ict_v2", help="prompt name from ict_chart_prompts.PROMPTS, or 'generic' for baseline")
    ap.add_argument("--model", default="gemma4:cloud", help="Ollama VLM model")
    ap.add_argument("--save", default=None, help="dir to dump full JSON outputs (optional)")
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    prompt, prompt_label = get_prompt(args.prompt)
    print(f"prompt: {args.prompt}  ({prompt_label})")
    print(f"model:  {args.model}\n")

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    rows = []
    correct_kind = 0
    correct_seq = 0
    seq_penalty = 0  # over-fabrication on case A (the poison-the-KB error)

    for fname, gt in GROUND_TRUTH.items():
        path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(path):
            print(f"  ! missing: {path}")
            continue
        print(f"[{fname}]  true={gt['true_kind']}/{gt['true_case']}")
        try:
            r = run_one(args.model, b64_image(path), prompt)
        except Exception as e:
            print(f"    ERROR {str(e)[:120]}")
            rows.append((fname, gt, {"error": str(e)[:200]}))
            continue

        # v4 has no `kind` — score on path_is_method instead.
        # true_case A: path_is_method should be false (text is payload)
        # true_case B: path_is_method should be true (path is method)
        # true_case C: weak (either)
        has_kind = r.get("kind") is not None
        if has_kind:
            kind_match = r["kind"] == gt["true_kind"]
            if kind_match:
                correct_kind += 1
        else:
            # v4 mode: score path_is_method against true_case
            pim = r.get("path_is_method")
            if gt["true_case"] == "A":
                kind_match = (pim == False)
            elif gt["true_case"] == "B":
                kind_match = (pim == True)
            else:  # C
                kind_match = True  # weak — don't penalize
            if kind_match:
                correct_kind += 1

        seq_state = seq_correctness(r, gt)
        if seq_state == "OVER":
            seq_penalty += 1
        elif seq_state.startswith("OK") or seq_state == "weak":
            correct_seq += 1

        label = r.get("image_type") if not has_kind else r.get("kind")
        print(f"    {'kind':<5s}={str(r.get('kind')):20s} {'itype':<6s}={str(r.get('image_type')):14s} "
              f"pim={str(r.get('path_is_method')):5s} {'✓' if kind_match else '✗'}  "
              f"seq={r['n_seq']:2d} [{seq_state}]  text={r['n_text']:3d}  "
              f"name={str(r['name'])[:30]}  {r['elapsed_s']:.1f}s")
        if r.get("framework") or r.get("educator_guess"):
            print(f"    framework={r['framework']}  educator={r['educator_guess']}")
        if r.get("entry_mechanics"):
            print(f"    entry_mechanics: {r['entry_mechanics']}")
        print(f"    concepts: {r['concepts_raw'][:8]}")
        if r["inferred"]:
            print(f"    inferred: {r['inferred'][:3]}")

        if args.save:
            with open(os.path.join(args.save, f"{os.path.splitext(fname)[0]}.json"), "w", encoding="utf-8") as f:
                json.dump({"file": fname, "ground_truth": gt, "result": r}, f, indent=2)

        rows.append((fname, gt, r))

    n = len([x for x in rows if isinstance(x[2], dict) and ("kind" in x[2] or "path_is_method" in x[2])])
    print(f"\n=== SUMMARY ({n}/{len(GROUND_TRUTH)} ran) ===")
    print(f"classification accuracy: {correct_kind}/{n}  ({correct_kind*100//max(n,1)}%)  "
          f"[v3: kind vs true_kind; v4: path_is_method vs true_case]")
    print(f"seq presence correct:  {correct_seq}/{n}  (case A=0 steps, B=>0 steps, C=weak)")
    print(f"seq fabrication (A):  {seq_penalty}  (over-emitting steps on case A — the poison-KB error)")

    # per-kind error matrix
    print("\nper-image:")
    print(f"  {'file':<38s} {'true':<22s} {'got':<22s} {'case':<5s} {'seq':<12s} {'match'}")
    for fname, gt, r in rows:
        if not isinstance(r, dict) or "kind" not in r:
            print(f"  {fname:<38s} {gt['true_kind']:<22s} {'ERROR':<22s}")
            continue
        kind_match = "✓" if r["kind"] == gt["true_kind"] else "✗"
        print(f"  {fname:<38s} {gt['true_kind']:<22s} {str(r['kind']):<22s} {gt['true_case']:<5s} "
              f"{seq_correctness(r, gt):<12s} {kind_match}")

    print(f"\nprompt: {args.prompt}  model: {args.model}")


if __name__ == "__main__":
    main()