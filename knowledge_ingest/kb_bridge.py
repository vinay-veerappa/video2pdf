"""
Bridge: ICT Knowledge Base → Narrative Engine.

Connects the video2pdf knowledge base (LanceDB + RAG) to the tvdownloadOHLC
narrative engine. Instead of a static ICT_CONCEPTS_KB.md, the narrative LLM
gets dynamically retrieved, grounded ICT knowledge that matches the day's
actual market context.

Integration points:
  1. Cheat sheet augmentation: inject KB-retrieved context into the cheat sheet
  2. Prompt augmentation: add a "KB Context" section to the narrative prompt
  3. Post-narrative verification: check narrative claims against KB sources

Usage in tvdownloadOHLC/scripts/trader/briefing_core.py:

    from knowledge_kb_bridge import get_kb_context_for_narrative

    # In build_trader_cheat_sheet():
    kb_ctx = get_kb_context_for_narrative(cheat_sheet_text)
    cheat_sheet += "\\n\\n# ICT KNOWLEDGE BASE CONTEXT (RAG)\\n" + kb_ctx

    # Or in trader_narrative.py, before sending to LLM:
    kb_ctx = get_kb_context_for_narrative(cheat_sheet)
    prompt = prompt.replace("{{INSERT_CHEAT_SHEET}}", cheat_sheet + kb_ctx)

The KB server must be running: python -m knowledge_ingest.serve --port 8900
"""

import os, sys, json, requests, re
from typing import Optional

# ─── Configuration ─────────────────────────────────────────────────────────── #

KB_API_URL = os.environ.get("KB_API_URL", "http://127.0.0.1:8900")

# Concepts the narrative engine cares about, mapped to KB search queries.
# When the cheat sheet mentions these, we pull relevant KB units.
CONCEPT_TRIGGERS = {
    # Cheat sheet keyword → KB search query
    "FVG": "fair value gap imbalance entry",
    "CSD": "change in state of delivery CSD entry",
    "MSS": "market structure shift MSS",
    "order block": "order block entry OB",
    "liquidity sweep": "liquidity sweep buy-side sell-side",
    "Judas": "Judas swing fake move London session",
    "Power of Three": "power of three accumulation manipulation distribution",
    "Po3": "power of three accumulation manipulation distribution",
    "MMXM": "market maker buy sell model MMXM",
    "Silver Bullet": "silver bullet entry window",
    "OTE": "optimal trade entry OTE",
    "killzone": "killzone trading session timing",
    "overnight session": "overnight session ONS profile trading",
    "premium": "premium discount dealing range",
    "discount": "premium discount dealing range",
    "PDH": "prior day high low reference level",
    "PDL": "prior day high low reference level",
    "midnight open": "midnight open reference level",
    "7 Rule": "Kish 7 Rules execution framework",
    "trendline": "trendline entry model",
    "breaker": "breaker block entry",
    "turtle soup": "turtle soup liquidity sweep",
}


# ─── Public API ─────────────────────────────────────────────────────────────── #

def get_kb_context_for_narrative(cheat_sheet_text: str, k_per_concept: int = 3,
                                  max_context_chars: int = 2000) -> str:
    """
    Scan the cheat sheet for ICT concepts, retrieve relevant KB units, and
    format them as a context block for the narrative LLM.

    Args:
        cheat_sheet_text: The full cheat sheet text (from briefing_core)
        k_per_concept: How many KB units to retrieve per concept found
        max_context_chars: Total context budget (to avoid bloating the prompt)

    Returns:
        Formatted KB context block, or empty string if KB unavailable/no matches.
    """
    # Find which concepts are mentioned in the cheat sheet
    found_concepts = _detect_concepts(cheat_sheet_text)
    if not found_concepts:
        return ""

    # Retrieve relevant KB units for each found concept
    all_units = []
    seen_ids = set()
    for concept, query in found_concepts.items():
        units = _kb_search(query, k=k_per_concept)
        for u in units:
            uid = u.get("source_file", "") + str(u.get("confidence", ""))
            if uid not in seen_ids:
                all_units.append(u)
                seen_ids.add(uid)

    if not all_units:
        return ""

    # Format as context block, respecting the character budget
    lines = []
    total = 0
    for u in all_units:
        ktype = u.get("knowledge_type", "?")
        src = u.get("source_file", "?")
        conf = u.get("confidence", 0)
        text = (u.get("retrieval_text") or "")[:300]
        concepts = u.get("concepts", "")

        block = f"[{ktype}] {src} (conf={conf:.1f})\n  Concepts: {concepts}\n  {text}\n"
        if total + len(block) > max_context_chars:
            break
        lines.append(block)
        total += len(block)

    header = (
        f"# ICT KNOWLEDGE BASE CONTEXT (retrieved {len(lines)} units)\n"
        f"# Concepts detected in today's data: {', '.join(found_concepts.keys())}\n"
        f"# These are grounded source materials — use for terminology, definitions,\n"
        f"# and methodology context. Do NOT treat as trade signals.\n"
    )
    return "\n".join([header] + lines)


def answer_narrative_question(question: str, k: int = 8) -> dict:
    """
    Full RAG: ask a question about ICT methodology, get a grounded answer.
    Useful for interactive narrative refinement or verification.

    Returns: {"answer": "...", "sources": [...]}
    """
    try:
        r = requests.post(f"{KB_API_URL}/ask", json={
            "question": question, "k": k, "min_confidence": 0.5,
        }, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"answer": f"KB unavailable: {e}", "sources": []}


def verify_narrative_claim(claim: str, k: int = 5) -> dict:
    """
    Check if a narrative claim is grounded in the KB.
    Useful for post-narrative fact-checking.

    Returns: {"supported": bool, "sources": [...], "verdict": "..."}
    """
    try:
        r = requests.post(f"{KB_API_URL}/search", json={
            "query": claim, "k": k, "min_confidence": 0.5,
        }, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return {"supported": False, "sources": [], "verdict": "No KB match found"}
        # If top results are closely related, claim is likely grounded
        return {
            "supported": True,
            "sources": [{"type": u.get("knowledge_type"), "source": u.get("source_file")} for u in results[:3]],
            "verdict": f"Found {len(results)} related units in KB",
        }
    except Exception as e:
        return {"supported": False, "sources": [], "verdict": f"KB error: {e}"}


# ─── Internal helpers ───────────────────────────────────────────────────────── #

def _detect_concepts(text: str) -> dict:
    """Find ICT concepts mentioned in the cheat sheet."""
    found = {}
    text_upper = text.upper()
    for trigger, query in CONCEPT_TRIGGERS.items():
        # Check both exact and case-insensitive
        if trigger.lower() in text.lower() or trigger.upper() in text_upper:
            found[trigger] = query
    return found


def _kb_search(query: str, k: int = 3) -> list:
    """Call the KB API search endpoint."""
    try:
        r = requests.post(f"{KB_API_URL}/search", json={
            "query": query, "k": k, "min_confidence": 0.5,
        }, timeout=30)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []


# ─── CLI for testing ─────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    # Test with a sample cheat sheet
    sample = """
    BIAS CONSENSUS: BEARISH
    GEX Regime: NEGATIVE
    Key Levels: PDH 20150, PDL 19980, Midnight Open 20050
    Overnight Session: tight range, <50 points
    FVG below 20030 (discount zone)
    Liquidity: BSL above 20150, SSL below 19980
    Expected Move: 75 points
    Killzone: NY AM 09:30-11:00
    """

    print("=" * 60)
    print("TESTING KB BRIDGE WITH SAMPLE CHEAT SHEET")
    print("=" * 60)
    print("\nSample cheat sheet:")
    print(sample)

    ctx = get_kb_context_for_narrative(sample)
    print("\n" + "=" * 60)
    print("KB CONTEXT RETRIEVED:")
    print("=" * 60)
    print(ctx if ctx else "(no context — is KB server running on port 8900?)")

    print("\n" + "=" * 60)
    print("VERIFICATION TEST:")
    print("=" * 60)
    claim = "FVG in the discount zone is a high-probability short entry"
    v = verify_narrative_claim(claim)
    print(f"Claim: {claim}")
    print(f"Supported: {v['supported']}")
    print(f"Verdict: {v['verdict']}")