"""
ICT-aware prompt overrides for the text-pipeline (segment → classify → extract).

These are drop-in replacements for the domain-agnostic prompts in pipeline/prompts.py.
They inject ICT domain knowledge (educators, frameworks, concepts, session structure)
into the CLASSIFY and EXTRACT stages so the model understands what it's reading and
produces richer, more grounded extractions.

Key design lessons from the chart-prompt iteration (§17i → §18):
  - DO lean into the model's existing ICT/SMC knowledge for general concepts
  - DO embed corpus-specific details (educators, dialect terms, frameworks)
  - DO NOT over-prescribe classification rules (that caused the §17i over-correction)
  - DO NOT change the output JSON schema (pipeline code must work unchanged)

Usage:
  from knowledge_ingest.sources.ict_text_prompts import (
      ICT_CLASSIFY_PROMPT, ICT_CLASSIFY_BATCH_PROMPT,
      ICT_EXTRACT_SYSTEM, ict_extract_prompt, ict_extract_batch_prompt,
  )
  # Monkey-patch or pass into IngestPipeline constructor

The domain knowledge block is shared with the chart prompt (ict_chart_prompts.py)
to avoid duplication — if you update educator profiles there, they're updated here.
"""

from knowledge_ingest.sources.ict_chart_prompts import ICT_DOMAIN_KNOWLEDGE
from knowledge_ingest.pipeline.prompts import _CONCEPTS_RAW_SPEC, _PAYLOAD_SPECS, _TYPE_LABELS


# --------------------------------------------------------------------------- #
# Stage 2: CLASSIFY — ICT-aware
# --------------------------------------------------------------------------- #

ICT_CLASSIFY_SYSTEM = (
    "You are an expert in ICT / Smart Money Concepts (SMC) trading methodology. "
    "You classify a single unit of trading-education content from an ICT/SMC "
    "educator's teaching. Be decisive and literal. Return JSON only."
)

ICT_CLASSIFY_PROMPT = """You are an expert in {domain_knowledge_short}

Classify this trading-education unit from an ICT/SMC educator's teaching.

Choose exactly one knowledge_type:
- "setup": a mechanical, repeatable trade rule with some notion of entry/trigger/target.
  In ICT context: a specific entry model (e.g. "9:12 macro CSD short", "Silver Bullet
  long", "Judas Swing fade", "FVG rejection entry"), a dealing-range play, an OLR entry.
- "contextual": calendar or regime anticipation (e.g. Opex week, NFP, day-of-week
  behavior, Profile 1-6 classification, overnight session efficiency analysis).
- "framework": an analytical METHOD / how-to for reading the market (not a single
  trade rule). In ICT context: explaining MMXM stages, Po3 phases, SMR steps, how to
  frame a dealing range, how to read ONS efficiency, the 7 Rules.
- "tip": a small heuristic or rule-of-thumb (e.g. "don't trade first M5 candle",
  "two losses and stop for the day", "partials at 1.5R").
- "psychology": mindset, discipline, emotional management (e.g. "accept the loss",
  "don't chase", "FOMO management").
- "anecdote": a story about a past trade/event with no directly reusable rule.

Also return:
- "testability": "backtestable" | "partially" | "not_testable"
- "session_applicability": array from ["asia_tokyo","london","ny_am","ny_pm","overnight","any"]
  (ICT sessions: Asia=asia_tokyo, London=london, NY AM=ny_am, NY PM=ny_pm, ONS=overnight)
- "instrument_applicability": array from ["NQ","ES","YM","RTY","CL","GC","any"]
- "concepts_raw": ICT concept names as phrased in the unit. Use standard ICT
  terminology: FVG, OB, CSD (Change in State of Delivery), MSS, Po3, MMXM, SMR, OTE,
  BSL/SSL, ERL/IRL, Judas Swing, Silver Bullet, killzones, PD Array, premium/discount,
  BPR, ONS, draw on liquidity, etc. If the speaker uses a non-standard spelling
  (e.g. "FEG", "CISD", "CSV"), keep their spelling — it will be mapped later.
{concepts_raw_spec}
- "extraction_worthwhile": boolean. False only for pure narrative anecdotes with no lesson.

Return ONLY JSON:
{{"knowledge_type":"...","testability":"...","session_applicability":[...],
"instrument_applicability":[...],"concepts_raw":[...],"extraction_worthwhile":true}}

UNIT:
{unit}
""".replace(
    "{domain_knowledge_short}",
    "ICT / Smart Money Concepts (SMC) methodology. You know the frameworks: "
    "Power of Three (Po3), Market Maker Buy/Sell Model (MMXM), Smart Money Reversal "
    "(SMR), Optimal Trade Entry (OTE), Silver Bullet windows, Judas Swing, dealing "
    "ranges, ONS profiles, killzones, CSD/MSS, FVG/OB, liquidity sweeps, premium/"
    "discount, PD Arrays, and the 7-Rule execution framework. You understand the "
    "vocabulary of ICT educators (ICT, LumiTrader, Kish, Flux, Arjo, etc.)."
).replace("{concepts_raw_spec}", _CONCEPTS_RAW_SPEC)


# --------------------------------------------------------------------------- #
# Stage 2: CLASSIFY BATCH — ICT-aware (all segments in one call)
# --------------------------------------------------------------------------- #

ICT_CLASSIFY_BATCH_PROMPT = """You are an expert in ICT / Smart Money Concepts (SMC)
methodology. You know the frameworks: Po3, MMXM, SMR, OTE, Silver Bullet, Judas Swing,
dealing ranges, ONS profiles, killzones, CSD/MSS, FVG/OB, liquidity sweeps, premium/
discount, PD Arrays, and the 7-Rule execution framework.

Classify EACH trading-education unit below (all from the same ICT/SMC educator session).
Return a JSON array with one object per unit, in the SAME ORDER, each of the form:

{{"idx": <0-based index>, "knowledge_type":"setup|contextual|framework|tip|psychology|anecdote",
"testability":"backtestable|partially|not_testable",
"session_applicability":[...from asia_tokyo,london,ny_am,ny_pm,overnight,any],
"instrument_applicability":[...from NQ,ES,YM,RTY,CL,GC,any],
"concepts_raw":[ICT concept names as phrased, using standard terminology],
"extraction_worthwhile": true|false}}

ICT-aware type definitions:
- setup: mechanical repeatable trade rule — a specific entry model, dealing-range play,
  OLR entry, Silver Bullet trade, Judas Swing fade, FVG rejection entry.
- contextual: calendar/regime anticipation — Opex, NFP, day-of-week, Profile 1-6, ONS
  efficiency analysis, weekly profile expectations.
- framework: analytical METHOD — explaining MMXM stages, Po3 phases, SMR steps, how to
  frame a dealing range, how to read ONS, the 7 Rules.
- tip: small heuristic — "don't trade first M5", "two losses stop", "partials at 1.5R".
- psychology: mindset/discipline — "accept the loss", "don't chase", FOMO management.
- anecdote: story with no reusable rule. extraction_worthwhile=false ONLY for these.

For concepts_raw, use standard ICT names: FVG, OB, CSD, MSS, Po3, MMXM, SMR, OTE, BSL/SSL,
ERL/IRL, Judas Swing, Silver Bullet, killzones, PD Array, premium/discount, BPR, ONS, DOL.
{concepts_raw_spec}

Return ONLY the JSON array.

UNITS (index: text):
{numbered_units}
""".replace("{concepts_raw_spec}", _CONCEPTS_RAW_SPEC)


# --------------------------------------------------------------------------- #
# Stage 3: EXTRACT — ICT-aware system prompt + per-type prompts
# --------------------------------------------------------------------------- #

ICT_EXTRACT_SYSTEM = (
    "You are an expert in ICT / Smart Money Concepts (SMC) trading methodology. "
    "You extract STRUCTURED, GROUNDED knowledge from a unit of ICT/SMC teaching. "
    "You understand: Power of Three (Po3 = AMD), Market Maker Buy/Sell Model (MMXM), "
    "Smart Money Reversal (SMR), Optimal Trade Entry (OTE = 62-79% retracement), "
    "Silver Bullet windows, Judas Swing, dealing ranges, ONS profiles, CSD/MSS, "
    "FVG/OB, liquidity sweeps/raids, premium/discount, PD Arrays, BPR, killzones, "
    "the 7-Rule execution framework, and session-based analysis. "
    "CRITICAL RULES: "
    "1) Only fill a field if the text actually states or clearly implies it. "
    "2) Use null for anything not present. Never invent price levels, rules, or numbers. "
    "3) If you infer a field rather than find it stated, list that field name in "
    "'inferred_fields'. "
    "4) Keep 'verbatim_anchor' under 15 words, copied from the text. "
    "5) Rate 'extraction_confidence' 0-1 for how grounded your output is. "
    "6) Use standard ICT terminology in concepts (e.g. CSD not 'change in state', "
    "FVG not 'fair value gap' unless the speaker uses the full form). "
    "Return JSON only."
)

# ICT-aware payload specs — same fields, but with ICT context explaining what each
# field MEANS in ICT methodology so the model fills them more accurately.

ICT_PAYLOAD_SPECS = {
    "setup": """payload fields (null if not stated):
- "name": short label (e.g. "9:12 macro CSD short", "Silver Bullet long", "Judas fade")
- "regime_precondition": required market state (e.g. "ONS inefficient", "Profile 6",
  "tight overnight range <50pt", "Monday-Tuesday range holding", "bearish order flow")
- "bias_source": where bias comes from (e.g. "HTF draw to 13,350", "Wed expansion
  cycle", "daily order flow bearish — Friday's high taken then rejected", "weekly
  draw to sell-side liquidity at Tuesday's low")
- "timing_gate": required time window (e.g. "9:12 macro", "11:00 rebalance macro",
  "Silver Bullet 10-11am", "8:15 offset check", "9:30 equity open", "equities open")
- "trigger": event arming entry (e.g. "CSD after liquidity sweep", "Judas completion
  + MSS", "down-candle high taken after FVG", "OLS raid and reclaim")
- "entry": entry rule (e.g. "50%/CE of expansion candle", "FVG edge", "below M1 CSD
  low", "retrace to OB 50% after CSD", "IOFED into breakaway gap")
- "invalidation": what kills it (e.g. "M5 close above down-candle high (speed rule)",
  "3-shot rule", "stop above swept liquidity", "close above key level")
- "target_logic": target derivation (e.g. "1 SD from session open", "ONS 50%",
  "unfinished-business liquidity", "sell-side at Friday's low", "next ERL")
- "management": post-entry rules (e.g. "partials at 1.5R, move to BE if no expansion
  in 2-3 M5 candles", "partial at 20pt, BE the rest")
- "stop_philosophy": explicit stop logic if separate (e.g. "stop above swept BSL",
  "stop below FVG low", "stop at protected high")
- "quality_notes": aggressive vs rules-based, conviction level, any stated win-rate/RR,
  whether this is a "Coke trade" vs mechanical""",

    "contextual": """payload fields (null if not stated):
- "event_or_condition": e.g. "Options Expiry week", "NFP day", "Wednesday of the
  week", "summer regime", "Profile 1 (ONS low taken before 9:45)"
- "expected_behavior": what price is expected to do, as concrete as stated
  (e.g. "Monday-Tuesday creates a range, Wednesday sweeps one side, Thursday expands")
- "directional_lean": e.g. "bearish", "bullish", or null
- "testable_claim": a crisp falsifiable version for backtesting, if one exists
- "date_scope": e.g. "Tue/Wed of Opex week", "monthly", "T+2 settlement\"""",

    "framework": """payload fields (null/empty if not stated):
- "method_name": e.g. "Power of Three (Po3)", "MMXM", "SMR", "7-Rule Framework",
  "Dealing Range Logic", "ONS Profile Analysis"
- "what_it_answers": the question this method helps answer (e.g. "How to frame
  intraday bias from the daily chart", "How to identify the draw on liquidity")
- "steps": ordered array of steps if given (e.g. MMXM stages: 1. Accumulation,
  2. Manipulation, 3. Distribution; or SMR steps: 1. Sweep SSL, 2. MSS, 3. FVG, etc.)
- "inputs_required": array (e.g. ["daily high/low", "ONS high/low", "4am open"])
- "when_to_apply": e.g. "after ONS completes", "when ONS is efficient", "during
  killzones", "for NFP days\"""",

    "tip": """payload fields:
- "heuristic": the rule of thumb (e.g. "Don't trade the first M5 candle of equities
  open", "Two losses and stop for the day", "Take partials at 20 points")
- "rationale": why, if stated (e.g. "equities open candle is too volatile")
- "conditions": when it applies/doesn't (e.g. "applies on no-news days, not FOMC")""",

    "psychology": """payload fields:
- "principle": e.g. "Accept the loss and move on", "Don't force a narrative"
- "trigger_situation": when this should surface (e.g. "after 2 stops", "when
  missing a move", "after a big win")
- "prescribed_action": what to do (e.g. "walk away from charts", "reduce size",
  "journal the trade")""",

    "anecdote": """payload fields:
- "embedded_heuristic": the transferable lesson, or null if pure narrative
  (e.g. "Don't chase the entry — wait for confirmation even if it means missing
  the move")
- "context": the story context""",
}


def ict_extract_prompt(knowledge_type: str, unit_text: str) -> str:
    """Build an ICT-aware extraction prompt for a single unit."""
    from knowledge_ingest.pipeline.prompts import _EXTRACT_BASE
    return _EXTRACT_BASE.format(
        type_label=_TYPE_LABELS[knowledge_type],
        payload_spec=ICT_PAYLOAD_SPECS[knowledge_type],
        unit=unit_text,
    )


def ict_extract_batch_prompt(knowledge_type: str, numbered_units: str) -> str:
    """Build an ICT-aware batched extraction prompt for same-type units."""
    from knowledge_ingest.pipeline.prompts import EXTRACT_BATCH_BASE
    return EXTRACT_BATCH_BASE.format(
        type_label=_TYPE_LABELS[knowledge_type],
        payload_spec=ICT_PAYLOAD_SPECS[knowledge_type],
        numbered_units=numbered_units,
    )


def ict_classify_batch_prompt(numbered_units: str) -> str:
    return ICT_CLASSIFY_BATCH_PROMPT.format(numbered_units=numbered_units)