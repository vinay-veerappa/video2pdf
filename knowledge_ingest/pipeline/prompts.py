"""
Prompt templates for the three stages. Kept in one place so you can iterate on
wording without touching pipeline logic.

Stage 1 SEGMENT  (cheap model): break a transcript into concept-coherent units.
Stage 2 CLASSIFY (cheap model): label each segment's knowledge_type + metadata hints.
Stage 3 EXTRACT  (strong model): grounded JSON payload for the segment's type.
"""

# --------------------------------------------------------------------------- #
# Stage 1: SEGMENT
# --------------------------------------------------------------------------- #
SEGMENT_SYSTEM = (
    "You segment trading-education transcripts into coherent units. "
    "A unit is one self-contained idea: one setup, one concept explanation, one "
    "piece of calendar guidance, one psychology point, or one story. "
    "Do NOT split a setup's entry from its invalidation. Do NOT merge unrelated ideas. "
    "Return JSON only."
)

SEGMENT_PROMPT = """Split the transcript below into coherent knowledge units.

A UNIT is one complete teaching idea — a full setup (with its context), one
concept explanation, one piece of calendar/market guidance, one psychology point,
or one story. A unit is usually 30 seconds to 4 minutes of talk.

CRITICAL sizing rules:
- Do NOT split one idea into sentence-level fragments. If several consecutive
  timestamps all elaborate the SAME setup or concept, they are ONE unit.
- Do NOT collapse a long session into one giant unit. A 60+ minute session
  contains MANY units — expect roughly one unit per distinct idea, typically
  8-30 units for a full session.
- A greeting, sign-off, or "see you tomorrow" is trivial — mark it its own tiny
  unit so it classifies as anecdote and gets dropped.

For each unit return start_ts and end_ts (copy them exactly as they appear,
e.g. "00:02:12") and the verbatim text of that unit.

Return ONLY a JSON array:
[{{"start_ts": "HH:MM:SS", "end_ts": "HH:MM:SS", "text": "..."}}, ...]

Preserve original wording in "text"; do not paraphrase or summarize.

TRANSCRIPT:
{chunk}
"""

# --------------------------------------------------------------------------- #
# Stage 2: CLASSIFY
# --------------------------------------------------------------------------- #
CLASSIFY_SYSTEM = (
    "You classify a single unit of trading-education content. "
    "Be decisive and literal. Return JSON only."
)

# Shared, stricter spec for concepts_raw — used by BOTH the single-unit and
# batched classify prompts so they can't drift apart again. Fixes the junk-
# concept bug (§10c): the old one-liner ("exactly as phrased", two positive
# examples, zero negative examples) gave the model nothing to reject an
# instruction/action/timestamp phrase with, so it grabbed whatever noun-ish
# text was nearby (e.g. "analyze 8 o'clock").
_CONCEPTS_RAW_SPEC = """- "concepts_raw": array of NAMED trading concepts/techniques/patterns
  mentioned in the unit, exactly as phrased (e.g. "rebalance macro", "Judas
  swing", "FVG", "profile 5", "CSD").
  A concept is a TERM you could look up in a glossary — a named pattern,
  technique, structure, level, or model. It is NOT an instruction, an action,
  a plain timestamp, or a sentence.
  Test before including a phrase: "Would this make sense as a glossary
  entry?" If it describes what to DO rather than naming a THING, leave it out.
  DO NOT include:
    - instructions/actions ("analyze the 8 o'clock candle", "look at the
      chart", "wait for confirmation") — extract the concept if one is named
      inside the instruction (e.g. from "wait for the CSD" take "CSD", not
      the whole instruction)
    - bare timestamps or clock times with no named pattern attached ("8:00",
      "9:30", "8 o'clock") — only include a time if it IS itself a named
      macro/window (e.g. "9:12 macro", "11:00 rebalance macro")
    - full sentences or clauses
    - generic English words with no trading-specific meaning ("the market",
      "price", "today")"""

CLASSIFY_PROMPT = """Classify this trading-education unit.

Choose exactly one knowledge_type:
- "setup": a mechanical, repeatable trade rule with some notion of entry/trigger/target.
- "contextual": calendar or regime anticipation (e.g. Opex week, NFP, day-of-week behavior).
- "framework": an analytical METHOD / how-to for reading the market (not a single trade rule).
- "tip": a small heuristic or rule-of-thumb.
- "psychology": mindset, discipline, emotional management.
- "anecdote": a story about a past trade/event with no directly reusable rule.

Also return:
- "testability": "backtestable" | "partially" | "not_testable"
- "session_applicability": array from ["asia_tokyo","london","ny_am","ny_pm","overnight","any"]
- "instrument_applicability": array from ["NQ","ES","YM","RTY","CL","GC","any"]
{concepts_raw_spec}
- "extraction_worthwhile": boolean. False only for pure narrative anecdotes with no lesson.

Return ONLY JSON:
{{"knowledge_type":"...","testability":"...","session_applicability":[...],
"instrument_applicability":[...],"concepts_raw":[...],"extraction_worthwhile":true}}

UNIT:
{unit}
""".replace("{concepts_raw_spec}", _CONCEPTS_RAW_SPEC)

# --------------------------------------------------------------------------- #
# Stage 3: EXTRACT  (one prompt per type; strong model, grounded)
# --------------------------------------------------------------------------- #
EXTRACT_SYSTEM = (
    "You extract STRUCTURED, GROUNDED knowledge from a trading-education unit. "
    "CRITICAL RULES: "
    "1) Only fill a field if the text actually states or clearly implies it. "
    "2) Use null for anything not present. Never invent price levels, rules, or numbers. "
    "3) If you infer a field rather than find it stated, list that field name in 'inferred_fields'. "
    "4) Keep 'verbatim_anchor' under 15 words, copied from the text. "
    "5) Rate 'extraction_confidence' 0-1 for how grounded your output is. "
    "Return JSON only."
)

# Per-type field guidance appended to a shared base.
_EXTRACT_BASE = """Extract from this {type_label} unit.

Always return these common fields:
- "summary": one faithful sentence in plain language.
- "verbatim_anchor": <15-word representative phrase copied from the text (or null).
- "extraction_confidence": 0-1.
- "inferred_fields": array of field names you inferred rather than found stated.

Plus the type-specific "payload" object described below.

{payload_spec}

Return ONLY JSON:
{{"summary":"...","verbatim_anchor":"...","extraction_confidence":0.0,
"inferred_fields":[],"payload":{{...}}}}

UNIT:
{unit}
"""

_PAYLOAD_SPECS = {
    "setup": """payload fields (null if not stated):
- "name": short label
- "regime_precondition": required market state (e.g. "ONS inefficient", "Profile 6")
- "bias_source": where bias comes from (e.g. "HTF draw to 13,350", "Wed expansion cycle")
- "timing_gate": required time window (e.g. "9:12 macro", "11:00 rebalance macro")
- "trigger": event arming entry (e.g. "CSD after sweep", "Judas completion + MSS")
- "entry": entry rule (e.g. "50%/CE of expansion candle", "FVG edge")
- "invalidation": what kills it (e.g. "M5 close above down-candle high", "3-shot rule")
- "target_logic": target derivation (e.g. "1 SD from open", "ONS 50%")
- "management": post-entry rules (partials, move-to-BE conditions)
- "stop_philosophy": explicit stop logic if separate
- "quality_notes": aggressive/Coke vs rules-based, any stated win-rate/RR""",

    "contextual": """payload fields (null if not stated):
- "event_or_condition": e.g. "Options Expiry week", "NFP day", "Wednesday"
- "expected_behavior": what price is expected to do, as concrete as stated
- "directional_lean": e.g. "bearish", or null
- "testable_claim": a crisp falsifiable version for backtesting, if one exists
- "date_scope": e.g. "Tue/Wed of Opex week", "monthly\"""",

    "framework": """payload fields (null/empty if not stated):
- "method_name"
- "what_it_answers": the question this method helps answer
- "steps": ordered array of steps if given
- "inputs_required": array
- "when_to_apply\"""",

    "tip": """payload fields:
- "heuristic": the rule of thumb
- "rationale": why, if stated
- "conditions": when it applies/doesn't""",

    "psychology": """payload fields:
- "principle"
- "trigger_situation": when this should surface (e.g. "after 2 stops")
- "prescribed_action\"""",

    "anecdote": """payload fields:
- "embedded_heuristic": the transferable lesson, or null if pure narrative
- "context\"""",
}

_TYPE_LABELS = {
    "setup": "mechanical trade setup",
    "contextual": "calendar/regime anticipation",
    "framework": "analytical method",
    "tip": "heuristic",
    "psychology": "trading-psychology",
    "anecdote": "trade-story",
}


def extract_prompt(knowledge_type: str, unit_text: str) -> str:
    return _EXTRACT_BASE.format(
        type_label=_TYPE_LABELS[knowledge_type],
        payload_spec=_PAYLOAD_SPECS[knowledge_type],
        unit=unit_text,
    )


# --------------------------------------------------------------------------- #
# BATCHED variants (large-context cloud model)
# --------------------------------------------------------------------------- #

# Classify ALL segments of a file in one call -> array, preserving per-unit output.
CLASSIFY_BATCH_PROMPT = """Classify EACH trading-education unit below. Return a JSON
array with one object per unit, in the SAME ORDER, each of the exact form:

{{"idx": <0-based index>, "knowledge_type":"setup|contextual|framework|tip|psychology|anecdote",
"testability":"backtestable|partially|not_testable",
"session_applicability":[...from asia_tokyo,london,ny_am,ny_pm,overnight,any],
"instrument_applicability":[...from NQ,ES,YM,RTY,CL,GC,any],
"concepts_raw":[concept names EXACTLY as phrased],
"extraction_worthwhile": true|false}}

Definitions:
- setup: mechanical repeatable trade rule (entry/trigger/target notion).
- contextual: calendar/regime anticipation (Opex, NFP, day-of-week behavior).
- framework: analytical METHOD/how-to (not a single trade rule).
- tip: small heuristic. psychology: mindset/discipline. anecdote: story, no reusable rule.
extraction_worthwhile=false ONLY for pure narrative anecdotes with no lesson.

concepts_raw — same rule as the single-unit prompt:
{concepts_raw_spec}

Return ONLY the JSON array.

UNITS (index: text):
{numbered_units}
""".replace("{concepts_raw_spec}", _CONCEPTS_RAW_SPEC)

# Extract a BATCH of same-type units in one call. The model sees them together
# (shared session regime/bias/HTF draw) but MUST emit one grounded record each.
EXTRACT_BATCH_BASE = """These are multiple {type_label} units from the SAME trading
session, so they may share context (the day's regime, bias, HTF draw). Use that
shared context to complete each unit — but extract ONE grounded record PER unit.

CRITICAL: only fill a field if stated or clearly implied for THAT unit. Use null
otherwise. Never invent numbers. If you infer a field, list it in that unit's
"inferred_fields". Rate each unit's "extraction_confidence" 0-1.

For each unit return an object; return a JSON ARRAY in input order:
{{"idx": <0-based index>, "summary":"one faithful sentence",
"verbatim_anchor":"<15-word phrase copied from text or null",
"extraction_confidence":0.0, "inferred_fields":[], "payload":{{...}}}}

payload for this type:
{payload_spec}

Return ONLY the JSON array.

UNITS (index: text):
{numbered_units}
"""


def classify_batch_prompt(numbered_units: str) -> str:
    return CLASSIFY_BATCH_PROMPT.format(numbered_units=numbered_units)


# --------------------------------------------------------------------------- #
# Segment prompt variant for PROSE sources (blogs, markdown, text PDFs).
# These have no timestamps — segment on headers / paragraph topic shifts.
# --------------------------------------------------------------------------- #
SEGMENT_PROSE_PROMPT = """Split the document below into coherent knowledge units.

A UNIT is one complete idea — a concept explanation, one setup, one piece of
calendar/market guidance, one psychology point. Use the document's own structure:
headers, sections, and paragraph topic shifts are natural unit boundaries.

CRITICAL sizing rules:
- Do NOT fragment one idea across multiple units. A section explaining one
  concept is ONE unit even if it spans several paragraphs.
- Do NOT merge unrelated sections into one giant unit.
- Boilerplate (nav text, subscribe prompts, author bios, ads) → mark as its own
  tiny unit so it classifies as anecdote/noise and gets dropped.

There are no timestamps. Use the section header (or first few words) as a label
in "start_ts" and leave "end_ts" null.

Return ONLY a JSON array:
[{{"start_ts": "section label", "end_ts": null, "text": "..."}}, ...]

Preserve original wording in "text".

DOCUMENT:
{chunk}
"""


def extract_batch_prompt(knowledge_type: str, numbered_units: str) -> str:
    return EXTRACT_BATCH_BASE.format(
        type_label=_TYPE_LABELS[knowledge_type],
        payload_spec=_PAYLOAD_SPECS[knowledge_type],
        numbered_units=numbered_units,
    )