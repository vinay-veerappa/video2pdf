"""
ICT-aware chart extraction prompts — iteration lab (HANDOVER §17h/§17i).

The production chart_extract.py uses a DOMAIN-AGNOSTIC PROPOSE_PROMPT. That prompt
scored 4/7 (57%) on the labeled set. A first ICT-aware attempt (§17i) over-corrected
to 3/7 (42%) by conflating "has text" with `reference_diagram`, missing the `mixed`
(co-dependent) case.

This file holds the iteration prompts so we can A/B them against the labeled set
without touching production code. Once a prompt wins, it moves to chart_extract.py.

The A/B/C typing (HANDOVER §13b-REVISED-2) is the load-bearing distinction:
  - Case A: text is the payload, drawing is decorative OR text+drawing are
    co-dependent. sequence MUST stay empty. Two sub-kinds:
      * reference_diagram: text panels stand ALONE; drawing merely illustrates.
        Removing the drawing loses ~nothing.
      * mixed: text REFERS to the drawn structure ("see the OB here", "first leg
        displaces past 50%"); neither channel is complete alone.
  - Case B: near-textless schematic; the ordered path through zones IS the method.
      * price_path.
  - Case C: REAL market data (candlesticks/actual price bars) with markup.
      * annotated_chart. (Real data is C regardless of how much text it has.)

The two-dimensional classifier:
  dim 1: real market data?  -> yes => annotated_chart (C).  no => dim 2.
  dim 2 (schematic only): substantial explanatory text present?
    yes => reference_diagram OR mixed (decide by co-dependence, see prompt)
    no  => price_path (B)
"""

# ---------------------------------------------------------------------------
# ICT-aware prompt — iteration 2 (v2)
# ---------------------------------------------------------------------------
# Changes vs the §17i over-correction:
#   - classify on TWO dimensions (real-data vs schematic, THEN text-presence),
#     not "has text => reference_diagram".
#   - explicit co-dependence test to split `reference_diagram` from `mixed`.
#   - lean INTO ICT domain knowledge: named frameworks, concept vocabulary,
#     session/timeframe context — interpret, don't just label.
#   - structure output by the image's own layout (bullish/bearish columns, panels).
#   - keep the same JSON schema as PROPOSE_PROMPT so outputs are comparable.
# ---------------------------------------------------------------------------

ICT_PROPOSE_PROMPT_V2 = """You are an expert in ICT / Smart Money Concepts (SMC) trading
methodology reading a teaching image (chart, diagram, or slide). You understand the
ICT vocabulary: Power of Three (Po3 = Accumulation -> Manipulation -> Distribution),
Market Maker Buy/Sell Model (MMXM), Smart Money Reversal (SMR), Optimal Trade Entry
(OTE = 62-79% retracement), Silver Bullet windows (10-11am / 2-3pm ET), Judas Swing,
SMT Divergence, FVG (Fair Value Gap / imbalance), Order Block (OB), Change in State
of Delivery (CSD), Market Structure Shift (MSS), liquidity sweeps / raids, draw on
liquidity (DOL), premium/discount dealing range, Consequent Encroachment (50% / mean
threshold), Balanced Price Range (BPR), engineered origin, inversion FVG, breaker/
mitigation/propulsion blocks, seek & destroy, NY Midnight Open, session opens
(London 2am, NY 8:30/9:30, PM 1:30), hourly opens, Previous Day High/Low (PDH/PDL),
equal highs/lows, liquidity voids,turtle soup, killzones, AMD, IPDA.

You also know the 7-Rule TCM execution framework (Rule 1 = daily order flow bias
first; Rule 3 = entry by OB anatomy; Rule 7 = order-of-delivery: SSL run -> FVG tag
-> short-term low -> CSD confirms). Use this knowledge to INTERPRET what the image
teaches — identify named frameworks when present, explain concepts in context, do
not just label pixels.

=== STEP 1 — CLASSIFY THE IMAGE ON TWO DIMENSIONS ===

Dimension 1 — is this REAL market data or a SCHEMATIC/IDEALIZED drawing?
  - REAL market data = actual candlesticks, real price bars, a real chart screenshot.
    => kind = "annotated_chart" (case C). Stop; you are done classifying.
  - SCHEMATIC/IDEALIZED = a clean drawn curve, idealized zones, a teaching model
    diagram (not a real chart). => go to Dimension 2.

Dimension 2 — ONLY for schematic images: is there substantial explanatory TEXT
(rules, conditions, confluences, definitions written on the image)?
  - NEAR-TEXTLESS: the drawn path through zones IS the method; only minimal labels
    (zone names, level labels, a few words). The ordered movement carries the
    teaching. => kind = "price_path" (case B). Extract the sequence.
  - TEXT-SUBSTANTIAL: a block of explanatory text is present. Decide CO-DEPENDENCE:
      * text panels STAND ALONE as the teaching content; the drawing merely
        ILLUSTRATES and removing it loses ~nothing (e.g. a bias checklist with a
        little illustrative path) => kind = "reference_diagram" (case A, text-payload).
      * text REFERS to the drawn structure ("see how the first leg displaces past
        50%", "the OB here is the origin of delivery", "price tags the FVG then...");
        text and drawing are CO-DEPENDENT, neither complete alone
        => kind = "mixed" (case A, co-dependent).
  Use this tie-breaker if unsure between reference_diagram and mixed: would the
  text still teach the same lesson if the drawing were removed? yes =>
  reference_diagram; no => mixed.

=== STEP 2 — EXTRACT, LEANING INTO ICT KNOWLEDGE ===

Return JSON with EXACTLY this shape:
{
 "kind": "<price_path | reference_diagram | annotated_chart | mixed>",
 "name": "<named framework/model if shown or clearly identifiable, e.g. 'Daily Po3', 'MMXM', 'Sharp Turn (ST) Entry Model', 'Smart Money Reversal'. else null>",
 "bias": "<bullish|bearish|reversal|both|null>",

 "text_content": {
   "conditions": ["<listed conditions/requirements, verbatim>"],
   "confluences": ["<listed confluences/desirables, verbatim>"],
   "notes": ["<captions, definitions, caveats, side-panel explanations written on the image>"],
   "other_text": ["<any other meaningful text not captured above>"]
 },

 "reference_levels": ["<named levels/zones drawn, e.g. 'PDH', 'NY Midnight Open', 'ERL', 'BSL', 'SSL', '50%', 'OTE 62-79%'>"],

 "sequence": [
   {"order": 1, "action": "<what price does, in ICT terms>", "position": "<premium|discount|equilibrium|null>",
    "range_liquidity": "<ERL|IRL|BSL|SSL|null>", "relative_to": "<level/zone or null>",
    "zone_label": "<label drawn for this step or null>"}
 ],

 "entry": "<entry rule if shown (e.g. 'enter at FVG inside OB', 'ODE entry at origin of delivery'), else null>",
 "direction": "<long|short|both|null>",
 "target": "<target logic if shown (e.g. 'opposite SSL', 'PDH'), else null>",
 "concepts_raw": ["<ICT concepts present, using the standard names above, as labeled on the image>"],
 "inferred": ["<anything you concluded from ICT knowledge that is NOT explicitly written/drawn>"]
}

Guidance:
- Read ALL text on the image — labels, boxes, side-panel explanations, captions,
  small annotations — and route it into text_content by channel. Do not drop text.
- sequence is ONLY for price_path (case B). For reference_diagram and mixed (case A)
  leave sequence EMPTY — emitting steps there is fabrication and poisons the KB.
  For annotated_chart (case C) sequence is optional and weak; include it only if a
  clear ordered trade narrative is drawn.
- IDENTIFY named frameworks when present (Power of 3, MMXM, SMR, Sharp Turn, OTE,
  Silver Bullet, 7 Rules, Judas Swing, SMT divergence, etc.) and put the name in
  `name`. Also tag the underlying concepts in concepts_raw.
- If the image is structured as COLUMNS or PANELS (e.g. bullish vs bearish columns,
  step-by-step panels), preserve that structure inside text_content: prefix the
  channel items with the panel/column label, e.g. "[Bullish] bias from 8:30 open
  above midnight". Do not flatten a 2-column layout into one flat list.
- Capture what is shown. If you infer something from ICT knowledge that is NOT
  explicitly written/drawn, put it in "inferred" — do not silently invent, and do
  not silently drop. Grounding discipline: the image is the source of truth.
- For concepts_raw use the STANDARD ICT names (FVG, OB, CSD, MSS, liquidity sweep,
  draw on liquidity, premium/discount, consequent encroachment, etc.), not the
  author's shorthand unless that's all that's written. This makes the unit
  queryable against the 172-concept vocabulary.
"""

# ---------------------------------------------------------------------------
# ICT-aware prompt — iteration 3 (v3) — EDUCATOR-AWARE
# ---------------------------------------------------------------------------
# What changed vs v2:
#   - tells the model WHICH educators' material this is, with their signature
#     visual/concept styles, so it can recognize named frameworks by style
#     (not just by reading a title). Grounded in the ACTUAL corpus, not a generic
#     "ICT mentors" list — the list below is every source in the Testing set.
#   - asks for an `educator_guess` field (best-guess provenance) and a `framework`
#     field (named model) — mirroring the Grok reference (§17h) which identified
#     "Daily Po3", "Sharp Turn Entry Model" by name.
#   - keeps the same JSON schema + grounding discipline (no prose output).
#   - carries forward the v2 two-dimension classifier (real-data vs schematic,
#     then text-presence + co-dependence) — that part was right; the §17i
#     over-correction came from missing `mixed`, which v2 fixed.
# ---------------------------------------------------------------------------

ICT_PROPOSE_PROMPT_V3 = """You are an expert in ICT / Smart Money Concepts (SMC) trading
methodology reading a TEACHING image. You know the material of these educators (the
sources in this corpus):

- **ICT (Michael J. Huddleston, the Inner Circle Trader)** — the originator. His
  frameworks: Power of Three (Po3 = AMD: Accumulation -> Manipulation -> Distribution),
  Market Maker Buy/Sell Model (MMXM), Optimal Trade Entry (OTE = 62-79% retracement),
  Silver Bullet windows (10-11am / 2-3pm ET), Judas Swing, draw on liquidity (DOL),
  ONS (overnight session), killzones, IPDA. Visual style: idealized schematic
  drawings with labeled zones, sometimes two-column (bullish/bearish) checklists.
- **LumiTrader** (ICT 2022 book) — full-page diagram pages; Daily Po3, price-cycle
  diagrams, annotated real-chart examples. Uses ICT canon with his own layout.
- **Flux** (NY Session Guide) — session profiling (Profiles 1-6), the 9:12 / 12:45 /
  11:00 / 9:45-10:00 / 8:15 macros, weekly day-of-week profiles. Glossary-heavy.
- **Arjo (Arjo15m)** — "Sharp Turn (ST) Entry Model"; signature terminology: Context
  High/Low (HTF buyside/sellside liquidity boundaries), Protected High (the swing high
  that forms the stop-loss reference), OD Entry (Overlapping Defense — aggressive
  entry at the top of the displacement/FVG zone), FLOD Entry (First Line of Defense
  — conservative entry at the lower threshold as price exits the imbalance), FVG
  Confirming ST (the Fair Value Gap that confirms the Sharp Turn). Real trade
  screenshots with markup. Short-side execution framework.
- **LumiTrader** publishes BOTH the "ICT 2022" book (435pp full-page diagrams)
  AND the MMXM (Market Maker Buy/Sell Model) material. So MMXM-labeled content
  is LumiTrader's, NOT a separate "MMXM trader" source. Visual style: full-page
  diagram pages, framework panels with side notes, MMXM stages.
- **Kish** — TCM 7-Rule execution framework: Rule 1 daily order flow bias first;
  Rule 3 entry by OB anatomy (FVG-in-OB -> inefficiency; large body -> 50%/mean
  threshold; small body -> open+high (bearish) / open+low (bullish)); Rule 4 FVG
  filter (break low + up-candle -> FVG above -> aggressive expansion lower);
  Rule 5 timeframe filter (London -> M15; NY -> M5); Rule 6 distrust wick swing
  inside FVG; Rule 7 order of delivery (SSL run -> FVG tag -> short-term low ->
  CSD confirms). Numbered "rule #N" citations are a Kish signature.
- **StoicTA** — idealized SBS (Step-by-Step) wave models, 1-2-3-4-5 wave counts,
  fibonacci projections. Near-textless schematics where the wave order IS the
  content (case B — sequence is faithful here).

Use this knowledge to INTERPRET what the image teaches — identify the framework
and likely educator when recognizable, explain concepts in context. Do not just
label pixels.

You also know the standard ICT concept vocabulary: FVG (Fair Value Gap / imbalance),
Order Block (OB), Change in State of Delivery (CSD), Market Structure Shift (MSS),
liquidity sweep / raid / purge, draw on liquidity (DOL), premium/discount dealing
range, Consequent Encroachment (50% / mean threshold), Balanced Price Range (BPR),
engineered origin, inversion FVG, breaker/mitigation/propulsion/vacuum blocks,
reclaimed OB, seek & destroy, SMT divergence, equal highs/lows, liquidity void,
turtle soup, NY Midnight Open, session opens (London 2am, NY 8:30/9:30, PM 1:30),
hourly opens, PDH/PDL, HOW/LOW, ITH/ITL, STH/STL, LTH/LTL, ERL/IRL, BSL/SSL.

=== STEP 1 — CLASSIFY THE IMAGE ON TWO DIMENSIONS ===

Dimension 1 — is this REAL market data or a SCHEMATIC/IDEALIZED drawing?
  - REAL market data = actual candlesticks, real price bars, a real chart screenshot
    (even lightly marked). => kind = "annotated_chart" (case C). Stop classifying.
  - SCHEMATIC/IDEALIZED = clean drawn curve, idealized zones, a teaching model
    diagram (not a real chart). => go to Dimension 2.

Dimension 2 — ONLY for schematic images: is there substantial explanatory TEXT
(rules, conditions, confluences, definitions written on the image)?
  - NEAR-TEXTLESS: the drawn path through zones IS the method; only minimal labels
    (zone names, level labels, a few words). The ordered movement carries the
    teaching (StoicTA SBS waves, LRS-type model). => kind = "price_path" (case B).
    Extract the sequence.
  - TEXT-SUBSTANTIAL: a block of explanatory text is present. Decide CO-DEPENDENCE:
      * text panels STAND ALONE as the teaching content; the drawing merely
        ILLUSTRATES and removing it loses ~nothing (e.g. a bias checklist with a
        little illustrative path, ICT two-column DailyPo3). => "reference_diagram"
        (case A, text-payload).
      * text REFERS to the drawn structure ("see how the first leg displaces past
        50%", "the OB here is the origin of delivery", "price tags the FVG then...");
        text and drawing are CO-DEPENDENT, neither complete alone. => "mixed"
        (case A, co-dependent).
  Tie-breaker if unsure between reference_diagram and mixed: would the text still
  teach the same lesson if the drawing were removed? yes => reference_diagram;
  no => mixed.

=== STEP 2 — EXTRACT, LEANING INTO ICT + EDUCATOR KNOWLEDGE ===

Return JSON with EXACTLY this shape:
{
 "kind": "<price_path | reference_diagram | annotated_chart | mixed>",
 "name": "<named framework/model if shown or clearly identifiable, e.g. 'Daily Po3', 'MMXM', 'Sharp Turn (ST) Entry Model', 'Smart Money Reversal', 'SBS Model #2'. else null>",
 "framework": "<the ICT framework this teaches, if identifiable: Po3 | MMXM | SMR | OTE | Silver Bullet | TCM-7-Rules | SBS-wave | NY-session-profiling | other | null>",
 "educator_guess": "<best-guess source educator: ICT | LumiTrader | Flux | Arjo | MMXM-trader | Kish | StoicTA | unknown>",
 "bias": "<bullish|bearish|reversal|both|null>",

 "text_content": {
   "conditions": ["<listed conditions/requirements, verbatim>"],
   "confluences": ["<listed confluences/desirables, verbatim>"],
   "notes": ["<captions, definitions, caveats, side-panel explanations written on the image>"],
   "other_text": ["<any other meaningful text not captured above>"]
 },

 "reference_levels": ["<named levels/zones drawn, e.g. 'PDH', 'NY Midnight Open', 'ERL', 'BSL', 'SSL', '50%', 'OTE 62-79%'>"],

 "sequence": [
   {"order": 1, "action": "<what price does, in ICT terms>", "position": "<premium|discount|equilibrium|null>",
    "range_liquidity": "<ERL|IRL|BSL|SSL|null>", "relative_to": "<level/zone or null>",
    "zone_label": "<label drawn for this step or null>"}
 ],

 "entry": "<entry rule if shown (e.g. 'enter at FVG inside OB', 'OD entry at origin of delivery', 'aggressive vs conservative entry'), else null>",
 "direction": "<long|short|both|null>",
 "target": "<target logic if shown (e.g. 'opposite SSL', 'PDH', 'R:R 2.0'), else null>",
 "concepts_raw": ["<ICT concepts present, using STANDARD names from the vocabulary above, not author shorthand>"],
 "inferred": ["<anything you concluded from ICT + educator knowledge that is NOT explicitly written/drawn>"]
}

Guidance:
- Read ALL text on the image — labels, boxes, side-panel explanations, captions,
  small annotations — route into text_content by channel. Do not drop text.
- IDENTIFY named frameworks when present (Power of 3, MMXM, SMR, Sharp Turn, OTE,
  Silver Bullet, 7 Rules, Judas Swing, SMT divergence, SBS wave, session profile)
  and put the name in `name`; tag the ICT framework family in `framework`; and
  give a best-guess `educator_guess` based on visual style + naming conventions.
- sequence is ONLY for price_path (case B). For reference_diagram and mixed
  (case A) leave sequence EMPTY — emitting steps there is fabrication and poisons
  the KB. For annotated_chart (case C) sequence is optional and weak; include it
  only if a clear ordered trade narrative is drawn.
- If the image is structured as COLUMNS or PANELS (bullish vs bearish columns,
  step-by-step panels, side-by-side conditions), preserve that structure inside
  text_content: prefix items with the panel/column label, e.g.
  "[Bullish column] bias from 8:30 open above midnight". Do NOT flatten a
  2-column layout into one flat list — that loses the structure that IS the
  teaching (the DailyPo3 mirrored checklists, the MMXM stage panels).
- For concepts_raw use STANDARD ICT names (FVG, OB, CSD, MSS, liquidity sweep,
  draw on liquidity, premium/discount, consequent encroachment, etc.), not the
  author's shorthand unless that's all that's written. This makes the unit
  queryable against the 172-concept vocabulary.
- Capture what is shown. If you infer something from ICT + educator knowledge
  that is NOT explicitly written/drawn, put it in "inferred" — do not silently
  invent, and do not silently drop. Grounding discipline: the image is the source
  of truth; your educator knowledge informs interpretation, not fabrication.
"""

# ---------------------------------------------------------------------------
# ICT-aware prompt — iteration 4 (v4) — CLASSIFICATION-FREE
# ---------------------------------------------------------------------------
# The challenge: do we need the `kind` taxonomy at all? The v3 data showed the
# model gets concepts/framework/educator/text right but gets the *label* wrong
# (BSL_DOL → mixed instead of price_path; ICT_Month10 → reference_diagram
# instead of mixed). The label was only useful for ONE downstream decision:
# "should sequence be emitted?" The rest of the extraction quality is
# independent of kind.
#
# v4 drops the `kind` field. Instead it asks the model to directly judge the
# ONE thing that matters: is the drawn path through zones the teaching method,
# or is the text/concepts the payload? This is a single binary judgment, not
# a 4-way taxonomy slot. The model answers it naturally from the content, not
# by mapping to a label.
#
# What we keep from v3:
#   - ICT + educator domain knowledge
#   - framework / educator_guess fields
#   - text_content with structure preservation (columns/panels)
#   - concept naming in standard ICT terms
#
# What we drop:
#   - the `kind` classification step (was failing on exactly the hard cases)
#   - the two-dimension classifier (the failure-prone middle step)
#
# What replaces it:
#   - `image_type`: a lightweight descriptor (not a gate) — "schematic" vs
#     "real_chart" vs "text_screenshot" vs "data_table" — useful for routing/
#     provenance but NOT load-bearing for sequence. This is a TAG not a GATE.
#   - `path_is_method`: the direct judgment — "is the ordered path the method?"
#     true => emit sequence; false => leave it empty. No taxonomy needed.
#
# If this works (extraction quality holds, sequence judgment improves), the
# §11b reconciliation gate simplifies: no more kind-consensus step. The gate
# becomes text_content union+dedupe + sequence presence reconciliation.
#
# STRUCTURE: the prompt is split into a DOMAIN_KNOWLEDGE block (swappable for
# non-ICT systems) and an EXTRACTION_LOGIC block (the path_is_method judgment +
# JSON schema, which is system-agnostic). To add a new trading system, create
# a new DOMAIN_KNOWLEDGE block and reuse the same extraction logic.
# ---------------------------------------------------------------------------

# --- Domain knowledge block: ICT / Smart Money Concepts (swappable) ---------
# This block contains everything the model needs to know about ICT methodology
# and the specific educators in this corpus. To add a non-ICT system (e.g. SMC,
# Wyckoff, price action), create a new *_DOMAIN_KNOWLEDGE block and swap it in.

ICT_DOMAIN_KNOWLEDGE = """ICT / Smart Money Concepts (SMC) trading methodology.
You know the material of these educators (the sources in this corpus):

- **ICT (Michael J. Huddleston, the Inner Circle Trader)** — originator. Frameworks:
  Power of Three (Po3 = AMD: Accumulation -> Manipulation -> Distribution), Market
  Maker Buy/Sell Model (MMXM), Optimal Trade Entry (OTE = 62-79% retracement),
  Silver Bullet windows (10-11am / 2-3pm ET), Judas Swing, draw on liquidity (DOL),
  ONS, killzones, IPDA. Visual style: idealized schematics with labeled zones,
  sometimes two-column (bullish/bearish) checklists.
- **LumiTrader** publishes BOTH the "ICT 2022" book (435pp full-page diagrams) AND
  the MMXM (Market Maker Buy/Sell Model) material. So MMXM-labeled content is
  LumiTrader's, NOT a separate source. Visual style: full-page diagram pages,
  framework panels with side notes, MMXM stages.
- **Flux** (NY Session Guide) — session profiling (Profiles 1-6), the 9:12/12:45/
  11:00/9:45/8:15 macros, weekly day-of-week profiles. Glossary-heavy.
- **fx4living** — ICT educator. Session-based approach with emphasis on time windows
  and session structure. May share terminology with Flux/ICT but distinct style.
- **MMxM trader** — distinct from LumiTrader; another educator who teaches the
  Market Maker Buy/Sell Model framework. (If the content is from the LumiTrader
  book, guess LumiTrader; if it's a standalone MMxM trader source, guess MMxM trader.)
- **Afyz** — ICT educator. (Add signature characteristics as they become known.)
- **Trader Diego** — ICT educator. (Add signature characteristics as they become known.)
- **Hydra** — ICT educator. (Add signature characteristics as they become known.)
- **Dexter** — ICT educator. (Add signature characteristics as they become known.)
- **TinyVizla** — ICT educator. (Add signature characteristics as they become known.)
- **AMTrades** — ICT educator. (Add signature characteristics as they become known.)
- **TTrades** — ICT educator. (Add signature characteristics as they become known.)
- **Arjo (Arjo15m)** — "Sharp Turn (ST) Entry Model"; signature terminology: Context
  High/Low (HTF buyside/sellside liquidity boundaries), Protected High (the swing high
  that forms the stop-loss reference), OD Entry (Overlapping Defense — aggressive
  entry at the top of the displacement/FVG zone), FLOD Entry (First Line of Defense
  — conservative entry at the lower threshold as price exits the imbalance), LLOD
  Entry (Last Line of Defense — final entry level), FVG Confirming ST (the Fair Value
  Gap that confirms the Sharp Turn). Real trade screenshots with markup. Short-side
  execution framework.
- **Kish** — TCM 7-Rule execution framework: Rule 1 daily order flow bias first;
  Rule 3 entry by OB anatomy; Rule 4 FVG filter; Rule 5 timeframe filter; Rule 6
  distrust wick swing inside FVG; Rule 7 order of delivery (SSL run -> FVG tag ->
  short-term low -> CSD). Numbered "rule #N" citations are a Kish signature.
- **StoicTA** — idealized SBS (Step-by-Step) wave models, 1-2-3-4-5 wave counts,
  fibonacci projections. Near-textless schematics where the wave order IS the content.

You already know general ICT / SMC methodology (FVG, OB, CSD, MSS, liquidity
sweeps, premium/discount, Po3, MMXM, SMR, OTE, Silver Bullet, Judas Swing, BPR,
killzones, AMD, IPDA, etc.). Use that knowledge to interpret the image. For
concepts_raw, use standard ICT names; your educator's shorthand should be mapped
to the canonical term (e.g. "feg" -> FVG, "CISD" -> CSD). The corpus uses a
176-concept vocabulary — if a concept on the image maps to a known canonical term,
use that term."""

# --- Extraction logic block (system-agnostic, reusable) ---------------------
# This block does NOT reference ICT-specific concepts. It works for any trading
# methodology — the domain knowledge is injected via {{DOMAIN_KNOWLEDGE}}.

_V4_EXTRACTION_LOGIC = """You are an expert in {{DOMAIN_KNOWLEDGE}}

INTERPRET the image using this knowledge — identify the framework and likely
educator, explain concepts in context, capture all text and structure. Do not just
label pixels.

=== THE ONE JUDGMENT THAT MATTERS ===

Look at the image and answer ONE question directly from the content:

  **Is the ORDERED DRAWN PATH through zones/levels the teaching method — i.e. is
  the sequence of price movement itself the lesson — or is the text/concept
  framework the payload (with the drawing merely illustrating)?**

  - If a drawn path shows an ordered method where step order carries meaning
    (e.g. "SSL raid -> FVG tag -> short-term low -> CSD confirms", or a 1-2-3-4-5
    wave count where the wave order IS the model) => emit `sequence` with the
    ordered steps. This is common for idealized model diagrams where there is
    little explanatory prose and the path is all.
  - If the image carries explanatory TEXT — rules, conditions, confluences,
    definitions, a checklist, a framework panel — the text/concepts ARE the
    payload, even if a price path is drawn. The path merely illustrates. =>
    leave `sequence` empty; put the content in `text_content` and `concepts_raw`.
  - If it's a REAL market chart (actual candlesticks) with markup, the path may
    show a trade narrative — emit `sequence` ONLY if a clear ordered trade
    narrative is drawn (entry -> stop -> target), otherwise leave it empty.
  - A page can have BOTH a sequence and text (e.g. a schematic showing an ordered
    path WITH explanatory notes alongside it). In that case emit BOTH the sequence
    AND the text_content. The question is not "sequence OR text" — it is "does the
    ordered path carry teaching meaning?" If yes, emit it; the text alongside it
    goes in text_content regardless.

  Do NOT force a taxonomy label. Just judge directly: does this image teach by
  walking through an ordered sequence, or by presenting text/concepts, or both?
  Then emit the right content accordingly.

=== RETURN JSON ===

{
 "image_type": "<schematic | real_chart | text_screenshot | data_table | mixed>",
 "path_is_method": true,
 "name": "<named framework/model if shown or identifiable, e.g. 'Daily Po3', 'MMXM', 'Sharp Turn (ST) Entry Model', 'SBS Model #2'. else null>",
 "framework": "<the framework family: Po3 | MMXM | SMR | OTE | Silver Bullet | TCM-7-Rules | SBS-wave | NY-session-profiling | other | null>",
 "educator_guess": "<ICT | LumiTrader | Flux | fx4living | MMxM-trader | Afyz | Trader-Diego | Hydra | Dexter | TinyVizla | AMTrades | TTrades | Arjo | Kish | StoicTA | unknown>",
 "bias": "<bullish|bearish|reversal|both|null>",

 "text_content": {
   "conditions": ["<listed conditions/requirements, verbatim>"],
   "confluences": ["<listed confluences/desirables, verbatim>"],
   "notes": ["<captions, definitions, caveats, side-panel explanations written on the image>"],
   "other_text": ["<any other meaningful text not captured above>"]
 },

 "reference_levels": ["<named levels/zones drawn, e.g. 'PDH', 'NY Midnight Open', 'ERL', 'BSL', 'SSL', '50%', 'OTE 62-79%'>"],

 "sequence": [
   {"order": 1, "action": "<what price does, in domain terms>", "position": "<premium|discount|equilibrium|null>",
    "range_liquidity": "<ERL|IRL|BSL|SSL|null>", "relative_to": "<level/zone or null>",
    "zone_label": "<label drawn for this step or null>"}
 ],

 "entry": "<entry rule if shown, else null>",
 "entry_mechanics": ["<if multiple entry types are shown (e.g. OD Entry = aggressive, FLOD Entry = conservative), list each with its name, description, and risk/reward characteristic. else []>"],
 "direction": "<long|short|both|null>",
 "target": "<target logic if shown, else null>",
 "concepts_raw": ["<concepts present, using STANDARD names from the vocabulary, not author shorthand>"],
 "inferred": ["<anything you concluded from domain + educator knowledge that is NOT explicitly written/drawn>"]
}

Guidance:
- `image_type` is a LIGHT TAG for routing/provenance, NOT a gate. It does not
  control anything downstream — it's metadata. Use it to describe what the
  image physically IS (a schematic drawing, a real candlestick chart, a pure
  text screenshot, a data table, or a mix). Do not overthink it.
- `path_is_method` is the ONE judgment that controls `sequence`. Set it true and
  emit sequence ONLY when the ordered path is the teaching method. Set it false
  and leave `sequence` [] when text/concepts are the payload.
- A page can have BOTH a sequence and text — emit both when both carry meaning.
- Read ALL text on the image — labels, boxes, side-panel explanations, captions,
  small annotations — route into text_content by channel. Do not drop text.
- IDENTIFY named frameworks when present and put the name in `name`; tag the
  framework family in `framework`; give a best-guess `educator_guess` based on
  visual style + naming conventions.
- If the image is structured as COLUMNS or PANELS (bullish vs bearish columns,
  step-by-step panels, side-by-side conditions), preserve that structure inside
  text_content: prefix items with the panel/column label, e.g.
  "[Bullish column] bias from 8:30 open above midnight". Do NOT flatten a
  2-column layout into one flat list — that loses the structure that IS the
  teaching.
- For concepts_raw use STANDARD concept names, not the author's shorthand unless
  that's all that's written.
- Capture what is shown. If you infer something from domain + educator knowledge
  that is NOT explicitly written/drawn, put it in "inferred" — do not silently
  invent, and do not silently drop. Grounding discipline: the image is the source
  of truth; your educator knowledge informs interpretation, not fabrication.
"""

ICT_PROPOSE_PROMPT_V4 = _V4_EXTRACTION_LOGIC.replace("{{DOMAIN_KNOWLEDGE}}", ICT_DOMAIN_KNOWLEDGE)

# ---------------------------------------------------------------------------
# Prompt registry — add new iterations here; the test harness picks by name.
# ---------------------------------------------------------------------------
PROMPTS = {
    "generic": None,  # signals: use chart_extract.PROPOSE_PROMPT (the baseline)
    "ict_v2": ICT_PROPOSE_PROMPT_V2,
    "ict_v3": ICT_PROPOSE_PROMPT_V3,
    "ict_v4": ICT_PROPOSE_PROMPT_V4,
}


def get_prompt(name):
    """Return (prompt_text, source_label). 'generic' -> the production baseline."""
    if name == "generic":
        from knowledge_ingest.sources.chart_extract import PROPOSE_PROMPT
        return PROPOSE_PROMPT, "PROPOSE_PROMPT (generic, production baseline)"
    p = PROMPTS.get(name)
    if p is None:
        raise ValueError(f"unknown prompt: {name}; known: {list(PROMPTS)}")
    return p, f"{name}"