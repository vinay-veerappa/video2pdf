# ICT Knowledge Ingestion System — Handover

**Purpose:** Turn heterogeneous ICT trading-education sources (video transcripts,
markdown, PDFs, blogs, chart/diagram images) into a typed, filterable, provenance-
tracked knowledge base that can later feed a trading companion (setup retrieval,
backtest validation, live analysis). Built to capture sources ONCE — provenance
and re-canonicalization mean nothing needs re-ingesting when the vocabulary grows.

This document is the single source of truth for state, architecture, decisions,
and how to run everything. Hand it to any future session to continue.

---

## 1. Current status (as of handover)

**Working & tested (text path):** transcripts, markdown, text PDFs → segment →
classify → typed extraction → units → vocabulary mapping → re-canonicalize →
vector store. Model choices settled by bake-off.

**Working (source front-stages):** blog fetcher (image-aware), mixed-PDF router
(text pages + chart pages), chart/diagram extractor (VLM-proposes → human-verifies).

**Chart path (VLM extractor):** RESOLVED — see §10a. The §6 prompt-loosening
hypothesis was correct (prompt/surfacing issue, not a VLM reading weakness); no
OCR needed. Model choice is KIND-DEPENDENT (§10b), and the kind taxonomy itself
grew substantially through real-world testing (§12, §13) — see those sections
before touching the chart path again. The reconciliation gate design (§11) is
NOT yet built.

**Deferred by design:** live/unannotated chart derivation (derive FVG/CISD/MSS from
raw candles — the hard "Aspect 3", better done from OHLCV data + detection code than
pixels); headless-browser web capture (manual save-as-PDF chosen instead); trade-journal
population (schema built, not yet used).

**Vocabulary:** 172 canonical concepts (grew from 56 → 82 via published-glossary
seeding, then 82 → 172 via three full-corpus harmonization rounds against
report_unmapped — see §15). Full-corpus unmapped rate went 79% → 51% → 37% across
those three rounds; remaining tail is mostly generic single-word noise
deliberately left unmapped (see §15a). VOCABULARY_REVIEW.md may be behind this
handover — treat this document as authoritative for vocab state.

---

## 2. Architecture — the whole system in one view

Everything reduces to TWO primitives: **text to extract** and **image to interpret**.
Every source type is a front-stage that feeds one or both.

```
SOURCES                     FRONT STAGE                    CORE PIPELINE
transcripts/.txt/.md  ──────────────────────────┐
text PDFs         ──► pdf_extract ───────────────┤
mixed PDFs/Slides ──► pdf_extract --mixed ──┬────┤──► segment → classify → extract
blogs             ──► blog_fetch ───────────┤    │   (typed KnowledgeUnit JSONL)
                                            │    │            │
chart/diagram images ───────────────────────┴────┼──► chart_extract (VLM propose
  (also from mixed-PDF/blog image pages)          │      → human verify → commit)
                                                  │            │
                                                  ▼            ▼
                                            units/*.jsonl (one store)
                                                  │
                            report_unmapped → grow vocab → recanonicalize
                                                  │
                                            build_lancedb → queryable vector store
```

**Two-channel principle for images:** an image carries meaning in BOTH its text
(labels, notes, conditions) AND its structure (the price path, spatial relations).
The chart extractor must capture both. (This drove the §6 prompt change.)

---

## 3. Module layout

```
knowledge_ingest/
  run.py                    entry point: ingest, --build-vectors
  config/config.py          ALL tunables: models, paths, batch sizes, thresholds
  schema/models.py          pydantic schemas: KnowledgeUnit, SetupPayload (+chart
                            fields: sequence/position/ERL-IRL/kind/text_content),
                            JournalEntry (separate store)
                            [FIXED §15c: ChartTextContent was defined AFTER
                            SetupPayload.model_rebuild() ran — deterministic
                            NameError on every fresh run. Moved above SetupPayload.]
  pipeline/
    ingest.py               orchestrator: segment→classify→extract, batched, resume,
                            collapse-guard, confidence-gate
    ollama_client.py        Ollama HTTP client, retry, robust JSON parse
    prompts.py              segment (timestamp + prose variants), classify, extract
                            [FIXED §15b: concepts_raw spec was too weak, emitted
                            junk like "analyze 8 o'clock" as if it were a concept]
    vector_store.py         LanceDB build + metadata-filtered search (multi-dir)
  vocab/
    ict_vocabulary.py       172 canonical concepts + aliases (canonical = YOUR name,
                            variants = aliases) — see §15 for harmonization history
    registry.py             domain registry: ict now; gex/etc. later
    VOCABULARY_REVIEW.md    decisions + [CHK] items to confirm
  sources/
    pdf_extract.py          text PDFs + --mixed per-page router (text/chart split)
    blog_fetch.py           fetch → clean text + download in-content images +
                            flag JS-rendered charts
    chart_extract.py        propose / compare / commit  (VLM chart → verify → unit)
    INPUTS.md               how to feed every source type
  report_unmapped.py        multi-dir: unmapped concepts by frequency → vocab growth
  recanonicalize.py         multi-dir: re-map concepts_canonical after vocab grows
  multidir.py               shared filename-collision guard for multi-dir tools
  examples/
    model_bakeoff.py        extractor grounding/calibration bake-off
    segmenter_bakeoff.py    segmenter verbatim-fidelity bake-off
    vlm_bakeoff.py          chart-reading grounding bake-off
    test_*.py               offline logic tests (no network/LLM needed)
```

---

## 4. Model choices (all settled empirically via bake-offs)

| Stage | Model | Why |
|---|---|---|
| Segmenter | gemma4:31b-cloud | 1.00 verbatim fidelity, fast (6.7s); copy-not-reason task |
| Classifier | gemma4:31b-cloud | cheap bounded judgment |
| Extractor (text) | deepseek-v4-flash:cloud | best calibration (honest confidence + inferred_fields), fast, no hallucination |
| Embeddings | nomic-embed-text-v2-moe | user-pulled |
| Chart VLM | KIND-DEPENDENT (see §10b, §11a) | minimax-m3 best for price_path (thorough sequences); qwen3.5 best for reference_diagram (branch-keys checklists correctly, doesn't fabricate). Do NOT lock one model globally. |

**Rejected & why:** qwen3.5/qwen3.6 (reasoning models → empty output under json_mode,
33–88s latency); gemma4 as extractor (miscalibrated — reports 1.0 confidence even when
dropping fields); coder/OCR/medical models for extraction (wrong domain). qwen3.5 &
gemma4 as chart VLM: flatten sequence / over-segment vs minimax.

**The bake-off discipline is the reusable lesson:** never trust a model choice by
reputation — test on known-answer inputs, score the failure mode that matters
(grounding/calibration, not fluency), pick the one needing least correction.

---

## 5. Key design decisions & rationale

- **Typed knowledge, not prose blobs.** Every unit is one of: setup / contextual /
  framework / tip / psychology / anecdote — each with its own schema. Only `setup`
  and testable `contextual` flow toward backtest validation.
- **Grounding discipline.** Extraction fills only what's stated; inferences go in
  `inferred_fields`; per-unit `extraction_confidence`. This keeps the eventual
  execution layer auditable.
- **Canonical = your dialect.** Vocabulary canonical id is YOUR preferred term;
  every other phrasing (incl. transcription errors like "feg"→fvg, "CISD"→csd) is an
  alias. RULE REFINED in §15a: bare short acronyms were originally banned wholesale
  (over-match risk: "eo"→video, "breaker"→circuit breaker, "smt "→wasmt) — but that
  risk applies to matching against free TEXT. Matching here is actually against
  already-EXTRACTED short concept phrases (raw_concepts), where the risk is much
  lower. Re-added many bare acronyms deliberately (eo, ob, breaker, smt, rr, gap,
  dr) once this was recognized. Re-verify this assumption if a future extractor
  version starts emitting longer free-text concept strings instead of short phrases.
- **Capture once.** Provenance (source_type, image_path, source_page, url) on every
  unit + idempotent recanonicalize = grow vocabulary anytime, never re-ingest.
- **Charts: human-in-the-loop.** VLM proposes structured setup; you verify (it's not
  reliable enough unverified — misreads geometry/levels). No prose output (you have
  the source images). Charts are mostly ILLUSTRATIONS of concepts you already have.
- **Journal is separate.** Your trades ≠ teaching knowledge. Different schema
  (JournalEntry), different store, feeds "review my execution", not concept KB.
- **Two-channel images.** Text + structure both captured (§6).
- **Web JS-charts:** manual save-as-PDF → mixed-PDF route chosen over a headless
  browser dependency. blog_fetch flags JS-rendered charts it can't download.

---

## 6. Chart-path text-channel fix (RESOLVED — see §10a)

The original "immediate next step" tested a hypothesis: the chart extractor's
missing side-panel text was a PROMPT/SCHEMA issue, not a VLM reading weakness.
**Confirmed RESOLVED in §10a** — no OCR needed. The prompt was loosened (added
`kind`, `text_content` block, optional `sequence`, `inferred` list, jfif/gif/bmp
globs) and all three VLMs then captured the text channel. The remaining chart work
is the reconciliation gate (§11b) and the VLM-consensus founding-premise test (§13g) —
see §16 for the current plan.

---

## 7. How to run — full workflow

```
# TEXT SOURCES
python -m knowledge_ingest.run --input <transcripts> --output <out>
python -m knowledge_ingest.run --input <dir> --output <out> --source-type markdown

# PDFs (text) and mixed PDFs / Google Slides (export Slides as PDF)
python -m knowledge_ingest.sources.pdf_extract --in <pdfs> --out <textout>            # text-only
python -m knowledge_ingest.sources.pdf_extract --in <pdfs> --out <textout> --mixed    # + chart pages
python -m knowledge_ingest.run --input <textout> --output <out> --source-type pdf

# BLOGS (image-aware; flags JS-rendered charts)
python -m knowledge_ingest.sources.blog_fetch --url-file urls.txt --out <blogout> --source <key>
python -m knowledge_ingest.run --input <blogout> --output <out> --source-type blog

# CHART / DIAGRAM IMAGES  (propose → verify → commit)
python -m knowledge_ingest.sources.chart_extract propose --images <charts> --output <out>
#   ...edit <out>/_review/*.json: fix sequence/relations, set "status":"approved"
python -m knowledge_ingest.sources.chart_extract commit  --output <out>
#   (compare mode to test/choose a VLM: chart_extract compare --images ... --models ...)

# AFTER A BATCH (any source) — grow vocab & build store
python -m knowledge_ingest.report_unmapped  --units <dir1> <dir2> ... --vocab ict --min-count 5
#   ...add worthwhile concepts to vocab/ict_vocabulary.py (your names canonical)
python -m knowledge_ingest.recanonicalize   --units <dir1> <dir2> ... --vocab ict   # dry-run first
python -m knowledge_ingest.run --build-vectors --units <dir1> <dir2> ... --db knowledge.lancedb
```

**Sequencing rules:** do ONE domain (ICT) at a time so vocabularies don't mix; keep
all runs on the SAME vocabulary file; grow vocab once at the end from the full
corpus; re-canonicalize (cheap, idempotent) every time vocab changes; re-EXTRACTION
is never repeated (expensive), only re-canonicalization.

---

## 8. Dependencies

```
pip install pydantic requests lancedb pyarrow          # core + vector store
pip install pdfplumber                                  # PDF fallback (poppler-utils preferred)
pip install trafilatura beautifulsoup4                  # blog fetch
# poppler-utils (pdftotext/pdffonts/pdfimages/pdftoppm) for PDF text+mixed routing
# Ollama running locally with the models in config/config.py
```

---

## 9. The bigger roadmap (what this foundation enables)

1. **Knowledge base** (this system) — DONE for text, in-test for charts.
2. **Strategy-candidate registry** — filter setups/testable-contextual units → the
   backtestable idea bank.
3. **Backtest-validation loop** — test candidates against 20y OHLCV + existing
   detection code (Edgeful/FVG) → writes validated stats, flips epistemic_status,
   populates linked_stat_ids. Closes concept→candidate→stat→reasoning loop.
4. **Live chart analysis (Aspect 3)** — derive structure from OHLCV data + detection
   code + KB definitions (reliable) rather than pixels (unreliable). The hard payoff.
5. **Trade journal** — schema exists; populate from journaled charts; feeds companion.
6. **The companion** — reads KB + live data → daily plan, levels, caution score,
   intraday updates. Semi-automated execution is the eventual goal (audit-first).

---

## 10. Post-extraction work queue & session findings (added after chart-path work)

This section records decisions and diagnostics from the session that fixed the
chart text-channel and surveyed the text corpus. Read before the next work block.

### 10a. Chart path — RESOLVED (supersedes §6)
The §6 hypothesis was correct: the missing side-panel text was a PROMPT/SURFACING
problem, not a VLM reading weakness. Two fixes landed:
- **compare() was discarding text_content** before writing the _compare JSON — so the
  §6 re-test was structurally blind to the field it was meant to judge. Fixed to
  surface kind / text_content / text_content_counts / bias / reference_levels /
  inferred, not just sequence. (n_steps==0 on a reference_diagram is CORRECT, not a
  failure — judge those on text_content.)
- With the loosened prompt, all three VLMs captured the text channel fully on the
  Daily Po3 diagram. **No OCR needed** — the glm-ocr / deepseek-ocr merge stays
  deferred and can be dropped from the plan unless a genuinely text-heavy image later
  fails.
- **commit() was dropping text_content / kind / inferred** — it wrote only
  sequence/reference_levels into the setup unit. Fixed to persist all three.
- **schema/models.py**: SetupPayload did NOT actually have `kind` or `text_content`
  (§3 claimed it did — §3 was aspirational/wrong). Added a `ChartTextContent`
  sub-model and `kind` / `text_content` / `chart_inferred` fields. This is
  LOAD-BEARING, not cosmetic: KnowledgeUnit.retrieval_text() calls
  payload.model_dump_json() and build_lancedb loads units back through the model, so
  an undeclared field is silently dropped at vector-build time. Without the schema
  change the commit fix would look right and still lose the text one stage later.
  **UPDATE (§15c): this schema change itself had a bug — see §15c, now fixed.**

### 10b. Model choice for charts is KIND-DEPENDENT (refines §4)
On the reference_diagram test, ranking INVERTED vs the §4 price-path finding:
- **qwen3.5** gave the best reference_diagram read — it correctly branch-keyed the
  mirrored bullish/bearish checklists (least correction needed).
- **minimax-m3** fabricated a 7-step sequence for a diagram with no single price path
  (exactly the "force it into a shape it isn't" failure) and inflated text totals via
  duplication across channels. Best price-path reader (§4), worst here.
- **gemma4** shredded the checklist across notes/other_text and lost the bias split.
So: do NOT "lock minimax" globally as §6 anticipated. Route by `kind` —
minimax for price_path, qwen3.5 for reference_diagram. (Still verify per-batch.)

### 10c. text-corpus diagnostics (6-file sample, 245 units — CONFIRM AT FULL CORPUS)
- **~36% of units are below 0.6 confidence** (88/245). Either transcripts are
  genuinely loose (live thinking-aloud) or the gate is too aggressive. Not decidable
  until full corpus.
- **~47% of units have raw concepts that map to ZERO canonical vocab** (116/245) —
  and the rate is HIGHER in high-confidence units (53%) than low (36%). KEY INSIGHT:
  vocabulary gaps and extraction confidence are SEPARATE problems. Confident, correct
  extractions are failing to map simply because the vocab doesn't contain the term yet
  (or the "concept" is junk — see below). Do not conflate the two.
  **UPDATE (§15): the 47%-unmapped side of this is now resolved — full-corpus
  unmapped went 79%→37% via three harmonization rounds. Re-run the confidence
  histogram on the real corpus now that vocab isn't confounding it (§15d, still
  open).**
- **Junk concepts**: extractor emits non-concepts as concepts (e.g.
  concepts_raw:["analyze 8 o'clock"]). This is a prompts.py EXTRACT issue — not fixable
  by vocab growth or review. Inspect the extract prompt's concept instruction.
  **FIXED (§15b): concepts_raw spec in prompts.py was underspecified — no negative
  examples, no test for what counts as a "concept". Rewritten with a shared spec
  used by both the single and batched classify prompts.**
- **Held-back units are NOT flagged in the units JSONL.** The run log reports "N held
  back for review" but the emitted .jsonl carries no held/review/status field (only
  epistemic_status, uniformly unvalidated_concept). OPEN QUESTION for next session:
  where does the pipeline write the held-back set — separate file, separate dir, or
  logged-and-left? This determines whether triage targets a marked subset or
  re-triages by confidence threshold. **STILL OPEN.**

### 10d. Post-extraction sequence (do IN THIS ORDER when the full run completes)
1. ~~**report_unmapped across the WHOLE corpus** (§7) → decide real ICT terms vs
   junk → grow ict_vocabulary.py ONCE → recanonicalize~~ **DONE (§15).** Ran across
   all 261 files / 8724 raw concepts, three rounds: 79%→51%→37% unmapped. Vocab grew
   56→82 (glossary seed, prior session) →172 (harmonization, this session). The Flux
   p66 glossary seed WAS folded in (§12d/§14) before this pass, as planned.
2. ~~**Fix the extract prompt's concept emission** (junk like "analyze 8 o'clock").~~
   **DONE (§15b).**
3. **Re-assess the low-confidence set** against the real (post-recanonicalize)
   numbers before building any review tooling. **STILL OPEN — next concrete step.**
   User ran recanonicalize and reported "the report did not throw out anything as
   such" for report_unmapped (i.e. mapping is stable/good) — but the CONFIDENCE
   histogram from §10c (36% below 0.6) has not been recomputed since vocab grew.
   That recomputation is what "re-assess" means here; it isn't a tool that exists
   yet, it's a small script to write (walk units JSONL, histogram
   extraction_confidence, ideally cross-tab against whether concepts_canonical is
   now non-empty).
4. **NotebookLM reconciliation gate — ONLY IF the remaining low-confidence set
   justifies it.** Design (deferred, not built): NotebookLM re-reads the full
   transcript per low-confidence unit (anchored by timestamp_range) and re-proposes in
   STRICT structured output; a local script compares its knowledge_type + grounded
   claim to the original and routes → auto-promote (both agree) / auto-drop (both say
   thin) / human-review (disagree). Only the disagreement set needs human eyes —
   this inverts the ~4000-item manual review into a small ambiguous queue.
   CAVEATS: (a) NotebookLM is a UI product, not API-scriptable — the middle step is
   semi-manual (generate prompts → paste → save structured answer → local reconcile);
   verify current NotebookLM batch/API capabilities before architecting. (b) NotebookLM
   validates only that the TRANSCRIPT says something, never that it's TRUE — it may
   flip knowledge_type and rescue grounding but must NEVER touch epistemic_status
   (stays unvalidated_concept until the backtest loop). Wire that guard into the review
   step. (c) Calibrate auto-promote/auto-drop thresholds on a hand-checked transcript
   FIRST (bake-off discipline); auto-dropping a real setup is the expensive error.
   **STILL OPEN — decide after step 3.**

---

## 11. Chart/image reconciliation gate & unified image-with-text path (design note)

Written after a 7-chart bake-off (DailyPo3, LRS, ict_mmxm_notes, RTH ORG,
Arjo15m, BSL_DOL, ICT_Month10) across gemma4/qwen3.5/minimax-m3. NOT yet built —
this is the design to build against once the text extraction run completes.
**Text extraction run + vocab harmonization are now done (§15) — this is the next
major body of work once §10d step 3/4 are settled.**

### 11a. What the 7-chart batch actually showed (refines §10b)
The clean "route by kind" rule from §10b is NOT directly implementable, because
`kind` is the thing the models DISAGREE on — and the disagreement correlates with
the failure mode:
- Models agreed on kind for only 3/7 charts. minimax labeled 5/7 as price_path
  (its bias), and it is also the model that FABRICATES sequences (6 steps on the
  DailyPo3 reference_diagram that gemma+qwen both scored 0; 18 steps on ICT_Month10
  where qwen scored 0).
- Where kind was unambiguous and agreed: LRS (true price_path) → minimax's thorough
  11-step read is an ASSET; DailyPo3 (true reference_diagram) → minimax fabricates,
  qwen correctly branch-keys with 0 steps. The §10b inversion HOLDS where it applies.
- CONCLUSION: you cannot route on minimax's own kind label (says price_path → sent
  to price_path path → its over-segmentation is the weakness). kind-detection must
  come from the HONEST classifiers (gemma+qwen agree 5/7; neither shows the
  price_path bias), and minimax used only as a READER once kind is decided.
- `text_items_total` is a BAD metric — it rewards duplication. minimax "wins" it
  5/7 partly by echoing content across conditions/confluences/notes/other_text and
  listing axis labels twice (LRS: "-3,-2.5,-2..." counted in two channels). Do not
  score models on it; drop or rename it in the compare summary.

### 11b. The reconciliation gate (scale: ~100 charts — the awkward middle)
100 is too many to human-verify all (a full tedious day) but too few to justify
high-precision auto-resolve. Target: turn 100 into a ~20-30 human queue. That
allows a CONSERVATIVE gate — when unsure, kick to human, because the queue is small
enough to absorb it. No pressure to auto-resolve aggressively.

Gate logic over the 3-model compare output:
1. **kind** = gemma+qwen consensus (IGNORE minimax as a classification vote; use it
   only as a reader). If gemma+qwen disagree → human queue.
2. **route the reader**: minimax for price_path; qwen for reference_diagram / mixed.
   Take the routed reader's structured output as the draft.
3. **sequence guard (conservative, asymmetric)**: require corroboration to KEEP a
   sequence, not to drop one. If the routed reader emits steps but the other two
   scored ~0, suppress the sequence (keep text_content) and/or flag for human.
   Rationale: auto-keeping a FABRICATED sequence poisons the KB silently; auto-
   dropping a REAL one is recoverable (still in text_content + reference_levels).
   Bias toward the recoverable error.
4. **text_content / concepts**: UNION across all three models, then de-duplicate.
   Cross-model agreement on a text item = strong evidence it's really on the image.
   This simultaneously fixes under-reading (qwen/gemma miss) and minimax's
   duplication inflation, in one step.
5. **auto-commit** only where kind agreed AND sequence-existence agreed. Everything
   else → small human queue.
Build it THRESHOLD-PARAMETERIZED so it can tighten if chart count ever grows; tuned
conservative for ~100.

### 11c. Calibration
Hand-label the 7 existing charts (correct kind; does a real sequence exist?) as the
starter calibration set; measure the gate's auto-resolve accuracy against it before
turning it loose. 7 is thin — grow it with reference_diagram / mixed pages harvested
from mixed PDFs (see 11d), which are exactly the kind-detection stress cases.
**Grew to ~17 images by §13g — still no VLM consensus run against these hand
labels (needs local Ollama). That remains the untested founding premise (§12e).**

### 11d. Unifying charts and mixed-PDF pages (the "broadbase" answer)
Everything reduces to §2's two primitives: text-to-extract, image-to-interpret. A
mixed PDF is not a new problem — it decomposes into those two. PROPOSED CHANGE to
the pdf_extract --mixed router:
- CURRENT: router splits each page into a text-stream + image-stream (a third,
  untested decision point).
- PROPOSED: router classifies each PAGE as image-dominant vs text-dominant and sends
  the WHOLE page down ONE path. Image-dominant → the SAME VLM-does-both + reconciliation
  gate the charts use (§10a proved a single VLM with a both-channels prompt beats
  OCR+VLM — this transfers to PDF pages). Text-dominant → the existing text pipeline
  (preserves deepseek-v4-flash, the bake-off-winning text extractor, and the
  segment/confidence-gate machinery).
- WHY: keeps the best text extractor where prose dominates; applies the proven
  single-VLM approach where images dominate; collapses charts and PDF-image-pages
  into ONE code path with ONE calibration set. Fewer moving parts than router+2 gates.
- COST/CATCH: this does not eliminate classification, it MOVES it — from "split page
  into streams" to "is this page image- or text-dominant?" That page-dominance
  classifier is the new critical untested component. It's a cousin of the chart-kind
  classifier, which we just learned is unreliable (models disagreed 4/7). So it needs
  the same treatment: honest classifier (gemma+qwen consensus, NOT minimax), and
  calibrate on known pages first. Calibration set must span standalone charts AND PDF
  pages of each type. **REFINED in §12b: page-dominance is 3-way, not binary — see
  below.**

### 11e. Sequencing (do NOT broadbase early)
Running PDFs through NOW — before the gates exist — just generates more unsorted
output into a system whose triage isn't built. Order:
1. Finish the full text extraction run (already waiting). **DONE.**
2. Build + calibrate the two reconciliation gates (transcript low-confidence §10d;
   chart/image §11b) on data in hand. **NOT YET DONE — this is the next major
   body of work.**
3. THEN run a few mixed PDFs deliberately, to (a) test the page-dominance classifier
   — the one truly untested link — and (b) harvest chart pages to grow the §11c
   calibration set as fresh validation of already-built machinery.
The general solution is TWO reconciliation gates (one per primitive), not a per-
source-type solution. PDFs don't need their own gate; they need the router to send
each page to the correct existing gate.

---

## 12. Findings from first real PDFs + off-distribution images (validates & corrects §11)

Reviewed 3 sources OUTSIDE the original 7-chart set to test the §11 design before
building: Flux "New York Session Guide" (67pp mixed PDF), a Profiling flowchart PNG,
and a Notion "Advanced Order Block Theory" doc (prose + annotated trade screenshots).

### 12a. The 4-kind taxonomy is INCOMPLETE (corrects §11a / §10a schema)
The flowchart PNG is a decision tree (economic-calendar → day branches → outcomes
like Classic expansion / Void profile / Midweek reversal). It is NONE of the four
kinds (price_path / reference_diagram / annotated_chart / mixed). Its meaning lives
in the GRAPH EDGES (which node leads to which) — and NO current schema field
(sequence / text_content / reference_levels) captures edges. Forcing it into a kind
would flatten the branching, which for a decision tree is the whole content.
- ACTION: add a 5th kind `flow_diagram` (a.k.a. decision_tree) with an edge-list
  representation (nodes + directed edges), OR accept these dump into text_content as
  a flat node list and lose the logic. For ICT weekly/daily profiling trees the
  branching is load-bearing → an edge list is worth it. Decide during schema work.
- This is the concrete proof that off-distribution images exist and the taxonomy
  must be tested, not assumed. Fold the flowchart into the §11c calibration set.

### 12b. Mixed-ness is NOT binary — at least THREE profiles (refines §11d)
The page-dominance classifier in §11d cannot be a simple text-vs-image call. Three
distinct mixed profiles seen across the samples:
- **illustration-subordinate** (Flux mixed pages 6,7,13,14,16,59-61): diagram merely
  illustrates prose that stands alone. Route whole page → TEXT pipeline; run VLM only
  as optional supplementary pass. Losing the figure loses ~nothing. §11d WORKS here.
- **decision-logic** (the flowchart): whole page is the graph. → image/flow path.
- **commentary-on-chart** (Notion OB doc): prose explicitly refers to THIS screenshot
  ("look how first leg displaces past 50%", "point out the inversion OBs"). Text and
  image are CO-DEPENDENT — neither channel is complete alone. A hard page-split
  BREAKS this case. The classifier MUST detect co-dependence and keep both channels
  together (closest to the annotated_chart / journal case).
- So the page classifier is 3-way at least (text-dominant / image-dominant /
  co-dependent), not binary. Co-dependence detection is the new hard sub-problem.

### 12c. §11d page-split VALIDATED on a real document (with the mixed caveat now evidenced)
Flux 67pp splits cleanly ~60% text-dominant / ~25% image-dominant / ~15% truly mixed,
and the mixed pages are illustration-subordinate (diagram inessential). So on THIS
document the "send whole page down one path" approach loses almost nothing and
preserves deepseek-v4-flash where prose dominates. §11d holds — but this is ONE
document; the flowchart + Notion cases prove it won't hold universally. Keep the
router THRESHOLD-PARAMETERIZED and calibrate the page classifier on all three mixed
profiles, not just clean pages.

### 12d. The Flux glossary (p66) is a ready-made VOCABULARY SEED (feeds §10d step 1)
Page 66 is an author-provided glossary: ~30 acronyms with expansions — DOL, DOT,
SIBI, BISI, FVG, OB, CISD, BRKR, V.I, NYO, LC, LO, ERL, IRL, REQH, REQL, ITH/L,
STH/L, LTH/L, HRLR, LRLR, SMT, TS, IFVG, SMR, MMXM, MSS, IOF, S.D, LOD/HOD, LOS/HOS.
This maps straight onto the 47%-unmapped-concept problem (§10c) — many unmapped raw
concepts are almost certainly these exact acronyms. Fold this glossary into
ict_vocabulary.py as a seed (canonical term + expansion) BEFORE/ALONGSIDE the §10d
report_unmapped pass — the author handed us the list, no discovery needed. NOTE:
these are general-ICT canon confirmed by a TCM-adjacent educator; still tag any that
need confirming against YOUR dialect as [CHK] in VOCABULARY_REVIEW.md. **DONE — see
§14.**

### 12e. Still-unverified assumption (unchanged, now more urgent)
These PDFs did NOT test the reconciliation gate's founding premise — that cross-model
CONSENSUS predicts CORRECTNESS. That still needs ground-truth labeling of the 7
charts + the flowchart + a couple of PDF chart pages (hand-label: correct kind; does
a real sequence/edge-set exist?). More urgent now because the flowchart proves the
TAXONOMY needs validation, not just the models. Do this before building either gate.
**STILL OPEN — needs local Ollama run against the ~17-image calibration set (§13g).**

---

## 13. The "sequence" ambiguity — resolved (this is the key insight of the labeling session)

Ground-truth labeling forced a definition of "sequence" and exposed that the field
was underspecified. This underspecification — NOT model quality — is the real root of
the biggest apparent model disagreement (ICT_Month10: minimax 18 steps, gemma/qwen 0).

### 13a. Two questions were being conflated under "sequence"
1. Is there a price PATH drawn on the image? (a squiggly line through zones — almost
   EVERY ICT teaching chart has one; it's how ICT explains anything)
2. Should the extractor emit an ordered STEP LIST as the primary representation —
   i.e. is the teaching point "do step 1, then 2, then 3" such that flattening to an
   ordered list PRESERVES the meaning?
minimax answered #1 mechanically (saw a line → emitted 18 steps). gemma/qwen answered
#2 (content is labeled STAGES, not a recipe → 0 steps). Neither misread the pixels —
they answered different questions because the prompt never said which. The 18-vs-0
"disagreement" is a SPEC gap, fixable upstream in the prompt/schema, cheaper and more
reliable than any downstream gate. (Tempers the "disagreement = ambiguity" premise:
some disagreement is just spec ambiguity. **Same lesson recurred in §15b — the
concepts_raw junk-concept bug was also a spec gap, not a model gap.**)

### 13b. A/B/C typing — CONFIRMED against user's ICT knowledge; basis for schema + gate
Every teaching image is one of:
- **(A) framework-illustrated-on-a-path** — price line is a teaching aid; real content
  is labeled stages/checklist. `sequence` MUST stay empty; content → labeled-stages /
  text_content. Emitting steps here is the fabrication / poison-the-KB error.
  Examples: DailyPo3, ICT_Month10.
- **(B) true price_path** — the ordered movement IS the method; step order carries
  meaning. `sequence` is the faithful container. Residual risk is MILDER: granularity
  disagreement (models split 5/6/8 step count), NOT invention. Examples: LRS, BSL_DOL.
- **(C) everything else** — reference_diagram (checklists), flow_diagram (§12a
  flowchart), annotated_chart (trade screenshots).
The A-vs-B call is a property of `kind`, which gemma/qwen classify correctly and
minimax does not. So the rule is implementable: classify A/B/C via gemma+qwen
consensus, THEN decide whether `sequence` is even the right container. minimax is a
READER (used in B, where its thoroughness helps), never the classifier.

### 13b-REVISED. The A/B line is TEXT-PRESENCE (user's actual criterion — supersedes above)
User correction on RTH_ORG (which lists ordered steps yet is A) pinned the real line:
**if the image carries a block of explanatory/rule/framework TEXT, the text is the
payload and the path merely illustrates → A. B is only when there is essentially NO
explanatory text and the drawn path is all there is to read (pure idealized model,
e.g. LRS: zone labels on a curve, no conditions/rules prose).**
This is near-mechanically detectable and aligns with the existing `text_content`
field: substantial text_content → A; near-empty text_content + a path → B.

CONSEQUENCE (big, simplifying):
- ICT teaching charts are heavily annotated → the VAST MAJORITY are case A.
  Pure geometry-only B diagrams (LRS-type) are the RARE case.
- For case A the correct output HAS NO SEQUENCE → the fabricated-sequence problem
  (minimax's 18 steps etc.) largely DISSOLVES. You're not guarding a sequence, you're
  simply not requesting one. The asymmetric sequence-guard of §11b becomes a
  rare-case afterthought, not the centre of the gate.
- The chart gate is therefore mostly about the TEXT channel: union-and-dedupe
  text_content across the 3 models (already established to fix both under-reading and
  minimax duplication). Sequence handling only matters for the rare B images, where
  the sole concern is granularity reconciliation (median / union-then-collapse).
- Prompt/schema implication: for A, prompt should NOT push for an ordered sequence;
  it should extract conditions/confluences/notes/labeled-stages. `sequence` optional,
  populated only when text is near-absent and a path is present (B).
  **CORRECTED AGAIN in §13b-REVISED-2 below — "rare" turned out wrong on both counts.**

### 13b-REVISED-2. Text-presence is NOT sufficient — needs a 2ND dimension (LumiTrader p205)
Ground-truth extension into the LumiTrader book (§13e) BROKE the one-dimensional rule.
p205 is two REAL annotated price charts (candlesticks + marked zones), almost no prose
— so by text-presence alone it would classify B. But it is NOT B; it's an
`annotated_chart` (case C). Both a pure idealized model (LRS → B) AND a lightly-marked
real chart (p205 → C) have little text. Text-presence cannot separate them.
CORRECTED RULE — classify on TWO dimensions:
  1. **schematic/idealized drawing vs real market data?**
     - real market data (candlesticks, actual price) → annotated_chart (C), regardless
       of how little text.
     - idealized/schematic drawing (clean curve, drawn zones) → go to dim 2.
  2. **among schematic images: text-present → A; geometry-only → B.**
So B is doubly narrow: schematic AND near-textless (LRS, BSL_DOL). "B is rare" still
holds — arguably MORE rare, since low-text real charts are C, not B.
**FURTHER CORRECTED in §13g: B is not rare overall, it CLUSTERS by source. See below.**

### 13c. Two different guards, not one (refines §11b sequence-guard)
- Case A: guard against sequence EXISTING at all (suppress fabricated steps).
- Case B: sequence is fine; guard against granularity noise (reconcile step count,
  e.g. take median or union-then-collapse across models).
The single "sequence guard" in §11b was too blunt — it conflated these. Fabrication
risk lives only in A; granularity noise lives only in B.

### 13d. Ground-truth labels — COMPLETE (user-confirmed)
All images are co-dependent (text+image read together) EXCEPT pure-text-payload and
pure-geometry edge cases; user confirmed the price_path/mixed charts are co-dependent.
| image | kind | case | sequence faithful? | co-dependent? |
|---|---|---|---|---|
| DailyPo3 | reference_diagram | A | no (mirrored bias checklists) | no (text is payload) |
| LRS | price_path | B | yes (path = method, no rule-text) | image-led |
| ICT_Month10 | mixed/reference | A | no (labeled MMXM stages) | yes |
| BSL_DOL | price_path | B | yes (granularity split 5/6/8) | yes |
| Profiling_FlowChart | flow_diagram | C | no (branching logic) | no (self-contained graph) |
| ict_mmxm_notes | mixed | A | no (framework + notes panel) | yes |
| RTH_ORG | mixed | A | no (rules/confluences text = payload) | yes |
| Arjo15m | annotated_chart | C | weak (minimal trade path) | yes |

Key: only LRS and BSL_DOL are case B (geometry-carries-content). Everything with a
real text block is A. This 8-image set (+ the flowchart) is the STARTER CALIBRATION
SET for the chart gate (§11c) — hand-labeled, ready to measure auto-resolve accuracy
against. Grow it with reference_diagram / mixed / flow pages harvested from PDFs.

### 13e. Ground-truth EXTENDED into LumiTrader book (3 pages, high return)
Rendered 3 diagram pages from the 435-page LumiTrader book to test the labeling rules
beyond the 8-image set. NOTE: every page of that book is a full-page rendered image
(no text layer with separable figures — unlike Flux), so harvesting = render + eyeball.
2 of 3 pages CORRECTED or EXTENDED the model:
| image | kind | case | seq faithful? | co-dep? | what it showed |
|---|---|---|---|---|---|
| LumiTrader p145 (Daily Po3) | reference_diagram | A | no | no | CONFIRMS A rule (annotated framework) |
| LumiTrader p205 (annotated charts) | annotated_chart | C | weak | yes | BREAKS text-only rule → see §13b-REVISED-2 |
| LumiTrader p150 (price cycle) | cycle_diagram (NEW) | C | no | no | NEW taxonomy type (cyclical states) |

Two model corrections came out of it:
1. **A/B needs TWO dimensions** (schematic-vs-real-data, THEN text-presence) — §13b-REVISED-2.
2. **Taxonomy now has ≥2 graph-type diagrams**: flow_diagram (§12a flowchart, directed
   branching) and cycle_diagram (p150, cyclical states). CONSIDER collapsing both into
   ONE `relationship_diagram` / `graph` kind with an edge-list representation (directed
   edges for flow, cyclic edges for cycle) rather than proliferating kinds. Decide at
   schema time. Either way the schema needs an EDGE representation these both require
   and no current field provides (same gap flagged in §12a).

VLM consensus test (does agreement predict correctness?) still NOT run on these — that
needs the local Ollama models. These hand labels are the ground truth to measure that
test against when it runs. This 11-image set (8 original + flowchart + p150 + p205,
plus p145) is the current calibration set. **Grew to ~17 by §13g.**

### 13f. TWO more off-distribution types + a THIRD primitive (user-confirmed)
Two more uploads pushed the taxonomy again:
- **Text-as-image screenshot** (PF/EV notes PNG): pure prose, NO diagram. User says
  these are COMMON (Notion notes etc.). CONSEQUENCE: the page/image-dominance router
  (§11d/§12b) MUST detect "pure text screenshot" and route it to the TEXT pipeline
  (deepseek), NOT the chart VLM. This is a required router branch, not an edge case —
  a common input type mangled if sent to the chart gate. Likely the EASIEST classify
  call (lots of text, no chart geometry), so cheap to get right, but must be explicit.
  Kind label: `pure_text` → text pipeline. Knowledge-type is framework/psychology/tip.
- **Data table / stat matrix** (3rd-hour probability grid): a MATRIX of conditional
  probabilities (condition → outcome → %). Fits NONE of the chart kinds; content is
  cell values + row/col structure. User: RARE (one-off), but wants it recognized as a
  THIRD PRIMITIVE. So §2's "two primitives" (text-to-extract / image-to-interpret)
  becomes THREE: + structured/tabular data.
  - Because rare: DON'T build a VLM table-parser for one image. Pragmatic path =
    manual transcription into structured rows. The value of naming the primitive is
    routing clarity: the NEXT table already has a lane (neither chart gate nor text).
  - SPECIAL ROLE: this table is BACKTESTABLE STAT content — the thing the §9 loop
    consumes. But the numbers are THIRD-PARTY (someone else's backtest), so they must
    NOT flow into linked_stat_ids as if self-computed. Needs its own epistemic_status
    (e.g. "third_party_stat"), distinct from "validated" (your own loop) and
    "unvalidated_concept". Wire that distinction if tables become more common.

TAXONOMY as it now stands (grew 3x this session — assume it will grow again):
  price_path | reference_diagram | annotated_chart | mixed  (original 4)
  + flow_diagram (§12a)  + cycle_diagram (§13e)  → consider merging as relationship/graph
  + pure_text (route to text pipeline, not a chart kind at all)
  + data_table / stat_matrix (the third primitive)
Lesson: the taxonomy is NOT closed. Every batch of real off-distribution inputs has
found a new type. Build the router/gate to have an explicit "none of the above →
human queue" fallback rather than assuming the kind list is complete.

### 13g. Case B is CLUSTERED-by-source, not vanishingly rare (corrects §13b-REVISED-2)
A 6-image batch (Breakout/Reversal MFE framework, annotated Q1-Q4 hourly chart, 3x
SBS/wave models, ORB midline chart) mostly CONFIRMED existing types — first batch
this session to find NO new kind. Signal: taxonomy is stabilizing (user agrees).
Labels: img1 = reference_diagram/A (has probability-rule text panel); img2 & img6 =
annotated_chart/C (real market data + markup, confirms p205 rule — real data is C even
when text-rich); img3/4/5 (SBS Model #1/#2, fib-wave) = price_path/B.

CORRECTION to §13b-REVISED-2's "B is arguably MORE rare": WRONG framing. B is a
minority of total images but CLUSTERS by source — model-drawing educators (StoicTA
SBS, wave/fib diagrams, LRS-type schematics) produce MANY B images. User confirms
idealized model diagrams are a recurring genre from a few sources. CONSEQUENCE: the
sequence-handling path is NOT a rare-case afterthought — it lights up predictably for
known model-drawing sources. If provenance says "source X draws idealized models,"
expect B and extract the sequence; don't treat it as an anomaly.

SEQUENCE-FAITHFUL confirmed on a direct test: user says the SBS 1-2-3-4-5 wave order
IS the content. This is the CLEAN case B — same surface feature (numbered points on a
path) that is FABRICATION in case A (ICT_Month10 MMXM stages) is FAITHFUL here. What
separates them is exactly the 2-D rule: SBS = near-textless schematic (B, extract
sequence); MMXM = text-heavy framework (A, don't). The A/B distinction holds under
direct test.

OPEN (noted, not decided): SBS/wave models are NAMED patterns ("SBS Model #2",
5-wave). Case B may need BOTH a `sequence` AND a pattern-NAME field — and named
patterns (SBS, etc.) may deserve vocab concepts. Don't over-engineer; revisit at
schema time.

Calibration set now ~17 images. Still NO VLM consensus run on any of them (local
Ollama needed) — that remains the untested founding premise.

### 13h. Review-queue flagging tool BUILT (chart_review_queue.py)
User asked: don't solve named-model acquisition now — just FLAG low-confidence images
so the user can decide (1) is it needed? (2) is there a better source? Built a
standalone tool that reads the *_compare.json files and emits a scannable review queue
with TWO orthogonal, differently-actionable flags:
- **model_disagreement** (COMPUTABLE): the 3 VLMs read differently. Sub-signals:
  kind_disagree (gemma vs qwen — the honest pair — split on kind), seq_fabrication
  (n_steps 0/0/N — the minimax pattern), text_divergence (honest pair's text totals
  diverge). Action: verify/correct the read.
- **poor_source_candidate** (HEURISTIC): even if models AGREE, the image looks like a
  named/idealized MODEL (case B) better captured elsewhere (SBS-type). Proxy: both
  honest models say price_path + near-textless + a real sequence. Action: user decides
  needed?/alt-source? ORTHOGONAL to disagreement — the SBS case is where models AGREE
  yet the source is poor, so a disagreement-only filter would MISS it. That's why two
  flags.
KEY BUILD LESSONS (bugs found + fixed against the 7-chart ground truth):
- minimax pollution: first version let minimax's KNOWN text-inflation into both
  text_divergence and near_textless — which SUPPRESSED poor_source on LRS (the very
  chart it should catch) and re-flagged minimax's personality everywhere. FIX: compute
  both flags from the HONEST pair (gemma+qwen) only; minimax is a reader, never a
  vote or a text-count source. (Same §11a lesson, re-learned the hard way.)
- annotated_chart false positive: Arjo15m (real trade screenshot, case C) got
  poor_source-flagged because "has a short sequence + low text" isn't enough. FIX:
  case B requires both honest models to say `price_path` (the SCHEMATIC dimension of
  the §13b-REVISED-2 two-dimensional rule) — real-data annotated_charts are excluded.
- near_textless is a WEAK proxy: text-item COUNT can't separate "zone labels on an
  idealized curve" (LRS, case B, ~17 items) from "rules/conditions text" (RTH_ORG,
  case A, ~24 items) — no clean gap. The compare JSON lacks a labels-vs-rules signal.
  User chose CONSERVATIVE over-flagging (default --text-threshold=18 catches LRS,
  accepts some case-A false positives) because for a REVIEW-candidate flag a false
  positive costs a glance while a false negative lets a poor source enter the KB
  silently. Threshold is a documented, tunable knob — NOT a magic number; recalibrate
  on the full calibration set.
Result on the 7-chart set (matches ground truth): read-verification = BSL_DOL,
DailyPo3, ICT_Month10; poor_source = LRS; auto-OK = Arjo15m, RTH_ORG, ict_mmxm_notes.
Tool is threshold-parameterized (§11b discipline) and has an "unknown_kind → queue"
fallback for the open taxonomy (§13f). Files: chart_review_queue.py +
review_queue_sample.md in outputs.

---

## 14. Vocabulary seeded from published glossaries (56 → 82 entries)

Merged the glossaries from TWO authoritative ICT sources into ict_vocabulary.py:
LumiTrader "ICT 2022" book (comprehensive A–Z terminology page) and Flux "NY Session
Guide" (p66 glossary, §12d). Cross-referenced against the existing 56 entries; only
GENUINE gaps added. Net: 56 → 82 canonical concepts. **Superseded numerically by
§15 (82 → 172) but the decisions below still stand as the reasoning trail.**

### 14a. What was added
- **~15 genuine concepts**: volume_imbalance (VI), iof, iofed, hrlr, lrlr, smr, cbdr,
  adr, dot, seek_and_destroy, measuring_gap, redelivered_rebalanced (RDRB), amd, mmxm,
  mean_threshold, return_to_origin (RHO), risk_reward (RR). Placed in existing
  categories.
- **9 structural reference levels** under a NEW `reference_level` category: HOD/LOD,
  HOW/LOW, PDH/PDL, PWH/PWL, ITH/ITL, STH/STL, LTH/LTL, NY midnight open, opening
  price. Separated so coordinates-that-setups-target don't dilute concept grouping.

### 14b. User-confirmed dialect calls (were ambiguous, now settled)
- **IOFED, RDRB**: confirmed STANDARD ICT terms (not LumiTrader-specific). Flags dropped.
- **SMT**: ONE concept, not two. "SMT" (tool/technique) and "SMT divergence" (signal)
  fold into a single canonical `smt_divergence`. Both "Smart Money Technique" AND
  "Smart Money Tool" are live expansions → both kept as aliases.
- **Generic trading terms** (TF, TP, SL, TA, SH, TZ): deliberately EXCLUDED — too
  generic, not ICT concepts, and short forms over-match. **PARTIALLY REVISED in §15:
  TP and RR did end up added (as take_profit and risk_reward respectively) once the
  short-phrase-matching insight (§15a) made bare short forms safer than assumed.**
- **DOL (Draw on Liquidity)**: NOT added — already covered by existing `htf_draw`
  (alias "draw on liquidity"). Adding it would fragment one concept. **RESOLVED in
  §15: user confirmed DOL = htf_draw, exactly as this file's own reasoning predicted.
  Alias "dol" added to htf_draw.**
- **first_presented_fvg (FPFVG)**: [CHK] placeholder — RESOLVED in a later session
  (see the FPFVG Option-B note that was in an earlier vocab-review pass, now folded
  into ict_vocabulary.py's reference_level anchors: ny_midnight_open, opening_price,
  pm_session_open_1330, hourly_open).

### 14c. Over-match discipline (regression-tested, not assumed)
Short/2-letter acronyms that are common English substrings were made phrase-only or
boundary-safe, following the file's existing convention ("ob ", "ce "). A regression
+ decoy test suite CAUGHT real over-matches during the build and they were fixed:
- VI "vi" matched "vivid"/"servicing" → phrase-only now.
- mean_threshold "mt " matched "wasmt " → phrase-only now.
- RR made ratio/spaced forms only (bare "rr" matches "arrow"/"current").
Bare forms are fine only for distinctive acronyms (hrlr, lrlr, iofed, cbdr, rdrb, pdh).
**§15a insight applies here too: this over-match concern is about matching against
free TEXT. Against short extracted concept phrases the risk is much lower — several
of these "phrase-only" calls were revisited and loosened in §15.**

### 14d. KNOWN pre-existing issues surfaced (documented in file header, NOT yet fixed)
- **map_to_canonical returns ONE concept per raw string** (breaks after first match).
  A raw phrase naming two concepts loses one (e.g. "mmxm sell into premium" → only
  premium_discount). If multi-concept raw strings are common in the corpus, change the
  break to collect all matches. This affects how completely the corpus maps.
  **STILL OPEN — not addressed during §15's harmonization work. Worth checking
  whether the full-corpus run has meaningfully many multi-concept raw strings before
  prioritizing this.**
- **Pre-existing over-match**: "ce " (consequent_encroachment) matches "servi[ce] ".
  Left as-is (changing a load-bearing existing alias is a call about existing data);
  flagged in the file header. **STILL OPEN.**

### 14e. Still empirical, not "done"
This is a strong SEED, not a finished vocabulary. Completeness is confirmed by
report_unmapped on the FULL corpus (§10d step 1), which shows which of these 82 earn
their place and what dialect-specific terms no published glossary contains. The
LumiTrader S–Z glossary tail was partly unreadable via pdftotext (OCR garbling) and
was filled from a user paste; the authoritative source is the book's actual glossary
pages (~3–4 pp near the front) if true completeness is needed. **§15 is that
completeness pass — see there for the actual corpus-driven results.**

---

## 15. Full-corpus vocabulary harmonization + two production bugs fixed (this session)

With text extraction complete on the full 261-file corpus, ran `report_unmapped`
end-to-end for the first time and harmonized the vocabulary against real usage
across three rounds, then hit and fixed two real bugs while trying to run the
pipeline again. This is the §10d step 1/2 work, DONE.

### 15a. Vocabulary harmonization — 56 → 82 → 172 concepts, 79% → 37% unmapped
Three rounds of `report_unmapped` → triage → vocab edit → repeat, against the full
261-file / 8724-raw-concept corpus:

| round | unmapped | key finds |
|---|---|---|
| 1 | 79% → after fixes | EO bare (291x+61x+...) was the single biggest miss; session/delivery/order-flow/timeframe/profile/psychology/macro clusters had NO home at all |
| 2 | → 51% | missing profile_2/3/5 (existing file only had 1/4/6 — a real gap, not corpus noise); bare OB/SMT/breaker; DR mis-slotted into PD-array bucket (later corrected) |
| 3 | → 37% | mostly single generic words left (liquidity, structure, direction, entry...) — see below on why those stay unmapped |

**KEY METHODOLOGICAL INSIGHT (supersedes the old blanket "no bare acronyms" rule in
§5/§14c):** the vocabulary file's original over-match caution ("eo"→video,
"breaker"→circuit breaker) is about matching against free PROSE TEXT. But
`map_to_canonical` is actually called against `concepts_raw` entries — already
LLM-extracted, short, concept-only phrases — not raw transcript sentences. Against
that kind of input, bare short aliases are much safer: a phrase like "video" or
"circuit breaker" is not itself emitted as a raw concept in this corpus, so the old
fear doesn't apply. Re-verify this if a future extractor version starts emitting
longer, sentence-like concept strings instead of short phrases — this whole insight
would need re-examining.

**Confirmed real transcription-error clusters** (same pattern as the pre-existing
"feg"→fvg): FPG/FBG/FDG → fvg; CSB/CSV/CST → csd. User-confirmed, not guessed.

**Confirmed dialect-specific terms, now modeled:**
- BAG = Breakaway Gap → folded into `breakaway_gap`.
- "Business Card Model" and "Zoom Model" are specific NAMED trading models (their
  own author/methodology models, not generic ICT jargon) → new concepts
  `business_card_model`, `zoom_model`.
- "distortion" = consolidation in this dialect → folded into `trading_range`.
- DR = Dealing Range (NOT PD Array — moved out of the PDRA bucket where it was
  first mis-slotted) → folded into `premium_discount` (which already had "dealing
  range" as an alias).
- PDR/PDRs/PDRAs = PD Array (Premium/Discount Reference Array) → own concept
  `pdra_cluster`, now cleanly separated from DR.
- WIC = wick → folded into `candle_polarity`.
- "gap" bare = FVG in this corpus specifically → added to `fvg`.
- "4 a.m." = a specific time he references consistently (significance not yet
  known — news window? session boundary? personal macro?) → new concept
  `macro_4am`. Label is deliberately generic pending that context; rename once known.

**Kish's 7 Rules — a real numbered TCM framework, now fully modeled (`tcm_rule_1`
through `tcm_rule_7`):** what started as one throwaway unmapped phrase ("rule number
seven", 12x) turned out to be a citation into a specific, documented 7-rule
execution framework from Kish's TCM methodology (presentation slides/notes):
1. Determine Daily Order Flow before any setup (bullish/bearish bias first).
2. Identify Entry Formation Location (e.g. above/below the ONS range).
3. Entry Confluences by OB anatomy (FVG-in-OB → enter at inefficiency; large body →
   50%/mean threshold; small body → open+high (bearish) / open+low (bullish)).
4. The FVG Filter — continuation: low break + up-candle → next candle trades into
   an FVG above it → aggressive expansion lower.
5. The Timeframe Filter — London session → M15 structure; NY session → M5 structure.
6. Liquidity at Swing Points (stop placement) — distrust a swing high/low formed by
   a wick, especially one inside an FVG (expect it to be raided).
7. Order of Delivery ("kissing" fractal continuation) — SSL run → FVG tag →
   short-term low → CSD confirms a high-probability short. Selling an FVG
   immediately rebalanced right after an SSL run is explicitly called LOW
   probability — this is a real precision/false-positive filter, not just
   sequence-matching.
Each rule's concept aliases are the CITATION forms ("rule 3", "rule #3", "rule
number three"), not the descriptive content — the descriptive content already maps
to existing concepts (Rule 1 ≈ `order_flow`, Rule 7 invokes `state_of_delivery` /
`csd` / `liquidity_sweep`). A unit can and should end up tagged with BOTH the
specific rule number AND the underlying concept(s) it invokes — that's intentional,
not a duplication bug. **OPEN DESIGN QUESTION carried forward:** this concept-level
tagging only tells you "a numbered rule was cited," not which one is THE model for
a given setup in a structured, queryable way. If Rule 7's exact sequence (SSL →
FVG tag → short-term low → CSD) is central to his methodology, it may eventually
deserve to be a first-class `SetupPayload.sequence` TEMPLATE the extractor checks
for directly (a schema-level change), rather than four separately-tagged concepts.
Not built — flagged for whenever schema work is next on the table.

**Deliberately left unmapped (noise, not a gap):** `liquidity`, `structure`,
`direction`, `entry`, `setup`, `bullish`/`bearish` bare, `model`, `context`, `high`/
`low` bare, and similar single generic words. These are the umbrella nouns that the
now-172 specific concepts already carve up (liquidity → liquidity_sweep / htf_draw /
buy_side_sell_side / stop_hunt / reference levels, etc.) — force-mapping the bare
word to any ONE of those would blur precise concepts that already exist. Their raw
occurrence count is healthy background noise, not a missing concept.

**Still open, low-value, genuinely unclear:** `autoblock`, `OMS`, `WIC` (resolved —
see above), `form fit` (confirmed generic, left unmapped), `step function`/`four
steps` (possible named sub-model, not confirmed), `true price`, `volume balance`.
None were guessed at; revisit only if they recur at higher frequency in future
corpus growth.

### 15b. prompts.py bug fixed — concepts_raw junk-concept emission (closes §10c)
Root cause confirmed: `concepts_raw`'s instruction (in BOTH `CLASSIFY_PROMPT` and
`CLASSIFY_BATCH_PROMPT` — same weak one-liner duplicated in two places) gave the
model two positive examples and ZERO negative ones, with no test for what counts as
a "concept" at all. "Exactly as phrased" actually invited grabbing whatever
noun-ish text was nearby, producing junk like `concepts_raw: ["analyze 8 o'clock"]`
— an instruction, not a named concept.

FIX: introduced one shared `_CONCEPTS_RAW_SPEC` constant, used by both prompts so
they can't drift apart again, containing:
- a positive test ("would this make sense as a glossary entry?")
- explicit negative examples: instructions/actions, bare timestamps (unless the
  time itself is a NAMED macro/window, e.g. "9:12 macro"), full sentences, generic
  English words
- guidance to extract the concept EMBEDDED in an instruction rather than dropping
  the whole phrase (e.g. "wait for the CSD" → take "CSD")

Same underlying lesson as §13a: some model "quality" problems are actually
underspecified prompts, cheaper to fix upstream than to gate downstream.

**NOT YET VALIDATED against a real model run** (no model access in the session that
wrote the fix) — the fix addresses the specification gap the bug traces to, but
confirm on a real batch that junk concepts actually stop appearing before trusting
it fully. If they don't, the problem may be deeper than the prompt spec.

### 15c. schema/models.py bug fixed — ChartTextContent forward-reference ordering
Hit while re-running the full pipeline on Windows (fresh environment, pydantic
2.13): a deterministic `PydanticUndefinedAnnotation: name 'ChartTextContent' is not
defined` at import time, every run, no flakiness.

ROOT CAUSE: `ChartTextContent` (added per §10a) was defined at the very BOTTOM of
the file — after `JournalEntry`, inside what read like unmerged "patch instructions"
for a change that had actually already been half-applied (the fields were live in
`SetupPayload`'s body, but the referenced class itself was stranded below where it
was needed). `SetupPayload.model_rebuild()` runs much earlier in the file (right
after `KnowledgeUnit`), and since the file uses `from __future__ import
annotations`, all type hints are strings resolved LAZILY at that rebuild call. At
the point `model_rebuild()` executed, `ChartTextContent` simply didn't exist yet in
the module namespace.

FIX: moved the `ChartTextContent` class definition to directly before
`SetupPayload` (which references it) — i.e. before `model_rebuild()` runs. Removed
the stale trailing comment block describing the "changes to apply," since those
changes were already live in the class body and the leftover notes were actively
confusing, not just redundant.

Verified two ways (no pydantic available in the fixing session — no network access
to install it):
1. AST-level check: confirmed `ChartTextContent` and `SetupStep` both precede the
   `model_rebuild()` call in source order — the ordering fix that matters.
2. Structural diff: confirmed all 16 classes and every field are unchanged from
   what was pasted — only line order moved, nothing dropped or altered.

**Still needs a real run on the user's machine** to confirm pydantic accepts it
end-to-end — the AST check proves the ordering bug is fixed, not that nothing else
is wrong. If a NEW error surfaces on the next run, it's a different bug — don't
assume this fix was insufficient without seeing the new traceback first.

### 15d. What's actually next (supersedes §10d's remaining open items)
1. ~~**Confirm the prompts.py and models.py fixes work on a real run**~~ **DONE
   (this session, §16a)** — full corpus re-ran clean; the §15c schema fix holds.
2. ~~**Re-assess the low-confidence set**~~ **PARTIALLY DONE (§16a)** — code bugs
   blocking re-run are fixed; the confidence histogram script itself still to write.
3. **Decide on the NotebookLM gate** (§10d step 4) — only after step 2 shows a real
   problem once vocab isn't confounding it.
4. **Build the vector store** (`--build-vectors`) once 1-3 are settled.
5. **THEN move to §11-13** — the chart/image reconciliation gates are fully designed
   but NOT built, and the calibration set (~17 hand-labeled images) has never been
   run against the local Ollama models to test the founding premise (does cross-model
   consensus predict correctness?). That test should happen before writing the gate
   logic, not after.

---

## 16. Session 2026-07-20 — code hardening + corpus re-canon verified

This session's work on the TEXT path. Image/PDF work is the next major body —
see §17 for the plan.

### 16a. Code bugs fixed + locked with regression tests
Fixed concrete, unambiguous bugs. All guarded by `knowledge_ingest/tests/`
(8 tests, all pass; run with `python knowledge_ingest/tests/test_regressions.py -v`):
- **config.py**: removed duplicate `extractor_*` field definitions (F811 — second
  block silently overwrote the first).
- **ingest.py `_segment`**: was using `classifier_num_ctx` instead of
  `segmenter_num_ctx` — silently truncated whole-file segmentation.
- **report_unmapped.py**: removed shadowed `collect(units_dir)` stub (dead code).
- **schema/models.py**: added `model_validator` enforcing that the populated
  payload matches `metadata.knowledge_type` (prevents silent retrieval corruption).
- **ingest.py `_extract_batch`**: warns on hallucinated out-of-range idx.
- **vocab/ict_vocabulary.py `map_to_canonical`**: removed the `break` after first
  match — now collects ALL canonical ids per raw concept (closes §14d). A phrase
  like "mmxm sell into premium" now maps to BOTH `mmxm` AND `premium_discount`,
  not just one.

### 16b. Recanonicalize verified safe across the full corpus
Re-ran recanonicalize across all 5 units dirs (the multi-match fix needed
propagation). Verified with `tests/verify_recanonicalize.py` (new tool):
- **11,592 units** total across 3 active dirs (2024/2025 empty).
- **4,483 units changed** (gained ≥1 canonical id).
- **0 units lost any id** (no regression), **0 parse errors**.
- **81 distinct ids gained** new occurrences; top gains: `trading_range` +647,
  `fvg` +492, `rejection` +477, `consequent_encroachment` +447, `overnight_session`
  +433. All real ICT concepts — no junk (no `video`/`circuit_breaker`), confirming
  the §15a insight (matching against short concept phrases is safe).
- 1 file in 2023 dir has no `.bak` (harmless — content is valid).

### 16c. Open items carried forward
- **Low-confidence histogram script** (§15d step 2): still to write — walk units
  JSONL, histogram `extraction_confidence`, cross-tab against whether
  `concepts_canonical` is non-empty. Do this before deciding on the NotebookLM gate.
- **`map_to_canonical` over-match watch** (§14d / §15a): the multi-match change is
  safe against the CURRENT corpus (verified), but if a future extractor version
  emits longer sentence-like concept strings, re-verify — bare short aliases could
  start over-matching.
- **`ce ` over-match** (§14d): still open, pre-existing, left as-is.

---

## 17. The image/PDF plan (next major body of work)

Goal: get images and PDFs ingesting at usable accuracy. User explicitly wants this.
Text path is done and verified; image path is the bottleneck. Founding premise
(cross-model consensus predicts correctness) is UNTESTED — test before building.

**STATUS: Phase 0 done, Phase 1 in progress. Read §11-13 for the full design history.**

### 17a. Phase 0 — input triage — DONE (2026-07-20)
User provided `C:\ICT_Videos\Testing\` (7 standalone images + 8 PDFs, 600 pages
total). Triage via `tests/triage_pdf.py` (new, PyMuPDF-based, no VLM):

| PDF | pp | text | image | co-dep | empty | what it is |
|---|---|---|---|---|---|---|
| Delirium | 3 | 2 | 0 | 1 | 0 | mostly text |
| Flux_NY_Guide | 68 | 0 | 67 | 0 | 1 | full-page images, NO text layer |
| ICT_Bond_Trading_Notes | 6 | 6 | 0 | 0 | 0 | pure text |
| Lecture 1-5 | 24 | 0 | 23 | 0 | 1 | slide deck (full-page images) |
| MMXM | 33 | 6 | 11 | 16 | 0 | **genuinely mixed** — the stress case |
| PRE-MARKET PLAN | 26 | 6 | 13 | 6 | 1 | mixed, image-heavy |
| TimeTheoryNYAM | 5 | 5 | 0 | 0 | 0 | pure text |
| lumitrader book | 435 | 0 | 435 | 0 | 0 | full-page images (every page) |
| **TOTAL** | **600** | **4%** | **91%** | **4%** | **0.5%** | |

**KEY FINDINGS (correct the handover):**
- **§12c is WRONG about Flux.** The handover says Flux splits ~60% text-dominant /
  25% image / 15% mixed. Structural reality: 0% text layer, 98% full-page image.
  The §12c eyeball analysis was reading text BAKED INTO the rendered image, not a
  PDF text layer. Flux can't go through the text pipeline at all — every page needs
  the VLM. **This corrects §12c.**
- **Two sub-types of `image_dominant` surfaced** (user-approved split):
  - `image_dominant_no_text_layer` (Flux 68pp, Lumitrader 435pp, Lecture 24pp):
    `get_text()` returns 0 — VLM must read EVERYTHING.
  - `image_dominant_with_text_layer` (MMXM image pages): real text layer alongside
    the image — could route text to deepseek and image to VLM, OR send whole page
    to VLM. This is the actual §11d design question.
- **MMXM.pdf (33pp) + PRE-MARKET PLAN (26pp) are the genuine mixed-doc test cases.**
  Very little co_dependent overall (23 pages, 4%); the hard co-dependence-detection
  problem lives in just 2 PDFs in this sample.
- Heuristic bug found+fixed: `img_area>=0.85` → `image_dominant` regardless of
  `text_chars` (text is baked in, needs VLM not text pipeline); and `img_count==0`
  → `text_dominant` regardless of length (nothing to be co-dependent WITH).
- **572 pages rendered to PNG** for the bake-off (only image_dominant + co_dependent).
- Files: `tests/triage_pdf.py`, `tests/_show_summary.py`,
  `C:\ICT_Videos\Testing\_triage_summary.json`, `_triage_renders\`.

### 17b. Phase 1 — VLM consensus bake-off — IN PROGRESS (2026-07-20)
`examples/vlm_calibration_run.py` (new) running on 63 inputs × 3 VLMs
(gemma4:cloud, qwen3.5:cloud, minimax-m3:cloud) + OCR+VLM variant on Flux/Lumitrader
pages (per user decision: "test both, let data decide" — bake-off discipline applied
to the OCR question).

- Input set: `C:\ICT_Videos\Testing\bakeoff_inputs.jsonl` (7 labeled standalone +
  22 MMXM co-dep + 13 PRE-MARKET co-dep + 10 Flux sampled + 10 Lumitrader sampled +
  1 Delirium). Built by `tests/build_bakeoff_inputs.py`.
- Output: `bakeoff_results.jsonl` + `bakeoff_report.md` (the latter has the 5
  decision tables: (a) honest-pair consensus vs ground truth, (b) kind
  misclassification patterns, (c) text_content union vs single, (d) minimax seq
  fabrication on case-A, (e) OCR+VLM vs pure-VLM on no-text-layer pages).
- Resume supported (`--resume`); ~40s/model/image so full run ~60-75 min + OCR.

**EARLY SIGNAL (4/63 done — confirm at full run):**
- DailyPo3 (true=reference_diagram): ALL 3 models agree `reference_diagram`, seq=0 →
  perfect consensus, AND minimax did NOT fabricate a sequence here (contra the §11a
  pattern). Promising for the founding premise.
- Arjo15m (true=annotated_chart): all 3 agree `annotated_chart` ✓.
- ICT_Month10 (true=mixed, case A): gemma+qwen both say `mixed`, seq=0 (correct for
  case A — no fabrication).
- BSL_DOL (true=price_path, case B): honest pair DISAGREES — gemma=`price_path`,
  qwen=`annotated_chart`. If this pattern holds, the §11b gate's "disagree → human
  queue" branch fires more than assumed. Founding premise may need refinement
  (consensus not always available).

**VERDICT (55/63 results analyzed — `tests/_bakeoff_analysis.py`):**
- **(a) Founding premise: PARTIALLY HOLDS — 5/7 (71%) on labeled set.** Real
  accuracy ~6/7 (86%) if we discount the gemma timeout on RTH_ORG. The BSL_DOL
  disagreement (gemma=price_path, qwen=annotated_chart) is the real failure — a
  case-B image where honest pair splits. A 2-of-3 vote (including minimax, which
  agreed with gemma) would get it right, but §11a says don't use minimax as a
  classifier. **The §11b "disagree → human queue" branch will fire on real case-B
  images — this is the expected stress case, not a gate failure.**
- **(b) Honest-pair disagreement: 20% across all 55 results** (78% agreement, 20%
  disagree, 2% missing). For 100 images → ~20 to human-review. Workable.
- **(b) Minimax price_path bias: 18% here vs 71% in §11a's 7-chart batch.** The
  §11a bias was chart-specific, not universal. On broader PDF pages minimax favors
  `reference_diagram` (29/55) over `price_path` (10/55). Still shouldn't be a
  classifier vote, but the bias is less severe than §11a feared.
- **(c) Kind distributions:** gemma and qwen are close (both favor
  reference_diagram + mixed). Minimax is the outlier (rarely mixed, often
  annotated_chart/price_path). Confirms §11a "don't use minimax as classifier."
- **(e) OCR: glm-ocr returned 0 chars on 13/13 no-text-layer pages.** MinerU's
  PP-OCRv6 succeeded on all. MinerU is the right OCR tool, NOT glm-ocr. The
  OCR+VLM comparison is moot — MinerU wins by default.

**IMPLICATIONS:** Phase 3 (chart gate) is buildable as designed (§11b holds,
disagree→human handles the 20%). Phase 5 = call MinerU (confirmed). The two
tools complement: MinerU extracts text + layout; the 3-VLM gate classifies kind.

### 17c. Phase 2 — update chart prompt + schema to the grown taxonomy — DEFERRED
Per user decision 2026-07-20: WAIT for Phase 1 data. The kind list in
`chart_extract.py PROPOSE_PROMPT` and the schema `SetupPayload.kind` docstring both
depend on what the bake-off discovers (whether the 3 models already classify the
grown-taxonomy kinds correctly → minimal change, vs whether the prompt needs the
full open taxonomy + `unknown → human queue` fallback). Phase 2 is small and
independent but uninformative until Phase 1 reports.

### 17d. Phase 3 — build the chart reconciliation gate (§11b) — DEFERRED
Only after Phase 1 confirms the premise (or tells us how to redesign). Threshold-
parameterized, conservative for ~100 charts. Honest pair classifies kind; routed
model reads; sequence guard is asymmetric (drop fabricated, keep real);
text_content union+dedupe.

### 17e. Phase 4 — eval set (alongside, per user decision)
20-30 hand-written Q&A pairs with known-correct unit_ids from the 300 transcripts.
`tests/eval_retrieval.py` runs `search()` for each, reports recall@k / MRR.
User writes the Q&A (only they know right answers); I build the harness.

### 17f. Phase 5 — PDF mixed-page router (§11d/§12b, last)
Only after the chart gate works. Test the 3-way page-dominance classifier
(text-dominant / image-dominant / co-dependent) on real PDF pages. Charts and
PDF-image-pages collapse into ONE code path. **Phase 0 already gives us the
ground-truth page-dominance labels for this** (`_triage_summary.json`).

**CANDIDATE SHORTCUT — MinerU (opendatalab/MinerU, 75k stars):** a purpose-built
document parser with a VLM+OCR dual engine (109 languages, scanned/handwritten/
multi-column), three backends (`pipeline` no-hallucination CPU, `vlm-engine`
high-accuracy, `hybrid-engine` native-text-where-it-exists + VLM-where-it-doesn't).
The hybrid engine IS the §11d page-dominance router, already engineered and
benchmarked — directly addresses the Phase 0 finding that 91% of the Testing
sample is full-page-image-no-text-layer (Flux/Lumitrader/Lecture). Outputs
Markdown + JSON with layout reconstruction, reading order, header/footer
removal, cross-page table merging, table→HTML. MCP server + LangChain.
Runs locally/offline. **If MinerU's hybrid engine works on Flux/Lumitrader, Phase
5 collapses from "build a 3-way classifier" to "call MinerU."** The riskiest
unbuilt piece (per §11d/§12e) becomes a maintained external dependency.
MinerU does NOT replace: the kind taxonomy, the reconciliation gate (Phase 3),
the grounding discipline, the typed KnowledgeUnit schema, provenance, or the
vocab/recanonicalize machinery — those still run on MinerU's output.

**INSTALL STATE (2026-07-20):** MinerU 3.4.4 installed OK into isolated venv
`C:\Users\vinay\mineru_venv` (created with `py -3.12`, since MinerU needs
`>=3.10,<3.14` and the main venv is Python 3.14). CLI at
`C:\Users\vinay\mineru_venv\Scripts\mineru.exe`. Backends: `pipeline` (no-
hallucination CPU), `vlm-engine` (high-accuracy), `hybrid-engine` (native-text
+ VLM-where-needed — the §11d router). `--effort high` required for image/chart
analysis (medium disables it). Models (opendatalab/MinerU2.5-Pro-2605-1.2B)
download to `C:\Users\vinay\.cache\huggingface\hub`.

**SMOKE TEST — PASSED (2026-07-20, after user enabled Developer Mode):**
MinerU `hybrid-engine --effort high` on Flux_NY_Guide.pdf pages 0-2 SUCCEEDED.
PyMuPDF got 0 chars on these pages (full-page images, no text layer); MinerU
extracted real text. Key results:
- **Disclaimer block** (~500 words all-caps) — clean extraction. PyMuPDF: 0 chars.
- **Title** "NEW YORK SESSION GUIDE 2024" — correct.
- **Small data table** (`| X | Y |` 5 rows) — MinerU detected and structured it as
  Markdown. This is the §13f `data_table` primitive handled automatically.
- **Image-text separation**: `natural_image` ("Dark, grainy image with no visible
  text") vs `text_image` vs `line` — layout-aware.
- **Structured JSON** (`*_content_list.json`) is per-element: `type` (image |
  footer | text | chart), `sub_type` (text_image | natural_image | line),
  `bbox` per element, `page_idx` (provenance), images saved separately with
  stable content hashes. This is significantly more useful than raw Markdown —
  it's a per-element structured breakdown that maps cleanly to our KnowledgeUnit
  pipeline. The `chart` type with `sub_type: "line"` is exactly the routing signal
  we'd otherwise have to build.

**IMPLICATION — Phase 5 mostly collapses to "call MinerU."** The §11d 3-way
page-dominance router (the riskiest unbuilt piece per §11d/§12e) becomes MinerU's
job. MinerU does NOT replace: the kind taxonomy (Phase 3 gate), the grounding
discipline, the typed KnowledgeUnit schema, provenance, or the vocab/recanonicalize
machinery — those still run on MinerU's per-element output. But the page-routing
+ OCR + layout-structure problem is now a maintained external dependency instead
of custom code. Outputs at `C:\ICT_Videos\Testing\_mineru_out\`.

**STILL TO VERIFY:** (a) MinerU on a genuinely mixed doc (MMXM.pdf — DONE, see
below). (b) MinerU on Lumitrader (435pp full-page images — scale test, pending
CUDA). (c) Whether MinerU's `chart`/`sub_type` labels align with our
`price_path`/`reference_diagram`/etc. taxonomy (almost certainly NOT — MinerU's
labels are layout-type, not ICT-pedagogy-type; our Phase 3 gate still needed to
map MinerU's elements to our kind taxonomy).

**MMXM (mixed doc) — DONE (2026-07-20):** MinerU hybrid-engine on MMXM.pdf p0-3
SUCCEEDED. Cleanly separated text elements (with `text_level` for headings:
"MMXM", "MARKET MAKER MODEL", "LUMITRADERS") from image elements (with
`sub_type: "natural_image"` + auto-captions like "Symmetrical black-and-white
geometric pattern resembling a stylized eye or infinity symbol"). Each element
has `bbox` + `page_idx`. **This confirms MinerU handles the
`image_dominant_with_text_layer` sub-type** — the harder routing case where text
and image coexist on the same page.

**GPU ACCELERATION (2026-07-20):** CPU-only torch took ~22 min for 3 Flux pages
(unusable for 435pp Lumitrader). User has NVIDIA RTX 4060 (8GB VRAM, CUDA 13.1
driver). Reinstalling torch+torchvision with CUDA 12.8 support into the MinerU
venv (`pip install torch torchvision --index-url
https://download.pytorch.org/whl/cu128 --force-reinstall`, terminal
`4af8453b-422c-42d4-b085-05d6d263c765`). Expected ~10-50x speedup, making
Lumitrader (435pp) feasible in minutes instead of hours.

**BAKE-OFF OCR FINDING:** glm-ocr:cloud returned 0 chars on ALL 5 Flux pages
tested in the bake-off. MinerU's PP-OCRv6 OCR (in the hybrid-engine backend)
succeeded where glm-ocr failed. This means MinerU is the right OCR tool for
full-page-image PDFs, NOT the Ollama glm-ocr model we'd considered for the
OCR+VLM comparison. The bake-off's OCR+VLM comparison (§17b table e) is
therefore moot — MinerU wins by default on the no-text-layer case.

### 17g. Tools built this session (image/PDF work)
- `tests/triage_pdf.py` — PDF page-dominance structural triage (PyMuPDF, no VLM).
- `tests/build_bakeoff_inputs.py` — assembles the bake-off input manifest.
- `examples/vlm_calibration_run.py` — Phase 1 bake-off harness (3 VLMs + OCR
  variant, resume, writes results JSONL + report MD).
- `tests/verify_recanonicalize.py` — (text-path; verifies recanon was safe).
- `tests/_show_summary.py` — tiny helper to print the triage summary table.

### 17h. GROK REFERENCE OUTPUT — the target quality (2026-07-20)
User pasted the DailyPo3 and Arjo15m images into Grok (xAI) and got rich, ICT-aware
analysis. Key quotes from the DailyPo3 output:
- "This image is a trading strategy guide for Daily Po3 (Power of 3 — a common ICT
  /Smart Money Concept framework)."
- Correctly identified: bias determination (8:30 open vs midnight open), Asia+London
  session analysis, HTF PD Array Discount, SMT divergence, CSD (Change in State of
  Delivery), LTF Liquidity Raid, London Judas Swing, NY 8:30 news, Power of 3
  (Accumulation → Manipulation → Distribution).
- Structured the output by bullish/bearish columns (matching the image layout).
- Explained each concept in context, not just labeled it.

And from the Arjo15m output:
- Identified "Sharp Turn (ST) Entry Model" — a short-side execution framework.
- Correctly named: Context High/Low, Protected High (SL), FVG Confirming ST,
  OD Entry (Origin of Delivery), FLOD Entry (First Level of Delivery).
- Extracted trade metrics: R:R 2.0, SL 31.75pts, target 63.50pts.
- Explained the two entry mechanics (aggressive vs conservative).

**THIS IS THE TARGET QUALITY.** The Grok output demonstrates that:
1. A vision-capable LLM WITH ICT domain knowledge can extract rich, structured, 
   accurate interpretation from these images — not just labels.
2. The bottleneck is the PROMPT, not the model. Our generic PROPOSE_PROMPT never
   told the model about ICT concepts, the A/B/C rule, or what to look for.
3. The architecture should be: MinerU (text/layout) → ICT-aware LLM (interpretation
   using domain knowledge) → structured KnowledgeUnit. The LLM should LEAN INTO its
   ICT knowledge, not try to be domain-agnostic.
4. **"If not everything we are doing is useless probably"** — user's words. If we
   can't get this quality of output, the image ingestion isn't useful. The prompt
   iteration is the critical path.

**NEXT STEP:** Iterate on the ICT-aware prompt (§17h insight) to match Grok's quality.
Test against the 7 labeled images with gemma4. If the ICT-aware prompt gets close to
Grok's quality with a single cheap model, we don't need 3-model consensus — we need
one good prompt + one good model + MinerU for text. The prompt is the lever.

### 17i. ICT-AWARE PROMPT TEST — FAILED (over-corrected) + TEXT PIPELINE FLAW (2026-07-20)
Tested the ICT-aware prompt (§17h) on the 7 labeled images with gemma4 + MinerU OCR
grounding. Results: **3/7 (42%) — WORSE than generic (4/7, 57%).**

The failure: the ICT-aware prompt told the model "if there's substantial text →
reference_diagram, don't extract sequence." But `mixed` images also have substantial
text. The model over-corrected and called everything with text a `reference_diagram`:
- BSL_DOL: true=`price_path`, ICT-aware=`reference_diagram` (wrong — it's a near-textless
  schematic, but the prompt's text emphasis confused it)
- ict_mmxm_notes: true=`mixed`, generic=`mixed` (correct), ICT-aware=`reference_diagram`
  (wrong — text + image are co-dependent, not text-as-payload)
- RTH_ORG: true=`mixed`, generic=`mixed` (correct), ICT-aware=`reference_diagram` (wrong)

**LESSON:** Domain knowledge in the prompt is necessary but not sufficient. Too
prescriptive → over-correction. The prompt must distinguish:
- `reference_diagram` = text panels ARE the content, image is decoration
- `mixed` = text and image are CO-DEPENDENT — neither complete alone
- `price_path` = near-textless schematic, path IS the method

The first ICT-aware prompt conflated "has text" with "reference_diagram." The next
iteration must be more nuanced.

**BIGGER REALIZATION — the TEXT pipeline has the same flaw (user insight):**
User: "this might be a flaw in the entire process we are doing with the transcripts
also."

This is correct. The text pipeline's prompts (prompts.py) are domain-agnostic:
- CLASSIFY_PROMPT doesn't explain what setup/contextual/framework look like in ICT terms
- EXTRACT prompt doesn't explain what regime_precondition/bias_source/timing_gate/trigger
  MEAN in ICT methodology
- concepts_raw spec doesn't reference the 172-concept vocabulary
- No ICT framework context (Power of 3, MMXM, 7 Rules, sessions, macros)

This likely explains:
- The 36% below-0.6 confidence (§10c) — model doesn't understand what it's extracting
- The junk concepts ("analyze 8 o'clock") — model doesn't know "8 o'clock candle" is
  an ICT timing reference
- The 47% unmapped rate (before vocab harmonization) — model emits non-standard names
- Many held-back units might be low-confidence BECAUSE the model lacks ICT context

**NEXT SESSION PRIORITY:** Rewrite BOTH the text AND image prompts with ICT domain
knowledge — carefully, avoiding the over-correction from the first attempt. The Grok
output (§17h) is the gold standard. The text pipeline prompt rewrite may improve
the 300 already-ingested transcripts (re-extraction is expensive, but re-classify +
re-extract with an ICT-aware prompt could be worth it for the low-confidence subset).

---

## 18. Session — ICT-aware v4 prompt: classification-free, 7/7 accuracy, full production run

**Date:** session following §17i. This section supersedes §17i's failure and
documents the successful resolution.

### 18a. The breakthrough: drop the kind taxonomy entirely

The §17i failure taught us that ICT domain knowledge in the prompt causes
over-correction on the 4-way `kind` classification (reference_diagram / mixed /
price_path / annotated_chart). User challenged the premise: **"Do we need the
classification if we can interpret the context of the image correctly?"**

The answer was NO. The v4 prompt replaces the entire `kind` taxonomy with a
single binary judgment: **`path_is_method`** — does the drawn path/shape itself
encode the trading method, or is it just a visual reference while text carries
the method? This is the only distinction that matters for downstream extraction:

- `path_is_method=True` → the image IS the setup (schematic, annotated chart with
  markings that ARE the methodology — e.g., Arjo's ST entry model, BSL/DOL
  schematic). Extract the sequence from the drawn path.
- `path_is_method=False` → the image accompanies text (reference diagram, mixed
  text+image where text carries the method — e.g., DailyPo3, ICT Month 10).
  Sequence is 0 or absent.

The A/B/C ground-truth cases map cleanly:
- Case A (reference/text-dominant, seq=0): pim=False
- Case B (price path IS the method, seq≥1): pim=True
- Case C (mixed, seq≥1): pim=True (path contributes to method)

### 18b. v4 results — 7/7 (100%) on gemma4:31b-cloud

| Image | Educator | Framework | pim | seq | Correct? |
|---|---|---|---|---|---|
| Arjo15m | Arjo | SMR | True | 5 | ✓ |
| BSL_DOL | — | — | True | 6 | ✓ (v3 said mixed — fixed) |
| DailyPo3 | LumiTrader | Po3 | False | 0 | ✓ |
| ICT_Month10 | ICT | — | False | 0 | ✓ (v3 said reference_diagram — fixed) |
| ict_mmxm_notes | MMXM trader | MMXM | True | — | ✓ |
| LRS | — | SMR | True | 8 | ✓ |
| RTH_ORG | — | — | False | 0 | ✓ |

Both v3 failures (BSL_DOL, ICT_Month10) are fixed. Single model (gemma4) achieves
what the 3-model consensus couldn't. **Multi-model consensus is unnecessary when
the prompt is good enough.**

### 18c. Prompt architecture — modular and reusable

The prompt is now in `knowledge_ingest/sources/ict_chart_prompts.py`:

```
ICT_DOMAIN_KNOWLEDGE   ← swappable block (15 educators, dialect terms)
                       ← replace with non-ICT domain knowledge for other systems
_V4_EXTRACTION_LOGIC   ← system-agnostic, reusable across domains
ICT_PROPOSE_PROMPT_V4  ← assembled from both
PROMPTS                ← registry dict: {generic, v2, v3, v4}
```

**Design decision:** lean on the model's built-in ICT knowledge for general
concepts (FVG, OB, CSD, liquidity, etc. — the model already knows these). Embed
only **corpus-specific** knowledge: educator profiles and dialect terms that the
model would NOT know from training data. This keeps the prompt lean and avoids
over-prescription.

### 18d. 15 educators profiled in the prompt

ICT, LumiTrader (also publishes MMXM material), Flux, fx4living, MMxM trader,
Afyz, Trader Diego, Hydra, Dexter, TinyVizla, AMTrades, TTrades, Arjo, Kish,
StoicTA.

Each profile includes: known frameworks, terminology dialect, what their charts
typically look like. This lets the model attribute material correctly (e.g.,
MMXM pages → LumiTrader, not a separate "unknown" educator).

### 18e. Arjo terminology corrected

The model was producing wrong expansions (OD="Origin of Delivery", FLOD="First
Level of Delivery"). Correct terms embedded directly in the prompt:
- OD = Overlapping Defense (aggressive entry)
- FLOD = First Line of Defense (conservative entry)
- LLOD = Last Line of Defense
- ST = Sharp Turn

Vocabulary updated 172 → 176 concepts (added: sharp_turn, overlapping_defense,
first_line_of_defense, last_line_of_defense).

### 18f. JSON schema additions

- `path_is_method` (boolean) — replaces `kind` (string enum)
- `image_type` (lightweight tag, not a strict taxonomy) — optional context
- `entry_mechanics` (string) — captures entry model details (e.g., "OD Entry:
  aggressive, enters at Overlapping Defense level")
- `educator` and `framework` fields (were in v3, retained in v4)

### 18g. Full production run — IN PROGRESS

`knowledge_ingest/tests/run_v4_full.py` — processes all 818 inputs (7 standalone
images + 811 rendered PDF pages from `_triage_renders/`).

- Model: gemma4:31b-cloud
- Output: `C:\ICT_Videos\Testing\_v4_full_run\`
- Resume supported (`--resume` flag)
- Progress printed every 10 images with ETA
- Final summary includes educator/framework/path_is_method distributions
- ~100-175s per image → ~30-40h total ETA (first image was slowest at 175s)

### 18h. Multi-model finding (from test_multimodel_vlm.py)

Before v4's breakthrough, we tested multiple VLMs on the focused problem set:
- **kimi-k2.7-code:cloud** — best at case-B (price_path) detection
- **minimax-m3:cloud + kimi** — best at mixed detection
- But v4 with a single model (gemma4) solved both, making multi-model consensus
  unnecessary. The lever was the prompt, not the model roster.

### 18i. Key files

| File | Purpose |
|---|---|
| `knowledge_ingest/sources/ict_chart_prompts.py` | Central prompt registry (v4 + legacy) |
| `knowledge_ingest/tests/test_prompt_iteration.py` | Single-model test on 7 labeled images |
| `knowledge_ingest/tests/test_multimodel_vlm.py` | Multi-VLM comparison (standalone + PDF pages) |
| `knowledge_ingest/tests/run_v4_full.py` | Full production run (818 images) |
| `knowledge_ingest/vocab/ict_vocabulary.py` | 176 canonical ICT concepts + aliases |

### 18j. Next steps after the production run completes

1. **Analyze v4 full-run results** — educator/framework/pim distributions, spot-check
   extractions for quality, identify systematic failure modes.
2. **Compare to Grok reference (§17h)** — does v4 match the target quality?
3. **If quality is good → build the downstream pipeline**: v4 output → typed
   KnowledgeUnit → vocab mapping → vector store. The chart path becomes production.
4. **If quality is insufficient → iterate prompt v5** — the modular architecture
   makes this easy (swap ICT_DOMAIN_KNOWLEDGE, adjust _V4_EXTRACTION_LOGIC).
5. **Text pipeline prompt rewrite (§17i insight)** — apply the same ICT-aware
   approach to `prompts.py` for transcripts. May improve the 300 already-ingested
   transcripts (re-classify + re-extract the low-confidence subset).
6. **MinerU integration (Phase 5, §17f)** — MinerU handles page routing + OCR;
   v4 prompt handles interpretation. Together they form the full image/PDF pipeline.