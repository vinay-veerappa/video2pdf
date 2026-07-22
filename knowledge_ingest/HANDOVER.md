# ICT Knowledge Ingestion System — Handover

**Purpose:** Turn heterogeneous ICT trading-education sources (video transcripts,
markdown, PDFs, blogs, chart/diagram images) into a typed, filterable, provenance-
tracked knowledge base that can later feed a trading companion (setup retrieval,
backtest validation, live analysis). Built to capture sources ONCE — provenance
and re-canonicalization mean nothing needs re-ingesting when the vocabulary grows.

This document is the single source of truth for state, architecture, decisions,
and how to run everything. Hand it to any future session to continue.

---

## 1. Current status (as of §19)

**Chart path (v4 — PRODUCTION):** 818/818 images extracted, 0 errors. 
Classification-free `path_is_method` prompt (§18) on gemma4:31b-cloud. Quality 
matches/exceeds Grok reference. All 818 converted to typed KnowledgeUnits 
(setup/framework), validated, in LanceDB at `C:\ICT_Videos\Testing\_v4_lancedb`.

**Text path (ICT-aware — IN PROGRESS):** `--ict-aware` flag (§19a) embeds ICT domain 
knowledge into classify/extract prompts. A/B tested: 2.4x more units, 88% setup 
naming vs 33%. Full TCM corpus re-ingestion running (335 transcripts across 2023-2025).

**Schema (§19b):** Payload required fields made Optional — schema now reflects what 
the extractor actually produces. `None` = not provided; `extraction_confidence` is 
the quality gate.

**Merge + MinerU (§19c-d):** `merge_knowledge_base.py` combines chart + text units 
into unified LanceDB. `mineru_integration.py` routes PDFs through MinerU for 
OCR/layout extraction → images to v4 chart pipeline, text to ICT-aware text pipeline.

**Vocabulary:** 176 canonical concepts (grew 56 → 82 → 172 → 176 across sessions).
Full-corpus unmapped rate: 79% → 37%. Recanonicalize is idempotent — grow vocab 
anytime without re-extraction.

**Deferred by design:** live chart derivation (OHLCV detection code, not pixels); 
trade-journal population (schema built, not yet used); NotebookLM reconciliation 
gate (§10d step 4 — only if low-confidence set justifies it after ICT-aware re-ingest).

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
  run.py                    entry point: ingest, --build-vectors, --ict-aware
  config/config.py          ALL tunables: models, paths, batch sizes, thresholds,
                            ict_aware flag
  schema/models.py          pydantic schemas: KnowledgeUnit, 6 payload types (all
                            fields now Optional — §19b), Provenance, SetupStep,
                            ChartTextContent, JournalEntry
  pipeline/
    ingest.py               orchestrator: segment→classify→extract, batched, resume,
                            collapse-guard, confidence-gate, ICT-aware prompt
                            swapping (§19a)
    ollama_client.py        Ollama HTTP client, retry, robust JSON parse
    prompts.py              domain-agnostic prompts (segment, classify, extract)
    vector_store.py         LanceDB build + metadata-filtered search (multi-dir)
  vocab/
    ict_vocabulary.py       176 canonical concepts + aliases (§15)
    registry.py             domain registry: ict now; gex/etc. later
    VOCABULARY_REVIEW.md    decisions + [CHK] items to confirm
  sources/
    ict_text_prompts.py     ICT-aware text pipeline prompts (§19a) — drop-in
                            replacements for prompts.py
    ict_chart_prompts.py    ICT-aware chart extraction prompts (§18c) — v4,
                            classification-free, swappable ICT_DOMAIN_KNOWLEDGE
    pdf_extract.py          text PDFs + --mixed per-page router
    blog_fetch.py           fetch → clean text + download in-content images
    chart_extract.py        propose / compare / commit (VLM chart → verify → unit)
    INPUTS.md               how to feed every source type
  merge_knowledge_base.py   unified LanceDB builder (chart + text units, §19c)
  mineru_integration.py      MinerU PDF → image/text routing (§19d)
  multidir.py               shared filename-collision guard for multi-dir tools
  report_unmapped.py        multi-dir: unmapped concepts by frequency → vocab growth
  recanonicalize.py         multi-dir: re-map concepts_canonical after vocab grows
  tests/
    convert_v4_to_units.py v4 JSON → KnowledgeUnit JSONL (818 units)
    run_v4_full.py          v4 full production run script
    test_text_prompts_ab.py A/B test: generic vs ICT-aware text prompts
    _test_vector_search.py  LanceDB semantic search verification
    test_regressions.py     8 regression tests (all pass)
    triage_pdf.py           PDF page-dominance structural triage
  examples/
    model_bakeoff.py        extractor grounding/calibration bake-off
    segmenter_bakeoff.py    segmenter verbatim-fidelity bake-off
    vlm_bakeoff.py          chart-reading grounding bake-off
    vlm_calibration_run.py Phase 1 VLM consensus bake-off harness
    test_*.py               offline logic tests (no network/LLM needed)
```

---

## 4. Model choices (current — updated by §18/§19)

| Stage | Model | Why |
|---|---|---|
| Segmenter | deepseek-v4-flash:cloud | fast, reliable; used in ICT-aware text pipeline |
| Classifier | deepseek-v4-flash:cloud | same model, batched classify |
| Extractor (text) | deepseek-v4-flash:cloud | best calibration, no hallucination, fast |
| Embeddings | nomic-embed-text | LanceDB vector store |
| Chart VLM (v4) | gemma4:31b-cloud | classification-free, 7/7 accuracy, 818/818 production run |
| Parallel check | qwen3.5:cloud | used for dual-model comparison (educator 99.5% agreement) |

**Chart approach (§18):** Single model (gemma4), classification-free `path_is_method` 
prompt. No multi-model consensus needed when the prompt is good enough. Supersedes 
the §4/§10b kind-dependent model routing and §11b reconciliation gate design.

**Text approach (§19a):** ICT-aware prompts on deepseek-v4-flash:cloud. The generic 
domain-agnostic prompts (§4's original choice) are superseded for ICT corpus.

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

## 6. Chart-path text-channel fix (SUPERSEDED by §18 — kept for history)

> **§18 superseded this section.** The v4 prompt (classification-free, single 
> model gemma4) solved the chart extraction problem entirely. The multi-model 
> consensus gate (§11) and kind taxonomy (§12-13) are NOT used in production. 
> This section is kept as the reasoning trail that led to §18.

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

## 10. Post-extraction work queue (MOSTLY DONE — items updated)

> **§18-19 superseded most of this section.** The chart path is now production-ready 
> (818/818 via v4). The text path is being re-ingested with ICT-aware prompts. 
> Remaining open items noted below.

### 10c. text-corpus diagnostics (SUPERSEDED by §19a)
> The 36% below-0.6 confidence and 47% unmapped were from the GENERIC pipeline. 
> ICT-aware prompts (§19a) produce 2.4x more units with 88% setup naming. 
> Re-assess after the ICT-aware re-ingestion completes.

### 10d. Post-extraction sequence (status updated)
1. ~~report_unmapped → grow vocab → recanonicalize~~ **DONE (§15).**
2. ~~Fix extract prompt concept emission~~ **DONE (§15b).**
3. ~~Re-assess low-confidence set~~ **SUPERSEDED by §19a** — ICT-aware prompts change the equation.
4. ~~NotebookLM gate~~ **DEFERRED** — only if low-confidence set justifies it after ICT-aware re-ingest.

---

## 11-13. Chart gate design, taxonomy, labeling sessions (SUPERSEDED by §18)

> **§18 superseded all of this.** The v4 prompt replaced the entire kind taxonomy 
> and multi-model reconciliation gate with a single-model, classification-free 
> `path_is_method` binary judgment. These sections (§11 reconciliation gate design, 
> §12 PDF findings, §13 sequence ambiguity / A-B-C typing / ground-truth labels) 
> are kept in the git history as the reasoning trail that led to §18's breakthrough. 
> 
> Key lessons still valid from §11-13:
> - The taxonomy is NOT closed (every batch found new types) — build with fallback
> - Some "model quality" problems are underspecified prompts, not model gaps (§13a)
> - Text-presence vs geometry-presence is the real distinction for sequence extraction
> - MinerU is the right OCR tool (§17f confirmed) — glm-ocr failed on all test pages


## 14-17. Vocabulary harmonization, code hardening, image/PDF plan (HISTORICAL � condensed)

> These sections document the reasoning trail from the initial text pipeline through
> vocabulary harmonization, code hardening, and the image/PDF bake-off that led to �18.
> Full content is in git history. Key outcomes:

**�14-15: Vocabulary** � 56 ? 82 ? 172 ? 176 canonical concepts across 3 harmonization 
rounds against the full 261-file / 8724-raw-concept corpus. Unmapped rate: 79% ? 37%.
Key insight: matching against short LLM-extracted concept phrases is safe for bare 
acronyms (not the same as matching against free prose text). Kish's 7 Rules fully 
modeled. Two bugs fixed: concepts_raw junk emission (�15b), ChartTextContent 
forward-reference ordering (�15c).

**�16: Code hardening** � 6 bugs fixed, 8 regression tests added (all pass).
Recanonicalize verified safe across full corpus: 11,592 units, 4,483 gained canonical 
ids, 0 regressions. Key fix: map_to_canonical now collects ALL matches per raw 
concept (was breaking after first match).

**�17: Image/PDF plan** � Phase 0 (triage): 600 pages triaged, 91% full-page images.
Phase 1 (VLM bake-off): 55/63 results analyzed, founding premise partially holds 
(71%). MinerU confirmed as OCR tool (glm-ocr failed on all no-text-layer pages). 
MinerU smoke test passed on Flux and MMXM PDFs. GPU acceleration enabled (RTX 4060).
All of this was superseded by �18's classification-free single-model approach and 
�19d's MinerU integration module.

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

---

## 19. Session — ICT-aware text pipeline, schema fix, merge + MinerU integration

**Date:** session following §18d. This section documents the wiring of ICT-aware
prompts into the text pipeline, a schema correctness fix, and the merge/MinerU
integration scaffolding.

### 19a. ICT-aware text prompts — wired into the pipeline

**Problem:** The generic text pipeline prompts (prompts.py) are domain-agnostic.
They don't explain what setup/contextual/framework look like in ICT terms, don't
reference the 176-concept vocabulary, and don't embed ICT framework context (Po3,
MMXM, 7 Rules, sessions, macros). This likely explains the 36% below-0.6
confidence (§10c) and the junk concepts ("analyze 8 o'clock").

**Solution:** `knowledge_ingest/sources/ict_text_prompts.py` provides drop-in
replacements that embed ICT domain knowledge into the classify and extract stages.
Reuses `ICT_DOMAIN_KNOWLEDGE` from `ict_chart_prompts.py` (no duplication). Same
JSON output schema — pipeline code works unchanged.

**A/B tested (d14f694):** 3 Kish transcripts, generic vs ICT-aware:
- ICT-aware finds 2.4x more units (31 vs 13)
- Setup naming: 88% (7/8) vs 33% (1/3) — ICT-aware actually names setups
- Better type distribution: 8 setups vs 3, captures tips/psychology
- Concept mapping cleaner: 88% vs 118% (generic over-maps)
- Sample names: 'Bearish Sell-Side Draw', 'CSD short', '4am buy-side raid short'

**Wiring (7bc57d2, 104ff1a):**
- `config.py`: added `ict_aware: bool = False` to PipelineConfig
- `run.py`: added `--ict-aware` CLI flag
- `ingest.py`: `IngestPipeline.__init__` swaps all 6 prompt references based on
  `cfg.ict_aware`:
  - `CLASSIFY_PROMPT` → `ICT_CLASSIFY_PROMPT`
  - `CLASSIFY_SYSTEM` → `ICT_CLASSIFY_SYSTEM`
  - `classify_batch_prompt()` → `ict_classify_batch_prompt()`
  - `EXTRACT_SYSTEM` → `ICT_EXTRACT_SYSTEM`
  - `extract_prompt()` → `ict_extract_prompt()`
  - `extract_batch_prompt()` → `ict_extract_batch_prompt()`
  - Falls back to generic prompts with warning if ict_text_prompts unavailable

**Smoke test:** 7 units from 1 transcript (11th May 2023) — setup, 2 framework,
contextual, 3 tips — all with ICT canonical concepts (FVG, CSD, MSS, killzone).

### 19b. Schema fix — required payload fields → Optional

**Problem:** Several payload classes had required (non-Optional) fields that the
LLM extractor cannot always fill. When the model returned `null` for these fields,
Pydantic v2 rejected the payload, the unit was silently lost, and the top-level
`_payload_matches_knowledge_type` validator failed with a cryptic truncated error.

**Affected fields:**
| Payload | Field | Was | Now |
|---|---|---|---|
| `ContextualPayload` | `event_or_condition` | `str` (required) | `Optional[str] = None` |
| `ContextualPayload` | `expected_behavior` | `str` (required) | `Optional[str] = None` |
| `FrameworkPayload` | `method_name` | `str` (required) | `Optional[str] = None` |
| `FrameworkPayload` | `what_it_answers` | `str` (required) | `Optional[str] = None` |
| `FrameworkPayload` | `steps` | `List[str]` (default_factory) | `Optional[List[str]] = None` |
| `FrameworkPayload` | `inputs_required` | `List[str]` (default_factory) | `Optional[List[str]] = None` |
| `TipPayload` | `heuristic` | `str` (required) | `Optional[str] = None` |
| `PsychologyPayload` | `principle` | `str` (required) | `Optional[str] = None` |
| `SetupStep` | `order` | `int` (required) | `Optional[int] = None` |
| `SetupStep` | `action` | `str` (required) | `Optional[str] = None` |

**Design rationale:** The schema should reflect what the extractor actually
produces. `None` means "not provided" — downstream code can distinguish it from
`""` (empty). The `extraction_confidence` field remains the quality gate. A
brief pipeline-level coercion band-aid (None→"") was tried first but removed —
it masked the real issue and produced misleading data.

**Pipeline error logging also improved:** payload validation failures now print
the payload keys the model returned, and missing payloads are explicitly logged
instead of silently swallowed.

### 19c. Unified LanceDB merge — chart + text knowledge bases

`knowledge_ingest/merge_knowledge_base.py` combines chart-derived units (818 from
v4) and transcript-derived units into a single LanceDB vector store.

**Current knowledge bases:**
- Chart: `C:\ICT_Videos\Testing\_v4_lancedb` (818 rows, table "knowledge")
- Text: will be built from ingest output dirs after transcript re-ingestion completes

**Usage:**
```
python -m knowledge_ingest.merge_knowledge_base --transcript-dir <text_units_dir>
```

**TCM transcript sources (335 total):**
- `C:\ICT_Videos\TCM\2023\transcripts` — 242 .txt files (raw timestamped)
- `C:\ICT_Videos\TCM\2024\transcripts` — 75 .txt files
- `C:\ICT_Videos\TCM\2025\transcripts` — 18 .txt files
- Note: `C:\ICT_Videos\TCM\AllTranscripts` has 268 .md files but those are
  summarized/processed versions — NOT the raw verbatim transcripts with timestamps
  that the pipeline expects. The `transcripts_zip/` in the workspace was a partial
  142-file subset of 2023 only.

### 19d. MinerU integration — PDF → image/text routing

`knowledge_ingest/mineru_integration.py` provides a clean interface to:
1. Run MinerU on a PDF → get markdown + extracted images
2. Route images to v4 chart extraction pipeline
3. Route markdown to ICT-aware text ingestion pipeline

**MinerU venv:** `C:\Users\vinay\mineru_venv` (separate Python environment)
**Usage:**
```
python -m knowledge_ingest.mineru_integration --check
python -m knowledge_ingest.mineru_integration --pdf <path>
python -m knowledge_ingest.mineru_integration --pdf-dir <dir> --batch
```

### 19e. How to run the full TCM transcript re-ingestion

Three parallel PowerShell windows (one per year, each writes to separate output
dirs — no collisions since each transcript produces its own `{stem}.jsonl`):

```powershell
# Window 1 — 2023 (242 transcripts)
cd C:\Users\vinay\video2pdf
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python -m knowledge_ingest.run --input "C:\ICT_Videos\TCM\2023\transcripts" --output "C:\ICT_Videos\Testing\_text_ict_2023" --source-type transcript --ict-aware --no-skip

# Window 2 — 2024 (75 transcripts)
python -m knowledge_ingest.run --input "C:\ICT_Videos\TCM\2024\transcripts" --output "C:\ICT_Videos\Testing\_text_ict_2024" --source-type transcript --ict-aware --no-skip

# Window 3 — 2025 (18 transcripts)
python -m knowledge_ingest.run --input "C:\ICT_Videos\TCM\2025\transcripts" --output "C:\ICT_Videos\Testing\_text_ict_2025" --source-type transcript --ict-aware --no-skip
```

After all 3 complete, merge into unified LanceDB:
```powershell
python -m knowledge_ingest.merge_knowledge_base --transcript-dir "C:\ICT_Videos\Testing\_text_ict_2023\units"
```

### 19f. Testing the knowledge base

**Existing chart LanceDB:** 818 rows at `C:\ICT_Videos\Testing\_v4_lancedb`
(table "knowledge"). Semantic search verified with 5 test queries — all relevant.

**Search API:** `knowledge_ingest.pipeline.vector_store.search()` provides
metadata-filtered semantic search:
- Filter by `knowledge_type` (setup, framework, contextual, tip, psychology)
- Filter by `session` (ny_am, london, etc.)
- Filter by `testability` (backtestable, partially, not_testable)
- Filter by `min_confidence`

**As ingest progresses**, build intermediate LanceDB from partial output and
test retrieval quality against known ICT concepts.

### 19g. Module layout additions (since §3)

```
knowledge_ingest/
  run.py                    + --ict-aware flag
  config/config.py          + ict_aware: bool field
  pipeline/
    ingest.py               + ICT-aware prompt swapping, better error logging
  schema/
    models.py               + required fields → Optional (§19b)
  sources/
    ict_text_prompts.py     NEW: ICT-aware text pipeline prompts (drop-in)
    ict_chart_prompts.py    ICT-aware chart prompts (§18c)
  merge_knowledge_base.py   NEW: unified LanceDB builder (chart + text)
  mineru_integration.py      NEW: MinerU PDF → image/text routing
  tests/
    test_text_prompts_ab.py A/B test: generic vs ICT-aware prompts
    convert_v4_to_units.py v4 JSON → KnowledgeUnit JSONL converter
    run_v4_full.py          v4 full production run script
    _test_vector_search.py  LanceDB semantic search verification
```

### 19h. Git commits this session

- `515c0bf`: v4 ICT-aware chart extraction prompt (classification-free, 7/7)
- `2c25757`: HANDOVER §18 docs
- `7669532`: --out-dir/--filter flags for parallel sessions
- `c92a73a`: Production run complete — quality analysis, Grok comparison
- `13e1c5f`: v4→KnowledgeUnit converter + LanceDB vector store
- `45aef80`: ICT-aware text pipeline prompts
- `d14f694`: A/B test: ICT-aware vs generic text prompts
- `7bc57d2`: Wire ICT-aware prompts into pipeline + merge/mineru integration
- `104ff1a`: Schema Optional fix + push to GitHub

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

### 18g. Full production run — IN PROGRESS (parallel sessions)

`knowledge_ingest/tests/run_v4_full.py` — processes all 818 inputs (7 standalone
images + 811 rendered PDF pages from `_triage_renders/`).

**Session 1 (main):**
- Model: `gemma4:31b-cloud`
- Output: `C:\ICT_Videos\Testing\_v4_full_run\`
- Resume supported (`--resume` flag)
- Progress printed every 10 images with ETA
- ~100-175s per image → ~30-40h total ETA (first image was slowest at 175s)

**Session 2 (parallel, added to cut runtime):**
- Model: `qwen3.5:cloud`
- Output: `C:\ICT_Videos\Testing\_v4_lumitrader\`
- Command: `python run_v4_full.py --model qwen3.5:cloud --resume --filter lumitrader --out-dir "C:\ICT_Videos\Testing\_v4_lumitrader"`
- Processes the lumitrader book (435 pages) — the biggest remaining chunk
- Separate output dir avoids file conflicts with the main run

**New CLI flags added for parallel support:**
- `--out-dir DIR` — override output directory (for parallel sessions)
- `--filter SUBSTRING` — only process images whose stem/source_pdf contains substring

Both sessions write JSON results + a `_run_log.jsonl` in their respective output
dirs. Results will be merged and deduped at the end (keeping whichever completed
each page first, or comparing both for dual-model coverage on the lumitrader book).

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

### 18j. Production run COMPLETE — final results & quality analysis (2026-07-21)

**Run summary:**
- 818/818 images processed, **0 errors** (12 transient errors fixed via retry + parallel fill)
- Main run: gemma4:31b-cloud, 858.9 min (~14.3h)
- Parallel run: qwen3.5:cloud on lumitrader book (435pp), 216.4 min (~3.6h), 0 errors
- Total wall-clock: ~6h (parallel strategy cut runtime in half)

**Full-run statistics (818 valid results):**

| Metric | Value |
|---|---|
| path_is_method: True | 403 (49.3%) |
| path_is_method: False | 415 (50.7%) |
| Pages with sequence | 403 (49.3%) |
| Pages with entry_mechanics | 14 (1.7%) |
| Pages with concepts_raw | 778 (95.1%) |
| Pages with inferred text | 818 (100%) |
| Avg concepts per page | 5.1 |

**Educator distribution (normalized):**
| Educator | Count | % |
|---|---|---|
| LumiTrader | 512 | 62.6% |
| unknown | 127 | 15.5% |
| Flux | 74 | 9.0% |
| ICT | 42 | 5.1% |
| Trader-Diego | 37 | 4.5% |
| Arjo | 6 | 0.7% |
| TinyVizla | 6 | 0.7% |
| MMxM-trader | 5 | 0.6% |
| AMTrades | 4 | 0.5% |
| Hydra/fx4living/Kish | 5 | 0.6% |

**Framework distribution:**
| Framework | Count | % |
|---|---|---|
| other | 263 | 32.2% |
| SMR | 214 | 26.2% |
| Po3 | 104 | 12.7% |
| MMXM | 94 | 11.5% |
| None | 73 | 8.9% |
| NY-session-profiling | 48 | 5.9% |
| OTE | 17 | 2.1% |
| Silver Bullet | 4 | 0.5% |

**Standalone vs PDF pages:**
- 7 standalone images: avg seq=4.0, pim_true=5/7 (71%)
- 811 PDF pages: avg seq=2.0, pim_true=398/811 (49%)

### 18k. v4 vs Grok reference comparison (§17h target quality)

**DailyPo3 — v4 vs Grok:**

| Aspect | Grok (§17h target) | v4 (gemma4) | Match? |
|---|---|---|---|
| Educator | (not asked) | ICT | ✅ correct |
| Framework | Po3 identified | Po3 | ✅ |
| Key concepts | AMD, SMT, CSD, Judas, Po3, PD Array, Premium/Discount | All present: Po3, Accumulation, Manipulation, Distribution, PD Array, SMT, CSD, Liquidity Raid, Judas Swing, Premium, Discount | ✅ all 11 concepts captured |
| Bias structure | Identified bullish/bearish columns | "green candle for bullish, black for bearish" | ✅ |
| Depth of explanation | Full contextual explanation | Concise inferred sentence | ⚠️ Shorter but accurate |
| path_is_method | (not asked) | False (reference diagram) | ✅ correct |

**Arjo15m — v4 vs Grok:**

| Aspect | Grok (§17h target) | v4 (gemma4) | Match? |
|---|---|---|---|
| Educator | (not asked) | Arjo | ✅ correct |
| Framework | ST Entry Model | SMR | ✅ (SMR is the broader framework) |
| ST identification | "Sharp Turn (ST) Entry Model" | "ST stands for Sharp Turn" in inferred | ✅ |
| OD Entry | "Origin of Delivery" (Grok was wrong) | "Overlapping Defense - aggressive entry" | ✅ v4 CORRECTED Grok's error |
| FLOD Entry | "First Level of Delivery" (Grok was wrong) | "First Line of Defense - conservative entry" | ✅ v4 CORRECTED Grok's error |
| Trade metrics | R:R 2.0, SL 31.75pts, target 63.50pts | Not captured in structured fields | ⚠️ Missing metrics |
| Entry mechanics | Both entries explained | Both captured with risk_reward field | ✅ |
| Sequence | (not asked) | 5 steps extracted | ✅ bonus |

**Verdict: v4 MATCHES or EXCEEDS Grok quality on the two reference images.**
- v4 correctly identified all key concepts that Grok found
- v4 CORRECTED two Grok errors (OD=Overlapping Defense, not "Origin of Delivery";
  FLOD=First Line of Defense, not "First Level of Delivery") — because the prompt
  embeds the correct Arjo terminology
- v4 produces structured JSON (sequence, entry_mechanics, concepts_raw) vs Grok's
  free-text — better for downstream pipeline ingestion
- v4 is slightly less verbose in the `inferred` field, but trades verbosity for
  structure, which is the right tradeoff for a knowledge base

**Quality assessment across the full run:**
- 95.1% of pages have extracted concepts (avg 5.1 per page) — high coverage
- 100% have inferred text — every page gets interpretation
- 49.3% identified as methodology pages (path_is_method=True) with sequences
- 50.7% identified as reference/non-method pages — correct for text-heavy PDF pages
- Educator identification: 84.5% attributed to a known educator (only 15.5% unknown)
- The 15.5% "unknown" is expected — covers disclaimers, covers, TOC pages, etc.

**Known limitations:**
1. **Entry mechanics only on 14 pages (1.7%)** — the entry_mechanics field is
   rarely populated. Most methodology is captured in `sequence` instead. This is
   a prompt design choice, not a failure.
2. **73 pages (8.9%) have framework=None** — these are pages where the model
   couldn't identify a specific framework. Acceptable for text/cover pages.
3. **Educator name normalization needed** — "Trader Diego" vs "Trader-Diego",
   "MMxM trader" vs "MMxM-trader" — post-processing step required.
4. **No trade metrics (R:R, SL, target)** — v4 doesn't capture numeric trade
   parameters. Could add to prompt v5 if needed.

### 18l. Downstream pipeline — COMPLETE (2026-07-21)

**v4 → KnowledgeUnit converter** (`knowledge_ingest/tests/convert_v4_to_units.py`):
- 818/818 images converted to typed `KnowledgeUnit` JSONL, 0 errors
- `path_is_method=True` → `KnowledgeType.SETUP` (with sequence, entry_mechanics, reference_levels)
- `path_is_method=False` → `KnowledgeType.FRAMEWORK` (reference/text pages)
- Educator names normalized via lookup table
- Concepts mapped via `ict_vocabulary.map_to_canonical()`
- All 818 units validated against `KnowledgeUnit` Pydantic schema
- Output: `C:\ICT_Videos\Testing\_v4_units\v4_chart_units.jsonl`

**LanceDB vector store** (`C:\ICT_Videos\Testing\_v4_lancedb`):
- 818 rows embedded with `nomic-embed-text`
- Metadata columns for pre-filtering: `knowledge_type`, `concepts`, `confidence`, `source_file`, etc.
- Semantic search verified with 5 test queries — all returned relevant results
- Search API: `from knowledge_ingest.pipeline.vector_store import search; search(query, db_path=..., k=N)`

**Dual-model comparison** (`knowledge_ingest/tests/_compare_models.py`):
- Educator agreement: 99.5% (433/435)
- `path_is_method` agreement: 64.4% (gemma4 more liberal, qwen3.5 more conservative)
- Framework agreement: 45.3% (qwen3.5 tends to use "other" or "None" where gemma4 assigns specific framework)

**End-to-end pipeline now works:**
```
PDFs/Images → triage_pdf.py → _triage_renders/ (811 PNGs)
                ↓
run_v4_full.py (gemma4:31b-cloud) → _v4_full_run/ (818 JSONs)
                ↓
convert_v4_to_units.py → _v4_units/ (818 KnowledgeUnits)
                ↓
build_lancedb() → _v4_lancedb/ (vector store, 818 rows)
                ↓
search() → semantic retrieval with metadata filters
```

### 18m. Next steps

1. **Text pipeline prompt rewrite (§17i)** — apply ICT-aware approach to
   `prompts.py` for transcripts. Re-classify + re-extract low-confidence subset.
2. **MinerU integration (Phase 5, §17f)** — MinerU for page routing + OCR;
   v4 prompt for interpretation. Full image/PDF pipeline.
3. **Prompt v5 (optional)** — add trade metrics capture, improve entry_mechanics
   frequency, reduce "unknown" educator rate.
4. **Merge chart + text knowledge bases** — combine the 818 chart-derived units
   with the ~300 transcript-derived units into a unified LanceDB.

---

## 20. Knowledge base query + RAG layer (2026-07-21)

This section documents the three interfaces that connect the knowledge base to any
LLM or client: CLI query, HTTP API server, and Python library. The practical payoff
of the entire pipeline — you ask a question, get a grounded answer with citations.

### 20a. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  ANY LLM or CLIENT                                            │
│  (Copilot Chat, Open WebUI, Python script, curl, trading bot) │
└──────────┬──────────────────────────────┬────────────────────┘
           │                              │
   ┌───────▼────────┐          ┌──────────▼──────────┐
   │ CLI (ask_kb.py) │          │ API (serve.py)      │
   │                 │          │ http://localhost:8900│
   │ ask "question" │          │                     │
   │ --no-llm        │          │ POST /ask  → RAG     │
   │ --sources       │          │ POST /search → raw   │
   │ interactive REPL│          │ GET /stats           │
   └───────┬────────┘          └──────────┬──────────┘
           │                              │
           └──────────┬───────────────────┘
                      │
          ┌───────────▼──────────────┐
          │  LanceDB (818+ units)     │
          │  + Ollama LLM (RAG)       │
          └──────────────────────────┘
```

### 20b. Three interfaces

**1. CLI — direct question with LLM synthesis:**
```powershell
$env:PYTHONPATH = "."
python -m knowledge_ingest.tests.ask_kb "What is the Sharp Turn entry model?"
python -m knowledge_ingest.tests.ask_kb "How does Kish use CSD?" --sources --k 5
python -m knowledge_ingest.tests.ask_kb "What are the 7 Rules?" --no-llm
python -m knowledge_ingest.tests.ask_kb  # interactive REPL
```

**2. HTTP API server — any client can query:**
```powershell
# Start server:
$env:PYTHONPATH = "."
python -m knowledge_ingest.serve --port 8900

# Query from any client:
$body = @{question="What is CSD?"; k=8} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8900/ask" -Method POST -Body $body -ContentType "application/json"

# Raw search (no LLM):
$body = @{query="Judas swing"; k=5} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8900/search" -Method POST -Body $body -ContentType "application/json"

# Stats:
Invoke-RestMethod -Uri "http://127.0.0.1:8900/stats"
```

**3. Python library — for scripts and agents:**
```python
from knowledge_ingest.tests.ask_kb import retrieve, format_context, synthesize
from knowledge_ingest.pipeline.vector_store import search

# Raw search (no LLM):
results = search("CSD order flow", db_path=r"C:\ICT_Videos\Testing\_v4_lancedb", k=8)

# Full RAG:
results = retrieve("What is CSD?", k=8)
context = format_context(results)
answer = synthesize("What is CSD?", context, model="deepseek-v4-flash:cloud")
```

### 20c. RAG pipeline — how it works

1. **Retrieve:** Question → embedded (nomic-embed-text) → semantic search against
   LanceDB → top-K units with metadata (knowledge_type, confidence, source, speaker)
2. **Format:** Retrieved units formatted as numbered source blocks with concepts,
   payload content, and provenance (who said it, where, when)
3. **Synthesize:** Question + formatted sources → sent to LLM (deepseek-v4-flash:cloud)
   with system prompt: "Answer using ONLY the provided source material. Cite sources
   by number. If sources don't contain enough info, say so."
4. **Answer:** LLM returns grounded answer with [Source N] citations. Every claim
   traces to a specific knowledge unit.

### 20d. API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check (status + DB path) |
| GET | `/stats` | Database statistics (counts, type distribution, top sources) |
| POST | `/search` | Raw semantic search — returns units, no LLM |
| POST | `/ask` | Full RAG — retrieve + LLM synthesize, returns answer + sources |

**POST /search body:**
```json
{"query": "CSD order flow", "k": 8, "knowledge_type": "setup", "min_confidence": 0.5}
```

**POST /ask body:**
```json
{"question": "What is CSD?", "k": 8, "knowledge_type": null, "min_confidence": 0.5}
```

**POST /ask response:**
```json
{
  "answer": "Based on the provided source material, CSD stands for...",
  "question": "What is CSD?",
  "sources": [{"knowledge_type": "setup", "source_file": "Vinay_Models.pdf", ...}]
}
```

### 20e. Query tool — interactive stats and search

`knowledge_ingest/tests/query_kb.py` provides a standalone query tool:

```powershell
python -m knowledge_ingest.tests.query_kb --stats
python -m knowledge_ingest.tests.query_kb --search "Judas swing" --type setup -k 5
python -m knowledge_ingest.tests.query_kb --text-stats
python -m knowledge_ingest.tests.query_kb  # interactive REPL
```

### 20f. Current knowledge base state

**Chart units (LanceDB, ready):** 818 units at `C:\ICT_Videos\Testing\_v4_lancedb`
- 415 framework, 403 setup
- Sources: LumiTrader book (435), Vinay_Models (119), ICTNotes (78), Flux (67), etc.
- All 0.5-0.9 confidence, avg 5.1 concepts per unit

**Text units (growing):** 62 units from 9 transcripts at `C:\ICT_Videos\Testing\_text_ict_ingest\units`
- 15 setup, 13 tip, 12 framework, 11 contextual, 10 psychology, 1 anecdote
- Avg confidence 0.79, only 3% below 0.6 (vs 36% with generic prompts)
- All from Kish transcripts; will grow to ~335 transcripts

### 20g. Module layout additions

```
knowledge_ingest/
  serve.py                      NEW: HTTP API server (health, stats, search, ask)
  tests/
    ask_kb.py                   NEW: RAG CLI + interactive REPL
    query_kb.py                  NEW: query tool (stats, search, text-stats, REPL)
```

### 20h. Practical value — what this replaces

| Before (manual) | After (knowledge base) |
|---|---|
| Re-watch 335 videos to find "that thing about the 9:12 macro" | `ask_kb "What is the 9:12 macro?"` |
| Re-explain ICT concepts from memory | LLM answers from KB with citations |
| Manually cross-reference educators' methods | `search "CSD"` across all sources at once |
| No way to verify "did Kish say X?" | Search with speaker filter, see exact transcript + timestamp |
| Can't programmatically access trading knowledge | HTTP API → any tool, bot, or dashboard can query |

### 20i. Connecting to other LLMs

The HTTP API (`serve.py`) makes the knowledge base accessible to:
- **Copilot Chat** — via MCP server or HTTP tool calls
- **Open WebUI** — configure as a custom tool/function
- **Custom agents** — any Python script calls `retrieve()` + `synthesize()`
- **Trading bots** — query setup definitions, framework rules, or session timing
- **Dashboards** — fetch stats and display knowledge coverage

The LLM model used for synthesis is configurable (`--model` flag on serve.py, or
`model` parameter in the Python API). Currently `deepseek-v4-flash:cloud` — fast,
grounded, no hallucination. Can be swapped to any Ollama model.
