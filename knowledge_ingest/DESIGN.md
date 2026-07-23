# Knowledge Ingest System — Architecture & Design

**Status:** Living document. Updated as the system evolves.
**Location:** `C:\Users\vinay\video2pdf\knowledge_ingest\DESIGN.md`

## Document map

| Document | Purpose | When to read |
|---|---|---|
| **DESIGN.md** (this file) | Architecture, goals, design decisions, roadmap | First — the big picture |
| [README.md](README.md) | How to run the pipeline (commands, quick start) | When you need to run ingest |
| [sources/INPUTS.md](sources/INPUTS.md) | How to feed each source type (routing guide) | When feeding new sources |
| [HANDOVER.md](HANDOVER.md) | Session-by-session state log (what's done, in progress) | Continuing from a prior session |
| [vocab/VOCABULARY_REVIEW.md](vocab/VOCABULARY_REVIEW.md) | Vocabulary decisions and open items | Editing the vocabulary |
| [tvDownloadOHLC CLAUDE.md](file:///c:/Users/vinay/tvDownloadOHLC/CLAUDE.md) | Consumer repo (detection code, backtest, trading) | Working in the consumer repo |

---

## 1. Goal

Build a typed, queryable knowledge base from heterogeneous trading-education
sources (video transcripts, PDFs, chart images, blogs) that can be used by
**Python scripts** and **LLMs** to:

1. **Generate backtestable trading strategies** from extracted setup definitions
2. **Derive confluences** across domains at runtime (ICT + GEX + volume profile)
3. **Automate trading** — either via standalone detection scripts or LLM-driven
   decision-making with grounded KB citations

The knowledge base is the **concept layer**. Confluences and strategies are
**derived at runtime** from the concept layer + live market data — NOT baked in
during ingestion. Educators teach concepts individually; the system combines them.

---

## 2. System Architecture

```
                    INGESTION (capture once)                         RUNTIME (daily)
                    
  SOURCES            FRONT STAGE         CORE PIPELINE                EXECUTION
    
  transcripts ──► segment ──► classify ──► extract                   ┌──────────────┐
  text PDFs    ──► MinerU   ────────────► (typed                    │ Confluence   │
  mixed PDFs   ──► MinerU   ─┬─ text ──► KnowledgeUnit              │ Engine       │
  chart images ──► triage    ─┴─ img  ──► v4 chart extract           │              │
  blogs        ──► fetch     ────────────►                          │ Live data +  │
                                                          ┌────────┤ KB query    │
  ┌─────────────────────────────────────────────┐        │        │ = confluence │
  │  UNITS STORE (JSONL)                         │        │        │ = trade plan │
  │  6 typed payloads (setup/framework/ctx/tip/  │        │        └──────────────┘
  │  psych/anecdote) + provenance + metadata     │        │              │
  └─────────────────────┬───────────────────────┘        │              ▼
                        │ recanonicalize                  │        ┌──────────────┐
                        ▼ (vocab grows)                   │        │ Backtest     │
  ┌─────────────────────────────────────────────┐        │        │ Loop         │
  │  LanceDB (vector store)                      │────────┘        │ candidate →  │
  │  semantic search + metadata filters          │                 │ test → stat → │
  │  4,153+ units (chart + transcript + PDF)     │                 │ validate     │
  └─────────────────────┬───────────────────────┘                 └──────────────┘
                        │                                               │
                        ▼                                               ▼
  ┌─────────────────────────────────────────────┐        ┌──────────────────────┐
  │  Strategy Candidate Registry (PLANNED)       │        │ Automated Trading    │
  │  KB setups → structured detection candidates│        │ (scripts or LLM)     │
  │  trigger/entry/invalidation → fn names + args│        └──────────────────────┘
  └─────────────────────────────────────────────┘
```

### The three layers

| Layer | Purpose | Status |
|---|---|---|
| **1. Ingestion** | Capture knowledge ONCE from any source → typed KnowledgeUnits | ✅ Production |
| **2. Knowledge Base** | Queryable store with semantic search + metadata filters | ✅ Production |
| **3. Strategy → Execution** | KB setups → backtestable candidates → confluence → trade | ❌ Not started |

---

## 3. Layer 1 — Ingestion

### Design principles

- **Capture once.** Provenance on every unit. Recanonicalize (not re-extract)
  when vocabulary grows.
- **Typed knowledge.** Every unit is one of 6 types (setup, contextual,
  framework, tip, psychology, anecdote). Only `setup` and testable
  `contextual` flow toward backtesting.
- **Grounding discipline.** Extractor fills only what's stated; inferences go
  in `inferred_fields`. `extraction_confidence` is the quality gate.
- **Two channels for images.** Text (labels, notes) + structure (price path,
  spatial relations) both captured.
- **Domain-aware prompts.** The pipeline uses domain-specific prompt profiles
  so the model understands what it's reading.

### Source types and routing

| Source type | Front stage | Pipeline | Output |
|---|---|---|---|
| `.txt` transcripts | (direct) | segment → classify → extract | KnowledgeUnit JSONL |
| `.md` markdown | (direct) | segment (prose mode) → classify → extract | KnowledgeUnit JSONL |
| `.pdf` text-heavy | MinerU extract → .txt | text pipeline | KnowledgeUnit JSONL + chart images |
| `.pdf` chart-heavy | triage → render pages to PNG | v4 chart extraction (VLM) | KnowledgeUnit JSONL |
| `.png/.jpg` charts | (direct) | v4 chart extraction (propose → verify → commit) | KnowledgeUnit JSONL |
| blogs | blog_fetch | text pipeline + image extraction | KnowledgeUnit JSONL |

### Prompt profiles (extensible)

Each domain provides its own prompt profile — domain knowledge, type-specific
examples, and concept vocabulary injected into the generic prompt template.

**Current profiles:**
- `ict` — ICT/Smart Money Concepts (176 concepts, 15 educators, 8+ frameworks)
- `generic` — domain-agnostic (fallback)

**Planned profiles:**
- `gex` — Gamma exposure / options flow
- `volume_profile` — Volume profile / market profile analysis
- `market_structure` — Pure market structure (S/R, trend, break/retest)
- `statistics` — Statistical/quantitative methods

**Composition (planned):**
`--profile ict+gex` uses the ICT prompt structure + appends GEX domain
knowledge. Confluence examples can be added to the composed profile.

**Registry pattern (mirrors `vocab/registry.py`):**
```
domains/registry.py     PROFILES = {"ict": "ict_profile", "gex": "gex_profile", ...}
domains/ict_profile.py  CONCEPTS, FRAMEWORKS, EDUCATORS, TYPE_EXAMPLES, prompts
domains/gex_profile.py  (same schema)
pipeline/prompt_builder.py  renders structured data → prompt context block
```

Adding a new domain = 1 file + 1 line in registry. No prompt template editing.

### Segmenter

Segmentation splits raw text into units before classification. Transcript mode
segments on timestamps; prose mode (PDF/blog/markdown) segments on topic
boundaries. The segmenter also needs domain awareness — domain-specific hints
for what constitutes a topic boundary in trading material.

### Models

| Stage | Model | Why |
|---|---|---|
| Segmenter | deepseek-v4-flash:cloud | fast, reliable, verbatim fidelity |
| Classifier | deepseek-v4-flash:cloud | batched classify, fast |
| Extractor | deepseek-v4-flash:cloud | best calibration, no hallucination |
| Embeddings | nomic-embed-text | LanceDB vector store |
| Chart VLM | gemma4:31b-cloud | classification-free, 7/7 accuracy |

---

## 4. Layer 2 — Knowledge Base

### Schema (KnowledgeUnit)

Every unit shares a common envelope:

```
KnowledgeUnit
├── unit_id: str (deterministic hash)
├── summary: str
├── verbatim_anchor: Optional[str] (exact quote from source)
├── metadata: KnowledgeMetadata
│   ├── knowledge_type: setup | contextual | framework | tip | psychology | anecdote
│   ├── testability: backtestable | partially | not_testable
│   ├── epistemic_status: unvalidated | validated | contradicted | mixed
│   ├── domains: List[str]           ← PLANNED (which domains this unit covers)
│   ├── session_applicability: List[Session]
│   ├── instrument_applicability: List[Instrument]
│   ├── concepts_raw: List[str]     (as the source phrased them)
│   ├── concepts_canonical: List[str] (mapped to controlled vocab)
│   ├── linked_stat_ids: List[str]  (filled by backtest loop)
│   ├── inferred_fields: List[str]  (field names the extractor guessed)
│   └── extraction_confidence: float (0-1, quality gate)
├── provenance: Provenance
│   ├── source_file, source_type, source_credibility
│   ├── session_date, speaker, timestamp_range
│   ├── image_path, source_page, source_url
│   └── extractor_model, extracted_at
└── payload: one of 6 typed payloads
    ├── SetupPayload (the key one for trading)
    ├── ContextualPayload
    ├── FrameworkPayload
    ├── TipPayload
    ├── PsychologyPayload
    └── AnecdotePayload
```

### SetupPayload — the bridge to executable strategies

This is the payload type that flows toward backtesting:

```
SetupPayload
├── name: "9:12 Macro CSD short"
├── regime_precondition: "ONS inefficient"          ← market state filter
├── bias_source: "HTF draw to 13,350"                ← directional bias
├── timing_gate: "9:12 macro"                        ← time window
├── trigger: "CSD after liquidity sweep"             ← event that arms entry
├── entry: "FVG edge"                                ← entry rule
├── invalidation: "M5 close above down-candle high"   ← what kills the trade
├── target_logic: "draw to opposite liquidity"       ← exit target
├── management: "partials at 1.5R"                   ← post-entry rules
├── stop_philosophy: "above swept liquidity"         ← stop placement
├── quality_notes: "aggressive variant, 2R"          ← conviction level
├── sequence: List[SetupStep]                         ← chart-derived ordered path
├── reference_levels: List[str]                      ← named price levels
└── text_content: ChartTextContent                   ← written labels from chart
```

**The gap:** these fields are free-text strings. They're perfect for LLM reasoning
and human review, but a Python script can't execute `"CSD after liquidity sweep"`.
This is the fundamental bridge that Layer 3 provides.

### Vector store (LanceDB)

- **DB:** `C:\ICT_Videos\Testing\unified_knowledge.lancedb`
- **Table:** `knowledge`
- **Embedding:** nomic-embed-text via Ollama
- **Filters:** knowledge_type, session, testability, min_confidence
- **Known gap:** `source_type` not surfaced as a top-level column (nested in
  `full_json`). Fix: add to `build_lancedb()` record dict.

### Vocabulary

176 canonical ICT concepts + aliases. Recanonicalize is idempotent — grow vocab
anytime without re-extraction. The vocab registry (`vocab/registry.py`) supports
multiple domains: `{"ict": "ict_vocabulary", "gex": "gex_vocabulary", ...}`.

**Alignment goal:** the prompt profile's CONCEPTS and the vocab module's CANONICAL
should be the same data (single source per concept, no drift).

### Query interfaces

| Interface | Entry point | Use case |
|---|---|---|
| Python library | `vector_store.search()` | Scripts, agents |
| CLI | `tests/ask_kb.py`, `tests/query_kb.py` | Interactive queries |
| HTTP API | `serve.py` (port 8900) | Any LLM, tool, or dashboard |
| KB bridge | `kb_bridge.py` | Connects to tvdownloadOHLC narrative engine |

---

## 5. Layer 3 — Strategy → Execution (PLANNED)

This is the layer that turns the KB from a reference into a trading system.

### 5.1 Strategy Candidate Registry

**Purpose:** Translate KB setups (free-text) into structured, executable
candidates that reference detection functions.

**Process:**
1. Query LanceDB for units with `knowledge_type=setup` and `testability=backtestable`
2. For each setup, an LLM maps the free-text fields to detection function names:
   ```
   Input:  trigger: "CSD after liquidity sweep"
   Output: {"fn": "csd_after_liq_sweep", "args": {"direction": "sell_side", "timeframe": "M1"}}
   ```
3. Store the candidate with a link back to the source unit (`source_unit_id`)

**Candidate schema:**
```python
StrategyCandidate
├── candidate_id: str (deterministic)
├── source_unit_id: str (links back to KB unit)
├── domains: List[str] (e.g. ["ict"], or ["ict", "gex"] for confluence)
├── trigger: DetectionSpec      # {"fn": "liquidity_sweep", "args": {...}}
├── entry: DetectionSpec        # {"fn": "csd_confirmation", "args": {...}}
├── invalidation: DetectionSpec
├── target: DetectionSpec
├── management: DetectionSpec
├── timing_gate: Optional[DetectionSpec]
├── regime_filter: Optional[DetectionSpec]
├── concepts: List[str]         # canonical concept IDs
├── epistemic_status: str       # unvalidated → validated/contradicted after backtest
├── backtest_results: Optional[BacktestResult]
└── parameters: dict            # tunable params (timeframe, lookback, thresholds)
```

**Detection function library (lives in tvdownloadOHLC):**
The `tvdownloadOHLC` repo already has ICT detection code (FVG, CSD, MSS, etc.
per the ICT engine spec). The candidate registry references these by name.
New detection functions are added there; the registry maps KB prose → function
names.

### 5.2 Confluence Engine

**Purpose:** At runtime (intraday or premarket), derive confluences across
domains from live market data + KB knowledge.

**How it works:**
1. **Live data input:** GEX levels, session timing, price structure, volume profile
2. **KB query:** Search for setups/concepts matching current conditions
3. **Candidate matching:** Which strategy candidates' trigger conditions are met?
4. **Confluence detection:** Cross-domain alignment:
   - ICT setup (CSD after BSL sweep) + GEX (negative gamma at price level) = confluence
   - Market structure (break/retest) + Volume profile (POC rejection) = confluence
5. **Output:** Trade plan with cited sources, confidence score, and risk parameters

**Two modes:**
- **Script mode:** Python code checks candidate conditions against bar data + GEX
- **LLM mode:** LLM reads KB units + live data, reasons about confluences, generates
  a narrative trade plan with citations

### 5.3 Backtest Validation Loop

**Purpose:** Test candidates against 20y OHLCV data, write results back to the KB.

```
StrategyCandidate (unvalidated)
    → run backtest (detection code + OHLCV data)
    → BacktestResult (win rate, RR, drawdown, conditions)
    → write back: candidate.epistemic_status = validated/contradicted
    → write back: source_unit.metadata.linked_stat_ids = [stat_id]
    → write back: source_unit.metadata.epistemic_status = validated/contradicted
```

This closes the loop: concept → candidate → backtest → validation → updated KB.
The KB becomes self-improving — validated concepts are marked, contradicted
ones are flagged, and the confluence engine can prefer validated setups.

### 5.4 Automated Trading

**Two paths:**
1. **Standalone scripts:** candidate detection conditions → NinjaTrader/TradingView
   alerts → manual or semi-automated execution
2. **LLM-driven:** LLM reads KB + live data → generates trade plan → executes via
   API (with audit trail: every decision cites KB units + live data)

**Safety:** Every trade decision traces to:
- A KB unit (who said this, when, with what confidence)
- A backtest result (does it actually work)
- Live data (what are current conditions)
- A confluence score (how many domains agree)

---

## 6. Current State

### What works (Layer 1 + 2)

| Component | Status | Details |
|---|---|---|
| Text ingest pipeline | ✅ | segment→classify→extract, batched, ICT-aware |
| Chart v4 extraction | ✅ | 818/818 production, gemma4 |
| MinerU PDF integration | ✅ | MinerU→text ingest (fixed this session) |
| Schema (6 typed payloads) | ✅ | grounding discipline, provenance |
| Vocabulary (176 concepts) | ✅ | recanonicalize idempotent |
| LanceDB (4,153 units) | ✅ | semantic search, metadata filters |
| RAG/API layer | ✅ | serve.py, ask_kb.py, query_kb.py |
| KB bridge | ✅ | kb_bridge.py to tvdownloadOHLC |

### What's missing (Layer 3 + enhancements)

| Component | Priority | Effort | Description |
|---|---|---|---|
| `source_type` in LanceDB | High | 5 min | Add column so queries can filter by origin |
| Prompt profile registry | High | 1-2 sessions | `--profile ict+gex` instead of `--ict-aware` bool |
| `domains` field in schema | High | 15 min | Tag each unit with which domain(s) it covers |
| Strategy candidate registry | **Critical** | 2-3 sessions | KB setups → structured executable candidates |
| Confluence engine | **Critical** | 2-3 sessions | Runtime cross-domain confluence detection |
| Backtest validation loop | High | 2-3 sessions | Candidate → backtest → write back to KB |
| Detection function catalog | High | ongoing | Map KB prose → function names (LLM-assisted) |
| Dedup across ingest runs | Medium | 1 hour | Content-hash dedup in merge |
| Image-heavy PDF auto-route | Medium | 30 min | Detect 0 text units + N images → suggest v4 |
| Backtest candidate export | Medium | 1-2 hours | Export backtestable setups as candidate list |
| serve.py default DB | Low | 5 min | Point to unified_knowledge.lancedb |

---

## 7. Design Decisions

### D1: Knowledge is domain-specific; confluences are runtime
Ingestion captures domain knowledge faithfully (ICT concepts get ICT prompts,
GEX concepts get GEX prompts). Confluences are NOT extracted during ingestion —
they're derived at runtime when live data + multiple KB domains are available.
Rationale: educators teach concepts individually; combining them requires
knowing current market context.

### D2: Free-text payloads + structured candidates (not one or the other)
KB units store free-text payloads (for LLM reasoning + human review). Strategy
candidates store structured detection specs (for script execution). The bridge
is LLM-assisted mapping (one-time batch job per setup). Both representations
coexist — the KB stays human-readable, the candidates are machine-executable.

### D3: Producer/consumer split
`video2pdf` is the producer (ingestion, KB, candidate generation). `tvdownloadOHLC`
is the consumer (detection code, backtest, live trading, narrative engine). They
communicate via the KB API (port 8900) or the candidate registry file. No data
duplication; `tvdownloadOHLC`'s `CLAUDE.md` points to the canonical HANDOVER.

### D4: Capture once, recanonicalize freely
Re-extraction is expensive (LLM calls). Re-canonicalization is cheap (string
matching). Grow the vocabulary anytime; recanonicalize is idempotent. Never
re-ingest to fix vocabulary mapping — only to fix extraction quality.

### D5: Grounding over completeness
The extractor fills only what's stated. Missing fields are `None`, not guessed.
Inferences go in `inferred_fields` so they can never be silently promoted to fact.
`extraction_confidence` is the quality gate. This keeps the execution layer
auditable — every trade decision traces to a grounded source.

### D6: Charts are human-in-the-loop
VLM chart extraction proposes structured setups; a human reviews and approves.
This is intentional — charts misread geometry/levels too often for fully
automated trust. The review step is the quality gate for chart-derived units.

---

## 8. File Layout (current)

```
knowledge_ingest/
  DESIGN.md                  THIS FILE — architecture & goals
  HANDOVER.md                Session-by-session state log (canonical)
  run.py                     Entry point: ingest, --build-vectors, --ict-aware
  config/config.py           ALL tunables: models, paths, batch sizes, thresholds
  schema/models.py           Pydantic schemas: KnowledgeUnit + 6 payload types
  pipeline/
    ingest.py                Orchestrator: segment→classify→extract, batched
    ollama_client.py         Ollama HTTP client, retry, JSON parse
    prompts.py               Generic (domain-agnostic) prompts
    vector_store.py          LanceDB build + metadata-filtered search
  sources/
    ict_text_prompts.py      ICT-aware text pipeline prompts (current "ict" profile)
    ict_chart_prompts.py     ICT-aware chart extraction prompts (v4)
    pdf_extract.py           Text PDF + --mixed (broken — no poppler; superseded)
    blog_fetch.py            Fetch → clean text + download images
    chart_extract.py         Propose/compare/commit (VLM chart → verify → unit)
  vocab/
    ict_vocabulary.py        176 canonical concepts + aliases
    registry.py              Domain registry: {"ict", "gex", "nqstats", ...}
  merge_knowledge_base.py    Unified LanceDB builder (chart + text + PDF units)
  mineru_integration.py      MinerU PDF → image/text routing (fixed)
  multidir.py                Filename-collision guard for multi-dir tools
  report_unmapped.py         Multi-dir: unmapped concepts → vocab growth
  recanonicalize.py          Multi-dir: re-map concepts after vocab grows
  kb_bridge.py               Connects KB to tvdownloadOHLC narrative engine
  serve.py                   HTTP API server (port 8900)
  tests/
    ask_kb.py                RAG CLI + interactive REPL
    query_kb.py              Query tool (stats, search, text-stats, REPL)
    triage_pdf.py            PDF page-dominance structural triage
    run_v4_full.py           v4 full production run (818 images)
    convert_v4_to_units.py   v4 JSON → KnowledgeUnit JSONL converter
    test_regressions.py      8 regression tests
    ...                      (bake-offs, comparisons, A/B tests)
```

### Planned additions

```
knowledge_ingest/
  domains/                   NEW: prompt profile registry (§3)
    registry.py              PROFILES = {"ict": ..., "gex": ...}
    ict_profile.py           Structured ICT domain knowledge
    gex_profile.py           (future)
  pipeline/
    prompt_builder.py        NEW: renders structured data → prompt context
  strategy_candidates.py     NEW: KB setups → executable candidates (§5.1)
  confluence_engine.py       NEW: runtime cross-domain confluence (§5.2)
  backtest_loop.py           NEW: candidate → backtest → write back (§5.3)
```

---

## 9. Roadmap

### Phase 1: Foundation fixes (this session)
- [ ] `source_type` column in LanceDB
- [ ] `domains` field in KnowledgeMetadata
- [ ] serve.py default → unified_knowledge.lancedb
- [ ] Re-merge LanceDB with new PDF units

### Phase 2: Prompt profile registry
- [x] `domains/registry.py` + `domains/ict_profile.py`
      (+ `domains/generic_profile.py`, `domains/gex_profile.py` stub,
       `pipeline/prompt_builder.py`)
- [x] `pipeline/prompt_builder.py` — `resolve_active_profile(cfg)`;
      `--profile ict+gex` combines domains (primary wording + unioned tags)
- [x] Wire into ingest.py / run.py / mineru_integration.py
      (`--profile` flag; `--ict-aware` kept as deprecated alias; back-compat
      via `resolve_active_profile` honoring the legacy boolean)
- [ ] A/B test: structured prompts vs current prose prompts (verify equal-or-better)
      — registry is the foundation; the A/B harness is deferred until a second
      domain corpus exists (gex stub registered but no GEX educator ingested yet)

### Phase 3: Strategy candidate registry
- [ ] `strategy_candidates.py` — LLM-assisted candidate generation
- [ ] Detection function catalog (map KB prose → function names)
- [ ] Export backtestable setups as candidate JSON
- [ ] Link candidates back to source units (bidirectional)

### Phase 4: Backtest validation loop
- [ ] `backtest_loop.py` — candidate → backtest → results
- [ ] Write back: candidate.epistemic_status, unit.metadata.linked_stat_ids
- [ ] Integration with tvdownloadOHLC's prop firm simulator (ADR-021)

### Phase 5: Confluence engine
- [ ] `confluence_engine.py` — runtime cross-domain confluence
- [ ] Script mode: candidate conditions + live data → confluence score
- [ ] LLM mode: KB + live data → narrative trade plan with citations
- [ ] Integration with tvdownloadOHLC narrative engine

### Phase 6: Automated trading
- [ ] Trade execution via API (NinjaTrader or broker API)
- [ ] Audit trail: every decision → KB unit + backtest result + live data
- [ ] Semi-automated (alerts + manual confirm) → fully automated (with guardrails)

---

## 10. Key Constraints (from tvdownloadOHLC ADRs)

These constraints from the consumer repo apply to any strategy generated from
the KB:

- **ADR-017 (Zero-Loop):** Detection code must be vectorized NumPy/Pandas. No
  `for` loops in calculation paths.
- **ADR-020 (Prop Firm Liquidation):** Intraday positions must exit by 16:00 ET.
- **ADR-021 (Unified Simulator):** Use only `PropFirmSimulator` for prop firm
  viability evaluation.
- **ADR-001 (Timezone):** Charts take UTC; calculations use ET sessions; storage
  uses UTC epoch.
- **ADR-002 (Normalization):** Performance metrics as price percentage, not
  absolute points.