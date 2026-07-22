# Knowledge Ingestion Pipeline

Turns trading-education transcripts (and later PDFs / your own design docs) into
**typed, grounded, filterable knowledge units** with full provenance — the
foundation for the concept → candidate → backtest → stat → reasoning loop.

This replaces the extraction core of the old `process_transcripts_ollama.py`
while keeping its proven skeleton (batch loop, resume, per-stage artifacts, retry).

## What changed vs. the old script

| Old | New |
|---|---|
| "Extract EVERYTHING" into free-text markdown | Structured JSON validated against a schema |
| One treatment for all content | Per-unit **classification** into 6 knowledge types, each with its own schema |
| No metadata | `session`, `instrument`, `testability`, `epistemic_status`, provenance on every unit |
| Char-count chunking (splits concepts) | Timestamp/topic **segmentation** → one concept per unit |
| One model for everything | **Tiered**: small model classifies/segments, strong model extracts |
| Open-ended prompt (hallucination-prone) | Grounded prompt: null-if-not-stated, `inferred_fields`, confidence score |
| Concept names free-form | Raw names kept **and** mapped to a controlled vocabulary (dedupe) |

The good parts of the old script were kept: resume with sanity-check, saving
every intermediate artifact, retry-with-backoff, timestamp preservation.

## Pipeline stages (large-context cloud via Ollama)

1. **Segment** (whole file, one call) — a large cloud context window means the segmenter
   sees the *entire* session at once, so boundary decisions improve: a concept introduced
   early and referenced later stays coherent. Splits only on truly enormous files.
2. **Classify** (batched, one call per file) — all segments classified together, returned
   as an ordered array; still one classification per unit.
3. **Extract** (batched PER TYPE) — same-type units share one call so the model sees the
   day's shared context (regime, bias, HTF draw) and nothing gets orphaned from context,
   while still emitting one grounded record per unit with its own confidence + inferred_fields.
   Large groups chunk by `extract_batch_size`. Skipped for low-value units (pure anecdotes).
4. **Assemble** — validate into `KnowledgeUnit` (pydantic), map concepts to controlled
   vocabulary, write JSONL + readable `.md`. Missing units from any batch fall back to
   per-unit calls, so nothing is silently dropped.
5. **Vectorize** (separate step) — load into LanceDB with metadata columns for
   filter-before-rank retrieval.

**Why batched-per-type, not per-unit or whole-file:** per-unit extraction has the best
isolation but can orphan a unit from context stated in a sibling segment (e.g. the ONS
precondition described three segments earlier). Whole-file mixes types and dilutes
attention. Batching *within a type* keeps per-unit grounding + audit signal while
letting shared session context complete each record.

## Cloud models via Ollama

Ollama exposes cloud models through the same `/api/generate` endpoint, so only the model
NAME and `num_ctx` change in `config/config.py`. Set `segmenter_model` / `classifier_model`
/ `extractor_model` to your cloud model name and raise `*_num_ctx` to the window you have.

## Install

```bash
pip install pydantic requests lancedb pyarrow
# Ollama running locally with the models named in config/config.py, e.g.:
ollama pull llama3.2:3b        # classifier / segmenter (cheap)
ollama pull qwen3:14b          # extractor (strong) — use your largest comfortable model
ollama pull nomic-embed-text   # embeddings
```

## Run

```bash
# 1) ingest transcripts -> units/*.jsonl + notes/*.md
python -m knowledge_ingest.run --input /path/to/transcripts --output /path/to/out

# for your own design docs / PDFs (converted to text), tag credibility:
python -m knowledge_ingest.run --input /path/to/own_docs --source-type own_doc

# 2) load units into LanceDB
python -m knowledge_ingest.run --build-vectors --output /path/to/out --db ./knowledge.lancedb
```

## Query example

```python
from knowledge_ingest.pipeline.vector_store import search

# "what setups apply to NY AM, testable, high confidence"
hits = search("liquidity sweep reversal entry",
              knowledge_type="setup", session="ny_am",
              testability="backtestable", min_confidence=0.6, k=8)
```

## Tuning

Everything lives in `config/config.py`: model names per tier, temperatures,
context sizes, segment sizes, which types to skip extracting, resume on/off.

## Where this plugs into the bigger system

- `units/*.jsonl` → **vector store** (semantic Q&A) **and** the **strategy-candidate
  registry** (filter `knowledge_type == "setup"`, `testability == "backtestable"`).
- `contextual` units with a `testable_claim` → feed the **backtest scaffolder**
  (validate against your 20-year data) → results populate the **stats registry**.
- Validated stats write back `linked_stat_ids` + flip `epistemic_status` on the unit,
  closing the loop and making the eventual execution layer auditable.

## Next steps to discuss

- **Controlled vocabulary** is seeded from 6 transcripts (27 concepts). It will grow;
  consider an "unmapped raw concepts" report after a full run to find gaps.
- **Segmentation quality** is the biggest quality lever — verify on ~5 files before
  the full 200-file run.
- **Eval set**: 20–30 known Q&A to verify retrieval before wiring into reasoning.
