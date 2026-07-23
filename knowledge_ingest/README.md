# Knowledge Ingestion Pipeline

Turns heterogeneous trading-education sources (video transcripts, PDFs, chart
images, blogs) into **typed, grounded, filterable knowledge units** with full
provenance — the foundation for the concept -> candidate -> backtest -> trade loop.

## Documents

| Document | Purpose | Audience |
|---|---|---|
| [DESIGN.md](DESIGN.md) | Architecture, goals, design decisions, roadmap | Anyone who needs to understand the whole system |
| [HANDOVER.md](HANDOVER.md) | Session-by-session state log (what's done, what's in progress) | Next session / next developer |
| **README.md** (this file) | How to run the pipeline | Operator (you, running ingest) |
| [sources/INPUTS.md](sources/INPUTS.md) | How to feed each source type | Operator (feeding new sources) |
| [vocab/VOCABULARY_REVIEW.md](vocab/VOCABULARY_REVIEW.md) | Vocabulary decisions and open items | Anyone editing the vocabulary |

Start with **DESIGN.md** for the big picture, then come here for commands.

---

## Quick start

```powershell
cd C:\Users\vinay\video2pdf
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
```

### Transcripts (text)

```powershell
python -m knowledge_ingest.run `
  --input "C:\ICT_Videos\TCM\2023\transcripts" `
  --output "C:\ICT_Videos\Testing\_text_ict_2023" `
  --source-type transcript --ict-aware --no-skip
```

### Text-heavy PDFs (MinerU -> text ingest)

```powershell
# Single command: MinerU extract + ICT-aware text ingest
python -m knowledge_ingest.mineru_integration `
  --pdf "C:\path\to\file.pdf" `
  --text-output "C:\ICT_Videos\Testing\_text_ict_pdf"

# Batch: all PDFs in a directory
python -m knowledge_ingest.mineru_integration `
  --pdf-dir "C:\path\to\pdfs" --batch

# Check MinerU installation
python -m knowledge_ingest.mineru_integration --check
```

### Chart/image-heavy PDFs (triage -> render -> v4 chart extraction)

```powershell
# 1. Triage: render PDF pages to PNG, route text pages to text pipeline
python -m knowledge_ingest.tests.triage_pdf `
  --pdfs "C:\path\to\pdfs" --render

# 2. V4 chart extraction on rendered images (human-in-the-loop)
python -m knowledge_ingest.tests.run_v4_full --out-dir "C:\ICT_Videos\Testing\_v4_run"

# 3. Convert v4 JSON -> KnowledgeUnit JSONL
python -m knowledge_ingest.tests.convert_v4_to_units
```

### Merge into unified LanceDB

```powershell
python -m knowledge_ingest.merge_knowledge_base `
  --transcript-dir "C:\ICT_Videos\Testing\_text_ict_ingest\units" `
                   "C:\ICT_Videos\Testing\_text_ict_2025\units" `
                   "C:\ICT_Videos\Testing\_text_ict_pdf_tcm\units" `
  --db "C:\ICT_Videos\Testing\unified_knowledge.lancedb"
```

### Query the knowledge base

```powershell
# CLI -- RAG with LLM synthesis
python -m knowledge_ingest.tests.ask_kb "What is the Sharp Turn entry model?"
python -m knowledge_ingest.tests.ask_kb "How does Kish use CSD?" --sources --k 5

# Interactive REPL
python -m knowledge_ingest.tests.query_kb

# HTTP API server
python -m knowledge_ingest.serve --port 8900
# Then: POST http://localhost:8900/ask {"question": "What is CSD?"}
```

### Python library

```python
from knowledge_ingest.pipeline.vector_store import search

# Semantic search with metadata filters
hits = search("liquidity sweep reversal entry",
              db_path=r"C:\ICT_Videos\Testing\unified_knowledge.lancedb",
              knowledge_type="setup", session="ny_am",
              testability="backtestable", min_confidence=0.6, k=8)
```

---

## Pipeline stages

1. **Segment** (whole file, one call) -- splits into topic-coherent units
2. **Classify** (batched, one call per file) -- each unit -> one of 6 knowledge types
3. **Extract** (batched per type) -- typed payload + concepts + confidence
4. **Assemble** -- validate into KnowledgeUnit (pydantic), map to vocab, write JSONL
5. **Vectorize** (separate step) -- load into LanceDB for semantic search

See [DESIGN.md section 3](DESIGN.md) for the full architecture.

## Knowledge types

| Type | Description | Flows to backtest? |
|---|---|---|
| `setup` | Mechanical, repeatable trade rule | yes |
| `contextual` | Calendar/regime anticipation (partly testable) | yes (if `testable_claim`) |
| `framework` | Analytical method / how-to | no (guides analysis) |
| `tip` | Heuristic rule-of-thumb | no |
| `psychology` | Mindset / discipline | no |
| `anecdote` | War story with embedded heuristic | no |

## Prompt profiles

The pipeline uses domain-specific prompt profiles so the model understands what
it's reading:

- `--ict-aware` -- ICT/Smart Money Concepts (176 concepts, 15 educators, 8+ frameworks)
- (no flag) -- generic domain-agnostic prompts

**Planned:** `--profile ict+gex` for composable multi-domain prompts. See
[DESIGN.md section 3](DESIGN.md).

## Tuning

Everything lives in `config/config.py`: model names per stage, temperatures,
context sizes, segment sizes, batch sizes, confidence thresholds.

## Dependencies

```bash
pip install pydantic requests lancedb pyarrow pandas
# Ollama running locally with models from config/config.py:
ollama pull deepseek-v4-flash:cloud   # segmenter/classifier/extractor
ollama pull gemma4:31b-cloud          # chart VLM
ollama pull nomic-embed-text          # embeddings
```

**MinerU** (for PDFs) lives in a separate venv: `C:\Users\vinay\mineru_venv`
(Python 3.12.10, `mineru` v3.4.4). See [DESIGN.md](DESIGN.md) and
[HANDOVER.md section 23d](HANDOVER.md) for details.

## Current knowledge base state

- **4,168 units** in `C:\ICT_Videos\Testing\unified_knowledge.lancedb`
- 818 chart units (v4, gemma4) + 3,350 text units (ICT-aware, deepseek-v4-flash)
- 176 canonical ICT concepts
- See [HANDOVER.md](HANDOVER.md) for the latest state