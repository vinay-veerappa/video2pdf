# INPUTS — how to feed every source type

One rule underneath everything: **every source reduces to two primitives** —
*text to extract* or *image to interpret*. Each tool below routes a source into
one of those. Provenance (source_type, image_path, source_page, url) is captured
on every unit so nothing needs re-ingesting later.

Keep ICT sources separate from other domains (GEX/MenthorQ). Do one domain at a
time, then the report→grow-vocab→recanonicalize pass, so vocabularies don't mix.

## Text sources

**Transcripts (.txt), markdown (.md)** — directly:
```
python -m knowledge_ingest.run --input <dir> --output <out>          # transcripts
python -m knowledge_ingest.run --input <dir> --output <out> --source-type markdown
```

**Text PDFs** — extract first, then ingest:
```
python -m knowledge_ingest.sources.pdf_extract --in <pdfs> --out <text_out>
python -m knowledge_ingest.run --input <text_out> --output <out> --source-type pdf
```

## Mixed PDFs (text + charts) and Google Slides

Export Google Slides as **PDF** (File → Download → PDF). Then use `--mixed`:
```
python -m knowledge_ingest.sources.pdf_extract --in <pdfs> --out <text_out> --mixed
```
Per-page routing:
- text pages → one `.md` per doc (page-tagged) → ingest with `--source-type pdf`
- image-bearing pages → rasterized to `<text_out>/_chart_images/` → chart-extract

Then:
```
# text side
python -m knowledge_ingest.run --input <text_out> --output <out> --source-type pdf
# image side (the chart pages)
python -m knowledge_ingest.sources.chart_extract propose --images <text_out>/_chart_images --output <out>
# ...verify the _review JSONs, then:
python -m knowledge_ingest.sources.chart_extract commit --output <out>
```

## Standalone chart / diagram images (annotated setups like LRS, EURUSD)

These encode a METHOD in their structure. VLM proposes a structured setup; YOU
verify (it's not reliable enough unverified). No prose is produced — you have the
source image.
```
# 1) propose
python -m knowledge_ingest.sources.chart_extract propose --images <charts> --output <out>
# 2) edit each <out>/_review/*.json: fix sequence/relations, set "status":"approved"
# 3) commit approved ones into the units store
python -m knowledge_ingest.sources.chart_extract commit --output <out>
```
Verification is fast: open the image, check the ordered `sequence` and each step's
`position` (premium/discount) and `range_liquidity` (ERL/IRL). Fix, approve, commit.

## Blogs (MenthorQ etc.)

```
python -m knowledge_ingest.sources.blog_fetch --url-file urls.txt --out <blog_out> --source menthorq
python -m knowledge_ingest.run --input <blog_out> --output <out> --source-type blog
```
Blogs map to their source's vocab DOMAIN (MenthorQ → gex, not ict). Respect
robots.txt / terms; prefer RSS/API if available.

## Journal (your own trades) — SEPARATE store

A journaled trade chart uses the same vision machinery but writes the JournalEntry
schema, not the knowledge base:
```
python -m knowledge_ingest.sources.chart_extract propose --images <my_trades> --output <out> --target journal
# verify, then commit -> writes to <out>/journal/ (not units/)
python -m knowledge_ingest.sources.chart_extract commit --output <out>
```
The journal is your execution record (entry/exit/outcome/what-I-missed), feeds the
companion's "how am I actually trading" role — NOT concept knowledge.

## After ingesting a batch (any source)

1. `report_unmapped --units <all dirs> --vocab ict` — see unmapped concepts
2. grow `ict_vocabulary.py` (your preferred names canonical, variants as aliases)
3. `recanonicalize --units <all dirs> --vocab ict` — re-map everything
4. `run --build-vectors --units <all dirs>` — build the queryable store

## Not yet built (deferred by design)

- **Unannotated live-chart derivation** (derive FVG/CISD/MSS from raw candles) —
  the hard Aspect 3; better done from OHLCV data + your detection code than from
  pixels. Depends on a mature knowledge base first.
- **OCR for scanned PDFs / precise level extraction** — pair OCR with the VLM when
  exact numbers matter (VLMs misread digits). Add when a source needs it.
