# INPUTS — how to feed every source type

One rule underneath everything: **every source reduces to two primitives** —
*text to extract* or *image to interpret*. Each tool below routes a source into
one of those. Provenance (source_type, image_path, source_page, url) is captured
on every unit so nothing needs re-ingesting later.

Keep ICT sources separate from other domains (GEX/VP). Do one domain at a time,
then the report->grow-vocab->recanonicalize pass, so vocabularies don't mix.

See [DESIGN.md](../DESIGN.md) for the full architecture and [README.md](../README.md)
for command syntax.

---

## Text sources

**Transcripts (.txt), markdown (.md)** — directly:
```
python -m knowledge_ingest.run --input <dir> --output <out> --ict-aware
python -m knowledge_ingest.run --input <dir> --output <out> --source-type markdown --ict-aware
```

---

## PDFs — two paths depending on content

### Path A: Chart/image-heavy PDFs (Lumi book, Flux NY Guide, MMXM)

Use when the PDF is mostly chart screenshots, diagrams, or full-page images.
The PDF pages get rendered to PNG and processed through the v4 chart extraction
pipeline (VLM interprets each page).

```
# 1. Triage: structural analysis + render pages to PNG
python -m knowledge_ingest.tests.triage_pdf --pdfs <pdfs> --render
#    -> renders to _triage_renders/*.png

# 2. V4 chart extraction on rendered images (human-in-the-loop)
python -m knowledge_ingest.tests.run_v4_full --out-dir <out>
#    -> processes each PNG with gemma4 ICT-aware chart prompt

# 3. Convert to KnowledgeUnit JSONL
python -m knowledge_ingest.tests.convert_v4_to_units
#    -> _v4_units/v4_chart_units.jsonl
```

Triage routes text-dominant pages to the text pipeline for free; only image
pages need the VLM (expensive). This produced the 818 chart units already in
the unified LanceDB (Lumi 435pp, Flux 67pp, MMXM, Vinay_Models 119pp, etc.).

### Path B: Text-heavy PDFs (Trader Blue Print Series, You Tube notes)

Use when the PDF is mostly text/notes. MinerU extracts text + images, then
the text goes through the ICT-aware text pipeline.

```
# Single command: MinerU extract + ICT-aware text ingest
python -m knowledge_ingest.mineru_integration --pdf <path> --text-output <out>

# Batch: all PDFs in a directory
python -m knowledge_ingest.mineru_integration --pdf-dir <dir> --batch --text-output <out>

# Skip text ingest (only extract + stage images)
python -m knowledge_ingest.mineru_integration --pdf <path> --no-text
```

MinerU also extracts embedded images to `_mineru_images/` for optional v4 chart
processing later. These are NOT auto-extracted (v4 is human-in-the-loop).

**MinerU environment:** `C:\Users\vinay\mineru_venv` (Python 3.12.10, `mineru`
v3.4.4). The CLI is `mineru.exe`, not the old `magic_pdf`. See
[HANDOVER.md section 23d](../HANDOVER.md) for details.

### Choosing the right path

- PDF mostly chart screenshots/diagrams -> Path A (triage + render + v4)
- PDF mostly text/notes -> Path B (MinerU + text ingest)
- Mixed -> Path A handles both (triage routes text pages to text pipeline,
  image pages to v4)
- `pdf_extract.py` (poppler-utils) is broken and superseded by both paths above

---

## Chart/diagram images (standalone .png/.jpg)

Human-in-the-loop: VLM proposes structured setup, you review, then commit.
```
# 1. Propose (VLM generates JSON for each image)
python -m knowledge_ingest.sources.chart_extract propose --images <dir> --output <out>

# 2. Review: edit <out>/_review/*.json, fix sequence/relations, set "status":"approved"

# 3. Commit (validated units -> JSONL)
python -m knowledge_ingest.sources.chart_extract commit --output <out>
```

---

## Blogs

```
python -m knowledge_ingest.sources.blog_fetch --url-file urls.txt --out <blogout> --source <key>
python -m knowledge_ingest.run --input <blogout> --output <out> --source-type blog --ict-aware
```

`blog_fetch` downloads in-content images and flags JS-rendered charts it can't
download (those need manual save-as-PDF -> mixed-PDF path).

---

## After a batch (any source) — grow vocab & build store

```
# 1. Report unmapped concepts -> find vocab gaps
python -m knowledge_ingest.report_unmapped --units <dir1> <dir2> ... --vocab ict --min-count 5

# 2. Add worthwhile concepts to vocab/ict_vocabulary.py

# 3. Recanonicalize (cheap, idempotent — never re-extract)
python -m knowledge_ingest.recanonicalize --units <dir1> <dir2> ... --vocab ict

# 4. Merge into unified LanceDB
python -m knowledge_ingest.merge_knowledge_base --transcript-dir <dir1>\units <dir2>\units ... --db <db_path>
```

**Sequencing rules:** do ONE domain at a time so vocabularies don't mix; keep
all runs on the SAME vocabulary file; grow vocab once at the end from the full
corpus; re-canonicalize every time vocab changes; re-EXTRACTION is never
repeated (expensive), only re-canonicalization.