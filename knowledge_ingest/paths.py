"""Central path resolver for knowledge-base data locations.

Single source of truth for where the KB stores its produced artifacts (the
unified LanceDB, ingest units JSONL, mineru text outputs). All producer
scripts import from here instead of hardcoding paths.

ARCHITECTURE (decided 2026-07-23):
  The consumer repo (`tvDownloadOHLC`) OWNS all produced KB data. The producer
  repo (`video2pdf/knowledge_ingest`) is just the ingest tool. Raw inputs
  (transcripts, PDFs, chart renders) stay external — they can live anywhere —
  because the producer is a generic tool. Everything the tool PRODUCES lives
  under the consumer's `data/knowledge/` tree so the running services (narrative
  engine, KB API, future confluence engine) all read from one place.

Resolution: `KB_DATA_DIR` env var, defaulting to
  `C:\\Users\\vinay\\tvDownloadOHLC\\data\\knowledge`.
Overridable per-run so tests, alternate machines, or CI can point elsewhere
without editing code. The launch scripts set it once.

Layout under KB_DATA_DIR:
  <KB_DATA_DIR>/
    unified_knowledge.lancedb      # canonical vector store (4,168 units)
    units/                        # produced KnowledgeUnit JSONL ingest outputs
      tcm_2023/  tcm_2024/  ...
    ingest/                       # mineru text extracts (transient build artifacts)
      mineru_text_staged/  mineru_text_ingest/
"""

from __future__ import annotations

import os
from pathlib import Path

# Default consumer data root. The consumer repo owns produced data; the
# producer is a generic tool that should not know about the consumer, so this
# default is a *fallback* only — override via KB_DATA_DIR for any other setup.
_DEFAULT_CONSUMER_DATA = r"C:\Users\vinay\tvDownloadOHLC\data\knowledge"


def kb_data_dir() -> str:
    """Root of all produced KB data. Honors KB_DATA_DIR, else the consumer default."""
    return os.environ.get("KB_DATA_DIR", _DEFAULT_CONSUMER_DATA)


def unified_db_path() -> str:
    """Canonical unified LanceDB path."""
    return os.path.join(kb_data_dir(), "unified_knowledge.lancedb")


def units_dir() -> str:
    """Root for produced KnowledgeUnit JSONL ingest outputs (subdirs per source)."""
    return os.path.join(kb_data_dir(), "units")


def ingest_dir() -> str:
    """Root for transient ingest build artifacts (mineru text extracts, etc.)."""
    return os.path.join(kb_data_dir(), "ingest")


def mineru_text_staged_dir() -> str:
    """Where mineru_integration stages .txt files for ingest."""
    return os.path.join(ingest_dir(), "mineru_text_staged")


def mineru_text_ingest_dir() -> str:
    """Where mineru_integration writes ingest output (units/, classified/, ...)."""
    return os.path.join(ingest_dir(), "mineru_text_ingest")


def ensure_dirs() -> None:
    """Create the KB data dir tree if missing. Safe to call repeatedly."""
    for d in (kb_data_dir(), units_dir(), ingest_dir(),
             mineru_text_staged_dir(), mineru_text_ingest_dir()):
        Path(d).mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """One-line summary of resolved paths (for logging at startup)."""
    return (f"KB_DATA_DIR={kb_data_dir()} | "
            f"unified_db={unified_db_path()}")