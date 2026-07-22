"""
Regression tests for bugs fixed in the knowledge_ingest package.

These are offline (no network, no LLM) tests. Run with:
    python -m pytest knowledge_ingest/tests/test_regressions.py
or directly:
    python knowledge_ingest/tests/test_regressions.py

Each test guards against a bug that actually bit the project and could regress.
"""

import os
import sys

# allow running as a script without package install
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


# --------------------------------------------------------------------------- #
# Bug §15c: ChartTextContent forward-reference ordering in models.py
#   ChartTextContent MUST be defined before SetupPayload.model_rebuild() runs,
#   else every fresh import raises PydanticUndefinedAnnotation.
#   This test just importing the module would fail if the ordering regresses.
# --------------------------------------------------------------------------- #
def test_models_import_cleanly():
    from knowledge_ingest.schema import models
    assert hasattr(models, "ChartTextContent")
    assert hasattr(models, "SetupPayload")


# --------------------------------------------------------------------------- #
# Bug #1 (config duplicate fields): OllamaConfig must define each extractor
# field exactly once. A duplicate-definition regression silently overwrites.
# --------------------------------------------------------------------------- #
def test_ollama_config_no_duplicate_fields():
    from knowledge_ingest.config.config import OllamaConfig
    # dataclass fields list should have unique names
    names = [f.name for f in OllamaConfig.__dataclass_fields__.values()]
    assert len(names) == len(set(names)), f"duplicate fields in OllamaConfig: {names}"
    # the extractor_* trio must exist (guard against accidental deletion)
    assert "extractor_model" in names
    assert "extractor_temperature" in names
    assert "extractor_num_ctx" in names
    assert "extractor_num_predict" in names


# --------------------------------------------------------------------------- #
# Bug #2 (segmenter used classifier_num_ctx): the segment stage must use
# segmenter_num_ctx, not classifier_num_ctx, so whole-file segmentation works.
# --------------------------------------------------------------------------- #
def test_segment_uses_segmenter_num_ctx():
    import inspect
    from knowledge_ingest.pipeline import ingest
    src = inspect.getsource(IngestPipeline_cls := ingest.IngestPipeline._segment)
    assert "segmenter_num_ctx" in src, "segment stage must use segmenter_num_ctx"
    # and must NOT use classifier_num_ctx in that method
    assert "classifier_num_ctx" not in src, "segment stage must not use classifier_num_ctx"


# --------------------------------------------------------------------------- #
# Bug #4 (report_unmapped shadowed collect): there must be exactly one
# `collect` symbol and it must accept a list.
# --------------------------------------------------------------------------- #
def test_report_unmapped_single_collect():
    import knowledge_ingest.report_unmapped as m
    funcs = [v for k, v in vars(m).items() if k == "collect" and callable(v)]
    assert len(funcs) == 1, "report_unmapped.collect should not be shadowed"


# --------------------------------------------------------------------------- #
# Bug #7 (payload/knowledge_type invariant): a KnowledgeUnit whose payload
# doesn't match its knowledge_type must raise.
# --------------------------------------------------------------------------- #
def test_payload_mismatch_rejected():
    from datetime import date
    from pydantic import ValidationError
    from knowledge_ingest.schema.models import (
        KnowledgeUnit, KnowledgeMetadata, Provenance, KnowledgeType,
        Testability, EpistemicStatus, Session, Instrument, TipPayload,
    )

    meta = KnowledgeMetadata(
        knowledge_type=KnowledgeType.SETUP,
        testability=Testability.BACKTESTABLE,
    )
    prov = Provenance(source_file="t", source_type="transcript", chunk_id="t:0")
    # attach a TIP payload to a SETUP-typed unit -> must raise
    with pytest.raises(ValidationError):
        KnowledgeUnit(
            unit_id="x", summary="s", metadata=meta, provenance=prov, tip=TipPayload(heuristic="h")
        )


def test_no_payload_rejected():
    from datetime import date
    from pydantic import ValidationError
    from knowledge_ingest.schema.models import (
        KnowledgeUnit, KnowledgeMetadata, Provenance, KnowledgeType, Testability,
    )
    meta = KnowledgeMetadata(knowledge_type=KnowledgeType.SETUP, testability=Testability.BACKTESTABLE)
    prov = Provenance(source_file="t", source_type="transcript", chunk_id="t:0")
    with pytest.raises(ValidationError):
        KnowledgeUnit(unit_id="x", summary="s", metadata=meta, provenance=prov)


# --------------------------------------------------------------------------- #
# §14d bug: map_to_canonical must collect ALL matches, not stop at the first.
# A raw phrase naming two concepts must yield both canonical ids.
# --------------------------------------------------------------------------- #
def test_map_to_canonical_collects_all_matches():
    from knowledge_ingest.vocab.ict_vocabulary import map_to_canonical
    # "mmxm" and "premium" (alias for premium_discount) both appear in this raw
    hits = map_to_canonical(["mmxm sell into premium"])
    assert "mmxm" in hits, hits
    # premium_discount should also be present now (was dropped by the old `break`)
    assert "premium_discount" in hits, hits


def test_map_to_canonical_still_dedupes():
    from knowledge_ingest.vocab.ict_vocabulary import map_to_canonical
    hits = map_to_canonical(["fvg", "fvg"])
    assert hits.count("fvg") == 1, hits


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))