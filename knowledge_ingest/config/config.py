"""
Central config. Everything tunable lives here so retuning for your hardware or
swapping models never touches call sites.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"

    # Cloud models are accessed through Ollama's same /api/generate endpoint,
    # so only the model NAME and context sizes change vs. local. With a large
    # cloud context window, segmentation sees whole files and extraction batches
    # many units per call.

    # --- Segmenter: sees the WHOLE file in one call --------------------- #
    # Chosen via examples/segmenter_bakeoff.py on VERBATIM FIDELITY + speed:
    #   gemma4:31b-cloud: 1.00 fidelity, 6.7s, well-formed -> CHOSEN.
    #   deepseek-v4-flash: also 1.00 fidelity but ~20s (3x slower) -> backup.
    #   gemma4:latest (small): ignored the array schema (emitted a flat
    #     {timestamp: text} map) + 47s -> rejected.
    #   qwen3.6:latest: empty response at 88s (reasoning-model behavior) -> rejected.
    # Segmentation is a copy-not-reason task, so gemma's confidence miscalibration
    # (which barred it from extraction) is irrelevant here.
    segmenter_model: str = "deepseek-v4-flash:cloud" #"gemma4:31b-cloud"
    segmenter_temperature: float = 0.0
    segmenter_num_ctx: int = 131072                # raise to your model's window

    # --- Classifier: batched, short bounded judgment -> small/fast, temp 0 -- #
    # gemma4:31b-cloud is the safe default. gemma4:latest (small local) is cheaper
    # but botched the SEGMENT schema in testing — classification is a simpler
    # output shape so it may be fine; verify before trusting it here.
    classifier_model: str = "deepseek-v4-flash:cloud" #"gemma4:31b-cloud"
    classifier_temperature: float = 0.0
    classifier_num_ctx: int = 65536

    # --- Extractor: the quality-deciding stage -> strongest REASONING model -- #
    # Chosen via examples/model_bakeoff.py on GROUNDING DISCIPLINE + calibration:
    #   deepseek-v4-flash: honest confidence (0.8 w/ populated inferred_fields when
    #     it guesses), fast (4-7s), no invented specifics -> PRIMARY.
    #   glm-5.2: strong second; best at crisp falsifiable testable_claims, 0.95 conf.
    #   gemma4:31b: clean output but reports 1.0 confidence even when dropping fields
    #     (miscalibrated) -> fallback only.
    #   qwen3.5:cloud: too slow (33-76s) + empty responses -> rejected.
    # OPTION for high-value setup/contextual units: run deepseek + glm and flag
    # disagreements for review (glm's stricter claim phrasing complements deepseek).
    extractor_model: str = "deepseek-v4-flash:cloud"
    extractor_temperature: float = 0.15
    extractor_num_ctx: int = 131072
    extractor_num_predict: int = 16384             # room for a batch of records

    # --- Embeddings: pull a dedicated embedder (none of the chat models fit) - #
    #   ollama pull nomic-embed-text   (lighter)   OR   ollama pull bge-m3 (stronger)
    embed_model: str = "nomic-embed-text"

    # cap units per extraction call so output JSON stays reliable
    extract_batch_size: int = 12

    request_timeout: int = 1200
    max_retries: int = 3
    retry_backoff_s: int = 5


@dataclass
class PipelineConfig:
    input_dir: str = r"C:\ICT_Videos\TCM\2023\transcripts"
    output_dir: str = r"C:\ICT_Videos\TCM\2023\ingest_output"

    # source metadata defaults (override per-batch)
    source_type: str = "transcript"        # transcript | pdf | own_doc
    source_credibility: str = "trusted_educator"
    default_speaker: str = "Kish"
    default_instrument: str = "NQ"

    # segmentation — with a large cloud window the whole file goes in one call.
    # This ceiling only triggers a split for truly enormous files.
    target_segment_chars: int = 3500       # aim ~1 concept per segment
    max_segment_chars: int = 60000         # effectively whole-file for transcripts
    segment_overlap_lines: int = 2

    # skip low-value extraction to save strong-model calls
    skip_extract_types: List[str] = field(default_factory=lambda: [])  # e.g. ["anecdote"]

    # resume
    skip_existing: bool = True

    # process only the first N files (after sorting); None = all.
    # useful for test runs before the full corpus. Combine with skip_existing=False
    # to force a clean re-run of the same N files while iterating on fixes.
    max_files: int = None

    # optional glob to pick WHICH files (e.g. "*Review*"); None = all matching txt/md.
    # more useful than max_files for testing a specific short+long contrast.
    file_filter: str = None

    # use ICT-aware prompts (embeds ICT domain knowledge into classify/extract stages)
    ict_aware: bool = False

    # prompt profile registry key (DESIGN.md §9 Phase 2). A single name ("ict",
    # "generic", "gex") or a "+"-joined combination ("ict+gex"). When set this
    # takes precedence over the legacy `ict_aware` boolean. Empty/None -> fall
    # back to the ict_aware flag, then to the "ict" default. See
    # pipeline/prompt_builder.resolve_active_profile.
    profile: str = ""

    # units below this extraction_confidence are logged to <stem>_lowconf.json but
    # NOT written to the vector-store JSONL (likely segmentation fragments).
    min_unit_confidence: float = 0.35

    # segmenter collapse guard: if a file > this many chars yields fewer than
    # (chars // resegment_chars_per_unit) segments, force a windowed re-segment.
    resegment_min_chars: int = 15000
    resegment_chars_per_unit: int = 8000
    resegment_forced_window: int = 8000

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
