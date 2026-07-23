"""GEX (Gamma Exposure / options dealer-flow) profile — STUB.

This is a placeholder profile so the registry demonstrates multi-domain
combination (`--profile ict+gex`) without waiting for a full GEX educator
corpus to be ingested. It REUSES the generic prompt wording (no GEX-specific
vocabulary has been iterated on yet) but stamps domains=['gex'] so units
extracted under it are tagged for the Phase 5 confluence engine.

WHEN to make this a real profile: once you have a GEX / dealer-flow / 0DTE
options educator corpus to ingest, replace the prompt callables below with
GEX-domain-knowledge-embedded versions (mirror the ict_text_prompts.py
pattern). Until then this stub lets `--profile ict+gex` resolve and lets you
tag a run as GEX-flavored without a registry KeyError.

See DESIGN.md §9 Phase 2 and the confluence-engine roadmap (Phase 5).
"""

from .registry import Profile, register
from knowledge_ingest.pipeline.prompts import (
    CLASSIFY_SYSTEM,
    CLASSIFY_PROMPT,
    EXTRACT_SYSTEM,
    extract_prompt,
    extract_batch_prompt,
    classify_batch_prompt,
)


GEX_PROFILE = register(Profile(
    name="gex",
    domains=["gex"],
    # STUB wording: reuse generic prompts until a GEX corpus is ingested and
    # GEX-specific prompt text is written (mirror ict_text_prompts.py).
    classify_system=CLASSIFY_SYSTEM,
    classify_prompt=CLASSIFY_PROMPT,
    classify_batch_prompt_fn=classify_batch_prompt,
    extract_system=EXTRACT_SYSTEM,
    extract_prompt_fn=extract_prompt,
    extract_batch_prompt_fn=extract_batch_prompt,
    description=(
        "STUB. Gamma Exposure / dealer-flow / 0DTE options domain. Reuses "
        "generic prompt wording until a GEX educator corpus is ingested; "
        "stamps domains=['gex']. Useful for combining via `--profile ict+gex`."
    ),
))