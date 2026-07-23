"""Generic (domain-agnostic) profile.

Wraps the domain-neutral prompts in `pipeline/prompts.py` into a Profile. Use
when ingesting sources that don't belong to a known domain (e.g. a general
trading blog, an unknown educator) so the model isn't biased toward ICT
vocabulary it can't ground.

This is what every extraction used BEFORE the `--ict-aware` flag existed.
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


GENERIC_PROFILE = register(Profile(
    name="generic",
    domains=["generic"],
    classify_system=CLASSIFY_SYSTEM,
    classify_prompt=CLASSIFY_PROMPT,
    classify_batch_prompt_fn=classify_batch_prompt,
    extract_system=EXTRACT_SYSTEM,
    extract_prompt_fn=extract_prompt,
    extract_batch_prompt_fn=extract_batch_prompt,
    description=(
        "Domain-agnostic. Uses the neutral classify/extract prompts with no "
        "embedded domain vocabulary. For unknown-educator or general-trading "
        "sources. Stamps domains=['generic'] on extracted units."
    ),
))