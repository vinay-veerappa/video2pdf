"""ICT / Smart Money Concepts profile.

Wraps the existing ICT-aware prompt text in `sources/ict_text_prompts.py` into a
Profile. This does NOT duplicate the prompt wording — it imports the constants
and callables already iterated on through §17i-§18m, so the registry is a
purely structural refactor with zero prompt-text drift.

Backward compatibility: the legacy `cfg.ict_aware = True` path maps to this
profile (see pipeline/prompt_builder.py).
"""

from .registry import Profile, register
from knowledge_ingest.sources.ict_text_prompts import (
    ICT_CLASSIFY_SYSTEM,
    ICT_CLASSIFY_PROMPT,
    ICT_EXTRACT_SYSTEM,
    ict_extract_prompt,
    ict_extract_batch_prompt,
    ict_classify_batch_prompt,
)


ICT_PROFILE = register(Profile(
    name="ict",
    domains=["ict"],
    classify_system=ICT_CLASSIFY_SYSTEM,
    classify_prompt=ICT_CLASSIFY_PROMPT,
    classify_batch_prompt_fn=ict_classify_batch_prompt,
    extract_system=ICT_EXTRACT_SYSTEM,
    extract_prompt_fn=ict_extract_prompt,
    extract_batch_prompt_fn=ict_extract_batch_prompt,
    description=(
        "ICT / Smart Money Concepts (SMC) methodology. Embeds corpus-specific "
        "educator vocabulary (ICT, Kish, LumiTrader, Flux, Arjo) and frameworks "
        "(Po3, MMXM, SMR, OTE, Silver Bullet, Judas, dealing ranges, ONS "
        "profiles, CSD/MSS, FVG/OB, PD Arrays, 7-Rule). The default profile."
    ),
))