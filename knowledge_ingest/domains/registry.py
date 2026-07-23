"""Profile registry + resolver.

A Profile is a frozen bundle of the three-stage prompt callables plus the
domain tag(s) it stamps onto extracted KnowledgeMetadata.domains.

Profiles are registered by a short name (e.g. "ict", "gex"). The `--profile`
CLI flag accepts a single name OR a "+"-joined combination ("ict+gex") which is
resolved by `resolve_profile` into a merged Profile: the first named profile's
prompt callables win (it is the "primary" domain for prompt wording), and the
domains list is the union of all named profiles' domains. This keeps prompt
text from one domain while tagging units with every applicable domain so the
confluence engine (Phase 5) can find cross-domain units.

Design rationale: prompt *wording* cannot be meaningfully concatenated across
domains (you'd get a contradictory wall of text), but the *domain tag* is purely
metadata and unions cleanly. So combination = "use ICT wording, tag both".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass(frozen=True)
class Profile:
    """One prompt bundle for a knowledge domain.

    Attributes:
        name: short registry key, e.g. "ict", "gex".
        domains: domain tag(s) stamped onto KnowledgeMetadata.domains for every
            unit extracted under this profile. e.g. ["ict"], ["gex"], or
            ["ict", "gex"] for a combined profile.
        classify_system: system prompt for the (single + batch) classify stage.
        classify_prompt: single-unit classify user prompt (has {unit}).
        classify_batch_prompt_fn: callable(numbered_units) -> batch classify prompt.
        extract_system: system prompt for the (single + batch) extract stage.
        extract_prompt_fn: callable(knowledge_type, unit_text) -> single extract prompt.
        extract_batch_prompt_fn: callable(knowledge_type, numbered_units) -> batch extract prompt.
        description: human-readable blurb for `--profile list`.
    """
    name: str
    domains: List[str]
    classify_system: str
    classify_prompt: str
    classify_batch_prompt_fn: Callable[..., str]
    extract_system: str
    extract_prompt_fn: Callable[..., str]
    extract_batch_prompt_fn: Callable[..., str]
    description: str = ""


# Populated by each domain module below. Order is registration order; dict
# preserves it on >=3.7.
REGISTRY: Dict[str, Profile] = {}


def register(profile: Profile) -> Profile:
    """Add a Profile to the global registry under profile.name. Idempotent."""
    REGISTRY[profile.name] = profile
    return profile


def get_profile(name: str) -> Profile:
    """Fetch a single registered profile. Raises KeyError with available names."""
    _ensure_registered()
    if name not in REGISTRY:
        avail = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(f"unknown profile '{name}'. Available: {avail}")
    return REGISTRY[name]


def resolve_profile(spec: str) -> Profile:
    """Resolve a --profile spec into one Profile.

    Accepts either a single name ("ict") or a "+"-joined combination
    ("ict+gex"). For combinations the FIRST named profile supplies the prompt
    callables (primary wording) and the domains list is the union of all named
    profiles, preserving order and de-duplicating.
    """
    _ensure_registered()
    names = [n.strip() for n in spec.split("+") if n.strip()]
    if not names:
        raise ValueError(f"empty profile spec '{spec}'")
    primary = get_profile(names[0])
    # union of domains, order-stable, de-duped
    merged_domains: List[str] = []
    for n in names:
        for d in get_profile(n).domains:
            if d not in merged_domains:
                merged_domains.append(d)
    if len(names) == 1:
        return primary
    # combined: same callables as primary, unioned domain tags
    return Profile(
        name=spec,
        domains=merged_domains,
        classify_system=primary.classify_system,
        classify_prompt=primary.classify_prompt,
        classify_batch_prompt_fn=primary.classify_batch_prompt_fn,
        extract_system=primary.extract_system,
        extract_prompt_fn=primary.extract_prompt_fn,
        extract_batch_prompt_fn=primary.extract_batch_prompt_fn,
        description=f"combined: {' + '.join(names)} (primary wording: {primary.name})",
    )


def list_profiles() -> List[Profile]:
    """Return all registered profiles (for `--profile list`)."""
    _ensure_registered()
    return list(REGISTRY.values())


def _ensure_registered() -> None:
    """Import the domain modules so they register themselves. Idempotent."""
    from . import ict_profile  # noqa: F401
    from . import gex_profile  # noqa: F401
    from . import generic_profile  # noqa: F401