"""Prompt profile registry.

A *profile* bundles the stage prompts (classify + extract, single + batch) with
the domain label(s) it covers. This replaces the old boolean `ict_aware` switch
with an extensible registry so multiple domains (ICT, GEX, volume profile, ...)
can be combined at runtime via `--profile ict+gex`.

See DESIGN.md §9 Phase 2 for the roadmap context.
"""

from .registry import (
    Profile,
    REGISTRY,
    get_profile,
    resolve_profile,
    list_profiles,
)

__all__ = [
    "Profile",
    "REGISTRY",
    "get_profile",
    "resolve_profile",
    "list_profiles",
]