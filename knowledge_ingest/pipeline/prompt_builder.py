"""Prompt profile resolver used by the ingest pipeline.

Single entry point: `resolve_active_profile(cfg)` returns the Profile the
pipeline should run under, derived from `cfg.profile` (the new field) with a
backward-compatibility fallback to the legacy `cfg.ict_aware` boolean.

Why a builder module instead of inline code in IngestPipeline:
  - keeps the registry import lazy (only imported when ingesting, not when
    e.g. building vectors) so vector-store-only runs don't pull ollama deps;
  - gives one place to evolve the resolution rules (defaults, deprecation,
    future per-source-type defaults) without touching pipeline plumbing.
"""

from __future__ import annotations

from ..config.config import PipelineConfig
from ..domains.registry import Profile, resolve_profile, get_profile, list_profiles

# The default profile for the whole system. ICT is the corpus the KB was
# built on, so it's the safest default — a generic default would silently
# downgrade every existing run.
DEFAULT_PROFILE = "ict"


def resolve_active_profile(cfg: PipelineConfig) -> Profile:
    """Return the Profile the pipeline should run under for this config.

    Resolution order (first wins):
      1. cfg.profile (the new `--profile` field) if set and non-empty.
      2. cfg.ict_aware legacy flag: True -> "ict", False -> "generic".
      3. DEFAULT_PROFILE ("ict").

    cfg.profile is authoritative once set; the legacy flag is honored only
    when profile was never assigned, preserving old behavior for callers that
    still flip `ict_aware` directly (e.g. mineru_integration default=True).
    """
    prof = getattr(cfg, "profile", None)
    if prof:
        return resolve_profile(prof)
    if getattr(cfg, "ict_aware", False):
        return resolve_profile("ict")
    return resolve_profile(DEFAULT_PROFILE)


def profile_help() -> str:
    """One-line summary of registered profiles for CLI --help epilog."""
    try:
        profs = list_profiles()
        return "; ".join(f"{p.name}={p.description.split('.')[0]}" for p in profs)
    except Exception:
        return "ict, generic, gex (+ to combine, e.g. ict+gex)"


def is_list_request(spec: str) -> bool:
    """True if the user asked to list profiles rather than run one."""
    return spec is not None and spec.lower() in ("list", "?", "help")