"""
Vocabulary domain registry.

You are building SEVERAL knowledge bases, each with its own vocabulary:
ICT/market-structure, GEX/gamma, volume profile, expected move, etc. Rather than
one flat vocabulary that serves none well, each domain is its own module exposing
the same interface:

    CANONICAL: dict[str, {"label","category","aliases"}]
    map_to_canonical(raw_concepts: list[str]) -> list[str]
    canonical_label(cid: str) -> str

Register a new domain by dropping a module in vocab/ and adding it to DOMAINS.
Then any run or report can select it by name (--vocab <domain>).

To add the GEX knowledge base later:
    1. create vocab/gex_vocabulary.py with the same CANONICAL structure
    2. add "gex": "gex_vocabulary" below
    3. run:  python -m knowledge_ingest.report_unmapped --units ... --vocab gex
"""

import importlib

DOMAINS = {
    "ict": "ict_vocabulary",
    "gex": "gex_vocabulary",
    "nqstats": "nqstats_vocabulary",
    # "volume_profile": "volume_profile_vocabulary",
    # "expected_move": "expected_move_vocabulary",
}


def load_domain(name: str):
    """Return the vocabulary module for a domain name."""
    if name not in DOMAINS:
        raise ValueError(
            f"Unknown vocab domain '{name}'. Known: {list(DOMAINS)}. "
            f"Add it to vocab/registry.py DOMAINS after creating the module."
        )
    return importlib.import_module(f".{DOMAINS[name]}", package="knowledge_ingest.vocab")
