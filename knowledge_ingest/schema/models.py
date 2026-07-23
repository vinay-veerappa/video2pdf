"""
Typed knowledge schema for trading-transcript ingestion.

Every extracted unit is one of several KNOWLEDGE TYPES. Each type has its own
payload schema, but all share a common metadata/provenance envelope so the
retrieval store, the strategy-candidate registry, and the eventual backtest ->
stats loop can all consume the same records with full audit trail.

Design principle: extraction is GROUNDED. Fields that are directly stated get
filled; anything the extractor infers goes in `inferred_fields` so it can never
be silently promoted to fact. `testability` governs what is ever allowed to flow
toward execution.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from datetime import date

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enums / controlled vocabularies
# --------------------------------------------------------------------------- #

class KnowledgeType(str, Enum):
    SETUP = "setup"                 # mechanical, testable rule
    CONTEXTUAL = "contextual"       # calendar / regime anticipation (partly testable)
    FRAMEWORK = "framework"         # how-to analytical method (not rule-testable)
    TIP = "tip"                     # heuristic rule-of-thumb
    PSYCHOLOGY = "psychology"       # discipline / mindset
    ANECDOTE = "anecdote"           # war story; extract embedded heuristic if any


class Testability(str, Enum):
    BACKTESTABLE = "backtestable"       # can be validated against bar data now
    PARTIALLY = "partially"             # some conditions testable, some not
    NOT_TESTABLE = "not_testable"       # method / mindset only


class EpistemicStatus(str, Enum):
    UNVALIDATED = "unvalidated_concept"   # default for anything from a transcript
    VALIDATED = "validated"               # cross-linked to a confirming stat
    CONTRADICTED = "contradicted"         # a stat disproved it
    MIXED = "mixed"                       # holds under some conditions only


class Session(str, Enum):
    ASIA_TOKYO = "asia_tokyo"
    LONDON = "london"
    NY_AM = "ny_am"
    NY_PM = "ny_pm"
    OVERNIGHT = "overnight"
    ANY = "any"


class Instrument(str, Enum):
    NQ = "NQ"
    ES = "ES"
    YM = "YM"
    RTY = "RTY"
    CL = "CL"
    GC = "GC"
    ANY = "any"


# --------------------------------------------------------------------------- #
# Shared metadata / provenance envelope
# --------------------------------------------------------------------------- #

class Provenance(BaseModel):
    source_file: str
    source_type: str = Field(description="own_doc | transcript | pdf | blog | chart | diagram")
    source_credibility: str = Field(
        default="unrated",
        description="own_doc | trusted_educator | community | vendor | unrated",
    )
    session_date: Optional[date] = None
    speaker: Optional[str] = None
    chunk_id: str
    timestamp_range: Optional[str] = Field(
        default=None, description="e.g. '00:02:12 - 00:04:34' from the transcript"
    )
    # image/chart provenance (null for text sources)
    image_path: Optional[str] = Field(default=None, description="path to source chart/diagram image")
    source_page: Optional[int] = Field(default=None, description="page number if from a PDF/slide deck")
    source_url: Optional[str] = Field(default=None, description="origin URL if from a blog/web source")
    extractor_model: Optional[str] = None
    extracted_at: Optional[str] = None


class KnowledgeMetadata(BaseModel):
    knowledge_type: KnowledgeType
    testability: Testability
    epistemic_status: EpistemicStatus = EpistemicStatus.UNVALIDATED
    domains: List[str] = Field(
        default_factory=lambda: ["ict"],
        description="Which knowledge domain(s) this unit covers. "
                    "e.g. ['ict'], ['gex'], ['ict','gex'] for confluence. "
                    "Set by the prompt profile used during extraction.",
    )
    session_applicability: List[Session] = Field(default_factory=lambda: [Session.ANY])
    instrument_applicability: List[Instrument] = Field(default_factory=lambda: [Instrument.ANY])
    concepts_raw: List[str] = Field(
        default_factory=list,
        description="Concept names exactly as the source phrased them.",
    )
    concepts_canonical: List[str] = Field(
        default_factory=list,
        description="Raw concepts mapped to controlled vocabulary; may differ in count.",
    )
    linked_stat_ids: List[str] = Field(
        default_factory=list,
        description="IDs in the stats registry that validate/contradict this. Empty until the loop runs.",
    )
    inferred_fields: List[str] = Field(
        default_factory=list,
        description="Names of any fields the extractor inferred rather than found stated.",
    )
    extraction_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Extractor self-rated confidence that the payload is grounded in the text.",
    )


# --------------------------------------------------------------------------- #
# Type-specific payloads
# --------------------------------------------------------------------------- #

class ChartTextContent(BaseModel):
    """The TEXT channel of a chart/diagram image.

    For reference_diagram / checklist images (e.g. the Daily Po3 diagram) the
    written panels ARE the payload — conditions, confluences, notes. Captured
    verbatim from the image. This is separate from `sequence`, which is only the
    price-path channel. An image may populate one channel or both.

    NOTE: this class MUST be defined before SetupPayload (and before
    SetupPayload.model_rebuild() is called below) — SetupPayload.text_content
    references it by name, and with `from __future__ import annotations` that
    reference is a string resolved lazily at model_rebuild() time. If this
    class isn't in the module namespace yet when rebuild runs, you get
    `PydanticUndefinedAnnotation: name 'ChartTextContent' is not defined`.
    (This bit us once already — it was defined at the bottom of the file,
    after model_rebuild(); moved up here to fix it for good.)
    """
    conditions: List[str] = Field(
        default_factory=list,
        description="Listed requirements/rules, verbatim from the image.",
    )
    confluences: List[str] = Field(
        default_factory=list,
        description="Listed confluences/desirables, verbatim.",
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Context notes, captions, caveats, definitions written on the image.",
    )
    other_text: List[str] = Field(
        default_factory=list,
        description="Meaningful text not captured by the above.",
    )
    # FUTURE (leave commented until branch-keyed diagrams prove common across the
    # corpus — one instance is not enough): bias-grouped panels so mirrored
    # bullish/bearish checklists don't collapse into one flat list. The Daily Po3
    # diagram is the motivating case; qwen currently preserves the split only by
    # string-prefixing ("Manipulation (Bullish): ...").
    # by_bias: Optional[dict[str, "ChartTextContent"]] = None


class SetupPayload(BaseModel):
    """
    Mechanical, backtestable setup. Field anatomy derived from the TCM/ICT
    transcripts: regime precondition -> bias source -> timing gate -> trigger ->
    entry -> invalidation -> target -> management. Each field is designed to have
    a testable analog in bar-based OHLCV data.
    """
    name: Optional[str] = Field(default=None, description="Short label, e.g. '9:12 Macro CSD short'")
    regime_precondition: Optional[str] = Field(
        default=None,
        description="Required market state, e.g. 'ONS inefficient', 'Profile 6', 'tight overnight range <50pt'",
    )
    bias_source: Optional[str] = Field(
        default=None,
        description="Where directional bias comes from, e.g. 'HTF draw to 13,350', 'weekly Wed expansion cycle'",
    )
    timing_gate: Optional[str] = Field(
        default=None,
        description="Time window that must be active, e.g. '8:15 offset check', '9:12 macro', '11:00 rebalance macro'",
    )
    trigger: Optional[str] = Field(
        default=None,
        description="Event that arms entry, e.g. 'CSD after liquidity sweep', 'Judas completion + MSS'",
    )
    entry: Optional[str] = Field(
        default=None,
        description="Entry rule, e.g. '50%/CE of expansion candle', 'FVG edge', 'below M1 CSD low'",
    )
    invalidation: Optional[str] = Field(
        default=None,
        description="What kills the trade, e.g. 'M5 close above down-candle high (speed rule)', '3-shot rule', 'stop above swept liquidity'",
    )
    target_logic: Optional[str] = Field(
        default=None,
        description="Target derivation, e.g. '1 SD from session open', 'ONS 50%', 'unfinished-business liquidity'",
    )
    management: Optional[str] = Field(
        default=None,
        description="Post-entry rules, e.g. 'partials at 1.5R, move to BE if no expansion in 2-3 M5 candles'",
    )
    stop_philosophy: Optional[str] = Field(
        default=None,
        description="Explicit stop-placement logic if stated separately from invalidation.",
    )
    quality_notes: Optional[str] = Field(
        default=None,
        description="Aggressive/Coke vs rules-based, conviction level, any stated win-rate/RR claims.",
    )

    # --- chart-derived setups: ordered path + spatial relations ----------- #
    # Populated when the setup comes from a chart/diagram where the MODEL is the
    # sequence of moves through zones/levels (e.g. the LRS Smart Money Reversal).
    # Null for text-derived setups that don't specify a spatial sequence.
    sequence: Optional[List["SetupStep"]] = Field(
        default=None,
        description="Ordered steps of the model. Each step: what happens, where (position), relative to which level.",
    )
    reference_levels: Optional[List[str]] = Field(
        default=None,
        description="Named levels the model references, e.g. ['HTF PDA premium','BSL/ERL','SSL/ERL','HTF PDA discount']",
    )
    kind: Optional[str] = Field(
        default=None,
        description="price_path | reference_diagram | annotated_chart | mixed. "
                    "How the chart encodes meaning; governs which channel(s) are "
                    "populated. Null for text-derived setups.",
    )
    text_content: Optional[ChartTextContent] = Field(
        default=None,
        description="The written channel of the image (panels/checklists). For a "
                    "reference_diagram this holds the payload; sequence stays empty.",
    )
    chart_inferred: List[str] = Field(
        default_factory=list,
        description="Things the VLM concluded that are NOT explicitly written/drawn. "
                    "Mirror of the review's `inferred`; kept here so the payload is "
                    "self-contained. Also copied to metadata.inferred_fields at commit.",
    )
    # NOTE on naming: SetupPayload has no `inferred` field of its own elsewhere;
    # metadata carries `inferred_fields`. `chart_inferred` is deliberately named
    # differently to avoid implying it's the same thing as metadata.inferred_fields
    # (which lists FIELD NAMES, whereas this lists free-text VLM inferences). If
    # you'd rather unify them, rename — but then fix commit() to not double-store.
    # Keeping them distinct is cleaner.


class SetupStep(BaseModel):
    """One ordered step in a chart-derived setup's path."""
    order: Optional[int] = None
    action: Optional[str] = Field(default=None, description="e.g. 'consolidate', 'sweep SSL', 'reverse', 'accumulate', 'rebalance', 'target'")
    position: Optional[str] = Field(default=None, description="premium | discount | equilibrium | null")
    range_liquidity: Optional[str] = Field(default=None, description="ERL | IRL | null (external/internal range liquidity)")
    relative_to: Optional[str] = Field(default=None, description="level/zone this step relates to, e.g. 'HTF PDA', 'BSL'")
    zone_label: Optional[str] = Field(default=None, description="label drawn on the chart for this step, e.g. 'OG Consolidation'")


class ContextualPayload(BaseModel):
    """Calendar / regime anticipation. Partly testable against 20y data."""
    event_or_condition: Optional[str] = Field(
        default=None,
        description="e.g. 'Options Expiry week', 'NFP', 'CPI/PPI day', 'Wednesday of the week', 'summer regime'"
    )
    expected_behavior: Optional[str] = Field(
        default=None,
        description="What price is expected to do, stated as concretely as the source allows."
    )
    directional_lean: Optional[str] = None
    testable_claim: Optional[str] = Field(
        default=None,
        description="A crisp, falsifiable version of the claim suitable for the backtest scaffolder, if one exists.",
    )
    date_scope: Optional[str] = Field(
        default=None, description="e.g. 'Tue/Wed of Opex week', 'monthly', 'T+2 settlement'"
    )


class FrameworkPayload(BaseModel):
    """Analytical method / how-to. Guides the companion's own analysis; not a rule."""
    method_name: Optional[str] = None
    what_it_answers: Optional[str] = Field(default=None, description="The question the method helps answer.")
    steps: Optional[List[str]] = Field(default=None, description="Ordered steps if the source gives them.")
    inputs_required: Optional[List[str]] = None
    when_to_apply: Optional[str] = None


class TipPayload(BaseModel):
    heuristic: Optional[str] = None
    rationale: Optional[str] = None
    conditions: Optional[str] = Field(default=None, description="When it applies / doesn't.")


class PsychologyPayload(BaseModel):
    principle: Optional[str] = None
    trigger_situation: Optional[str] = Field(
        default=None, description="When this should surface to the trader, e.g. 'after 2 stops', 'consolidation at entry'"
    )
    prescribed_action: Optional[str] = None


class AnecdotePayload(BaseModel):
    """War story. We keep only an embedded heuristic if present; discard narrative."""
    embedded_heuristic: Optional[str] = Field(
        default=None, description="The transferable lesson, if any. Null means the unit is pure narrative -> drop."
    )
    context: Optional[str] = None


# --------------------------------------------------------------------------- #
# Top-level record
# --------------------------------------------------------------------------- #

class KnowledgeUnit(BaseModel):
    """One extracted, typed, grounded unit of knowledge with full provenance."""
    unit_id: str = Field(description="Stable hash id: sourcefile + chunk + ordinal.")
    summary: str = Field(description="One-sentence, faithful summary in plain language.")
    verbatim_anchor: Optional[str] = Field(
        default=None,
        description="Short (<15 word) representative phrase from the source, for retrieval display / audit.",
    )
    metadata: KnowledgeMetadata
    provenance: Provenance

    # exactly one payload is populated, matching metadata.knowledge_type
    setup: Optional[SetupPayload] = None
    contextual: Optional[ContextualPayload] = None
    framework: Optional[FrameworkPayload] = None
    tip: Optional[TipPayload] = None
    psychology: Optional[PsychologyPayload] = None
    anecdote: Optional[AnecdotePayload] = None

    def retrieval_text(self) -> str:
        """The text that gets embedded for semantic search."""
        parts = [self.summary]
        parts += self.metadata.concepts_canonical or self.metadata.concepts_raw
        payload = (
            self.setup or self.contextual or self.framework
            or self.tip or self.psychology or self.anecdote
        )
        if payload is not None:
            parts.append(payload.model_dump_json())
        return "\n".join(p for p in parts if p)

    @model_validator(mode="after")
    def _payload_matches_knowledge_type(self):
        """Exactly one payload must be populated, matching metadata.knowledge_type.

        Guards against silent data corruption: a bug in _build_unit or a hand-edited
        unit could otherwise produce a record carrying knowledge_type=setup with a
        tip payload (or no payload at all). A KnowledgeUnit with the wrong/no payload
        is invisible at retrieval (retrieval_text reads only the typed payload),
        so this is load-bearing, not cosmetic.
        """
        expected = {
            KnowledgeType.SETUP: "setup",
            KnowledgeType.CONTEXTUAL: "contextual",
            KnowledgeType.FRAMEWORK: "framework",
            KnowledgeType.TIP: "tip",
            KnowledgeType.PSYCHOLOGY: "psychology",
            KnowledgeType.ANECDOTE: "anecdote",
        }
        attr = expected[self.metadata.knowledge_type]
        populated = [name for name in expected.values() if getattr(self, name) is not None]
        if populated != [attr]:
            raise ValueError(
                f"payload/{self.metadata.knowledge_type} mismatch: "
                f"expected only {attr!r} populated, got {populated}"
            )
        return self


# resolve the SetupStep / ChartTextContent forward references used in SetupPayload
SetupPayload.model_rebuild()


# =========================================================================== #
# JOURNAL SCHEMA — SEPARATE STORE, not the knowledge base.
# The knowledge base holds generalizable RULES; the journal holds YOUR TRADES.
# Different purpose (review your execution), different lifecycle, feeds the
# companion's "how am I actually trading" role. Chart-extract can write to THIS
# schema instead of the setup schema when the image is a journaled trade rather
# than teaching material — same vision machinery, different output target.
# =========================================================================== #

class JournalEntry(BaseModel):
    entry_id: str
    trade_date: Optional[date] = None
    instrument: Optional[Instrument] = None
    session: Optional[Session] = None
    direction: Optional[str] = Field(default=None, description="long | short")

    # what actually happened
    entry_price: Optional[str] = None
    exit_price: Optional[str] = None
    stop: Optional[str] = None
    target: Optional[str] = None
    outcome: Optional[str] = Field(default=None, description="win | loss | breakeven | scratch")
    r_multiple: Optional[float] = Field(default=None, description="realized R")

    # the analysis / learning
    setup_used: Optional[str] = Field(default=None, description="which model/setup was traded; can link to a knowledge-base setup name")
    what_i_saw: Optional[str] = Field(default=None, description="the read at the time")
    what_i_missed: Optional[str] = Field(default=None, description="in hindsight — the learning")
    discipline_notes: Optional[str] = Field(default=None, description="execution/psychology observations")
    concepts: List[str] = Field(default_factory=list, description="ICT concepts present in the trade (canonical)")

    # provenance — journal charts are images too
    chart_image_path: Optional[str] = None
    linked_setup_id: Optional[str] = Field(default=None, description="knowledge-base setup this trade instantiates")
    created_at: Optional[str] = None