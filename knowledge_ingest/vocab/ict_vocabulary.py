"""
Controlled ICT / TCM concept vocabulary.

Purpose: collapse the many surface phrasings a live-session speaker uses for the
same concept ("11:00 macro", "rebalance macro", "the rebalance window") down to
one canonical concept id, so the strategy-candidate registry can group and dedupe.

Design per the user's choice: keep BOTH the raw phrasing (on the record) AND the
canonical mapping. This file only supplies the canonical map; raw is preserved
on every KnowledgeUnit.

Structure:
  CANONICAL[canonical_id] = {
      "label": human label,
      "category": grouping,
      "aliases": [lowercased surface forms / regex-ish substrings to match],
  }

Matching is substring-based, case-insensitive, longest-alias-first, so
"rebalance macro" wins over a bare "macro". Extend freely; this is seeded from
the six sample transcripts and is meant to grow.

KNOWN LIMITATIONS (surfaced during the LumiTrader/Flux glossary merge, not yet
fixed because they touch existing behavior — verify against real corpus first):
  1. map_to_canonical returns ONE concept per raw string (breaks after first
     match). A raw phrase naming two concepts (e.g. "mmxm sell into premium")
     only tags the longest-alias winner ("premium_discount"), silently losing
     the other. If multi-concept raw strings are common, change the break to
     collect all matches.
  2. Pre-existing substring over-matches from short aliases still present:
     - "ce " (consequent_encroachment) matches "servi[ce] " → false positive.
     - some existing 2-letter/short aliases may similarly over-match.
     New entries in this session were made phrase-only or boundary-safe to
     avoid adding more; the older ones predate that discipline.
"""

CANONICAL = {
    # --- Time windows / macros -------------------------------------------- #
    "macro_rebalance_1100": {
        "label": "11:00 Rebalance Macro",
        "category": "time_macro",
        "aliases": ["rebalance macro", "11:00 macro", "11 am macro", "1100 macro", "11:00 am rebalance"],
    },
    "macro_lunch_1245": {
        "label": "12:45 Lunch Expansion Macro",
        "category": "time_macro",
        "aliases": ["lunch macro", "12:45 expansion", "12:45 macro", "1245 macro"],
    },
    "macro_912": {
        "label": "9:12 Macro (timed CSD)",
        "category": "time_macro",
        "aliases": ["9:12 macro", "9:12 am", "912 macro", "912"],
    },
    "macro_offset_945": {
        "label": "9:45-10:00 Offset Macro",
        "category": "time_macro",
        "aliases": ["offset macro", "9:45 macro", "9:45 - 10:00", "9:45-10:00",
                    "9.45", "945"],
    },
    "offset_window_815": {
        "label": "8:15 Offset Window",
        "category": "time_macro",
        "aliases": ["8:15 offset", "offset time window", "8:15 am transition", "8.15", "8:15 am",
                    "815", "815 open"],
    },
    "london_open_0200": {
        "label": "2:00 AM London Open",
        "category": "time_macro",
        "aliases": ["london open", "2:00 am london", "london sweep", "2 am open", "2 a.m."],
    },

    # --- Structural concepts ---------------------------------------------- #
    "fvg": {
        "label": "Fair Value Gap",
        "category": "pd_array",
        # "feg" = recurring transcription error for FVG (confirmed in corpus).
        # FPG/FBG/FDG: user-confirmed transcription typos for FVG.
        "aliases": ["fair value gap", "fvg", "feg", "imbalance", "rebalance",
                    "fair value by", "fair value", "fpg", "fbg", "fdg", "fdgs",
                    "rebalancing", "fair valley for buying", "gap"],
    },
    "order_block": {
        "label": "Order Block",
        "category": "pd_array",
        "aliases": ["order block", "ob ", "obs ", "ob"],
    },
    "csd": {
        "label": "Change in State of Delivery",
        "category": "order_flow",
        # CSB(45x)/CSV(26x)/CST(23x): user-confirmed transcription typos for CSD.
        "aliases": ["change in state of delivery", "change in the state of delivery",
                    "change state of delivery", "csd", "change in status of delivery",
                    "change the state of delivery", "csb", "csv", "cst"],
    },
    "mss": {
        "label": "Market Structure Shift",
        "category": "order_flow",
        "aliases": ["market structure shift", "mss", "shift structure", "structure shift"],
    },
    "judas_swing": {
        "label": "Judas Swing",
        "category": "manipulation",
        "aliases": ["judas swing", "judas", "false move"],
    },
    "liquidity_sweep": {
        "label": "Liquidity Sweep / Purge",
        "category": "liquidity",
        # OLR = opposing liquidity run (confirmed by user). Liquidity-pool phrasings
        # and internal/drawn liquidity all fold into the liquidity concept here.
        "aliases": ["liquidity purge", "sweep", "raid", "purge", "stop run",
                    "self-sad liquidity", "sellside liquidity", "sell-side liquidity",
                    "sell side liquidity", "buy-side liquidity", "buyside",
                    "liquidity pool", "pool of liquidity", "internal liquidity",
                    "drawn liquidity", "opposing liquidity run", "olr",
                    # corpus additions (report_unmapped, 261 files):
                    "liquidity run", "run on liquidity", "runs on liquidity",
                    "run to liquidity", "liquidity taken", "engineer liquidity",
                    "swept liquidity", "liquidity exchange", "untapped liquidity",
                    "untapped low", "untapped lows", "opposing liquidity",
                    "external liquidity", "internal range liquidity",
                    "external range liquidity", "housing liquidity",
                    "run on buy side", "run on sell side", "buy side run",
                    "sell side run", "buy side runs"],
    },
    "consequent_encroachment": {
        "label": "Consequent Encroachment (50%)",
        "category": "pd_array",
        "aliases": ["consequent encroachment", "50% (consequent", "ce ", "mean threshold",
                    "50% mean threshold", "equilibrium", "50%",
                    # corpus additions:
                    "50 percent", "eq", "ons eq", "overnight session eq",
                    "midpoint", "average price", "average pricing"],
    },
    "bpr": {
        "label": "Balanced Price Range / Matched Price Range",
        "category": "pd_array",
        "aliases": ["matched price range", "balanced price range", "bpr", "brick wall"],
    },

    # --- Regime / profile ------------------------------------------------- #
    "ons_efficiency": {
        "label": "Overnight Session Efficiency Check",
        "category": "regime",
        "aliases": ["efficient overnight", "overnight session efficiency", "efficient range",
                    "inefficient range", "ons efficient", "efficiency check",
                    "efficient vs inefficient", "inefficient", "inefficiency",
                    "efficiently traded range", "overnight session profile",
                    # corpus additions:
                    "efficient", "efficiently traded", "efficiency", "range efficiency",
                    "efficient delivery", "overnight session efficient", "inefficiencies"],
    },
    "profile_1": {"label": "Profile 1 (one-sided run)", "category": "regime", "aliases": ["profile 1", "profile one"]},
    "profile_2": {"label": "Profile 2", "category": "regime", "aliases": ["profile 2", "profile two"]},
    "profile_3": {"label": "Profile 3", "category": "regime", "aliases": ["profile 3", "profile three"]},
    "profile_4": {"label": "Profile 4", "category": "regime", "aliases": ["profile 4", "profile four"]},
    "profile_5": {"label": "Profile 5", "category": "regime", "aliases": ["profile 5", "profile five"]},
    "profile_6": {"label": "Profile 6 (inefficient ONS / news)", "category": "regime", "aliases": ["profile 6", "profile six"]},
    "holding_pattern": {
        "label": "Holding Pattern Protocol",
        "category": "regime",
        "aliases": ["holding pattern"],
    },

    # --- Calendar / seasonal ---------------------------------------------- #
    "opex_week": {
        "label": "Options Expiry Week",
        "category": "calendar",
        # "silver bullet" removed — that's the intraday ICT time-window concept
        # (see silver_bullet_window), not an opex term. Was a misattribution.
        "aliases": ["options expiry", "options expiration", "opex", "quad witching",
                    "options expire", "options expire week", "options expiring"],
    },
    "weekly_profile": {
        "label": "Weekly Day-of-Week Profile",
        "category": "calendar",
        "aliases": ["weekly profile", "wednesday is", "real open", "settlement cycle", "t+2", "t+3", "cme timeline", "monday/tuesday range",
                    "monday tuesday range", "daily profile", "wednesday", "weekly",
                    "friday's high", "thursday's high"],
    },
    "news_delivery": {
        "label": "News Delivery (CPI/PPI/NFP/FOMC)",
        "category": "calendar",
        "aliases": ["news delivery", "cpi", "ppi", "nfp", "fomc", "news day", "high-impact news",
                    "news", "news release", "high impact news", "calendar"],
    },

    # --- Targets / draws -------------------------------------------------- #
    "htf_draw": {
        "label": "Higher-Timeframe Draw on Liquidity",
        "category": "targeting",
        "aliases": ["htf draw", "draw on liquidity", "unfinished business", "objective", "terminus",
                    # corpus additions: "draw" bare is a genuine ICT shorthand for
                    # this concept in the corpus (not generic English "draw" —
                    # context is short extracted concept phrases). Flag [CHK] if
                    # this over-broadens; recanonicalize is cheap to fix.
                    "draw liquidity", "draw", "take liquidity", "drawing liquidity",
                    "upside draw", "run liquidity", "dol"],
    },
    "std_dev_projection": {
        "label": "Standard Deviation Projection from Open",
        "category": "targeting",
        "aliases": ["standard deviation", "1 sd", "one standard deviation", "std dev"],
    },

    # --- Risk / discipline ------------------------------------------------ #
    "three_shot_rule": {
        "label": "Three-Shot Rule",
        "category": "risk",
        "aliases": ["three shot", "three-shot", "three attempts", "3 shots", "three shots per idea"],
    },
    "speed_invalidation": {
        "label": "Speed / Mile-Marker Invalidation",
        "category": "risk",
        "aliases": ["speed invalidation", "mile marker", "we need speed", "2-3 candles", "2-3 m5", "speed"],
    },
    "stop_first": {
        "label": "Stop-Placement-First Principle",
        "category": "risk",
        "aliases": ["where do i put my stop", "stop placement", "stop first"],
    },

    # ===================================================================== #
    # CORPUS-CONFIRMED ADDITIONS — surfaced by report_unmapped across the
    # run directories and confirmed as real concepts. See VOCABULARY_REVIEW.md.
    # ===================================================================== #
    "rejection_block": {
        "label": "Rejection Block",
        "category": "pd_array",
        "aliases": ["rejection block"],
    },
    "failure_swing": {  # plural "failure swings" folded in
        "label": "Failure Swing",
        "category": "manipulation",
        "aliases": ["failure swing", "failure swings"],
    },
    "breakaway_gap": {
        "label": "Breakaway Gap",
        "category": "pd_array",
        # "bag" confirmed by user as shorthand for Breakaway Gap.
        "aliases": ["breakaway gap", "bag", "break away gap", "failed bag", "breakaway"],
    },
    "premium_discount": {
        "label": "Premium / Discount (dealing range)",
        "category": "pd_array",
        "aliases": ["premium array", "discount array", "premium zone", "discount zone",
                    "dealing range", "into premium", "into discount",
                    # corpus additions: bare forms re-added — matching is against
                    # short extracted concept phrases, not prose, so the original
                    # "too broad" concern is much lower here.
                    "premium", "discount", "premium run", "discount run",
                    "deep premium", "deep discount", "dr", "dr logic"],
    },
    "engineered_origin": {  # EO = engineered origin (confirmed by user)
        "label": "Engineered Origin",
        "category": "manipulation",
        # UPDATE (full-corpus report_unmapped, 261 files): bare "EO"/"eo" is by
        # far the single most common unmapped raw concept (291x + 61x + 16 "EOs"
        # + dozens of "EO range/high/low/candle/open" compounds). The over-match
        # risk noted below applies to free TEXT ("video", "rodeo"); it does NOT
        # apply here because raw_concepts are already short EXTRACTED noun
        # phrases, not prose — a phrase like "video" is not itself emitted as a
        # concept in this corpus. Adding the bare alias resolves EO, eo, EO
        # range(s), EO high/low/candle/open, EOs, timed EOs, H1EO, H1 EO, EO
        # loop, EO deliveries in one shot via substring matching. Re-verify this
        # assumption if a future extractor version starts emitting longer
        # free-text concept strings.
        "aliases": ["engineered origin", "eo"],
    },
    "three_hour_cycle": {  # confirmed TCM concept
        "label": "Three-Hour Cycle",
        "category": "time_macro",
        "aliases": ["three-hour cycle", "three hour cycle", "3 hour cycle", "3-hour cycle"],
    },

    # ===================================================================== #
    # GENERAL-ICT EXPANSION — standard ICT concepts NOT seen in your corpus
    # yet but likely to appear across 200 files. These come from general ICT
    # knowledge, NOT your transcripts, so EVERY entry here is flagged [CHK] in
    # VOCABULARY_REVIEW.md for you to confirm against TCM's actual usage.
    # Short acronyms use full phrases only (no bare 2-3 letter substring aliases)
    # to avoid the over-match problem.
    # ===================================================================== #

    # --- Block-trio PD arrays --------------------------------------------- #
    "breaker_block": {
        "label": "Breaker Block",
        "category": "pd_array",
        # bare "breaker" [CHK] re-added — 30x/28 files in corpus, dominates any
        # circuit-breaker collision risk at this scale; revert if wrong.
        "aliases": ["breaker block", "return to breaker", "breaker", "bullish breaker"],
    },
    "mitigation_block": {
        "label": "Mitigation Block",
        "category": "pd_array",
        "aliases": ["mitigation block"],
    },
    "propulsion_block": {
        "label": "Propulsion Block",
        "category": "pd_array",
        "aliases": ["propulsion block"],
    },
    "vacuum_block": {
        "label": "Vacuum Block",
        "category": "pd_array",
        "aliases": ["vacuum block"],
    },
    "reclaimed_ob": {
        "label": "Reclaimed Order Block",
        "category": "pd_array",
        "aliases": ["reclaimed order block", "reclaimed ob", "reclaimed"],
    },
    "inversion_fvg": {
        "label": "Inversion Fair Value Gap",
        "category": "pd_array",
        "aliases": ["inversion fair value gap", "inversion fvg", "inverted fvg", "ifvg"],
    },
    "unicorn": {
        "label": "Unicorn (Breaker + FVG overlap)",
        "category": "pd_array",
        "aliases": ["unicorn model", "unicorn setup"],
    },

    # --- Structure / order flow ------------------------------------------- #
    "choch": {
        "label": "Change of Character (CHoCH)",
        "category": "order_flow",
        "aliases": ["change of character", "choch"],
    },
    "bos": {
        "label": "Break of Structure",
        "category": "order_flow",
        "aliases": ["break of structure", "market structure break"],
    },
    "displacement": {
        "label": "Displacement (energetic move leaving FVG)",
        "category": "order_flow",
        "aliases": ["displacement", "displacement leg", "energetic move"],
    },
    "smt_divergence": {
        "label": "SMT Divergence (Smart Money Technique / Tool)",
        "category": "order_flow",
        # bare "smt " removed — matched inside "wasmt"/typos. Phrases only.
        # User: SMT (tool/technique) and SMT divergence are ONE concept; both
        # "technique" and "tool" expansions are in use → both kept as aliases.
        "aliases": ["smt divergence", "smt div", "smart money technique",
                    "smart money tool", "smt"],
    },

    # --- Liquidity -------------------------------------------------------- #
    "equal_highs_lows": {
        "label": "Equal Highs / Equal Lows",
        "category": "liquidity",
        "aliases": ["equal highs", "equal lows", "relative equal highs",
                    "relative equal lows"],
    },
    "liquidity_void": {
        "label": "Liquidity Void",
        "category": "liquidity",
        # "void" bare added [CHK] — 27x in corpus as its own raw concept string;
        # verify it isn't being used for something else (e.g. a gap/FVG synonym).
        "aliases": ["liquidity void", "void"],
    },

    # --- Entry models ----------------------------------------------------- #
    "ote": {
        "label": "Optimal Trade Entry (OTE)",
        "category": "entry_model",
        "aliases": ["optimal trade entry", "ote"],
    },
    "turtle_soup": {
        "label": "Turtle Soup (false-breakout reversal)",
        "category": "entry_model",
        "aliases": ["turtle soup"],
    },
    "silver_bullet_window": {
        "label": "Silver Bullet Time Window (10-11am / 2-3pm ET)",
        "category": "time_macro",
        "aliases": ["silver bullet window", "silver bullet setup", "silver bullet",
                    "10 a.m. open", "10 o'clock", "10am"],
    },

    # --- Narrative / regime ----------------------------------------------- #
    "po3": {
        "label": "Power of Three (accum-manip-distribution)",
        "category": "regime",
        "aliases": ["power of three", "accumulation manipulation distribution",
                    "open high low close model", "daily cycle", "daily cycles", "cycle"],
    },
    "ipda": {
        "label": "IPDA (Interbank Price Delivery Algorithm)",
        "category": "regime",
        "aliases": ["ipda", "interbank price delivery", "price delivery algorithm"],
    },
    "killzone": {
        "label": "Killzone (session time window)",
        "category": "time_macro",
        "aliases": ["killzone", "kill zone", "london killzone", "new york killzone",
                    "ny killzone", "asian killzone"],
    },

    # --- Opening gaps (recent ICT concepts) ------------------------------- #
    "nwog": {
        "label": "New Week Opening Gap",
        "category": "pd_array",
        # Friday close -> Sunday/Monday open gap; algorithmic draw, 50% CE watched.
        "aliases": ["new week opening gap", "nwog", "weekly opening gap"],
    },
    "ndog": {
        "label": "New Day Opening Gap",
        "category": "pd_array",
        # 17:00 close -> 18:00 reopen daily gap (Mon-Thu). Daily repricing draw.
        "aliases": ["new day opening gap", "ndog", "daily opening gap"],
    },
    "org": {
        "label": "Opening Range Gap",
        "category": "pd_array",
        # DISTINCT from NDOG/NWOG: ORG is the RTH gap; NDOG/NWOG are ETH gaps.
        "aliases": ["opening range gap", "org "],
    },
    "first_presented_fvg": {
        # CONFIRMED (user): a GENERAL pattern, not tied to one clock time — the
        # FIRST FVG to form after a SIGNIFICANT OPEN. Significant opens include
        # midnight, the 9:30 NY open, the 1:30 PM open, and (user's method) the
        # first FVG of EVERY hour. Bullish or bearish. Spotted on 1m, traded 5m/15m.
        # MODELING (Option B): this is ONE concept = "first-FVG-after-an-open".
        # WHICH open (midnight / 9:30 / 1:30 PM / top-of-hour) is captured
        # SEPARATELY via the reference_level/time anchors below (ny_midnight_open,
        # opening_price, pm_session_open_1330, hourly_open), so the registry can
        # still distinguish a midnight FPFVG from a 9:30 FPFVG for backtesting.
        # Aliases here are time-AGNOSTIC on purpose; do not bake a clock time in.
        "label": "First Presented FVG (FPFVG) — first FVG after a significant open",
        "category": "pd_array",
        "aliases": ["first presented fair value gap", "first presented fvg",
                    "first presented f v g", "fpfvg", "first fvg",
                    "1st presented fvg", "1st presented fair value gap",
                    "first presented", "first fair value gap",
                    "first fvg of the hour", "first fvg of the session",
                    "first fvg after the open", "opening fvg"],
    },

    # ====================================================================== #
    # ADDITIONS from glossaries: LumiTrader ICT-2022 book + Flux NY guide.
    # Cross-referenced against the 56 existing entries; only genuine gaps added.
    # Two tranches: (1) genuine concepts, (2) structural reference levels.
    # Short/common-substring acronyms use boundary-safe alias forms (trailing
    # space, period, or spelled-out) to avoid the over-match bug the docstring
    # warns about (e.g. bare "ad" would match inside "trade"). This follows the
    # file's existing convention (see "ob ", "ce " above).
    # ====================================================================== #

    # --- Tranche 1: genuine concepts -------------------------------------- #
    "volume_imbalance": {
        "label": "Volume Imbalance",
        "category": "pd_array",
        # VI is important (user-flagged). Bare/short "vi" forms over-match badly
        # ("vivid", "service", "advice"), so phrase-only. A bare "VI" mention is
        # rare and is still preserved in concepts_raw per the capture-once design.
        "aliases": ["volume imbalance", "vol imbalance", "v.i "],
    },
    "iof": {
        "label": "Institutional Order Flow",
        "category": "order_flow",
        "aliases": ["institutional order flow", "iof", "institutional orderflow"],
    },
    "iofed": {
        "label": "Institutional Order Flow Entry Drill (IOFED)",
        "category": "entry_model",
        "aliases": ["institutional order flow entry", "iofed", "iofe drill"],
    },
    "hrlr": {
        "label": "High Resistance Liquidity Run",
        "category": "regime",
        "aliases": ["high resistance liquidity run", "high resistance liquidity",
                    "hrlr", "high resistance run", "jagged to smooth", "high resistance"],
    },
    "lrlr": {
        "label": "Low Resistance Liquidity Run",
        "category": "regime",
        "aliases": ["low resistance liquidity run", "low resistance liquidity",
                    "lrlr", "low resistance run", "smooth to jagged", "low resistance"],
    },
    "smr": {
        "label": "Smart Money Reversal",
        "category": "order_flow",
        "aliases": ["smart money reversal", "smr ", " smr", "smart-money reversal"],
    },
    "cbdr": {
        "label": "Central Bank Dealers Range",
        "category": "time_macro",
        "aliases": ["central bank dealers range", "central bank dealer range", "cbdr"],
    },
    "adr": {
        "label": "Average Daily Range",
        "category": "targeting",
        "aliases": ["average daily range", "adr "],
    },
    "dot": {
        "label": "Draw on Time",
        "category": "targeting",
        # bare "dot" over-matches ("dot com", punctuation); require context.
        "aliases": ["draw on time", "draw-on-time", "dot ("],
    },
    # UPDATE: "Draw on Liquidity" (DOL) is now folded into `htf_draw` (alias
    # "dol") per this file's own original reasoning below — DOL and HTF-draw
    # are the same concept in this dialect. If they should be DISTINCT, split
    # htf_draw instead of re-adding a parallel `dol` entry.
    "seek_and_destroy": {
        "label": "Seek and Destroy (choppy 2-sided day)",
        "category": "regime",
        "aliases": ["seek and destroy", "seek & destroy", "s&d", "seek-and-destroy", "chop"],
    },
    "measuring_gap": {
        "label": "Measuring Gap",
        "category": "pd_array",
        "aliases": ["measuring gap"],  # "mg" too collision-prone; spell out only
    },
    "redelivered_rebalanced": {
        "label": "Redelivered Rebalanced PD Array (RDRB)",
        "category": "pd_array",
        "aliases": ["redelivered rebalanced", "rdrb", "redelivery rebalance"],
    },
    "return_to_origin": {
        "label": "Return to Origin / Return to OB (RHO)",
        "category": "order_flow",
        "aliases": ["return to origin", "return to ob", "rho ", "return-to-origin"],
    },
    "risk_reward": {
        "label": "Risk/Reward Ratio (RR)",
        "category": "risk",
        # bare "rr" over-matches in free TEXT ("current", "arrow"); risk is much
        # lower here since raw_concepts are short extracted phrases, not prose.
        "aliases": ["risk/reward", "risk reward", "reward/risk", "r/r", "r:r",
                    "risk to reward", "rr ratio", "rr"],
    },
    "amd": {
        "label": "Accumulation-Manipulation-Distribution (AMD)",
        "category": "regime",
        # This is the Po3 cycle; kept distinct since AMD is named as its own term.
        "aliases": ["accumulation manipulation distribution",
                    "accumulation, manipulation, distribution", "amd ", " amd",
                    "accumulation-manipulation-distribution"],
    },
    "mmxm": {
        "label": "Market Maker Buy/Sell Model (MMXM)",
        "category": "entry_model",
        "aliases": ["market maker buy model", "market maker sell model",
                    "market maker buy/sell model", "market maker model", "mmxm",
                    "mmbm", "mmsm", "market maker x model", "market maker"],
    },
    "mean_threshold": {
        "label": "Mean Threshold (of a PD array)",
        "category": "pd_array",
        # "mt " removed — matched "wasmt " and similar. Phrase forms only.
        "aliases": ["mean threshold", "mean-threshold"],
    },

    # --- Tranche 2: structural reference levels --------------------------- #
    # Coordinates/levels that setups TARGET or invalidate against, not concepts.
    # Own category so they don't dilute concept grouping in the registry.
    # Two-letter forms included per user request; kept boundary-safe where the
    # bare form is a common English substring.
    "high_low_of_day": {
        "label": "High / Low of Day (HOD/LOD)",
        "category": "reference_level",
        "aliases": ["high of day", "low of day", "hod", "lod", "hod/lod",
                    "lod/hod", "high of the day", "low of the day",
                    # corpus additions:
                    "daily high", "daily low", "daily lows", "daily highs",
                    "daily range", "daily levels", "daily level"],
    },
    "high_low_of_week": {
        "label": "High / Low of Week (HOW/LOW)",
        "category": "reference_level",
        # "low" alone is a ubiquitous word; only match the acronym/phrase forms.
        "aliases": ["high of week", "low of week", "how ", "high of the week",
                    "low of the week", "hoW/loW", "weekly high", "weekly low",
                    "weekly range"],
    },
    "previous_day_high_low": {
        "label": "Previous Day High / Low (PDH/PDL)",
        "category": "reference_level",
        "aliases": ["previous day high", "previous day low", "pdh", "pdl",
                    "prev day high", "prev day low", "pdh/pdl", "pdl/pdh",
                    # corpus additions:
                    "previous day's low", "previous day's range",
                    "previous days high and low", "previous daily low",
                    "previous range"],
    },
    "previous_week_high_low": {
        "label": "Previous Week High / Low (PWH/PWL)",
        "category": "reference_level",
        "aliases": ["previous week high", "previous week low", "pwh", "pwl",
                    "prev week high", "prev week low", "pwh/pwl",
                    "previous weekly range"],
    },
    "intermediate_term_hl": {
        "label": "Intermediate-Term High / Low (ITH/ITL)",
        "category": "reference_level",
        "aliases": ["intermediate-term high", "intermediate-term low",
                    "intermediate term high", "intermediate term low",
                    "ith", "itl", "ith/itl"],
    },
    "short_term_hl": {
        "label": "Short-Term High / Low (STH/STL)",
        "category": "reference_level",
        "aliases": ["short-term high", "short-term low", "short term high",
                    "short term low", "sth", "stl", "sth/stl"],
    },
    "long_term_hl": {
        "label": "Long-Term High / Low (LTH/LTL)",
        "category": "reference_level",
        "aliases": ["long-term high", "long-term low", "long term high",
                    "long term low", "lth", "ltl", "lth/ltl"],
    },
    "ny_midnight_open": {
        "label": "NY Midnight Opening Price (NMO/Midnight Open)",
        "category": "reference_level",
        "aliases": ["midnight open", "midnight opening", "ny midnight open",
                    "nmo", "midnight opening price", "true day open", "true day"],
    },
    "opening_price": {
        "label": "Opening Price (session/8:30/9:30)",
        "category": "reference_level",
        # bare "op" over-matches badly ("open", "stop"); phrase forms only.
        "aliases": ["opening price", "session open", "8:30 open", "9:30 open",
                    "true open", "true session open", "open price", "930",
                    "830 candle"],
    },
    "pm_session_open_1330": {
        # Anchor for the 1:30 PM open (PM-session FPFVG, opening range). Distinct
        # from the 9:30 AM open captured by opening_price.
        "label": "PM Session Open (1:30 PM ET)",
        "category": "reference_level",
        "aliases": ["1:30 open", "1:30 pm open", "130 open", "13:30 open",
                    "pm session open", "pm open", "afternoon open",
                    "1:30 opening range"],
    },
    "hourly_open": {
        # Anchor for the top-of-hour open (user's per-hour FPFVG method).
        "label": "Top-of-Hour Open (hourly open)",
        "category": "reference_level",
        "aliases": ["top of hour", "top-of-hour", "hourly open", "start of the hour",
                    "opening of the hour", "hour open"],
    },

    # ====================================================================== #
    # HARMONIZATION PASS — full-corpus report_unmapped (261 files, 8724 raw
    # concepts, 79% unmapped). These are NEW concepts for clusters that showed
    # up at real frequency/file-spread and did not fit any existing entry
    # above (existing entries were extended in place instead, see diffs).
    # Umbrella concepts are used where the corpus draws a finer distinction
    # than is worth encoding yet (e.g. one "trading_psychology" bucket instead
    # of 15 separate emotion concepts) — split later via recanonicalize if the
    # registry needs the finer grain; that's cheap and lossless (concepts_raw
    # is preserved on every unit).
    # ====================================================================== #

    # --- Order flow / delivery (framework language, distinct from CSD) ----- #
    "order_flow": {
        "label": "Order Flow (bullish/bearish)",
        "category": "order_flow",
        "aliases": ["order flow", "bullish order flow", "bearish order flow",
                    "order flow is bearish", "daily order flow",
                    "higher time frame order flow"],
    },
    "state_of_delivery": {
        "label": "State of Delivery (one-sided / new / perfect / efficient)",
        "category": "order_flow",
        "aliases": ["state of delivery", "order of delivery", "delivery", "deliveries",
                    "new delivery", "perfect delivery", "one-sided delivery",
                    "one sided delivery", "buy side delivery", "sell side delivery",
                    "buy-side delivery", "bullish delivery", "bearish delivery",
                    "eo deliveries", "time delivery", "delivery origin",
                    "delivery fractal", "delivery fractals"],
    },
    "confirmation": {
        "label": "Directional Confirmation",
        "category": "order_flow",
        "aliases": ["confirmation", "directional confirmation", "confirmation of direction",
                    "direction is confirmed", "m5 confirmation", "entry confirmation",
                    "confirmations"],
    },
    "bias": {
        "label": "Directional Bias",
        "category": "order_flow",
        "aliases": ["bias", "bearish bias", "bullish bias", "directional bias", "daily bias",
                    "directional premise"],
    },
    "narrative": {
        "label": "Narrative (directional story)",
        "category": "order_flow",
        "aliases": ["narrative", "support narrative", "bullish narrative",
                    "h1 narrative", "m5 narrative"],
    },

    # --- Sessions (distinct from the ons_efficiency JUDGMENT concept) ------ #
    "overnight_session": {
        "label": "Overnight Session (ONS)",
        "category": "session",
        "aliases": ["overnight session", "ons", "overnight session low", "OMS",
                    "overnight session high", "overnight session range",
                    "overnight session eq", "refined ons", "ons high", "ons low",
                    "ons range", "ons eq", "ons profiles"],
    },
    "asian_session": {
        "label": "Asian Session",
        "category": "session",
        "aliases": ["asian session", "asian range"],
    },
    "london_session": {
        "label": "London Session",
        "category": "session",
        "aliases": ["london session"],
    },
    "equities_open": {
        "label": "Equities/RTH Open",
        "category": "session",
        "aliases": ["equities open"],
    },
    "forum_open": {
        "label": "Forum Open",
        "category": "session",
        "aliases": ["forum open"],
    },
    "weekly_open_level": {
        "label": "Weekly Open (level)",
        "category": "reference_level",
        "aliases": ["weekly open"],
    },

    # --- Timeframe tags (context, not techniques) -------------------------- #
    "timeframe_m1": {
        "label": "M1 (1-minute chart)",
        "category": "timeframe",
        "aliases": ["m1"],
    },
    "timeframe_m5": {
        "label": "M5 (5-minute chart)",
        "category": "timeframe",
        "aliases": ["m5"],
    },
    "timeframe_h1": {
        "label": "H1 (1-hour chart)",
        "category": "timeframe",
        "aliases": ["h1", "hourly candle"],
    },

    # --- Retracement family -------------------------------------------------#
    "retracement": {
        "label": "Retracement (general)",
        "category": "structure",
        "aliases": ["retracement", "retracements", "retrace", "retracing",
                    "retracement target", "retracement protocol", "retraces"],
    },
    "deep_retracement": {
        "label": "Deep Retracement",
        "category": "structure",
        "aliases": ["deep retracement", "deep retracement protocol"],
    },
    "shallow_retracement": {
        "label": "Shallow Retracement",
        "category": "structure",
        "aliases": ["shallow retracement", "shallow retracement protocol"],
    },

    # --- Structural / candle patterns --------------------------------------#
    "swing_point": {
        "label": "Swing High / Low",
        "category": "structure",
        "aliases": ["swing high", "swing low", "swing highs", "swing lows",
                    "higher low", "higher lows", "highest high"],
    },
    "three_bar_swing": {
        "label": "Three-Bar Swing High / Low",
        "category": "structure",
        "aliases": ["three bar swing high", "three bar swing low", "three bar high",
                    "three bar low", "three candle swing high", "three candle swing low",
                    "three-bar swing low", "three bar swing", "three bar lows",
                    "h1 three bar high", "m5 three bar high"],
    },
    "candle_polarity": {
        "label": "Up / Down Candle",
        "category": "structure",
        "aliases": ["down candle", "up candle", "down candles", "up candles low",
                    "down close candle", "up close candle", "candle body",
                    "down candle high", "down candle body", "down candle bodies",
                    "up candle low", "up candle body", "bodies", "wick",
                    "highest candle", "highest up candle", "highest closing up candle",
                    "lowest closing down candle", "lowest down candle",
                    "m5 down candle", "body above", "body below", "wic"],
    },
    "candle_close_relative": {
        "label": "Close/Body Above or Below a Level",
        "category": "structure",
        "aliases": ["close below", "close above", "close above 50", "body closure"],
    },
    "trigger_candle": {
        "label": "Trigger Candle",
        "category": "entry_model",
        "aliases": ["trigger candle", "propulsion candle", "trigger", "triggers"],
    },
    "trading_range": {
        "label": "Trading Range (general)",
        "category": "structure",
        "aliases": ["trading range", "range low", "range high", "consolidation",
                    "consolidating", "consolidation profile", "large range day",
                    "inside day", "key level", "key levels",
                    # user confirmed: distortion == consolidation in this dialect.
                    "distortion", "distortion profile", "range", "ranges"],
    },
    "business_card_model": {
        "label": "Business Card Model (named trading model)",
        "category": "entry_model",
        "aliases": ["business card model", "biz card model", "biz card",
                    "base card model"],
    },
    "zoom_model": {
        "label": "Zoom Model (named trading model)",
        "category": "entry_model",
        "aliases": ["zoom model"],
    },
    "accumulation_phase": {
        "label": "Accumulation / Reaccumulation / Distribution (phase)",
        "category": "regime",
        "aliases": ["accumulation", "reaccumulation", "re-accumulation", "reaccumulate",
                    "distribution", "distribute", "manipulation"],
    },

    # --- Risk / trade management -------------------------------------------#
    "stop_loss": {
        "label": "Stop Loss",
        "category": "risk",
        "aliases": ["stop loss", "stop point", "stop point leg", "invalidation point",
                    "invalidation"],
    },
    "break_even": {
        "label": "Break Even",
        "category": "risk",
        "aliases": ["break even", "breakeven"],
    },
    "partial_profit_taking": {
        "label": "Partials / Scaling Out",
        "category": "risk",
        "aliases": ["partials", "partial", "scaling out", "runners"],
    },
    "lot_size": {
        "label": "Lot / Contract Size",
        "category": "risk",
        "aliases": ["lot size", "contract size"],
    },
    "risk_management": {
        "label": "Risk Management (general practice)",
        "category": "risk",
        "aliases": ["risk management", "risk tolerance", "risk parameters", "risk appetite",
                    "accepting risk", "accept risk", "accept the risk", "manage the risk",
                    "control the risk", "cut risk", "reduced risk", "remove the risk",
                    "risk is managed", "low risk", "risk"],
    },

    # --- Trading psychology (umbrella — split later if the registry needs it) #
    "trading_psychology": {
        "label": "Trading Psychology (discipline/emotion/mindset)",
        "category": "psychology",
        "aliases": ["discipline", "disciplined", "patience", "patient", "confidence",
                    "ego", "fomo", "greed", "revenge trading", "revenge trade",
                    "overtrading", "hope", "composure", "overthinking",
                    "mental capital", "mindset", "trading mindset",
                    "psychological factors", "unrealistic expectations",
                    "trading psychology", "emotions", "conviction", "self-awareness",
                    "chasing", "journaling", "journal", "progression plan",
                    "capital preservation", "stay flat",
                    # corpus additions (round 2):
                    "rules", "rules of engagement", "rule-based loss", "process",
                    "precision", "edge", "psychology", "mechanical", "back testing",
                    "backtest", "trade management", "trade plan", "leverage",
                    "refinement"],
    },

    # --- Macro / fundamental context ---------------------------------------#
    "macro_context": {
        "label": "Macro / Fundamental Context (DXY, rates, data)",
        "category": "macro",
        "aliases": ["dxy", "dxy strength", "dxy weakness", "dollar", "dollar strength",
                    "dollar weakness", "bullish dollar", "pmi", "interest rates",
                    "unemployment claims", "consumer sentiment", "consumer confidence",
                    "inflation", "cme", "earnings", "sentiment", "sentiment driven"],
    },

    # ====================================================================== #
    # HARMONIZATION PASS 2 — 79% -> 51% unmapped. Same discipline: existing
    # concepts extended in place above; genuinely new clusters below.
    # ====================================================================== #

    "buy_side_sell_side": {
        "label": "Buy-Side / Sell-Side (directional, generic)",
        "category": "order_flow",
        "aliases": ["buy side", "sell side", "buy-side", "sell-side", "internal buy side",
                    "by side", "buy to sell"],
    },
    "expansion": {
        "label": "Expansion (range/price expansion)",
        "category": "order_flow",
        "aliases": ["expansion", "bearish expansion", "last expansion", "expand"],
    },
    "offsetting": {
        "label": "Offsetting (position)",
        "category": "order_flow",
        "aliases": ["offset", "offsetting"],
    },
    "volatility": {
        "label": "Volatility",
        "category": "macro",
        "aliases": ["volatility"],
    },
    "fractal": {
        "label": "Fractal (general market structure)",
        "category": "structure",
        "aliases": ["fractal", "fractals"],
    },
    "chart_view": {
        "label": "Chart Timeframe View (daily/monthly)",
        "category": "timeframe",
        "aliases": ["daily chart", "monthly chart"],
    },
    "confluence": {
        "label": "Confluence",
        "category": "framework",
        "aliases": ["confluence"],
    },
    "trade_probability": {
        "label": "Trade Probability Assessment",
        "category": "framework",
        "aliases": ["high probability", "low probability"],
    },
    "tape_reading": {
        "label": "Tape Reading",
        "category": "framework",
        "aliases": ["tape reading", "tape read", "read the tape"],
    },
    "macro_window_generic": {
        "label": "Macro Window (unspecified time)",
        "category": "time_macro",
        "aliases": ["macro", "macros"],
    },
    "hyperscalping": {
        "label": "Hyperscalping",
        "category": "entry_model",
        "aliases": ["hyperscalping", "hyperscalp"],
    },
    "loop_closure": {
        "label": "Loop Closure (trade idea completes)",
        "category": "framework",
        "aliases": ["loop closure", "loops", "loop", "closed the loop"],
    },
    "spread_cost": {
        "label": "Spread (trading cost)",
        "category": "risk",
        "aliases": ["spreads", "spread"],
    },
    "stop_hunt": {
        "label": "Stop Hunt",
        "category": "liquidity",
        "aliases": ["stop hunt", "stop hunting", "run on stops"],
    },
    "market_structure": {
        "label": "Market Structure (general)",
        "category": "order_flow",
        "aliases": ["market structure", "micro market structure"],
    },
    "signature": {
        "label": "Signature (recognizable pattern)",
        "category": "structure",
        "aliases": ["signatures", "bullish signature"],
    },
    "fibonacci": {
        "label": "Fibonacci",
        "category": "targeting",
        "aliases": ["fib"],
    },
    "reversal": {
        "label": "Reversal",
        "category": "order_flow",
        "aliases": ["reversal", "turning point"],
    },
    "fade_setup": {
        "label": "Fade (counter-trend entry)",
        "category": "entry_model",
        "aliases": ["fade"],
    },
    "support_resistance": {
        "label": "Support / Resistance",
        "category": "structure",
        "aliases": ["support", "resistance", "support and resistance"],
    },
    "order_pairing": {
        "label": "Order Pairing",
        "category": "order_flow",
        "aliases": ["order pairing", "order peering"],
    },
    "technical_analysis": {
        "label": "Technical Analysis (general)",
        "category": "framework",
        "aliases": ["technical analysis"],
    },
    "entry_timing": {
        "label": "Entry Timing (aggressive/early)",
        "category": "entry_model",
        "aliases": ["aggressive entry", "aggressive entries", "early entry",
                    "early entries", "aggressive trade"],
    },
    "prop_firm": {
        "label": "Prop Firm (context)",
        "category": "risk",
        "aliases": ["prop firm"],
    },
    "take_profit": {
        "label": "Take Profit (TP)",
        "category": "risk",
        "aliases": ["tp"],
    },
    "rejection": {
        "label": "Rejection (price rejects a level)",
        "category": "structure",
        "aliases": ["rejection"],
    },
    "scalping": {
        "label": "Scalping",
        "category": "entry_model",
        "aliases": ["scalp", "scalping"],
    },
    "order_book": {
        "label": "Order Book",
        "category": "order_flow",
        "aliases": ["order book"],
    },
    "timeframe_analysis": {
        "label": "Higher/Lower Timeframe Analysis (generic)",
        "category": "framework",
        "aliases": ["higher time frame", "higher time frame analysis",
                    "lower time frame", "time frame", "time and price"],
    },
    "extended_hours": {
        "label": "Extended / External Trading Hours",
        "category": "session",
        "aliases": ["external trading hours", "extended trading hours"],
    },
    "utc_minus_five": {
        "label": "UTC-5 Timezone Reference",
        "category": "time_macro",
        "aliases": ["utc minus five"],
    },
    "impulse_move": {
        "label": "Impulse Move",
        "category": "order_flow",
        "aliases": ["impulse"],
    },
    "instrument_type": {
        "label": "Instrument Type (futures/CFD)",
        "category": "context",
        "aliases": ["futures", "cfd", "cfds"],
    },
    "pullback_move": {
        "label": "Pullback",
        "category": "structure",
        "aliases": ["pullback"],
    },
    "hourly_rotation": {
        "label": "Hourly Rotation",
        "category": "structure",
        "aliases": ["hourly rotation", "rotation"],
    },
    "buy_program": {
        "label": "Buy Program",
        "category": "order_flow",
        "aliases": ["buy program"],
    },
    "short_squeeze": {
        "label": "Short Squeeze",
        "category": "regime",
        "aliases": ["short squeeze"],
    },
    "breakout": {
        "label": "Breakout",
        "category": "structure",
        "aliases": ["breakout"],
    },
    "moving_average": {
        "label": "Moving Average",
        "category": "targeting",
        "aliases": ["moving average"],
    },
    "projection_block": {
        "label": "Projection Block",
        "category": "pd_array",
        "aliases": ["projection block"],
    },
    "setup_quality": {
        "label": "Setup Quality (A-plus / textbook)",
        "category": "entry_model",
        "aliases": ["a plus setup", "textbook setup"],
    },
    "end_of_month": {
        "label": "End of Month (calendar)",
        "category": "calendar",
        "aliases": ["end of the month"],
    },

    # --- Flagged [CHK] — grouped by plausible relation, NOT confidently
    # defined. Confirm meaning before trusting; safe to delete if wrong. ----- #
    "pdra_cluster": {
        "label": "PD Array / PDRA (Premium/Discount Reference Array)",
        "category": "pd_array",
        # confirmed by user: PDRA = PD Array. NOTE: bare "DR"/"DR logic" moved
        # OUT of here — user clarified DR = Dealing Range, see premium_discount.
        "aliases": ["pdr", "pdrs", "pdras", "pd arrays", "pd array"],
    },
    "bookmaking": {
        "label": "Bookmaking",
        "category": "framework",
        "aliases": ["bookmaking fractal", "bookmaking process"],
    },
    "macro_4am": {
        # User confirmed: this is a specific time he refers to consistently,
        # not a generic timestamp. Label is generic pending the exact
        # significance (news window? session boundary?) — refine if you want
        # more than just the time anchor captured.
        "label": "4:00 AM Reference Time",
        "category": "time_macro",
        "aliases": ["4 a.m.", "4am", "4:00 am"],
    },
    # --- Kish's 7 Rules (TCM numbered execution framework) -----------------
    # User provided the full content of each rule from his presentation
    # slides/notes. These replace the earlier generic "personal_rule_citation"
    # placeholder — now that the actual content is known, each rule gets its
    # own concept rather than one undifferentiated bucket. Aliases are the
    # CITATION forms ("rule 3", "rule number three", etc.); the label carries
    # the actual rule content for reference. Some rule content overlaps
    # existing concepts (Rule 1 = order_flow, Rule 7 references state_of_delivery/
    # csd/liquidity_sweep) — that's fine, a unit can be tagged with both the
    # specific numbered rule AND the underlying concept it invokes.
    "tcm_rule_1": {
        "label": "TCM Rule #1 \u2014 Determine Daily Order Flow before looking for setups "
                 "(establish bullish/bearish bias first)",
        "category": "framework",
        "aliases": ["rule number one", "rule 1", "rule #1", "rule one"],
    },
    "tcm_rule_2": {
        "label": "TCM Rule #2 \u2014 Identify Entry Formation Location "
                 "(e.g. above/below the ONS range)",
        "category": "entry_model",
        "aliases": ["rule number two", "rule 2", "rule #2", "rule two"],
    },
    "tcm_rule_3": {
        "label": "TCM Rule #3 \u2014 Entry Confluences by OB anatomy "
                 "(FVG-in-OB \u2192 enter at inefficiency; large body \u2192 enter at 50%/mean "
                 "threshold; small body \u2192 enter open+high (bearish) or open+low (bullish))",
        "category": "entry_model",
        "aliases": ["rule number three", "rule 3", "rule #3", "rule three",
                    "entry confluences"],
    },
    "tcm_rule_4": {
        "label": "TCM Rule #4 \u2014 The FVG Filter (continuation): break of low + up-candle "
                 "\u2192 next candle trades into an FVG above it \u2192 aggressive expansion lower",
        "category": "entry_model",
        "aliases": ["rule number four", "rule 4", "rule #4", "rule four", "fvg filter"],
    },
    "tcm_rule_5": {
        "label": "TCM Rule #5 \u2014 The Timeframe Filter (London session \u2192 M15 structure; "
                 "New York session \u2192 M5 structure)",
        "category": "framework",
        "aliases": ["rule number five", "rule 5", "rule #5", "rule five", "timeframe filter"],
    },
    "tcm_rule_6": {
        "label": "TCM Rule #6 \u2014 Liquidity at Swing Points (stop placement): distrust a "
                 "swing high/low formed by a wick, especially a wick inside an FVG \u2014 "
                 "expect it to be raided",
        "category": "risk",
        "aliases": ["rule number six", "rule 6", "rule #6", "rule six"],
    },
    "tcm_rule_7": {
        "label": "TCM Rule #7 \u2014 Order of Delivery (\"kissing\" fractal continuation): "
                 "SSL run \u2192 FVG tag \u2192 short-term low \u2192 CSD confirms high-probability short "
                 "(selling an FVG immediately rebalanced right after an SSL run is LOW probability)",
        "category": "entry_model",
        "aliases": ["rule number seven", "rule 7", "rule #7", "rule seven",
                    "kissing fractal", "kissing fractal continuation"],
    },
    # --- Arjo (Arjo15m) entry-model terminology (user-confirmed 2026-07-21) ---
    # These are Arjo-specific named entry mechanics, distinct from generic ICT concepts.
    # OD = Overlapping Defense (aggressive entry at top of displacement/FVG zone)
    # FLOD = First Line of Defense (conservative entry at lower threshold)
    # LLOD = Last Line of Defense (final entry level)
    # ST = Sharp Turn (the structural rejection that confirms the entry model)
    "sharp_turn": {
        "label": "Sharp Turn (ST) \u2014 Arjo entry model: structural rejection at a liquidity "
                 "point confirmed by an FVG (displacement candle creates the inefficiency)",
        "category": "entry_model",
        "aliases": ["sharp turn", "st entry", "st model", "sharpturn"],
    },
    "overlapping_defense": {
        "label": "Overlapping Defense (OD) \u2014 Arjo: aggressive entry at the top of the "
                 "displacement/FVG zone; tighter risk, higher R:R",
        "category": "entry_model",
        "aliases": ["overlapping defense", "od entry", "od ", "overlapping defense entry"],
    },
    "first_line_of_defense": {
        "label": "First Line of Defense (FLOD) \u2014 Arjo: conservative/confirmation entry at "
                 "the lower threshold as price exits the imbalance zone",
        "category": "entry_model",
        "aliases": ["first line of defense", "flod entry", "flod ", "first line of defense entry"],
    },
    "last_line_of_defense": {
        "label": "Last Line of Defense (LLOD) \u2014 Arjo: final entry level at the last "
                 "defensive threshold before the target",
        "category": "entry_model",
        "aliases": ["last line of defense", "llod entry", "llod ", "last line of defense entry"],
    },
}


def _alias_index():
    """Build a (alias, canonical_id) list sorted longest-first for greedy matching."""
    pairs = []
    for cid, spec in CANONICAL.items():
        for alias in spec["aliases"]:
            pairs.append((alias.lower(), cid))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


_ALIAS_INDEX = _alias_index()


def map_to_canonical(raw_concepts):
    """
    Map a list of raw concept strings to a de-duplicated list of canonical ids.
    Substring match, longest-alias-first. Unmatched raw concepts are dropped from
    the canonical list (but remain in concepts_raw on the record).

    Multi-concept support: a raw phrase may name MORE than one concept (e.g.
    "mmxm sell into premium" -> both `mmxm` and `premium_discount`). We collect
    ALL matching canonical ids per raw concept, not just the first. This was a
    known gap (§14d of HANDOVER.md) where the old `break` after first match
    silently dropped the second concept, fragmenting the KB's concept index.
    """
    found = []
    for raw in raw_concepts:
        r = raw.lower()
        for alias, cid in _ALIAS_INDEX:
            if alias in r and cid not in found:
                found.append(cid)
    return found


def canonical_label(cid):
    return CANONICAL.get(cid, {}).get("label", cid)