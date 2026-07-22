# Vocabulary Review Notes

Decisions from consolidating the `report_unmapped` candidate list (from the run
directories) into `ict_vocabulary.py`. Review the flagged items when convenient.

## What was added

### Alias gaps fixed on EXISTING concepts (no new entry — just more aliases)
- `fvg`  += "feg" (recurring transcription error), "fair value"
- `csd`  += "change in the state of delivery" (the "the" variant)
- `liquidity_sweep` += "sell side liquidity", "liquidity pool", "pool of liquidity",
  "internal liquidity", "drawn liquidity", "opposing liquidity run", "olr"
  (OLR = opposing liquidity run, confirmed by user)
- `ons_efficiency` += "inefficient", "inefficiency", "efficiently traded range",
  "overnight session profile"
- `consequent_encroachment` += "50%"

### New concepts added (corpus-confirmed)
- `rejection_block` — part of the ICT block trio
- `failure_swing` — plural "failure swings" folded in
- `breakaway_gap`
- `premium_discount` — from "dealing range"; bare "premium"/"discount" deliberately
  NOT aliased (too broad)
- `engineered_origin` — EO, confirmed by user
- `three_hour_cycle` — confirmed TCM concept

## What was deliberately CUT (and why)

### Too generic to be a useful filter
mindset, liquidity, order flow, expansion, consolidation, confluence,
risk management, risk tolerance, trading psychology, confirmation, opening price,
down candle, rules of engagement, premium.
These are so broad that filtering by them returns everything — a useless bucket.
They stay unmapped by design. (If any turns out to matter as a filter, add it.)

### Timeframe tokens — belong in a schema field, not the concept vocab
m5, m1, h1. These are TIMEFRAMES, not concepts. Recommendation: add a
`timeframe` field to the extraction schema (extracted separately) rather than
polluting concept retrieval. Dropped from vocabulary for now.

## GENERAL-ICT EXPANSION — [CHK] ALL of these, they're from general ICT
## knowledge, NOT your corpus. Confirm each matches how TCM actually teaches.

Added 21 standard ICT concepts likely to appear across 200 files but absent from
your sample so far. Delete any TCM doesn't use; correct any whose meaning differs.

- Block trio + variants: breaker_block, mitigation_block, propulsion_block,
  vacuum_block, reclaimed_ob, inversion_fvg, unicorn
- Structure/flow: choch (CHoCH), bos (break of structure), displacement, smt_divergence
- Liquidity: equal_highs_lows, liquidity_void
- Entry models: ote (OTE), turtle_soup, silver_bullet_window
- Narrative/regime: po3 (power of three), ipda, killzone

**Specific things to check:**
- **`silver_bullet_window`** — RESOLVED. "Silver bullet" now maps only to the
  intraday ICT time-window concept (10-11am / 2-3pm ET). The earlier opex
  "silver bullet" mapping was a corpus misattribution and has been removed from
  opex_week.
- **`choch` vs `csd`.** Some traders equate Change of Character with CSD/CISD,
  others treat them as distinct. Decide for TCM.
- **`bos` vs `mss`.** Break of Structure and Market Structure Shift are used
  interchangeably by some, distinctly by others. You now have both as separate
  entries — merge if TCM treats them as one.
- **Over-match fixes applied during expansion:** removed bare "breaker" (matched
  "circuit breaker") and bare "smt " (matched "wasmt"/typos). Both concepts still
  map via full phrases.

## FLAGGED — verify when you have a moment

0. **Opening-gap concepts added (NWOG/NDOG/ORG/FPFVG) — verified definitions:**
   - `nwog` New Week Opening Gap: Friday close -> Sunday/Monday open gap.
   - `ndog` New Day Opening Gap: 17:00 close -> 18:00 reopen (Mon-Thu).
   - `org` Opening Range Gap: DISTINCT — the RTH gap, vs NDOG/NWOG which are ETH.
   - `first_presented_fvg` (FPFVG) [CHK]: added with common aliases, but the exact
     definition TCM uses (first FVG after WHICH reference — NWOG? session open?)
     was NOT authoritatively confirmed. Verify against your source.

   **"EVENT HORIZON" was NOT added.** Could not confirm it as an established ICT
   term — all references were astrophysics. If TCM or your sources use it, tell me
   the definition and I'll add it precisely rather than inventing one.

1. **Bare "EO" acronym is no longer captured.** "eo" as a substring alias matched
   inside "video", "rodeo", etc., so it was removed — only the full phrase
   "engineered origin" maps now. If the corpus uses standalone "EO" often, the
   right fix is a WORD-BOUNDARY matcher in `map_to_canonical` (regex `\beo\b`)
   rather than a substring alias. Same caution applies to any 2-letter acronym.

2. **CHoCH vs CSD/CISD.** Not in this candidate list, but when it appears: some
   traders treat Change of Character (CHoCH) as the same as CSD/CISD, others as
   distinct. Decide how TCM uses it before merging or separating.

3. **`three_hour_cycle` category** — filed under `time_macro`. Confirm that's the
   right grouping vs. a `regime` concept.

4. **`premium_discount`** — folded "dealing range" here. If TCM treats "dealing
   range" as its own distinct concept (not just the premium/discount container),
   split it out.

## Recommended `map_to_canonical` enhancement (future)

Short acronyms (EO, OB, CE, MT, H1...) are risky as substring aliases. Consider
supporting a per-alias "word_boundary": true flag, or auto-applying `\b...\b`
regex matching for aliases <= 3 chars. This would let you safely add bare
acronyms without the over-match problem.
