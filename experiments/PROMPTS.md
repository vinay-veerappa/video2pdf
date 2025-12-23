# Prompts for Independent Experimentation

Use these prompts with **Claude 3.5 Sonnet**, **Gemini 1.5 Pro**, or **ChatGPT (GPT-4o)** to test the logic independently.

---

## 1. Crisp Technical Notes (Transcript Cleaning)
**Goal:** Remove banter/filler, fix grammar, and keep timestamps/technical details.

**System Prompt:**
> You are an expert technical editor for financial trading content.

**User Prompt:**
```text
Convert the following raw transcript segment into CRISP TECHNICAL NOTES.

STRICT INSTRUCTIONS:
1. REMOVE ALL BANTER, small talk, jokes, and introductory pleasantries (e.g., "Good morning," "How's the audio," "Crazy market right?").
2. EXTRACT the core technical points, trading rules, and specific market observations.
3. PRESERVE THE TIMESTAMPS [HH:MM:SS] for each significant point.
4. If a block of text is just conversation without technical value, DISCARD it completely.
5. Use bullet points for clarity.
6. Preserve all technical terms exactly (e.g., "Liquidity," "Order Block," "FVG," "MSS," "Standard Deviation").

TRANSCRIPT SEGMENT:
[PASTE YOUR TRANSCRIPT CHUNK HERE]

CRISP TECHNICAL NOTES:
```

---

## 2. Robust Crop Coordinate Extraction (Vision)
**Goal:** Identifying the crop box to remove UI toolbars from a trading chart screenshot.

**System Prompt:**
> You are a computer vision assistant specialized in UI detection.

**User Prompt:**
*(Attach the Screenshot Image)*

```text
Look at this trading chart. I need to crop it to keep only the main chart area.
Identify the normalized boundaries (0.0 to 1.0) to remove:
1. Top toolbar (usually contains URL or timeframe settings)
2. Left toolbar (drawing tools)
3. Right sidebar/watchlist
4. Bottom date axis (keep if relevant/readable, otherwise crop)

Provide the crop box in this EXACT format:
ymin: [number]
xmin: [number]
ymax: [number]
xmax: [number]

Do not explain. Just give the numbers.
```

---

## 3. Semantic Deduplication (Experimental)
**Goal:** Deciding if two similar slides are effectively the same content (Duplicate) or have meaningful changes (Unique).

**User Prompt:**
*(Attach Image 1 and Image 2)*

```text
Compare these two slides from a trading presentation.
Ignore minor compression artifacts or tiny mouse cursor movements.

Are these slides SEMANTICALLY DUPLICATES? 
- YES if the chart content, drawings, and text are identical.
- NO if a new drawing appeared, the price moved significantly, or the slide text changed.

Reply with JSON only:
{
  "is_duplicate": boolean,
  "confidence": number (0-1),
  "reason": "short explanation"
}
```
