---
description: Process next TCM transcript and create study notes
---

# TCM Transcript Notes Workflow

This workflow processes one transcript at a time and creates detailed study notes.

## Steps:

1. **Find the next transcript**

```bash
// turbo
.venv\Scripts\python find_next_transcript.py
```

2. **Read the transcript** - The agent will automatically read the next transcript file

3. **Generate notes** - The agent will create detailed study notes following the TCM format

4. **Save the notes** - The agent will save the notes to the correct location

5. **Repeat** - Run this workflow again to process the next transcript

## Notes:

- This processes ONE transcript per run
- You can switch models between runs if you hit token limits
- Progress: Check with `python find_next_transcript.py`
- Total remaining: 92 transcripts
