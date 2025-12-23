# Task: Finalize Advanced Transcript Cleaning Integration

## Review & Assessment
- [x] Review recent changes in `app.py`, `main.py`, and `index.html`
- [x] Identify missing `/transcript_status` route
- [x] Identify unreachable code in `app.py`
- [x] Analyze `clean_transcript_background` status tracking gaps

## Bug Fixes & Implementation
- [x] Implement `/transcript_status/<video_id>` route in `app.py`
- [x] Add global `CLEANING_JOBS` state to track transcript cleaning tasks
- [x] Update `clean_transcript_background` and `run_clean` to track their progress
- [x] Rename transcripts from `transcript.txt` to `<video_name>.txt` (and similar for cleaned versions)
- [x] Ensure prioritize cleaned/Ollama transcripts in `prepare_slides_data` for editing phase
- [x] Add UI validation/popup for missing images/video when "Skip" is checked
- [x] Improve curation UI/UX in `curate_v2.html` (group actions, remove/restore, better status)
- [x] Clean up unreachable code in `app.py`

## Verification
- [ ] Run the app and verify the progress bar shows "Cleaning transcript..."
- [ ] Test Ollama cleaning with different styles (condensed, clean)
- [ ] Verify background threads don't block main process

## Strict Deduplication (Completed)
- [x] Add "Strict Deduplication" Option to UI
- [x] Implement backend logic for strict threshold (8 vs 12)

## Refactor Edit Workflow (Grouped Interface)
- [x] **Design & Backend**
    - [x] Create `prepare_grouped_data` in `app.py` (Group by 5min intervals)
    - [x] Create `/edit_v2/<video_id>` route
    - [x] Create `/save_grouped_data` route (Handle grouped text/images)
- [x] **Frontend (Edit V2)**
    - [x] Create `edit_v2.html` template
    - [x] Implement Split View (Text Left, Image Grid Right)
    - [x] Debugging empty page issue (Backend Data Mismatch Fixed)
    - [x] Add "Remove Image" functionality
- [x] Validate PDF/DOCX generation

## Phase 4: Enhanced Group Management
- [x] Implement "Add Group" functionality
    - [x] Add "Add Group" button to UI
    - [x] Update JS data model to support empty/new groups
- [x] Implement Drag-and-Drop for Images
    - [x] Make images draggable
    - [x] Make group containers drop targets
    - [x] Handle data model updates on drop
- [x] Implement Drag-and-Drop for GROUPS
    - [x] Add drag handle to group header
    - [x] Make group containers sortable
    - [x] Update group order in data model
- [ ] Update Save Logic
    - [ ] Ensure empty groups are handled or cleaned up
    - [ ] Persist custom group structure

## Phase 5: Project Cleanup
- [x] Create `archive/` directory
- [x] Move temporary files/scripts to `archive/` or `scripts/`
- [x] Move generic scripts to `scripts/`
- [x] Update `.gitignore` if necessary
- [x] Verify application still runs after moves

## Phase 6: Final Documentation & Removal
- [x] Create `docs/` directory
- [x] Move miscellaneous `.md` files to `docs/`
- [x] Delete "junk" from `archive/` (after confirmation)
- [x] Update `README.md` to reference `docs/`
- [x] Final verification of app functionality
- [x] **PDF Generation**
    - [x] Logic to flatten groups back to linear slides for PDF generation
    - [x] Verify PDF output quality

## Reference-Based Automation
- [x] **Global Crop Logic**
    - [x] Create `references/crop` directory mechanism
    - [x] Implement UI "Set Global Crop" and "Save as Default"
    - [x] Backend prioritized logic: Reference Image -> Saved Config -> Auto-Crop
    - [x] Support for **Drawn Rectangles** (Red/Green/Blue box) on reference images
- [x] **Visual Filtering (Ignore List)**
    - [x] Create `references/ignore` directory mechanism
    - [x] Implement Perceptual Hash matching to auto-discard ignored images
