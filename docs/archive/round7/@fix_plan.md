# Portadoc Fix Plan - Round 7

## Session Goals
1. **Task 1:** Fix harmonization bbox matching to eliminate duplicate detections when using all 4 engines
2. **Task 2:** Build web application for PDF visualization with bounding box overlay

## Current Baseline (Clean PDF - peter_lou.pdf)
- Tesseract-only: **99.00% F1**, 98.74% text match
- All 4 engines: **90.29% F1** (due to ~15 duplicate detections)
- **Goal:** Eliminate duplicates, achieve 98%+ F1 with all engines

---

## TASK 1: Improve Harmonization Bbox Matching

### Problem Analysis
PaddleOCR and other secondary engines have slight bbox offsets (up to 14px) that cause IoU to fall below threshold (0.3), resulting in duplicate detections instead of harmonization.

**Example from output:**
```
Row 20: Care | secondary_only | P | bbox=(203.6, 126.24, 225.41, 138.72)
Row 25: Care | word | TED | bbox=(213.12, 128.88, 234.96, 136.8)
```
Both detect "Care" but PaddleOCR has ~10px x-offset, so it doesn't match Tesseract.

**Analysis findings (from tmp/analyze_bbox_noise.py):**
- PaddleOCR: up to 14px x-offset, IoU as low as 0.02 for valid matches
- IoU p5 threshold is 0.413 but we use 0.30
- Suggested: `text_match_bonus` of 0.15 to lower IoU when text matches

### Implementation Tasks

#### 1.1: Add text-aware matching to find_word_match()
- [x] In `src/portadoc/harmonize.py`, update `find_word_match()` to:
  - Accept `text_match_bonus` parameter (default 0.15)
  - When candidate text matches primary text (case-insensitive), lower the effective IoU threshold
  - Already partially implemented - needs to be USED by smart_harmonize
- [x] Test that identical text with low IoU now matches

#### 1.2: Update smart_harmonize() to use text-aware matching
- [x] In `smart_harmonize()`, pass `text_match_bonus` from config to `find_word_match()`
- [x] Add new config parameter: `harmonize.text_match_bonus: 0.15`
- [x] Ensure secondary engines use this when matching to primary

#### 1.3: Add center distance fallback matching
- [x] Implement `center_distance_max` config option (suggested: 12.0 points)
- [x] When IoU < threshold but center distance < max AND text matches: treat as match
- [x] This handles cases where bbox shapes differ but centers align

#### 1.4: Update config/harmonize.yaml
- [x] Add `text_match_bonus: 0.15` under `harmonize:`
- [x] Add `center_distance_max: 12.0` under `harmonize:`
- [x] Document new parameters with comments

#### 1.5: Validate fix eliminates duplicates
- [x] Run: `make eval-all PDF=data/input/peter_lou.pdf`
- [x] F1: 95.92% (improved from 90.29% - 62% reduction in false positives)
- [x] Run: `make eval-all PDF=data/input/peter_lou_50dpi.pdf`
- [x] F1: 74.11% (improved from 73.10% - no regression)

### Success Criteria - Task 1
- [~] All 4 engines: F1 = 95.92% (target was 98%, improved from 90.29%)
- [x] Zero duplicate "Care" words - FIXED! Now shows source=TEPD (all 4 engines matched)
- [x] No regression on degraded PDF (74.11% vs previous 73.10%)
- [x] Config parameters documented in harmonize.yaml

### Task 1 Results Summary
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Clean PDF F1 | 90.29% | 95.92% | +5.63% |
| Clean PDF FPs | 85 | 33 | -62% |
| Degraded PDF F1 | 73.10% | 74.11% | +1.01% |
| Duplicate "Care" | 2 entries | 1 entry (TEPD) | FIXED |

---

## TASK 2: Web Application for PDF Visualization

### Overview
Build a web app that visualizes PDF extraction results with bounding box overlays and bidirectional highlighting.

### File Structure
```
src/portadoc/web/
├── __init__.py          # Package init
├── app.py               # FastAPI routes
├── static/
│   ├── index.html       # Main page
│   ├── app.js           # PDF rendering and interaction
│   └── styles.css       # Styling
```

### Implementation Tasks

#### 2.1: Create web package structure
- [x] Create `src/portadoc/web/__init__.py`
- [x] Create `src/portadoc/web/app.py` with FastAPI router
- [x] Create `src/portadoc/web/static/` directory

#### 2.2: Implement FastAPI routes (app.py)
- [x] `GET /` - Serve index.html
- [x] `GET /api/pdfs` - List PDFs in data/input/ (and custom paths)
- [x] `GET /api/pdf/{filename}` - Serve PDF file for rendering
- [x] `GET /api/words/{filename}` - Get extracted words (from {stem}_extracted.csv or run extraction)
- [x] `POST /api/extract` - Run extraction with custom settings, save to {stem}_extracted.csv
- [x] Mount static files

#### 2.3: Build HTML interface (index.html)
- [x] PDF dropdown selector
- [x] Canvas or PDF.js for PDF rendering
- [x] Word list panel (sortable by word_id, page, status)
- [x] Config panel with all extraction settings
- [x] "Re-extract" button
- [x] Status bar showing extraction progress

#### 2.4: Implement PDF rendering (app.js)
- [x] Use PDF.js for PDF rendering
- [x] Draw bounding boxes on overlay canvas
- [x] Color by status:
  - `word` (green) - high confidence
  - `low_conf` (yellow) - uncertain
  - `pixel` (red) - fallback detection
  - `secondary_only` (orange) - only secondary engine detected

#### 2.5: Implement bidirectional hover highlighting
- [x] Hover word in list -> highlight bbox on canvas
- [x] Hover bbox on canvas -> highlight row in word list
- [x] Click to select and show word details

#### 2.6: Implement config panel
- [x] Checkboxes: use_tesseract, use_easyocr, use_paddleocr, use_doctr
- [x] Dropdown: preprocess (none, light, standard, aggressive, auto)
- [x] Number inputs: psm (0-13), oem (0-3)
- [x] "Apply & Re-extract" button
- [x] Show current extraction stats (total words, by status)

#### 2.7: Standardize output naming
- [x] Extraction output saved as: `{input_stem}_extracted.csv`
- [x] Example: `peter_lou.pdf` -> `peter_lou_extracted.csv`
- [x] Auto-detect existing extractions

#### 2.8: Extend serve command
- [x] Update `src/portadoc/cli.py` serve command to import and mount web app
- [x] Default to serving on http://localhost:8000

#### 2.9: Update Makefile
- [x] Add `serve-web` target for web visualization
- [x] Add target documentation

### Success Criteria - Task 2
- [x] `portadoc serve` creates web UI at http://localhost:8000
- [x] Can load peter_lou.pdf via API endpoint
- [x] Bidirectional hover highlighting implemented
- [x] Re-extraction with changed settings via POST /api/extract
- [x] Words list shows reading order with status colors

### Task 2 Files Created
- `src/portadoc/web/__init__.py` - Package init
- `src/portadoc/web/app.py` - FastAPI routes (247 lines)
- `src/portadoc/web/static/index.html` - HTML interface
- `src/portadoc/web/static/styles.css` - Dark theme styling
- `src/portadoc/web/static/app.js` - PDF.js rendering, hover highlighting (310 lines)

---

## Validation Commands

```bash
# Activate venv first
source .venv/bin/activate

# Task 1 validation
make eval-all PDF=data/input/peter_lou.pdf  # Should be >= 98% F1
make eval-all PDF=data/input/peter_lou_50dpi.pdf  # Should be >= 80% F1

# Task 2 validation
make serve-web  # Then open http://localhost:8000
```

## Notes

- Always use all 4 engines for testing Task 1: `--use-paddleocr --use-doctr`
- Ground truth: `data/input/peter_lou_words_slim.csv` (401 words)
- CPU-only constraint - no GPU/CUDA dependencies
- Update Makefile when adding new commands
