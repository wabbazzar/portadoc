# Ralph Development Tasks

## Session Date: 2026-01-17
## Status: ACTIVE

---

## TASK 1: Improve Harmonization Bbox Matching

### Problem
PaddleOCR and other secondary engines have slight bbox offsets that cause IoU to fall below threshold (0.3), resulting in duplicate detections instead of harmonization.

**Example** (word "Care"):
```
Tesseract: x0=213.12, y0=128.88, x1=234.96, y1=136.8  (area=173)
PaddleOCR: x0=203.6,  y0=126.24, x1=225.41, y1=138.72 (area=272)
IoU = 0.28 (below 0.3 threshold!)
```

Both detect "Care" correctly but with ~10px x-offset and ~2.5px y-offset.

### Requirements
1. **Learn noise tolerances** from test documents (`data/input/peter_lou_words_slim.csv`)
2. **Implement fuzzy matching** that considers:
   - Text similarity (same text should match even with bbox offset)
   - Center-point distance (if centers are close, allow lower IoU)
   - Aspect ratio similarity
3. **Auto-tune thresholds** based on ground truth
4. **Eliminate duplicate detections** like:
   ```
   20,0,...,Care,secondary_only,P   <- DUPLICATE
   25,0,...,Care,word,TED          <- CORRECT
   ```

### Acceptance Criteria
- [ ] No duplicate words in output when text matches across engines
- [ ] PaddleOCR matches correctly (source=TEPD for most words)
- [ ] F1 score maintains or improves
- [ ] Document learned tolerances in config

### Test Command
```bash
make eval-all PDF=data/input/peter_lou.pdf GROUND_TRUTH=data/input/peter_lou_words_slim.csv
```

---

## TASK 2: Web Application for PDF Visualization

### Overview
Build a web app that displays PDF pages with OCR bounding boxes overlaid, plus a reading-order word list with interactive highlighting.

### Technical Requirements

#### 2.1 Output File Standardization
- Output file name derived from input: `{input_stem}_extracted.csv`
- Example: `peter_lou.pdf` → `peter_lou_extracted.csv`
- Store in same directory as input or configurable output dir

#### 2.2 Backend (FastAPI)
- Extend existing `portadoc serve` command
- Endpoints:
  - `GET /api/pdfs` - List available PDFs
  - `GET /api/pdf/{name}/page/{num}` - Render page as image
  - `GET /api/pdf/{name}/words` - Get extracted words with coords
  - `POST /api/extract` - Run extraction with config options
  - `GET /api/config` - Get current config options
  - `POST /api/config` - Update config options

#### 2.3 Frontend (React or vanilla JS)
- **Left pane**: PDF page with bbox overlays
  - Draw colored boxes around detected words
  - Color-code by status: word (green), low_conf (yellow), pixel (red), secondary_only (orange)
  - Hover highlights word and shows tooltip with details

- **Right pane**: Word list in reading order
  - Table: word_id, text, source, confidence, status
  - Sortable/filterable
  - Hover highlights corresponding bbox in left pane

- **Config panel**:
  - Checkboxes: use_tesseract, use_easyocr, use_paddleocr, use_doctr
  - Dropdown: preprocess (none, light, standard, aggressive, auto)
  - Number inputs: psm, oem, iou_threshold
  - Button: "Re-extract" to run with new settings

#### 2.4 Interactive Features
- **Bidirectional hover**:
  - Hover word in list → highlight bbox on page
  - Hover bbox on page → highlight row in list
- **Click to select**: Click locks the highlight
- **Zoom**: Mouse wheel zoom on PDF pane
- **Page navigation**: Prev/Next buttons, page number input

### File Structure
```
src/portadoc/
├── web/
│   ├── __init__.py
│   ├── app.py          # FastAPI app with routes
│   ├── static/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   └── templates/      # Optional Jinja templates
```

### Acceptance Criteria
- [ ] Web app loads at `http://localhost:8000`
- [ ] PDF pages render with bboxes
- [ ] Word list shows in reading order
- [ ] Hover highlighting works bidirectionally
- [ ] Config panel allows re-extraction
- [ ] Output files use standardized naming

### Test Command
```bash
make serve-dev
# Open http://localhost:8000
```

---

## Configuration: Force All 4 Engines

All tasks must use all 4 OCR engines. Config already updated in `config/harmonize.yaml`:
```yaml
engines:
  easyocr:
    enabled: true
  doctr:
    enabled: true
  paddleocr:
    enabled: true
```

CLI: `--use-paddleocr --use-doctr`

---

## Ground Truth Reference

**File**: `data/input/peter_lou_words_slim.csv`
- 401 words total
- Columns: page, word_id, text, x0, y0, x1, y1, engine, ocr_confidence

---

## Current Metrics (Baseline)

| Metric | Tess-only | All 4 Engines |
|--------|-----------|---------------|
| F1 (clean) | 99.00% | 90.29% |
| F1 (degraded) | 81.55% | 73.10% |
| Text Match (degraded) | 40.35% | 68.75% |
| Duplicate detections | 0 | ~15 |

**Goal**: Reduce duplicates to 0 while maintaining text match improvement.

---

## Diagnostic Tools

- `tmp/diagnose_ocr.py` - Test each engine individually
- `tmp/debug_harmonize.py` - Trace harmonization matching
- `make benchmark` - Compare configurations
