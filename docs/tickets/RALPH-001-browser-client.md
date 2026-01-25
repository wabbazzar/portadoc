# [RALPH] Browser-Based Portadoc Client

> **Type:** Ralph (autonomous agent)
> **Status:** ready
> **Priority:** P2 (medium)
> **Created:** 2025-01-25
> **Updated:** 2025-01-25

## Objective

Build a fully client-side version of Portadoc that runs OCR entirely in the browser using Tesseract.js and docTR-TFJS.

## Context

- Portadoc currently runs server-side with Python OCR engines
- Users want offline/private document processing without server round-trips
- Mindee has already ported docTR to TensorFlow.js: https://github.com/mindee/doctr-tfjs-demo
- Tesseract.js is mature and widely used
- This enables: offline use, lower operational cost, mobile support, privacy-first workflows

---

## Ralph Setup Checklist

Before starting Ralph, ensure these files exist in the project root:

### Required Files

- [ ] **PROMPT.md** - Ralph's instructions for this ticket (see template below)
- [ ] **@fix_plan.md** - Task checklist with `[ ]` / `[x]` items
- [ ] **@AGENT.md** - Build, test, and run instructions
- [ ] **specs/browser-client.md** - Browser client specification

### Optional but Recommended

- [ ] **.claude/settings.json** - Sandbox permissions (allows uninterrupted work)
- [ ] **screenshots/** directory - For GUI verification (if using dev-browser)

---

## Scope

### In Scope
- New `src/portadoc/browser/` directory with TypeScript/JavaScript client
- PDF.js integration for PDF → image conversion
- Tesseract.js integration for Tesseract OCR
- docTR-TFJS integration for docTR OCR
- Harmonization logic ported to TypeScript
- Single-page web app that works offline
- Model preloading with progress indicator
- Export results as CSV/JSON

### Out of Scope
- Surya, EasyOCR, PaddleOCR, Kraken (no browser ports)
- Server-side components (this is client-only)
- React/Vue/framework - vanilla JS/TS for simplicity
- Mobile-specific optimizations (works but not optimized)
- Service worker / PWA features (future enhancement)

---

## Architecture

```
src/portadoc/browser/
├── index.html              # Main page
├── app.ts                  # Main application logic
├── pdf-loader.ts           # PDF.js wrapper
├── ocr/
│   ├── tesseract.ts        # Tesseract.js wrapper
│   └── doctr.ts            # docTR-TFJS wrapper
├── geometric-clustering.ts # Reading order algorithm (port of geometric_clustering.py)
├── harmonize.ts            # Multi-engine result fusion
├── models.ts               # TypeScript interfaces (Word, BBox, Cluster, etc.)
└── styles.css              # Styling
```

### Pipeline

```
PDF → PDF.js → Image (canvas)
                  ↓
          ┌──────┴──────┐
          ↓             ↓
     Tesseract.js   docTR-TFJS
          ↓             ↓
          └──────┬──────┘
                 ↓
          Harmonize (TS)
                 ↓
         Display + Export
```

---

## Task Breakdown (@fix_plan.md content)

Copy this to `@fix_plan.md`:

```markdown
# Fix Plan - Browser-Based Portadoc Client

## Phase 1: Project Setup
- [ ] Create `src/portadoc/browser/` directory structure
- [ ] Set up TypeScript build with esbuild or Vite
- [ ] Create `index.html` with basic layout
- [ ] Add npm dependencies: pdf.js, tesseract.js, @tensorflow/tfjs

## Phase 2: PDF Loading
- [ ] Implement `pdf-loader.ts` with PDF.js
- [ ] Convert PDF pages to canvas/ImageData
- [ ] Add page navigation UI
- [ ] Handle multi-page PDFs

## Phase 3: Tesseract.js Integration
- [ ] Implement `ocr/tesseract.ts` wrapper
- [ ] Load Tesseract worker and language data
- [ ] Extract words with bounding boxes
- [ ] Add progress callback for UI

## Phase 4: docTR-TFJS Integration
- [ ] Clone/adapt models from doctr-tfjs-demo
- [ ] Implement `ocr/doctr.ts` wrapper
- [ ] Load detection model (db_mobilenet_v2)
- [ ] Load recognition model (crnn_vgg16_bn)
- [ ] Extract words with bounding boxes
- [ ] Add progress callback for UI

## Phase 5: Reading Order and Harmonization
- [ ] Port `Word` and `BBox` interfaces to `models.ts`
- [ ] Port geometric clustering algorithm from `src/portadoc/geometric_clustering.py`:
  - [ ] Union-Find data structure
  - [ ] `calculate_distance_thresholds()` - Q1-based threshold calculation
  - [ ] `detect_column_boundaries()` - row-based gap analysis
  - [ ] `build_clusters()` - spatial proximity clustering
  - [ ] `sort_words_within_cluster()` - y-fuzz row grouping
  - [ ] `group_clusters_into_row_bands()` - cluster ordering
  - [ ] `order_words_by_reading()` - main entry point
- [ ] Port `find_word_match()` to TypeScript
- [ ] Port `harmonize_words()` to TypeScript
- [ ] Port deduplication logic

## Phase 6: UI and Export
- [ ] Display extracted words with bbox overlays
- [ ] Add word list panel (like existing web UI)
- [ ] Implement CSV export
- [ ] Implement JSON export
- [ ] Add model loading progress indicators

## Phase 7: Validation and Polish
- [ ] Benchmark: peter_lou.pdf with Tesseract.js only → verify ≥80% word match
- [ ] Benchmark: peter_lou.pdf with docTR-TFJS only → verify ≥80% word match
- [ ] Verify reading order is correct (top-to-bottom, left-to-right)
- [ ] Stress test: peter_lou_50dpi.pdf processes without crashing
- [ ] Add error handling for model load failures
- [ ] Screenshot validation with dev-browser

## Completed
(Ralph moves items here as they finish)

## Notes
- Models are large (~50-100MB), need loading indicator
- Tesseract.js needs language data (~10MB for eng)
- Use Web Workers for OCR to avoid blocking UI
- Test in Chrome first, then Firefox/Safari
- **Reading order is non-trivial**: Must port geometric clustering algorithm (~800 lines)
  - Union-Find clustering, column detection, row-band grouping
  - Reference: `src/portadoc/geometric_clustering.py`
  - Validate against: `data/input/peter_lou_words_slim.csv`
```

---

## PROMPT.md Template

Copy and customize for `PROMPT.md`:

```markdown
# Ralph Development Instructions - Browser Portadoc Client

## Context
You are Ralph, an autonomous AI agent working on **Portadoc**.
Portadoc extracts text with word-level bounding boxes from PDFs using multi-engine OCR.

## Session Objective
Build a fully client-side browser version using Tesseract.js and docTR-TFJS.

## CRITICAL: Environment Setup
**For TypeScript/browser work:**
```bash
cd src/portadoc/browser
npm install
npm run dev
```

## Test Data
- `data/input/peter_lou.pdf` - Clean 3-page test PDF
- `data/input/peter_lou_50dpi.pdf` - Degraded version
- `data/input/peter_lou_words_slim.csv` - Ground truth (401 words)

## Task Summary
Port Portadoc's OCR extraction to run entirely in the browser.

### The Problem
Current Portadoc requires a Python server. Users want offline/private processing.

### The Solution
Use existing browser ports of OCR engines:
- PDF.js for PDF rendering
- Tesseract.js for Tesseract OCR
- docTR-TFJS for docTR OCR
- Port harmonization logic to TypeScript

### Key Resources
- https://github.com/naptha/tesseract.js - Tesseract.js
- https://github.com/mindee/doctr-tfjs-demo - docTR browser demo
- `src/portadoc/geometric_clustering.py` - Reading order algorithm to port (~800 lines)
- `src/portadoc/harmonize.py` - Multi-engine fusion logic to port
- `data/input/peter_lou_words_slim.csv` - Ground truth with correct reading order

### Validation
Ground truth: `data/input/peter_lou_words_slim.csv` (401 words)

**Benchmark (peter_lou.pdf - clean)**:
- Tesseract.js alone: ≥80% word match (≥320 words)
- docTR-TFJS alone: ≥80% word match (≥320 words)
- Reading order: correct (first words: "7/24/25," "10:28" "AM")

**Stress test (peter_lou_50dpi.pdf - degraded)**:
- Must process without crashing
- Text accuracy not required to match clean doc

```bash
# Start dev server
cd src/portadoc/browser && npm run dev

# Use dev-browser to:
# 1. Upload peter_lou.pdf, run each engine separately
# 2. Export CSV, count matching words against ground truth
# 3. Upload peter_lou_50dpi.pdf, verify it completes
```

## Key Principles
- Complete tasks in order from @fix_plan.md
- Test after each phase
- Update @fix_plan.md with [x] when done
- Keep code simple - vanilla TS, no frameworks

## Status Reporting (CRITICAL)

At the end of EVERY response, include:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING | DEBUGGING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary of next action>
---END_RALPH_STATUS---
```

### EXIT_SIGNAL = true when:
1. All items in @fix_plan.md are marked [x]
2. peter_lou.pdf: Both Tesseract.js and docTR-TFJS achieve ≥80% word match
3. peter_lou.pdf: Reading order matches ground truth (word sequence comparison)
4. peter_lou_50dpi.pdf: Processes without crashing
```

---

## Exit Conditions

All must be true for Ralph to exit successfully:

- [ ] All @fix_plan.md items marked `[x]`
- [ ] Browser app loads at localhost without console errors

### Benchmark: peter_lou.pdf (clean document)
Ground truth: `data/input/peter_lou_words_slim.csv` (401 words across 3 pages)

- [ ] **Tesseract.js alone**: ≥80% word match against ground truth
- [ ] **docTR-TFJS alone**: ≥80% word match against ground truth
- [ ] **Reading order**: Word sequence matches ground truth CSV order (validated by comparing word_id sequences)
- [ ] **Bounding boxes**: Visually correct - boxes align with text in document

**Reading order note**: The ground truth CSV defines the correct reading order. The Python version uses a sophisticated geometric clustering algorithm (`src/portadoc/geometric_clustering.py`) that handles multi-column layouts, row-band detection, and cluster-based ordering. The browser version must port this logic to achieve correct reading order.

### Stress test: peter_lou_50dpi.pdf (degraded document)
- [ ] App loads and processes all 3 pages without crashing
- [ ] Words are extracted (count may be lower due to degradation)
- [ ] Bounding boxes are displayed (accuracy not required to match clean doc)

### Export
- [ ] CSV export produces valid file with columns: page, word_id, text, x0, y0, x1, y1, engine
- [ ] JSON export works

---

## Validation Protocol

### 1. Start dev server
```bash
cd src/portadoc/browser && npm run dev
```

### 2. Clean document benchmark (peter_lou.pdf)
```
Use dev-browser skill:
1. Navigate to http://localhost:5173
2. Upload data/input/peter_lou.pdf
3. Run Tesseract.js only → export CSV → count words matching ground truth
4. Run docTR-TFJS only → export CSV → count words matching ground truth
5. Verify reading order: first words should be "7/24/25," "10:28" "AM" (top-left)
6. Screenshot: screenshots/browser-clean-tesseract.png
7. Screenshot: screenshots/browser-clean-doctr.png
```

**Word match criteria**: Both engines ≥80% match (≥320 of 401 words)

**Reading order validation**: Export CSV, compare word sequence against ground truth.
The first 10 words in ground truth order are:
```
0: "7/24/25,"  1: "10:28"  2: "AM"  3: "Patient"  4: "Intake"
5: "Summary"   6: "-"      7: "Peter"  8: "Lou"  9: (pixel_detector)
10: "NORTHWEST"  11: "VETERINARY"  12: "ASSOCIATES" ...
```
Note: words 9, 29 are pixel_detector entries (non-text regions) - browser version may skip these.

### 3. Degraded document stress test (peter_lou_50dpi.pdf)
```
Use dev-browser skill:
1. Upload data/input/peter_lou_50dpi.pdf
2. Run both engines
3. Verify app doesn't crash, words are extracted
4. Screenshot: screenshots/browser-degraded.png
```

**Pass criteria**: App completes without errors, produces output

---

## GUI Testing (required)

This ticket heavily involves UI work. Ralph should use **dev-browser** for verification:

```
Use the dev-browser skill to:
1. Navigate to http://localhost:5173
2. Upload data/input/peter_lou_50dpi_page1.jpg (or full PDF)
3. Wait for OCR to complete
4. Take screenshot showing extracted words
5. Verify bounding boxes are displayed
Save screenshots to screenshots/browser-client-[step].png
```

---

## Dependencies

### NPM Packages
```json
{
  "dependencies": {
    "pdfjs-dist": "^4.0.0",
    "tesseract.js": "^5.0.0",
    "@tensorflow/tfjs": "^4.0.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "vite": "^5.0.0"
  }
}
```

### docTR Models (download from Mindee)
- `db_mobilenet_v2` - Text detection
- `crnn_vgg16_bn` - Text recognition

---

## Progress Log

Ralph updates this as work progresses:

```
[YYYY-MM-DD HH:MM] Loop 1: ...
```

---

## Results

Final metrics and outcomes (filled on completion):

### peter_lou.pdf (clean) - Ground truth: 401 words

| Engine | Words Found | Matches | Match % | Target | Pass? |
|--------|-------------|---------|---------|--------|-------|
| Tesseract.js | - | - | - | ≥80% | - |
| docTR-TFJS | - | - | - | ≥80% | - |

### peter_lou_50dpi.pdf (degraded)

| Engine | Words Found | Completed? | Errors? |
|--------|-------------|------------|---------|
| Tesseract.js | - | - | - |
| docTR-TFJS | - | - | - |

### Performance

| Metric | Target | Actual |
|--------|--------|--------|
| Initial model load (cold) | <60s | - |
| OCR time per page (clean) | <10s | - |
| OCR time per page (degraded) | <15s | - |

---

## Completion Notes

Summary of what was done and any follow-up tickets needed.
