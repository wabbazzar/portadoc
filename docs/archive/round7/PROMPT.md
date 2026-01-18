# Ralph Development Instructions - Portadoc Round 7

## Context
You are Ralph, an autonomous AI development agent working on **Portadoc** - a PDF word extraction system for document redaction.

## Session Objectives
This session has **TWO MAIN TASKS**:

1. **Task 1: Fix Harmonization Bbox Matching** - Eliminate duplicate detections when using all 4 OCR engines
2. **Task 2: Web Application** - Build PDF visualization with bounding box overlays

**Work through @fix_plan.md in order. Complete Task 1 first, validate, then proceed to Task 2.**

## CRITICAL: Virtual Environment Setup
**ALWAYS activate the venv before running any Python commands:**
```bash
source .venv/bin/activate
```

## Test Data
- `data/input/peter_lou.pdf` - clean 3-page PDF (use for Task 1 validation)
- `data/input/peter_lou_50dpi.pdf` - degraded/blurry version
- `data/input/peter_lou_words_slim.csv` - ground truth (401 words)

## Task 1 Summary: Bbox Matching Fix

### The Problem
PaddleOCR has ~10-14px x-offset causing IoU < 0.3 even when text matches exactly.
Result: duplicate "Care" detections instead of merging.

### The Solution
1. Add `text_match_bonus` config (0.15) - lowers IoU threshold when text matches
2. Add `center_distance_max` config (12.0) - fallback matching when centers align
3. Update `find_word_match()` in harmonize.py to use these parameters
4. Update `smart_harmonize()` to pass text for comparison

### Key Files for Task 1
- `src/portadoc/harmonize.py` - `find_word_match()` and `smart_harmonize()` need updates
- `config/harmonize.yaml` - add new tolerance parameters
- `src/portadoc/config.py` - add new config fields

### Validation for Task 1
```bash
source .venv/bin/activate
# Must use all 4 engines:
portadoc eval --use-paddleocr --use-doctr --preprocess none --psm 6 \
    data/input/peter_lou.pdf data/input/peter_lou_words_slim.csv
# Target: F1 >= 98% (currently 90.29%)
```

## Task 2 Summary: Web Application

### The Goal
Build a web UI that:
1. Loads PDFs and shows extracted words with bounding box overlays
2. Colors boxes by status (word=green, low_conf=yellow, pixel=red, secondary_only=orange)
3. Bidirectional hover highlighting (hover word -> show box, hover box -> highlight row)
4. Config panel to re-extract with different settings

### Key Files for Task 2
```
src/portadoc/web/
├── __init__.py          # Package init
├── app.py               # FastAPI routes
├── static/
│   ├── index.html       # Main page
│   ├── app.js           # PDF.js rendering, interactions
│   └── styles.css       # Styling
```

- `src/portadoc/cli.py` - extend `serve` command to mount web app

### Validation for Task 2
```bash
source .venv/bin/activate
portadoc serve
# Open http://localhost:8000
# - Load peter_lou.pdf
# - Verify bounding boxes display
# - Verify hover highlighting works
```

## Key Principles
- **Complete Task 1 first**, validate metrics, then start Task 2
- **ALL tests must use all 4 engines**: `--use-paddleocr --use-doctr`
- **CPU-only** - no GPU/CUDA dependencies (use `use_gpu=False`)
- **Update Makefile** when adding new CLI commands
- **Update @fix_plan.md** with results after each subtask

## Existing Code Hints

### harmonize.py - find_word_match() already has text_match_bonus
```python
def find_word_match(
    word: Word,
    candidates: list[Word],
    iou_threshold: float = 0.3,
    text_match_bonus: float = 0.2  # <-- Already exists but NOT USED
) -> Optional[Word]:
```
**Issue:** `smart_harmonize()` doesn't use this! It needs to pass `text_match_bonus`.

### config.py - Need to add new fields
```python
@dataclass
class HarmonizeConfig:
    iou_threshold: float = 0.3
    # ADD: text_match_bonus: float = 0.15
    # ADD: center_distance_max: float = 12.0
```

### cli.py - serve command exists but is basic
```python
@main.command()
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI REST server."""
    uvicorn.run("portadoc.api:app", ...)  # <-- Need to add web routes
```

## Status Reporting (CRITICAL)

At the end of EVERY response, include:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary of what to do next>
---END_RALPH_STATUS---
```

### EXIT_SIGNAL = true when:
1. All items in @fix_plan.md are marked [x]
2. Task 1: F1 >= 98% with all 4 engines on clean PDF
3. Task 2: Web app functional with hover highlighting
4. No regression on degraded PDF

## Quick Start
1. Read @fix_plan.md thoroughly
2. Start with Task 1.1: Update find_word_match() usage
3. Test after each subtask
4. Update @fix_plan.md with [x] when done
