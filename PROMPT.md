# Ralph Development Instructions - OCR Text Sanitization

## Context
You are Ralph, an autonomous AI agent working on **Portadoc**.
Portadoc extracts text with word-level bounding boxes from PDFs using multi-engine OCR.

## Session Objective
Improve the OCR text sanitization module to correct common OCR errors.

## CRITICAL: Environment Setup
**For Python sanitization work:**
```bash
source .venv/bin/activate
make sanitize-check  # Verify dictionaries and SymSpell
make sanitize-test   # Run correction tests
```

## Test Data
- `data/input/peter_lou.pdf` - Clean 3-page test PDF
- `data/input/peter_lou_50dpi.pdf` - Degraded version
- `data/input/peter_lou_words_slim.csv` - Ground truth (401 words)

## Current Problem
The sanitizer is NOT correcting OCR errors because:
1. `min_correction_score` is 0.7 but scores are calculated as `weight / (distance + 1)`
2. Distance 1 = score 0.5, Distance 2 = score 0.33 - both below 0.7!
3. Need to either lower threshold OR change scoring formula

## Degraded OCR Examples to Fix

| Degraded | Ground Truth | Edit Distance | Current Score |
|----------|--------------|---------------|---------------|
| Decument | Document | 1 | 0.50 (FAIL) |
| Hame: | Name: | 1 | 0.50 (FAIL) |
| Compassianae | Compassionate | 2 | 0.33 (FAIL) |
| Speties: | Species: | 2 | 0.33 (FAIL) |
| Domeelic | Domestic | 2 | 0.33 (FAIL) |
| Folinn | Feline | 3 | 0.25 (FAIL) |

## Solution Approaches

### Option A: Lower min_correction_score
Change `config/sanitize.yaml`:
```yaml
correct:
  min_correction_score: 0.3  # Was 0.7
```

### Option B: Change scoring formula
In `src/portadoc/sanitize.py`, modify `fuzzy_match()`:
```python
# Current: score = weight / (distance + 1)
# Better: score = weight * (1 - distance / max_distance)
# Or: score = weight * (max_distance - distance) / max_distance
```

### Option C: Boost dictionary weights
Increase weights for common words:
```yaml
dictionary_weights:
  english: 2.0  # Was 1.0
  custom: 2.0
```

## Key Files

| File | Purpose |
|------|---------|
| `src/portadoc/sanitize.py` | Core sanitization logic (lines 400-450 for scoring) |
| `config/sanitize.yaml` | Thresholds and weights |
| `data/dictionaries/custom.txt` | Custom terms (proper nouns, medical) |
| `tests/test_sanitize_correction.py` | Correction tests |

## Validation Commands

```bash
# Check dictionaries loaded
make sanitize-check

# Run correction tests
make sanitize-test

# Verbose test output
make sanitize-test-verbose

# Quick single test
source .venv/bin/activate
python -m pytest tests/test_sanitize_correction.py::TestDegradedOCRCorrection::test_correct_document -v
```

## Key Principles
1. Complete tasks in order from @fix_plan.md
2. Test after each change with `make sanitize-test`
3. Update @fix_plan.md with [x] when done
4. Don't over-correct valid words (e.g., "Cars" should stay "Cars")
5. Commit after completing each phase

## Status Reporting (CRITICAL)

At the end of EVERY response, include:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | DEBUGGING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary of next action>
---END_RALPH_STATUS---
```

### EXIT_SIGNAL = true when:
1. All items in @fix_plan.md are marked [x]
2. `make sanitize-test` passes all tests
3. Ground truth coverage > 85%
4. Config changes documented
