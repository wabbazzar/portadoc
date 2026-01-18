# Ralph Development Instructions - Degraded Harmonization

## Context
You are Ralph, an autonomous AI agent working on **Portadoc**.
Portadoc extracts words from PDFs using multi-engine OCR with smart harmonization.

## Session Objective
Improve harmonization for degraded (50dpi) PDFs by fixing overlapping bbox suppression, paddle concatenation detection, and fragment merging.

## CRITICAL: Environment Setup
**ALWAYS activate venv before Python commands:**
```bash
source .venv/bin/activate
```

## Test Data
- `data/input/peter_lou.pdf` - Clean test PDF (should not regress)
- `data/input/peter_lou_50dpi.pdf` - Degraded test PDF (main target)
- `data/input/peter_lou_words_slim.csv` - Ground truth (401 words)
- `data/output/peter_lou_extracted.csv` - Gold standard extraction
- `data/output/peter_lou_50dpi_extracted.csv` - Current degraded extraction

## Task Summary
Fix harmonization issues identified in docs/tickets/RALPH-001-degraded-harmonization.md

### The Problem
Degraded PDFs have several harmonization failures:
1. Overlapping garbage detections not suppressed
2. Paddle concatenates adjacent words incorrectly
3. Multi-line text merged incorrectly
4. Long numbers fragmented into garbage
5. Secondary OCR had best answer but lost to garbage winner

### The Solution
Add detection and filtering in harmonize.py:
1. Suppress low-conf secondaries overlapping multiple primaries
2. Penalize paddle text when concatenated (len > 1.5x, no spaces)
3. Check vertical alignment before accepting paddle matches
4. Merge adjacent fragments when secondary has better detection
5. Use Levenshtein distance to other sanitized words found in document when no clear winner

### Key Files
- `src/portadoc/harmonize.py` - Main harmonization logic
- `src/portadoc/config.py` - Configuration classes
- `config/harmonize.yaml` - Thresholds and settings

### Validation
```bash
# Test degraded PDF
make eval-smart PDF=data/input/peter_lou_50dpi.pdf

# Regression test on clean PDF
make eval-smart PDF=data/input/peter_lou.pdf
# Expected: F1 >= 99% (should not regress)
```

## Key Principles
- Complete tasks in order from @fix_plan.md
- Test after each change on BOTH clean and degraded PDFs
- Update @fix_plan.md with [x] when done
- Keep changes minimal and focused
- Do not over-engineer

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
2. All validation commands pass
3. Clean PDF does not regress (F1 >= 99%)

## Quick Start
1. Read @fix_plan.md thoroughly
2. Read src/portadoc/harmonize.py to understand current logic
3. Start with first unchecked item (Issue 1)
4. Test after each change
5. Update @fix_plan.md with [x] when done
