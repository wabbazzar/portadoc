# [RALPH] Improve Harmonization for Degraded 50dpi PDFs

> **Type:** Ralph (autonomous agent)
> **Status:** draft
> **Priority:** P1 (high)
> **Created:** 2026-01-17
> **Updated:** 2026-01-17

## Objective

Improve harmonization pipeline to handle degraded (50dpi) PDF extractions by fixing overlapping bbox suppression, paddle concatenation splitting, and garbage text filtering.

## Context

Background information Ralph needs:
- The current harmonization works well for clean PDFs (99% F1 on peter_lou.pdf)
- For degraded PDFs (peter_lou_50dpi.pdf), quality drops significantly
- Several systematic issues cause false positives and incorrect word boundaries
- This ticket addresses post-OCR harmonization, not OCR engine quality itself

### Test Files
- `data/output/peter_lou_extracted.csv` - Gold standard (clean PDF extraction)
- `data/output/peter_lou_50dpi_extracted.csv` - Degraded extraction to improve
- `data/input/peter_lou_50dpi.pdf` - Source degraded PDF
- `data/input/peter_lou_words_slim.csv` - Ground truth for evaluation

---

## Ralph Setup Checklist

Before starting Ralph, ensure these files exist in the project root:

### Required Files

- [ ] **PROMPT.md** - Ralph's instructions for this ticket (see template below)
- [ ] **@fix_plan.md** - Task checklist with `[ ]` / `[x]` items
- [ ] **@AGENT.md** - Build, test, and run instructions
- [ ] **specs/** directory - Specifications Ralph references

### Optional but Recommended

- [ ] **.claude/settings.json** - Sandbox permissions (allows uninterrupted work)

---

## Issue Analysis

### Issue 1: Overlapping Pixel/Low-conf Noise

**50dpi rows 13-19** (header "Compassionate Care for Your Feline Friends"):
```csv
13,0,137.04,127.92,209.04,135.12,Compasaionaie,word,TEPD,77.5,...
14,0,213.12,127.92,234.96,135.12,Care,word,TED,95.8,...
15,0,230.62,125.76,275.39,139.2,"Iar""kur",pixel,E,2.3,...  # OVERLAPS Care/for/Your
16,0,239.04,127.92,250.08,135.12,tor,low_conf,TD,54.0,...
17,0,252.96,127.92,274.08,135.12,Your,word,TPD,77.5,...
```

**Gold standard** (clean PDF):
```csv
13,0,137.28,128.88,209.28,138.72,Compassionate,word,TEPD,98.5,...
14,0,213.12,128.88,234.96,136.8,Care,word,TEPD,99.3,...
15,0,238.32,128.88,250.56,136.56,for,word,TEPD,99.9,...
16,0,253.44,129.12,274.56,136.56,Your,word,TEPD,99.9,...
```

**Problem**: Row 15 "Iar"kur" (bbox 230.62-275.39) overlaps with rows 14 ("Care" ends at 234.96) and 17 ("Your" starts at 252.96). Low-conf garbage is not being suppressed when it overlaps legitimate words.

**Fix**: In harmonize.py Phase 2, add overlap suppression for low-conf secondary words that overlap multiple existing bboxes.

---

### Issue 2: Multi-line Document ID Mismerge

**50dpi row 22**:
```csv
22,0,498.0,82.08,539.04,99.12,INK-2025-,word,TEPD,68.9,ee,ImLIudz,INK-2025-,08347,...
```

**Gold standard** (two separate lines):
```csv
21,0,493.6,82.08,539.48,92.16,INK-2025-,secondary_only,E,99.8,,INK-2025-,,,
22,0,502.8,92.64,540.24,102.72,0923847,secondary_only,E,100.0,,0923847,,,
```

**Problem**: The paddle text "08347" is a fragment of "0923847" from the line below. The INK-2025- bbox (y0=82.08, y1=99.12) incorrectly spans two visual lines (gold has y1=92.16 for first line, y0=92.64 for second).

**Fix**: Detect when paddle text appears to be from a different vertical band (y-offset > threshold) and exclude it from voting.

---

### Issue 3: Paddle Concatenation Artifacts

**50dpi examples**:
```csv
9,0,137.04,79.92,252.0,96.0,NORTHWEST,word,TEPD,98.9,...,NORTHWESTVETERINARY  # paddle col
31,0,89.04,207.12,144.0,216.96,PATIENT,word,TEPD,99.9,...,PATIENTINFORMATION
37,0,132.0,275.04,162.96,283.92,Birth:,word,TEPD,97.8,...,ofBirth:
83,0,192.96,537.12,220.08,546.0,4827,word,TEPD,88.6,...,4827Maple
```

**Problem**: Paddle often concatenates adjacent words. When the paddle text is significantly longer than other engines, it should be split
**Fix**: Add paddle text length check - if paddle sub text is within the other words, add a space at the substring portion. adjust coordinates for what is before or after

---

### Issue 4: Microchip Number Fragmentation

**50dpi rows 65-70** (microchip "985141004729856"):
```csv
65,0,364.08,364.08,468.72,377.76,935141004729890,pixel,E,19.3,,935141004729890,,,
66,0,366.96,366.96,385.92,376.08,OBS,low_conf,T,36.0,OBS,,,
67,0,389.04,366.96,405.12,376.08,141,pixel,T,0,141,,,
68,0,409.44,366.0,423.84,380.16,OM,pixel,T,0,OM,,,
69,0,428.4,366.0,445.2,380.16,725,low_conf,T,36.0,725,,,
70,0,449.76,366.0,467.04,380.16,R66,low_conf,T,42.0,R66,,,
```

**Gold standard**:
```csv
60,0,367.68,367.92,466.8,376.8,985141004729856,word,TEPD,100.0,...
```

**Problem**: Tesseract fragments the microchip into garbage pieces. EasyOCR detects it correctly but as pixel (conf=19.3). These overlapping fragments should be merged.

**Fix**: When multiple low-conf/pixel entries from primary engine have bboxes that are sequentially adjacent (x1 of prev ≈ x0 of next) on same y-band, check if a secondary engine has a single detection covering the span. If so, prefer the secondary detection.

---

### Issue 5: Email Address Fragmentation

**50dpi rows 93-96** (email "r.martinez.pdx@gmail.com"):
```csv
93,0,382.08,537.12,522.96,549.12,Limartined,word,TEPD,81.4,...,r.maronez.pdxggrail.com
94,0,438.72,536.16,460.32,550.56,ado,low_conf,T,20.0,ado,,,
95,0,471.12,536.16,499.92,550.56,"oral,",pixel,TE,0,"oral,",pregtail,,
96,0,503.52,536.16,523.92,550.56,com,word,TE,81.0,com,cII,,
```

**Gold standard**:
```csv
80,0,382.08,537.84,522.96,549.36,r.martinez.pdx@gmail.com,word,TEPD,100.0,...
```

**Problem**: Email is fragmented into multiple garbage entries. The paddle text "r.maronez.pdxggrail.com" is close but row 93 winner is "Limartined".

**Fix**: the paddle text had the closest sensible answer. might use an english dictionary or levenstein distance to other owrds found in text eg maronez distance to martinez (found elsewhere). solution needs to scale well

---

### Issue 6: Header Complete Failure

**50dpi rows 0-8** (header "7/24/25, 10:28 AM ... Peter Lou"):
```csv
0,0,24.0,16.08,49.92,22.08,"vied,",pixel,TED,0,"vied,","3115,","21125,",
1,0,53.04,16.08,84.96,22.08,MA,pixel,TED,16.0,MA,Imz9x4,IRAM,
2,0,294.0,16.08,315.12,22.08,Veit,pixel,TD,0,Veit,,Pusinl,
```

**Gold standard**:
```csv
0,0,24.24,17.04,50.4,23.76,"7/24/25,",word,TEPD,99.9,...
1,0,53.28,17.04,70.08,22.8,10:28,word,TEPD,99.0,...
```

**Problem**: At 50dpi, all engines fail on the header text. This is an OCR quality issue, not harmonization.

**Note**: This may require preprocessing improvements or is out of scope for harmonization. Flag for documentation. probably find to just mark for pixel detection.

---

## Scope

### In Scope
1. Overlapping bbox suppression for low-conf secondaries (Issue 1)
2. Paddle concatenation detection and penalization (Issue 3)
3. Paddle vertical misalignment detection (Issue 2)
4. Adjacent fragment merging with secondary preference (Issue 4)
5. Levenstein distance to other sanitized words found in document when no clear winner is presented (Issue 5)

### Out of Scope
- OCR engine improvements (Issue 6 header failure)
- Image preprocessing changeis
- New OCR engine integration

---

## Task Breakdown (@fix_plan.md content)

Copy this to `@fix_plan.md`:

```markdown
# Fix Plan - Degraded Harmonization Improvements

## High Priority

### Issue 1: Overlapping Bbox Suppression
- [ ] Add `suppress_overlapping_secondaries()` function in harmonize.py
- [ ] Detect when a secondary-only word overlaps >1 existing primary word
- [ ] Suppress such words (don't add to result) if conf < 50
- [ ] Test: Row 15 "Iar"kur" should not appear in output
- [ ] Validation: `make eval-smart PDF=data/input/peter_lou_50dpi.pdf`

### Issue 3: Paddle Concatenation Penalty
- [ ] Add `is_paddle_concat()` helper to detect concatenated paddle text
- [ ] Check: paddle_text length > 1.5x winner_text length AND no spaces
- [ ] When detected, apply 0.1 weight multiplier to paddle vote
- [ ] Test: "NORTHWESTVETERINARY" should not affect "NORTHWEST" vote
- [ ] Validation: Row 9 text should be "NORTHWEST" not affected by paddle

### Issue 2: Paddle Vertical Misalignment
- [ ] Add `check_vertical_alignment()` in weighted_vote or find_word_match
- [ ] If paddle bbox y-band differs from primary y-band by >10px, exclude
- [ ] Test: Row 22 paddle "08347" should not merge with INK-2025-
- [ ] Validation: INK-2025- and 0923847 should be separate entries (like gold)

## Medium Priority

### Issue 4: Adjacent Fragment Merging
- [ ] Add `merge_adjacent_fragments()` post-processing step
- [ ] Detect sequences of low-conf/pixel entries on same y-band
- [ ] Check if secondary engine has covering detection with better conf
- [ ] If so, replace fragments with secondary detection
- [ ] Test: Microchip rows 65-70 should become single entry
- [ ] Validation: Microchip should match gold row 60

### Issue 5: Levenstein distance to other words found in document (or english dictionary)

- [ ] Document Issue 6 (header failure) as OCR limitation, not harmonization
- [ ] Update specs/portadoc.md with degraded PDF handling notes

## Completed
(Ralph moves items here as they finish)

## Notes
- Always run `source .venv/bin/activate` before Python commands
- Test on both peter_lou.pdf (should not regress) and peter_lou_50dpi.pdf
- Key file: src/portadoc/harmonize.py
```

---

## PROMPT.md Template

Copy and customize for `PROMPT.md`:

```markdown
# Ralph Development Instructions - Degraded Harmonization

## Context
You are Ralph, an autonomous AI agent working on **Portadoc**.
Portadoc extracts words from PDFs using multi-engine OCR with smart harmonization.

## Session Objective
Improve harmonization for degraded (50dpi) PDFs by fixing overlapping bbox suppression, paddle concatenation detection, and fragment merging.

## CRITICAL: Environment Setup
**ALWAYS activate venv before Python commands:**
\`\`\`bash
source .venv/bin/activate
\`\`\`

## Test Data
- `data/input/peter_lou.pdf` - Clean test PDF (should not regress)
- `data/input/peter_lou_50dpi.pdf` - Degraded test PDF (main target)
- `data/input/peter_lou_words_slim.csv` - Ground truth (401 words)
- `data/output/peter_lou_extracted.csv` - Gold standard extraction
- `data/output/peter_lou_50dpi_extracted.csv` - Current degraded extraction

## Task Summary
Fix harmonization issues identified in RALPH-001-degraded-harmonization.md

### The Problem
Degraded PDFs have several harmonization failures:
1. Overlapping garbage detections not suppressed
2. Paddle concatenates adjacent words incorrectly
3. Multi-line text merged incorrectly
4. Long numbers fragmented into garbage
5. Secondary OCR had best answer

### The Solution
Add detection and filtering in harmonize.py:
1. Suppress low-conf secondaries overlapping multiple primaries
2. Penalize paddle text when concatenated (len > 1.5x, no spaces)
3. Check vertical alignment before accepting paddle matches
4. Merge adjacent fragments when secondary has better detection
5. Levenstein distance to real words or words in document

### Key Files
- `src/portadoc/harmonize.py` - Main harmonization logic
- `src/portadoc/config.py` - Configuration classes
- `config/harmonize.yaml` - Thresholds and settings

### Validation
\`\`\`bash
# Test degraded PDF
make eval-smart PDF=data/input/peter_lou_50dpi.pdf

# Regression test on clean PDF
make eval-smart PDF=data/input/peter_lou.pdf
# Expected: F1 >= 99% (should not regress)
\`\`\`

## Key Principles
- Complete tasks in order from @fix_plan.md
- Test after each change on BOTH clean and degraded PDFs
- Update @fix_plan.md with [x] when done
- Keep changes minimal and focused
- Do not over-engineer

## Status Reporting (CRITICAL)

At the end of EVERY response, include:

\`\`\`
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING | DEBUGGING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary of next action>
---END_RALPH_STATUS---
\`\`\`

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
```

---

## Exit Conditions

All must be true for Ralph to exit successfully:

- [ ] All @fix_plan.md items marked `[x]`
- [ ] Validation: `make eval-smart PDF=data/input/peter_lou_50dpi.pdf` shows improvement
- [ ] No regression: `make eval-smart PDF=data/input/peter_lou.pdf` F1 >= 99%
- [ ] Specific tests pass:
  - Row 15 "Iar"kur" suppressed (Issue 1)
  - Row 9 "NORTHWEST" not affected by paddle concat (Issue 3)
  - INK-2025- and 0923847 separate (Issue 2)

---

## Validation Commands

```bash
# Primary validation - degraded PDF improvement
source .venv/bin/activate
make eval-smart PDF=data/input/peter_lou_50dpi.pdf

# Regression check - clean PDF
make eval-smart PDF=data/input/peter_lou.pdf
# Expected: F1 >= 99%

# Quick extraction test
make extract-smart PDF=data/input/peter_lou_50dpi.pdf
```

---

## Progress Log

Ralph updates this as work progresses:

```
[YYYY-MM-DD HH:MM] Loop 1: Started Issue 1, reading harmonize.py
[YYYY-MM-DD HH:MM] Loop 2: Implemented overlap suppression, testing...
```

---

## Results

Final metrics and outcomes (filled on completion):

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| 50dpi F1 | ~74% | TBD | Improve |
| Clean F1 | 99% | TBD | >= 99% |
| Overlap suppressions | 0 | TBD | Row 15 gone |

---

## Completion Notes

Summary of what was done and any follow-up tickets needed.
