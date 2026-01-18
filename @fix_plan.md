# Fix Plan - Degraded Harmonization Improvements

## High Priority

### Issue 1: Overlapping Bbox Suppression
- [x] Add `suppress_overlapping_secondaries()` function in harmonize.py
- [x] Detect when a secondary-only word overlaps >1 existing primary word
- [x] Suppress such words (don't add to result) if conf < 50
- [x] Test: Row 15 "Iar"kur" should not appear in output
- [x] Validation: `make eval-smart PDF=data/input/peter_lou_50dpi.pdf`

### Issue 3: Paddle Concatenation Penalty
- [x] Add `is_paddle_concat()` helper to detect concatenated paddle text
- [x] Check: paddle_text length > 1.5x winner_text length AND no spaces
- [x] When detected, apply 0.1 weight multiplier to paddle vote
- [x] Test: "NORTHWESTVETERINARY" should not affect "NORTHWEST" vote
- [x] Validation: Row 9 text should be "NORTHWEST" not affected by paddle

### Issue 2: Paddle Vertical Misalignment
- [x] Add `check_vertical_alignment()` in weighted_vote or find_word_match
- [x] If paddle bbox y-band differs from primary y-band by >10px, exclude
- [x] Test: Row 22 paddle "08347" does not affect INK-2025- vote (correctly handled)
- [x] Note: 0923847 missing is a detection issue, not harmonization (Tesseract bbox spans 2 lines)

## Medium Priority

### Issue 4: Adjacent Fragment Merging
- [x] Add `merge_adjacent_fragments()` post-processing step
- [x] Detect sequences of low-conf/pixel entries on same y-band
- [x] Check if secondary engine has covering detection with better conf
- [x] If so, replace fragments with secondary detection
- [x] Test: Microchip rows 65-70 removed (fragments), row 64 kept (paddle detection)
- [x] Validation: F1 improved from 77.57% to 78.00%, 5 FPs removed

### Issue 5: Levenshtein Distance for Unclear Winners
- [ ] When no clear winner, compare candidates to other sanitized words in document
- [ ] Use Levenshtein distance to find closest match to known-good text
- [ ] Apply this for email/name fragments (e.g., "maronez" -> "martinez")
- [ ] Test: Email "r.martinez.pdx@gmail.com" should be correctly assembled
- [ ] Validation: Row 93 should have correct email text

## Documentation

- [ ] Document Issue 6 (header failure) as OCR limitation, not harmonization
- [ ] Update specs/portadoc.md with degraded PDF handling notes

## Completed
(Ralph moves items here as they finish)

## Notes
- Always run `source .venv/bin/activate` before Python commands
- Test on both peter_lou.pdf (should not regress) and peter_lou_50dpi.pdf
- Key file: src/portadoc/harmonize.py
