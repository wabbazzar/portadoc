# Sanitization OCR Correction Improvement

## Objective
Improve the OCR text sanitization to correct degraded OCR output. The sanitizer should correct common OCR errors (1-3 edit distance) while preserving valid words.

## Validation Command
```bash
make sanitize-test
```

## Success Criteria
- All tests in `tests/test_sanitize_correction.py` pass
- Ground truth coverage > 85%
- No over-correction of valid words (don't change "Cars" to something else)

## Test Data
- Ground truth: `data/input/peter_lou_words_slim.csv` (401 words)
- Degraded OCR input examples below

## Current Status
Run `make sanitize-test` to see pass/fail status.

---

## Tasks

### Phase 1: Dictionary Coverage
- [ ] Add missing proper nouns to `data/dictionaries/custom.txt`
- [ ] Add common words that might be OCR'd incorrectly to dictionaries
- [ ] Verify dictionaries load with `make sanitize-check`
- [ ] Ensure `Name`, `Owner`, `Species`, `Breed`, `Document` are in dictionaries

### Phase 2: Single-Character Edit Distance Corrections
These have edit distance 1 and should definitely correct:
- [ ] Fix `test_correct_document` - "Decument" -> "Document" (o->e)
- [ ] Fix `test_correct_name` - "Hame:" -> "Name:" (N->H)
- [ ] Fix `test_correct_martinez` - "Marlinez" -> "Martinez" (t->l)

### Phase 3: Two-Character Edit Distance Corrections
- [ ] Fix `test_correct_compassionate` - "Compassianae" -> "Compassionate" (dist 2)
- [ ] Fix `test_correct_species` - "Speties:" -> "Species:" (c->t, i deleted)
- [ ] Fix `test_correct_domestic` - "Domeelic" -> "Domestic" (st->el)
- [ ] Fix `test_correct_owner` - "Dmner:" -> "Owner:" (Ow->Dm)

### Phase 4: Higher Edit Distance (Optional - Context Mode)
These have edit distance 3+ and may require context correction:
- [ ] Fix `test_correct_feline` - "Folinn" -> "Feline" (edit dist 3)
- [ ] Fix `test_correct_january` - "lanuiry" -> "January" (edit dist 3)
- [ ] Fix `test_correct_shorthair` - "Shorifair" -> "Shorthair" (edit dist 3)
- [ ] Fix `test_correct_rebecca` - "Reteia" -> "Rebecca" (edit dist 4)

### Phase 5: Validation
- [ ] Run `make sanitize-test` - all deterministic tests pass
- [ ] Run `make sanitize-test-verbose` - verify correction output
- [ ] Verify `test_ground_truth_coverage` achieves >85%
- [ ] Document config changes made

---

## Files to Modify
- `src/portadoc/sanitize.py` - Core sanitization logic
- `config/sanitize.yaml` - Configuration thresholds
- `data/dictionaries/custom.txt` - Custom dictionary
- `data/dictionaries/medical_terms.txt` - Medical terms

## Degraded OCR Examples (from 50 DPI scan)

### Page 1 Header
```
Degraded: NORTHWEST VETERINARY . ASSOCIATES Compassianae Cars lor our Folinn Friends Decument ID: INa-2n25-

Ground Truth: NORTHWEST VETERINARY ASSOCIATES Compassionate Care for Your Feline Friends Document ID: INK-2025-
```

### Page 1 Patient Section
```
Degraded: PATIENT INFORMATION Hame: Peterlau Speties: Felina Domeelic Shorifair

Ground Truth: PATIENT INFORMATION Name: Peter Lou Species: Feline Domestic Shorthair
```

### Page 1 Owner Section
```
Degraded: Dmner: Reteia Marlinez Phone: (503) 555-0133

Ground Truth: Owner: Rebecca Martinez Phone: (503) 555-0123
```

## Algorithm Notes

The sanitizer uses SymSpell for O(1) fuzzy matching. Key thresholds:
- `correct.max_edit_distance`: Maximum Levenshtein distance (default: 2)
- `correct.min_correction_score`: Minimum score to apply correction (default: 0.7)
- `context.max_edit_distance`: Higher distance allowed with context (default: 3)

Scoring formula: `score = dictionary_weight / (edit_distance + 1)`

Example: "Document" in English dictionary (weight 1.0) matched with distance 1:
- Score = 1.0 / (1 + 1) = 0.5

If min_correction_score is 0.7, this would NOT correct by default!
Consider lowering min_correction_score or boosting dictionary weights.
