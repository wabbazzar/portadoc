# [CLAUDE] VLM Fusion TDD - Cat in the Hat

**Type:** CLAUDE (interactive session)
**Status:** COMPLETE
**Created:** 2026-01-20
**Completed:** 2026-01-20
**Parent:** CLAUDE-002-vlm-bbox-fusion.md

## Objective

Test-driven development of the VLM+BBox fusion algorithm using:
- **Real PDF**: Generated "cat in the hat" test document
- **Real Tesseract**: Actual OCR output (no mocking)
- **Real geometric clustering**: Existing `geometric_clustering.py`
- **Real pixel detection**: Existing `detection.py`
- **Mocked VLM**: Text-only output (trivial to mock since VLMs just return text)

## Test Document

Create `data/input/cat_in_hat.pdf` with:

```
Line 1 (40pt): "The cat in the hat did a flip and a splat,"
Line 2 (20pt): "then he tipped his tall hat and said, 'How about that!'"
```

This gives us:
- Two font sizes (for char size clustering tests)
- Repeated words ("the", "hat") for context matching tests
- Punctuation for edge cases

## Test Scenarios

### Level 0: Perfect VLM
VLM output matches document exactly.
- Validates basic matching pipeline works

### Level 1: VLM Misspellings
```
VLM: "The cat in the hat did a flp and a splat, then he tipped his tall hat and said, 'How about that!'"
                              ^^^
```
- "flip" → "flp" (missing 'i')
- Tests fuzzy matching with Levenshtein

### Level 2: VLM Missing Words
```
VLM: "The cat in the hat did a flip a splat, then he tipped his tall hat and said, 'How about that!'"
                                  ^ missing "and"
```
- Tests orphan OCR word handling

### Level 3: VLM Extra Words (hallucination)
```
VLM: "The fat cat in the hat did a flip and a splat, then he tipped his tall hat and said, 'How about that!'"
         ^^^
```
- "fat" inserted (doesn't exist in OCR)
- Tests VLM orphan word placement

### Level 4: Tesseract Errors + Perfect VLM
Force Tesseract errors by using degraded image:
```
Tesseract: "The crt in thr hat did a flip and a splat..."
VLM:       "The cat in the hat did a flip and a splat..."
```
- Tests that VLM text replaces OCR text for matches

### Level 5: Both Have Errors (Different)
```
Tesseract: "The crt in thr hat..."
VLM:       "The cat in the hat did a flp..."
```
- VLM fixes "crt"→"cat", "thr"→"the"
- VLM introduces "flp" (but has bbox from Tesseract)

### Level 6: Multi-line Orphan
```
Tesseract: "The cat" ... "splat, then" (misses middle of line 1)
VLM:       Full text
```
- Tests orphan placement across detected pixel regions

## File Structure

```
tests/
└── test_vlm_fusion.py      # All TDD tests

src/portadoc/
└── ocr/
    └── vlm_fusion.py       # Implementation (not vlm_ocr.py yet - no API)

data/input/
└── cat_in_hat.pdf          # Test document
```

## Success Criteria

- [x] Test document created and renders correctly
- [x] Tesseract produces expected output on test doc
- [x] Level 0-6 tests written and passing (19 tests)
- [x] Context matching works within clusters
- [x] Character size estimation produces reasonable values
- [x] Orphan placement uses neighbor interpolation (pixel detection deferred)
- [x] All status values correctly assigned (vlm_matched, vlm_interpolated, ocr_only)

## Notes

- No VLM API calls in this phase - just mocked text strings
- Can test with real GOT later if available locally
- Focus on algorithm correctness, not API integration
