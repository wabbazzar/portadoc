# [CLAUDE] Kraken OCR Integration Preparation

> **Type:** Claude (interactive session)
> **Status:** ready
> **Priority:** P1 (high)
> **Created:** 2026-01-20
> **Depends On:** None
> **Produces:** PROMPT.md, @fix_plan.md, @AGENT.md for Ralph execution

## Objective

Prepare all files needed for Ralph to integrate Kraken OCR into Portadoc's multi-engine pipeline, with rigorous testing on the peter_lou test suite.

---

## Research Summary (Answers to Key Questions)

### Is Kraken Open Source for Commercial Use?

**YES** - Apache 2.0 license. Fully permissive for commercial use.

- Code: https://github.com/mittagessen/kraken
- Models: Hosted on Zenodo with various licenses (CC-BY-4.0, Apache 2.0)
- PyPI: `pip install kraken[pdf]`

### Which Model to Use?

For modern printed English documents (like peter_lou.pdf), the recommended models are:

| Model | Description | Best For |
|-------|-------------|----------|
| **CATMuS-Print** | Multi-language (EN, FR, DE, ES, etc), 16th-21st century | Modern printed text |
| **McCATMuS** | 7 languages, 16th-21st century, trained on 118k+ lines | General documents |
| **en_best** | English-specific if available | English-only docs |

**Recommendation:** Start with **CATMuS-Print** or **McCATMuS** since they cover modern printed English.

To list available models: `kraken list`
To download: `kraken get <model_doi>`

### Kraken's Bounding Box Capabilities

Kraken provides **word-level AND character-level** coordinates:

```python
from kraken import rpred
from kraken.lib import models

# Load model
model = models.load_any('path/to/model.mlmodel')

# Run recognition on line segmentation
for record in rpred.rpred(model, image, segmentation):
    record.prediction   # text string
    record.cuts         # character cut positions [(x0,x1), ...]
    record.confidences  # per-character confidence
    record.line         # line bounding box
    record.baseline     # baseline coordinates (if baseline seg)
```

**Word extraction strategy:**
1. Get character cuts from `record.cuts`
2. Find spaces in `record.prediction`
3. Aggregate character cuts between spaces into word bboxes

### Expected Performance on peter_lou Documents

| File | Description | Kraken Advantage |
|------|-------------|------------------|
| `peter_lou.pdf` | Clean document | Sanity check - should match Tesseract |
| `peter_lou_50dpi.pdf` | Low resolution | Kraken trained on degraded scans |
| `peter_hard.pdf` | Challenging layout | Trainable segmentation |
| `peter_hard_50dpi.pdf` | Low res + challenging | Full Kraken strength |

Kraken is specifically designed for historical/degraded documents. Its trainable segmentation and recognition models should excel on the degraded 50dpi variants.

---

## What This Session Must Produce

### 1. PROMPT.md (Ralph instructions)
Contains:
- Context about Portadoc and Kraken
- Clear objectives for the implementation
- Key files to modify
- Validation commands
- Status reporting format

### 2. @fix_plan.md (Task checklist)
Contains:
- Ordered, specific tasks
- Each task testable and atomic
- Clear exit criteria

### 3. @AGENT.md (Build/run instructions)
Contains:
- Venv activation
- Kraken installation
- Model download commands
- Test commands

### 4. specs/kraken-integration.md (Technical spec)
Contains:
- Python API patterns
- Word bbox extraction algorithm
- Harmonization integration points
- Config schema additions

---

## Implementation Architecture

### New File: `src/portadoc/ocr/kraken_ocr.py`

```python
# Pattern matches existing OCR modules
def extract_words_kraken(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    model_path: Optional[str] = None,
) -> list[Word]:
    """Extract words using Kraken OCR with word-level bboxes."""
    ...

def is_kraken_available() -> bool:
    """Check if Kraken is installed."""
    ...

def get_kraken_version() -> Optional[str]:
    """Get Kraken version string."""
    ...
```

### Config Additions (`config/harmonize.yaml`)

```yaml
harmonize:
  secondary:
    engines:
      kraken:
        enabled: false  # Off by default until tested
        weight: 1.0     # TBD based on testing
        garbage_penalty: 0.1
        bbox_type: word
        model: null     # Path to model file or DOI

ocr:
  kraken:
    model_path: null    # Default model path
    blla_model: null    # Segmentation model path (optional)
```

### Model Additions (`src/portadoc/models.py`)

Add to `HarmonizedWord`:
```python
kraken_text: str = ""
dist_kraken: int = -1
```

### Engine Code in harmonize.py

Add `"K"` to `ENGINE_CODES` mapping.

### Extractor Integration

Wire up `use_kraken` flag following existing patterns.

### CLI Updates (`src/portadoc/cli.py`)

Add `--use-kraken` / `--no-kraken` flags.

---

## Test Plan (CRITICAL)

### Baseline Metrics (Capture BEFORE Implementation)

Run on all 4 test files and record:

```bash
# Clean document baseline
./portadoc eval data/input/peter_lou.pdf data/input/peter_lou_words_slim.csv -v

# Degraded document baseline
./portadoc eval data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv -v

# Hard document baselines (if ground truth exists)
./portadoc extract data/input/peter_hard.pdf -o /tmp/peter_hard_baseline.csv
./portadoc extract data/input/peter_hard_50dpi.pdf -o /tmp/peter_hard_50dpi_baseline.csv
```

### Regression Tests (MUST NOT DEGRADE)

| File | Current F1 | Target | Notes |
|------|------------|--------|-------|
| peter_lou.pdf | 99.00% | >= 99.00% | Sanity check |
| peter_lou_50dpi.pdf | 81.55% | >= 81.55% | Should improve |

### Kraken-Specific Tests

1. **Word count consistency:** Kraken words extracted == expected
2. **Bbox precision:** Visual inspection on web UI
3. **Reading order:** Geometric clustering still works
4. **Harmonization:** Kraken votes integrate correctly

### Geometric Clustering Regression

The reading order must remain correct. Test by:
1. Extract with Kraken enabled
2. Verify word_id sequence matches expected reading order
3. Check multi-column documents cluster correctly

---

## Task Breakdown Preview

These will go in @fix_plan.md:

### Phase 1: Setup
- [ ] Install Kraken in venv: `pip install kraken[pdf]`
- [ ] Download CATMuS-Print or McCATMuS model
- [ ] Capture baseline F1 metrics on all 4 test files

### Phase 2: Core Implementation
- [ ] Create `src/portadoc/ocr/kraken_ocr.py` with `extract_words_kraken()`
- [ ] Implement word-level bbox extraction from character cuts
- [ ] Add `is_kraken_available()` and `get_kraken_version()`
- [ ] Add Kraken to `HarmonizedWord` (kraken_text, dist_kraken)
- [ ] Add "K" to ENGINE_CODES in harmonize.py

### Phase 3: Integration
- [ ] Add Kraken config section to `config/harmonize.yaml`
- [ ] Wire Kraken into `extractor.py` (use_kraken flag)
- [ ] Update CLI with --use-kraken / --no-kraken flags
- [ ] Update `make check` to detect Kraken availability

### Phase 4: Validation
- [ ] Run eval on peter_lou.pdf - F1 >= 99.00%
- [ ] Run eval on peter_lou_50dpi.pdf - F1 >= 81.55%
- [ ] Extract peter_hard.pdf - verify reasonable output
- [ ] Extract peter_hard_50dpi.pdf - verify reasonable output
- [ ] Visual inspection in web UI - bboxes look correct
- [ ] Reading order check - geometric clustering works

### Phase 5: Documentation
- [ ] Update CLAUDE.md with Kraken in engine table
- [ ] Add Kraken to Makefile help

---

## Exit Criteria for Ralph

All of the following must be true:

1. **All @fix_plan.md items marked `[x]`**
2. **F1 on peter_lou.pdf >= 99.00%** (no regression)
3. **F1 on peter_lou_50dpi.pdf >= 81.55%** (no regression)
4. **Kraken extracts words from all 4 test files without errors**
5. **Reading order (geometric clustering) works correctly**
6. **Web UI displays Kraken results correctly**

---

## Session Instructions

When running this Claude session:

1. **First:** Read the current codebase files listed above
2. **Then:** Create PROMPT.md in project root
3. **Then:** Create @fix_plan.md with detailed tasks
4. **Then:** Create @AGENT.md with build instructions
5. **Then:** Create specs/kraken-integration.md with technical details
6. **Finally:** User runs Ralph to execute

---

## Files to Create

| File | Location | Purpose |
|------|----------|---------|
| PROMPT.md | project root | Ralph instructions |
| @fix_plan.md | project root | Task checklist |
| @AGENT.md | project root | Build/run commands |
| specs/kraken-integration.md | specs/ | Technical specification |

---

## Reference: Current OCR Module Pattern

From `src/portadoc/ocr/tesseract.py`:

```python
def extract_words_tesseract(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    lang: str = "eng",
    config: str = "--psm 3 --oem 3"
) -> list[Word]:
    # 1. Get image dimensions
    # 2. Run OCR to get word-level data
    # 3. Convert image coords to PDF coords
    # 4. Create Word objects with BBox
    # 5. Return list

def is_tesseract_available() -> bool:
    ...

def get_tesseract_version() -> Optional[str]:
    ...
```

Kraken implementation must follow this exact pattern.

---

## Notes

- Kraken is CPU-friendly but can use GPU if available
- Model download is one-time; cache in ~/.kraken or project directory
- Kraken's segmentation is trainable - future improvement opportunity
- The hardest part is word bbox extraction from character cuts
