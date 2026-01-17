# OCR Engine Benchmarks
**Date**: 2026-01-17

## Bug Fix: Multi-Engine Harmonization

**Problem**: Config file had `doctr.enabled: false` and `paddleocr.enabled: false`, causing CLI flags `--use-paddleocr` and `--use-doctr` to run the engines but the harmonizer would skip their output.

**Fix**: Updated `config/harmonize.yaml` to enable all engines.

---

## Benchmark Results

### Clean PDF (peter_lou.pdf) - 401 ground truth words

| Configuration        | Precision | Recall | F1 Score | Text Match |
|---------------------|-----------|--------|----------|------------|
| Tesseract only      | 99.00%    | 99.00% | **99.00%** | 98.74%   |
| Tess + EasyOCR      | 88.69%    | 99.75% | 93.90%   | 98.75%     |
| All 4 engines       | 82.47%    | 99.75% | 90.29%   | 98.25%     |

**Winner**: Tesseract-only (99% F1)

### Degraded PDF (peter_lou_50dpi.pdf)

| Configuration        | Precision | Recall | F1 Score | Text Match |
|---------------------|-----------|--------|----------|------------|
| Tesseract only      | 77.11%    | 86.53% | **81.55%** | 40.35%   |
| Tess + EasyOCR      | 68.76%    | 87.28% | 76.92%   | 40.29%     |
| All 4 engines       | 62.63%    | 87.78% | 73.10%   | **68.75%** |

**F1 Winner**: Tesseract-only (81.55%)
**Text Match Winner**: All 4 engines (68.75%)

---

## Key Findings

1. **More engines = lower precision** - Secondary engines add false positive detections
2. **Text accuracy improves with more engines** - 68.75% vs 40.35% on degraded docs
3. **Trade-off**: Bbox accuracy vs text accuracy

## Recommended Usage

| Use Case | Command |
|----------|---------|
| Redaction (bbox accuracy) | `--smart --no-easyocr --preprocess none --psm 6` |
| Text extraction (text accuracy) | `--smart --use-paddleocr --use-doctr` |

## Per-Engine Word Counts (Clean PDF, Page 0)

| Engine | Words |
|--------|-------|
| Tesseract | 90 |
| EasyOCR | 95 |
| PaddleOCR | 94 |
| docTR | 92 |

All engines now produce similar word counts when working correctly.
