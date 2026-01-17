# Portadoc Fix Plan

## High Priority (Core Pipeline)

- [x] Set up Python project structure with requirements.txt
- [x] Install Python dependencies (pymupdf, pytesseract, opencv-python-headless, numpy, click, easyocr)
- [x] Implement PDF to image conversion using PyMuPDF
- [x] Implement Tesseract OCR wrapper with word-level bounding boxes
- [x] Implement EasyOCR wrapper with word-level output
- [x] Build coordinate transformation (image coords → PDF coords)
- [x] Create basic CLI with `extract` command
- [x] Implement pixel detection fallback for OCR misses
- [x] Install tesseract-ocr binary (tesseract 5.3.4 verified working)

## Medium Priority (Quality Improvements)

- [x] Implement OpenCV preprocessing pipeline
  - Implemented in preprocess.py: grayscale, denoise, CLAHE contrast, sharpen, binarize
  - Levels: none, light, standard, aggressive, auto
  - Auto-detection uses Laplacian variance + contrast std
- [x] Build OCR result harmonization logic (Tesseract + EasyOCR)
  - Implemented in harmonize.py: matches by IoU, votes on text, merges confidences
- [x] Add confidence-based triage system
  - Implemented in triage.py: strict, normal, permissive levels
  - Filters by: min confidence, text length, aspect ratio, area, punctuation
  - CLI: --triage flag (strict|normal|permissive)
  - Results: no triage=418, permissive=415, normal=407, strict=402 (GT=401)
- [x] Create evaluation metrics (recall, precision, IoU vs ground truth)
  - Implemented in metrics.py: precision, recall, F1, mean IoU, text match rate
  - CLI: `eval` command compares extraction to ground truth CSV
  - Results (no triage): 98.50% recall, 94.50% precision, 96.46% F1

## Low Priority (Web Service & Polish) - DONE

- [x] Build FastAPI REST endpoints
- [x] Add async job processing for large PDFs
- [x] Implement JSON output format
- [x] Add progress reporting for CLI

---

## CURRENT FOCUS: Degraded Document Performance

**Baseline (peter_lou_50dpi.pdf):** 51.61% precision, 55.86% recall, 53.65% F1, 22.77% text match
**Target:** Match clean PDF performance (~95% F1, ~90% text match)

### Key Files
- `src/portadoc/cli.py` - CLI commands (extract, eval, serve)
- `src/portadoc/preprocess.py` - OpenCV preprocessing (exists but not wired to CLI)
- `src/portadoc/extractor.py` - Main extraction pipeline
- `src/portadoc/ocr/tesseract.py` - Tesseract wrapper
- `src/portadoc/ocr/easyocr.py` - EasyOCR wrapper
- `src/portadoc/harmonize.py` - Multi-engine result fusion
- `src/portadoc/metrics.py` - Evaluation metrics

### Phase 1: Wire Up Preprocessing to CLI - DONE
- [x] Add `--preprocess` flag to `extract` command in cli.py (none|light|standard|aggressive|auto)
- [x] Add `--preprocess` flag to `eval` command in cli.py
- [x] Test all preprocess levels on degraded PDF, record metrics in this file
- [x] Verify `--dpi` flag works for upscaling (already exists, test if it helps degraded docs)

**Preprocessing Results (degraded 50dpi PDF, DPI=300):**
| Preprocess | Precision | Recall | F1 | Text Match |
|------------|-----------|--------|-----|------------|
| none | **70.39%** | 84.79% | **76.92%** | **38.53%** |
| light | 68.36% | **87.28%** | 76.67% | 33.43% |
| standard | 66.03% | 86.28% | 74.81% | 30.92% |
| aggressive | 30.21% | 21.70% | 25.25% | 3.45% |
| auto | 51.61% | 55.86% | 53.65% | 22.77% |

**KEY INSIGHT:** `preprocess=none` performs BEST on degraded docs. Auto-detection incorrectly applies aggressive preprocessing.

**DPI Scaling Results (preprocess=none):**
| DPI | Precision | Recall | F1 | Text Match |
|-----|-----------|--------|-----|------------|
| 72 | 59.10% | 59.10% | 59.10% | 58.65% |
| 100 | 63.21% | 69.83% | 66.35% | **69.29%** |
| 150 | 68.48% | 81.80% | 74.55% | 50.30% |
| 300 | 70.39% | 84.79% | 76.92% | 38.53% |
| 600 | 63.79% | 83.04% | 72.16% | 34.53% |

**KEY INSIGHT:** Lower DPI (100-150) improves text match rate. DPI 100 + preprocess=none achieves **69.29% text match** (best so far)

### Phase 2: Tune Preprocessing for Degraded Docs
- [ ] In preprocess.py: experiment with CLAHE clip limit (currently 2.0, try 1.0-4.0)
- [ ] In preprocess.py: test denoise h parameter (currently 10, try 5-30)
- [ ] In preprocess.py: try adaptive thresholding vs Otsu for binarization
- [ ] Add cv2.resize upscaling (INTER_CUBIC or INTER_LANCZOS4) before OCR
- [ ] Test different sharpening kernel strengths

### Phase 3: OCR Engine Tuning - PARTIAL
- [x] Add `--psm` and `--oem` CLI flags, pass to tesseract wrapper
- [x] In tesseract.py: test PSM modes 6 (block), 11 (sparse), 12 (sparse + OSD)
- [x] In tesseract.py: test OEM 0 (legacy), 1 (LSTM), 2 (combined)
  - OEM 0/2 require legacy traineddata (not installed), only OEM 1/3 work
- [ ] In easyocr.py: test decoder='beamsearch' (currently 'greedy')
- [ ] In easyocr.py: tune contrast_ths (default 0.1), adjust_contrast (default True)
- [ ] In easyocr.py: tune text_threshold (default 0.7), width_ths (default 0.5)

**PSM Mode Results (preprocess=none, DPI=300):**
| PSM | Precision | Recall | F1 | Text Match |
|-----|-----------|--------|-----|------------|
| 3 (default) | 70.39% | 84.79% | 76.92% | 38.53% |
| **6 (block)** | 70.42% | 87.28% | **77.95%** | 40.29% |
| 11 (sparse) | 67.51% | 86.53% | 75.85% | 39.77% |
| 12 (sparse+OSD) | 66.60% | 86.03% | 75.08% | 39.42% |

**Best PSM 6 + DPI Combinations:**
| Config | Precision | Recall | F1 | Text Match |
|--------|-----------|--------|-----|------------|
| PSM 6, DPI 300 | 70.42% | 87.28% | **77.95%** | 40.29% |
| PSM 6, DPI 100 | 66.44% | 74.56% | 70.27% | 66.56% |
| PSM 3, DPI 100 | 63.21% | 69.83% | 66.35% | **69.29%** |

**KEY INSIGHT:** PSM 6 gives best F1 (77.95%) at DPI 300. For text accuracy, lower DPI (100) still wins.

### Phase 4: Alternative OCR Engines
- [ ] Install PaddleOCR: `pip install paddlepaddle paddleocr`
  - Create src/portadoc/ocr/paddleocr.py wrapper
  - PaddleOCR is known for excellent degraded doc handling
- [ ] Install docTR: `pip install python-doctr`
  - Create src/portadoc/ocr/doctr.py wrapper
- [ ] Benchmark each engine standalone on degraded PDF
- [ ] Add best performer to harmonize.py pipeline

### Phase 5: Bounding Box Accuracy
- [ ] In extractor.py: improve coord scaling when source DPI differs from render DPI
- [ ] Add sub-pixel bbox interpolation for low-res → high-res scaling
- [ ] In metrics.py: test IoU thresholds (currently 0.5, try 0.3-0.7) for degraded matching

### Validation Command
After each change, run:
```bash
PYTHONPATH=src python3 -m portadoc.cli eval data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv
```

### Success Criteria
- [ ] F1 Score > 80% on degraded PDF
- [ ] Text Match Rate > 60% on degraded PDF
- [ ] No regression on clean PDF (maintain ~96% F1)

## Notes

- CPU-only constraint - no CUDA/GPU dependencies
- Bounding boxes must be in PDF coordinate space (points, origin top-left)
- Target: 401 words across 3 pages matching ground truth

## Baseline Results

**Clean PDF (peter_lou.pdf):**
| Config | Precision | Recall | F1 | Words |
|--------|-----------|--------|-----|-------|
| No triage | 94.50% | 98.50% | 96.46% | 418 |

**Degraded PDF (peter_lou_50dpi.pdf):**
| Config | Precision | Recall | F1 | Text Match |
|--------|-----------|--------|-----|------------|
| auto (old default) | 51.61% | 55.86% | 53.65% | 22.77% |
| none, DPI=100 | 63.21% | 69.83% | 66.35% | **69.29%** |
| **none, PSM=6, DPI=300** | 70.42% | 87.28% | **77.95%** | 40.29% |
