# Portadoc Fix Plan

## High Priority (Core Pipeline) - DONE

- [x] Set up Python project structure with requirements.txt
- [x] Install Python dependencies (pymupdf, pytesseract, opencv-python-headless, numpy, click, easyocr)
- [x] Implement PDF to image conversion using PyMuPDF
- [x] Implement Tesseract OCR wrapper with word-level bounding boxes
- [x] Implement EasyOCR wrapper with word-level output
- [x] Build coordinate transformation (image coords → PDF coords)
- [x] Create basic CLI with `extract` command
- [x] Implement pixel detection fallback for OCR misses
- [x] Install tesseract-ocr binary (tesseract 5.3.4 verified working)

## Medium Priority (Quality Improvements) - DONE

- [x] Implement OpenCV preprocessing pipeline
- [x] Build OCR result harmonization logic (Tesseract + EasyOCR)
- [x] Add confidence-based triage system
- [x] Create evaluation metrics (recall, precision, IoU vs ground truth)

## Low Priority (Web Service & Polish) - DONE

- [x] Build FastAPI REST endpoints
- [x] Add async job processing for large PDFs
- [x] Implement JSON output format
- [x] Add progress reporting for CLI

---

## CURRENT FOCUS: Achieve 95%+ Accuracy on Degraded Documents

**Current Best (peter_lou_50dpi.pdf):** 81.45% F1, 40.11% text match
**Target:** 95%+ F1, 90%+ text match (comparable to AWS Textract's 99%)

### Phase 1: Implement PaddleOCR Engine (DONE)
- [x] Create `src/portadoc/ocr/paddleocr.py` wrapper
  - Use `from paddleocr import PaddleOCR`
  - Initialize with `use_angle_cls=True, lang='en', use_gpu=False`
  - Extract word-level boxes and text from result structure
  - Return list of Word objects matching existing interface
- [x] Add `is_paddleocr_available()` function
- [x] Add `--use-paddleocr` flag to CLI extract command
- [x] Test PaddleOCR standalone on degraded PDF, record metrics
- [x] Integrated into extractor.py with 3-engine harmonization

**PaddleOCR Results (degraded PDF, preprocess=none):**
| Config | Precision | Recall | F1 | Text Match |
|--------|-----------|--------|-----|------------|
| PaddleOCR only | 44.66% | 28.18% | 34.56% | 38.94% |
| Tess+Easy (baseline) | 76.54% | 87.03% | 81.45% | 40.11% |
| Tess+Easy+Paddle | 71.37% | 87.03% | 78.43% | 35.24% |

**KEY INSIGHT:** PaddleOCR underperforms on this degraded doc. Adding it to harmonization reduces F1 due to more false positives. The problem is image quality, not OCR engines - need super-resolution.

### Phase 2: Implement Image Super-Resolution (DONE)
- [x] Create `src/portadoc/superres.py` module
  - Implemented `upscale_image(image, scale, method)` function
  - Support methods: 'bicubic', 'lanczos', 'espcn', 'fsrcnn'
  - Uses OpenCV DNN: `cv2.dnn_superres.DnnSuperResImpl_create()` with ESPCN/FSRCNN models
  - Models downloaded to `models/` directory
- [x] Add `--upscale` and `--upscale-method` flags to CLI
- [x] Apply super-resolution BEFORE preprocessing and OCR
- [x] Tested on degraded PDF

**Super-Resolution Results (degraded PDF, preprocess=none, PSM=6, text_ths=0.95):**
| Config | Precision | Recall | F1 | Text Match |
|--------|-----------|--------|-----|------------|
| No upscale (baseline) | 76.54% | 87.03% | **81.45%** | **40.11%** |
| 2x ESPCN | 70.98% | 84.79% | 77.27% | 36.76% |
| 4x ESPCN | 68.63% | 81.30% | 74.43% | 38.96% |
| 4x Lanczos | 69.09% | 83.04% | 75.42% | 34.23% |

**KEY INSIGHT:** Super-resolution DOES NOT HELP on this degraded document. The baseline without upscaling performs best. The OCR engines (Tesseract+EasyOCR) may already have internal upscaling or the DNN models aren't suited for text. The fundamental problem is that this is a 50 DPI rasterized PDF - no amount of upscaling can recover information that isn't there.

### Phase 3: Implement docTR Engine (MEDIUM PRIORITY)
- [ ] Create `src/portadoc/ocr/doctr_ocr.py` wrapper
  - Use `from doctr.models import ocr_predictor`
  - Initialize with `pretrained=True`
  - Extract word-level boxes from document result
  - Return list of Word objects
- [ ] Add `is_doctr_available()` function
- [ ] Add `--use-doctr` flag to CLI
- [ ] Test docTR standalone on degraded PDF, record metrics

### Phase 4: Fix "degraded" Preprocessing Level
- [ ] In `preprocess.py`: implement the DEGRADED case in `preprocess_for_ocr()`
  - Currently defined in enum but not handled (falls through to else)
  - Should: upscale 2x with Lanczos → bilateral filter → CLAHE → unsharp mask
- [ ] Update `auto_detect_quality()` to return DEGRADED for very low quality images
  - Detect low DPI by checking image dimensions vs expected page size
  - If laplacian_var < 50, return DEGRADED instead of AGGRESSIVE

### Phase 5: Enhanced Multi-Engine Harmonization
- [ ] Update `harmonize.py` to support 3+ OCR engines
  - Current: takes tess_words, easy_words
  - New: take list of (engine_name, words) tuples
  - Implement voting across all engines for each word region
- [ ] Add weighted voting based on engine confidence and known accuracy
- [ ] Test 3-engine (Tesseract + EasyOCR + PaddleOCR) harmonization

### Validation Commands
```bash
# Activate venv first
source .venv/bin/activate

# Test single engine
portadoc eval data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv --use-paddleocr --no-tesseract --no-easyocr

# Test with super-resolution
portadoc eval data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv --upscale 4x

# Test full pipeline
portadoc eval data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv
```

### Success Criteria
- [ ] F1 Score > 95% on degraded PDF
- [ ] Text Match Rate > 90% on degraded PDF
- [ ] No regression on clean PDF (maintain ~96% F1)

---

## Historical Results

**Clean PDF (peter_lou.pdf):**
| Config | Precision | Recall | F1 | Words |
|--------|-----------|--------|-----|-------|
| No triage | 94.50% | 98.50% | 96.46% | 418 |

**Degraded PDF (peter_lou_50dpi.pdf):**
| Config | Precision | Recall | F1 | Text Match |
|--------|-----------|--------|-----|------------|
| auto (old default) | 51.61% | 55.86% | 53.65% | 22.77% |
| none, DPI=100 | 63.21% | 69.83% | 66.35% | 69.29% |
| none, PSM=6, DPI=300 | 70.42% | 87.28% | 77.95% | 40.29% |
| none, PSM=6, text_ths=0.95 | 76.54% | 87.03% | 81.45% | 40.11% |

**Key Insights from Previous Work:**
- `preprocess=none` outperforms all preprocessing on already-degraded images
- Lower DPI (100) improves text match but hurts F1
- Tesseract PSM 6 (block mode) works best
- EasyOCR text_threshold=0.95 reduces false positives significantly
- The core problem: 50 DPI source is too low for Tesseract/EasyOCR
- Solution: Real super-resolution + better OCR engines (PaddleOCR)

## Notes

- CPU-only constraint - no CUDA/GPU dependencies
- Virtual environment at `.venv/` - activate with `source .venv/bin/activate`
- Bounding boxes must be in PDF coordinate space (points, origin top-left)
- Target: 401 words across 3 pages matching ground truth
- AWS Textract achieves ~99% on this document - that's our benchmark
