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

## Low Priority (Web Service & Polish)

- [x] Build FastAPI REST endpoints
  - Implemented in api.py: /health, /extract, / endpoints
  - POST /extract accepts PDF upload, returns JSON with words and bounding boxes
  - Supports dpi, triage, preprocess query parameters
  - CLI: `portadoc serve --host 0.0.0.0 --port 8000`
- [x] Add async job processing for large PDFs
  - POST /jobs submits PDF for background processing
  - GET /jobs/{job_id} polls for status and result
  - GET /jobs lists all jobs, DELETE /jobs/{job_id} cleans up
  - Uses ThreadPoolExecutor with 2 workers for CPU-bound OCR
- [x] Implement JSON output format (in output.py)
- [ ] Add progress reporting for CLI
- [ ] Research additional CPU-compatible OCRs (PaddleOCR, docTR)
- [ ] Performance optimization

## Completed

- [x] Project initialization
- [x] Create test data (peter_lou.pdf, peter_lou_50dpi.pdf, words CSV)
- [x] Define project specifications

## Validation Results

**Latest extraction test (peter_lou.pdf with harmonization + pixel detection):**
- Ground truth: 401 entries
- Our output: 420 entries (over-extraction is OK for redaction recall)
- Tesseract alone: 399 words
- EasyOCR alone: 378 words
- Harmonized: 414 words
- Pixel detector regions: 6 (3 match ground truth, 3 extra)

**Text coverage analysis:**
- Missing texts (4): `0923847`, `4_`, `peter lou`, `rmartinez pdx@gmail.com`
- Extra texts (5+): minor variations like `,`, email formatting differences
- Most differences are OCR transcription variations (e.g., `10.28` vs `10:28`)

**Harmonization strategy:**
- Match words by bounding box IoU (threshold 0.3)
- Vote on text: prefer Tesseract for character accuracy
- Include unmatched Tesseract words (high recall)
- Include unmatched EasyOCR words if not covered by Tesseract

**Next steps to improve accuracy:**
1. Tune pixel detection to reduce false positives
2. Add confidence-based filtering for extra detections
3. Implement preprocessing pipeline for degraded PDFs

## Notes

- CPU-only constraint - no CUDA/GPU dependencies
- Bounding boxes must be in PDF coordinate space (points, origin top-left)
- Empty `engine` field means harmonized result from multiple engines
- `pixel_detector` engine = fallback for OCR-missed content with confidence 0.0
- Target: 401 words across 3 pages matching ground truth

## Current Status

**✓ Tesseract installed and verified** (v5.3.4)
**✓ Harmonization implemented** (harmonize.py - Tesseract + EasyOCR fusion)
**✓ OpenCV preprocessing implemented** (preprocess.py - auto-detects quality level)

**Preprocessing test results (degraded 50dpi PDF):**
- No preprocess: 476 words (noisy)
- Light: 506 words
- Standard: 517 words
- Aggressive: 267 words (over-filtered)
- Auto: 422 words (balanced)

**✓ Triage system implemented** (triage.py - filters low-confidence detections)

**Triage test results:**
- No triage: 418 words
- Permissive: 415 words
- Normal: 407 words
- Strict: 402 words (very close to GT=401)

**✓ Evaluation metrics implemented** (metrics.py)

**Evaluation results (peter_lou.pdf vs ground truth):**
| Config | Precision | Recall | F1 | Words |
|--------|-----------|--------|-----|-------|
| No triage | 94.50% | **98.50%** | 96.46% | 418 |
| Normal | 94.59% | 96.01% | 95.30% | 407 |
| Strict | 94.78% | 95.01% | 94.89% | 402 |

For redaction, no triage recommended (highest recall).

All Medium Priority tasks complete. Next: Low Priority items (FastAPI, async, etc.)
