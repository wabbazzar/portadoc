# Portadoc Fix Plan

## High Priority (Core Pipeline)

- [x] Set up Python project structure with requirements.txt
- [x] Install Python dependencies (pymupdf, pytesseract, opencv-python-headless, numpy, click, easyocr)
- [x] Implement PDF to image conversion using PyMuPDF
- [x] Implement Tesseract OCR wrapper with word-level bounding boxes
- [x] Implement EasyOCR wrapper with word-level output
- [x] Build coordinate transformation (image coords → PDF coords)
- [x] Create basic CLI with `extract` command
- [ ] **BLOCKED**: Install tesseract-ocr binary - needs sudo: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`

## Medium Priority (Quality Improvements)

- [ ] Implement OpenCV preprocessing pipeline
  - Grayscale, denoising, contrast enhancement, binarization
- [ ] Build OCR result harmonization logic
  - Merge overlapping boxes, vote on text content
- [ ] Implement pixel detection fallback for OCR misses
  - Contour detection for text-like regions
- [ ] Add confidence-based triage system
- [ ] Create evaluation metrics (recall, precision, IoU vs ground truth)

## Low Priority (Web Service & Polish)

- [ ] Build FastAPI REST endpoints
- [ ] Add async job processing for large PDFs
- [x] Implement JSON output format (in output.py)
- [ ] Add progress reporting for CLI
- [ ] Research additional CPU-compatible OCRs (PaddleOCR, docTR)
- [ ] Performance optimization

## Completed

- [x] Project initialization
- [x] Create test data (peter_lou.pdf, peter_lou_50dpi.pdf, words CSV)
- [x] Define project specifications

## Validation Results

**EasyOCR extraction test (peter_lou.pdf):**
- Ground truth: 401 words
- EasyOCR output: 378 words
- Gap: 23 words (~94% recall)

Missing items likely include:
- Pixel detector regions (logos, signatures)
- Some low-confidence text
- Small punctuation marks

Next step: Implement pixel detection fallback and OCR harmonization to improve recall.

## Notes

- CPU-only constraint - no CUDA/GPU dependencies
- Bounding boxes must be in PDF coordinate space (points, origin top-left)
- Empty `engine` field means harmonized result from multiple engines
- `pixel_detector` engine = fallback for OCR-missed content with confidence 0.0
- Target: 401 words across 3 pages matching ground truth

## Current Blocker

**Tesseract binary not installed.** Run this command to unblock:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

Then verify with: `PYTHONPATH=src python3 -m portadoc.cli check`
