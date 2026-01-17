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
- [ ] **BLOCKED**: Install tesseract-ocr binary - needs sudo: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`

## Medium Priority (Quality Improvements)

- [ ] Implement OpenCV preprocessing pipeline
  - Grayscale, denoising, contrast enhancement, binarization
- [ ] Build OCR result harmonization logic (Tesseract + EasyOCR)
  - Merge overlapping boxes, vote on text content
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

**Latest extraction test (peter_lou.pdf with EasyOCR + pixel detection):**
- Ground truth: 401 entries
- Our output: 385 entries
- OCR words: 378
- Pixel detector regions: 7 (3 match ground truth, 4 extra detections)

**Pixel detector accuracy:**
- Page 0 logo: ✓ detected (bbox matches within 1-2 pts)
- Page 0 horizontal line: ✓ detected
- Page 2 vertical line: ✓ detected

**Missing content analysis:**
- Most differences are OCR transcription variations (e.g., `10.28` vs `10:28`)
- Single-character entries (`e`, `o`, `i`) not detected - likely low-confidence
- Some multi-word items split differently

**Next steps to improve recall:**
1. Install Tesseract for better OCR accuracy
2. Implement harmonization to combine Tesseract + EasyOCR results
3. Tune pixel detection to reduce false positives

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
