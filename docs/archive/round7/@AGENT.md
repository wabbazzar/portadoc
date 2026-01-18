# Portadoc - Agent Build Instructions

## Project Setup

### System Dependencies
```bash
# Tesseract OCR (already installed)
tesseract --version  # Should show 5.3.4
```

### Python Environment
```bash
# ALWAYS activate the virtual environment first
source .venv/bin/activate

# Verify installation
python -c "import paddleocr; print('PaddleOCR OK')"
python -c "from doctr.models import ocr_predictor; print('docTR OK')"
python -c "import realesrgan; print('RealESRGAN OK')"
```

## Running the Application

### CLI Usage
```bash
source .venv/bin/activate

# Extract words from PDF to CSV
portadoc extract data/input/peter_lou.pdf -o output.csv

# Extract with specific options
portadoc extract data/input/peter_lou_50dpi.pdf --preprocess none --psm 6

# Evaluate against ground truth
portadoc eval data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv
```

### FastAPI Server
```bash
source .venv/bin/activate
uvicorn portadoc.api:app --reload --port 8000
```

## Running Tests
```bash
source .venv/bin/activate
pytest
pytest --cov=src/portadoc tests/ --cov-report=term-missing
```

## Validation
```bash
source .venv/bin/activate

# Degraded PDF evaluation (this is the hard one)
portadoc eval data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv

# Clean PDF evaluation (should maintain ~96% F1)
portadoc eval data/input/peter_lou.pdf data/input/peter_lou_words_slim.csv
```

## Key Learnings
- Tesseract provides word-level boxes via `image_to_data()` with Output.DICT
- EasyOCR returns line-level boxes that need word decomposition
- PaddleOCR handles degraded documents better than Tesseract/EasyOCR
- PDF coordinates use points (1/72 inch), origin at top-left
- Image coordinates use pixels, origin at top-left - scaling required
- For degraded images: super-resolution > preprocessing
- 50 DPI source needs 4-6x upscale before OCR works well

## Project Structure
```
src/portadoc/
├── __init__.py
├── cli.py           # Command line interface
├── api.py           # FastAPI endpoints
├── config.py        # Configuration loader (TO CREATE)
├── pdf.py           # PDF loading and conversion
├── preprocess.py    # OpenCV image enhancement
├── superres.py      # Image super-resolution
├── extractor.py     # Main extraction pipeline
├── harmonize.py     # Multi-engine result merging (REWRITE - smart_harmonize)
├── detection.py     # Pixel-based text detection fallback
├── triage.py        # Confidence-based filtering
├── metrics.py       # Evaluation metrics
├── models.py        # Data structures (Word, BBox, Page, HarmonizedWord)
├── output.py        # CSV/JSON formatters
└── ocr/
    ├── __init__.py
    ├── tesseract.py   # Tesseract wrapper
    ├── easyocr.py     # EasyOCR wrapper (LINE-level bbox)
    ├── paddleocr.py   # PaddleOCR wrapper (LINE-level bbox)
    └── doctr_ocr.py   # docTR wrapper (WORD-level bbox)

config/
└── harmonize.yaml   # All OCR and harmonization thresholds
```

## Dependencies
Core (in .venv):
- pymupdf (PDF rendering)
- opencv-python (image preprocessing)
- pytesseract (Tesseract OCR)
- easyocr (secondary OCR)
- paddleocr (PaddleOCR - best for degraded docs)
- python-doctr (docTR OCR)
- realesrgan, basicsr (super-resolution)
- fastapi + uvicorn (web API)
- python-Levenshtein (text distance for harmonization)
- pyyaml (config file parsing)

## Key Architecture Notes

### OCR Engine Bbox Granularity
- **Tesseract**: WORD-level bboxes (precise, use as primary)
- **EasyOCR**: LINE-level bboxes (decomposed to words by char proportion - imprecise)
- **PaddleOCR**: LINE-level bboxes (same issue as EasyOCR)
- **docTR**: WORD-level bboxes (can use directly)

### Harmonization Strategy
See `@fix_plan.md` "Priority 2: Smart Harmonization Logic" for full design.
- Tesseract is PRIMARY engine (trusted for bbox)
- Secondary engines vote on TEXT only
- LINE-level engines matched via containment, not IoU
- All detections go to CSV with status: word/low_conf/pixel/secondary_only

## Quality Standards

### Testing Requirements
- All tests must pass before committing
- Test on both clean and degraded PDFs
- Record metrics in @fix_plan.md

### Git Workflow
1. Commit with conventional messages: `feat:`, `fix:`, `test:`, etc.
2. Update @fix_plan.md with progress and metrics
3. Push after each completed feature

### Validation Checkpoints
After each feature:
- [ ] Tests pass with `pytest`
- [ ] Degraded PDF: F1 improved or maintained
- [ ] Clean PDF: no regression (~96% F1)
- [ ] @fix_plan.md updated with results
- [ ] Changes committed
