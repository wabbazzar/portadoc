# Portadoc - Agent Build Instructions

## Project Setup

### System Dependencies
```bash
# Install Tesseract OCR
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# Verify installation
tesseract --version
```

### Python Environment
```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Running the Application

### CLI Usage
```bash
# Extract words from PDF to CSV
python -m portadoc.cli extract data/input/peter_lou.pdf -o output.csv

# Extract with JSON output
python -m portadoc.cli extract data/input/peter_lou.pdf --format json -o output.json

# Process degraded PDF
python -m portadoc.cli extract data/input/peter_lou_50dpi.pdf -o degraded_output.csv
```

### FastAPI Server (when implemented)
```bash
# Start development server
uvicorn portadoc.api:app --reload --port 8000

# API endpoints:
# POST /api/v1/extract - Upload PDF for processing
# GET /api/v1/jobs/{job_id} - Check job status
```

## Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/portadoc tests/ --cov-report=term-missing

# Run specific test file
pytest tests/test_ocr.py -v
```

## Validation
```bash
# Compare output to ground truth
python -m portadoc.cli extract data/input/peter_lou.pdf -o data/output/test.csv
diff data/output/test.csv data/input/peter_lou_words_slim.csv

# Check word count
wc -l data/output/test.csv  # Should be 402 (401 words + header)
```

## Key Learnings
- Tesseract provides word-level boxes via `image_to_data()` with Output.DICT
- EasyOCR returns line-level boxes that need word decomposition
- PDF coordinates use points (1/72 inch), origin at bottom-left
- Image coordinates use pixels, origin at top-left - transformation required
- For degraded images: upscale before OCR, use adaptive thresholding
- Pixel detection fallback catches logos, signatures, and obscured text

## Project Structure
```
src/portadoc/
├── __init__.py
├── cli.py           # Command line interface
├── api.py           # FastAPI endpoints
├── pdf.py           # PDF loading and conversion
├── preprocessing.py # OpenCV image enhancement
├── ocr/
│   ├── __init__.py
│   ├── tesseract.py # Tesseract wrapper
│   ├── easyocr.py   # EasyOCR wrapper
│   └── harmonizer.py # Multi-engine result merging
├── detection.py     # Pixel-based text detection fallback
├── models.py        # Data structures (Word, BBox, Page)
└── output.py        # CSV/JSON formatters
```

## Feature Development Quality Standards

**CRITICAL**: All new features MUST meet these requirements before being considered complete.

### Testing Requirements
- Minimum 85% code coverage for new code
- All tests must pass
- Unit tests for business logic
- Integration tests comparing output to ground truth

### Git Workflow
1. Commit with conventional messages: `feat:`, `fix:`, `test:`, etc.
2. Push to remote after each completed feature
3. Update @fix_plan.md with progress

### Validation Checkpoints
After each feature:
- [ ] Tests pass with `pytest`
- [ ] Output matches ground truth (401 words, correct bounding boxes)
- [ ] Works on both clean and degraded PDFs
- [ ] @fix_plan.md updated
- [ ] Changes committed and pushed

## Dependencies
See `requirements.txt` for full list. Core dependencies:
- pymupdf (PDF rendering)
- opencv-python (image preprocessing)
- pytesseract (Tesseract OCR)
- easyocr (secondary OCR)
- fastapi + uvicorn (web API)
