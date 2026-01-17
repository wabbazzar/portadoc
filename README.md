# Portadoc

PDF word extraction system for document redaction workflows.

## Quick Start

```bash
# Install system dependencies
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Install Python dependencies
pip install -r requirements.txt

# Extract words from PDF
python -m portadoc.cli extract input.pdf -o output.csv
```

## Features

- Multi-engine OCR (Tesseract + EasyOCR)
- Handles degraded/blurry documents
- Word-level bounding boxes
- Pixel detection fallback for OCR misses
- CSV and JSON output formats

See `specs/portadoc.md` for full specification.
