# Portadoc - PDF Word Extraction and Redaction System

## Overview
Portadoc is a CPU-bound document processing system that extracts words and their bounding boxes from PDF documents. The system handles degraded/blurry documents using multiple OCR engines, preprocessing techniques, and pixel detection fallbacks.

## Primary Goal
Extract ALL text content from PDF documents with word-level bounding boxes, achieving 100% recall suitable for document redaction workflows.

## Test Case
- **Input files**: `data/input/peter_lou.pdf` (clean) and `data/input/peter_lou_50dpi.pdf` (blurry/degraded)
- **Ground truth**: `data/input/peter_lou_words_slim.csv` - 401 words across 3 pages
- **Success criteria**: Output must match ground truth CSV (words + bounding boxes)

## Architecture

### Phase 1: CLI Tool
Build a command-line tool that processes PDFs and outputs word extraction results.

```bash
portadoc extract input.pdf -o output.csv
portadoc extract input.pdf --format json -o output.json
```

### Phase 2: FastAPI Web Service
REST API for document processing.

```
POST /api/v1/extract
  - Upload PDF, returns extracted words with bounding boxes

GET /api/v1/jobs/{job_id}
  - Check processing status for async jobs
```

## Core Processing Pipeline

### 1. PDF to Image Conversion (PyMuPDF)
- Convert each PDF page to high-resolution image (300+ DPI for clean, upscale for degraded)
- Preserve original PDF coordinate system for bounding box mapping

### 2. Image Preprocessing (OpenCV)
Apply preprocessing to improve OCR accuracy:
- Grayscale conversion
- Noise reduction (Gaussian blur, bilateral filter)
- Contrast enhancement (CLAHE)
- Binarization (adaptive thresholding, Otsu's method)
- Deskewing
- For degraded images: super-resolution upscaling

### 3. Multi-Engine OCR

#### Primary: Tesseract
- Word-level bounding boxes with confidence scores
- Configure for best accuracy (--psm 3, --oem 3)

#### Secondary: EasyOCR
- Line-level bounding boxes (decompose to word-level)
- Good for handwriting and stylized text

#### Tertiary: Other CPU-compatible OCRs (research needed)
- Must provide at minimum line-level bounding boxes
- Examples to investigate: PaddleOCR (CPU mode), docTR, keras-ocr

### 4. Result Harmonization
Combine results from multiple OCR engines:
- Merge overlapping bounding boxes
- Vote on text content when engines disagree
- Track confidence per engine
- Flag low-confidence regions for review

### 5. Fallback Strategies (Triage)

#### Level 1: Standard OCR
Extract words using Tesseract + EasyOCR harmonization.

#### Level 2: Low-Confidence Region Detection
For regions where OCR confidence < threshold (e.g., 60%):
- Still include the word but flag as uncertain
- Expand bounding box slightly for safety in redaction

#### Level 3: Pixel Detection
For content OCR completely misses:
- Detect text-like pixel regions using contour detection
- Identify horizontal lines of dark pixels
- Create bounding boxes for unrecognized content
- Mark as `engine: pixel_detector, confidence: 0.0`

## Output Format

### CSV (Primary)
```csv
page,word_id,text,x0,y0,x1,y1,engine,ocr_confidence
0,0,"7/24/25,",24.24,17.04,50.4,23.76,,93.0
0,9,,72.72,81.72,121.68,141.84,pixel_detector,0.0
```

### JSON (Alternative)
```json
{
  "pages": [
    {
      "page_number": 0,
      "words": [
        {"word_id": 0, "text": "7/24/25,", "bbox": [24.24, 17.04, 50.4, 23.76], "engine": "harmonized", "confidence": 93.0}
      ]
    }
  ]
}
```

## Bounding Box Coordinate System
- Origin: Top-left of page
- Units: PDF points (1/72 inch)
- Format: `(x0, y0, x1, y1)` where (x0, y0) is top-left corner, (x1, y1) is bottom-right corner

## Dependencies (CPU-only)

```
pymupdf>=1.23.0
opencv-python>=4.8.0
pytesseract>=0.3.10
easyocr>=1.7.0
numpy>=1.24.0
pillow>=10.0.0
fastapi>=0.109.0
uvicorn>=0.27.0
python-multipart>=0.0.6
```

System dependencies:
- tesseract-ocr
- tesseract-ocr-eng (English language pack)

## Quality Metrics

### Recall (Primary)
- Percentage of ground truth words found
- Target: 100%

### Precision
- Percentage of extracted words that are correct
- Target: >95%

### Bounding Box Accuracy
- IoU (Intersection over Union) with ground truth boxes
- Target: >0.8 average IoU

## Testing Strategy

### Unit Tests
- PDF loading and conversion
- Image preprocessing functions
- Individual OCR engine wrappers
- Bounding box coordinate transformations

### Integration Tests
- Full pipeline with test PDFs
- Compare output to ground truth CSV
- Measure recall/precision/IoU metrics

### Test Files
- `data/input/peter_lou.pdf` - clean document (baseline)
- `data/input/peter_lou_50dpi.pdf` - degraded document (stress test)
- `data/input/peter_lou_words_slim.csv` - ground truth

## Non-Goals (For Now)
- GPU acceleration
- Table structure extraction
- Form field detection
- Multi-language support (English only for MVP)
- Real-time streaming
