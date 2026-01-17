"""FastAPI REST API for Portadoc."""

import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .extractor import extract_words
from .models import Document


app = FastAPI(
    title="Portadoc API",
    description="PDF word extraction API for document redaction",
    version="0.1.0",
)


class BBoxResponse(BaseModel):
    """Bounding box in PDF coordinate space."""

    x0: float = Field(description="Left edge (points)")
    y0: float = Field(description="Top edge (points)")
    x1: float = Field(description="Right edge (points)")
    y1: float = Field(description="Bottom edge (points)")


class WordResponse(BaseModel):
    """A single extracted word."""

    word_id: int
    text: str
    bbox: BBoxResponse
    page: int
    engine: str = ""
    confidence: float = 0.0


class PageResponse(BaseModel):
    """A single page of extracted words."""

    page_number: int
    width: float
    height: float
    words: list[WordResponse]


class ExtractionResponse(BaseModel):
    """Complete extraction result."""

    filename: str
    total_words: int
    pages: list[PageResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    tesseract_available: bool
    easyocr_available: bool


def document_to_response(doc: Document) -> ExtractionResponse:
    """Convert internal Document model to API response."""
    pages = []
    for page in doc.pages:
        words = [
            WordResponse(
                word_id=w.word_id,
                text=w.text,
                bbox=BBoxResponse(
                    x0=w.bbox.x0,
                    y0=w.bbox.y0,
                    x1=w.bbox.x1,
                    y1=w.bbox.y1,
                ),
                page=w.page,
                engine=w.engine,
                confidence=w.confidence,
            )
            for w in page.words
        ]
        pages.append(
            PageResponse(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                words=words,
            )
        )
    return ExtractionResponse(
        filename=doc.filename,
        total_words=doc.total_words,
        pages=pages,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and OCR engine availability."""
    from .ocr.tesseract import is_tesseract_available
    from .ocr.easyocr import is_easyocr_available

    return HealthResponse(
        status="ok",
        tesseract_available=is_tesseract_available(),
        easyocr_available=is_easyocr_available(),
    )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_pdf(
    file: UploadFile = File(..., description="PDF file to process"),
    dpi: int = Query(300, ge=72, le=600, description="DPI for rendering"),
    triage: Optional[str] = Query(
        None,
        pattern="^(strict|normal|permissive)$",
        description="Triage level for filtering",
    ),
    preprocess: str = Query(
        "auto",
        pattern="^(none|light|standard|aggressive|auto)$",
        description="Preprocessing level",
    ),
):
    """
    Extract words and bounding boxes from a PDF.

    Upload a PDF file and receive structured word extraction results
    with bounding boxes suitable for redaction workflows.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="File must be a PDF",
        )

    # Save uploaded file to temp location
    temp_dir = Path(tempfile.gettempdir()) / "portadoc"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex}.pdf"

    try:
        # Write uploaded content to temp file
        content = await file.read()
        temp_path.write_bytes(content)

        # Extract words
        doc = extract_words(
            temp_path,
            dpi=dpi,
            triage=triage,
            preprocess=preprocess,
        )

        # Update filename to original
        doc.filename = file.filename

        return document_to_response(doc)

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@app.get("/")
async def root():
    """API root - returns basic info."""
    return {
        "name": "Portadoc API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
