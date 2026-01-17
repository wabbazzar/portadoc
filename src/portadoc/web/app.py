"""FastAPI web application for PDF visualization with bounding box overlays."""

import csv
import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Default directories
DEFAULT_PDF_DIR = Path("data/input")
DEFAULT_OUTPUT_DIR = Path("data/output")

router = APIRouter()


class ExtractionRequest(BaseModel):
    """Request body for extraction endpoint."""
    pdf_path: str
    use_tesseract: bool = True
    use_easyocr: bool = True
    use_paddleocr: bool = False
    use_doctr: bool = False
    preprocess: str = "none"
    psm: int = 6
    oem: int = 3
    smart: bool = True


class WordData(BaseModel):
    """Word data for JSON response."""
    word_id: int
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    status: str
    source: str
    confidence: float
    tess_text: Optional[str] = None
    easy_text: Optional[str] = None
    doctr_text: Optional[str] = None
    paddle_text: Optional[str] = None


def get_output_path(pdf_path: Path) -> Path:
    """Get standardized output path for a PDF."""
    return pdf_path.parent / f"{pdf_path.stem}_extracted.csv"


def load_words_from_csv(csv_path: Path) -> list[dict]:
    """Load words from extracted CSV file."""
    if not csv_path.exists():
        return []

    words = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = {
                "word_id": int(row.get("word_id", 0)),
                "page": int(row.get("page", 0)),
                "x0": float(row.get("x0", 0)),
                "y0": float(row.get("y0", 0)),
                "x1": float(row.get("x1", 0)),
                "y1": float(row.get("y1", 0)),
                "text": row.get("text", ""),
                "status": row.get("status", ""),
                "source": row.get("source", ""),
                "confidence": float(row.get("confidence", 0) or 0),
                "tess_text": row.get("tess", ""),
                "easy_text": row.get("easy", ""),
                "doctr_text": row.get("doctr", ""),
                "paddle_text": row.get("paddle", ""),
            }
            words.append(word)
    return words


@router.get("/api/pdfs")
async def list_pdfs(directory: str = None):
    """List available PDF files."""
    pdf_dir = Path(directory) if directory else DEFAULT_PDF_DIR
    if not pdf_dir.exists():
        return {"pdfs": [], "error": f"Directory not found: {pdf_dir}"}

    pdfs = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        output_path = get_output_path(pdf_file)
        pdfs.append({
            "name": pdf_file.name,
            "path": str(pdf_file),
            "has_extraction": output_path.exists(),
            "extraction_path": str(output_path) if output_path.exists() else None,
        })

    return {"pdfs": pdfs}


@router.get("/api/pdf/{filename:path}")
async def get_pdf(filename: str):
    """Serve a PDF file for rendering."""
    # Try as absolute path first
    pdf_path = Path(filename)
    if not pdf_path.exists():
        # Try in default directory
        pdf_path = DEFAULT_PDF_DIR / filename

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@router.get("/api/words/{filename:path}")
async def get_words(filename: str):
    """Get extracted words for a PDF."""
    # Determine PDF path
    pdf_path = Path(filename)
    if not pdf_path.exists():
        pdf_path = DEFAULT_PDF_DIR / filename

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {filename}")

    # Check for existing extraction
    output_path = get_output_path(pdf_path)

    if not output_path.exists():
        return {
            "words": [],
            "extracted": False,
            "message": f"No extraction found. Run extraction first.",
            "pdf_path": str(pdf_path),
        }

    words = load_words_from_csv(output_path)

    # Group by page
    pages = {}
    for word in words:
        page = word["page"]
        if page not in pages:
            pages[page] = []
        pages[page].append(word)

    # Stats
    status_counts = {}
    source_counts = {}
    for word in words:
        status = word["status"]
        source = word["source"]
        status_counts[status] = status_counts.get(status, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1

    return {
        "words": words,
        "extracted": True,
        "total_words": len(words),
        "pages": len(pages),
        "status_counts": status_counts,
        "source_counts": source_counts,
        "pdf_path": str(pdf_path),
        "csv_path": str(output_path),
    }


@router.post("/api/extract")
async def extract_pdf(request: ExtractionRequest):
    """Run extraction on a PDF with specified settings."""
    from ..extractor import extract_words_smart
    from ..output import write_harmonized_csv

    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        pdf_path = DEFAULT_PDF_DIR / request.pdf_path

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {request.pdf_path}")

    output_path = get_output_path(pdf_path)

    try:
        # Run extraction with smart harmonization
        harmonized_words = extract_words_smart(
            pdf_path=pdf_path,
            dpi=300,
            use_tesseract=request.use_tesseract,
            use_easyocr=request.use_easyocr,
            use_paddleocr=request.use_paddleocr,
            use_doctr=request.use_doctr,
            preprocess=request.preprocess,
            tesseract_psm=request.psm,
            tesseract_oem=request.oem,
        )

        # Save to CSV
        write_harmonized_csv(harmonized_words, output_path)

        # Load and return words
        words = load_words_from_csv(output_path)

        return {
            "success": True,
            "words": words,
            "total_words": len(words),
            "pdf_path": str(pdf_path),
            "csv_path": str(output_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/config")
async def get_current_config():
    """Get current configuration values."""
    from ..config import get_config

    config = get_config()

    return {
        "harmonize": {
            "iou_threshold": config.harmonize.iou_threshold,
            "text_match_bonus": config.harmonize.text_match_bonus,
            "center_distance_max": config.harmonize.center_distance_max,
        },
        "ocr": {
            "tesseract": {
                "psm": config.ocr.tesseract.psm,
                "oem": config.ocr.tesseract.oem,
            },
        },
        "engines": {
            "tesseract": True,  # Always enabled
            "easyocr": config.harmonize.secondary.engines.get("easyocr", {}).enabled if hasattr(config.harmonize.secondary.engines.get("easyocr", {}), 'enabled') else True,
            "paddleocr": config.harmonize.secondary.engines.get("paddleocr", {}).enabled if hasattr(config.harmonize.secondary.engines.get("paddleocr", {}), 'enabled') else True,
            "doctr": config.harmonize.secondary.engines.get("doctr", {}).enabled if hasattr(config.harmonize.secondary.engines.get("doctr", {}), 'enabled') else True,
        },
    }


def create_app(pdf_dir: Path = None, static_dir: Path = None) -> FastAPI:
    """Create the FastAPI application."""
    global DEFAULT_PDF_DIR

    if pdf_dir:
        DEFAULT_PDF_DIR = pdf_dir

    app = FastAPI(
        title="Portadoc Web",
        description="PDF visualization with OCR bounding box overlays",
        version="0.1.0",
    )

    # Include API routes
    app.include_router(router)

    # Serve static files
    if static_dir is None:
        static_dir = Path(__file__).parent / "static"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Serve index.html at root
    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return index_path.read_text()
        return """
        <html>
            <head><title>Portadoc Web</title></head>
            <body>
                <h1>Portadoc Web</h1>
                <p>Static files not found. Check static directory.</p>
                <p>API available at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """

    return app
