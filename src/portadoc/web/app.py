"""FastAPI web application for PDF visualization with bounding box overlays."""

import csv
import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Default directories
DEFAULT_PDF_DIR = Path("data/input")
DEFAULT_OUTPUT_DIR = Path("data/output")

router = APIRouter()


class HarmonizeConfigOverride(BaseModel):
    """Override harmonization settings."""
    iou_threshold: Optional[float] = None
    text_match_bonus: Optional[float] = None
    center_distance_max: Optional[float] = None
    word_min_conf: Optional[float] = None
    low_conf_min_conf: Optional[float] = None


class GeometricClusteringConfigOverride(BaseModel):
    """Override geometric clustering settings."""
    y_fuzz_default: Optional[float] = None
    y_fuzz_multiplier: Optional[float] = None
    y_fuzz_max_height_ratio: Optional[float] = None
    x_overlap_min: Optional[float] = None
    y_overlap_min: Optional[float] = None


class ExtractionRequest(BaseModel):
    """Request body for extraction endpoint."""
    pdf_path: str
    use_tesseract: bool = True
    use_easyocr: bool = True
    use_paddleocr: bool = True
    use_doctr: bool = True
    use_surya: bool = True
    use_kraken: bool = False
    preprocess: str = "none"
    psm: int = 6
    oem: int = 3
    primary_engine: Optional[str] = None  # None = use config default
    # Config overrides
    harmonize: Optional[HarmonizeConfigOverride] = None
    geometric_clustering: Optional[GeometricClusteringConfigOverride] = None


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
    rotation: int = 0
    tess_text: Optional[str] = None
    easy_text: Optional[str] = None
    doctr_text: Optional[str] = None
    paddle_text: Optional[str] = None
    surya_text: Optional[str] = None
    kraken_text: Optional[str] = None


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
                "confidence": float(row.get("conf", 0) or 0),
                "rotation": int(row.get("rotation", 0) or 0),
                "tess_text": row.get("tess", ""),
                "easy_text": row.get("easy", ""),
                "doctr_text": row.get("doctr", ""),
                "paddle_text": row.get("paddle", ""),
                "surya_text": row.get("surya", ""),
                "kraken_text": row.get("kraken", ""),
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
    # Include PDFs from root dir and uploads subdirectory
    for pdf_file in pdf_dir.glob("**/*.pdf"):
        output_path = get_output_path(pdf_file)
        # Show relative path for nested files
        display_name = pdf_file.name
        if pdf_file.parent != pdf_dir:
            display_name = f"{pdf_file.parent.name}/{pdf_file.name}"
        pdfs.append({
            "name": display_name,
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

    # Group by page and collect rotation per page
    pages = {}
    page_rotations = {}
    for word in words:
        page = word["page"]
        if page not in pages:
            pages[page] = []
            page_rotations[page] = word.get("rotation", 0)
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
        "page_rotations": page_rotations,
        "status_counts": status_counts,
        "source_counts": source_counts,
        "pdf_path": str(pdf_path),
        "csv_path": str(output_path),
    }


@router.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Save to uploads directory
    uploads_dir = DEFAULT_PDF_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Use original filename
    file_path = uploads_dir / file.filename

    # Write the file
    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    return {
        "success": True,
        "filename": file.filename,
        "path": str(file_path),
    }


@router.post("/api/extract")
async def extract_pdf(request: ExtractionRequest):
    """Run extraction on a PDF with specified settings."""
    from ..extractor import extract_words
    from ..output import write_harmonized_csv
    from ..config import apply_config_overrides

    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        pdf_path = DEFAULT_PDF_DIR / request.pdf_path

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {request.pdf_path}")

    output_path = get_output_path(pdf_path)

    try:
        # Apply config overrides if provided
        overrides = {}
        if request.harmonize:
            overrides["harmonize"] = {
                "iou_threshold": request.harmonize.iou_threshold,
                "text_match_bonus": request.harmonize.text_match_bonus,
                "center_distance_max": request.harmonize.center_distance_max,
                "word_min_conf": request.harmonize.word_min_conf,
                "low_conf_min_conf": request.harmonize.low_conf_min_conf,
            }
        if request.geometric_clustering:
            overrides["geometric_clustering"] = {
                "y_fuzz_default": request.geometric_clustering.y_fuzz_default,
                "y_fuzz_multiplier": request.geometric_clustering.y_fuzz_multiplier,
                "y_fuzz_max_height_ratio": request.geometric_clustering.y_fuzz_max_height_ratio,
                "x_overlap_min": request.geometric_clustering.x_overlap_min,
                "y_overlap_min": request.geometric_clustering.y_overlap_min,
            }

        if overrides:
            apply_config_overrides(overrides)

        # Run extraction
        harmonized_words = extract_words(
            pdf_path=pdf_path,
            dpi=300,
            use_tesseract=request.use_tesseract,
            use_easyocr=request.use_easyocr,
            use_paddleocr=request.use_paddleocr,
            use_doctr=request.use_doctr,
            use_surya=request.use_surya,
            use_kraken=request.use_kraken,
            preprocess=request.preprocess,
            tesseract_psm=request.psm,
            tesseract_oem=request.oem,
            primary_engine=request.primary_engine,
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


@router.get("/api/reading-order/{filename:path}")
async def get_reading_order(filename: str, page: int = Query(0, ge=0)):
    """Get reconstructed reading order text for a PDF page."""
    from ..reading_order import reconstruct_lines, lines_to_json

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
            "lines": [],
            "extracted": False,
            "message": "No extraction found. Run extraction first.",
            "page": page,
        }

    words = load_words_from_csv(output_path)
    lines = reconstruct_lines(words, page)

    return {
        "lines": lines_to_json(lines),
        "extracted": True,
        "page": page,
        "total_lines": len(lines),
        "total_words": sum(len(line.words) for line in lines),
    }


@router.get("/api/config")
async def get_current_config():
    """Get current configuration values and UI descriptions."""
    from ..config import get_config, get_ui_descriptions

    config = get_config()
    ui_descriptions = get_ui_descriptions()

    return {
        "harmonize": {
            "iou_threshold": config.harmonize.iou_threshold,
            "text_match_bonus": config.harmonize.text_match_bonus,
            "center_distance_max": config.harmonize.center_distance_max,
            "word_min_conf": config.harmonize.status.word_min_conf,
            "low_conf_min_conf": config.harmonize.status.low_conf_min_conf,
        },
        "geometric_clustering": {
            "y_fuzz": {
                "default": config.geometric_clustering.y_fuzz.default,
                "multiplier": config.geometric_clustering.y_fuzz.multiplier,
                "max_height_ratio": config.geometric_clustering.y_fuzz.max_height_ratio,
            },
            "connection": {
                "x_overlap_min": config.geometric_clustering.connection.x_overlap_min,
                "y_overlap_min": config.geometric_clustering.connection.y_overlap_min,
            },
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
            "surya": config.harmonize.secondary.engines.get("surya", {}).enabled if hasattr(config.harmonize.secondary.engines.get("surya", {}), 'enabled') else True
        },
        "ui_descriptions": ui_descriptions,
    }


class RedactRequest(BaseModel):
    """Request body for redaction endpoint."""
    csv_path: str
    detect_names: bool = True
    detect_dates: bool = True
    detect_codes: bool = True
    names_path: Optional[str] = None


class ApplyRequest(BaseModel):
    """Request body for apply redactions endpoint."""
    pdf_path: str
    csv_path: str
    output_path: Optional[str] = None
    color: str = "black"
    preview: bool = False


@router.post("/api/redact")
async def redact_csv_endpoint(request: RedactRequest):
    """Detect entities and mark words for redaction."""
    from ..redact import redact_csv, get_redaction_stats

    csv_path = Path(request.csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {request.csv_path}")

    names_path = Path(request.names_path) if request.names_path else None
    if names_path and not names_path.exists():
        raise HTTPException(status_code=404, detail=f"Names file not found: {request.names_path}")

    try:
        count = redact_csv(
            csv_path,
            output_csv=None,  # Overwrite input
            names_path=names_path,
            detect_names=request.detect_names,
            detect_dates=request.detect_dates,
            detect_codes=request.detect_codes,
        )

        stats = get_redaction_stats(csv_path)

        # Reload words with new columns
        words = load_words_from_csv(csv_path)

        return {
            "success": True,
            "words": words,
            "redacted_count": count,
            "by_type": stats["by_type"],
            "total_words": stats["total_words"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/apply")
async def apply_redactions_endpoint(request: ApplyRequest):
    """Apply redactions to a PDF."""
    from ..apply import apply_redactions, apply_redactions_preview

    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        # Try in default directory
        pdf_path = DEFAULT_PDF_DIR / request.pdf_path
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {request.pdf_path}")

    csv_path = Path(request.csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {request.csv_path}")

    # Generate output path if not provided
    if request.output_path:
        output_path = Path(request.output_path)
    else:
        output_path = pdf_path.parent / f"{pdf_path.stem}_redacted.pdf"

    # Map color names to RGB tuples
    color_map = {
        "black": (0, 0, 0),
        "white": (1, 1, 1),
        "red": (1, 0, 0),
    }
    rgb_color = color_map.get(request.color, (0, 0, 0))

    try:
        if request.preview:
            count = apply_redactions_preview(
                pdf_path,
                csv_path,
                output_path,
                border_color=rgb_color,
            )
        else:
            count = apply_redactions(
                pdf_path,
                csv_path,
                output_path,
                color=rgb_color,
            )

        return {
            "success": True,
            "redacted_count": count,
            "output_path": str(output_path),
            "preview": request.preview,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/redacted-pdf/{filename:path}")
async def get_redacted_pdf(filename: str):
    """Serve a redacted PDF file."""
    pdf_path = Path(filename)
    if not pdf_path.exists():
        # Try in default directory
        pdf_path = DEFAULT_PDF_DIR / filename

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"Redacted PDF not found: {filename}")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
    )


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
