"""FastAPI REST API for Portadoc."""

import asyncio
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .extractor import extract_words
from .models import Document


# Thread pool for CPU-bound OCR work
_executor = ThreadPoolExecutor(max_workers=2)


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


class JobStatus(str, Enum):
    """Job processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Async job status response."""

    job_id: str
    status: JobStatus
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[ExtractionResponse] = None


class JobSubmitResponse(BaseModel):
    """Response when submitting a new async job."""

    job_id: str
    status: JobStatus
    message: str


# In-memory job store (for production, use Redis or a database)
_jobs: dict[str, dict] = {}


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


def _run_extraction(
    job_id: str,
    pdf_path: Path,
    filename: str,
    dpi: int,
    triage: Optional[str],
    preprocess: str,
):
    """Run extraction in background thread."""
    try:
        _jobs[job_id]["status"] = JobStatus.PROCESSING

        doc = extract_words(
            pdf_path,
            dpi=dpi,
            triage=triage,
            preprocess=preprocess,
        )
        doc.filename = filename

        _jobs[job_id]["status"] = JobStatus.COMPLETED
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        _jobs[job_id]["result"] = document_to_response(doc)

    except Exception as e:
        _jobs[job_id]["status"] = JobStatus.FAILED
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        _jobs[job_id]["error"] = str(e)

    finally:
        # Clean up temp file
        if pdf_path.exists():
            pdf_path.unlink()


@app.post("/jobs", response_model=JobSubmitResponse)
async def submit_job(
    background_tasks: BackgroundTasks,
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
    Submit a PDF for async processing.

    Returns a job ID immediately. Poll /jobs/{job_id} for results.
    Recommended for large PDFs (>10 pages or >5MB).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Generate job ID and save file
    job_id = uuid.uuid4().hex
    temp_dir = Path(tempfile.gettempdir()) / "portadoc" / "jobs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{job_id}.pdf"

    content = await file.read()
    temp_path.write_bytes(content)

    # Create job record
    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None,
        "result": None,
    }

    # Submit to thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_extraction,
        job_id,
        temp_path,
        file.filename,
        dpi,
        triage,
        preprocess,
    )

    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job submitted. Poll /jobs/{job_id} for status.",
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get the status and result of an async job.

    Poll this endpoint until status is 'completed' or 'failed'.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job["completed_at"],
        error=job["error"],
        result=job["result"],
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a completed or failed job from the store."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job["status"] in (JobStatus.PENDING, JobStatus.PROCESSING):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a job that is still processing",
        )

    del _jobs[job_id]
    return {"message": "Job deleted"}


@app.get("/jobs", response_model=list[JobResponse])
async def list_jobs():
    """List all jobs (for debugging/monitoring)."""
    return [
        JobResponse(
            job_id=job["job_id"],
            status=job["status"],
            created_at=job["created_at"],
            completed_at=job["completed_at"],
            error=job["error"],
            result=job["result"],
        )
        for job in _jobs.values()
    ]


@app.get("/")
async def root():
    """API root - returns basic info."""
    return {
        "name": "Portadoc API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "jobs": "/jobs",
    }
