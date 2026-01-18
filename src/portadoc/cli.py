"""Command line interface for Portadoc."""

import sys
from pathlib import Path

import click

from .extractor import extract_words, extract_document
from .output import format_output, write_harmonized_csv


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Portadoc - PDF word extraction for document redaction."""
    pass


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    help="Output file path (default: stdout for CSV)"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Output format (default: csv)"
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for PDF rendering (default: 300)"
)
@click.option(
    "--triage",
    type=click.Choice(["strict", "normal", "permissive"]),
    default=None,
    help="Triage level to filter low-confidence detections (default: none, requires --format json)"
)
@click.option(
    "--progress", "-p",
    is_flag=True,
    help="Show progress bar during extraction"
)
@click.option(
    "--preprocess",
    type=click.Choice(["none", "light", "standard", "aggressive", "degraded", "auto"]),
    default="auto",
    help="Preprocessing level for OCR (default: auto)"
)
@click.option(
    "--psm",
    type=int,
    default=3,
    help="Tesseract page segmentation mode 0-13 (default: 3)"
)
@click.option(
    "--oem",
    type=int,
    default=3,
    help="Tesseract OCR engine mode 0-3 (default: 3)"
)
@click.option(
    "--easyocr-decoder",
    type=click.Choice(["greedy", "beamsearch"]),
    default="greedy",
    help="EasyOCR decoder type (default: greedy)"
)
@click.option(
    "--easyocr-text-threshold",
    type=float,
    default=0.7,
    help="EasyOCR text detection threshold 0.0-1.0 (default: 0.7)"
)
@click.option(
    "--no-paddleocr",
    is_flag=True,
    help="Disable PaddleOCR engine"
)
@click.option(
    "--no-tesseract",
    is_flag=True,
    help="Disable Tesseract OCR (not recommended)"
)
@click.option(
    "--no-easyocr",
    is_flag=True,
    help="Disable EasyOCR"
)
@click.option(
    "--upscale",
    type=click.Choice(["none", "2", "4"]),
    default="none",
    help="Super-resolution upscale factor (default: none)"
)
@click.option(
    "--upscale-method",
    type=click.Choice(["espcn", "fsrcnn", "bicubic", "lanczos"]),
    default="espcn",
    help="Super-resolution method (default: espcn)"
)
@click.option(
    "--no-doctr",
    is_flag=True,
    help="Disable docTR engine"
)
@click.option(
    "--no-surya",
    is_flag=True,
    help="Disable Surya OCR engine (word-level, SOTA accuracy)"
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to config file (YAML) for harmonization thresholds"
)
@click.option(
    "--primary",
    type=click.Choice(["tesseract", "surya", "doctr", "easyocr", "paddleocr"]),
    default=None,
    help="Primary OCR engine for bbox authority (default: from config, usually tesseract)"
)
@click.option(
    "--no-reading-order",
    is_flag=True,
    help="Disable geometric reading order (use simple y,x coordinate sorting)"
)
@click.option(
    "--align/--no-align",
    default=True,
    help="Auto-detect and correct page orientation (default: enabled)"
)
def extract(
    pdf_path: Path,
    output: Path | None,
    format: str,
    dpi: int,
    triage: str | None,
    progress: bool,
    preprocess: str,
    psm: int,
    oem: int,
    easyocr_decoder: str,
    easyocr_text_threshold: float,
    no_paddleocr: bool,
    no_tesseract: bool,
    no_easyocr: bool,
    upscale: str,
    upscale_method: str,
    no_doctr: bool,
    no_surya: bool,
    config: Path | None,
    primary: str | None,
    no_reading_order: bool,
    align: bool,
):
    """
    Extract words and bounding boxes from a PDF.

    PDF_PATH is the path to the input PDF file.

    Uses smart harmonization with configurable primary engine and
    secondary engines (EasyOCR, PaddleOCR, docTR, Surya) for text voting.
    """
    try:
        # Set up progress callback if requested
        progress_bar = None
        if progress:
            from .pdf import load_pdf
            with load_pdf(pdf_path, dpi=dpi) as pdf:
                total_pages = len(pdf)
            progress_bar = click.progressbar(
                length=total_pages,
                label="Extracting",
                file=sys.stderr,
            )
            progress_bar.__enter__()

        def progress_callback(page_num, total, stage):
            if progress_bar:
                progress_bar.update(1)

        # Parse upscale factor
        upscale_factor = None if upscale == "none" else int(upscale)

        # JSON format requires Document object for backward compat
        if format == "json":
            if triage:
                click.echo("Note: Using triage filtering with JSON output", err=True)

            doc = extract_document(
                pdf_path,
                dpi=dpi,
                triage=triage,
                preprocess=preprocess,
                upscale=upscale_factor,
                upscale_method=upscale_method,
                tesseract_psm=psm,
                tesseract_oem=oem,
                easyocr_decoder=easyocr_decoder,
                easyocr_text_threshold=easyocr_text_threshold,
                use_paddleocr=not no_paddleocr,
                use_tesseract=not no_tesseract,
                use_easyocr=not no_easyocr,
                use_doctr=not no_doctr,
                use_surya=not no_surya,
                config_path=config,
                progress_callback=progress_callback if progress else None,
                use_reading_order=not no_reading_order,
                primary_engine=primary,
                align_pages=align,
            )

            if progress_bar:
                progress_bar.__exit__(None, None, None)

            if output:
                format_output(doc, output, format=format)
                click.echo(f"Extracted {doc.total_words} words to {output}", err=True)
            else:
                click.echo("Error: --output is required for JSON format", err=True)
                sys.exit(1)
        else:
            # CSV format - use full HarmonizedWord output
            if triage:
                click.echo("Warning: --triage is only supported with --format json", err=True)

            harmonized_words = extract_words(
                pdf_path,
                dpi=dpi,
                preprocess=preprocess,
                upscale=upscale_factor,
                upscale_method=upscale_method,
                tesseract_psm=psm,
                tesseract_oem=oem,
                easyocr_decoder=easyocr_decoder,
                easyocr_text_threshold=easyocr_text_threshold,
                use_paddleocr=not no_paddleocr,
                use_tesseract=not no_tesseract,
                use_easyocr=not no_easyocr,
                use_doctr=not no_doctr,
                use_surya=not no_surya,
                config_path=config,
                progress_callback=progress_callback if progress else None,
                use_reading_order=not no_reading_order,
                primary_engine=primary,
                align_pages=align,
            )

            if progress_bar:
                progress_bar.__exit__(None, None, None)

            if output:
                write_harmonized_csv(harmonized_words, output)
                click.echo(f"Extracted {len(harmonized_words)} words to {output}", err=True)
            else:
                write_harmonized_csv(harmonized_words, sys.stdout)

    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@main.command()
def check():
    """Check OCR engine availability."""
    from .ocr.tesseract import is_tesseract_available, get_tesseract_version
    from .ocr.easyocr import is_easyocr_available, get_easyocr_version
    from .ocr.paddleocr import is_paddleocr_available, get_paddleocr_version
    from .ocr.doctr_ocr import is_doctr_available, get_doctr_version
    from .ocr.surya_ocr import is_surya_available, get_surya_version

    click.echo("OCR Engine Status:")
    click.echo("-" * 40)

    if is_tesseract_available():
        version = get_tesseract_version()
        click.echo(f"Tesseract:  OK (version {version})")
    else:
        click.echo("Tesseract:  NOT FOUND (REQUIRED)")
        click.echo("  Install with: sudo apt-get install tesseract-ocr tesseract-ocr-eng")

    if is_easyocr_available():
        version = get_easyocr_version()
        click.echo(f"EasyOCR:    OK (version {version})")
    else:
        click.echo("EasyOCR:    NOT FOUND")
        click.echo("  Install with: pip install easyocr")

    if is_paddleocr_available():
        version = get_paddleocr_version()
        click.echo(f"PaddleOCR:  OK (version {version})")
    else:
        click.echo("PaddleOCR:  NOT FOUND")
        click.echo("  Install with: pip install paddleocr")

    if is_doctr_available():
        version = get_doctr_version()
        click.echo(f"docTR:      OK (version {version})")
    else:
        click.echo("docTR:      NOT FOUND")
        click.echo("  Install with: pip install python-doctr[torch]")

    if is_surya_available():
        version = get_surya_version()
        click.echo(f"Surya:      OK (version {version})")
    else:
        click.echo("Surya:      NOT FOUND")
        click.echo("  Install with: pip install surya-ocr")


@main.command("eval")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.argument("ground_truth", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for PDF rendering (default: 300)"
)
@click.option(
    "--iou-threshold",
    type=float,
    default=0.5,
    help="IoU threshold for matching (default: 0.5)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed match information"
)
@click.option(
    "--preprocess",
    type=click.Choice(["none", "light", "standard", "aggressive", "degraded", "auto"]),
    default="auto",
    help="Preprocessing level for OCR (default: auto)"
)
@click.option(
    "--psm",
    type=int,
    default=3,
    help="Tesseract page segmentation mode 0-13 (default: 3)"
)
@click.option(
    "--oem",
    type=int,
    default=3,
    help="Tesseract OCR engine mode 0-3 (default: 3)"
)
@click.option(
    "--easyocr-decoder",
    type=click.Choice(["greedy", "beamsearch"]),
    default="greedy",
    help="EasyOCR decoder type (default: greedy)"
)
@click.option(
    "--easyocr-text-threshold",
    type=float,
    default=0.7,
    help="EasyOCR text detection threshold 0.0-1.0 (default: 0.7)"
)
@click.option(
    "--no-paddleocr",
    is_flag=True,
    help="Disable PaddleOCR engine"
)
@click.option(
    "--no-tesseract",
    is_flag=True,
    help="Disable Tesseract OCR (not recommended)"
)
@click.option(
    "--no-easyocr",
    is_flag=True,
    help="Disable EasyOCR"
)
@click.option(
    "--upscale",
    type=click.Choice(["none", "2", "4"]),
    default="none",
    help="Super-resolution upscale factor (default: none)"
)
@click.option(
    "--upscale-method",
    type=click.Choice(["espcn", "fsrcnn", "bicubic", "lanczos"]),
    default="espcn",
    help="Super-resolution method (default: espcn)"
)
@click.option(
    "--no-doctr",
    is_flag=True,
    help="Disable docTR engine"
)
@click.option(
    "--no-surya",
    is_flag=True,
    help="Disable Surya OCR engine (word-level, SOTA accuracy)"
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to config file (YAML) for harmonization thresholds"
)
@click.option(
    "--align/--no-align",
    default=True,
    help="Auto-detect and correct page orientation (default: enabled)"
)
def evaluate_cmd(
    pdf_path: Path,
    ground_truth: Path,
    dpi: int,
    iou_threshold: float,
    verbose: bool,
    preprocess: str,
    psm: int,
    oem: int,
    easyocr_decoder: str,
    easyocr_text_threshold: float,
    no_paddleocr: bool,
    no_tesseract: bool,
    no_easyocr: bool,
    upscale: str,
    upscale_method: str,
    no_doctr: bool,
    no_surya: bool,
    config: Path | None,
    align: bool,
):
    """
    Evaluate extraction against ground truth CSV.

    PDF_PATH is the path to the input PDF file.
    GROUND_TRUTH is the path to the ground truth CSV.
    """
    from .metrics import evaluate

    try:
        # Parse upscale factor
        upscale_factor = None if upscale == "none" else int(upscale)

        harmonized_words = extract_words(
            pdf_path,
            dpi=dpi,
            preprocess=preprocess,
            upscale=upscale_factor,
            upscale_method=upscale_method,
            tesseract_psm=psm,
            tesseract_oem=oem,
            easyocr_decoder=easyocr_decoder,
            easyocr_text_threshold=easyocr_text_threshold,
            use_paddleocr=not no_paddleocr,
            use_tesseract=not no_tesseract,
            use_easyocr=not no_easyocr,
            use_doctr=not no_doctr,
            use_surya=not no_surya,
            config_path=config,
            align_pages=align,
        )

        # Evaluate
        result = evaluate(harmonized_words, ground_truth, iou_threshold=iou_threshold)

        # Print summary
        click.echo(result.summary())

        if verbose and result.unmatched_gt:
            click.echo("\nMissed Ground Truth Words:")
            for w in result.unmatched_gt[:10]:
                click.echo(f"  Page {w.page}: {repr(w.text)}")
            if len(result.unmatched_gt) > 10:
                click.echo(f"  ... and {len(result.unmatched_gt) - 10} more")

        if verbose and result.false_positive_words:
            click.echo("\nFalse Positive Words:")
            for w in result.false_positive_words[:10]:
                click.echo(f"  Page {w.page}: {repr(w.text)}")
            if len(result.false_positive_words) > 10:
                click.echo(f"  ... and {len(result.false_positive_words) - 10} more")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)"
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind to (default: 8000)"
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development"
)
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI REST server with web visualization UI."""
    import uvicorn

    click.echo(f"Starting Portadoc Web UI at http://{host}:{port}")
    click.echo("API docs available at /docs")

    uvicorn.run(
        "portadoc.web.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
