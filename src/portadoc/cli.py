"""Command line interface for Portadoc."""

import sys
from pathlib import Path

import click

from .extractor import extract_words
from .output import format_output


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
    help="Output file path (default: stdout for CSV, required for JSON)"
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
    help="Triage level to filter low-confidence detections (default: none)"
)
@click.option(
    "--progress", "-p",
    is_flag=True,
    help="Show progress bar during extraction"
)
@click.option(
    "--preprocess",
    type=click.Choice(["none", "light", "standard", "aggressive", "auto"]),
    default="auto",
    help="Preprocessing level for OCR (default: auto)"
)
def extract(pdf_path: Path, output: Path | None, format: str, dpi: int, triage: str | None, progress: bool, preprocess: str):
    """
    Extract words and bounding boxes from a PDF.

    PDF_PATH is the path to the input PDF file.
    """
    try:
        # Set up progress callback if requested
        progress_bar = None
        if progress:
            # We need to know total pages first
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

        # Extract words
        doc = extract_words(
            pdf_path,
            dpi=dpi,
            triage=triage,
            preprocess=preprocess,
            progress_callback=progress_callback if progress else None,
        )

        if progress_bar:
            progress_bar.__exit__(None, None, None)

        # Output results
        if output:
            format_output(doc, output, format=format)
            click.echo(f"Extracted {doc.total_words} words to {output}", err=True)
        else:
            if format == "json":
                click.echo("Error: --output is required for JSON format", err=True)
                sys.exit(1)
            # Write CSV to stdout
            from .output import write_csv
            write_csv(doc, sys.stdout)

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

    click.echo("OCR Engine Status:")
    click.echo("-" * 40)

    if is_tesseract_available():
        version = get_tesseract_version()
        click.echo(f"Tesseract: OK (version {version})")
    else:
        click.echo("Tesseract: NOT FOUND")
        click.echo("  Install with: sudo apt-get install tesseract-ocr tesseract-ocr-eng")

    if is_easyocr_available():
        version = get_easyocr_version()
        click.echo(f"EasyOCR:   OK (version {version})")
    else:
        click.echo("EasyOCR:   NOT FOUND")
        click.echo("  Install with: pip install easyocr")


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
    "--triage",
    type=click.Choice(["strict", "normal", "permissive"]),
    default=None,
    help="Triage level to filter low-confidence detections"
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
    type=click.Choice(["none", "light", "standard", "aggressive", "auto"]),
    default="auto",
    help="Preprocessing level for OCR (default: auto)"
)
def evaluate_cmd(
    pdf_path: Path,
    ground_truth: Path,
    dpi: int,
    triage: str | None,
    iou_threshold: float,
    verbose: bool,
    preprocess: str,
):
    """
    Evaluate extraction against ground truth CSV.

    PDF_PATH is the path to the input PDF file.
    GROUND_TRUTH is the path to the ground truth CSV.
    """
    from .metrics import evaluate

    try:
        # Extract words
        doc = extract_words(pdf_path, dpi=dpi, triage=triage, preprocess=preprocess)

        # Evaluate
        result = evaluate(doc, ground_truth, iou_threshold=iou_threshold)

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
    """Start the FastAPI REST server."""
    import uvicorn

    click.echo(f"Starting Portadoc API server at http://{host}:{port}")
    click.echo("API docs available at /docs")

    uvicorn.run(
        "portadoc.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
