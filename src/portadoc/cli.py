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
def extract(pdf_path: Path, output: Path | None, format: str, dpi: int):
    """
    Extract words and bounding boxes from a PDF.

    PDF_PATH is the path to the input PDF file.
    """
    try:
        # Extract words
        doc = extract_words(pdf_path, dpi=dpi)

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


if __name__ == "__main__":
    main()
