"""Apply redactions to PDF by drawing rectangles over marked words."""

import csv
from pathlib import Path
from typing import NamedTuple

import pymupdf


class RedactionBox(NamedTuple):
    """A bounding box to redact on a specific page."""
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


def load_redactions_from_csv(csv_path: Path) -> dict[int, list[RedactionBox]]:
    """
    Load redaction boxes from a CSV file.

    Args:
        csv_path: Path to CSV with redact column.

    Returns:
        Dictionary mapping page number to list of RedactionBox.
    """
    redactions: dict[int, list[RedactionBox]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only include rows marked for redaction
            if row.get("redact") != "true":
                continue

            page = int(row.get("page", 0))
            x0 = float(row.get("x0", 0))
            y0 = float(row.get("y0", 0))
            x1 = float(row.get("x1", 0))
            y1 = float(row.get("y1", 0))

            box = RedactionBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1)

            if page not in redactions:
                redactions[page] = []
            redactions[page].append(box)

    return redactions


def apply_redactions(
    pdf_path: Path,
    words_csv: Path,
    output_path: Path,
    color: tuple[float, float, float] = (0, 0, 0),
    padding: float = 0,
) -> int:
    """
    Draw filled rectangles over redacted words in a PDF.

    Args:
        pdf_path: Path to input PDF.
        words_csv: Path to CSV with redact column.
        output_path: Path for output redacted PDF.
        color: RGB tuple (0-1 range) for redaction color. Default black.
        padding: Extra padding around boxes in points.

    Returns:
        Number of redactions applied.
    """
    # Load redaction boxes grouped by page
    redactions = load_redactions_from_csv(words_csv)

    if not redactions:
        # No redactions - just copy the file
        import shutil
        shutil.copy(pdf_path, output_path)
        return 0

    # Open PDF
    doc = pymupdf.open(pdf_path)
    total_redactions = 0

    for page_num, boxes in redactions.items():
        if page_num >= len(doc):
            continue

        page = doc[page_num]

        for box in boxes:
            # Create rectangle with optional padding
            rect = pymupdf.Rect(
                box.x0 - padding,
                box.y0 - padding,
                box.x1 + padding,
                box.y1 + padding,
            )

            # Draw filled rectangle
            page.draw_rect(rect, color=color, fill=color)
            total_redactions += 1

    # Save to output path
    doc.save(output_path)
    doc.close()

    return total_redactions


def apply_redactions_preview(
    pdf_path: Path,
    words_csv: Path,
    output_path: Path,
    border_color: tuple[float, float, float] = (1, 0, 0),
    fill_color: tuple[float, float, float] | None = None,
    border_width: float = 1.5,
) -> int:
    """
    Draw preview rectangles (outlines) over redacted words without filling.

    Useful for reviewing what will be redacted before final application.

    Args:
        pdf_path: Path to input PDF.
        words_csv: Path to CSV with redact column.
        output_path: Path for output preview PDF.
        border_color: RGB tuple for border color. Default red.
        fill_color: Optional RGB tuple for semi-transparent fill.
        border_width: Width of border line in points.

    Returns:
        Number of redactions marked.
    """
    redactions = load_redactions_from_csv(words_csv)

    if not redactions:
        import shutil
        shutil.copy(pdf_path, output_path)
        return 0

    doc = pymupdf.open(pdf_path)
    total_marked = 0

    for page_num, boxes in redactions.items():
        if page_num >= len(doc):
            continue

        page = doc[page_num]

        for box in boxes:
            rect = pymupdf.Rect(box.x0, box.y0, box.x1, box.y1)

            # Draw outline with optional transparent fill
            page.draw_rect(
                rect,
                color=border_color,
                fill=fill_color,
                width=border_width,
            )
            total_marked += 1

    doc.save(output_path)
    doc.close()

    return total_marked
