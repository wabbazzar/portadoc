"""Output formatters for word extraction results."""

import csv
import json
from pathlib import Path
from typing import TextIO

from .models import Document, Word


def write_csv(document: Document, output: str | Path | TextIO) -> None:
    """
    Write document words to CSV format.

    CSV columns: page, word_id, text, x0, y0, x1, y1, engine, ocr_confidence
    """
    fieldnames = ["page", "word_id", "text", "x0", "y0", "x1", "y1", "engine", "ocr_confidence"]

    def write_to_file(f: TextIO) -> None:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for word in document.all_words():
            writer.writerow({
                "page": word.page,
                "word_id": word.word_id,
                "text": word.text,
                "x0": word.bbox.x0,
                "y0": word.bbox.y0,
                "x1": word.bbox.x1,
                "y1": word.bbox.y1,
                "engine": word.engine,
                "ocr_confidence": word.confidence,
            })

    if isinstance(output, (str, Path)):
        with open(output, "w", newline="", encoding="utf-8") as f:
            write_to_file(f)
    else:
        write_to_file(output)


def write_json(document: Document, output: str | Path | TextIO, indent: int = 2) -> None:
    """
    Write document words to JSON format.

    JSON structure:
    {
        "filename": "...",
        "pages": [
            {
                "page_number": 0,
                "width": 612.0,
                "height": 792.0,
                "words": [...]
            }
        ]
    }
    """
    data = {
        "filename": document.filename,
        "total_words": document.total_words,
        "pages": []
    }

    for page in document.pages:
        page_data = {
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "words": []
        }

        for word in page.words:
            word_data = {
                "word_id": word.word_id,
                "text": word.text,
                "bbox": [word.bbox.x0, word.bbox.y0, word.bbox.x1, word.bbox.y1],
                "engine": word.engine if word.engine else "harmonized",
                "confidence": word.confidence,
            }
            page_data["words"].append(word_data)

        data["pages"].append(page_data)

    if isinstance(output, (str, Path)):
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
    else:
        json.dump(data, output, indent=indent)


def format_output(document: Document, output_path: str | Path, format: str = "csv") -> None:
    """
    Write document to file in specified format.

    Args:
        document: Document with extracted words
        output_path: Output file path
        format: Output format ('csv' or 'json')
    """
    if format == "csv":
        write_csv(document, output_path)
    elif format == "json":
        write_json(document, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
