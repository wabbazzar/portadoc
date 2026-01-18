"""Output formatters for word extraction results."""

import csv
import json
from pathlib import Path
from typing import TextIO

from .models import Document, Word, HarmonizedWord


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


def write_harmonized_csv(
    words: list[HarmonizedWord],
    output: str | Path | TextIO,
    include_distances: bool = True,
    include_pixel: bool = True,
) -> None:
    """
    Write HarmonizedWord results to CSV with full tracking.

    CSV columns (dense format):
    word_id, page, x0, y0, x1, y1, text, status, source, conf,
    tess, easy, doctr, paddle, surya, dist_tess, dist_easy, dist_doctr, dist_paddle, dist_surya

    Args:
        words: List of HarmonizedWord objects
        output: Output file path or file handle
        include_distances: Whether to include Levenshtein distance columns
        include_pixel: Whether to include status=pixel rows
    """
    # Base fieldnames
    fieldnames = [
        "word_id", "page", "x0", "y0", "x1", "y1",
        "text", "status", "source", "conf", "rotation",
        "tess", "easy", "doctr", "paddle", "surya",
    ]

    if include_distances:
        fieldnames.extend(["dist_tess", "dist_easy", "dist_doctr", "dist_paddle", "dist_surya"])

    def write_to_file(f: TextIO) -> None:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for hw in words:
            # Skip pixel detections if not requested
            if not include_pixel and hw.status == "pixel":
                continue

            row = {
                "word_id": hw.word_id,
                "page": hw.page,
                "x0": round(hw.bbox.x0, 2),
                "y0": round(hw.bbox.y0, 2),
                "x1": round(hw.bbox.x1, 2),
                "y1": round(hw.bbox.y1, 2),
                "text": hw.text,
                "status": hw.status,
                "source": hw.source,
                "conf": round(hw.confidence, 1),
                "rotation": hw.rotation,
                "tess": hw.tess_text,
                "easy": hw.easy_text,
                "doctr": hw.doctr_text,
                "paddle": hw.paddle_text,
                "surya": hw.surya_text,
            }

            if include_distances:
                row["dist_tess"] = hw.dist_tess if hw.dist_tess >= 0 else ""
                row["dist_easy"] = hw.dist_easy if hw.dist_easy >= 0 else ""
                row["dist_doctr"] = hw.dist_doctr if hw.dist_doctr >= 0 else ""
                row["dist_paddle"] = hw.dist_paddle if hw.dist_paddle >= 0 else ""
                row["dist_surya"] = hw.dist_surya if hw.dist_surya >= 0 else ""

            writer.writerow(row)

    if isinstance(output, (str, Path)):
        with open(output, "w", newline="", encoding="utf-8") as f:
            write_to_file(f)
    else:
        write_to_file(output)
