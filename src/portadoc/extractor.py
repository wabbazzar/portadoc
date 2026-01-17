"""Main word extraction pipeline."""

from pathlib import Path

from .models import BBox, Document, Page, Word
from .pdf import load_pdf
from .ocr.tesseract import extract_words_tesseract, is_tesseract_available


def extract_words(
    pdf_path: str | Path,
    dpi: int = 300,
    use_tesseract: bool = True,
) -> Document:
    """
    Extract words from a PDF document.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering pages
        use_tesseract: Whether to use Tesseract OCR

    Returns:
        Document with extracted words
    """
    pdf_path = Path(pdf_path)
    doc = Document(filename=pdf_path.name)

    # Check OCR availability
    tesseract_ok = is_tesseract_available() if use_tesseract else False

    if use_tesseract and not tesseract_ok:
        raise RuntimeError(
            "Tesseract OCR is not installed. "
            "Install with: sudo apt-get install tesseract-ocr tesseract-ocr-eng"
        )

    word_id_counter = 0

    with load_pdf(pdf_path, dpi=dpi) as pdf:
        for page_num, image, page_width, page_height in pdf.pages():
            page = Page(
                page_number=page_num,
                width=page_width,
                height=page_height,
            )

            # Extract words using Tesseract
            if tesseract_ok:
                tess_words = extract_words_tesseract(
                    image, page_num, page_width, page_height
                )

                # Assign word IDs and add to page
                for word in tess_words:
                    word.word_id = word_id_counter
                    word_id_counter += 1
                    # Clear engine field for harmonized (single-engine) output
                    word.engine = ""
                    page.words.append(word)

            doc.pages.append(page)

    return doc


def extract_to_csv(
    pdf_path: str | Path,
    output_path: str | Path,
    dpi: int = 300,
) -> Document:
    """
    Extract words from PDF and write to CSV.

    Args:
        pdf_path: Path to input PDF
        output_path: Path to output CSV
        dpi: Resolution for rendering

    Returns:
        Document with extracted words
    """
    from .output import write_csv

    doc = extract_words(pdf_path, dpi=dpi)
    write_csv(doc, output_path)
    return doc
