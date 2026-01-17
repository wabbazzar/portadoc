"""Main word extraction pipeline."""

from pathlib import Path
from typing import Optional

from .models import BBox, Document, Page, Word
from .pdf import load_pdf
from .ocr.tesseract import extract_words_tesseract, is_tesseract_available
from .ocr.easyocr import extract_words_easyocr, is_easyocr_available
from .detection import detect_missed_content
from .harmonize import harmonize_words
from .preprocess import PreprocessLevel, preprocess_for_ocr, auto_detect_quality
from .triage import TriageLevel, triage_words


def extract_words(
    pdf_path: str | Path,
    dpi: int = 300,
    use_tesseract: bool = True,
    use_easyocr: bool = True,
    use_pixel_detection: bool = True,
    gpu: bool = False,
    preprocess: Optional[str] = "auto",
    triage: Optional[str] = None,
) -> Document:
    """
    Extract words from a PDF document.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering pages
        use_tesseract: Whether to use Tesseract OCR
        use_easyocr: Whether to use EasyOCR
        use_pixel_detection: Whether to use pixel detection fallback
        gpu: Whether to use GPU for EasyOCR (default: False)
        preprocess: Preprocessing level - "none", "light", "standard", "aggressive", or "auto"
        triage: Triage level - "strict", "normal", "permissive", or None (no triage)

    Returns:
        Document with extracted words
    """
    pdf_path = Path(pdf_path)
    doc = Document(filename=pdf_path.name)

    # Check OCR availability
    tesseract_ok = is_tesseract_available() if use_tesseract else False
    easyocr_ok = is_easyocr_available() if use_easyocr else False

    if not tesseract_ok and not easyocr_ok:
        raise RuntimeError(
            "No OCR engine available. Install tesseract-ocr or easyocr."
        )

    word_id_counter = 0

    with load_pdf(pdf_path, dpi=dpi) as pdf:
        for page_num, image, page_width, page_height in pdf.pages():
            page = Page(
                page_number=page_num,
                width=page_width,
                height=page_height,
            )

            # Apply preprocessing if requested
            ocr_image = image
            if preprocess and preprocess != "none":
                if preprocess == "auto":
                    level = auto_detect_quality(image)
                else:
                    level = PreprocessLevel(preprocess)
                ocr_image = preprocess_for_ocr(image, level=level, return_rgb=True)

            # Extract words from each OCR engine
            tess_words = []
            easy_words = []

            if tesseract_ok:
                tess_words = extract_words_tesseract(
                    ocr_image, page_num, page_width, page_height
                )

            if easyocr_ok:
                easy_words = extract_words_easyocr(
                    ocr_image, page_num, page_width, page_height, gpu=gpu
                )

            # Harmonize results if both engines available, else use what we have
            if tess_words and easy_words:
                ocr_words = harmonize_words(tess_words, easy_words)
            elif tess_words:
                ocr_words = tess_words
            else:
                ocr_words = easy_words

            # Assign word IDs to OCR words and add to page
            for word in ocr_words:
                word.word_id = word_id_counter
                word_id_counter += 1
                word.engine = ""  # Harmonized output
                page.words.append(word)

            # Pixel detection fallback for missed content
            if use_pixel_detection:
                existing_bboxes = [w.bbox for w in page.words]
                pixel_words = detect_missed_content(
                    image, page_num, page_width, page_height,
                    existing_bboxes=existing_bboxes
                )

                # Assign word IDs to pixel-detected regions
                for word in pixel_words:
                    word.word_id = word_id_counter
                    word_id_counter += 1
                    page.words.append(word)

            doc.pages.append(page)

    # Apply triage filtering if requested
    if triage:
        triage_level = TriageLevel(triage)
        for page in doc.pages:
            page.words = triage_words(page.words, level=triage_level)

        # Reassign word IDs after triage
        word_id_counter = 0
        for page in doc.pages:
            for word in page.words:
                word.word_id = word_id_counter
                word_id_counter += 1

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
