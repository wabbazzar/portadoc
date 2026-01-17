"""docTR OCR wrapper for word-level extraction."""

from typing import Optional

import numpy as np

from ..models import BBox, Word

# Lazy import to avoid slow startup
_predictor = None


def _get_predictor():
    """Get or create a cached docTR predictor."""
    global _predictor
    if _predictor is None:
        from doctr.models import ocr_predictor
        _predictor = ocr_predictor(pretrained=True)
    return _predictor


def extract_words_doctr(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
) -> list[Word]:
    """
    Extract words from an image using docTR.

    docTR returns word-level bounding boxes with normalized coordinates (0-1),
    which we convert to PDF coordinate space.

    Args:
        image: RGB image as numpy array
        page_num: Page number for word IDs
        page_width: Page width in PDF points
        page_height: Page height in PDF points

    Returns:
        List of Word objects with bounding boxes in PDF coordinates
    """
    predictor = _get_predictor()

    # docTR expects a list of images
    result = predictor([image])

    words = []

    # Result has pages -> blocks -> lines -> words
    for page in result.pages:
        # Page dimensions from docTR (height, width)
        img_height, img_width = page.dimensions

        for block in page.blocks:
            for line in block.lines:
                for word_obj in line.words:
                    text = word_obj.value.strip()
                    if not text:
                        continue

                    # docTR geometry is ((x0, y0), (x1, y1)) normalized 0-1
                    (x0_norm, y0_norm), (x1_norm, y1_norm) = word_obj.geometry

                    # Convert normalized coords to PDF coords
                    pdf_x0 = float(x0_norm) * page_width
                    pdf_y0 = float(y0_norm) * page_height
                    pdf_x1 = float(x1_norm) * page_width
                    pdf_y1 = float(y1_norm) * page_height

                    word = Word(
                        word_id=-1,
                        text=text,
                        bbox=BBox(x0=pdf_x0, y0=pdf_y0, x1=pdf_x1, y1=pdf_y1),
                        page=page_num,
                        engine="doctr",
                        confidence=float(word_obj.confidence) * 100,  # Convert to 0-100 scale
                    )
                    words.append(word)

    return words


def is_doctr_available() -> bool:
    """Check if docTR is installed."""
    try:
        import doctr
        return True
    except ImportError:
        return False


def get_doctr_version() -> Optional[str]:
    """Get docTR version string, or None if not available."""
    try:
        import doctr
        return doctr.__version__
    except (ImportError, AttributeError):
        return None
