"""EasyOCR wrapper for word-level extraction."""

from typing import Optional

import numpy as np

from ..models import BBox, Word

# Lazy import to avoid slow startup
_reader: Optional["easyocr.Reader"] = None


def _get_reader(lang: list[str] = ["en"], gpu: bool = False) -> "easyocr.Reader":
    """Get or create a cached EasyOCR reader."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(lang, gpu=gpu)
    return _reader


def extract_words_easyocr(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    lang: list[str] = ["en"],
    gpu: bool = False,
    decoder: str = "greedy",
    contrast_ths: float = 0.1,
    adjust_contrast: float = 0.5,
    text_threshold: float = 0.7,
    width_ths: float = 0.5,
) -> list[Word]:
    """
    Extract words from an image using EasyOCR.

    EasyOCR returns line-level bounding boxes, so we decompose them into
    word-level boxes by splitting text on whitespace and proportionally
    dividing the bounding box.

    Args:
        image: RGB image as numpy array
        page_num: Page number for word IDs
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        lang: List of language codes (default: English)
        gpu: Whether to use GPU (default: False for CPU-only)
        decoder: Decoder type - "greedy" or "beamsearch" (default: greedy)
        contrast_ths: Contrast threshold for auto-adjustment (default: 0.1)
        adjust_contrast: Contrast adjustment factor (default: 0.5)
        text_threshold: Text detection confidence threshold (default: 0.7)
        width_ths: Width threshold for merging boxes (default: 0.5)

    Returns:
        List of Word objects with bounding boxes in PDF coordinates
    """
    img_height, img_width = image.shape[:2]
    reader = _get_reader(lang, gpu=gpu)

    # EasyOCR returns: list of (bbox, text, confidence)
    # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (4 corners)
    results = reader.readtext(
        image,
        decoder=decoder,
        contrast_ths=contrast_ths,
        adjust_contrast=adjust_contrast,
        text_threshold=text_threshold,
        width_ths=width_ths,
    )

    words = []
    scale_x = page_width / img_width
    scale_y = page_height / img_height

    for bbox_corners, text, confidence in results:
        text = text.strip()
        if not text:
            continue

        # Convert 4-corner bbox to (x0, y0, x1, y1)
        xs = [p[0] for p in bbox_corners]
        ys = [p[1] for p in bbox_corners]
        line_x0 = min(xs)
        line_y0 = min(ys)
        line_x1 = max(xs)
        line_y1 = max(ys)
        line_width = line_x1 - line_x0

        # Split text into words
        text_words = text.split()
        if not text_words:
            continue

        # If single word, use the whole bounding box
        if len(text_words) == 1:
            pdf_x0 = line_x0 * scale_x
            pdf_y0 = line_y0 * scale_y
            pdf_x1 = line_x1 * scale_x
            pdf_y1 = line_y1 * scale_y

            word = Word(
                word_id=-1,
                text=text_words[0],
                bbox=BBox(x0=pdf_x0, y0=pdf_y0, x1=pdf_x1, y1=pdf_y1),
                page=page_num,
                engine="easyocr",
                confidence=confidence * 100,  # Convert to 0-100 scale
                easyocr_confidence=confidence * 100,
            )
            words.append(word)
        else:
            # Multiple words: distribute bounding box proportionally
            total_chars = sum(len(w) for w in text_words)
            # Add spacing estimate (half character per gap)
            total_chars_with_space = total_chars + (len(text_words) - 1) * 0.5

            current_x = line_x0
            for word_text in text_words:
                # Estimate word width proportionally
                word_proportion = (len(word_text) + 0.25) / total_chars_with_space
                word_width = line_width * word_proportion

                word_x0 = current_x
                word_x1 = current_x + word_width

                # Convert to PDF coordinates
                pdf_x0 = word_x0 * scale_x
                pdf_y0 = line_y0 * scale_y
                pdf_x1 = word_x1 * scale_x
                pdf_y1 = line_y1 * scale_y

                word = Word(
                    word_id=-1,
                    text=word_text,
                    bbox=BBox(x0=pdf_x0, y0=pdf_y0, x1=pdf_x1, y1=pdf_y1),
                    page=page_num,
                    engine="easyocr",
                    confidence=confidence * 100,
                    easyocr_confidence=confidence * 100,
                )
                words.append(word)

                current_x = word_x1

    return words


def is_easyocr_available() -> bool:
    """Check if EasyOCR is installed."""
    try:
        import easyocr
        return True
    except ImportError:
        return False


def get_easyocr_version() -> Optional[str]:
    """Get EasyOCR version string, or None if not available."""
    try:
        import easyocr
        return easyocr.__version__
    except (ImportError, AttributeError):
        return None
