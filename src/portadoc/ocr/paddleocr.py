"""PaddleOCR wrapper for word-level extraction."""

from typing import Optional

import numpy as np

from ..models import BBox, Word

# Lazy import to avoid slow startup
_ocr: Optional["paddleocr.PaddleOCR"] = None


def _get_ocr(lang: str = "en", use_gpu: bool = False) -> "paddleocr.PaddleOCR":
    """Get or create a cached PaddleOCR instance."""
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )
    return _ocr


def extract_words_paddleocr(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    lang: str = "en",
    use_gpu: bool = False,
) -> list[Word]:
    """
    Extract words from an image using PaddleOCR.

    PaddleOCR returns line-level bounding boxes, so we decompose them into
    word-level boxes by splitting text on whitespace and proportionally
    dividing the bounding box (similar to EasyOCR approach).

    Args:
        image: RGB image as numpy array
        page_num: Page number for word IDs
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        lang: PaddleOCR language code (default: "en")
        use_gpu: Whether to use GPU (default: False for CPU-only)

    Returns:
        List of Word objects with bounding boxes in PDF coordinates
    """
    img_height, img_width = image.shape[:2]
    ocr = _get_ocr(lang=lang, use_gpu=use_gpu)

    # PaddleOCR returns: list[list[detection]]
    # Each detection: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
    result = ocr.ocr(image, cls=True)

    words = []
    scale_x = page_width / img_width
    scale_y = page_height / img_height

    # Result is list of pages, we only have one image
    if not result or not result[0]:
        return words

    for detection in result[0]:
        bbox_corners, (text, confidence) = detection
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
                engine="paddleocr",
                confidence=confidence * 100,  # Convert to 0-100 scale
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
                    engine="paddleocr",
                    confidence=confidence * 100,
                )
                words.append(word)

                current_x = word_x1

    return words


def is_paddleocr_available() -> bool:
    """Check if PaddleOCR is installed."""
    try:
        import paddleocr
        return True
    except ImportError:
        return False


def get_paddleocr_version() -> Optional[str]:
    """Get PaddleOCR version string, or None if not available."""
    try:
        import paddleocr
        return paddleocr.__version__
    except (ImportError, AttributeError):
        return None
