"""Surya OCR wrapper for word-level extraction.

Note: Surya >=0.15.0 has a regression bug where word/character bboxes are
incorrect (all words in a line share the same bbox). We work around this by
estimating word bboxes from the line bbox using character count proportions.
See: https://github.com/VikParuchuri/surya/issues/450
"""

from typing import Optional

import numpy as np
from PIL import Image

from ..models import BBox, Word

# Lazy imports to avoid slow startup
_det_predictor = None
_rec_predictor = None
_foundation_predictor = None


def _estimate_word_bboxes(
    line_polygon: list[list[float]],
    word_texts: list[str],
) -> list[tuple[float, float, float, float]]:
    """
    Estimate word bounding boxes by proportionally dividing line bbox.

    Since Surya >=0.15.0 has broken word-level bboxes (all words share the
    same bbox), we estimate individual word positions by dividing the line
    width based on character counts.

    Args:
        line_polygon: Line polygon [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        word_texts: List of word strings in the line

    Returns:
        List of (x0, y0, x1, y1) tuples for each word
    """
    if not word_texts:
        return []

    # Extract line bbox from polygon
    xs = [p[0] for p in line_polygon]
    ys = [p[1] for p in line_polygon]
    line_x0 = min(xs)
    line_x1 = max(xs)
    line_y0 = min(ys)
    line_y1 = max(ys)
    line_width = line_x1 - line_x0

    # Calculate total characters including spaces between words
    total_chars = sum(len(w) for w in word_texts) + len(word_texts) - 1
    if total_chars <= 0:
        total_chars = 1

    char_width = line_width / total_chars

    # Estimate each word's bbox
    bboxes = []
    current_x = line_x0

    for i, word in enumerate(word_texts):
        word_width = len(word) * char_width
        word_x1 = current_x + word_width

        # Clamp to line bounds
        word_x1 = min(word_x1, line_x1)

        bboxes.append((current_x, line_y0, word_x1, line_y1))

        # Move past word + space
        current_x = word_x1 + char_width

    return bboxes


def _get_predictors():
    """Get or create cached Surya predictors."""
    global _det_predictor, _rec_predictor, _foundation_predictor

    if _det_predictor is None:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.foundation import FoundationPredictor

        _det_predictor = DetectionPredictor()
        _foundation_predictor = FoundationPredictor()
        _rec_predictor = RecognitionPredictor(_foundation_predictor)

    return _det_predictor, _rec_predictor


def extract_words_surya(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
) -> list[Word]:
    """
    Extract words from an image using Surya OCR.

    Surya returns word-level bounding boxes with polygon coordinates,
    which we convert to PDF coordinate space.

    Args:
        image: RGB image as numpy array
        page_num: Page number for word IDs
        page_width: Page width in PDF points
        page_height: Page height in PDF points

    Returns:
        List of Word objects with bounding boxes in PDF coordinates
    """
    det_predictor, rec_predictor = _get_predictors()

    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)
    img_width, img_height = pil_image.size

    # Run OCR with word-level output
    results = rec_predictor(
        [pil_image],
        det_predictor=det_predictor,
        return_words=True,
    )

    words = []

    # Results is a list with one OCRResult per image
    if not results:
        return words

    ocr_result = results[0]

    # Scale factors for converting image coords to PDF coords
    scale_x = page_width / img_width
    scale_y = page_height / img_height

    for text_line in ocr_result.text_lines:
        # Get word texts - prefer Surya's word segmentation, fallback to split
        if text_line.words:
            word_texts = [w.text.strip() for w in text_line.words if w.text.strip()]
            word_confidences = [
                w.confidence if w.confidence else 0 for w in text_line.words
            ]
        else:
            word_texts = [w for w in text_line.text.split() if w]
            # Use line confidence for all words when no word-level data
            line_conf = text_line.confidence if text_line.confidence else 0
            word_confidences = [line_conf] * len(word_texts)

        if not word_texts:
            continue

        # Estimate word bboxes from line polygon (workaround for Surya bug)
        estimated_bboxes = _estimate_word_bboxes(text_line.polygon, word_texts)

        for word_text, (img_x0, img_y0, img_x1, img_y1), conf in zip(
            word_texts, estimated_bboxes, word_confidences
        ):
            # Convert image coords to PDF coords
            pdf_x0 = float(img_x0) * scale_x
            pdf_y0 = float(img_y0) * scale_y
            pdf_x1 = float(img_x1) * scale_x
            pdf_y1 = float(img_y1) * scale_y

            word = Word(
                word_id=-1,
                text=word_text,
                bbox=BBox(x0=pdf_x0, y0=pdf_y0, x1=pdf_x1, y1=pdf_y1),
                page=page_num,
                engine="surya",
                confidence=float(conf) * 100,  # Convert to 0-100 scale
            )
            words.append(word)

    return words


def is_surya_available() -> bool:
    """Check if Surya OCR is installed."""
    try:
        import surya
        return True
    except ImportError:
        return False


def get_surya_version() -> Optional[str]:
    """Get Surya version string, or None if not available."""
    try:
        from importlib.metadata import version
        return version("surya-ocr")
    except Exception:
        return None
