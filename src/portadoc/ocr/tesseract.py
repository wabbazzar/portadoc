"""Tesseract OCR wrapper for word-level extraction."""

from typing import Optional

import numpy as np
import pytesseract
from pytesseract import Output

from ..models import BBox, Word


def extract_words_tesseract(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    lang: str = "eng",
    config: str = "--psm 3 --oem 3"
) -> list[Word]:
    """
    Extract words from an image using Tesseract OCR.

    Args:
        image: RGB image as numpy array
        page_num: Page number for word IDs
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        lang: Tesseract language code
        config: Tesseract configuration string

    Returns:
        List of Word objects with bounding boxes in PDF coordinates
    """
    img_height, img_width = image.shape[:2]

    # Get word-level data from Tesseract
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=Output.DICT
    )

    words = []
    n_boxes = len(data["text"])

    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])

        # Skip empty text or low confidence noise
        if not text or conf < 0:
            continue

        # Get bounding box in image coordinates
        img_x = data["left"][i]
        img_y = data["top"][i]
        img_w = data["width"][i]
        img_h = data["height"][i]

        # Convert to PDF coordinates
        scale_x = page_width / img_width
        scale_y = page_height / img_height

        x0 = img_x * scale_x
        y0 = img_y * scale_y
        x1 = (img_x + img_w) * scale_x
        y1 = (img_y + img_h) * scale_y

        word = Word(
            word_id=-1,  # Will be assigned later during harmonization
            text=text,
            bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
            page=page_num,
            engine="tesseract",
            confidence=conf,
            tesseract_confidence=conf,
        )
        words.append(word)

    return words


def is_tesseract_available() -> bool:
    """Check if Tesseract is installed and accessible."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False


def get_tesseract_version() -> Optional[str]:
    """Get Tesseract version string, or None if not available."""
    try:
        return str(pytesseract.get_tesseract_version())
    except pytesseract.TesseractNotFoundError:
        return None
