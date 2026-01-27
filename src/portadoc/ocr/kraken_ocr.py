"""Kraken OCR wrapper for word-level extraction."""

import logging
from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image

from ..models import BBox, Word

logger = logging.getLogger(__name__)

# Lazy-loaded models
_rec_model = None
_loaded_model_path = None

# Default model preferences (in order of preference)
# catmus-print-fondue-large: Best for printed text (modern, multi-language)
# mccatmus-nfd: Best for handwriting + printed + typewritten (multi-language)
DEFAULT_MODELS = [
    "catmus-print-fondue-large.mlmodel",  # Printed text (primary)
    "mccatmus-nfd.mlmodel",               # Handwriting/general (fallback)
]


def _find_model_path() -> Optional[str]:
    """Find the best available model in standard locations."""
    search_paths = [
        Path('models/kraken'),                    # Project directory
        Path.home() / '.local/share/kraken',      # User kraken directory
    ]

    # First, look for preferred models by name
    for model_name in DEFAULT_MODELS:
        for search_path in search_paths:
            model_path = search_path / model_name
            if model_path.exists():
                return str(model_path)

    # Fallback: find any .mlmodel file
    for search_path in search_paths:
        if search_path.exists():
            mlmodels = list(search_path.glob('*.mlmodel'))
            if mlmodels:
                return str(mlmodels[0])

    return None


def _get_model(model_path: Optional[str] = None):
    """Load or return cached Kraken recognition model."""
    global _rec_model, _loaded_model_path

    # If a specific model is requested and differs from loaded, reload
    if model_path and _loaded_model_path != model_path:
        _rec_model = None

    if _rec_model is None:
        from kraken.lib import models

        # Use provided model path or find best available
        if model_path is None:
            model_path = _find_model_path()

        if model_path is None:
            raise RuntimeError(
                "No Kraken model found. Download models to models/kraken/:\n"
                "  CATMuS-Print (printed): https://zenodo.org/records/10592716\n"
                "  McCATMuS (handwriting): https://zenodo.org/records/13788177"
            )

        logger.info(f"Loading Kraken model: {model_path}")
        _rec_model = models.load_any(model_path)
        _loaded_model_path = model_path

    return _rec_model


def extract_words_kraken(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    model_path: Optional[str] = None,
) -> list[Word]:
    """
    Extract words from an image using Kraken OCR.

    Args:
        image: RGB image as numpy array
        page_num: Page number for word IDs
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        model_path: Path to Kraken model file (optional)

    Returns:
        List of Word objects with bounding boxes in PDF coordinates
    """
    from kraken import blla, rpred

    rec_model = _get_model(model_path)

    img_height, img_width = image.shape[:2]
    pil_image = Image.fromarray(image)

    # Segment the page using default BLLA model
    segmentation = blla.segment(pil_image)

    # Scale factors for coordinate conversion
    scale_x = page_width / img_width
    scale_y = page_height / img_height

    words = []

    # Recognize each line
    for record in rpred.rpred(rec_model, pil_image, segmentation):
        text = record.prediction
        cuts = record.cuts

        # Get line bounding box from boundary polygon
        # record.boundary is a list of [x, y] points forming a polygon
        if not hasattr(record, 'boundary') or not record.boundary:
            continue

        boundary = record.boundary
        line_y0 = min(pt[1] for pt in boundary)
        line_y1 = max(pt[1] for pt in boundary)

        # Calculate average confidence for the line
        if record.confidences:
            line_conf = sum(record.confidences) / len(record.confidences) * 100
        else:
            line_conf = 0

        # Skip empty lines
        if not text or not text.strip():
            continue

        # Extract words from character cuts
        # In Kraken 6.x, cuts are 4-point polygons: ((p1, p2, p3, p4), ...)
        # Each point is [x, y]. We extract x coordinates for word boundaries.
        current_word = ""
        word_start_x = None
        word_end_x = None
        word_y0 = None
        word_y1 = None
        word_confs = []

        for i, char in enumerate(text):
            if i < len(cuts):
                cut = cuts[i]
                # Handle different cut formats
                if isinstance(cut, (list, tuple)) and len(cut) >= 2:
                    # New format: tuple of 4 points ((x,y), (x,y), (x,y), (x,y))
                    if isinstance(cut[0], (list, tuple)):
                        xs = [pt[0] for pt in cut]
                        ys = [pt[1] for pt in cut]
                        char_x0 = min(xs)
                        char_x1 = max(xs)
                        char_y0 = min(ys)
                        char_y1 = max(ys)
                    else:
                        # Old format: (x0, x1) - use line y bounds
                        char_x0, char_x1 = cut[0], cut[1] if len(cut) > 1 else cut[0]
                        char_y0, char_y1 = line_y0, line_y1
                else:
                    continue

                char_conf = record.confidences[i] if i < len(record.confidences) else 0
            else:
                continue

            if char.isspace():
                if current_word and word_start_x is not None:
                    # Create Word object
                    avg_conf = sum(word_confs) / len(word_confs) * 100 if word_confs else line_conf

                    word = Word(
                        word_id=-1,
                        text=current_word,
                        bbox=BBox(
                            x0=word_start_x * scale_x,
                            y0=word_y0 * scale_y,
                            x1=word_end_x * scale_x,
                            y1=word_y1 * scale_y,
                        ),
                        page=page_num,
                        engine="kraken",
                        confidence=avg_conf,
                    )
                    words.append(word)

                current_word = ""
                word_start_x = None
                word_end_x = None
                word_y0 = None
                word_y1 = None
                word_confs = []
            else:
                if word_start_x is None:
                    word_start_x = char_x0
                    word_y0 = char_y0
                    word_y1 = char_y1
                word_end_x = char_x1
                word_y0 = min(word_y0, char_y0) if word_y0 else char_y0
                word_y1 = max(word_y1, char_y1) if word_y1 else char_y1
                current_word += char
                word_confs.append(char_conf)

        # Don't forget the last word in line
        if current_word and word_start_x is not None:
            avg_conf = sum(word_confs) / len(word_confs) * 100 if word_confs else line_conf

            word = Word(
                word_id=-1,
                text=current_word,
                bbox=BBox(
                    x0=word_start_x * scale_x,
                    y0=word_y0 * scale_y,
                    x1=word_end_x * scale_x,
                    y1=word_y1 * scale_y,
                ),
                page=page_num,
                engine="kraken",
                confidence=avg_conf,
            )
            words.append(word)

    return words


def is_kraken_available() -> bool:
    """Check if Kraken is installed and a model is available."""
    try:
        import kraken
        # Also check if a model is available
        return _find_model_path() is not None
    except ImportError:
        return False


def get_available_model() -> Optional[str]:
    """Get the path to the available Kraken model, or None if not found."""
    return _find_model_path()


def get_kraken_version() -> Optional[str]:
    """Get Kraken version string, or None if not available."""
    try:
        from importlib.metadata import version
        return version("kraken")
    except Exception:
        return None
