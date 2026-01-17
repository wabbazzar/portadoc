"""Confidence-based triage system for filtering OCR results."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .models import Word


class TriageLevel(Enum):
    """Triage strictness levels."""
    STRICT = "strict"      # Higher precision, may miss some words
    NORMAL = "normal"      # Balanced trade-off
    PERMISSIVE = "permissive"  # Maximum recall, more false positives


@dataclass
class TriageConfig:
    """Configuration for triage filtering."""

    # Minimum confidence thresholds (0-100 scale)
    min_confidence: float = 0.0

    # Minimum text length (filter out single chars with low conf)
    min_text_length: int = 1

    # Maximum aspect ratio (width/height) to filter noise
    max_aspect_ratio: float = 50.0

    # Minimum area in PDF points squared
    min_area: float = 4.0

    # Filter words that are just punctuation with low confidence
    filter_punctuation_low_conf: bool = True
    punctuation_conf_threshold: float = 50.0

    # Require dual-engine confirmation for low-confidence words
    require_dual_engine_if_low_conf: bool = False
    dual_engine_threshold: float = 30.0


# Preset configurations for each triage level
TRIAGE_PRESETS: dict[TriageLevel, TriageConfig] = {
    TriageLevel.STRICT: TriageConfig(
        min_confidence=40.0,
        min_text_length=1,
        max_aspect_ratio=30.0,
        min_area=10.0,
        filter_punctuation_low_conf=True,
        punctuation_conf_threshold=60.0,
        require_dual_engine_if_low_conf=True,
        dual_engine_threshold=50.0,
    ),
    TriageLevel.NORMAL: TriageConfig(
        min_confidence=20.0,
        min_text_length=1,
        max_aspect_ratio=40.0,
        min_area=6.0,
        filter_punctuation_low_conf=True,
        punctuation_conf_threshold=40.0,
        require_dual_engine_if_low_conf=False,
    ),
    TriageLevel.PERMISSIVE: TriageConfig(
        min_confidence=0.0,
        min_text_length=1,
        max_aspect_ratio=100.0,
        min_area=2.0,
        filter_punctuation_low_conf=False,
        require_dual_engine_if_low_conf=False,
    ),
}


def is_punctuation_only(text: str) -> bool:
    """Check if text contains only punctuation/whitespace."""
    import string
    return all(c in string.punctuation or c.isspace() for c in text)


def word_passes_triage(word: Word, config: TriageConfig) -> bool:
    """
    Check if a word passes triage filters.

    Args:
        word: Word to evaluate
        config: Triage configuration

    Returns:
        True if word passes, False if it should be filtered
    """
    # Skip pixel_detector results - these have confidence 0 but are fallbacks
    if word.engine == "pixel_detector":
        return True  # Always keep pixel detector results for recall

    # Text length filter
    if len(word.text.strip()) < config.min_text_length:
        return False

    # Confidence filter
    if word.confidence < config.min_confidence:
        # Check for dual-engine confirmation exception
        if config.require_dual_engine_if_low_conf:
            has_both = (
                word.tesseract_confidence is not None and
                word.easyocr_confidence is not None
            )
            if not has_both:
                return False
        elif word.confidence < config.min_confidence:
            return False

    # Aspect ratio filter (catch horizontal line artifacts)
    bbox = word.bbox
    if bbox.height > 0:
        aspect = bbox.width / bbox.height
        if aspect > config.max_aspect_ratio:
            return False

    # Area filter (catch tiny noise)
    if bbox.area < config.min_area:
        return False

    # Punctuation filter
    if config.filter_punctuation_low_conf:
        if is_punctuation_only(word.text):
            if word.confidence < config.punctuation_conf_threshold:
                return False

    return True


def triage_words(
    words: list[Word],
    level: TriageLevel = TriageLevel.NORMAL,
    config: Optional[TriageConfig] = None,
) -> list[Word]:
    """
    Filter words based on confidence and quality metrics.

    Args:
        words: List of words to filter
        level: Triage level preset (ignored if config provided)
        config: Custom configuration (overrides level)

    Returns:
        Filtered list of words that pass triage
    """
    if config is None:
        config = TRIAGE_PRESETS[level]

    return [w for w in words if word_passes_triage(w, config)]


def triage_stats(
    original: list[Word],
    triaged: list[Word],
) -> dict:
    """
    Calculate triage statistics.

    Args:
        original: Original word list before triage
        triaged: Word list after triage

    Returns:
        Dictionary with stats
    """
    removed = len(original) - len(triaged)

    # Analyze what was removed
    original_set = {id(w) for w in original}
    triaged_set = {id(w) for w in triaged}

    removed_words = [w for w in original if id(w) not in triaged_set]

    # Categorize removals
    low_conf = sum(1 for w in removed_words if w.confidence < 30)
    punctuation = sum(1 for w in removed_words if is_punctuation_only(w.text))
    tiny = sum(1 for w in removed_words if w.bbox.area < 10)

    return {
        "original_count": len(original),
        "triaged_count": len(triaged),
        "removed_count": removed,
        "removed_low_confidence": low_conf,
        "removed_punctuation": punctuation,
        "removed_tiny": tiny,
    }
