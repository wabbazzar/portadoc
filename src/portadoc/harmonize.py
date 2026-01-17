"""OCR result harmonization - combines Tesseract and EasyOCR outputs."""

from dataclasses import dataclass
from typing import Optional

from .models import BBox, Word


@dataclass
class HarmonizedWord:
    """Intermediate representation during harmonization."""
    text: str
    bbox: BBox
    page: int
    tesseract_conf: Optional[float] = None
    easyocr_conf: Optional[float] = None
    source: str = ""  # "tesseract", "easyocr", or "both"


def bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    """
    Calculate how much of box B is covered by box A.

    Returns: Ratio of intersection area to B's area (0.0 to 1.0)
    """
    inter_x0 = max(a.x0, b.x0)
    inter_y0 = max(a.y0, b.y0)
    inter_x1 = min(a.x1, b.x1)
    inter_y1 = min(a.y1, b.y1)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    b_area = b.area

    return inter_area / b_area if b_area > 0 else 0.0


def find_matching_word(
    word: Word,
    candidates: list[Word],
    iou_threshold: float = 0.3,
    text_match_bonus: float = 0.2
) -> Optional[Word]:
    """
    Find the best matching word from candidates based on bbox overlap and text similarity.

    Args:
        word: Word to match
        candidates: List of candidate words to search
        iou_threshold: Minimum IoU to consider a match
        text_match_bonus: Extra threshold allowance if text matches

    Returns:
        Best matching word, or None if no match found
    """
    best_match = None
    best_score = 0.0

    for cand in candidates:
        if cand.page != word.page:
            continue

        iou = word.bbox.iou(cand.bbox)

        # Text similarity bonus - if texts match, lower threshold
        effective_threshold = iou_threshold
        if word.text.lower() == cand.text.lower():
            effective_threshold -= text_match_bonus

        if iou >= effective_threshold and iou > best_score:
            best_score = iou
            best_match = cand

    return best_match


def merge_bboxes(a: BBox, b: BBox) -> BBox:
    """Create a bounding box that encompasses both input boxes."""
    return BBox(
        x0=min(a.x0, b.x0),
        y0=min(a.y0, b.y0),
        x1=max(a.x1, b.x1),
        y1=max(a.y1, b.y1),
    )


def choose_text(tess_word: Optional[Word], easy_word: Optional[Word]) -> str:
    """
    Choose the best text between Tesseract and EasyOCR results.

    Strategy:
    1. If both agree, use that text
    2. If one is a subset of the other, use the longer one
    3. Prefer higher confidence result
    4. Prefer Tesseract for character-level accuracy
    """
    if tess_word is None and easy_word is None:
        return ""
    if tess_word is None:
        return easy_word.text
    if easy_word is None:
        return tess_word.text

    tess_text = tess_word.text
    easy_text = easy_word.text

    # Exact match
    if tess_text == easy_text:
        return tess_text

    # Case-insensitive match - prefer Tesseract capitalization
    if tess_text.lower() == easy_text.lower():
        return tess_text

    # One is substring of other - prefer longer
    if tess_text in easy_text:
        return easy_text
    if easy_text in tess_text:
        return tess_text

    # Compare confidence
    tess_conf = tess_word.confidence or 0
    easy_conf = easy_word.confidence or 0

    # Strong confidence difference (>20%) - go with higher
    if tess_conf > easy_conf + 20:
        return tess_text
    if easy_conf > tess_conf + 20:
        return easy_text

    # Similar confidence - prefer Tesseract for character accuracy
    return tess_text


def harmonize_words(
    tesseract_words: list[Word],
    easyocr_words: list[Word],
    iou_threshold: float = 0.3,
) -> list[Word]:
    """
    Harmonize OCR results from Tesseract and EasyOCR.

    Strategy:
    1. Match words by bounding box overlap (IoU)
    2. For matched pairs: vote on text, average confidence, union bbox
    3. For unmatched Tesseract words: include if confidence > threshold
    4. For unmatched EasyOCR words: include if not covered by Tesseract

    Args:
        tesseract_words: Words from Tesseract
        easyocr_words: Words from EasyOCR
        iou_threshold: Minimum IoU to consider words matched

    Returns:
        Harmonized list of words
    """
    result = []
    used_tess = set()
    used_easy = set()

    # Pass 1: Match Tesseract words to EasyOCR words
    for i, tess in enumerate(tesseract_words):
        match = find_matching_word(tess, easyocr_words, iou_threshold)

        if match is not None:
            j = easyocr_words.index(match)
            used_tess.add(i)
            used_easy.add(j)

            # Merge the pair
            text = choose_text(tess, match)
            bbox = tess.bbox  # Prefer Tesseract bbox (usually more precise)

            # Average confidence, weighted toward higher value
            tess_conf = tess.confidence or 0
            easy_conf = match.confidence or 0
            confidence = max(tess_conf, easy_conf) * 0.7 + min(tess_conf, easy_conf) * 0.3

            word = Word(
                word_id=-1,  # Will be assigned later
                text=text,
                bbox=bbox,
                page=tess.page,
                engine="",  # Harmonized
                confidence=confidence,
                tesseract_confidence=tess_conf,
                easyocr_confidence=easy_conf,
            )
            result.append(word)
        else:
            # Unmatched Tesseract word - include if reasonable confidence
            used_tess.add(i)
            word = Word(
                word_id=-1,
                text=tess.text,
                bbox=tess.bbox,
                page=tess.page,
                engine="",
                confidence=tess.confidence or 0,
                tesseract_confidence=tess.confidence,
                easyocr_confidence=None,
            )
            result.append(word)

    # Pass 2: Add unmatched EasyOCR words that don't overlap with Tesseract
    for j, easy in enumerate(easyocr_words):
        if j in used_easy:
            continue

        # Check if this word is already covered by a Tesseract word
        covered = False
        for i, tess in enumerate(tesseract_words):
            if tess.page != easy.page:
                continue
            overlap = bbox_overlap_ratio(tess.bbox, easy.bbox)
            if overlap > 0.5:  # More than half covered
                covered = True
                break

        if not covered:
            word = Word(
                word_id=-1,
                text=easy.text,
                bbox=easy.bbox,
                page=easy.page,
                engine="",
                confidence=easy.confidence or 0,
                tesseract_confidence=None,
                easyocr_confidence=easy.confidence,
            )
            result.append(word)

    # Sort by page, then by y position, then x position
    result.sort(key=lambda w: (w.page, w.bbox.y0, w.bbox.x0))

    return result
