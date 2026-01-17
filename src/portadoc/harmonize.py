"""OCR result harmonization - combines multiple OCR engine outputs using smart harmonization."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    # Fallback if python-Levenshtein not installed
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

from .models import BBox, Word, HarmonizedWord
from .config import PortadocConfig, get_config


def is_garbage_text(text: str, config: Optional[PortadocConfig] = None) -> bool:
    """
    Detect OCR garbage like 'DCCUZAn', 'ImLIudz'.

    Args:
        text: Text to check
        config: Configuration (uses defaults if None)

    Returns:
        True if text appears to be garbage
    """
    if config is None:
        config = get_config()

    gc = config.harmonize.garbage_detection
    if not gc.enabled:
        return False

    if not text or len(text) < 2:
        return True

    # Low alphanumeric ratio
    alnum = sum(c.isalnum() for c in text)
    if alnum / len(text) < gc.min_alnum_ratio:
        return True

    # Long consonant runs
    vowels = set('aeiouAEIOU')
    run = 0
    for c in text:
        if c.isalpha() and c not in vowels:
            run += 1
            if run >= gc.max_consonant_run:
                return True
        else:
            run = 0

    # Suspicious mixed case mid-word
    if gc.mixed_case_penalty and len(text) > 3:
        inner = text[1:-1]
        if inner and any(c.isupper() for c in inner) and any(c.islower() for c in inner):
            return True

    return False


def extract_word_at_position(line_text: str, rel_x: float) -> str:
    """
    Extract word from line text at a relative x position.

    Args:
        line_text: Full line text
        rel_x: Relative x position (0.0 to 1.0)

    Returns:
        Extracted word at that position
    """
    if not line_text:
        return ""

    words = line_text.split()
    if not words:
        return line_text.strip()

    if len(words) == 1:
        return words[0]

    # Map relative position to word index
    # Each word takes proportional space based on character count
    total_chars = sum(len(w) for w in words) + len(words) - 1  # +spaces
    char_pos = int(rel_x * total_chars)

    # Find which word this position falls into
    current_pos = 0
    for word in words:
        word_end = current_pos + len(word)
        if char_pos <= word_end:
            return word
        current_pos = word_end + 1  # +1 for space

    return words[-1]


@dataclass
class MatchResult:
    """Result of matching a primary word to secondary engine."""
    text: str
    confidence: float
    engine: str
    word_id: int  # ID for tracking which secondary word was matched


def find_line_match(
    primary: Word,
    lines: list[Word],
    matched_ids: set[int],
) -> Optional[MatchResult]:
    """
    Match a word-level primary to a line-level secondary.

    Uses containment: checks if primary center is inside line bbox,
    then extracts the word at that relative position.

    Args:
        primary: Primary (word-level) word
        lines: Secondary (line-level) words
        matched_ids: Set of already-matched word IDs

    Returns:
        MatchResult with extracted text, or None if no match
    """
    for i, line in enumerate(lines):
        if i in matched_ids or line.page != primary.page:
            continue

        # Check if primary center is inside line bbox
        if not line.bbox.contains_point(primary.bbox.center_x, primary.bbox.center_y):
            continue

        # Calculate relative x position within line
        if line.bbox.width > 0:
            rel_x = (primary.bbox.center_x - line.bbox.x0) / line.bbox.width
        else:
            rel_x = 0.5

        extracted = extract_word_at_position(line.text, rel_x)

        return MatchResult(
            text=extracted,
            confidence=line.confidence or 0,
            engine="",  # Will be set by caller
            word_id=i,
        )

    return None


def find_word_match(
    primary: Word,
    words: list[Word],
    matched_ids: set[int],
    iou_threshold: float,
    text_match_bonus: float = 0.15,
    center_distance_max: float = 12.0,
) -> Optional[MatchResult]:
    """
    Match a word-level primary to a word-level secondary using IoU.

    Uses text-aware matching: when text matches (case-insensitive), the IoU
    threshold is lowered by text_match_bonus. Additionally, if centers are
    within center_distance_max and text matches, treat as a match even with
    very low IoU.

    Args:
        primary: Primary word
        words: Secondary words
        matched_ids: Set of already-matched word IDs
        iou_threshold: Minimum IoU for match
        text_match_bonus: IoU threshold reduction when text matches (default 0.15)
        center_distance_max: Max center-to-center distance for fallback match (default 12.0)

    Returns:
        MatchResult or None
    """
    best_match = None
    best_score = 0.0  # Combined score considering IoU and text match

    primary_text_lower = primary.text.lower().strip()
    primary_cx = primary.bbox.center_x
    primary_cy = primary.bbox.center_y

    for i, word in enumerate(words):
        if i in matched_ids or word.page != primary.page:
            continue

        iou = primary.bbox.iou(word.bbox)
        word_text_lower = word.text.lower().strip()
        text_matches = (primary_text_lower == word_text_lower)

        # Calculate effective threshold based on text match
        effective_threshold = iou_threshold
        if text_matches:
            effective_threshold -= text_match_bonus

        # Primary matching: IoU meets effective threshold
        is_match = iou >= effective_threshold

        # Fallback matching: center distance + text match
        if not is_match and text_matches and center_distance_max > 0:
            center_dist = ((word.bbox.center_x - primary_cx) ** 2 +
                          (word.bbox.center_y - primary_cy) ** 2) ** 0.5
            if center_dist <= center_distance_max:
                is_match = True

        if is_match:
            # Score combines IoU and text match bonus
            score = iou + (0.5 if text_matches else 0.0)
            if score > best_score:
                best_score = score
                best_match = MatchResult(
                    text=word.text,
                    confidence=word.confidence or 0,
                    engine="",
                    word_id=i,
                )

    return best_match


def weighted_vote(
    votes: list[tuple[str, str, float]],  # (engine, text, confidence)
    config: PortadocConfig,
) -> str:
    """
    Vote on text with weights and garbage penalty.

    Args:
        votes: List of (engine, text, confidence) tuples
        config: Configuration

    Returns:
        Winning text
    """
    if len(votes) == 1:
        return votes[0][1]

    scores: dict[str, float] = defaultdict(float)
    originals: dict[str, str] = {}

    for engine, text, conf in votes:
        if not text.strip():
            continue

        normalized = text.lower().strip()

        # Get engine weight
        if engine == "tesseract":
            weight = config.harmonize.primary.weight
        else:
            eng_config = config.harmonize.secondary.engines.get(engine)
            weight = eng_config.weight if eng_config else 1.0

        # Confidence factor (0.5 to 1.0 range)
        weight *= (0.5 + 0.5 * (conf / 100.0))

        # Garbage penalty
        if is_garbage_text(text, config):
            eng_config = config.harmonize.secondary.engines.get(engine)
            penalty = eng_config.garbage_penalty if eng_config else 0.1
            weight *= penalty

        scores[normalized] += weight
        if normalized not in originals:
            originals[normalized] = text

    if not scores:
        return votes[0][1] if votes else ""

    winner = max(scores, key=scores.get)
    return originals[winner]


def harmonize(
    primary_words: list[Word],
    secondary_results: dict[str, list[Word]],
    config: Optional[PortadocConfig] = None,
) -> list[HarmonizedWord]:
    """
    Harmonize OCR results using primary-engine bbox and text voting.

    Primary engine (Tesseract) provides authoritative word boundaries.
    Secondary engines vote on text only and can add high-confidence detections.
    All detections are tracked with status and Levenshtein distances.

    Args:
        primary_words: Words from primary engine (Tesseract)
        secondary_results: Dict of {engine_name: words} for secondary engines
        config: Configuration (uses global config if None)

    Returns:
        List of HarmonizedWord with full tracking
    """
    if config is None:
        config = get_config()

    result: list[HarmonizedWord] = []
    matched_secondary: dict[str, set[int]] = defaultdict(set)
    all_bboxes: list[BBox] = []

    # === PHASE 1: Process ALL primary words ===
    for primary in primary_words:
        hw = HarmonizedWord(
            word_id=-1,
            page=primary.page,
            bbox=primary.bbox,
            text="",  # will be set by voting
            status="",  # will be set by confidence
            source="T",
            confidence=primary.confidence or 0,
        )
        hw.tess_text = primary.text

        # Collect text votes from all engines
        votes: list[tuple[str, str, float]] = [
            ("tesseract", primary.text, primary.confidence or 0)
        ]

        for engine, sec_words in secondary_results.items():
            eng_config = config.harmonize.secondary.engines.get(engine)
            if eng_config and not eng_config.enabled:
                continue

            # Determine matching strategy based on bbox type
            bbox_type = eng_config.bbox_type if eng_config else "word"

            if bbox_type == "line":
                match = find_line_match(primary, sec_words, matched_secondary[engine])
            else:
                match = find_word_match(
                    primary, sec_words, matched_secondary[engine],
                    config.harmonize.iou_threshold,
                    config.harmonize.text_match_bonus,
                    config.harmonize.center_distance_max,
                )

            if match:
                matched_secondary[engine].add(match.word_id)

                # Only vote if confidence above threshold
                if match.confidence >= config.harmonize.secondary.vote_min_conf:
                    votes.append((engine, match.text, match.confidence))

                # Update source string
                engine_code = engine[0].upper()  # T, E, D, P
                hw.source += engine_code

                # Store raw text
                if engine == "easyocr":
                    hw.easy_text = match.text
                elif engine == "doctr":
                    hw.doctr_text = match.text
                elif engine == "paddleocr":
                    hw.paddle_text = match.text

        # Vote on final text
        hw.text = weighted_vote(votes, config)

        # Compute Levenshtein distances
        final_lower = hw.text.lower()
        if hw.tess_text:
            hw.dist_tess = levenshtein_distance(hw.tess_text.lower(), final_lower)
        if hw.easy_text:
            hw.dist_easy = levenshtein_distance(hw.easy_text.lower(), final_lower)
        if hw.doctr_text:
            hw.dist_doctr = levenshtein_distance(hw.doctr_text.lower(), final_lower)
        if hw.paddle_text:
            hw.dist_paddle = levenshtein_distance(hw.paddle_text.lower(), final_lower)

        # Determine status by max confidence
        max_conf = max(v[2] for v in votes) if votes else 0
        hw.confidence = max_conf

        if max_conf >= config.harmonize.status.word_min_conf:
            hw.status = "word"
        elif max_conf >= config.harmonize.status.low_conf_min_conf:
            hw.status = "low_conf"
        else:
            hw.status = "pixel"

        result.append(hw)
        all_bboxes.append(hw.bbox)

    # === PHASE 2: Add UNMATCHED secondary words ===
    for engine, sec_words in secondary_results.items():
        eng_config = config.harmonize.secondary.engines.get(engine)
        if eng_config and not eng_config.enabled:
            continue

        for i, sec_word in enumerate(sec_words):
            if i in matched_secondary[engine]:
                continue

            # Skip if overlaps existing bbox
            overlaps = any(
                sec_word.bbox.iou(b) >= config.harmonize.iou_threshold
                for b in all_bboxes
            )
            if overlaps:
                continue

            engine_code = engine[0].upper()
            hw = HarmonizedWord(
                word_id=-1,
                page=sec_word.page,
                bbox=sec_word.bbox,
                text=sec_word.text,
                status="",  # will be set below
                source=engine_code,
                confidence=sec_word.confidence or 0,
            )

            # Store raw text and distance
            if engine == "easyocr":
                hw.easy_text = sec_word.text
                hw.dist_easy = 0
            elif engine == "doctr":
                hw.doctr_text = sec_word.text
                hw.dist_doctr = 0
            elif engine == "paddleocr":
                hw.paddle_text = sec_word.text
                hw.dist_paddle = 0

            # Check for corroboration from another secondary
            corroborated = False
            for other_eng, other_words in secondary_results.items():
                if other_eng == engine:
                    continue
                for other in other_words:
                    if sec_word.bbox.iou(other.bbox) >= config.harmonize.iou_threshold:
                        corroborated = True
                        break
                if corroborated:
                    break

            conf = sec_word.confidence or 0
            if conf >= config.harmonize.secondary.solo_high_conf:
                hw.status = "secondary_only"
            elif conf >= config.harmonize.secondary.solo_min_conf and corroborated:
                hw.status = "secondary_only"
            elif conf >= config.harmonize.status.low_conf_min_conf:
                hw.status = "low_conf"
            else:
                hw.status = "pixel"

            result.append(hw)
            all_bboxes.append(hw.bbox)

    # Assign word IDs and sort
    result.sort(key=lambda w: (w.page, w.bbox.y0, w.bbox.x0))
    for i, hw in enumerate(result):
        hw.word_id = i

    return result


# Backward compatibility alias
smart_harmonize = harmonize
