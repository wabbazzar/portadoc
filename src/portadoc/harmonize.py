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

    # Exempt email-like and URL-like patterns from garbage detection
    # These often have long consonant runs (e.g., "pdxgmail")
    if "@" in text or ".com" in text.lower() or ".org" in text.lower():
        return False

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
    y_band_tolerance: float = 10.0,
) -> Optional[MatchResult]:
    """
    Match a word-level primary to a word-level secondary using IoU.

    Uses text-aware matching: when text matches (case-insensitive), the IoU
    threshold is lowered by text_match_bonus. Additionally, if centers are
    within center_distance_max and text matches, treat as a match even with
    very low IoU.

    Also supports containment-based matching: if secondary has high confidence
    (>=90) and primary has low confidence (<=30), and one bbox's center is
    inside the other, treat as a match. This helps match Surya's accurate
    readings with Tesseract's garbage readings.

    Args:
        primary: Primary word
        words: Secondary words
        matched_ids: Set of already-matched word IDs
        iou_threshold: Minimum IoU for match
        text_match_bonus: IoU threshold reduction when text matches (default 0.15)
        center_distance_max: Max center-to-center distance for fallback match (default 12.0)
        y_band_tolerance: Max y-center difference for match (default 10.0)

    Returns:
        MatchResult or None
    """
    best_match = None
    best_score = 0.0  # Combined score considering IoU and text match

    primary_text_lower = primary.text.lower().strip()
    primary_cx = primary.bbox.center_x
    primary_cy = primary.bbox.center_y
    primary_conf = primary.confidence or 0

    for i, word in enumerate(words):
        if i in matched_ids or word.page != primary.page:
            continue

        # Check vertical alignment first (Issue 2)
        y_diff = abs(word.bbox.center_y - primary_cy)
        if y_diff > y_band_tolerance:
            continue

        iou = primary.bbox.iou(word.bbox)
        word_text_lower = word.text.lower().strip()
        word_conf = word.confidence or 0
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

        # Containment-based fallback: high-conf secondary + low-conf primary + bbox overlap
        # This helps match Surya's good readings (100%) with Tesseract's garbage (<30%)
        if not is_match and word_conf >= 90 and primary_conf <= 30:
            # Check if either center is inside the other bbox
            sec_cx, sec_cy = word.bbox.center_x, word.bbox.center_y
            primary_contains_sec = primary.bbox.contains_point(sec_cx, sec_cy)
            sec_contains_primary = word.bbox.contains_point(primary_cx, primary_cy)
            if primary_contains_sec or sec_contains_primary:
                is_match = True

        if is_match:
            # Score combines IoU and text match bonus
            score = iou + (0.5 if text_matches else 0.0)
            # Boost score for containment matches with high confidence
            if word_conf >= 90 and primary_conf <= 30:
                score += 0.3
            if score > best_score:
                best_score = score
                best_match = MatchResult(
                    text=word.text,
                    confidence=word.confidence or 0,
                    engine="",
                    word_id=i,
                )

    return best_match


def is_paddle_concat(paddle_text: str, other_texts: list[str]) -> bool:
    """
    Detect if paddle text is a concatenation of adjacent words.

    PaddleOCR sometimes reads adjacent words as one (e.g., "NORTHWESTVETERINARY").
    This detects such cases by checking if paddle text is much longer than others
    and contains no spaces.

    Args:
        paddle_text: Text from PaddleOCR
        other_texts: Texts from other engines for comparison

    Returns:
        True if paddle text appears to be concatenated
    """
    if not paddle_text or " " in paddle_text:
        return False

    # Compare to other engine texts
    for other in other_texts:
        if not other:
            continue
        # If paddle text is >1.5x longer and has no spaces, likely concatenated
        if len(paddle_text) > len(other) * 1.5:
            return True

    return False


def weighted_vote(
    votes: list[tuple[str, str, float]],  # (engine, text, confidence)
    config: PortadocConfig,
    known_words: set[str] | None = None,
) -> str:
    """
    Vote on text with weights and garbage penalty.

    Args:
        votes: List of (engine, text, confidence) tuples
        config: Configuration
        known_words: Optional set of known-good words for similarity checking

    Returns:
        Winning text
    """
    if len(votes) == 1:
        return votes[0][1]

    scores: dict[str, float] = defaultdict(float)
    originals: dict[str, str] = {}

    # Collect non-paddle texts for concatenation check
    non_paddle_texts = [text for eng, text, _ in votes if eng != "paddleocr" and text.strip()]
    paddle_text = next((text for eng, text, _ in votes if eng == "paddleocr"), None)

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

        # Paddle concatenation penalty (Issue 3)
        if engine == "paddleocr" and is_paddle_concat(text, non_paddle_texts):
            weight *= 0.1

        # Known-word similarity bonus (Issue 5)
        # If text contains parts similar to known words, boost its score
        if known_words and len(text) > 5:
            # Extract potential words from text (split by non-alphanumeric)
            import re
            parts = re.split(r'[^a-zA-Z]+', text.lower())
            best_similarity = 0.0
            for part in parts:
                if len(part) < 3:
                    continue
                for known in known_words:
                    if len(known) < 3:
                        continue
                    # Check similarity
                    dist = levenshtein_distance(part, known.lower())
                    similarity = 1 - (dist / max(len(part), len(known)))
                    if similarity > best_similarity:
                        best_similarity = similarity
            # Apply scaled boost based on best similarity (0.7+ threshold)
            if best_similarity > 0.7:
                # Boost scales from 1.0 (at 0.7) to 1.5 (at 1.0)
                weight *= 1.0 + (best_similarity - 0.7) * 1.67

        scores[normalized] += weight
        if normalized not in originals:
            originals[normalized] = text

    if not scores:
        return votes[0][1] if votes else ""

    winner = max(scores, key=scores.get)
    return originals[winner]


def merge_adjacent_fragments(
    words: list[HarmonizedWord],
    config: PortadocConfig,
) -> list[HarmonizedWord]:
    """
    Merge adjacent low-conf/pixel fragments when a secondary covers them.

    When Tesseract fragments a word (e.g., microchip "985141004729856" into
    "OBS 141 OM 725 R66"), check if a secondary-only detection covers the
    combined span. If so, remove the fragments (keep the secondary).

    Args:
        words: List of harmonized words
        config: Configuration

    Returns:
        Filtered list with fragments removed where secondary is better
    """
    # Identify secondary-only words with high confidence
    secondary_only = [
        w for w in words
        if w.status == "secondary_only" and w.confidence >= 90
    ]

    if not secondary_only:
        return words

    # For each secondary-only, check if it covers primary fragments
    to_remove: set[int] = set()  # indices of words to remove

    for sec in secondary_only:
        # Find primary fragments (primary-only, low-conf or pixel) overlapping this secondary
        fragments = []
        for i, w in enumerate(words):
            if w is sec:
                continue
            if w.page != sec.page:
                continue
            # Is it a primary-only fragment? (single-letter source = no secondary match)
            if len(w.source) != 1:
                continue
            if w.status not in ("low_conf", "pixel"):
                continue
            # Check vertical alignment (same y-band)
            y_diff = abs(w.bbox.center_y - sec.bbox.center_y)
            if y_diff > 15:  # Allow some tolerance
                continue
            # Check horizontal overlap
            overlap_x = min(w.bbox.x1, sec.bbox.x1) - max(w.bbox.x0, sec.bbox.x0)
            if overlap_x > 0:
                fragments.append(i)

        # If we found multiple fragments covered by this secondary, remove them
        if len(fragments) >= 2:
            to_remove.update(fragments)

    # Filter out removed words
    return [w for i, w in enumerate(words) if i not in to_remove]


def should_suppress_secondary(
    sec_word: Word,
    primary_words: list[Word],
    config: PortadocConfig,
) -> bool:
    """
    Check if a secondary-only word should be suppressed.

    Suppresses secondary words that overlap multiple primary words, which
    indicates line-level detection or concatenation rather than a true
    missed word.

    Args:
        sec_word: Secondary word candidate
        primary_words: All primary words
        config: Configuration

    Returns:
        True if the word should be suppressed
    """
    # Count how many primaries this secondary contains (center-based)
    contained_count = 0
    sec_bbox = sec_word.bbox

    for primary in primary_words:
        if primary.page != sec_word.page:
            continue
        # Check if primary center is inside secondary bbox
        if sec_bbox.contains_point(primary.bbox.center_x, primary.bbox.center_y):
            contained_count += 1
            if contained_count > 1:
                # Contains multiple primaries - suppress if low confidence
                conf = sec_word.confidence or 0
                if conf < 50:
                    return True

    return False


def harmonize(
    primary_words: list[Word],
    secondary_results: dict[str, list[Word]],
    config: Optional[PortadocConfig] = None,
    primary_engine: str = "tesseract",
) -> list[HarmonizedWord]:
    """
    Harmonize OCR results using primary-engine bbox and text voting.

    Primary engine provides authoritative word boundaries.
    Secondary engines vote on text only and can add high-confidence detections.
    All detections are tracked with status and Levenshtein distances.

    Args:
        primary_words: Words from primary engine
        secondary_results: Dict of {engine_name: words} for secondary engines
        config: Configuration (uses global config if None)
        primary_engine: Name of primary engine (for source code tracking)

    Returns:
        List of HarmonizedWord with full tracking
    """
    if config is None:
        config = get_config()

    result: list[HarmonizedWord] = []
    matched_secondary: dict[str, set[int]] = defaultdict(set)
    all_bboxes: list[BBox] = []

    # Build known-good word set from high-confidence primaries (Issue 5)
    known_words: set[str] = set()
    for primary in primary_words:
        if (primary.confidence or 0) >= 80 and len(primary.text) >= 3:
            # Add word and potential parts (split on non-alphanumeric)
            import re
            known_words.add(primary.text.lower())
            for part in re.split(r'[^a-zA-Z]+', primary.text.lower()):
                if len(part) >= 3:
                    known_words.add(part)

    # Also add known-good words from secondaries
    for engine, sec_words in secondary_results.items():
        for sw in sec_words:
            if (sw.confidence or 0) >= 90 and len(sw.text) >= 3:
                import re
                known_words.add(sw.text.lower())
                for part in re.split(r'[^a-zA-Z]+', sw.text.lower()):
                    if len(part) >= 3:
                        known_words.add(part)

    # Map engine names to source codes
    ENGINE_CODES = {
        "tesseract": "T",
        "easyocr": "E",
        "doctr": "D",
        "paddleocr": "P",
        "surya": "S",
        "kraken": "K",
    }
    primary_code = ENGINE_CODES.get(primary_engine, primary_engine[0].upper())

    # === PHASE 1: Process ALL primary words ===
    for primary in primary_words:
        hw = HarmonizedWord(
            word_id=-1,
            page=primary.page,
            bbox=primary.bbox,
            text="",  # will be set by voting
            status="",  # will be set by confidence
            source=primary_code,
            confidence=primary.confidence or 0,
        )

        # Store primary engine's raw text
        if primary_engine == "tesseract":
            hw.tess_text = primary.text
        elif primary_engine == "easyocr":
            hw.easy_text = primary.text
        elif primary_engine == "doctr":
            hw.doctr_text = primary.text
        elif primary_engine == "paddleocr":
            hw.paddle_text = primary.text
        elif primary_engine == "surya":
            hw.surya_text = primary.text
        elif primary_engine == "kraken":
            hw.kraken_text = primary.text

        # Collect text votes from all engines
        votes: list[tuple[str, str, float]] = [
            (primary_engine, primary.text, primary.confidence or 0)
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
                # Use per-engine overrides if specified
                iou_thresh = (
                    eng_config.iou_threshold_override
                    if eng_config and eng_config.iou_threshold_override is not None
                    else config.harmonize.iou_threshold
                )
                center_dist = (
                    eng_config.center_distance_max_override
                    if eng_config and eng_config.center_distance_max_override is not None
                    else config.harmonize.center_distance_max
                )
                # y_band_tolerance should be proportional to center_distance_max
                # For Surya (center_dist=50), use y_band=25; for others (12), use default 10
                y_band = max(10.0, center_dist / 2)
                match = find_word_match(
                    primary, sec_words, matched_secondary[engine],
                    iou_thresh,
                    config.harmonize.text_match_bonus,
                    center_dist,
                    y_band,
                )

            if match:
                matched_secondary[engine].add(match.word_id)

                # Only vote if confidence above threshold
                if match.confidence >= config.harmonize.secondary.vote_min_conf:
                    votes.append((engine, match.text, match.confidence))

                # Update source string
                engine_code = ENGINE_CODES.get(engine, engine[0].upper())
                hw.source += engine_code

                # Store raw text (avoid overwriting if already set as primary)
                if engine == "tesseract" and not hw.tess_text:
                    hw.tess_text = match.text
                elif engine == "easyocr" and not hw.easy_text:
                    hw.easy_text = match.text
                elif engine == "doctr" and not hw.doctr_text:
                    hw.doctr_text = match.text
                elif engine == "paddleocr" and not hw.paddle_text:
                    hw.paddle_text = match.text
                elif engine == "surya" and not hw.surya_text:
                    hw.surya_text = match.text
                elif engine == "kraken" and not hw.kraken_text:
                    hw.kraken_text = match.text

        # Vote on final text
        hw.text = weighted_vote(votes, config, known_words)

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
        if hw.surya_text:
            hw.dist_surya = levenshtein_distance(hw.surya_text.lower(), final_lower)
        if hw.kraken_text:
            hw.dist_kraken = levenshtein_distance(hw.kraken_text.lower(), final_lower)

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

            # Suppress secondaries that overlap multiple primaries (Issue 1)
            if should_suppress_secondary(sec_word, primary_words, config):
                continue

            # Text-aware deduplication: skip if same text already exists on same y-band
            # This handles wide Surya bboxes that have low IoU but duplicate text
            sec_text_lower = sec_word.text.lower().strip('.:,')
            sec_cy = sec_word.bbox.center_y
            text_duplicate = False
            for existing in result:
                if existing.page != sec_word.page:
                    continue
                # Same y-band (within 15 pixels)?
                if abs(existing.bbox.center_y - sec_cy) > 15:
                    continue
                # Same text?
                if existing.text.lower().strip('.:,') == sec_text_lower and sec_text_lower:
                    text_duplicate = True
                    break
            if text_duplicate:
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
            elif engine == "surya":
                hw.surya_text = sec_word.text
                hw.dist_surya = 0

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

    # === PHASE 3: Merge adjacent fragments (Issue 4) ===
    # Find low-conf/pixel primary fragments that a secondary covers better
    result = merge_adjacent_fragments(result, config)

    # Assign word IDs and sort
    result.sort(key=lambda w: (w.page, w.bbox.y0, w.bbox.x0))
    for i, hw in enumerate(result):
        hw.word_id = i

    return result


# Backward compatibility alias
smart_harmonize = harmonize
