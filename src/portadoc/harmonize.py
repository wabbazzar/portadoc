"""OCR result harmonization - combines multiple OCR engine outputs."""

from collections import defaultdict
from dataclasses import dataclass, field
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


# Default weights for each OCR engine based on observed performance
# Higher weight = more trusted for text accuracy
ENGINE_WEIGHTS = {
    "tesseract": 1.0,  # Good character-level accuracy
    "easyocr": 0.9,    # Good overall but sometimes misses
    "paddleocr": 0.6,  # Lower performance on degraded docs
    "doctr": 1.1,      # Best text accuracy (76.57% text match)
}


@dataclass
class LegacyHarmonizedWord:
    """Old intermediate representation (kept for backwards compatibility)."""
    text: str
    bbox: BBox
    page: int
    tesseract_conf: Optional[float] = None
    easyocr_conf: Optional[float] = None
    source: str = ""  # "tesseract", "easyocr", or "both"


@dataclass
class WordCluster:
    """Group of words from different engines that refer to the same text region."""
    words: list[tuple[str, Word]] = field(default_factory=list)  # (engine_name, word)

    def add(self, engine: str, word: Word):
        self.words.append((engine, word))

    @property
    def page(self) -> int:
        return self.words[0][1].page if self.words else 0

    def get_merged_bbox(self) -> BBox:
        """Get bounding box that encompasses all words in cluster."""
        if not self.words:
            return BBox(0, 0, 0, 0)
        bboxes = [w.bbox for _, w in self.words]
        return BBox(
            x0=min(b.x0 for b in bboxes),
            y0=min(b.y0 for b in bboxes),
            x1=max(b.x1 for b in bboxes),
            y1=max(b.y1 for b in bboxes),
        )

    def vote_text(self) -> str:
        """Vote on best text using weighted voting."""
        if not self.words:
            return ""
        if len(self.words) == 1:
            return self.words[0][1].text

        # Collect votes with weights
        text_votes: dict[str, float] = defaultdict(float)
        for engine, word in self.words:
            weight = ENGINE_WEIGHTS.get(engine, 1.0)
            conf = (word.confidence or 50) / 100.0  # Normalize to 0-1
            vote_weight = weight * (0.5 + 0.5 * conf)  # Weight by confidence

            # Normalize text for voting (lowercase)
            normalized = word.text.lower().strip()
            if normalized:
                text_votes[normalized] += vote_weight

        if not text_votes:
            return self.words[0][1].text

        # Find winning text
        best_normalized = max(text_votes, key=text_votes.get)

        # Return original casing from highest-weight engine that voted for this
        for engine, word in sorted(self.words, key=lambda x: -ENGINE_WEIGHTS.get(x[0], 1.0)):
            if word.text.lower().strip() == best_normalized:
                return word.text

        return self.words[0][1].text

    def get_confidence(self) -> float:
        """Get combined confidence score."""
        if not self.words:
            return 0.0
        confs = [w.confidence or 0 for _, w in self.words]
        # Weighted average favoring higher values
        return max(confs) * 0.7 + sum(confs) / len(confs) * 0.3


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


def harmonize_multi_engine(
    engine_results: list[tuple[str, list[Word]]],
    iou_threshold: float = 0.3,
) -> list[Word]:
    """
    Harmonize OCR results from multiple engines using clustering and voting.

    Args:
        engine_results: List of (engine_name, words) tuples
        iou_threshold: Minimum IoU to consider words as matching

    Returns:
        Harmonized list of words
    """
    if not engine_results:
        return []

    if len(engine_results) == 1:
        return engine_results[0][1]

    # Flatten all words with their engine names
    all_words: list[tuple[str, Word]] = []
    for engine_name, words in engine_results:
        for word in words:
            all_words.append((engine_name, word))

    if not all_words:
        return []

    # Cluster words by spatial overlap
    clusters: list[WordCluster] = []
    used = set()

    for i, (eng_i, word_i) in enumerate(all_words):
        if i in used:
            continue

        cluster = WordCluster()
        cluster.add(eng_i, word_i)
        used.add(i)

        # Find all overlapping words from other engines
        for j, (eng_j, word_j) in enumerate(all_words):
            if j in used or word_j.page != word_i.page:
                continue

            iou = word_i.bbox.iou(word_j.bbox)
            if iou >= iou_threshold:
                cluster.add(eng_j, word_j)
                used.add(j)

        clusters.append(cluster)

    # Convert clusters to words
    result = []
    for cluster in clusters:
        text = cluster.vote_text()
        if not text.strip():
            continue

        # Use bbox from highest-weight engine for precision
        best_bbox = None
        best_weight = -1
        for engine, word in cluster.words:
            weight = ENGINE_WEIGHTS.get(engine, 1.0)
            if weight > best_weight:
                best_weight = weight
                best_bbox = word.bbox

        word = Word(
            word_id=-1,
            text=text,
            bbox=best_bbox or cluster.get_merged_bbox(),
            page=cluster.page,
            engine="",
            confidence=cluster.get_confidence(),
        )
        result.append(word)

    # Sort by page, then by y position, then x position
    result.sort(key=lambda w: (w.page, w.bbox.y0, w.bbox.x0))

    return result


# =============================================================================
# Smart Harmonization (New Implementation)
# =============================================================================


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


def smart_harmonize(
    primary_words: list[Word],
    secondary_results: dict[str, list[Word]],
    config: Optional[PortadocConfig] = None,
) -> list[HarmonizedWord]:
    """
    Smart harmonization with primary-engine bbox and text voting.

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
