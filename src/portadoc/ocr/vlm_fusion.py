"""
VLM + BBox OCR Fusion algorithm.

Combines VLM text accuracy with traditional OCR bounding box precision.

Algorithm:
1. Geometric cluster OCR words to identify text blocks
2. Tokenize VLM text output
3. Match VLM tokens to OCR words within clusters using context-based matching
4. For matched pairs: use VLM text + OCR bbox
5. For VLM orphans: use pixel detection to find text regions, then place words
6. For OCR orphans: keep original OCR text + bbox
7. Order final output by reading order
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from statistics import mean, stdev

from portadoc.models import BBox, HarmonizedWord, Word
from portadoc.geometric_clustering import (
    order_words_by_reading,
    build_clusters,
    calculate_distance_thresholds,
    Cluster,
)
from portadoc.detection import detect_missed_content

if TYPE_CHECKING:
    from numpy import ndarray


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MatchResult:
    """Result of matching VLM tokens to OCR words."""
    matched: Dict[int, Tuple[str, HarmonizedWord]]  # vlm_idx -> (vlm_text, ocr_word)
    vlm_orphans: List[int]  # VLM token indices with no OCR match
    ocr_orphans: List[int]  # OCR word indices with no VLM match
    vlm_tokens: List[str]  # Original VLM tokens for reference


@dataclass
class CharSizes:
    """Character size estimates."""
    avg_char_width: float
    avg_char_height: float
    std_char_width: float = 0.0
    std_char_height: float = 0.0


@dataclass
class CharSizeCluster:
    """A cluster of character size measurements for similar font sizes."""
    measurements: List[Tuple[float, float]] = field(default_factory=list)  # (width, height)

    @property
    def avg_char_width(self) -> float:
        if not self.measurements:
            return 0.0
        return mean(w for w, _ in self.measurements)

    @property
    def avg_char_height(self) -> float:
        if not self.measurements:
            return 0.0
        return mean(h for _, h in self.measurements)

    @property
    def std_char_height(self) -> float:
        if len(self.measurements) < 2:
            return 0.0
        return stdev(h for _, h in self.measurements)


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_vlm_text(text: str) -> List[str]:
    """
    Tokenize VLM text output into words.

    Preserves punctuation attached to words (e.g., "splat," stays as one token).

    Args:
        text: Raw VLM text output

    Returns:
        List of word tokens
    """
    if not text or not text.strip():
        return []

    # Split on whitespace, preserving punctuation attached to words
    tokens = text.split()

    return tokens


# ============================================================================
# Context Matching
# ============================================================================

def build_context_signature(tokens: List[str], index: int, window: int = 1) -> str:
    """
    Build a context signature for a token at given index.

    The signature includes surrounding words to help disambiguate repeated words.

    Args:
        tokens: List of all tokens
        index: Index of the target token
        window: Number of words on each side to include

    Returns:
        Space-separated context string
    """
    start = max(0, index - window)
    end = min(len(tokens), index + window + 1)
    return " ".join(tokens[start:end])


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row is one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def context_distance(ctx1: str, ctx2: str) -> int:
    """
    Calculate distance between two context signatures.

    Uses Levenshtein on the full context string.
    """
    return levenshtein_distance(ctx1.lower(), ctx2.lower())


def match_vlm_to_ocr(
    ocr_words: List[HarmonizedWord],
    vlm_text: str,
    max_context_window: int = 4,
    max_levenshtein: int = 3,
) -> MatchResult:
    """
    Match VLM tokens to OCR words using context-based matching.

    Algorithm:
    1. Tokenize VLM text
    2. For each VLM token, try to find matching OCR word
    3. Use expanding context window to disambiguate repeated words
    4. Track orphans on both sides

    Args:
        ocr_words: Words from traditional OCR with bboxes
        vlm_text: Raw text from VLM
        max_context_window: Maximum context expansion
        max_levenshtein: Maximum edit distance for fuzzy matching

    Returns:
        MatchResult with matched pairs and orphans
    """
    vlm_tokens = tokenize_vlm_text(vlm_text)

    if not vlm_tokens:
        return MatchResult(
            matched={},
            vlm_orphans=[],
            ocr_orphans=list(range(len(ocr_words))),
            vlm_tokens=vlm_tokens,
        )

    if not ocr_words:
        return MatchResult(
            matched={},
            vlm_orphans=list(range(len(vlm_tokens))),
            ocr_orphans=[],
            vlm_tokens=vlm_tokens,
        )

    ocr_texts = [w.text for w in ocr_words]
    matched: Dict[int, Tuple[str, HarmonizedWord]] = {}
    ocr_used: set = set()

    # Pass 0: Direct word matching for single/few VLM tokens
    # (context matching doesn't work well with minimal context)
    if len(vlm_tokens) <= 3:
        for vlm_idx, vlm_token in enumerate(vlm_tokens):
            # Find exact match first
            for ocr_idx, ocr_word in enumerate(ocr_words):
                if ocr_idx in ocr_used:
                    continue
                if vlm_token.lower() == ocr_word.text.lower():
                    matched[vlm_idx] = (vlm_token, ocr_word)
                    ocr_used.add(ocr_idx)
                    break

        # Then fuzzy match for remaining
        for vlm_idx, vlm_token in enumerate(vlm_tokens):
            if vlm_idx in matched:
                continue
            best_match = None
            best_distance = max_levenshtein + 1
            for ocr_idx, ocr_word in enumerate(ocr_words):
                if ocr_idx in ocr_used:
                    continue
                dist = levenshtein_distance(vlm_token.lower(), ocr_word.text.lower())
                if dist <= max_levenshtein and dist < best_distance:
                    best_match = ocr_idx
                    best_distance = dist
            if best_match is not None:
                matched[vlm_idx] = (vlm_token, ocr_words[best_match])
                ocr_used.add(best_match)

        # Return early for short VLM inputs
        vlm_orphans = [i for i in range(len(vlm_tokens)) if i not in matched]
        ocr_orphans = [i for i in range(len(ocr_words)) if i not in ocr_used]
        return MatchResult(
            matched=matched,
            vlm_orphans=vlm_orphans,
            ocr_orphans=ocr_orphans,
            vlm_tokens=vlm_tokens,
        )

    # Pass 1: Exact matches with expanding context
    for vlm_idx, vlm_token in enumerate(vlm_tokens):
        if vlm_idx in matched:
            continue

        for window in range(1, max_context_window + 1):
            vlm_context = build_context_signature(vlm_tokens, vlm_idx, window)

            candidates = []
            for ocr_idx, ocr_word in enumerate(ocr_words):
                if ocr_idx in ocr_used:
                    continue

                ocr_context = build_context_signature(ocr_texts, ocr_idx, window)

                # Check if contexts match (exact)
                if context_distance(vlm_context, ocr_context) == 0:
                    candidates.append(ocr_idx)

            if len(candidates) == 1:
                matched[vlm_idx] = (vlm_token, ocr_words[candidates[0]])
                ocr_used.add(candidates[0])
                break

    # Pass 2: Fuzzy matches for remaining
    for max_dist in range(1, max_levenshtein + 1):
        for vlm_idx, vlm_token in enumerate(vlm_tokens):
            if vlm_idx in matched:
                continue

            for window in range(1, max_context_window + 1):
                vlm_context = build_context_signature(vlm_tokens, vlm_idx, window)

                best_match = None
                best_distance = max_dist + 1

                for ocr_idx, ocr_word in enumerate(ocr_words):
                    if ocr_idx in ocr_used:
                        continue

                    ocr_context = build_context_signature(ocr_texts, ocr_idx, window)
                    distance = context_distance(vlm_context, ocr_context)

                    if distance <= max_dist and distance < best_distance:
                        best_match = ocr_idx
                        best_distance = distance

                if best_match is not None:
                    matched[vlm_idx] = (vlm_token, ocr_words[best_match])
                    ocr_used.add(best_match)
                    break

    # Pass 3: Positional fallback for remaining orphans
    # When context matching fails (e.g., OCR missed a chunk), try matching
    # based on text similarity + reading order position
    vlm_orphan_indices = [i for i in range(len(vlm_tokens)) if i not in matched]
    ocr_orphan_indices = [i for i in range(len(ocr_words)) if i not in ocr_used]

    if vlm_orphan_indices and ocr_orphan_indices:
        # Try to match orphans with same/similar text in reading order
        for vlm_idx in vlm_orphan_indices[:]:  # Copy to allow modification
            if vlm_idx in matched:
                continue

            vlm_token = vlm_tokens[vlm_idx]

            # Find OCR orphans with matching text
            best_ocr_idx = None
            best_distance = max_levenshtein + 1

            for ocr_idx in ocr_orphan_indices:
                if ocr_idx in ocr_used:
                    continue

                ocr_text = ocr_words[ocr_idx].text
                dist = levenshtein_distance(vlm_token.lower(), ocr_text.lower())

                if dist <= max_levenshtein and dist < best_distance:
                    best_ocr_idx = ocr_idx
                    best_distance = dist

            if best_ocr_idx is not None:
                matched[vlm_idx] = (vlm_token, ocr_words[best_ocr_idx])
                ocr_used.add(best_ocr_idx)

    # Identify final orphans
    vlm_orphans = [i for i in range(len(vlm_tokens)) if i not in matched]
    ocr_orphans = [i for i in range(len(ocr_words)) if i not in ocr_used]

    return MatchResult(
        matched=matched,
        vlm_orphans=vlm_orphans,
        ocr_orphans=ocr_orphans,
        vlm_tokens=vlm_tokens,
    )


# ============================================================================
# Character Size Estimation
# ============================================================================

def estimate_char_sizes(matched_pairs: List[Tuple[str, BBox]]) -> CharSizes:
    """
    Estimate average character sizes from matched (text, bbox) pairs.

    Args:
        matched_pairs: List of (vlm_text, ocr_bbox) tuples

    Returns:
        CharSizes with average width and height
    """
    if not matched_pairs:
        return CharSizes(avg_char_width=10.0, avg_char_height=15.0)

    widths = []
    heights = []

    for text, bbox in matched_pairs:
        if len(text) == 0:
            continue

        char_width = bbox.width / len(text)
        char_height = bbox.height

        widths.append(char_width)
        heights.append(char_height)

    if not widths:
        return CharSizes(avg_char_width=10.0, avg_char_height=15.0)

    return CharSizes(
        avg_char_width=mean(widths),
        avg_char_height=mean(heights),
        std_char_width=stdev(widths) if len(widths) > 1 else 0.0,
        std_char_height=stdev(heights) if len(heights) > 1 else 0.0,
    )


def estimate_char_sizes_clustered(
    matched_pairs: List[Tuple[str, BBox]],
    std_multiplier: float = 2.0,
) -> List[CharSizeCluster]:
    """
    Estimate character sizes with clustering by font size.

    Groups measurements into clusters where heights are within std_multiplier * std.

    Args:
        matched_pairs: List of (vlm_text, ocr_bbox) tuples
        std_multiplier: Multiple of std for cluster membership

    Returns:
        List of CharSizeCluster objects
    """
    if not matched_pairs:
        return [CharSizeCluster(measurements=[(10.0, 15.0)])]

    # Calculate per-word char sizes
    measurements = []
    for text, bbox in matched_pairs:
        if len(text) == 0:
            continue
        char_width = bbox.width / len(text)
        char_height = bbox.height
        measurements.append((char_width, char_height))

    if not measurements:
        return [CharSizeCluster(measurements=[(10.0, 15.0)])]

    # Sort by height (proxy for font size)
    measurements.sort(key=lambda m: m[1])

    # Greedy clustering
    clusters: List[CharSizeCluster] = []
    current_cluster = CharSizeCluster(measurements=[measurements[0]])

    for m in measurements[1:]:
        _, height = m
        cluster_mean = current_cluster.avg_char_height
        cluster_std = current_cluster.std_char_height or 1.0

        if abs(height - cluster_mean) <= std_multiplier * cluster_std:
            current_cluster.measurements.append(m)
        else:
            clusters.append(current_cluster)
            current_cluster = CharSizeCluster(measurements=[m])

    if current_cluster.measurements:
        clusters.append(current_cluster)

    return clusters


# ============================================================================
# Pixel Detection for Orphan Placement
# ============================================================================

def detect_text_regions_in_area(
    image: "ndarray",
    search_region: BBox,
    page_width: float,
    page_height: float,
    existing_bboxes: List[BBox],
) -> List[BBox]:
    """
    Detect text regions within a search area using pixel detection.

    Args:
        image: Page image as numpy array
        search_region: Area to search within (PDF coordinates)
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        existing_bboxes: Already-placed bboxes to avoid overlap

    Returns:
        List of detected text region bboxes
    """
    import cv2

    if image is None:
        return []

    img_height, img_width = image.shape[:2]
    scale_x = img_width / page_width
    scale_y = img_height / page_height

    # Convert search region to image coordinates
    img_x0 = int(search_region.x0 * scale_x)
    img_y0 = int(search_region.y0 * scale_y)
    img_x1 = int(search_region.x1 * scale_x)
    img_y1 = int(search_region.y1 * scale_y)

    # Clamp to image bounds
    img_x0 = max(0, min(img_x0, img_width))
    img_y0 = max(0, min(img_y0, img_height))
    img_x1 = max(0, min(img_x1, img_width))
    img_y1 = max(0, min(img_y1, img_height))

    if img_x1 <= img_x0 or img_y1 <= img_y0:
        return []

    # Crop to search region
    cropped = image[img_y0:img_y1, img_x0:img_x1]

    # Convert to grayscale
    if len(cropped.shape) == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    else:
        gray = cropped

    # Binary threshold to find text pixels
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Dilate to connect nearby characters into words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Convert back to PDF coordinates (relative to search region)
        pdf_x0 = search_region.x0 + (x / scale_x)
        pdf_y0 = search_region.y0 + (y / scale_y)
        pdf_x1 = search_region.x0 + ((x + w) / scale_x)
        pdf_y1 = search_region.y0 + ((y + h) / scale_y)

        # Filter out tiny regions (noise)
        if (pdf_x1 - pdf_x0) < 5 or (pdf_y1 - pdf_y0) < 5:
            continue

        bbox = BBox(x0=pdf_x0, y0=pdf_y0, x1=pdf_x1, y1=pdf_y1)

        # Check for overlap with existing bboxes
        overlaps = False
        for existing in existing_bboxes:
            if bbox.iou(existing) > 0.3:
                overlaps = True
                break

        if not overlaps:
            regions.append(bbox)

    # Sort by reading order (top-to-bottom, left-to-right)
    regions.sort(key=lambda b: (b.y0, b.x0))

    return regions


# ============================================================================
# Orphan Segment Detection
# ============================================================================

def find_contiguous_orphan_segments(orphan_indices: List[int]) -> List[List[int]]:
    """
    Group orphan indices into contiguous segments.

    Consecutive orphan indices (e.g., [5, 6, 7]) are grouped together
    so they can be placed sequentially with proper line wrapping.

    Args:
        orphan_indices: List of VLM orphan indices (sorted)

    Returns:
        List of segments, where each segment is a list of consecutive indices
    """
    if not orphan_indices:
        return []

    # Ensure sorted
    sorted_indices = sorted(orphan_indices)
    segments = []
    current_segment = [sorted_indices[0]]

    for idx in sorted_indices[1:]:
        if idx == current_segment[-1] + 1:
            # Consecutive - add to current segment
            current_segment.append(idx)
        else:
            # Gap - start new segment
            segments.append(current_segment)
            current_segment = [idx]

    # Don't forget the last segment
    segments.append(current_segment)

    return segments


# ============================================================================
# Orphan Placement
# ============================================================================

def find_neighbor_anchors(
    vlm_idx: int,
    matched: Dict[int, Tuple[str, HarmonizedWord]],
    vlm_tokens: List[str],
) -> Tuple[Optional[HarmonizedWord], Optional[HarmonizedWord]]:
    """
    Find the nearest matched words before and after an orphan.

    Args:
        vlm_idx: Index of the orphan in VLM tokens
        matched: Dictionary of matched VLM indices
        vlm_tokens: Full list of VLM tokens

    Returns:
        (left_anchor, right_anchor) - either may be None
    """
    left_anchor = None
    right_anchor = None

    # Find left anchor (nearest matched word before vlm_idx)
    for i in range(vlm_idx - 1, -1, -1):
        if i in matched:
            _, ocr_word = matched[i]
            left_anchor = ocr_word
            break

    # Find right anchor (nearest matched word after vlm_idx)
    for i in range(vlm_idx + 1, len(vlm_tokens)):
        if i in matched:
            _, ocr_word = matched[i]
            right_anchor = ocr_word
            break

    return left_anchor, right_anchor


def are_on_same_line(bbox1: BBox, bbox2: BBox, tolerance: float = 20.0) -> bool:
    """Check if two bboxes are on the same line (y-ranges overlap or are close)."""
    y_overlap = min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0)
    if y_overlap > 0:
        return True
    y_gap = abs(bbox1.center_y - bbox2.center_y)
    return y_gap < tolerance


def interpolate_bbox(
    text: str,
    left_anchor: Optional[HarmonizedWord],
    right_anchor: Optional[HarmonizedWord],
    char_sizes: CharSizes,
) -> BBox:
    """
    Interpolate a bbox for orphan text based on neighbor anchors.

    Handles cross-line cases where anchors are on different lines.

    Args:
        text: The orphan word text
        left_anchor: Nearest matched word to the left
        right_anchor: Nearest matched word to the right
        char_sizes: Estimated character sizes

    Returns:
        Interpolated BBox
    """
    estimated_width = len(text) * char_sizes.avg_char_width

    if left_anchor and right_anchor:
        # Check if anchors are on the same line
        same_line = are_on_same_line(left_anchor.bbox, right_anchor.bbox)

        if same_line:
            # Place between anchors on same line
            x0 = left_anchor.bbox.x1 + 2
            available_width = right_anchor.bbox.x0 - x0 - 2

            if estimated_width <= available_width and available_width > 0:
                x1 = x0 + estimated_width
            elif available_width > 0:
                x1 = x0 + available_width
            else:
                # Not enough space - place after left anchor
                x1 = x0 + estimated_width

            y0 = (left_anchor.bbox.y0 + right_anchor.bbox.y0) / 2
            y1 = (left_anchor.bbox.y1 + right_anchor.bbox.y1) / 2
        else:
            # Cross-line case: anchors are on different lines
            # Place on left anchor's line, extending to the right
            x0 = left_anchor.bbox.x1 + 2
            x1 = x0 + estimated_width
            y0 = left_anchor.bbox.y0
            y1 = left_anchor.bbox.y1

    elif left_anchor:
        # Place to the right of left anchor
        x0 = left_anchor.bbox.x1 + 2
        x1 = x0 + estimated_width
        y0 = left_anchor.bbox.y0
        y1 = left_anchor.bbox.y1

    elif right_anchor:
        # Place to the left of right anchor
        x1 = right_anchor.bbox.x0 - 2
        x0 = max(0, x1 - estimated_width)  # Don't go negative
        y0 = right_anchor.bbox.y0
        y1 = right_anchor.bbox.y1

    else:
        # No anchors - use defaults
        x0 = 50
        x1 = x0 + estimated_width
        y0 = 100
        y1 = y0 + char_sizes.avg_char_height

    # Ensure valid bbox (x1 > x0)
    if x1 <= x0:
        x1 = x0 + estimated_width

    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def place_orphan_segment_with_wrap(
    orphan_words: List[str],
    left_anchor: Optional[HarmonizedWord],
    right_anchor: Optional[HarmonizedWord],
    char_sizes: CharSizes,
    cluster: Optional[Cluster] = None,
    page_width: float = 612.0,
) -> List[BBox]:
    """
    Place orphan words sequentially with line wrapping.

    When pixel detection fails and we have multiple consecutive orphan words,
    this function places them sequentially (not stacked) with proper line
    wrapping at cluster or page boundaries.

    Args:
        orphan_words: List of orphan word texts in reading order
        left_anchor: Last matched word before orphan segment
        right_anchor: First matched word after orphan segment (may be on different line)
        char_sizes: Character size estimates for this cluster
        cluster: Geometric cluster for boundary detection (optional)
        page_width: Page width in PDF points (fallback if no cluster)

    Returns:
        List of BBox for each orphan word
    """
    if not orphan_words:
        return []

    bboxes = []

    # Get cluster boundaries (or use page defaults)
    if cluster and cluster.bounding_box:
        cluster_x0 = cluster.bounding_box.x0
        cluster_x1 = cluster.bounding_box.x1
    else:
        cluster_x0 = 50  # Default left margin
        cluster_x1 = page_width - 50  # Default right margin

    # Estimate space width (typically 0.3 * char_width)
    space_width = char_sizes.avg_char_width * 0.3

    # Determine starting position
    if left_anchor:
        current_x = left_anchor.bbox.x1 + space_width
        current_y0 = left_anchor.bbox.y0
        current_y1 = left_anchor.bbox.y1
    elif right_anchor:
        # Start at cluster left edge, same line as right anchor
        current_x = cluster_x0 + 10  # Indent from left edge
        current_y0 = right_anchor.bbox.y0
        current_y1 = right_anchor.bbox.y1
    else:
        # No anchors - use defaults
        current_x = cluster_x0 + 10
        current_y0 = 100
        current_y1 = 100 + char_sizes.avg_char_height

    line_height = char_sizes.avg_char_height
    right_margin = 10  # Margin from right edge before wrapping

    for word in orphan_words:
        word_width = len(word) * char_sizes.avg_char_width

        # Check if word fits on current line
        if current_x + word_width > cluster_x1 - right_margin:
            # Wrap to next line
            current_x = cluster_x0 + 10  # Indent from left edge
            current_y0 += line_height + 5  # Line spacing
            current_y1 += line_height + 5

        # Place word
        bbox = BBox(
            x0=current_x,
            y0=current_y0,
            x1=current_x + word_width,
            y1=current_y1,
        )
        bboxes.append(bbox)

        # Move to next position
        current_x = bbox.x1 + space_width

    return bboxes


def find_cluster_for_word(
    word_bbox: BBox,
    clusters: List[Cluster],
) -> Optional[Cluster]:
    """
    Find the cluster that contains or is closest to a word's bbox.

    Args:
        word_bbox: The word's bounding box
        clusters: List of geometric clusters

    Returns:
        The containing or nearest cluster, or None if no clusters
    """
    if not clusters:
        return None

    # First, check if word is inside any cluster's bounding box
    for cluster in clusters:
        if cluster.bounding_box:
            cb = cluster.bounding_box
            # Check if word center is within cluster bounds (with margin)
            if (cb.x0 - 20 <= word_bbox.center_x <= cb.x1 + 20 and
                cb.y0 - 20 <= word_bbox.center_y <= cb.y1 + 20):
                return cluster

    # If no containing cluster, find the nearest one
    best_cluster = None
    best_distance = float('inf')

    for cluster in clusters:
        if cluster.bounding_box:
            cb = cluster.bounding_box
            # Distance from word center to cluster center
            dx = word_bbox.center_x - (cb.x0 + cb.x1) / 2
            dy = word_bbox.center_y - (cb.y0 + cb.y1) / 2
            distance = (dx * dx + dy * dy) ** 0.5

            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster

    return best_cluster


# ============================================================================
# Main Fusion Function
# ============================================================================

def fuse_vlm_with_ocr(
    ocr_words: List[HarmonizedWord],
    vlm_text: str,
    page_image: Optional["ndarray"] = None,
    page_width: float = 612.0,
    page_height: float = 792.0,
    max_context_window: int = 4,
    max_levenshtein: int = 3,
) -> List[HarmonizedWord]:
    """
    Fuse VLM text with OCR bounding boxes.

    This is the main entry point for VLM+OCR fusion.

    Algorithm:
    1. Geometric cluster OCR words to identify text blocks
    2. Match VLM tokens to OCR words using context-based matching
    3. Estimate character sizes from matched pairs (per cluster)
    4. Place VLM orphans using pixel detection + neighbor interpolation
    5. Keep OCR orphans with original text
    6. Order by reading order using geometric clustering

    Args:
        ocr_words: Words from traditional OCR with bboxes
        vlm_text: Raw text from VLM
        page_image: Optional page image for pixel detection (numpy array)
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        max_context_window: Maximum context expansion for matching
        max_levenshtein: Maximum edit distance for fuzzy matching

    Returns:
        List of HarmonizedWord with VLM text and OCR/estimated bboxes
    """
    # Handle empty inputs
    if not vlm_text or not vlm_text.strip():
        # Return OCR words as-is with ocr_only status
        result = []
        for word in ocr_words:
            new_word = HarmonizedWord(
                word_id=word.word_id,
                page=word.page,
                bbox=word.bbox,
                text=word.text,
                status="ocr_only",
                source=word.source,
                confidence=word.confidence,
            )
            result.append(new_word)
        return result

    if not ocr_words:
        # Can't place VLM words without OCR bboxes
        return []

    # Step 1: Geometric cluster OCR words
    x_thresh, y_thresh = calculate_distance_thresholds(ocr_words)
    clusters = build_clusters(ocr_words, x_thresh, y_thresh)

    # Step 2: Match VLM tokens to OCR words
    match_result = match_vlm_to_ocr(
        ocr_words, vlm_text, max_context_window, max_levenshtein
    )

    # Step 3: Estimate character sizes from matched pairs (per cluster)
    matched_pairs = [
        (vlm_txt, ocr_word.bbox)
        for _, (vlm_txt, ocr_word) in match_result.matched.items()
    ]
    char_sizes = estimate_char_sizes(matched_pairs)

    # Also get clustered sizes for different font sizes
    char_size_clusters = estimate_char_sizes_clustered(matched_pairs)

    # Step 4: Build output words
    result: List[HarmonizedWord] = []

    # Collect existing bboxes for overlap detection
    existing_bboxes = [w.bbox for w in ocr_words]

    # Add matched words (VLM text + OCR bbox)
    for vlm_idx, (vlm_token, ocr_word) in match_result.matched.items():
        new_word = HarmonizedWord(
            word_id=-1,  # Will be assigned later
            page=ocr_word.page,
            bbox=ocr_word.bbox,
            text=vlm_token,  # Use VLM text
            status="vlm_matched",
            source=f"vlm+{ocr_word.source}",
            confidence=1.0 if vlm_token == ocr_word.text else 0.9,
        )
        result.append(new_word)

    # Add VLM orphans (VLM text + pixel-detected or wrapped placement)
    # Group orphans into contiguous segments for sequential placement
    orphan_segments = find_contiguous_orphan_segments(match_result.vlm_orphans)

    for segment in orphan_segments:
        # Get word texts for this segment
        segment_words = [match_result.vlm_tokens[idx] for idx in segment]

        # Find anchors for the segment (first and last word)
        first_idx = segment[0]
        last_idx = segment[-1]

        left_anchor, _ = find_neighbor_anchors(
            first_idx, match_result.matched, match_result.vlm_tokens
        )
        _, right_anchor = find_neighbor_anchors(
            last_idx, match_result.matched, match_result.vlm_tokens
        )

        # Determine page from anchors
        page = 0
        if left_anchor:
            page = left_anchor.page
        elif right_anchor:
            page = right_anchor.page

        # Find the cluster for this segment (for boundary detection)
        anchor_bbox = left_anchor.bbox if left_anchor else (right_anchor.bbox if right_anchor else None)
        cluster = find_cluster_for_word(anchor_bbox, clusters) if anchor_bbox else None

        # Try pixel detection first if image is available
        segment_bboxes = []
        all_pixel_placed = False

        if page_image is not None and (left_anchor or right_anchor):
            # Define search region for the whole segment
            if left_anchor and right_anchor:
                if are_on_same_line(left_anchor.bbox, right_anchor.bbox):
                    search_region = BBox(
                        x0=left_anchor.bbox.x1,
                        y0=min(left_anchor.bbox.y0, right_anchor.bbox.y0) - 5,
                        x1=right_anchor.bbox.x0,
                        y1=max(left_anchor.bbox.y1, right_anchor.bbox.y1) + 5,
                    )
                else:
                    # Cross-line: search to the right of left anchor
                    search_region = BBox(
                        x0=left_anchor.bbox.x1,
                        y0=left_anchor.bbox.y0 - 5,
                        x1=page_width - 50,
                        y1=left_anchor.bbox.y1 + 5,
                    )
            elif left_anchor:
                search_region = BBox(
                    x0=left_anchor.bbox.x1,
                    y0=left_anchor.bbox.y0 - 5,
                    x1=page_width - 50,
                    y1=left_anchor.bbox.y1 + 5,
                )
            else:
                search_region = BBox(
                    x0=50,
                    y0=right_anchor.bbox.y0 - 5,
                    x1=right_anchor.bbox.x0,
                    y1=right_anchor.bbox.y1 + 5,
                )

            # Detect text regions in search area
            detected_regions = detect_text_regions_in_area(
                page_image, search_region, page_width, page_height, existing_bboxes
            )

            # Try to assign detected regions to words
            if detected_regions and len(detected_regions) >= len(segment_words):
                # Enough regions for all words - use them
                for word_text, region in zip(segment_words, detected_regions):
                    estimated_width = len(word_text) * char_sizes.avg_char_width
                    if region.width >= estimated_width * 0.5:  # 50% tolerance
                        segment_bboxes.append(region)
                        existing_bboxes.append(region)
                    else:
                        break

                if len(segment_bboxes) == len(segment_words):
                    all_pixel_placed = True

        # Fall back to sequential placement with wrapping if pixel detection didn't work
        if not all_pixel_placed:
            segment_bboxes = place_orphan_segment_with_wrap(
                segment_words,
                left_anchor,
                right_anchor,
                char_sizes,
                cluster=cluster,
                page_width=page_width,
            )

        # Create HarmonizedWord for each word in the segment
        for vlm_idx, (word_text, bbox) in zip(segment, zip(segment_words, segment_bboxes)):
            status = "vlm_pixel_placed" if all_pixel_placed else "vlm_interpolated"
            new_word = HarmonizedWord(
                word_id=-1,
                page=page,
                bbox=bbox,
                text=word_text,
                status=status,
                source="vlm",
                confidence=0.8 if status == "vlm_pixel_placed" else 0.7,
            )
            result.append(new_word)

    # Add OCR orphans (keep original text + bbox)
    for ocr_idx in match_result.ocr_orphans:
        ocr_word = ocr_words[ocr_idx]
        new_word = HarmonizedWord(
            word_id=-1,
            page=ocr_word.page,
            bbox=ocr_word.bbox,
            text=ocr_word.text,
            status="ocr_only",
            source=ocr_word.source,
            confidence=ocr_word.confidence,
        )
        result.append(new_word)

    # Step 5: Sort by reading order and assign word_ids using geometric clustering
    result = order_words_by_reading(result)

    return result
