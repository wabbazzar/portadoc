"""
Geometric clustering for document reading order.

This module implements a spatial clustering algorithm that determines the correct
reading order for extracted words in multi-column document layouts.

Algorithm Overview:
1. Calculate distance thresholds using Q1 * 1.5 of inter-word distances
2. Build clusters using union-find based on spatial proximity
3. Detect and reposition intra-cluster outliers
4. Order clusters top-to-bottom, words within clusters left-to-right/top-to-bottom
5. Assign sequential word_ids based on reading order

The Q1 (first quartile) threshold ensures aggressive gap detection, preventing
words from different columns from being grouped together.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from portadoc.models import HarmonizedWord, BBox


# Default thresholds when calculation isn't possible
DEFAULT_X_THRESHOLD = 50.0
DEFAULT_Y_THRESHOLD = 20.0
DEFAULT_Y_FUZZ = 5.0  # Default y-fuzz for row grouping when estimation fails

# Multipliers for threshold calculation
Q1_MULTIPLIER = 1.5  # For inter-cluster gaps
INTRA_CLUSTER_MULTIPLIER = 1.2  # For intra-cluster outlier detection
Y_FUZZ_MULTIPLIER = 2.0  # Multiplier on std for y-fuzz tolerance


@dataclass
class Cluster:
    """A group of spatially related words."""
    words: List[HarmonizedWord] = field(default_factory=list)

    @property
    def centroid(self) -> Tuple[float, float]:
        """Calculate centroid of all word centers."""
        if not self.words:
            return (0.0, 0.0)
        x_sum = sum(w.bbox.center_x for w in self.words)
        y_sum = sum(w.bbox.center_y for w in self.words)
        n = len(self.words)
        return (x_sum / n, y_sum / n)

    @property
    def bounding_box(self) -> Optional[BBox]:
        """Get bounding box encompassing all words."""
        if not self.words:
            return None
        x0 = min(w.bbox.x0 for w in self.words)
        y0 = min(w.bbox.y0 for w in self.words)
        x1 = max(w.bbox.x1 for w in self.words)
        y1 = max(w.bbox.y1 for w in self.words)
        return BBox(x0, y0, x1, y1)

    @property
    def min_y(self) -> float:
        """Minimum y coordinate (top of cluster)."""
        if not self.words:
            return 0.0
        return min(w.bbox.y0 for w in self.words)

    @property
    def min_x(self) -> float:
        """Minimum x coordinate (left edge of cluster)."""
        if not self.words:
            return 0.0
        return min(w.bbox.x0 for w in self.words)


class UnionFind:
    """Union-Find data structure for clustering."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def horizontal_gap(w1: HarmonizedWord, w2: HarmonizedWord) -> float:
    """
    Calculate horizontal gap between two word bboxes.

    Returns 0 if boxes overlap horizontally.
    """
    if w1.bbox.x1 <= w2.bbox.x0:
        # w1 is to the left of w2
        return w2.bbox.x0 - w1.bbox.x1
    elif w2.bbox.x1 <= w1.bbox.x0:
        # w2 is to the left of w1
        return w1.bbox.x0 - w2.bbox.x1
    else:
        # Overlapping horizontally
        return 0.0


def vertical_gap(w1: HarmonizedWord, w2: HarmonizedWord) -> float:
    """
    Calculate vertical gap between two word bboxes.

    Returns 0 if boxes overlap vertically.
    """
    if w1.bbox.y1 <= w2.bbox.y0:
        # w1 is above w2
        return w2.bbox.y0 - w1.bbox.y1
    elif w2.bbox.y1 <= w1.bbox.y0:
        # w2 is above w1
        return w1.bbox.y0 - w2.bbox.y1
    else:
        # Overlapping vertically
        return 0.0


def x_overlap_ratio(w1: HarmonizedWord, w2: HarmonizedWord) -> float:
    """
    Calculate horizontal overlap as a ratio of the smaller word's width.

    Returns value between 0.0 (no overlap) and 1.0+ (full overlap or more).
    """
    overlap_x0 = max(w1.bbox.x0, w2.bbox.x0)
    overlap_x1 = min(w1.bbox.x1, w2.bbox.x1)
    overlap_width = max(0, overlap_x1 - overlap_x0)

    min_width = min(w1.bbox.width, w2.bbox.width)
    if min_width <= 0:
        return 0.0

    return overlap_width / min_width


def y_overlap_ratio(w1: HarmonizedWord, w2: HarmonizedWord) -> float:
    """
    Calculate vertical overlap as a ratio of the smaller word's height.

    Returns value between 0.0 (no overlap) and 1.0+ (full overlap or more).
    """
    overlap_y0 = max(w1.bbox.y0, w2.bbox.y0)
    overlap_y1 = min(w1.bbox.y1, w2.bbox.y1)
    overlap_height = max(0, overlap_y1 - overlap_y0)

    min_height = min(w1.bbox.height, w2.bbox.height)
    if min_height <= 0:
        return 0.0

    return overlap_height / min_height


def calculate_distance_thresholds(words: List[HarmonizedWord]) -> Tuple[float, float]:
    """
    Calculate clustering thresholds using Q1 * 1.5 of inter-word distances.

    Uses first quartile to aggressively detect gaps between columns/sections.

    Args:
        words: List of words to analyze

    Returns:
        Tuple of (x_threshold, y_threshold)
    """
    if len(words) < 2:
        return (DEFAULT_X_THRESHOLD, DEFAULT_Y_THRESHOLD)

    # Sort by rough reading order (top-to-bottom, left-to-right)
    sorted_words = sorted(words, key=lambda w: (w.page, w.bbox.center_y, w.bbox.center_x))

    x_distances = []
    y_distances = []

    # Calculate distances between consecutive words
    for i in range(len(sorted_words) - 1):
        w1, w2 = sorted_words[i], sorted_words[i + 1]

        # Only consider words on same page
        if w1.page != w2.page:
            continue

        x_dist = horizontal_gap(w1, w2)
        y_dist = vertical_gap(w1, w2)

        if x_dist > 0:
            x_distances.append(x_dist)
        if y_dist > 0:
            y_distances.append(y_dist)

    # Calculate Q1 * multiplier for thresholds
    if x_distances:
        x_q1 = np.percentile(x_distances, 25)
        x_threshold = x_q1 * Q1_MULTIPLIER
    else:
        x_threshold = DEFAULT_X_THRESHOLD

    if y_distances:
        y_q1 = np.percentile(y_distances, 25)
        y_threshold = y_q1 * Q1_MULTIPLIER
    else:
        y_threshold = DEFAULT_Y_THRESHOLD

    # Ensure minimum thresholds
    x_threshold = max(x_threshold, 5.0)
    y_threshold = max(y_threshold, 3.0)

    return (x_threshold, y_threshold)


def estimate_y_fuzz(words: List[HarmonizedWord], x_threshold: float) -> float:
    """
    Estimate y-fuzz tolerance based on y-variability between horizontally adjacent words.

    The idea: words that are horizontally adjacent (small x-gap) but on the "same visual line"
    may have slight y-coordinate variations due to:
    - Different font sizes (e.g., "NORTHWEST" vs "VETERINARY")
    - Baseline variations
    - OCR bbox noise

    We find pairs of horizontally adjacent words and measure the std of their y-center
    differences. This gives us an estimate of the y-noise for same-row words.

    Args:
        words: List of words to analyze
        x_threshold: Maximum horizontal gap to consider words "adjacent"

    Returns:
        Y-fuzz tolerance value (in pixels/points)
    """
    if len(words) < 2:
        return DEFAULT_Y_FUZZ

    # Find pairs of horizontally adjacent words
    # These are words that are close horizontally (potential same-row)
    y_center_diffs = []

    # Use a slightly larger x_threshold to find more same-row candidates
    adjacent_x_threshold = x_threshold * 3.0

    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if i >= j:
                continue
            if w1.page != w2.page:
                continue

            x_gap = horizontal_gap(w1, w2)

            # Only consider horizontally adjacent pairs
            if x_gap > adjacent_x_threshold:
                continue

            # Also require some y-overlap to ensure these are plausibly same-row
            # Use a relaxed check: y-ranges should at least partially overlap
            # or be within a reasonable distance
            y_gap = vertical_gap(w1, w2)
            min_height = min(w1.bbox.height, w2.bbox.height)

            # Allow pairs that overlap or are within half a character height
            if y_gap > min_height * 0.5:
                continue

            # Calculate y-center difference
            y_diff = abs(w1.bbox.center_y - w2.bbox.center_y)
            y_center_diffs.append(y_diff)

    if len(y_center_diffs) < 3:
        # Not enough data, use default
        return DEFAULT_Y_FUZZ

    # Use std of y-center differences as noise estimate
    y_std = np.std(y_center_diffs)

    # Also consider the median - if median is low but std is high,
    # there might be outliers. Use a robust estimate.
    y_median = np.median(y_center_diffs)

    # Y-fuzz should be at least the median difference plus some margin
    # and at least 2x the std to capture most variation
    y_fuzz = max(
        y_median + y_std,  # Median plus one std
        y_std * Y_FUZZ_MULTIPLIER,  # 2x std
        DEFAULT_Y_FUZZ  # Minimum floor
    )

    # Cap at a reasonable maximum (half typical line height)
    avg_height = np.mean([w.bbox.height for w in words])
    y_fuzz = min(y_fuzz, avg_height * 0.5)

    return y_fuzz


def detect_column_boundaries(words: List[HarmonizedWord], y_fuzz: float = DEFAULT_Y_FUZZ) -> List[float]:
    """
    Detect major vertical column boundaries in the document layout.

    Uses row-based gap analysis: finds x-gaps within each row, then identifies
    boundaries that are consistent across multiple rows.

    Args:
        words: List of words to analyze
        y_fuzz: Y-tolerance for grouping words into rows (based on y-center)

    Returns:
        List of x-coordinates representing column boundaries
    """
    if len(words) < 2:
        return []

    # Group words into rows based on y-center proximity (using y_fuzz)
    rows = []
    sorted_by_y = sorted(words, key=lambda w: w.bbox.center_y)

    current_row = [sorted_by_y[0]]
    row_y_center_max = sorted_by_y[0].bbox.center_y

    for word in sorted_by_y[1:]:
        word_y_center = word.bbox.center_y
        if word_y_center <= row_y_center_max + y_fuzz:
            current_row.append(word)
            row_y_center_max = max(row_y_center_max, word_y_center)
        else:
            rows.append(current_row)
            current_row = [word]
            row_y_center_max = word_y_center
    rows.append(current_row)

    # Find gaps within each row
    all_gaps = []
    for row in rows:
        if len(row) < 2:
            continue
        sorted_row = sorted(row, key=lambda w: w.bbox.x0)
        for i in range(len(sorted_row) - 1):
            w1, w2 = sorted_row[i], sorted_row[i + 1]
            gap = w2.bbox.x0 - w1.bbox.x1
            if gap > 0:
                # Store gap with its approximate x-position (midpoint)
                all_gaps.append((gap, (w1.bbox.x1 + w2.bbox.x0) / 2))

    if not all_gaps:
        return []

    # Find significant gaps (using median as reference)
    gap_sizes = [g[0] for g in all_gaps]
    median_gap = np.median(gap_sizes)

    # A column boundary should be at least 3x the median gap
    # This helps distinguish column gaps from word spacing
    threshold = median_gap * 3

    # Only consider gaps significantly larger than typical word spacing
    significant_gaps = [(size, pos) for size, pos in all_gaps if size >= threshold]

    if not significant_gaps:
        return []

    # Return positions of significant gaps as boundaries
    boundaries = sorted(set(pos for _, pos in significant_gaps))

    return boundaries


def assign_column(word: HarmonizedWord, boundaries: List[float]) -> int:
    """
    Assign a word to a column based on its position relative to boundaries.

    Args:
        word: The word to assign
        boundaries: List of x-coordinates representing column boundaries

    Returns:
        Column index (0-based)
    """
    word_center_x = word.bbox.center_x
    for i, boundary in enumerate(boundaries):
        if word_center_x < boundary:
            return i
    return len(boundaries)


def build_clusters(
    words: List[HarmonizedWord],
    x_threshold: float,
    y_threshold: float,
    y_fuzz: float = DEFAULT_Y_FUZZ
) -> List[Cluster]:
    """
    Build clusters using union-find based on spatial proximity and column detection.

    Algorithm:
    1. Detect column boundaries using gap analysis
    2. Assign words to columns
    3. Only allow clustering between words in the same column
    4. Within columns, use proximity rules for clustering

    Clustering rules within a column:
    1. Same row (y-overlap) AND horizontally close
    2. Column aligned (x-overlap) AND vertically close
    3. Very close in both dimensions

    Args:
        words: List of words to cluster
        x_threshold: Maximum horizontal gap for same-row clustering
        y_threshold: Maximum vertical gap for adjacent lines
        y_fuzz: Y-tolerance for row grouping in column detection

    Returns:
        List of Cluster objects
    """
    if not words:
        return []

    n = len(words)
    uf = UnionFind(n)

    # Detect column boundaries (use y_fuzz for row grouping)
    boundaries = detect_column_boundaries(words, y_fuzz=y_fuzz)

    # Assign words to columns
    word_columns = [assign_column(w, boundaries) for w in words]

    # Thresholds for different connection types
    X_OVERLAP_MIN = 0.30  # 30% x-overlap needed for column alignment
    Y_OVERLAP_MIN = 0.50  # 50% y-overlap for same-row detection
    VERTICAL_MULTIPLIER = 3.0  # Allow larger y-gaps for column-aligned words
    SAME_ROW_X_MULTIPLIER = 8.0  # Allow larger x-gaps for same-row items (label-content)

    # Check all pairs and union if they should be connected
    for i in range(n):
        for j in range(i + 1, n):
            w1, w2 = words[i], words[j]

            # Different pages = different clusters
            if w1.page != w2.page:
                continue

            x_dist = horizontal_gap(w1, w2)
            y_dist = vertical_gap(w1, w2)
            x_overlap = x_overlap_ratio(w1, w2)
            y_overlap = y_overlap_ratio(w1, w2)

            should_connect = False

            # Rule 1: Same row (y-overlap) AND horizontally close
            # This connects words on the same visual line
            # Same-row words can ignore column boundaries ONLY if the gap is small
            # (This allows headers like "NORTHWEST VETERINARY" to connect while
            #  keeping multi-column content like "Grove" | "Email:" separate)
            if y_overlap >= Y_OVERLAP_MIN and x_dist <= x_threshold * SAME_ROW_X_MULTIPLIER:
                # Check if this is a small gap (typical word spacing) or a large gap (column gap)
                # For small gaps, ignore column boundaries; for large gaps, respect them
                is_small_gap = x_dist <= x_threshold * 2.0  # Small gap: 2x threshold
                same_column = word_columns[i] == word_columns[j]

                if same_column or is_small_gap:
                    should_connect = True

            # For rules 2 and 3, always respect column boundaries
            elif word_columns[i] == word_columns[j]:
                # Rule 2: Column aligned (x-overlap) AND vertically close
                # This connects words that stack vertically in the same column
                # e.g., "4827 Maple" / "Grove" / "Avenue, Apt"
                if x_overlap >= X_OVERLAP_MIN and y_dist <= y_threshold * VERTICAL_MULTIPLIER:
                    should_connect = True

                # Rule 3: Very close in both dimensions (fallback)
                elif x_dist <= x_threshold and y_dist <= y_threshold:
                    should_connect = True

            if should_connect:
                uf.union(i, j)

    # Group words by cluster root
    clusters_dict = {}
    for i, word in enumerate(words):
        root = uf.find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(word)

    # Create Cluster objects
    clusters = [Cluster(words=words_list) for words_list in clusters_dict.values()]

    return clusters


def detect_intra_cluster_outliers(cluster: Cluster) -> List[HarmonizedWord]:
    """
    Detect words that are outliers within a cluster.

    A word is an outlier if its distance to neighbors exceeds 1.2x the
    average intra-cluster distance.

    Args:
        cluster: Cluster to analyze

    Returns:
        List of outlier words
    """
    if len(cluster.words) < 3:
        return []

    # Sort words by reading order within cluster
    sorted_words = sorted(
        cluster.words,
        key=lambda w: (w.bbox.center_y, w.bbox.center_x)
    )

    # Calculate intra-cluster distances
    x_distances = []
    y_distances = []

    for i in range(len(sorted_words) - 1):
        w1, w2 = sorted_words[i], sorted_words[i + 1]
        x_dist = horizontal_gap(w1, w2)
        y_dist = vertical_gap(w1, w2)
        if x_dist > 0:
            x_distances.append((x_dist, i))
        if y_dist > 0:
            y_distances.append((y_dist, i))

    if not x_distances and not y_distances:
        return []

    # Calculate thresholds
    x_thresh = np.mean([d for d, _ in x_distances]) * INTRA_CLUSTER_MULTIPLIER if x_distances else float('inf')
    y_thresh = np.mean([d for d, _ in y_distances]) * INTRA_CLUSTER_MULTIPLIER if y_distances else float('inf')

    # Find gaps exceeding threshold
    outliers = set()
    centroid = cluster.centroid

    for dist, idx in x_distances:
        if dist > x_thresh:
            # Determine which side of gap is the outlier (furthest from centroid)
            w1, w2 = sorted_words[idx], sorted_words[idx + 1]
            d1 = abs(w1.bbox.center_x - centroid[0]) + abs(w1.bbox.center_y - centroid[1])
            d2 = abs(w2.bbox.center_x - centroid[0]) + abs(w2.bbox.center_y - centroid[1])
            outlier = w1 if d1 > d2 else w2
            outliers.add(id(outlier))

    for dist, idx in y_distances:
        if dist > y_thresh:
            w1, w2 = sorted_words[idx], sorted_words[idx + 1]
            d1 = abs(w1.bbox.center_x - centroid[0]) + abs(w1.bbox.center_y - centroid[1])
            d2 = abs(w2.bbox.center_x - centroid[0]) + abs(w2.bbox.center_y - centroid[1])
            outlier = w1 if d1 > d2 else w2
            outliers.add(id(outlier))

    return [w for w in cluster.words if id(w) in outliers]


def reposition_outlier(outlier: HarmonizedWord, cluster: Cluster) -> str:
    """
    Determine where an outlier should be repositioned within its cluster.

    Based on vector from word center to cluster centroid:
    - If outlier is left/above centroid: move to front
    - If outlier is right/below centroid: move to end

    Args:
        outlier: The outlier word
        cluster: The cluster it belongs to

    Returns:
        "front" or "end"
    """
    centroid = cluster.centroid
    word_center = (outlier.bbox.center_x, outlier.bbox.center_y)

    # Vector from word to centroid
    dx = centroid[0] - word_center[0]
    dy = centroid[1] - word_center[1]

    # If word is to the left or above centroid, it should go to front
    # Reading order is left-to-right, top-to-bottom
    # So if dx > 0 (word is left of centroid) or dy > 0 (word is above centroid)
    # it should come before the cluster

    # Weight x more heavily since columns are the main concern
    weighted_direction = dx * 2 + dy

    if weighted_direction > 0:
        return "front"
    else:
        return "end"


def sort_words_within_cluster(cluster: Cluster, y_fuzz: float = DEFAULT_Y_FUZZ) -> List[HarmonizedWord]:
    """
    Sort words within a cluster by reading order.

    Uses row detection with y-fuzz tolerance: words whose y-centers are within
    y_fuzz of each other are considered the same row, then sorted left-to-right.
    Rows are sorted top-to-bottom.

    Args:
        cluster: Cluster to sort
        y_fuzz: Y-tolerance for considering words on the same row (based on y-center)

    Returns:
        Sorted list of words
    """
    if not cluster.words:
        return []

    words = cluster.words.copy()

    # Group into rows based on y-center proximity (not just overlap)
    # This handles cases where words on the same visual line have slight y-variations
    rows = []
    sorted_by_y = sorted(words, key=lambda w: w.bbox.center_y)

    current_row = [sorted_by_y[0]]
    # Track the y-center range of the current row
    row_y_center_min = sorted_by_y[0].bbox.center_y
    row_y_center_max = sorted_by_y[0].bbox.center_y

    for word in sorted_by_y[1:]:
        word_y_center = word.bbox.center_y

        # Check if word's y-center is within y_fuzz of current row's y-center range
        # This allows words with slight y-variations to be grouped together
        if word_y_center <= row_y_center_max + y_fuzz:
            current_row.append(word)
            row_y_center_max = max(row_y_center_max, word_y_center)
            row_y_center_min = min(row_y_center_min, word_y_center)
        else:
            rows.append(current_row)
            current_row = [word]
            row_y_center_min = word_y_center
            row_y_center_max = word_y_center

    rows.append(current_row)

    # Sort each row left-to-right, then flatten
    result = []
    for row in rows:
        row_sorted = sorted(row, key=lambda w: w.bbox.x0)
        result.extend(row_sorted)

    return result


def group_clusters_into_row_bands(clusters: List[Cluster], y_tolerance: float = 15.0) -> List[List[Cluster]]:
    """
    Group clusters into row bands based on y-overlap.

    Clusters whose y-ranges overlap (or are within tolerance) are grouped
    into the same row band. This handles slight y-variations in same-row content.

    Args:
        clusters: List of clusters to group
        y_tolerance: Maximum y-gap to still consider same row

    Returns:
        List of row bands, each containing clusters on that row
    """
    if not clusters:
        return []

    # Sort clusters by min_y first
    sorted_clusters = sorted(clusters, key=lambda c: c.min_y)

    row_bands = []
    current_band = [sorted_clusters[0]]
    band_y_max = sorted_clusters[0].bounding_box.y1 if sorted_clusters[0].bounding_box else sorted_clusters[0].min_y

    for cluster in sorted_clusters[1:]:
        cluster_y_min = cluster.min_y

        # Check if this cluster overlaps with or is close to current band
        if cluster_y_min <= band_y_max + y_tolerance:
            current_band.append(cluster)
            if cluster.bounding_box:
                band_y_max = max(band_y_max, cluster.bounding_box.y1)
        else:
            # Start new band
            row_bands.append(current_band)
            current_band = [cluster]
            band_y_max = cluster.bounding_box.y1 if cluster.bounding_box else cluster.min_y

    row_bands.append(current_band)
    return row_bands


def order_words_by_reading(words: List[HarmonizedWord]) -> List[HarmonizedWord]:
    """
    Order words by proper reading sequence using geometric clustering.

    Main entry point for the reading order algorithm.

    The ordering strategy (ROW-FIRST, like reading a magazine):
    1. Build clusters to identify multi-line content blocks
    2. Group clusters into ROW BANDS (clusters with overlapping y-ranges)
    3. Within each row band, sort clusters left-to-right by min_x
    4. Order row bands top-to-bottom
    5. When a cluster is encountered, read ALL of it before moving on

    This handles:
    - Multi-line blocks (like addresses) being read completely
    - Same-row content with slight y-variations being read left-to-right
    - Magazine-style chaotic layouts

    Args:
        words: List of HarmonizedWord objects (any order)

    Returns:
        List of HarmonizedWord with:
        - Proper reading order (row-first, clusters read completely)
        - Sequential word_ids (0, 1, 2, ..., n-1)
    """
    if not words:
        return []

    # Group by page first
    pages = {}
    for word in words:
        if word.page not in pages:
            pages[word.page] = []
        pages[word.page].append(word)

    result = []

    for page_num in sorted(pages.keys()):
        page_words = pages[page_num]

        # Calculate thresholds for this page
        x_thresh, y_thresh = calculate_distance_thresholds(page_words)

        # Estimate y-fuzz for row grouping based on y-variability of adjacent words
        y_fuzz = estimate_y_fuzz(page_words, x_thresh)

        # Build clusters (pass y_fuzz for consistent row grouping)
        clusters = build_clusters(page_words, x_thresh, y_thresh, y_fuzz=y_fuzz)

        # Handle outliers within each cluster
        for cluster in clusters:
            outliers = detect_intra_cluster_outliers(cluster)

        # Group clusters into row bands (handles y-variations)
        # Use y_fuzz for the tolerance
        row_bands = group_clusters_into_row_bands(clusters, y_tolerance=y_fuzz)

        # Process each row band: sort clusters left-to-right within band
        for band in row_bands:
            # Sort clusters in this band by min_x (left-to-right)
            band_sorted = sorted(band, key=lambda c: c.min_x)

            # Add words from each cluster in order
            for cluster in band_sorted:
                ordered_cluster_words = sort_words_within_cluster(cluster, y_fuzz=y_fuzz)
                result.extend(ordered_cluster_words)

    # Assign sequential word_ids
    for i, word in enumerate(result):
        word.word_id = i

    return result


def reorder_for_output(words: List[HarmonizedWord]) -> List[HarmonizedWord]:
    """
    Convenience function to reorder words for CSV output.

    This is the function to call from the extraction pipeline.

    Args:
        words: Extracted words in any order

    Returns:
        Words ordered by reading sequence with sequential word_ids
    """
    return order_words_by_reading(words)
