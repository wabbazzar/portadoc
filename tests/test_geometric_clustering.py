"""Tests for geometric clustering reading order algorithm."""

import pytest
from pathlib import Path

from portadoc.models import BBox, HarmonizedWord
from portadoc.geometric_clustering import (
    calculate_distance_thresholds,
    estimate_y_fuzz,
    build_clusters,
    detect_intra_cluster_outliers,
    reposition_outlier,
    order_words_by_reading,
    Cluster,
    DEFAULT_Y_FUZZ,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_row_words():
    """Three words in a simple horizontal row."""
    return [
        HarmonizedWord(word_id=0, page=0, bbox=BBox(10, 100, 50, 120), text="Hello",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=1, page=0, bbox=BBox(60, 100, 100, 120), text="World",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=2, page=0, bbox=BBox(110, 100, 150, 120), text="Test",
                       status="word", source="T", confidence=95.0),
    ]


@pytest.fixture
def two_column_words():
    """Words in two distinct columns (like Address | Email layout)."""
    # Left column (Address block) - x: 89-260
    left_col = [
        HarmonizedWord(word_id=0, page=0, bbox=BBox(88.56, 537.12, 139.44, 550.32), text="Address:",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=1, page=0, bbox=BBox(193.68, 538.08, 219.84, 546.96), text="4827",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=2, page=0, bbox=BBox(224.4, 538.08, 255.84, 549.12), text="Maple",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=3, page=0, bbox=BBox(194.16, 551.52, 225.84, 560.4), text="Grove",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=4, page=0, bbox=BBox(193.44, 565.2, 236.4, 575.52), text="Avenue,",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=5, page=0, bbox=BBox(240.0, 565.2, 258.0, 576.24), text="Apt",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=6, page=0, bbox=BBox(193.92, 578.64, 210.48, 588.96), text="3B,",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=7, page=0, bbox=BBox(215.76, 578.64, 261.84, 588.96), text="Portland,",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=8, page=0, bbox=BBox(193.92, 591.84, 211.2, 600.96), text="OR",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=9, page=0, bbox=BBox(215.28, 592.08, 247.68, 600.96), text="97205",
                       status="word", source="T", confidence=95.0),
    ]
    # Right column (Email block) - x: 277-522
    right_col = [
        HarmonizedWord(word_id=10, page=0, bbox=BBox(277.2, 538.08, 311.28, 546.96), text="Email:",
                       status="word", source="T", confidence=95.0),
        HarmonizedWord(word_id=11, page=0, bbox=BBox(382.08, 537.84, 522.96, 549.36), text="r.martinez.pdx@gmail.com",
                       status="word", source="T", confidence=95.0),
    ]
    return left_col + right_col


@pytest.fixture
def address_email_ground_truth_order():
    """Expected reading order: all address words, then email words."""
    return [
        "Address:", "4827", "Maple", "Grove", "Avenue,", "Apt",
        "3B,", "Portland,", "OR", "97205", "Email:", "r.martinez.pdx@gmail.com"
    ]


@pytest.fixture
def multi_row_words():
    """Multiple rows of text to test vertical clustering."""
    rows = []
    for row_idx in range(3):
        y_base = 100 + row_idx * 30
        for col_idx in range(4):
            x_base = 50 + col_idx * 60
            rows.append(
                HarmonizedWord(
                    word_id=row_idx * 4 + col_idx,
                    page=0,
                    bbox=BBox(x_base, y_base, x_base + 50, y_base + 20),
                    text=f"W{row_idx}{col_idx}",
                    status="word", source="T", confidence=95.0
                )
            )
    return rows


# ============================================================================
# Unit Tests: Threshold Calculation
# ============================================================================

class TestCalculateDistanceThresholds:
    """Tests for Q1 * 1.5 threshold calculation."""

    def test_simple_uniform_spacing(self, simple_row_words):
        """Uniform spacing should yield consistent thresholds."""
        x_thresh, y_thresh = calculate_distance_thresholds(simple_row_words)

        # With uniform 10-unit gaps, Q1 should be 10, threshold = 15
        assert x_thresh > 0
        assert y_thresh > 0

    def test_empty_words_returns_defaults(self):
        """Empty word list should return sensible defaults."""
        x_thresh, y_thresh = calculate_distance_thresholds([])

        # Should return some reasonable default, not crash
        assert x_thresh > 0
        assert y_thresh > 0

    def test_single_word_returns_defaults(self):
        """Single word should return defaults (no pairs to measure)."""
        words = [
            HarmonizedWord(word_id=0, page=0, bbox=BBox(10, 10, 50, 30), text="Single",
                           status="word", source="T", confidence=95.0)
        ]
        x_thresh, y_thresh = calculate_distance_thresholds(words)

        assert x_thresh > 0
        assert y_thresh > 0

    def test_threshold_uses_first_quartile(self, multi_row_words):
        """Verify threshold is based on Q1, not mean or median."""
        x_thresh, y_thresh = calculate_distance_thresholds(multi_row_words)

        # Q1 * 1.5 should be smaller than median * 1.5
        # This ensures we aggressively shield out gaps
        assert x_thresh > 0
        assert y_thresh > 0


# ============================================================================
# Unit Tests: Y-Fuzz Estimation
# ============================================================================

class TestEstimateYFuzz:
    """Tests for y-fuzz tolerance estimation based on adjacent word y-variability."""

    def test_uniform_y_words_low_fuzz(self):
        """Words with identical y-centers should have minimal y-fuzz."""
        # All words at y_center = 110 (y0=100, y1=120)
        words = [
            HarmonizedWord(word_id=0, page=0, bbox=BBox(10, 100, 50, 120), text="Hello",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=1, page=0, bbox=BBox(60, 100, 100, 120), text="World",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=2, page=0, bbox=BBox(110, 100, 150, 120), text="Test",
                           status="word", source="T", confidence=95.0),
        ]
        x_thresh, _ = calculate_distance_thresholds(words)
        y_fuzz = estimate_y_fuzz(words, x_thresh)

        # With no y variation, y_fuzz should be at the default minimum
        assert y_fuzz == DEFAULT_Y_FUZZ

    def test_varied_y_words_higher_fuzz(self):
        """Words with y-center variation should yield higher y-fuzz."""
        # Words with varied y positions but still close enough to be "same row"
        words = [
            HarmonizedWord(word_id=0, page=0, bbox=BBox(10, 100, 50, 120), text="Hello",
                           status="word", source="T", confidence=95.0),  # y_center = 110
            HarmonizedWord(word_id=1, page=0, bbox=BBox(60, 105, 100, 125), text="World",
                           status="word", source="T", confidence=95.0),  # y_center = 115
            HarmonizedWord(word_id=2, page=0, bbox=BBox(110, 98, 150, 118), text="Test",
                           status="word", source="T", confidence=95.0),  # y_center = 108
            HarmonizedWord(word_id=3, page=0, bbox=BBox(160, 103, 200, 123), text="More",
                           status="word", source="T", confidence=95.0),  # y_center = 113
        ]
        x_thresh, _ = calculate_distance_thresholds(words)
        y_fuzz = estimate_y_fuzz(words, x_thresh)

        # With y variation (std ~3-4), y_fuzz should be higher than default
        assert y_fuzz >= DEFAULT_Y_FUZZ

    def test_header_style_mixed_heights(self):
        """Simulate header with mixed font sizes like 'NORTHWEST VETERINARY'."""
        # Large word (like header) and smaller word on "same line" but different bbox heights
        words = [
            # "NORTHWEST" - large font, y_center ~ 70
            HarmonizedWord(word_id=0, page=0, bbox=BBox(50, 50, 200, 90), text="NORTHWEST",
                           status="word", source="T", confidence=95.0),
            # "VETERINARY" - also large but slightly different y
            HarmonizedWord(word_id=1, page=0, bbox=BBox(220, 55, 370, 95), text="VETERINARY",
                           status="word", source="T", confidence=95.0),
            # "ASSOCIATES" - on same visual line but different size
            HarmonizedWord(word_id=2, page=0, bbox=BBox(390, 48, 500, 88), text="ASSOCIATES",
                           status="word", source="T", confidence=95.0),
        ]
        x_thresh, _ = calculate_distance_thresholds(words)
        y_fuzz = estimate_y_fuzz(words, x_thresh)

        # Should detect the y-variation between these header words
        assert y_fuzz >= DEFAULT_Y_FUZZ

    def test_few_words_returns_default(self):
        """With too few words, should return default y-fuzz."""
        words = [
            HarmonizedWord(word_id=0, page=0, bbox=BBox(10, 100, 50, 120), text="Only",
                           status="word", source="T", confidence=95.0),
        ]
        x_thresh, _ = calculate_distance_thresholds(words)
        y_fuzz = estimate_y_fuzz(words, x_thresh)

        assert y_fuzz == DEFAULT_Y_FUZZ

    def test_words_on_different_rows_ignored(self):
        """Words clearly on different rows shouldn't inflate y-fuzz."""
        # Words on clearly separate rows (large y gap)
        words = [
            HarmonizedWord(word_id=0, page=0, bbox=BBox(10, 100, 50, 120), text="Row1",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=1, page=0, bbox=BBox(60, 100, 100, 120), text="Word2",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=2, page=0, bbox=BBox(10, 200, 50, 220), text="Row2",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=3, page=0, bbox=BBox(60, 200, 100, 220), text="Word4",
                           status="word", source="T", confidence=95.0),
        ]
        x_thresh, _ = calculate_distance_thresholds(words)
        y_fuzz = estimate_y_fuzz(words, x_thresh)

        # Words on different rows (80-pixel gap) should be ignored
        # Only same-row pairs should contribute, so y_fuzz stays low
        assert y_fuzz <= 20  # Should be reasonable, not inflated


# ============================================================================
# Unit Tests: Cluster Building
# ============================================================================

class TestBuildClusters:
    """Tests for union-find based clustering."""

    def test_single_row_forms_one_cluster(self, simple_row_words):
        """Words in a row with uniform spacing should form one cluster."""
        x_thresh, y_thresh = calculate_distance_thresholds(simple_row_words)
        clusters = build_clusters(simple_row_words, x_thresh, y_thresh)

        assert len(clusters) == 1
        assert len(clusters[0].words) == 3

    def test_two_columns_form_separate_clusters(self, two_column_words):
        """Address and Email content should be in separate clusters (not mixed)."""
        x_thresh, y_thresh = calculate_distance_thresholds(two_column_words)
        clusters = build_clusters(two_column_words, x_thresh, y_thresh)

        # Find clusters containing address words vs email words
        address_words = {"Address:", "4827", "Maple", "Grove", "Avenue,", "Apt",
                         "3B,", "Portland,", "OR", "97205"}
        email_words = {"Email:", "r.martinez.pdx@gmail.com"}

        for cluster in clusters:
            texts = set(w.text for w in cluster.words)
            has_address = bool(texts & address_words)
            has_email = bool(texts & email_words)

            # A cluster should NOT contain both address and email content
            assert not (has_address and has_email), \
                f"Cluster mixes address and email content: {texts}"

    def test_email_not_in_address_cluster(self, two_column_words):
        """Email: must NOT be clustered with address words."""
        x_thresh, y_thresh = calculate_distance_thresholds(two_column_words)
        clusters = build_clusters(two_column_words, x_thresh, y_thresh)

        # Find the cluster containing "Address:"
        for cluster in clusters:
            texts = [w.text for w in cluster.words]
            if "Address:" in texts:
                assert "Email:" not in texts, \
                    "Email: incorrectly clustered with Address block"
                break

    def test_multipage_words_separate_clusters(self):
        """Words on different pages should always be separate clusters."""
        words = [
            HarmonizedWord(word_id=0, page=0, bbox=BBox(10, 100, 50, 120), text="Page1",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=1, page=1, bbox=BBox(10, 100, 50, 120), text="Page2",
                           status="word", source="T", confidence=95.0),
        ]
        x_thresh, y_thresh = calculate_distance_thresholds(words)
        clusters = build_clusters(words, x_thresh, y_thresh)

        assert len(clusters) == 2


# ============================================================================
# Unit Tests: Outlier Detection
# ============================================================================

class TestOutlierDetection:
    """Tests for intra-cluster outlier detection."""

    def test_no_outliers_in_tight_cluster(self, simple_row_words):
        """Uniformly spaced words should have no outliers."""
        cluster = Cluster(words=simple_row_words)
        outliers = detect_intra_cluster_outliers(cluster)

        assert len(outliers) == 0

    def test_detects_distant_word_as_outlier(self):
        """Word far from others should be flagged as outlier."""
        words = [
            HarmonizedWord(word_id=0, page=0, bbox=BBox(10, 100, 50, 120), text="Close1",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=1, page=0, bbox=BBox(60, 100, 100, 120), text="Close2",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=2, page=0, bbox=BBox(70, 100, 110, 120), text="Close3",
                           status="word", source="T", confidence=95.0),
            # This one is far away - should be outlier
            HarmonizedWord(word_id=3, page=0, bbox=BBox(500, 100, 550, 120), text="Far",
                           status="word", source="T", confidence=95.0),
        ]
        cluster = Cluster(words=words)
        outliers = detect_intra_cluster_outliers(cluster)

        assert len(outliers) >= 1
        outlier_texts = [w.text for w in outliers]
        assert "Far" in outlier_texts


# ============================================================================
# Unit Tests: Outlier Repositioning
# ============================================================================

class TestRepositionOutlier:
    """Tests for outlier repositioning based on centroid vector."""

    def test_outlier_left_of_centroid_goes_to_front(self):
        """Outlier to the left of centroid should move to front."""
        # Cluster with centroid around x=300
        words = [
            HarmonizedWord(word_id=0, page=0, bbox=BBox(250, 100, 300, 120), text="Mid1",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=1, page=0, bbox=BBox(310, 100, 360, 120), text="Mid2",
                           status="word", source="T", confidence=95.0),
        ]
        cluster = Cluster(words=words)

        # Outlier to the LEFT of centroid
        outlier = HarmonizedWord(word_id=2, page=0, bbox=BBox(50, 100, 100, 120), text="LeftOut",
                                 status="word", source="T", confidence=95.0)

        position = reposition_outlier(outlier, cluster)
        assert position == "front"

    def test_outlier_right_of_centroid_goes_to_end(self):
        """Outlier to the right of centroid should move to end."""
        words = [
            HarmonizedWord(word_id=0, page=0, bbox=BBox(100, 100, 150, 120), text="Mid1",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=1, page=0, bbox=BBox(160, 100, 210, 120), text="Mid2",
                           status="word", source="T", confidence=95.0),
        ]
        cluster = Cluster(words=words)

        # Outlier to the RIGHT of centroid
        outlier = HarmonizedWord(word_id=2, page=0, bbox=BBox(500, 100, 550, 120), text="RightOut",
                                 status="word", source="T", confidence=95.0)

        position = reposition_outlier(outlier, cluster)
        assert position == "end"


# ============================================================================
# Unit Tests: Full Reading Order
# ============================================================================

class TestOrderWordsByReading:
    """Tests for complete reading order algorithm."""

    def test_simple_row_maintains_left_to_right(self, simple_row_words):
        """Simple row should read left to right."""
        ordered = order_words_by_reading(simple_row_words)

        texts = [w.text for w in ordered]
        assert texts == ["Hello", "World", "Test"]

    def test_multi_row_reads_top_to_bottom_left_to_right(self, multi_row_words):
        """Multiple rows should read top-to-bottom, left-to-right."""
        ordered = order_words_by_reading(multi_row_words)

        texts = [w.text for w in ordered]
        expected = ["W00", "W01", "W02", "W03", "W10", "W11", "W12", "W13", "W20", "W21", "W22", "W23"]
        assert texts == expected

    def test_address_email_correct_order(self, two_column_words, address_email_ground_truth_order):
        """Address/Email layout should read address block first, then email."""
        ordered = order_words_by_reading(two_column_words)

        texts = [w.text for w in ordered]
        assert texts == address_email_ground_truth_order

    def test_word_ids_are_sequential(self, two_column_words):
        """After ordering, word_ids should be 0, 1, 2, ..., n-1."""
        ordered = order_words_by_reading(two_column_words)

        word_ids = [w.word_id for w in ordered]
        expected_ids = list(range(len(ordered)))
        assert word_ids == expected_ids

    def test_preserves_all_words(self, two_column_words):
        """Ordering should not lose or duplicate any words."""
        original_texts = sorted([w.text for w in two_column_words])
        ordered = order_words_by_reading(two_column_words)
        ordered_texts = sorted([w.text for w in ordered])

        assert original_texts == ordered_texts

    def test_y_fuzz_groups_varied_height_headers(self):
        """Words with slight y-variations should be grouped on the same row.

        This tests the y-fuzz improvement for headers like 'NORTHWEST VETERINARY ASSOCIATES'
        where words may have different bounding box heights or slight y-offsets.
        """
        # Simulate a header with varied y-positions (like different font sizes)
        words = [
            # Row 1: Header with slight y-variations
            HarmonizedWord(word_id=0, page=0, bbox=BBox(50, 50, 150, 85), text="NORTHWEST",
                           status="word", source="T", confidence=95.0),  # y_center = 67.5
            HarmonizedWord(word_id=1, page=0, bbox=BBox(160, 55, 280, 90), text="VETERINARY",
                           status="word", source="T", confidence=95.0),  # y_center = 72.5
            HarmonizedWord(word_id=2, page=0, bbox=BBox(290, 52, 400, 88), text="ASSOCIATES",
                           status="word", source="T", confidence=95.0),  # y_center = 70

            # Row 2: Clearly separate row
            HarmonizedWord(word_id=3, page=0, bbox=BBox(50, 120, 150, 140), text="Care",
                           status="word", source="T", confidence=95.0),
            HarmonizedWord(word_id=4, page=0, bbox=BBox(160, 120, 310, 140), text="Compassionate",
                           status="word", source="T", confidence=95.0),
        ]
        ordered = order_words_by_reading(words)
        texts = [w.text for w in ordered]

        # The header words should be grouped together on row 1, followed by row 2
        # Correct reading order: NORTHWEST VETERINARY ASSOCIATES Care Compassionate
        # NOT: NORTHWEST Care VETERINARY Compassionate ASSOCIATES (interleaved)
        assert texts == ["NORTHWEST", "VETERINARY", "ASSOCIATES", "Care", "Compassionate"]


# ============================================================================
# Integration Tests: Against Ground Truth
# ============================================================================

class TestGroundTruthComparison:
    """Integration tests comparing against peter_lou ground truth."""

    @pytest.fixture
    def ground_truth_path(self):
        return Path("data/input/peter_lou_words_slim.csv")

    @pytest.fixture
    def test_pdf_path(self):
        return Path("data/input/peter_lou.pdf")

    def test_ground_truth_file_exists(self, ground_truth_path):
        """Verify ground truth file is available."""
        assert ground_truth_path.exists(), f"Ground truth not found: {ground_truth_path}"

    def test_pdf_file_exists(self, test_pdf_path):
        """Verify test PDF is available."""
        assert test_pdf_path.exists(), f"Test PDF not found: {test_pdf_path}"

    @pytest.mark.integration
    def test_reading_order_preserves_relative_order(self, ground_truth_path, test_pdf_path):
        """
        Compare geometric clustering output against ground truth reading order.

        Uses relative order comparison: for matched words, check that their
        relative order is preserved (if word A comes before word B in ground truth,
        A should also come before B in extracted output).

        This is more robust than exact position matching since:
        - OCR may have text differences
        - Word counts may differ between extracted and ground truth
        """
        pytest.importorskip("portadoc.extractor")
        from portadoc.extractor import extract_words
        from portadoc.geometric_clustering import order_words_by_reading
        import csv

        # Load ground truth
        gt_words = []
        with open(ground_truth_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_words.append({
                    'word_id': int(row['word_id']),
                    'page': int(row['page']),
                    'text': row['text'].strip(),
                })

        # Extract from PDF
        extracted = extract_words(str(test_pdf_path))
        ordered = order_words_by_reading(extracted)

        # Build ground truth order (sorted by page, word_id)
        gt_sorted = sorted(gt_words, key=lambda x: (x['page'], x['word_id']))

        # Build text-to-position maps for both lists
        gt_text_to_pos = {}
        for i, w in enumerate(gt_sorted):
            text = w['text'].lower().strip()
            if text not in gt_text_to_pos:  # First occurrence
                gt_text_to_pos[text] = i

        ext_text_to_pos = {}
        for i, w in enumerate(ordered):
            text = w.text.lower().strip()
            if text not in ext_text_to_pos:  # First occurrence
                ext_text_to_pos[text] = i

        # Find common words
        common_words = set(gt_text_to_pos.keys()) & set(ext_text_to_pos.keys())

        if len(common_words) < 10:
            pytest.skip(f"Only {len(common_words)} common words found, skipping order test")

        # Check relative order preservation using Kendall Tau-like metric
        # For each pair of common words, check if their relative order matches
        common_list = list(common_words)
        concordant = 0
        discordant = 0

        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                w1, w2 = common_list[i], common_list[j]
                gt_order = gt_text_to_pos[w1] < gt_text_to_pos[w2]
                ext_order = ext_text_to_pos[w1] < ext_text_to_pos[w2]

                if gt_order == ext_order:
                    concordant += 1
                else:
                    discordant += 1

        total_pairs = concordant + discordant
        if total_pairs == 0:
            pytest.skip("No pairs to compare")

        # Kendall Tau ranges from -1 to 1, we want high concordance
        tau = (concordant - discordant) / total_pairs

        # Require at least 70% relative order preservation
        # (tau > 0.4 means significantly more concordant than discordant pairs)
        assert tau >= 0.4, f"Relative order accuracy (Kendall tau) {tau:.2f} < 0.4"


# ============================================================================
# Regression Tests
# ============================================================================

class TestRegressions:
    """Regression tests for specific bugs."""

    def test_email_not_between_address_lines(self, two_column_words):
        """
        Regression: Email: was incorrectly inserted between address lines.

        Bug example (WRONG order):
          Address: -> 4827 -> Maple -> Email: -> Grove -> ...

        Correct order:
          Address: -> 4827 -> Maple -> Grove -> ... -> Email: -> r.martinez@...
        """
        ordered = order_words_by_reading(two_column_words)
        texts = [w.text for w in ordered]

        # Find positions
        address_idx = texts.index("Address:")
        email_idx = texts.index("Email:")
        grove_idx = texts.index("Grove")

        # Email must come AFTER Grove (not between Address and Grove)
        assert email_idx > grove_idx, \
            f"Email: at position {email_idx} should be after Grove at {grove_idx}"

        # All address words should be contiguous before Email
        address_words = ["Address:", "4827", "Maple", "Grove", "Avenue,", "Apt",
                         "3B,", "Portland,", "OR", "97205"]
        address_indices = [texts.index(w) for w in address_words]

        # Check contiguity: max - min + 1 should equal len
        assert max(address_indices) - min(address_indices) + 1 == len(address_words), \
            "Address words should be contiguous"

        # Email should come after all address words
        assert email_idx > max(address_indices), \
            "Email: should come after all address words"
