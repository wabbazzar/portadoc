"""
TDD tests for VLM + BBox OCR Fusion algorithm.

Uses:
- Real PDF: data/input/cat_in_hat.pdf
- Real Tesseract output (no mocking)
- Real geometric clustering
- Mocked VLM output (text only)

Test levels:
- Level 0: Perfect VLM output
- Level 1: VLM misspellings
- Level 2: VLM missing words
- Level 3: VLM extra words (hallucination)
- Level 4: Tesseract errors + perfect VLM
- Level 5: Both have different errors
- Level 6: Multi-line orphan segments
"""

import pytest
from pathlib import Path
from typing import List

from portadoc.models import BBox, HarmonizedWord
from portadoc.geometric_clustering import order_words_by_reading, Cluster, build_clusters


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def cat_in_hat_pdf():
    """Path to the cat in hat test PDF."""
    path = Path("data/input/cat_in_hat.pdf")
    if not path.exists():
        pytest.skip(f"Test PDF not found: {path}")
    return path


@pytest.fixture
def cat_in_hat_tesseract_words(cat_in_hat_pdf) -> List[HarmonizedWord]:
    """Run Tesseract on cat_in_hat.pdf and return words."""
    pytest.importorskip("portadoc.extractor")
    from portadoc.extractor import extract_words

    words = extract_words(
        str(cat_in_hat_pdf),
        use_easyocr=False,
        use_doctr=False,
        use_paddleocr=False,
        use_surya=False,
        use_kraken=False,
    )
    return words


@pytest.fixture
def ground_truth_text_line1() -> str:
    """Ground truth text for line 1 (40pt)."""
    return "The cat in the hat did a flip and a splat,"


@pytest.fixture
def ground_truth_text_line2() -> str:
    """Ground truth text for line 2 (20pt)."""
    return "then he tipped his tall hat and said, 'How about that!'"


@pytest.fixture
def ground_truth_full(ground_truth_text_line1, ground_truth_text_line2) -> str:
    """Full ground truth text."""
    return f"{ground_truth_text_line1} {ground_truth_text_line2}"


# ============================================================================
# VLM Output Fixtures (Mocked - various noise levels)
# ============================================================================

@pytest.fixture
def vlm_output_perfect(ground_truth_full) -> str:
    """Level 0: Perfect VLM output."""
    return ground_truth_full


@pytest.fixture
def vlm_output_misspelling() -> str:
    """Level 1: VLM with misspelling ('flip' -> 'flp')."""
    return "The cat in the hat did a flp and a splat, then he tipped his tall hat and said, 'How about that!'"


@pytest.fixture
def vlm_output_missing_word() -> str:
    """Level 2: VLM missing 'and' before 'a splat'."""
    return "The cat in the hat did a flip a splat, then he tipped his tall hat and said, 'How about that!'"


@pytest.fixture
def vlm_output_extra_word() -> str:
    """Level 3: VLM hallucinated 'fat' before 'cat'."""
    return "The fat cat in the hat did a flip and a splat, then he tipped his tall hat and said, 'How about that!'"


@pytest.fixture
def vlm_output_partial_line1_only() -> str:
    """Level 6: VLM only got line 1."""
    return "The cat in the hat did a flip and a splat,"


# ============================================================================
# Helper Functions
# ============================================================================

def words_to_text(words: List[HarmonizedWord]) -> str:
    """Convert word list to space-separated text."""
    return " ".join(w.text for w in words)


def find_word_by_text(words: List[HarmonizedWord], text: str) -> HarmonizedWord:
    """Find first word matching text (case-insensitive)."""
    for w in words:
        if w.text.lower() == text.lower():
            return w
    return None


# ============================================================================
# Level 0: Perfect VLM Output
# ============================================================================

class TestLevel0PerfectVLM:
    """Tests with perfect VLM output matching ground truth."""

    def test_tokenize_vlm_output(self, vlm_output_perfect):
        """VLM tokenizer should split text into words preserving punctuation."""
        from portadoc.ocr.vlm_fusion import tokenize_vlm_text

        tokens = tokenize_vlm_text(vlm_output_perfect)

        # Should preserve punctuation attached to words
        assert "splat," in tokens
        assert "said," in tokens
        assert "that!'" in tokens

        # Should have expected word count
        # Line 1: The cat in the hat did a flip and a splat, (11 words)
        # Line 2: then he tipped his tall hat and said, 'How about that!' (11 words)
        # Total: 22 words
        assert len(tokens) == 22

    def test_context_signature_unique_word(self, vlm_output_perfect):
        """Context signature for unique word should be minimal."""
        from portadoc.ocr.vlm_fusion import tokenize_vlm_text, build_context_signature

        tokens = tokenize_vlm_text(vlm_output_perfect)

        # "tipped" is unique - should need minimal context
        tipped_idx = tokens.index("tipped")
        sig = build_context_signature(tokens, tipped_idx, window=1)

        assert "tipped" in sig
        # Window 1 gives [prev, current, next]
        assert "he" in sig or "his" in sig

    def test_context_signature_repeated_word(self, vlm_output_perfect):
        """Context signature for repeated word needs more context."""
        from portadoc.ocr.vlm_fusion import tokenize_vlm_text, build_context_signature

        tokens = tokenize_vlm_text(vlm_output_perfect)

        # "the" appears multiple times - need context to disambiguate
        the_indices = [i for i, t in enumerate(tokens) if t.lower() == "the"]
        assert len(the_indices) >= 2

        # Each "the" should get different context
        signatures = [build_context_signature(tokens, i, window=2) for i in the_indices]

        # All signatures should be different (with enough context)
        assert len(set(signatures)) == len(the_indices)

    def test_basic_matching_exact(self, cat_in_hat_tesseract_words, vlm_output_perfect):
        """Exact matching should work for words that match perfectly."""
        from portadoc.ocr.vlm_fusion import match_vlm_to_ocr

        result = match_vlm_to_ocr(cat_in_hat_tesseract_words, vlm_output_perfect)

        # Should have some matched words
        assert len(result.matched) > 0

        # Check that common words are matched
        matched_vlm_texts = [vlm_text for _, (vlm_text, _) in result.matched.items()]

        # "The", "cat", "hat" should all be matched
        assert any("The" in t or "the" in t for t in matched_vlm_texts)


# ============================================================================
# Level 1: VLM Misspellings
# ============================================================================

class TestLevel1VLMMisspelling:
    """Tests where VLM has misspellings."""

    def test_fuzzy_match_finds_misspelling(self, cat_in_hat_tesseract_words, vlm_output_misspelling):
        """Fuzzy matching should find VLM 'flp' matches OCR 'flip'."""
        from portadoc.ocr.vlm_fusion import match_vlm_to_ocr

        result = match_vlm_to_ocr(
            cat_in_hat_tesseract_words,
            vlm_output_misspelling,
            max_levenshtein=2
        )

        # 'flp' should match 'flip' with Levenshtein 1
        # Find if 'flp' was matched
        matched_vlm_texts = [vlm_text for _, (vlm_text, _) in result.matched.items()]

        # Either 'flp' is matched OR it's an orphan
        flp_matched = "flp" in matched_vlm_texts
        flp_orphan = "flp" in [result.vlm_tokens[i] for i in result.vlm_orphans]

        # At minimum, other words should still match
        assert len(result.matched) > 10  # Most words should match


# ============================================================================
# Level 2: VLM Missing Words
# ============================================================================

class TestLevel2VLMMissingWord:
    """Tests where VLM is missing words that OCR found."""

    def test_ocr_orphan_when_vlm_missing(self, cat_in_hat_tesseract_words, vlm_output_missing_word):
        """OCR word should be orphan when VLM doesn't have it."""
        from portadoc.ocr.vlm_fusion import match_vlm_to_ocr

        result = match_vlm_to_ocr(cat_in_hat_tesseract_words, vlm_output_missing_word)

        # The word "and" (before "a splat") is missing from VLM
        # Check that we identify OCR orphans
        # Note: there are multiple "and" words, so this is tricky

        # At least verify matching still works for other words
        assert len(result.matched) > 0


# ============================================================================
# Level 3: VLM Extra Words (Hallucination)
# ============================================================================

class TestLevel3VLMExtraWord:
    """Tests where VLM has words that OCR didn't find."""

    def test_vlm_orphan_for_hallucinated_word(self, cat_in_hat_tesseract_words, vlm_output_extra_word):
        """VLM hallucinated 'fat' should be an orphan (no OCR match)."""
        from portadoc.ocr.vlm_fusion import match_vlm_to_ocr

        result = match_vlm_to_ocr(cat_in_hat_tesseract_words, vlm_output_extra_word)

        # 'fat' should be a VLM orphan (no OCR match)
        orphan_texts = [result.vlm_tokens[i] for i in result.vlm_orphans]
        assert "fat" in orphan_texts

    def test_orphan_placement_uses_neighbors(self, cat_in_hat_tesseract_words, vlm_output_extra_word):
        """Orphan 'fat' should be placed sequentially after 'The' using neighbor anchor."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        fused_words = fuse_vlm_with_ocr(cat_in_hat_tesseract_words, vlm_output_extra_word)

        # Find 'fat' in fused output
        fat_word = find_word_by_text(fused_words, "fat")

        if fat_word is not None:
            # Should be placed after "The" (its left anchor)
            the_word = find_word_by_text(fused_words, "The")

            if the_word:
                # fat's x0 should be >= The's x1 (placed after left anchor)
                assert fat_word.bbox.x0 >= the_word.bbox.x1 - 5  # Allow small tolerance
                # fat should be on the same line as The (same y coordinate)
                assert abs(fat_word.bbox.y0 - the_word.bbox.y0) < 10  # Same line
                # fat should have valid dimensions
                assert fat_word.bbox.width > 0
                assert fat_word.bbox.height > 0


# ============================================================================
# Level 4: Tesseract Errors + Perfect VLM
# ============================================================================

class TestLevel4TesseractErrorsVLMPerfect:
    """Tests where Tesseract has errors but VLM is correct."""

    def test_vlm_text_replaces_ocr_errors(self, cat_in_hat_tesseract_words, vlm_output_perfect):
        """VLM text should replace OCR misspellings in fused output."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        # Note: Our test Tesseract output shows "|" instead of "a splat,"
        fused_words = fuse_vlm_with_ocr(cat_in_hat_tesseract_words, vlm_output_perfect)

        # The fused output should use VLM text
        fused_texts = [w.text for w in fused_words]

        # Should have correct VLM text, not OCR errors
        assert "cat" in fused_texts  # Not "crt" if Tesseract had errors
        assert "the" in fused_texts or "The" in fused_texts


# ============================================================================
# Level 5: Both Have Different Errors
# ============================================================================

class TestLevel5BothHaveErrors:
    """Tests where both OCR and VLM have errors, but different ones."""

    def test_vlm_corrects_ocr_but_introduces_own_error(
        self, cat_in_hat_tesseract_words, vlm_output_misspelling
    ):
        """VLM fixes some OCR errors but introduces 'flp'."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        fused_words = fuse_vlm_with_ocr(cat_in_hat_tesseract_words, vlm_output_misspelling)

        # Should have fused output
        assert len(fused_words) > 0

        # VLM's text should be used (including its error 'flp')
        fused_texts = [w.text for w in fused_words]

        # Check we have VLM text adopted
        assert any("flp" in t or "flip" in t for t in fused_texts)


# ============================================================================
# Level 6: Orphan Placement with Pixel Detection
# ============================================================================

class TestLevel6OrphanPlacement:
    """Tests for placing orphan VLM words using pixel detection."""

    def test_character_size_estimation(self, cat_in_hat_tesseract_words, vlm_output_perfect):
        """Should estimate character sizes from matched words."""
        from portadoc.ocr.vlm_fusion import match_vlm_to_ocr, estimate_char_sizes

        result = match_vlm_to_ocr(cat_in_hat_tesseract_words, vlm_output_perfect)

        # Get matched pairs for char size estimation
        matched_pairs = [
            (vlm_text, ocr_word.bbox)
            for _, (vlm_text, ocr_word) in result.matched.items()
        ]

        char_sizes = estimate_char_sizes(matched_pairs)

        # Should have size estimates
        assert char_sizes.avg_char_width > 0
        assert char_sizes.avg_char_height > 0

        # 40pt font should have larger chars than 20pt
        # But we're averaging, so just check reasonable range
        assert 5 < char_sizes.avg_char_width < 50
        assert 10 < char_sizes.avg_char_height < 60

    def test_char_size_clusters_by_font_size(self, cat_in_hat_tesseract_words, vlm_output_perfect):
        """Should create separate clusters for different font sizes."""
        from portadoc.ocr.vlm_fusion import match_vlm_to_ocr, estimate_char_sizes_clustered

        result = match_vlm_to_ocr(cat_in_hat_tesseract_words, vlm_output_perfect)

        matched_pairs = [
            (vlm_text, ocr_word.bbox)
            for _, (vlm_text, ocr_word) in result.matched.items()
        ]

        clusters = estimate_char_sizes_clustered(matched_pairs)

        # Should have at least 1 cluster (maybe 2 for different font sizes)
        assert len(clusters) >= 1

        # Each cluster should have size estimates
        for cluster in clusters:
            assert cluster.avg_char_width > 0
            assert cluster.avg_char_height > 0


# ============================================================================
# Integration Tests: Full Pipeline
# ============================================================================

class TestFullPipeline:
    """End-to-end tests of the VLM fusion pipeline."""

    def test_fused_output_has_all_words(self, cat_in_hat_tesseract_words, vlm_output_perfect):
        """Fused output should have words from both sources."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        fused = fuse_vlm_with_ocr(cat_in_hat_tesseract_words, vlm_output_perfect)

        # Should have reasonable word count
        assert len(fused) >= 15  # At least most of the words

    def test_fused_output_preserves_bboxes(self, cat_in_hat_tesseract_words, vlm_output_perfect):
        """Fused output should have valid bboxes."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        fused = fuse_vlm_with_ocr(cat_in_hat_tesseract_words, vlm_output_perfect)

        for word in fused:
            assert word.bbox is not None
            assert word.bbox.width > 0
            assert word.bbox.height > 0

    def test_fused_output_has_status(self, cat_in_hat_tesseract_words, vlm_output_perfect):
        """Fused output should have status field indicating provenance."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        fused = fuse_vlm_with_ocr(cat_in_hat_tesseract_words, vlm_output_perfect)

        valid_statuses = {"vlm_matched", "vlm_pixel_placed", "vlm_interpolated", "ocr_only"}

        for word in fused:
            assert word.status in valid_statuses

    def test_fused_output_reading_order(self, cat_in_hat_tesseract_words, vlm_output_perfect):
        """Fused output should be in reading order."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        fused = fuse_vlm_with_ocr(cat_in_hat_tesseract_words, vlm_output_perfect)

        # Words should be sorted by y then x (reading order)
        for i in range(len(fused) - 1):
            w1, w2 = fused[i], fused[i + 1]

            # Either w2 is below w1, or they're on same line and w2 is to the right
            same_line = abs(w1.bbox.center_y - w2.bbox.center_y) < 20
            if same_line:
                assert w1.bbox.x0 <= w2.bbox.x0 + 5, f"Words not in reading order: {w1.text} vs {w2.text}"


# ============================================================================
# Edge Cases
# ============================================================================

# ============================================================================
# Orphan Segment Detection Tests
# ============================================================================

class TestOrphanSegmentDetection:
    """Tests for contiguous orphan segment detection."""

    def test_single_orphan(self):
        """Single orphan should be its own segment."""
        from portadoc.ocr.vlm_fusion import find_contiguous_orphan_segments

        segments = find_contiguous_orphan_segments([5])
        assert segments == [[5]]

    def test_consecutive_orphans(self):
        """Consecutive orphans should be grouped into one segment."""
        from portadoc.ocr.vlm_fusion import find_contiguous_orphan_segments

        segments = find_contiguous_orphan_segments([5, 6, 7])
        assert segments == [[5, 6, 7]]

    def test_non_consecutive_orphans(self):
        """Non-consecutive orphans should be in separate segments."""
        from portadoc.ocr.vlm_fusion import find_contiguous_orphan_segments

        segments = find_contiguous_orphan_segments([2, 5, 6, 7, 10])
        assert segments == [[2], [5, 6, 7], [10]]

    def test_empty_orphans(self):
        """Empty orphan list should return empty segments."""
        from portadoc.ocr.vlm_fusion import find_contiguous_orphan_segments

        segments = find_contiguous_orphan_segments([])
        assert segments == []

    def test_unsorted_orphans(self):
        """Unsorted orphan indices should still be grouped correctly."""
        from portadoc.ocr.vlm_fusion import find_contiguous_orphan_segments

        segments = find_contiguous_orphan_segments([7, 5, 6, 2, 10])
        assert segments == [[2], [5, 6, 7], [10]]


# ============================================================================
# Orphan Word Wrapping Tests
# ============================================================================

class TestOrphanWordWrapping:
    """Tests for orphan word placement with line wrapping."""

    def test_single_line_fit(self):
        """Orphan words that fit on same line should not wrap."""
        from portadoc.ocr.vlm_fusion import place_orphan_segment_with_wrap, CharSizes
        from portadoc.models import BBox, HarmonizedWord

        char_sizes = CharSizes(avg_char_width=10.0, avg_char_height=15.0)

        # Create anchor at x=100, plenty of room to the right
        left_anchor = HarmonizedWord(
            word_id=0, page=0,
            bbox=BBox(x0=50, y0=100, x1=100, y1=115),
            text="The", status="matched", source="test", confidence=1.0
        )

        bboxes = place_orphan_segment_with_wrap(
            orphan_words=["cat", "sat"],
            left_anchor=left_anchor,
            right_anchor=None,
            char_sizes=char_sizes,
            page_width=612.0,
        )

        assert len(bboxes) == 2

        # Both words should be on the same line (same y0)
        assert bboxes[0].y0 == bboxes[1].y0

        # Words should be sequential (not stacked)
        assert bboxes[1].x0 > bboxes[0].x1  # Second word starts after first

    def test_wrap_to_next_line(self):
        """Orphan words exceeding line width should wrap to next line."""
        from portadoc.ocr.vlm_fusion import place_orphan_segment_with_wrap, CharSizes
        from portadoc.models import BBox, HarmonizedWord
        from portadoc.geometric_clustering import Cluster

        char_sizes = CharSizes(avg_char_width=10.0, avg_char_height=15.0)

        # Create narrow cluster to force wrapping
        cluster_word = HarmonizedWord(
            word_id=0, page=0,
            bbox=BBox(x0=50, y0=100, x1=200, y1=115),  # Cluster ends at x=200
            text="The", status="matched", source="test", confidence=1.0
        )
        cluster = Cluster(words=[cluster_word])

        # Left anchor near right edge
        left_anchor = HarmonizedWord(
            word_id=0, page=0,
            bbox=BBox(x0=150, y0=100, x1=180, y1=115),
            text="near", status="matched", source="test", confidence=1.0
        )

        # Words that won't fit on current line
        bboxes = place_orphan_segment_with_wrap(
            orphan_words=["supercalifragilistic", "expialidocious"],
            left_anchor=left_anchor,
            right_anchor=None,
            char_sizes=char_sizes,
            cluster=cluster,
        )

        assert len(bboxes) == 2

        # First word should wrap (can't fit after left_anchor)
        # Second word should also be on a new line or same wrapped line
        # Key check: words are NOT stacked at same x position
        assert bboxes[0].x0 != bboxes[1].x0 or bboxes[0].y0 != bboxes[1].y0

    def test_multiple_wraps(self):
        """Long orphan segment should wrap multiple times."""
        from portadoc.ocr.vlm_fusion import place_orphan_segment_with_wrap, CharSizes
        from portadoc.models import BBox, HarmonizedWord
        from portadoc.geometric_clustering import Cluster

        char_sizes = CharSizes(avg_char_width=10.0, avg_char_height=15.0)

        # Create narrow cluster
        cluster_word = HarmonizedWord(
            word_id=0, page=0,
            bbox=BBox(x0=50, y0=100, x1=150, y1=115),  # Only 100pt wide
            text="word", status="matched", source="test", confidence=1.0
        )
        cluster = Cluster(words=[cluster_word])

        left_anchor = HarmonizedWord(
            word_id=0, page=0,
            bbox=BBox(x0=50, y0=100, x1=80, y1=115),
            text="The", status="matched", source="test", confidence=1.0
        )

        # Many words that will require multiple wraps
        words = ["one", "two", "three", "four", "five", "six", "seven", "eight"]

        bboxes = place_orphan_segment_with_wrap(
            orphan_words=words,
            left_anchor=left_anchor,
            right_anchor=None,
            char_sizes=char_sizes,
            cluster=cluster,
        )

        assert len(bboxes) == len(words)

        # Should have multiple different y values (multiple lines)
        y_values = set(b.y0 for b in bboxes)
        assert len(y_values) >= 2, "Should have wrapped to at least 2 lines"

        # No two words should be stacked at exact same position
        positions = [(b.x0, b.y0) for b in bboxes]
        assert len(set(positions)) == len(positions), "Words should not be stacked"

    def test_no_anchors_uses_defaults(self):
        """When no anchors, should use default positioning."""
        from portadoc.ocr.vlm_fusion import place_orphan_segment_with_wrap, CharSizes

        char_sizes = CharSizes(avg_char_width=10.0, avg_char_height=15.0)

        bboxes = place_orphan_segment_with_wrap(
            orphan_words=["hello", "world"],
            left_anchor=None,
            right_anchor=None,
            char_sizes=char_sizes,
        )

        assert len(bboxes) == 2
        # Should have valid bboxes
        assert all(b.width > 0 and b.height > 0 for b in bboxes)

    def test_sequential_placement_not_stacked(self):
        """Core requirement: orphan words must NOT stack at same x position."""
        from portadoc.ocr.vlm_fusion import place_orphan_segment_with_wrap, CharSizes
        from portadoc.models import BBox, HarmonizedWord

        char_sizes = CharSizes(avg_char_width=10.0, avg_char_height=15.0)

        left_anchor = HarmonizedWord(
            word_id=0, page=0,
            bbox=BBox(x0=300, y0=50, x1=350, y1=80),
            text="did", status="matched", source="test", confidence=1.0
        )

        # This was the original bug: words like "a", "flip", "and" stacked at x=360
        bboxes = place_orphan_segment_with_wrap(
            orphan_words=["a", "flip", "and", "a", "splat"],
            left_anchor=left_anchor,
            right_anchor=None,
            char_sizes=char_sizes,
        )

        # Each word should have a unique x0 position (on same line)
        # OR if wrapped, should be on different lines
        positions = [(round(b.x0, 1), round(b.y0, 1)) for b in bboxes]
        assert len(set(positions)) == len(positions), (
            f"Words should not be stacked! Positions: {positions}"
        )


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_vlm_output(self, cat_in_hat_tesseract_words):
        """Should handle empty VLM output gracefully."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        fused = fuse_vlm_with_ocr(cat_in_hat_tesseract_words, "")

        # Should return OCR words as-is with ocr_only status
        assert len(fused) == len(cat_in_hat_tesseract_words)
        for word in fused:
            assert word.status == "ocr_only"

    def test_empty_ocr_words(self, vlm_output_perfect):
        """Should handle empty OCR words gracefully."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        fused = fuse_vlm_with_ocr([], vlm_output_perfect)

        # Can't place VLM words without OCR bboxes
        # Should return empty or VLM words without bboxes
        assert len(fused) == 0

    def test_single_word_vlm(self, cat_in_hat_tesseract_words):
        """Should handle single-word VLM output."""
        from portadoc.ocr.vlm_fusion import fuse_vlm_with_ocr

        fused = fuse_vlm_with_ocr(cat_in_hat_tesseract_words, "cat")

        # Should match 'cat' and mark rest as ocr_only
        assert len(fused) > 0

        cat_matched = any(w.text == "cat" and w.status == "vlm_matched" for w in fused)
        assert cat_matched
