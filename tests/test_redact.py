"""Tests for entity detection and redaction marking."""

import csv
import tempfile
from pathlib import Path

import pytest

from portadoc.redact import (
    EntityType,
    EntityDetector,
    load_names,
    strip_punctuation,
    redact_csv,
    get_redaction_stats,
)
from portadoc.patterns import (
    matches_date,
    matches_code,
    matches_exclusion,
)


class TestPatterns:
    """Test regex pattern matching."""

    def test_matches_date_us_format(self):
        assert matches_date("7/24/25")
        assert matches_date("12/31/2024")
        assert matches_date("1-15-24")
        assert matches_date("7/24/25,")  # trailing punctuation

    def test_matches_date_iso_format(self):
        assert matches_date("2024-12-31")
        assert matches_date("2025-01-15")

    def test_matches_date_negative(self):
        assert not matches_date("hello")
        assert not matches_date("123")
        assert not matches_date("12:30")  # time

    def test_matches_code_phone(self):
        assert matches_code("555-123-4567")
        assert matches_code("(555) 123-4567")
        assert matches_code("555.123.4567")

    def test_matches_code_ssn(self):
        assert matches_code("123-45-6789")

    def test_matches_code_generic(self):
        assert matches_code("12345")
        assert matches_code("123456789")
        assert matches_code("7829341")  # 7 digit MRN

    def test_matches_code_alphanumeric(self):
        assert matches_code("ABC12345")
        assert matches_code("A1234567")
        assert matches_code("INK12345")

    def test_matches_code_negative(self):
        assert not matches_code("hello")
        assert not matches_code("123")  # too short
        assert not matches_code("$100")

    def test_matches_exclusion_time(self):
        assert matches_exclusion("10:28")
        assert matches_exclusion("2:30")

    def test_matches_exclusion_currency(self):
        assert matches_exclusion("$100")
        assert matches_exclusion("$1,000.00")

    def test_matches_exclusion_percent(self):
        assert matches_exclusion("50%")
        assert matches_exclusion("3.5%")

    def test_matches_exclusion_abbreviations(self):
        assert matches_exclusion("AM")
        assert matches_exclusion("PM")
        assert matches_exclusion("ID")


class TestStripPunctuation:
    """Test punctuation stripping."""

    def test_trailing_comma(self):
        assert strip_punctuation("Peter,") == "Peter"

    def test_trailing_period(self):
        assert strip_punctuation("Lou.") == "Lou"

    def test_leading_punctuation(self):
        assert strip_punctuation("(Peter") == "Peter"

    def test_both_ends(self):
        assert strip_punctuation("\"Lou\"") == "Lou"

    def test_no_punctuation(self):
        assert strip_punctuation("Peter") == "Peter"

    def test_inner_punctuation_preserved(self):
        assert strip_punctuation("O'Brien") == "O'Brien"


class TestLoadNames:
    """Test names dictionary loading."""

    def test_load_default_names(self):
        names = load_names()
        assert len(names) > 0
        # Use names actually in the us_names.txt (female names)
        assert "abigail" in names
        assert "anna" in names
        assert "zoe" in names

    def test_load_nonexistent_file(self):
        names = load_names(Path("/nonexistent/file.txt"))
        assert len(names) == 0


class TestEntityDetector:
    """Test the EntityDetector class."""

    @pytest.fixture
    def detector(self):
        return EntityDetector()

    def test_detect_date(self, detector):
        entity, redact = detector.detect("7/24/25")
        assert entity == EntityType.DATE
        assert redact is True

    def test_detect_code(self, detector):
        entity, redact = detector.detect("7829341")
        assert entity == EntityType.CODE
        assert redact is True

    def test_detect_name(self, detector):
        # Use names that are in the us_names.txt dictionary
        entity, redact = detector.detect("Abigail")
        assert entity == EntityType.NAME
        assert redact is True

    def test_detect_name_with_punctuation(self, detector):
        entity, redact = detector.detect("Anna,")
        assert entity == EntityType.NAME
        assert redact is True

    def test_detect_name_case_insensitive(self, detector):
        entity, redact = detector.detect("ABIGAIL")
        assert entity == EntityType.NAME
        assert redact is True

    def test_detect_exclusion_time(self, detector):
        entity, redact = detector.detect("10:28")
        assert entity == EntityType.NONE
        assert redact is False

    def test_detect_normal_word(self, detector):
        entity, redact = detector.detect("Patient")
        # "Patient" is not in names dictionary
        entity, redact = detector.detect("Veterinary")
        assert entity == EntityType.NONE
        assert redact is False

    def test_detect_empty_string(self, detector):
        entity, redact = detector.detect("")
        assert entity == EntityType.NONE
        assert redact is False

    def test_detect_batch(self, detector):
        texts = ["Abigail", "7/24/25", "hello", "7829341"]
        results = detector.detect_batch(texts)
        assert len(results) == 4
        assert results[0] == (EntityType.NAME, True)
        assert results[1] == (EntityType.DATE, True)
        assert results[2] == (EntityType.NONE, False)
        assert results[3] == (EntityType.CODE, True)

    def test_disable_names(self):
        detector = EntityDetector(detect_names=False)
        entity, redact = detector.detect("Peter")
        assert entity == EntityType.NONE
        assert redact is False

    def test_disable_dates(self):
        detector = EntityDetector(detect_dates=False)
        entity, redact = detector.detect("7/24/25")
        assert entity == EntityType.NONE
        assert redact is False

    def test_disable_codes(self):
        detector = EntityDetector(detect_codes=False)
        entity, redact = detector.detect("7829341")
        assert entity == EntityType.NONE
        assert redact is False


class TestRedactCsv:
    """Test CSV redaction marking."""

    def test_redact_csv(self):
        # Create a test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["word_id", "page", "text", "x0", "y0", "x1", "y1"])
            writer.writerow([0, 0, "7/24/25", 0, 0, 10, 10])
            writer.writerow([1, 0, "Abigail", 0, 10, 10, 20])  # Name in dictionary
            writer.writerow([2, 0, "Intake", 0, 20, 10, 30])
            writer.writerow([3, 0, "7829341", 0, 30, 10, 40])
            input_path = Path(f.name)

        try:
            # Run redaction
            count = redact_csv(input_path)
            assert count == 3  # date, name, code

            # Read and verify output
            with open(input_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert rows[0]["entity"] == "DATE"
            assert rows[0]["redact"] == "true"

            assert rows[1]["entity"] == "NAME"
            assert rows[1]["redact"] == "true"

            assert rows[2]["entity"] == ""
            assert rows[2]["redact"] == "false"

            assert rows[3]["entity"] == "CODE"
            assert rows[3]["redact"] == "true"

        finally:
            input_path.unlink()

    def test_redact_csv_output_file(self):
        # Create input CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["word_id", "page", "text", "x0", "y0", "x1", "y1"])
            writer.writerow([0, 0, "Abigail", 0, 0, 10, 10])  # Name in dictionary
            input_path = Path(f.name)

        # Create output path
        output_path = Path(tempfile.gettempdir()) / "output.csv"

        try:
            count = redact_csv(input_path, output_csv=output_path)
            assert count == 1
            assert output_path.exists()

            # Verify original is unchanged
            with open(input_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert "entity" not in rows[0]

            # Verify output has new columns
            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert rows[0]["entity"] == "NAME"

        finally:
            input_path.unlink()
            if output_path.exists():
                output_path.unlink()


class TestGetRedactionStats:
    """Test redaction statistics."""

    def test_get_stats(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["word_id", "text", "entity", "redact"])
            writer.writerow([0, "Peter", "NAME", "true"])
            writer.writerow([1, "7/24/25", "DATE", "true"])
            writer.writerow([2, "hello", "", "false"])
            writer.writerow([3, "7829341", "CODE", "true"])
            path = Path(f.name)

        try:
            stats = get_redaction_stats(path)
            assert stats["total_words"] == 4
            assert stats["redacted_count"] == 3
            assert stats["by_type"]["NAME"] == 1
            assert stats["by_type"]["DATE"] == 1
            assert stats["by_type"]["CODE"] == 1

        finally:
            path.unlink()
