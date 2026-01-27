"""Tests for OCR text sanitization."""

import pytest
from pathlib import Path

from portadoc.sanitize import (
    Sanitizer,
    SanitizeConfig,
    SanitizeStatus,
    SanitizeResult,
    DictionaryManager,
    load_sanitize_config,
    check_symspell,
    _levenshtein_distance,
)


class TestLevenshteinDistance:
    """Test the Levenshtein distance calculation."""

    def test_identical_strings(self):
        assert _levenshtein_distance("hello", "hello") == 0

    def test_single_insertion(self):
        assert _levenshtein_distance("hello", "helllo") == 1

    def test_single_deletion(self):
        assert _levenshtein_distance("hello", "helo") == 1

    def test_single_substitution(self):
        assert _levenshtein_distance("hello", "hallo") == 1

    def test_empty_strings(self):
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("hello", "") == 5
        assert _levenshtein_distance("", "hello") == 5

    def test_completely_different(self):
        assert _levenshtein_distance("abc", "xyz") == 3


class TestSanitizeConfig:
    """Test configuration loading."""

    def test_default_config(self):
        config = SanitizeConfig()
        assert config.enabled is True
        assert config.correct.max_edit_distance == 2
        assert config.preserve.confidence_threshold == 100.0

    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
sanitize:
  enabled: true
  correct:
    max_edit_distance: 3
    min_correction_score: 0.5
  preserve:
    confidence_threshold: 95
""")
        config = SanitizeConfig.from_yaml(config_file)
        assert config.correct.max_edit_distance == 3
        assert config.correct.min_correction_score == 0.5
        assert config.preserve.confidence_threshold == 95.0


class TestSanitizer:
    """Test the main Sanitizer class."""

    @pytest.fixture
    def sanitizer(self, tmp_path):
        """Create a sanitizer with a test dictionary."""
        # Create test dictionary
        dict_path = tmp_path / "test_dict.txt"
        dict_path.write_text("""
hello
world
testing
Gabapentin
Portland
""")
        config = SanitizeConfig()
        config.dictionaries["test"] = type("DictionarySource", (), {
            "path": str(dict_path),
            "format": "wordlist",
            "min_word_length": 1,
            "min_frequency": 0,
        })()
        config.preserve.exact_match_dictionaries = ["test"]

        sanitizer = Sanitizer(config)
        sanitizer.dictionaries.base_path = tmp_path.parent  # Adjust base path
        sanitizer.dictionaries.config.dictionaries["test"].path = str(dict_path)
        return sanitizer

    def test_skip_pixel_detector(self):
        """Test that pixel_detector engine results are skipped."""
        sanitizer = Sanitizer()
        result = sanitizer.sanitize_word("", confidence=0.0, engine="pixel_detector")
        assert result.status == SanitizeStatus.SKIPPED

    def test_skip_low_confidence_single_char(self):
        """Test that low-confidence single characters are skipped."""
        sanitizer = Sanitizer()
        result = sanitizer.sanitize_word("e", confidence=40.0)
        assert result.status == SanitizeStatus.SKIPPED

    def test_preserve_high_confidence(self):
        """Test that high-confidence words are preserved."""
        sanitizer = Sanitizer()
        result = sanitizer.sanitize_word("anything", confidence=100.0)
        assert result.status == SanitizeStatus.PRESERVED
        assert result.sanitized_text == "anything"

    def test_preserve_numeric(self):
        """Test that numeric-heavy words are preserved."""
        sanitizer = Sanitizer()

        # Microchip number
        result = sanitizer.sanitize_word("985141004729856", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED

        # Phone number
        result = sanitizer.sanitize_word("555-0123", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED

    def test_preserve_email(self):
        """Test that email addresses are preserved."""
        sanitizer = Sanitizer()
        result = sanitizer.sanitize_word("test@example.com", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED

    def test_preserve_date(self):
        """Test that date patterns are preserved."""
        sanitizer = Sanitizer()

        result = sanitizer.sanitize_word("7/24/25,", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED

        result = sanitizer.sanitize_word("01-15-2025", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED

    def test_metrics_counting(self):
        """Test that metrics are correctly counted."""
        sanitizer = Sanitizer()

        # Process a few words
        sanitizer.sanitize_word("", confidence=0.0, engine="pixel_detector")  # skip
        sanitizer.sanitize_word("e", confidence=30.0)  # skip
        sanitizer.sanitize_word("hello", confidence=100.0)  # preserve

        metrics = sanitizer.get_metrics()
        assert metrics.total_words == 3
        assert metrics.skipped == 2
        assert metrics.preserved == 1


class TestDictionaryManager:
    """Test dictionary management."""

    def test_load_wordlist_dictionary(self, tmp_path):
        """Test loading a wordlist dictionary."""
        dict_path = tmp_path / "words.txt"
        dict_path.write_text("""
# Comment line
hello
world
Python
""")
        config = SanitizeConfig()
        config.dictionaries["test"] = type("DictionarySource", (), {
            "path": str(dict_path),
            "format": "wordlist",
            "min_word_length": 1,
            "min_frequency": 0,
        })()

        manager = DictionaryManager(config, base_path=tmp_path.parent)
        manager.config.dictionaries["test"].path = str(dict_path)
        manager.load()

        assert "hello" in manager._word_sets["test"]
        assert "world" in manager._word_sets["test"]
        assert "Python" in manager._word_sets["test"]
        assert "python" in manager._word_sets["test"]  # lowercase added

    def test_exact_match(self, tmp_path):
        """Test exact dictionary matching."""
        dict_path = tmp_path / "words.txt"
        dict_path.write_text("hello\nworld\n")

        config = SanitizeConfig()
        config.dictionaries["test"] = type("DictionarySource", (), {
            "path": str(dict_path),
            "format": "wordlist",
            "min_word_length": 1,
            "min_frequency": 0,
        })()
        config.preserve.exact_match_dictionaries = ["test"]

        manager = DictionaryManager(config, base_path=tmp_path.parent)
        manager.config.dictionaries["test"].path = str(dict_path)
        manager.load()

        assert manager.exact_match("hello") == "test"
        assert manager.exact_match("Hello") == "test"  # case insensitive
        assert manager.exact_match("xyz") is None


class TestIntegration:
    """Integration tests with real project files."""

    @pytest.fixture
    def project_root(self):
        return Path(__file__).parent.parent

    def test_load_project_config(self, project_root):
        """Test loading the project's sanitize.yaml config."""
        config_path = project_root / "config" / "sanitize.yaml"
        if not config_path.exists():
            pytest.skip("config/sanitize.yaml not found")

        config = load_sanitize_config(config_path)
        assert config.enabled is True

    def test_load_custom_dictionary(self, project_root):
        """Test loading the custom dictionary."""
        dict_path = project_root / "data" / "dictionaries" / "custom.txt"
        if not dict_path.exists():
            pytest.skip("custom.txt dictionary not found")

        config = SanitizeConfig()
        config.dictionaries["custom"] = type("DictionarySource", (), {
            "path": str(dict_path),
            "format": "wordlist",
            "min_word_length": 1,
            "min_frequency": 0,
        })()

        manager = DictionaryManager(config, base_path=project_root)
        manager.config.dictionaries["custom"].path = str(dict_path)
        manager.load()

        # Check for expected words from ground truth
        assert "Gabapentin" in manager._word_sets["custom"] or \
               "gabapentin" in manager._word_sets["custom"]

    def test_symspell_availability(self):
        """Check if SymSpell is available."""
        # This is informational, not a failure
        if check_symspell():
            print("\nSymSpell: Available (fast mode)")
        else:
            print("\nSymSpell: Not installed (using slow fallback)")


class TestEdgeCases:
    """Test edge cases from ground truth data."""

    def test_medical_terms(self, tmp_path):
        """Test handling of medical terminology."""
        dict_path = tmp_path / "medical.txt"
        dict_path.write_text("""
Gabapentin
mirtazapine
Clavamox
FLUTD
""")
        config = SanitizeConfig()
        config.dictionaries["medical"] = type("DictionarySource", (), {
            "path": str(dict_path),
            "format": "wordlist",
            "min_word_length": 1,
            "min_frequency": 0,
        })()
        config.preserve.exact_match_dictionaries = ["medical"]

        sanitizer = Sanitizer(config)
        sanitizer.dictionaries.base_path = tmp_path.parent
        sanitizer.dictionaries.config.dictionaries["medical"].path = str(dict_path)
        sanitizer.load_dictionaries()

        # Exact match should preserve
        result = sanitizer.sanitize_word("Gabapentin", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED

        # Case-insensitive match
        result = sanitizer.sanitize_word("gabapentin", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED

    def test_dosing_notation(self):
        """Test that dosing notation is handled correctly."""
        sanitizer = Sanitizer()

        # These should be preserved due to numeric content
        result = sanitizer.sanitize_word("50mg", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED

        result = sanitizer.sanitize_word("q12h", confidence=80.0)
        # This might be uncertain or preserved depending on config
        assert result.status in [SanitizeStatus.PRESERVED, SanitizeStatus.UNCERTAIN]

    def test_license_numbers(self):
        """Test that license numbers are preserved."""
        sanitizer = Sanitizer()

        result = sanitizer.sanitize_word("OR-VET-8847293", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED  # High numeric ratio

    def test_punctuation_handling(self, tmp_path):
        """Test that punctuation is preserved during correction."""
        dict_path = tmp_path / "words.txt"
        dict_path.write_text("hello\n")

        config = SanitizeConfig()
        config.dictionaries["test"] = type("DictionarySource", (), {
            "path": str(dict_path),
            "format": "wordlist",
            "min_word_length": 1,
            "min_frequency": 0,
        })()
        config.preserve.exact_match_dictionaries = ["test"]

        sanitizer = Sanitizer(config)
        sanitizer.dictionaries.base_path = tmp_path.parent
        sanitizer.dictionaries.config.dictionaries["test"].path = str(dict_path)
        sanitizer.load_dictionaries()

        # Word with punctuation should still match
        result = sanitizer.sanitize_word("hello,", confidence=80.0)
        assert result.status == SanitizeStatus.PRESERVED
        assert result.sanitized_text == "hello,"
