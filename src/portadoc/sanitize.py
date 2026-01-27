"""
OCR Text Sanitization using dictionary-based Levenshtein distance matching.

Uses SymSpell for O(1) fuzzy matching with configurable thresholds.
See specs/sanitize.md for full documentation.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

# Try to import symspellpy, fall back to basic implementation
try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False
    SymSpell = None
    Verbosity = None

# Import ranking module
from portadoc.ranking import (
    FrequencyRanker,
    DocumentRanker,
    BigramRanker,
    OCRErrorModel,
    MultiSignalRanker,
    FrequencyConfig,
    DocumentConfig,
    BigramConfig,
    OCRModelConfig,
)


class SanitizeStatus(Enum):
    """Status of a sanitized word."""
    SKIPPED = "skipped"           # Phase 0: pixel detector, artifacts
    PRESERVED = "preserved"       # Phase 1: high confidence, exact match
    CORRECTED = "corrected"       # Phase 2: single-word correction
    CONTEXT_CORRECTED = "context" # Phase 3: context-based correction
    UNCERTAIN = "uncertain"       # Phase 4: no confident correction


@dataclass
class SanitizeResult:
    """Result of sanitizing a single word."""
    original_text: str
    sanitized_text: str
    status: SanitizeStatus
    confidence: float             # Original OCR confidence
    edit_distance: int            # 0 if preserved/skipped
    correction_score: float       # Score that led to correction
    matched_dictionary: Optional[str] = None  # Which dictionary matched
    frequency_factor: float = 1.0  # Ranking: word frequency signal
    document_factor: float = 1.0   # Ranking: document context signal
    bigram_factor: float = 1.0     # Ranking: bigram context signal
    ocr_factor: float = 1.0        # Ranking: OCR error model signal


@dataclass
class SanitizeMetrics:
    """Aggregate metrics for sanitization."""
    total_words: int = 0
    skipped: int = 0
    preserved: int = 0
    corrected: int = 0
    context_corrected: int = 0
    uncertain: int = 0

    # Evaluation metrics (set when comparing to ground truth)
    true_positives: int = 0       # Correct corrections
    false_positives: int = 0      # Incorrect corrections (over-correction)
    false_negatives: int = 0      # Missed corrections (under-correction)
    true_negatives: int = 0       # Correctly preserved

    @property
    def precision(self) -> float:
        """Correct corrections / Total corrections."""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Correct corrections / Total errors in input."""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """(Preserved correct + Corrected to correct) / Total."""
        correct = self.true_positives + self.true_negatives
        total = self.total_words
        return correct / total if total > 0 else 0.0

    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"Total words: {self.total_words}",
            f"  Skipped: {self.skipped}",
            f"  Preserved: {self.preserved}",
            f"  Corrected: {self.corrected}",
            f"  Context corrected: {self.context_corrected}",
            f"  Uncertain: {self.uncertain}",
        ]
        if self.true_positives + self.false_positives > 0:
            lines.extend([
                "",
                f"Precision: {self.precision:.2%}",
                f"Recall: {self.recall:.2%}",
                f"F1 Score: {self.f1_score:.2%}",
                f"Accuracy: {self.accuracy:.2%}",
            ])
        return "\n".join(lines)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SkipConfig:
    """Phase 0: Skip rules configuration."""
    pixel_detector: bool = True
    single_char_max_conf: float = 50.0


@dataclass
class PreserveConfig:
    """Phase 1: Preserve rules configuration."""
    confidence_threshold: float = 100.0
    numeric_ratio: float = 0.5
    exact_match_dictionaries: list[str] = field(
        default_factory=lambda: ["english", "names", "medical", "custom"]
    )


@dataclass
class CorrectConfig:
    """Phase 2: Single-word correction configuration."""
    algorithm: str = "symspell"
    max_edit_distance: int = 2
    min_correction_score: float = 0.7
    case_sensitive: bool = False
    dictionary_weights: dict[str, float] = field(
        default_factory=lambda: {
            "english": 1.0,
            "names": 0.9,
            "medical": 0.8,
            "custom": 1.0,
        }
    )


@dataclass
class ContextConfig:
    """Phase 3: Context correction configuration."""
    enabled: bool = True
    ngram_window: int = 2
    min_context_score: float = 0.6
    max_edit_distance: int = 3


@dataclass
class FlagConfig:
    """Phase 4: Flagging configuration."""
    mark_uncertain: bool = True
    preserve_original: bool = True


@dataclass
class DictionarySource:
    """Configuration for a dictionary source."""
    path: str
    format: str = "wordlist"  # wordlist | frequency
    min_word_length: int = 1
    min_frequency: int = 0


@dataclass
class RankingSignalConfig:
    """Configuration for a ranking signal."""
    enabled: bool = True
    weight: float = 1.0


@dataclass
class FrequencyRankingConfig(RankingSignalConfig):
    """Frequency ranking configuration."""
    source: str = "data/frequencies/english_freq.txt"
    fallback_frequency: int = 1


@dataclass
class DocumentRankingConfig(RankingSignalConfig):
    """Document frequency configuration."""
    min_occurrences: int = 2


@dataclass
class BigramRankingConfig(RankingSignalConfig):
    """Bigram context configuration."""
    source: str = "data/bigrams/english_bigrams.txt"
    window: int = 1


@dataclass
class OCRModelRankingConfig(RankingSignalConfig):
    """OCR error model configuration."""
    source: str = "data/ocr_confusions.yaml"


@dataclass
class RankingConfig:
    """Multi-signal ranking configuration."""
    frequency: FrequencyRankingConfig = field(default_factory=FrequencyRankingConfig)
    document: DocumentRankingConfig = field(default_factory=DocumentRankingConfig)
    bigram: BigramRankingConfig = field(default_factory=BigramRankingConfig)
    ocr_model: OCRModelRankingConfig = field(default_factory=OCRModelRankingConfig)


@dataclass
class SanitizeConfig:
    """Complete sanitization configuration."""
    enabled: bool = True
    skip: SkipConfig = field(default_factory=SkipConfig)
    preserve: PreserveConfig = field(default_factory=PreserveConfig)
    correct: CorrectConfig = field(default_factory=CorrectConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    flag: FlagConfig = field(default_factory=FlagConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    dictionaries: dict[str, DictionarySource] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SanitizeConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls._parse(data)

    @classmethod
    def _parse(cls, data: dict) -> "SanitizeConfig":
        """Parse configuration from dict."""
        config = cls()

        if "sanitize" not in data:
            return config

        s = data["sanitize"]
        config.enabled = s.get("enabled", True)

        # Parse skip section
        if "skip" in s:
            sk = s["skip"]
            config.skip.pixel_detector = sk.get("pixel_detector", True)
            config.skip.single_char_max_conf = sk.get("single_char_max_conf", 50.0)

        # Parse preserve section
        if "preserve" in s:
            p = s["preserve"]
            config.preserve.confidence_threshold = p.get("confidence_threshold", 100.0)
            config.preserve.numeric_ratio = p.get("numeric_ratio", 0.5)
            config.preserve.exact_match_dictionaries = p.get(
                "exact_match_dictionaries",
                ["english", "names", "medical", "custom"]
            )

        # Parse correct section
        if "correct" in s:
            c = s["correct"]
            config.correct.algorithm = c.get("algorithm", "symspell")
            config.correct.max_edit_distance = c.get("max_edit_distance", 2)
            config.correct.min_correction_score = c.get("min_correction_score", 0.7)
            config.correct.case_sensitive = c.get("case_sensitive", False)
            if "dictionary_weights" in c:
                config.correct.dictionary_weights = c["dictionary_weights"]

        # Parse context section
        if "context" in s:
            ctx = s["context"]
            config.context.enabled = ctx.get("enabled", True)
            config.context.ngram_window = ctx.get("ngram_window", 2)
            config.context.min_context_score = ctx.get("min_context_score", 0.6)
            config.context.max_edit_distance = ctx.get("max_edit_distance", 3)

        # Parse flag section
        if "flag" in s:
            f = s["flag"]
            config.flag.mark_uncertain = f.get("mark_uncertain", True)
            config.flag.preserve_original = f.get("preserve_original", True)

        # Parse ranking section
        if "ranking" in s:
            r = s["ranking"]

            # Frequency ranking
            if "frequency" in r:
                freq = r["frequency"]
                config.ranking.frequency.enabled = freq.get("enabled", True)
                config.ranking.frequency.weight = freq.get("weight", 1.0)
                config.ranking.frequency.source = freq.get("source", "data/frequencies/english_freq.txt")
                config.ranking.frequency.fallback_frequency = freq.get("fallback_frequency", 1)

            # Document ranking
            if "document" in r:
                doc = r["document"]
                config.ranking.document.enabled = doc.get("enabled", True)
                config.ranking.document.weight = doc.get("weight", 0.3)
                config.ranking.document.min_occurrences = doc.get("min_occurrences", 2)

            # Bigram ranking
            if "bigram" in r:
                big = r["bigram"]
                config.ranking.bigram.enabled = big.get("enabled", True)
                config.ranking.bigram.weight = big.get("weight", 0.5)
                config.ranking.bigram.source = big.get("source", "data/bigrams/english_bigrams.txt")
                config.ranking.bigram.window = big.get("window", 1)

            # OCR model ranking
            if "ocr_model" in r:
                ocr = r["ocr_model"]
                config.ranking.ocr_model.enabled = ocr.get("enabled", True)
                config.ranking.ocr_model.weight = ocr.get("weight", 0.4)
                config.ranking.ocr_model.source = ocr.get("source", "data/ocr_confusions.yaml")

        # Parse dictionaries section
        if "dictionaries" in data:
            for name, d in data["dictionaries"].items():
                config.dictionaries[name] = DictionarySource(
                    path=d.get("path", ""),
                    format=d.get("format", "wordlist"),
                    min_word_length=d.get("min_word_length", 1),
                    min_frequency=d.get("min_frequency", 0),
                )

        return config


def load_sanitize_config(config_path: Optional[Path | str] = None) -> SanitizeConfig:
    """
    Load sanitization configuration.

    Args:
        config_path: Path to config file. If None, uses default config/sanitize.yaml

    Returns:
        SanitizeConfig object
    """
    if config_path is None:
        default_path = Path(__file__).parent.parent.parent / "config" / "sanitize.yaml"
        if default_path.exists():
            config_path = default_path
        else:
            return SanitizeConfig()

    return SanitizeConfig.from_yaml(config_path)


# =============================================================================
# Dictionary Management
# =============================================================================

class DictionaryManager:
    """Manages multiple dictionaries for fuzzy matching."""

    def __init__(self, config: SanitizeConfig, base_path: Optional[Path] = None):
        """
        Initialize dictionary manager.

        Args:
            config: Sanitization configuration
            base_path: Base path for resolving relative dictionary paths
        """
        self.config = config
        self.base_path = base_path or Path(__file__).parent.parent.parent

        # SymSpell instances per dictionary
        self._symspell: dict[str, SymSpell] = {}

        # Simple word sets for exact matching (fallback)
        self._word_sets: dict[str, set[str]] = {}

        # Combined word set for quick exact match check
        self._all_words: set[str] = set()

        self._loaded = False

    def load(self) -> None:
        """Load all configured dictionaries."""
        if self._loaded:
            return

        for name, source in self.config.dictionaries.items():
            self._load_dictionary(name, source)

        self._loaded = True

    def _load_dictionary(self, name: str, source: DictionarySource) -> None:
        """Load a single dictionary."""
        path = self.base_path / source.path
        if not path.exists():
            # Skip missing dictionaries silently
            return

        words = set()

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if source.format == "frequency":
                    # Format: word,frequency or word\tfrequency
                    parts = re.split(r"[,\t]", line)
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        try:
                            freq = int(parts[1].strip())
                            if freq < source.min_frequency:
                                continue
                        except ValueError:
                            pass
                    else:
                        word = parts[0].strip()
                else:
                    word = line

                if len(word) >= source.min_word_length:
                    words.add(word)
                    # Also add lowercase version for case-insensitive matching
                    words.add(word.lower())

        self._word_sets[name] = words
        self._all_words.update(words)

        # Initialize SymSpell for this dictionary if available
        if SYMSPELL_AVAILABLE and words:
            # Use max of correct and context edit distances
            max_dist = max(
                self.config.correct.max_edit_distance,
                self.config.context.max_edit_distance if self.config.context.enabled else 0
            )
            sym = SymSpell(max_dictionary_edit_distance=max_dist)
            for word in words:
                sym.create_dictionary_entry(word, 1)
            self._symspell[name] = sym

    def exact_match(self, word: str) -> Optional[str]:
        """
        Check if word has an exact match in any dictionary.

        Args:
            word: Word to check

        Returns:
            Dictionary name if matched, None otherwise
        """
        if not self._loaded:
            self.load()

        # Check case-sensitive first
        if word in self._all_words:
            for name in self.config.preserve.exact_match_dictionaries:
                if name in self._word_sets and word in self._word_sets[name]:
                    return name

        # Check case-insensitive
        word_lower = word.lower()
        if word_lower in self._all_words:
            for name in self.config.preserve.exact_match_dictionaries:
                if name in self._word_sets and word_lower in self._word_sets[name]:
                    return name

        return None

    def fuzzy_match(
        self,
        word: str,
        max_distance: Optional[int] = None,
        ranker: Optional[MultiSignalRanker] = None,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None
    ) -> list[tuple[str, int, str, float, float, float, float, float]]:
        """
        Find fuzzy matches for a word.

        Args:
            word: Word to match
            max_distance: Maximum edit distance (default from config)
            ranker: Optional MultiSignalRanker for improved scoring
            prev_word: Previous word for context (if using ranker)
            next_word: Next word for context (if using ranker)

        Returns:
            List of (matched_word, distance, dictionary_name, score,
                    freq_factor, doc_factor, bigram_factor, ocr_factor) tuples,
            sorted by score descending
        """
        if not self._loaded:
            self.load()

        if max_distance is None:
            max_distance = self.config.correct.max_edit_distance

        results = []

        if SYMSPELL_AVAILABLE:
            # Use SymSpell for fast matching
            lookup_word = word if self.config.correct.case_sensitive else word.lower()

            for name, sym in self._symspell.items():
                weight = self.config.correct.dictionary_weights.get(name, 1.0)

                suggestions = sym.lookup(
                    lookup_word,
                    Verbosity.CLOSEST,
                    max_edit_distance=max_distance,
                )

                for suggestion in suggestions:
                    # Base score = weight * (1 - distance / (max_distance + 1))
                    base_score = weight * (1 - suggestion.distance / (max_distance + 1))

                    # Apply ranking factors if ranker provided
                    freq_factor = 1.0
                    doc_factor = 1.0
                    bigram_factor = 1.0
                    ocr_factor = 1.0

                    if ranker:
                        freq_factor, doc_factor, bigram_factor, ocr_factor = ranker.rank_candidate(
                            word, suggestion.term, prev_word, next_word
                        )
                        # Apply factors to base score
                        base_score = base_score * freq_factor * doc_factor * bigram_factor * ocr_factor

                    results.append((
                        suggestion.term,
                        suggestion.distance,
                        name,
                        base_score,
                        freq_factor,
                        doc_factor,
                        bigram_factor,
                        ocr_factor,
                    ))
        else:
            # Fallback to basic Levenshtein
            results = self._fallback_fuzzy_match(word, max_distance, ranker, prev_word, next_word)

        # Sort by score descending
        results.sort(key=lambda x: x[3], reverse=True)
        return results

    def _fallback_fuzzy_match(
        self,
        word: str,
        max_distance: int,
        ranker: Optional[MultiSignalRanker] = None,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None
    ) -> list[tuple[str, int, str, float, float, float, float, float]]:
        """Fallback fuzzy matching using basic Levenshtein."""
        results = []
        lookup_word = word if self.config.correct.case_sensitive else word.lower()

        for name, words in self._word_sets.items():
            weight = self.config.correct.dictionary_weights.get(name, 1.0)

            for dict_word in words:
                compare_word = dict_word if self.config.correct.case_sensitive else dict_word.lower()
                dist = _levenshtein_distance(lookup_word, compare_word)

                if dist <= max_distance:
                    # Base score = weight * (1 - distance / (max_distance + 1))
                    base_score = weight * (1 - dist / (max_distance + 1))

                    # Apply ranking factors if ranker provided
                    freq_factor = 1.0
                    doc_factor = 1.0
                    bigram_factor = 1.0
                    ocr_factor = 1.0

                    if ranker:
                        freq_factor, doc_factor, bigram_factor, ocr_factor = ranker.rank_candidate(
                            word, dict_word, prev_word, next_word
                        )
                        # Apply factors to base score
                        base_score = base_score * freq_factor * doc_factor * bigram_factor * ocr_factor

                    results.append((dict_word, dist, name, base_score, freq_factor, doc_factor, bigram_factor, ocr_factor))

        return results


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# =============================================================================
# Main Sanitizer
# =============================================================================

class Sanitizer:
    """
    OCR text sanitizer using dictionary-based correction.

    Implements a 4-phase pipeline:
    1. Skip: Ignore pixel detections and low-confidence single chars
    2. Preserve: Keep high-confidence words and exact dictionary matches
    3. Correct: Apply single-word corrections within edit distance
    4. Context: Use adjacent words for harder corrections (future)
    """

    def __init__(
        self,
        config: Optional[SanitizeConfig] = None,
        config_path: Optional[Path | str] = None,
    ):
        """
        Initialize sanitizer.

        Args:
            config: SanitizeConfig object
            config_path: Path to config file (used if config is None)
        """
        if config is None:
            config = load_sanitize_config(config_path)

        self.config = config
        self.dictionaries = DictionaryManager(config)
        self.metrics = SanitizeMetrics()

        # Initialize multi-signal ranker
        self.ranker = MultiSignalRanker(
            frequency_config=FrequencyConfig(
                enabled=config.ranking.frequency.enabled,
                weight=config.ranking.frequency.weight,
                source=config.ranking.frequency.source,
                fallback_frequency=config.ranking.frequency.fallback_frequency,
            ),
            document_config=DocumentConfig(
                enabled=config.ranking.document.enabled,
                weight=config.ranking.document.weight,
                min_occurrences=config.ranking.document.min_occurrences,
            ),
            bigram_config=BigramConfig(
                enabled=config.ranking.bigram.enabled,
                weight=config.ranking.bigram.weight,
                source=config.ranking.bigram.source,
                window=config.ranking.bigram.window,
            ),
            ocr_config=OCRModelConfig(
                enabled=config.ranking.ocr_model.enabled,
                weight=config.ranking.ocr_model.weight,
                source=config.ranking.ocr_model.source,
            ),
        )

        # Compiled patterns for efficiency
        self._numeric_pattern = re.compile(r"\d")
        self._email_pattern = re.compile(r"@")
        self._date_pattern = re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}[,.]?$")

    def load_dictionaries(self) -> None:
        """Load all configured dictionaries."""
        self.dictionaries.load()

    def sanitize_word(
        self,
        text: str,
        confidence: float = 0.0,
        engine: str = "",
    ) -> SanitizeResult:
        """
        Sanitize a single word.

        Args:
            text: Word text
            confidence: OCR confidence (0-100)
            engine: Source OCR engine

        Returns:
            SanitizeResult with sanitized text and status
        """
        # Ensure dictionaries are loaded
        if not self.dictionaries._loaded:
            self.dictionaries.load()

        self.metrics.total_words += 1

        # Phase 0: Skip
        skip_result = self._phase_skip(text, confidence, engine)
        if skip_result:
            self.metrics.skipped += 1
            return skip_result

        # Phase 1: Preserve
        preserve_result = self._phase_preserve(text, confidence)
        if preserve_result:
            self.metrics.preserved += 1
            return preserve_result

        # Phase 2: Correct
        correct_result = self._phase_correct(text, confidence)
        if correct_result:
            self.metrics.corrected += 1
            return correct_result

        # Phase 3: Context (placeholder - requires word context)
        # Implemented in sanitize_words()

        # Phase 4: Flag as uncertain
        self.metrics.uncertain += 1
        return SanitizeResult(
            original_text=text,
            sanitized_text=text if self.config.flag.preserve_original else "",
            status=SanitizeStatus.UNCERTAIN,
            confidence=confidence,
            edit_distance=0,
            correction_score=0.0,
        )

    def _phase_skip(
        self,
        text: str,
        confidence: float,
        engine: str,
    ) -> Optional[SanitizeResult]:
        """Phase 0: Check if word should be skipped."""

        # Skip pixel detector results
        if self.config.skip.pixel_detector and engine == "pixel_detector":
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                status=SanitizeStatus.SKIPPED,
                confidence=confidence,
                edit_distance=0,
                correction_score=0.0,
            )

        # Skip low-confidence single characters
        if (
            len(text) == 1 and
            confidence < self.config.skip.single_char_max_conf
        ):
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                status=SanitizeStatus.SKIPPED,
                confidence=confidence,
                edit_distance=0,
                correction_score=0.0,
            )

        return None

    def _phase_preserve(
        self,
        text: str,
        confidence: float,
    ) -> Optional[SanitizeResult]:
        """Phase 1: Check if word should be preserved."""

        # Preserve high-confidence words
        if confidence >= self.config.preserve.confidence_threshold:
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                status=SanitizeStatus.PRESERVED,
                confidence=confidence,
                edit_distance=0,
                correction_score=1.0,
            )

        # Preserve words with high numeric ratio
        numeric_ratio = self._get_numeric_ratio(text)
        if numeric_ratio >= self.config.preserve.numeric_ratio:
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                status=SanitizeStatus.PRESERVED,
                confidence=confidence,
                edit_distance=0,
                correction_score=1.0,
            )

        # Preserve if it's an email (contains @)
        if self._email_pattern.search(text):
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                status=SanitizeStatus.PRESERVED,
                confidence=confidence,
                edit_distance=0,
                correction_score=1.0,
            )

        # Preserve if it's a date pattern
        if self._date_pattern.match(text):
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                status=SanitizeStatus.PRESERVED,
                confidence=confidence,
                edit_distance=0,
                correction_score=1.0,
            )

        # Preserve exact dictionary matches
        matched_dict = self.dictionaries.exact_match(text)
        if matched_dict:
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                status=SanitizeStatus.PRESERVED,
                confidence=confidence,
                edit_distance=0,
                correction_score=1.0,
                matched_dictionary=matched_dict,
            )

        # Strip common punctuation and try again
        stripped = text.strip(".,;:!?()[]{}\"'")
        if stripped != text and stripped:
            matched_dict = self.dictionaries.exact_match(stripped)
            if matched_dict:
                return SanitizeResult(
                    original_text=text,
                    sanitized_text=text,
                    status=SanitizeStatus.PRESERVED,
                    confidence=confidence,
                    edit_distance=0,
                    correction_score=1.0,
                    matched_dictionary=matched_dict,
                )

        return None

    def _phase_correct(
        self,
        text: str,
        confidence: float,
        ranker: Optional[MultiSignalRanker] = None,
        prev_word: Optional[str] = None,
        next_word: Optional[str] = None
    ) -> Optional[SanitizeResult]:
        """Phase 2: Apply single-word correction."""

        # Strip punctuation for matching, preserve for output
        stripped = text.strip(".,;:!?()[]{}\"'")
        prefix = text[:len(text) - len(text.lstrip(".,;:!?()[]{}\"'"))]
        suffix = text[len(text.rstrip(".,;:!?()[]{}\"'")):]

        if not stripped:
            return None

        # Find fuzzy matches with ranking
        matches = self.dictionaries.fuzzy_match(stripped, ranker=ranker, prev_word=prev_word, next_word=next_word)

        if not matches:
            return None

        best_match, distance, dict_name, score, freq_factor, doc_factor, bigram_factor, ocr_factor = matches[0]

        # Check if score meets threshold
        if score < self.config.correct.min_correction_score:
            return None

        # Don't correct if distance is 0 (should have been caught in preserve)
        if distance == 0:
            return SanitizeResult(
                original_text=text,
                sanitized_text=text,
                status=SanitizeStatus.PRESERVED,
                confidence=confidence,
                edit_distance=0,
                correction_score=score,
                matched_dictionary=dict_name,
                frequency_factor=freq_factor,
                document_factor=doc_factor,
                bigram_factor=bigram_factor,
                ocr_factor=ocr_factor,
            )

        # Preserve case pattern from original
        corrected = self._preserve_case(stripped, best_match)

        # Re-attach punctuation
        sanitized = prefix + corrected + suffix

        return SanitizeResult(
            original_text=text,
            sanitized_text=sanitized,
            status=SanitizeStatus.CORRECTED,
            confidence=confidence,
            edit_distance=distance,
            correction_score=score,
            matched_dictionary=dict_name,
            frequency_factor=freq_factor,
            document_factor=doc_factor,
            bigram_factor=bigram_factor,
            ocr_factor=ocr_factor,
        )

    def _get_numeric_ratio(self, text: str) -> float:
        """Calculate ratio of digits in text."""
        if not text:
            return 0.0
        # Count digits
        digit_count = len(self._numeric_pattern.findall(text))
        return digit_count / len(text)

    def _preserve_case(self, original: str, corrected: str) -> str:
        """Preserve case pattern from original word."""
        if not original or not corrected:
            return corrected

        # All uppercase
        if original.isupper():
            return corrected.upper()

        # All lowercase
        if original.islower():
            return corrected.lower()

        # Title case
        if original.istitle():
            return corrected.title()

        # Mixed case - just return corrected
        return corrected

    def sanitize_words(
        self,
        words: list[dict],
    ) -> list[SanitizeResult]:
        """
        Sanitize a list of words with context awareness.

        Args:
            words: List of word dicts with 'text', 'confidence', 'engine' keys

        Returns:
            List of SanitizeResult objects
        """
        # Reset metrics
        self.metrics = SanitizeMetrics()

        # Build document index for ranking
        word_texts = [w.get("text", "") for w in words]
        self.ranker.build_document_index(word_texts)

        results = []

        for i, word in enumerate(words):
            text = word.get("text", "")
            confidence = word.get("confidence", 0.0)
            engine = word.get("engine", "")

            # Get context words for ranking
            prev_word = words[i-1].get("text", "") if i > 0 else None
            next_word = words[i+1].get("text", "") if i < len(words) - 1 else None

            # Ensure dictionaries are loaded
            if not self.dictionaries._loaded:
                self.dictionaries.load()

            self.metrics.total_words += 1

            # Phase 0: Skip
            skip_result = self._phase_skip(text, confidence, engine)
            if skip_result:
                self.metrics.skipped += 1
                results.append(skip_result)
                continue

            # Phase 1: Preserve
            preserve_result = self._phase_preserve(text, confidence)
            if preserve_result:
                self.metrics.preserved += 1
                results.append(preserve_result)
                continue

            # Phase 2: Correct with ranking
            correct_result = self._phase_correct(text, confidence, self.ranker, prev_word, next_word)
            if correct_result:
                self.metrics.corrected += 1
                results.append(correct_result)
                continue

            # Phase 3: Context correction for uncertain words
            if self.config.context.enabled:
                context_result = self._phase_context(
                    text, confidence, words, i
                )
                if context_result:
                    self.metrics.context_corrected += 1
                    results.append(context_result)
                    continue

            # Phase 4: Flag as uncertain
            self.metrics.uncertain += 1
            results.append(SanitizeResult(
                original_text=text,
                sanitized_text=text if self.config.flag.preserve_original else "",
                status=SanitizeStatus.UNCERTAIN,
                confidence=confidence,
                edit_distance=0,
                correction_score=0.0,
            ))

        return results

    def _phase_context(
        self,
        text: str,
        confidence: float,
        words: list[dict],
        index: int,
    ) -> Optional[SanitizeResult]:
        """Phase 3: Context-based correction using adjacent words."""
        # Get context window
        window = self.config.context.ngram_window
        start = max(0, index - window)
        end = min(len(words), index + window + 1)

        # Build context (exclude current word)
        context_words = []
        for i in range(start, end):
            if i != index:
                context_words.append(words[i].get("text", ""))

        # For now, just try with higher edit distance
        # Future: Use n-gram frequencies for scoring
        stripped = text.strip(".,;:!?()[]{}\"'")
        if not stripped:
            return None

        # Get prev/next words for context
        prev_word = words[index-1].get("text", "") if index > 0 else None
        next_word = words[index+1].get("text", "") if index < len(words) - 1 else None

        matches = self.dictionaries.fuzzy_match(
            stripped,
            max_distance=self.config.context.max_edit_distance,
            ranker=self.ranker,
            prev_word=prev_word,
            next_word=next_word,
        )

        if not matches:
            return None

        best_match, distance, dict_name, score, freq_factor, doc_factor, bigram_factor, ocr_factor = matches[0]

        if score < self.config.context.min_context_score:
            return None

        if distance == 0:
            return None

        # Preserve case and punctuation
        prefix = text[:len(text) - len(text.lstrip(".,;:!?()[]{}\"'"))]
        suffix = text[len(text.rstrip(".,;:!?()[]{}\"'")):]
        corrected = self._preserve_case(stripped, best_match)
        sanitized = prefix + corrected + suffix

        return SanitizeResult(
            original_text=text,
            sanitized_text=sanitized,
            status=SanitizeStatus.CONTEXT_CORRECTED,
            confidence=confidence,
            edit_distance=distance,
            correction_score=score,
            matched_dictionary=dict_name,
            frequency_factor=freq_factor,
            document_factor=doc_factor,
            bigram_factor=bigram_factor,
            ocr_factor=ocr_factor,
        )

    def get_metrics(self) -> SanitizeMetrics:
        """Get current sanitization metrics."""
        return self.metrics


# =============================================================================
# Convenience Functions
# =============================================================================

def sanitize_csv(
    input_path: Path | str,
    output_path: Optional[Path | str] = None,
    config_path: Optional[Path | str] = None,
) -> SanitizeMetrics:
    """
    Sanitize words from a CSV file.

    Args:
        input_path: Path to input CSV (from portadoc extract)
        output_path: Path to output CSV (optional, prints to stdout if None)
        config_path: Path to config file

    Returns:
        SanitizeMetrics with results
    """
    import csv
    import sys

    input_path = Path(input_path)
    config = load_sanitize_config(config_path)
    sanitizer = Sanitizer(config)
    sanitizer.load_dictionaries()

    # Read input
    words = []
    fieldnames = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            words.append({
                "text": row.get("text", ""),
                "confidence": float(row.get("ocr_confidence", row.get("confidence", 0))),
                "engine": row.get("engine", ""),
                "_row": row,  # Keep original row for output
            })

    # Sanitize
    results = sanitizer.sanitize_words(words)

    # Write output
    output_fieldnames = list(fieldnames) + [
        "sanitized_text", "sanitize_status", "edit_distance", "correction_score"
    ]

    out_file = open(output_path, "w", newline="", encoding="utf-8") if output_path else sys.stdout

    try:
        writer = csv.DictWriter(out_file, fieldnames=output_fieldnames)
        writer.writeheader()

        for word, result in zip(words, results):
            row = dict(word["_row"])
            row["sanitized_text"] = result.sanitized_text
            row["sanitize_status"] = result.status.value
            row["edit_distance"] = result.edit_distance
            row["correction_score"] = f"{result.correction_score:.4f}"
            writer.writerow(row)
    finally:
        if output_path:
            out_file.close()

    return sanitizer.get_metrics()


def check_symspell() -> bool:
    """Check if SymSpell is available."""
    return SYMSPELL_AVAILABLE
