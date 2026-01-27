"""Multi-signal ranking for OCR correction candidates.

Provides frequency-based, document context, bigram, and OCR error model ranking
to improve correction accuracy beyond simple edit distance.
"""

import math
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RankingConfig:
    """Configuration for ranking signals."""
    enabled: bool = True
    weight: float = 1.0


@dataclass
class FrequencyConfig(RankingConfig):
    """Frequency ranking configuration."""
    source: str = "data/frequencies/english_freq.txt"
    fallback_frequency: int = 1


@dataclass
class DocumentConfig(RankingConfig):
    """Document frequency configuration."""
    min_occurrences: int = 2


@dataclass
class BigramConfig(RankingConfig):
    """Bigram context configuration."""
    source: str = "data/bigrams/english_bigrams.txt"
    window: int = 1


@dataclass
class OCRModelConfig(RankingConfig):
    """OCR error model configuration."""
    source: str = "data/ocr_confusions.yaml"


class FrequencyRanker:
    """Ranks correction candidates by word frequency."""

    def __init__(self, config: FrequencyConfig):
        self.config = config
        self.frequencies: Dict[str, int] = {}
        self.max_frequency = 1

        if config.enabled and Path(config.source).exists():
            self._load_frequencies()

    def _load_frequencies(self):
        """Load word frequencies from file."""
        try:
            with open(self.config.source, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        word, freq = parts
                        freq_val = int(freq)
                        self.frequencies[word.lower()] = freq_val
                        self.max_frequency = max(self.max_frequency, freq_val)
        except Exception as e:
            print(f"Warning: Failed to load frequency data: {e}")
            self.frequencies = {}

    def get_frequency_factor(self, word: str) -> float:
        """Get frequency-based ranking factor for a word.

        Args:
            word: Word to rank

        Returns:
            Factor between 0 and 1, where 1 is most frequent
        """
        if not self.config.enabled or not self.frequencies:
            return 1.0

        freq = self.frequencies.get(word.lower(), self.config.fallback_frequency)

        # Log normalization to compress range
        factor = math.log10(freq + 1) / math.log10(self.max_frequency + 1)

        # Apply weight
        return factor ** (1.0 / self.config.weight) if self.config.weight != 1.0 else factor


class DocumentRanker:
    """Ranks candidates by frequency in the current document."""

    def __init__(self, config: DocumentConfig):
        self.config = config
        self.document_index: Dict[str, int] = {}

    def build_document_index(self, words: List[str]):
        """Build case-insensitive index of word counts in document.

        Args:
            words: List of all words in the document
        """
        self.document_index = {}
        for word in words:
            key = word.lower()
            self.document_index[key] = self.document_index.get(key, 0) + 1

    def get_document_factor(self, word: str) -> float:
        """Get document frequency factor for a word.

        Args:
            word: Word to rank

        Returns:
            Factor >= 1.0, boosted if word appears frequently in document
        """
        if not self.config.enabled:
            return 1.0

        count = self.document_index.get(word.lower(), 0)

        # Only boost if meets minimum threshold
        if count < self.config.min_occurrences:
            return 1.0

        # Logarithmic boost
        boost = 1.0 + self.config.weight * math.log10(count + 1)
        return boost


class BigramRanker:
    """Ranks candidates by bigram context probability."""

    def __init__(self, config: BigramConfig):
        self.config = config
        self.bigrams: Dict[Tuple[str, str], int] = {}
        self.unigram_counts: Dict[str, int] = {}

        if config.enabled and Path(config.source).exists():
            self._load_bigrams()

    def _load_bigrams(self):
        """Load bigram frequencies from file."""
        try:
            with open(self.config.source, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        words, freq = parts
                        word_parts = words.split()
                        if len(word_parts) == 2:
                            w1, w2 = word_parts
                            freq_val = int(freq)
                            self.bigrams[(w1.lower(), w2.lower())] = freq_val

                            # Track unigram counts for probability calculation
                            self.unigram_counts[w1.lower()] = self.unigram_counts.get(w1.lower(), 0) + freq_val
                            self.unigram_counts[w2.lower()] = self.unigram_counts.get(w2.lower(), 0) + freq_val
        except Exception as e:
            print(f"Warning: Failed to load bigram data: {e}")
            self.bigrams = {}

    def get_bigram_factor(self, word: str, prev_word: Optional[str] = None,
                          next_word: Optional[str] = None) -> float:
        """Get bigram context factor for a word.

        Args:
            word: Word to rank
            prev_word: Previous word in context (optional)
            next_word: Next word in context (optional)

        Returns:
            Factor based on conditional probability of word given context
        """
        if not self.config.enabled or not self.bigrams:
            return 1.0

        word_lower = word.lower()
        scores = []

        # P(word | prev_word)
        if prev_word:
            prev_lower = prev_word.lower()
            bigram_count = self.bigrams.get((prev_lower, word_lower), 0)
            unigram_count = self.unigram_counts.get(prev_lower, 0)
            if unigram_count > 0:
                prob = bigram_count / unigram_count
                scores.append(prob)

        # P(next_word | word)
        if next_word:
            next_lower = next_word.lower()
            bigram_count = self.bigrams.get((word_lower, next_lower), 0)
            unigram_count = self.unigram_counts.get(word_lower, 0)
            if unigram_count > 0:
                prob = bigram_count / unigram_count
                scores.append(prob)

        if not scores:
            return 1.0

        # Average probability, normalized and weighted
        avg_prob = sum(scores) / len(scores)

        # Convert to factor (probabilities are very small, so use log scale)
        if avg_prob > 0:
            factor = 1.0 + self.config.weight * math.log10(avg_prob * 1000000 + 1) / 10
            return max(0.5, min(2.0, factor))  # Clamp to reasonable range

        return 1.0


class OCRErrorModel:
    """Ranks candidates by likelihood of OCR error patterns."""

    def __init__(self, config: OCRModelConfig):
        self.config = config
        self.confusions: List[Dict] = []

        if config.enabled and Path(config.source).exists():
            self._load_confusions()

    def _load_confusions(self):
        """Load OCR confusion patterns from YAML."""
        try:
            with open(self.config.source, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.confusions = data.get('confusions', [])
        except Exception as e:
            print(f"Warning: Failed to load OCR confusion data: {e}")
            self.confusions = []

    def get_ocr_factor(self, original: str, candidate: str) -> float:
        """Get OCR error likelihood factor.

        Analyzes character-level differences between original and candidate
        to determine if they match known OCR confusion patterns.

        Args:
            original: Original OCR text
            candidate: Correction candidate

        Returns:
            Factor >= 1.0 if differences match known OCR errors
        """
        if not self.config.enabled or not self.confusions:
            return 1.0

        # Simple character alignment and difference detection
        factors = []

        # Check for direct substitutions
        for i, (o_char, c_char) in enumerate(zip(original, candidate)):
            if o_char != c_char:
                factor = self._check_confusion(o_char, c_char, i, len(original))
                if factor > 1.0:
                    factors.append(factor)

        # Check for insertions/deletions (pattern-based)
        if len(original) != len(candidate):
            # Check multi-character patterns
            for confusion in self.confusions:
                pattern = confusion['pattern']
                if len(pattern) > 1:
                    for confused in confusion['confused_with']:
                        if pattern in original and confused in candidate:
                            factors.append(confusion['probability'])
                        elif confused in original and pattern in candidate:
                            factors.append(confusion['probability'])

        if not factors:
            return 1.0

        # Combine factors (geometric mean to avoid over-boosting)
        product = 1.0
        for f in factors:
            product *= f

        result = product ** (1.0 / len(factors))

        # Apply weight
        if self.config.weight != 1.0:
            result = 1.0 + (result - 1.0) * self.config.weight

        return result

    def _check_confusion(self, char1: str, char2: str, position: int, word_len: int) -> float:
        """Check if character substitution matches known confusion."""
        for confusion in self.confusions:
            pattern = confusion['pattern']
            confused_with = confusion['confused_with']
            probability = confusion['probability']

            # Check if single-character pattern
            if len(pattern) == 1:
                # Check context if specified
                context = confusion.get('context')
                if context == 'end' and position != word_len - 1:
                    continue
                if context == 'middle' and (position == 0 or position == word_len - 1):
                    continue

                # Check both directions
                if (pattern == char1 and char2 in confused_with) or \
                   (pattern == char2 and char1 in confused_with):
                    return 1.0 + probability

        return 1.0


class MultiSignalRanker:
    """Orchestrates all ranking signals."""

    def __init__(self,
                 frequency_config: Optional[FrequencyConfig] = None,
                 document_config: Optional[DocumentConfig] = None,
                 bigram_config: Optional[BigramConfig] = None,
                 ocr_config: Optional[OCRModelConfig] = None):

        self.frequency_ranker = FrequencyRanker(frequency_config or FrequencyConfig())
        self.document_ranker = DocumentRanker(document_config or DocumentConfig())
        self.bigram_ranker = BigramRanker(bigram_config or BigramConfig())
        self.ocr_ranker = OCRErrorModel(ocr_config or OCRModelConfig())

    def build_document_index(self, words: List[str]):
        """Build document index for context-based ranking."""
        self.document_ranker.build_document_index(words)

    def rank_candidate(self,
                       original: str,
                       candidate: str,
                       prev_word: Optional[str] = None,
                       next_word: Optional[str] = None) -> Tuple[float, float, float, float]:
        """Calculate all ranking factors for a candidate.

        Args:
            original: Original OCR text
            candidate: Correction candidate
            prev_word: Previous word in context
            next_word: Next word in context

        Returns:
            Tuple of (frequency_factor, document_factor, bigram_factor, ocr_factor)
        """
        freq_factor = self.frequency_ranker.get_frequency_factor(candidate)
        doc_factor = self.document_ranker.get_document_factor(candidate)
        bigram_factor = self.bigram_ranker.get_bigram_factor(candidate, prev_word, next_word)
        ocr_factor = self.ocr_ranker.get_ocr_factor(original, candidate)

        return freq_factor, doc_factor, bigram_factor, ocr_factor

    def get_composite_factor(self,
                             original: str,
                             candidate: str,
                             prev_word: Optional[str] = None,
                             next_word: Optional[str] = None) -> float:
        """Calculate composite ranking factor from all signals.

        Args:
            original: Original OCR text
            candidate: Correction candidate
            prev_word: Previous word in context
            next_word: Next word in context

        Returns:
            Composite factor (product of all enabled signals)
        """
        freq, doc, bigram, ocr = self.rank_candidate(original, candidate, prev_word, next_word)
        return freq * doc * bigram * ocr
