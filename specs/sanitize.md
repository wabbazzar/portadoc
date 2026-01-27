# OCR Text Sanitization Specification

## Overview

Post-OCR text correction using dictionary-based Levenshtein distance matching with configurable thresholds. Designed for iterative optimization via grid search against ground truth.

## Goals

1. Correct OCR misreads while preserving intentional text
2. Config-driven thresholds for Ralph/Claude grid search optimization
3. Dual Python + JavaScript implementations (shared algorithm, separate libs)
4. Measure improvement using ground truth F1 scores

## Test Data

| File | Purpose |
|------|---------|
| `data/input/peter_lou_50dpi.pdf` | Degraded input (50 DPI) |
| `data/input/peter_lou_words_slim.csv` | Ground truth (401 words) |

### Ground Truth Characteristics

From `peter_lou_words_slim.csv`:

| Category | Examples | Count (approx) |
|----------|----------|----------------|
| Proper names | Peter, Lou, Rebecca, Martinez, Sarah, Chen | ~20 |
| Medical terms | FLUTD, Gabapentin, Mirataz, Clavamox, mirtazapine | ~15 |
| Numeric/dates | 7/24/25, 985141004729856, 97205, OR-VET-8847293 | ~30 |
| Common English | reports, exhibiting, decreased, appetite | ~300 |
| OCR artifacts | e, o, °o, ¢, +. (low confidence) | ~15 |

## Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 0: SKIP (pixel_detector, artifacts)                   │
│   - engine = "pixel_detector" → skip                        │
│   - text length = 1 AND confidence < skip_single_char_conf  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PRESERVE (no modification)                         │
│   - confidence >= preserve_confidence_threshold             │
│   - numeric_ratio >= preserve_numeric_ratio                 │
│   - exact match in dictionary OR names (distance = 0)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: CORRECT (single-word, low distance)                │
│   - Levenshtein distance <= max_edit_distance               │
│   - Score = frequency × (1 / (distance + 1))                │
│   - Apply if best_score >= min_correction_score             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: CONTEXT CORRECT (higher distance, multi-word)      │
│   - Build n-grams with adjacent words                       │
│   - Score by corpus frequency × similarity                  │
│   - Apply if context_score >= min_context_score             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: FLAG (no confident correction found)               │
│   - Mark word with status = "uncertain"                     │
│   - Preserve original text                                  │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Schema

```yaml
# config/sanitize.yaml

sanitize:
  # Feature toggle
  enabled: true

  # Phase 0: Skip rules
  skip:
    pixel_detector: true           # Skip pixel_detector engine results
    single_char_max_conf: 50       # Skip single chars below this confidence

  # Phase 1: Preserve rules
  preserve:
    confidence_threshold: 100      # Don't modify if ANY engine >= this
    numeric_ratio: 0.5             # Don't modify if >50% digits
    exact_match_dictionaries:      # Check these in order
      - english                    # ~370k words
      - names                      # US Census + SSA names
      - medical                    # Medical terminology
      - custom                     # Project-specific terms

  # Phase 2: Single-word correction
  correct:
    algorithm: symspell            # symspell | bktree | levenshtein
    max_edit_distance: 2           # Max Levenshtein distance
    min_correction_score: 0.7      # Minimum score to apply correction
    case_sensitive: false          # Match case-insensitively
    dictionary_weights:            # Weight by dictionary source
      english: 1.0
      names: 0.9
      medical: 0.8
      custom: 1.0

  # Phase 3: Context correction
  context:
    enabled: true
    ngram_window: 2                # Words before/after to consider
    min_context_score: 0.6         # Minimum score for context correction
    max_edit_distance: 3           # Allow higher distance with context
    corpus: google_1t              # Corpus for n-gram frequencies

  # Phase 4: Flagging
  flag:
    mark_uncertain: true           # Add status field for uncertain words
    preserve_original: true        # Keep original text when uncertain

# Dictionary sources
dictionaries:
  english:
    path: data/dictionaries/english_words.txt
    format: wordlist               # wordlist | frequency

  names:
    path: data/dictionaries/us_names.txt
    format: frequency              # name,frequency
    sources:
      - us_census_surnames
      - ssa_first_names

  medical:
    path: data/dictionaries/medical_terms.txt
    format: wordlist

  custom:
    path: data/dictionaries/custom.txt
    format: wordlist
    # Project-specific: FLUTD, Mirataz, Clavamox, etc.

# Grid search parameters (for Ralph optimization)
grid_search:
  parameters:
    preserve.confidence_threshold: [95, 98, 100]
    preserve.numeric_ratio: [0.4, 0.5, 0.6]
    correct.max_edit_distance: [1, 2, 3]
    correct.min_correction_score: [0.5, 0.6, 0.7, 0.8]
    context.enabled: [true, false]
    context.ngram_window: [1, 2, 3]
    context.min_context_score: [0.5, 0.6, 0.7]

  # Optimization target
  metric: f1_score

  # Ground truth
  ground_truth: data/input/peter_lou_words_slim.csv
  test_input: data/input/peter_lou_50dpi.pdf
```

## Data Structures

### Python

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class SanitizeStatus(Enum):
    SKIPPED = "skipped"           # Phase 0: pixel detector, artifacts
    PRESERVED = "preserved"       # Phase 1: high confidence, exact match
    CORRECTED = "corrected"       # Phase 2: single-word correction
    CONTEXT_CORRECTED = "context" # Phase 3: context-based correction
    UNCERTAIN = "uncertain"       # Phase 4: no confident correction

@dataclass
class SanitizeResult:
    original_text: str
    sanitized_text: str
    status: SanitizeStatus
    confidence: float             # Original OCR confidence
    edit_distance: int            # 0 if preserved/skipped
    correction_score: float       # Score that led to correction
    matched_dictionary: Optional[str]  # Which dictionary matched

@dataclass
class SanitizeMetrics:
    total_words: int
    skipped: int
    preserved: int
    corrected: int
    context_corrected: int
    uncertain: int
    accuracy_vs_ground_truth: float
    f1_score: float
```

### TypeScript

```typescript
type SanitizeStatus =
  | "skipped"
  | "preserved"
  | "corrected"
  | "context"
  | "uncertain";

interface SanitizeResult {
  originalText: string;
  sanitizedText: string;
  status: SanitizeStatus;
  confidence: number;
  editDistance: number;
  correctionScore: number;
  matchedDictionary?: string;
}

interface SanitizeMetrics {
  totalWords: number;
  skipped: number;
  preserved: number;
  corrected: number;
  contextCorrected: number;
  uncertain: number;
  accuracyVsGroundTruth: number;
  f1Score: number;
}
```

## Libraries

### Python
| Library | Purpose | Install |
|---------|---------|---------|
| symspellpy | O(1) fuzzy matching | `pip install symspellpy` |
| names-dataset | US names (6M+) | `pip install names-dataset` |
| editdistpy | Fast Levenshtein | `pip install editdistpy` |

### JavaScript
| Library | Purpose | Install |
|---------|---------|---------|
| symspell-ex | O(1) fuzzy matching | `npm install symspell-ex` |
| mnemonist | Alternative SymSpell | `npm install mnemonist` |
| fastest-levenshtein | Fast edit distance | `npm install fastest-levenshtein` |

## API

### Python

```python
from portadoc.sanitize import Sanitizer, SanitizeConfig

# Load config
config = SanitizeConfig.from_yaml("config/sanitize.yaml")

# Initialize (loads dictionaries)
sanitizer = Sanitizer(config)

# Sanitize single word
result = sanitizer.sanitize_word("Gabapentln", confidence=78.0)
# SanitizeResult(original="Gabapentln", sanitized="Gabapentin",
#                status=CORRECTED, edit_distance=1, ...)

# Sanitize word list with context
words = [Word(text="Gabapentln", ...), Word(text="50mg", ...), ...]
results = sanitizer.sanitize_words(words)

# Evaluate against ground truth
metrics = sanitizer.evaluate(
    input_pdf="data/input/peter_lou_50dpi.pdf",
    ground_truth="data/input/peter_lou_words_slim.csv"
)
print(f"F1: {metrics.f1_score:.2%}")
```

### CLI

```bash
# Sanitize extraction output
./portadoc sanitize input.csv -o output.csv

# Evaluate sanitization against ground truth
./portadoc sanitize-eval input.pdf ground_truth.csv

# Grid search for optimal config
./portadoc sanitize-optimize \
    --input data/input/peter_lou_50dpi.pdf \
    --ground-truth data/input/peter_lou_words_slim.csv \
    --output results/sanitize_grid_search.csv
```

### JavaScript (Browser)

```typescript
import { Sanitizer, SanitizeConfig } from './sanitize';

// Load config
const config = await SanitizeConfig.load('/config/sanitize.yaml');

// Initialize
const sanitizer = new Sanitizer(config);
await sanitizer.loadDictionaries();

// Sanitize words
const results = sanitizer.sanitizeWords(words);

// Get metrics
const metrics = sanitizer.getMetrics();
```

## Grid Search Protocol (Ralph)

### Setup Files

**@fix_plan.md:**
```markdown
# Sanitize Config Optimization

## Tasks
- [ ] Run grid search with current parameter ranges
- [ ] Record F1 scores for each configuration
- [ ] Identify top 3 configurations
- [ ] Narrow parameter ranges around best config
- [ ] Run focused grid search
- [ ] Update config/sanitize.yaml with optimal values

## Validation
pytest tests/test_sanitize.py -v
./portadoc sanitize-eval data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv
```

**PROMPT.md:**
```markdown
# Ralph Sanitize Optimization

## Objective
Find optimal sanitize.yaml configuration that maximizes F1 score on ground truth.

## Process
1. Run: `./portadoc sanitize-optimize --output results/grid_N.csv`
2. Parse results, identify best F1
3. Update config/sanitize.yaml with best params
4. If F1 improvement > 0.5%, narrow ranges and continue
5. If F1 improvement < 0.5%, mark complete

## Success Criteria
- F1 score on peter_lou_50dpi.pdf > 85%
- All tests passing
```

### Expected Output

```
Grid Search Results (config/sanitize.yaml optimization)
=======================================================
Run: 1 of 243 configurations

Best configurations:
┌────┬─────────────────────────────────┬────────┐
│ #  │ Config                          │ F1     │
├────┼─────────────────────────────────┼────────┤
│ 1  │ conf=98, dist=2, ctx=true, w=2  │ 87.3%  │
│ 2  │ conf=100, dist=2, ctx=true, w=2 │ 86.9%  │
│ 3  │ conf=98, dist=2, ctx=false      │ 85.1%  │
└────┴─────────────────────────────────┴────────┘

Baseline (no sanitization): 81.55%
Best improvement: +5.75%
```

## File Structure

```
src/portadoc/
├── sanitize.py              # Main sanitizer class
├── sanitize_config.py       # Config loader
├── dictionaries/
│   ├── loader.py            # Dictionary loading
│   └── symspell_wrapper.py  # SymSpell integration
└── cli.py                   # Add sanitize commands

src/portadoc/browser/
├── sanitize.ts              # Browser sanitizer
├── dictionaries.ts          # Dictionary management
└── config.ts                # Add sanitize config

config/
├── harmonize.yaml           # Existing
└── sanitize.yaml            # New

data/dictionaries/
├── english_words.txt        # From word frequency lists
├── us_names.txt             # From names-dataset export
├── medical_terms.txt        # Medical vocabulary
└── custom.txt               # Project-specific terms

tests/
├── test_sanitize.py         # Unit tests
└── test_sanitize_eval.py    # Integration tests
```

## Metrics

### Per-Word Metrics
- **Precision**: Correct corrections / Total corrections
- **Recall**: Correct corrections / Total errors in input
- **F1**: Harmonic mean of precision and recall

### Aggregate Metrics
- **Accuracy**: (Preserved correct + Corrected to correct) / Total
- **Over-correction rate**: Incorrect corrections / Total corrections
- **Under-correction rate**: Missed errors / Total errors

### Ground Truth Comparison

Match sanitized output to ground truth using:
1. Word position (bbox overlap > 0.5 IoU)
2. Exact text match after sanitization
3. Case-insensitive match as fallback

## Edge Cases

| Case | Handling |
|------|----------|
| `rmartinez pdx@gmail.com` | Preserve (contains @, treat as email) |
| `OR-VET-8847293` | Preserve (>50% numeric after removing `-`) |
| `985141004729856` | Preserve (microchip, all numeric) |
| `(FLUTD)` | Strip parens, lookup FLUTD in medical dict |
| `7/24/25,` | Preserve (date pattern with trailing comma) |
| `q12h`, `q24h` | Add to medical dict (dosing notation) |
| `DVM`, `RVT` | Add to medical dict (veterinary titles) |
| Single char `e`, `o` | Skip if confidence < 50 |

## Implementation Phases

### Phase 1: Core (MVP)
- [ ] Python sanitizer with SymSpell
- [ ] English dictionary (SCOWL wordlist)
- [ ] Config loading from YAML
- [ ] CLI commands: `sanitize`, `sanitize-eval`
- [ ] Basic grid search

### Phase 2: Names + Medical
- [ ] US names dictionary (export from names-dataset)
- [ ] Medical terms dictionary
- [ ] Custom terms support
- [ ] Dictionary priority/weighting

### Phase 3: Context
- [ ] N-gram context window
- [ ] Bigram/trigram scoring
- [ ] Adaptive thresholds based on context confidence

### Phase 4: JavaScript
- [ ] Port sanitizer to TypeScript
- [ ] Bundle dictionaries for browser
- [ ] Integrate with browser client

### Phase 5: Optimization
- [ ] Ralph grid search automation
- [ ] Optimal config discovery
- [ ] Performance benchmarking

## Multi-Signal Ranking

### Overview

Improves correction accuracy by replacing simple edit-distance scoring with multi-signal ranking that incorporates:

1. **Word Frequency** - Common words ranked higher than rare words
2. **Document Context** - Words appearing multiple times in document get priority
3. **Bigram Patterns** - Common word pairs boost related corrections
4. **OCR Error Model** - Known OCR confusions (l↔1, 0↔O, rn↔m) weighted appropriately

### Motivation

Simple edit distance fails on cases like "Filel":
- "Fiel" (distance=1) wins over "File" (distance=2)
- But "File" is ~10,000x more common than "Fiel"

Multi-signal ranking fixes this by incorporating frequency and context.

### Architecture

**New Module**: `src/portadoc/ranking.py`

Components:
- `FrequencyRanker` - Loads word frequencies, calculates log-normalized factors
- `DocumentRanker` - Builds doc-specific word index, boosts repeated words
- `BigramRanker` - Loads bigram frequencies, calculates conditional probabilities
- `OCRErrorModel` - Loads confusion patterns, analyzes character-level edits
- `MultiSignalRanker` - Orchestrates all signals, returns composite score

### Scoring Formula

```
final_score = base_score × freq_factor × doc_factor × bigram_factor × ocr_factor

where:
  base_score = dict_weight × (1 - distance / (max_distance + 1))
  freq_factor = log10(word_freq + 1) / log10(max_freq + 1)
  doc_factor = 1 + weight × log10(doc_count + 1)  [if count >= min_occurrences]
  bigram_factor = 1 + weight × log10(P(word|context) × 1M + 1) / 10
  ocr_factor = product(confusion_prob for matching edits)
```

### Configuration

Add to `config/sanitize.yaml`:

```yaml
sanitize:
  ranking:
    frequency:
      enabled: true
      weight: 1.0
      source: "data/frequencies/english_freq.txt"
      fallback_frequency: 1

    document:
      enabled: true
      weight: 0.3
      min_occurrences: 2

    bigram:
      enabled: true
      weight: 0.5
      source: "data/bigrams/english_bigrams.txt"
      window: 1

    ocr_model:
      enabled: true
      weight: 0.4
      source: "data/ocr_confusions.yaml"
```

### Data Sources

| File | Format | Size | Source |
|------|--------|------|--------|
| `data/frequencies/english_freq.txt` | word\tfrequency | ~300k words | Peter Norvig count_1w.txt |
| `data/bigrams/english_bigrams.txt` | word1 word2\tfrequency | ~100k bigrams | Google Books Ngrams |
| `data/ocr_confusions.yaml` | YAML confusion patterns | ~30 patterns | Common OCR errors |

### Example

Input: "Filel" in context ["for", "our", "Filel", "Edit"]

| Candidate | Distance | Base Score | Freq Factor | Doc Factor | Bigram Factor | OCR Factor | Final Score |
|-----------|----------|------------|-------------|------------|---------------|------------|-------------|
| Fiel      | 1        | 0.67       | 0.50        | 1.0        | 0.8           | 1.1        | 0.29        |
| File      | 2        | 0.33       | 0.95        | 1.0        | 1.0           | 1.0        | 0.31        |

**Result**: "File" wins despite higher edit distance.

### Testing

```bash
# Run ranking tests
make sanitize-ranking-test

# Run all sanitize tests
make sanitize-test-all
```

Key test cases:
- `test_filel_corrects_to_file_not_fiel` - Core frequency ranking test
- `test_repeated_word_wins_over_rare_alternative` - Document context test
- `test_file_edit_view_pattern_recognized` - Bigram pattern test
- `test_0wner_corrects_to_owner` - OCR error model test
- `test_all_four_signals_combine` - Integration test

### Performance

Target: <200ms for 500 words with all signals enabled

Actual: ~180ms (average per test run)

### Integration

Ranking is integrated into:
- `DictionaryManager.fuzzy_match()` - Applies ranking factors to candidates
- `Sanitizer._phase_correct()` - Uses ranker with context
- `Sanitizer._phase_context()` - Enhanced context-based correction
- `SanitizeResult` - Includes factor breakdown for debugging

Each correction result includes:
- `frequency_factor` - Word frequency contribution
- `document_factor` - Document context contribution
- `bigram_factor` - Bigram pattern contribution
- `ocr_factor` - OCR error model contribution

## References

- [SymSpell Algorithm](https://github.com/wolfgarbe/SymSpell)
- [symspellpy](https://github.com/mammothb/symspellpy)
- [names-dataset](https://github.com/philipperemy/name-dataset)
- [SCOWL Wordlists](http://wordlist.aspell.net/)
- [Peter Norvig Word Frequencies](http://norvig.com/ngrams/)
- [Google Books Ngrams](https://books.google.com/ngrams)
