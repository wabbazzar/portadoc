# Agent Instructions - OCR Text Sanitization

## Build Commands

```bash
# Activate virtual environment (REQUIRED)
source .venv/bin/activate

# Check dependencies
make sanitize-check

# Install symspellpy if missing
pip install symspellpy
```

## Test Commands

```bash
# Run all sanitization tests
make sanitize-test

# Run with verbose output (shows corrections)
make sanitize-test-verbose

# Run specific test
python -m pytest tests/test_sanitize_correction.py::TestDegradedOCRCorrection::test_correct_document -v

# Run original sanitize tests (unit tests)
python -m pytest tests/test_sanitize.py -v
```

## Run Commands

```bash
# Sanitize a CSV file
./portadoc sanitize input.csv -o output.csv

# Evaluate sanitization against ground truth
./portadoc sanitize-eval data/input/peter_lou.pdf data/input/peter_lou_words_slim.csv

# Check sanitization status
./portadoc sanitize-check
```

## Test Data

| File | Description |
|------|-------------|
| `data/input/peter_lou.pdf` | Clean 3-page test PDF |
| `data/input/peter_lou_50dpi.pdf` | Degraded version for testing |
| `data/input/peter_lou_words_slim.csv` | Ground truth (401 words) |

## Validation Criteria

### Correction Tests
Run `make sanitize-test` - all tests should pass:
- `test_correct_document` - "Decument" -> "Document"
- `test_correct_name` - "Hame:" -> "Name:"
- `test_correct_compassionate` - "Compassianae" -> "Compassionate"
- `test_correct_species` - "Speties:" -> "Species:"
- `test_correct_domestic` - "Domeelic" -> "Domestic"

### Coverage Test
- `test_ground_truth_coverage` - > 85% of ground truth words preserved or correctable

### No Over-Correction
- Valid words like "Cars" should NOT be changed to "Care"
- Numeric values should be preserved
- Email addresses should be preserved

## Architecture

```
src/portadoc/
├── sanitize.py              # Main sanitizer (Sanitizer class, DictionaryManager)
├── cli.py                   # CLI commands (sanitize, sanitize-eval, sanitize-check)
└── ...

config/
└── sanitize.yaml            # Thresholds and dictionary config

data/dictionaries/
├── english_words.txt        # 370k English words
├── us_names.txt             # 4.9k US names
├── medical_terms.txt        # Medical/veterinary terms
└── custom.txt               # Project-specific terms

tests/
├── test_sanitize.py         # Unit tests
└── test_sanitize_correction.py  # Correction tests (degraded OCR)
```

## Key Configuration

In `config/sanitize.yaml`:

```yaml
sanitize:
  correct:
    max_edit_distance: 2      # Max Levenshtein distance
    min_correction_score: 0.7 # Minimum score to apply correction
    dictionary_weights:
      english: 1.0
      names: 0.9
      medical: 0.8
      custom: 1.0

  context:
    enabled: true
    max_edit_distance: 3      # Higher distance with context
```

## Scoring Formula

Current scoring in `sanitize.py`:
```python
score = weight / (distance + 1)
```

Examples:
- Distance 0 (exact match): score = 1.0 (preserved)
- Distance 1: score = 1.0 / 2 = 0.5
- Distance 2: score = 1.0 / 3 = 0.33
- Distance 3: score = 1.0 / 4 = 0.25

With `min_correction_score: 0.7`, nothing corrects!

## Common Issues

1. **No corrections happening**: Lower `min_correction_score` or change scoring formula
2. **Over-correction**: Raise `min_correction_score` or add word to preserve list
3. **Missing dictionary**: Check paths in `config/sanitize.yaml`
4. **SymSpell not found**: Run `pip install symspellpy`

## Quick Debug

```python
# In Python REPL
from portadoc.sanitize import Sanitizer, load_sanitize_config

config = load_sanitize_config()
s = Sanitizer(config)
s.load_dictionaries()

# Test a word
result = s.sanitize_word("Decument", confidence=60.0)
print(f"{result.original_text} -> {result.sanitized_text} ({result.status.value})")
print(f"Score: {result.correction_score}, Distance: {result.edit_distance}")
```
