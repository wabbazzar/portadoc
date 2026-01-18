# Agent Instructions - Portadoc

## Environment Setup

**ALWAYS run before any Python command:**
```bash
source .venv/bin/activate
```

## Build & Install

```bash
# Install in development mode
pip install -e .

# Install all dependencies
pip install -r requirements.txt
```

## Test Commands

```bash
# Run unit tests
pytest tests/ -v

# Run specific test
pytest tests/test_harmonize.py -v
```

## Extraction Commands

```bash
# Extract with smart harmonization (default)
make extract-smart PDF=data/input/peter_lou_50dpi.pdf

# Extract with all 4 engines
make extract-all PDF=data/input/peter_lou_50dpi.pdf

# View output
cat data/output/peter_lou_50dpi_extracted.csv | head -30
```

## Evaluation Commands

```bash
# Evaluate degraded PDF (main target)
make eval-smart PDF=data/input/peter_lou_50dpi.pdf

# Regression check - clean PDF (should stay >= 99% F1)
make eval-smart PDF=data/input/peter_lou.pdf
```

## Key Files

| File | Purpose |
|------|---------|
| `src/portadoc/harmonize.py` | Main harmonization logic - PRIMARY TARGET |
| `src/portadoc/config.py` | Configuration classes |
| `config/harmonize.yaml` | OCR settings and thresholds |
| `data/output/peter_lou_extracted.csv` | Gold standard (clean PDF) |
| `data/output/peter_lou_50dpi_extracted.csv` | Current degraded output |

## Debugging

```bash
# Extract with verbose output
portadoc extract data/input/peter_lou_50dpi.pdf --verbose

# Check specific rows in output
cat data/output/peter_lou_50dpi_extracted.csv | head -25
```

## Validation Checklist

Before marking EXIT_SIGNAL = true:

1. [ ] `make eval-smart PDF=data/input/peter_lou_50dpi.pdf` shows improvement
2. [ ] `make eval-smart PDF=data/input/peter_lou.pdf` F1 >= 99% (no regression)
3. [ ] All @fix_plan.md items marked [x]
