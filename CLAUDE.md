# Portadoc

PDF word extraction system for document redaction.

## Quick Start (Makefile)

Run `make help` to see all available commands. Common usage:

```bash
make check              # Check OCR engine availability
make extract-smart      # Extract with best config for degraded PDFs
make eval-smart         # Evaluate against ground truth
make peter-lou-eval     # Quick eval on test file

# Custom PDF:
make extract-smart PDF=path/to/your.pdf
make eval-smart PDF=path/to/your.pdf GROUND_TRUTH=path/to/truth.csv
```

## Running the CLI (Manual)

With venv (after running install script):
```bash
source .venv/bin/activate
portadoc extract data/input/peter_lou_50dpi.pdf
```

Without venv (fallback):
```bash
PYTHONPATH=src python3 -m portadoc.cli extract data/input/peter_lou_50dpi.pdf
```

## Evaluation Command

```bash
source .venv/bin/activate
portadoc eval data/input/peter_lou_50dpi.pdf data/input/peter_lou_words_slim.csv
```

## Directories

- `src/portadoc/` - Main source code
- `data/input/` - Test PDFs and ground truth CSVs
- `tmp/` - Temporary scripts and working files (not committed)
- `logs/` - Ralph loop logs

## tmp/ Directory

The `tmp/` directory is for temporary scripts, scratch files, and one-off utilities.
- `tmp/install_ocr_deps.sh` - Installs OCR dependencies (PaddleOCR, docTR, etc.)

## Key Files for OCR Pipeline

- `src/portadoc/extractor.py` - Main extraction pipeline
- `src/portadoc/preprocess.py` - OpenCV preprocessing (grayscale, denoise, CLAHE, sharpen, binarize)
- `src/portadoc/ocr/tesseract.py` - Tesseract wrapper
- `src/portadoc/ocr/easyocr.py` - EasyOCR wrapper
- `src/portadoc/harmonize.py` - Multi-engine result fusion
- `src/portadoc/metrics.py` - Evaluation metrics

## Current Best Configs

### For bbox accuracy (redaction):
```bash
portadoc extract --smart --no-easyocr --preprocess none --psm 6 <pdf>
# Or: make extract-smart PDF=<pdf>
```
- Clean PDFs: **99.00% F1**, 98.74% text match
- Degraded (50dpi): **81.55% F1**, 40.35% text match

### For text accuracy (OCR extraction):
```bash
portadoc extract --smart --use-paddleocr --use-doctr --preprocess none --psm 6 <pdf>
# Or: make extract-all PDF=<pdf>
```
- Clean PDFs: 90.29% F1, 98.25% text match
- Degraded (50dpi): 73.10% F1, **68.75% text match**

## Key CLI Flags

- `--smart` - Use smart harmonization with full tracking (recommended)
- `--no-easyocr` - Disable EasyOCR (better bbox precision)
- `--use-paddleocr` - Enable PaddleOCR (adds text accuracy)
- `--use-doctr` - Enable docTR (best text accuracy)
- `--preprocess none` - Skip preprocessing (optimal for already-degraded images)
- `--psm 6` - Tesseract uniform block mode (best for document text)
- `--config PATH` - Custom YAML config for harmonization thresholds

## Performance Notes

- **Trade-off**: More engines = better text accuracy but lower bbox precision
- Tesseract-only: Best F1 (fewest false positives)
- All 4 engines: Best text match (68.75% vs 40.35% on degraded docs)
- 50 DPI degraded PDFs: 81.55% F1 is the ceiling with Tesseract-only
- Clean PDFs: 99.00% F1 achievable with Tesseract-only

## OCR Engine Availability

All 4 engines enabled in `config/harmonize.yaml`:
- Tesseract (primary) - always used
- EasyOCR - enabled by default
- PaddleOCR - enabled, use `--use-paddleocr`
- docTR - enabled, use `--use-doctr`

Check with: `make check`

## Ralph Loop Instructions

**IMPORTANT: Always maintain the Makefile when making changes.**

When adding new CLI commands, features, or changing existing functionality:

1. **Update the Makefile** (`Makefile` in project root) to reflect the changes
2. Add new targets for any new CLI commands or common workflows
3. Update existing targets if command signatures change
4. Follow the existing pattern with `##` comments for the help system
5. Test with `make help` to verify the help output is correct

Example Makefile target format:
```makefile
##@ Category Name

target-name: ## Description of what this target does
	$(PORTADOC) command --flags $(VARIABLES)
```

**OpenCV Note**: This project requires `opencv-contrib-python` (not `opencv-python`) for super-resolution support. The install target handles this automatically.
