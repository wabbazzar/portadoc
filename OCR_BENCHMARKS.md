# OCR Engine Benchmarks
**Date**: 2026-01-17

## Surya / docTR version fix (2026-05-26)

**Symptoms**
- Default `portadoc extract` aborted with
  `Unexpected error: 'SuryaDecoderConfig' object has no attribute 'pad_token_id'`
  (Surya is in the default engine set, so *every* document failed).
- `portadoc check` reported `docTR: NOT FOUND` even though `python-doctr` was installed.

**Root cause — transformers/huggingface_hub version skew.** The environment had
`transformers==5.9.0` and `huggingface-hub==1.15.0`:
- **Surya** (`surya-ocr 0.17.0`): transformers 5.x changed `PretrainedConfig`
  attribute access, so `SuryaDecoderConfig` no longer exposes `pad_token_id` and
  `SuryaDecoderModel.__init__` raised `AttributeError`. surya-ocr 0.17.0 needs
  transformers **4.x** (`>=4.56.1`).
- **docTR** (`python-doctr 1.0.0`): imports `Repository` from `huggingface_hub`,
  which was **removed in huggingface_hub 1.0**, so the import failed and
  `is_doctr_available()` returned False. docTR needs huggingface-hub **<1.0**.

**Fix — pin transformers<5, which drags huggingface-hub back below 1.0.**
`transformers 4.57.x` already requires `huggingface-hub<1.0,>=0.34.0`, so a single
cap fixes both engines:

```bash
pip install 'transformers>=4.56.1,<5.0.0'   # resolved: transformers 4.57.6, huggingface-hub 0.36.2
```

Working CPU-only matrix (pinned in `requirements.txt` / `pyproject.toml [ocr]`):

| Package          | Version  | Constraint            |
|------------------|----------|-----------------------|
| transformers     | 4.57.6   | `>=4.56.1,<5.0.0`     |
| huggingface-hub  | 0.36.2   | `>=0.34.0,<1.0.0`     |
| surya-ocr        | 0.17.0   | `==0.17.0`            |
| python-doctr     | 1.0.0    | `==1.0.0` (`[torch]`) |
| torch            | 2.9.0    | `>=2.7.0,<3.0.0` CPU  |
| tokenizers       | 0.22.2   | (transformers dep)    |

**Resilience added alongside the fix:** each engine now runs in isolation inside
`extract_words()` — if one engine raises at runtime it logs a warning and
harmonization proceeds over the survivors (regression test:
`tests/test_engine_resilience.py`). `portadoc check` now *runs* each engine on a
tiny probe image, so "OK" means "actually produced output" rather than "imports".

**Multi-engine RAM (measured, CPU, single doc, peak RSS, single worker):**

| Config                                            | Peak RSS | Wall time |
|---------------------------------------------------|----------|-----------|
| Tesseract-only (`--no-easyocr --no-paddleocr --no-doctr --no-surya`) | ~0.35 GB | ~2 s/page |
| Full default (tesseract+easyocr+paddleocr+docTR+surya)               | ~9.0 GB  | ~2.5 min/page (Surya recognition dominates) |

Numbers from `EFTA01583993.pdf` (single page) via `/usr/bin/time -v`. The full
set loads torch + tensorflow + four model stacks, so budget **~9–10 GB/worker**.
On the 60 GB / 32-core box that means ~5 parallel single-doc workers with OS
headroom — do **not** assume 1 worker/core. For high-throughput batch redaction
where bbox precision matters more than text accuracy, the Tesseract-only path is
~25× faster and ~25× lighter.

---

## Bug Fix: Multi-Engine Harmonization

**Problem**: Config file had `doctr.enabled: false` and `paddleocr.enabled: false`, causing CLI flags `--use-paddleocr` and `--use-doctr` to run the engines but the harmonizer would skip their output.

**Fix**: Updated `config/harmonize.yaml` to enable all engines.

---

## Benchmark Results

### Clean PDF (peter_lou.pdf) - 401 ground truth words

| Configuration        | Precision | Recall | F1 Score | Text Match |
|---------------------|-----------|--------|----------|------------|
| Tesseract only      | 99.00%    | 99.00% | **99.00%** | 98.74%   |
| Tess + EasyOCR      | 88.69%    | 99.75% | 93.90%   | 98.75%     |
| All 4 engines       | 82.47%    | 99.75% | 90.29%   | 98.25%     |

**Winner**: Tesseract-only (99% F1)

### Degraded PDF (peter_lou_50dpi.pdf)

| Configuration        | Precision | Recall | F1 Score | Text Match |
|---------------------|-----------|--------|----------|------------|
| Tesseract only      | 77.11%    | 86.53% | **81.55%** | 40.35%   |
| Tess + EasyOCR      | 68.76%    | 87.28% | 76.92%   | 40.29%     |
| All 4 engines       | 62.63%    | 87.78% | 73.10%   | **68.75%** |

**F1 Winner**: Tesseract-only (81.55%)
**Text Match Winner**: All 4 engines (68.75%)

---

## Key Findings

1. **More engines = lower precision** - Secondary engines add false positive detections
2. **Text accuracy improves with more engines** - 68.75% vs 40.35% on degraded docs
3. **Trade-off**: Bbox accuracy vs text accuracy

## Recommended Usage

| Use Case | Command |
|----------|---------|
| Redaction (bbox accuracy) | `--no-easyocr --preprocess none --psm 6` |
| Text extraction (text accuracy) | `--use-paddleocr --use-doctr` |

## Per-Engine Word Counts (Clean PDF, Page 0)

| Engine | Words |
|--------|-------|
| Tesseract | 90 |
| EasyOCR | 95 |
| PaddleOCR | 94 |
| docTR | 92 |

All engines now produce similar word counts when working correctly.
