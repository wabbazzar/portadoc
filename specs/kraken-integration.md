# Kraken OCR Integration Technical Specification

## Overview

Integrate Kraken OCR as an additional engine in Portadoc's multi-engine pipeline. Kraken excels at historical and degraded documents with trainable segmentation.

## Kraken Basics

### Installation

```bash
pip install kraken[pdf]
```

### Model Management

```bash
# List available models
kraken list

# Download a model (example: McCATMuS)
kraken get 10.5281/zenodo.13788177

# Models are stored in ~/.local/share/kraken/
```

### Recommended Models for English

| Model | DOI | Description |
|-------|-----|-------------|
| McCATMuS | 10.5281/zenodo.13788177 | Multi-language (EN,FR,DE,ES,IT,LA,OC), 16th-21st century |
| CATMuS-Print | Check `kraken list` | Multi-language printed text |

## Python API

### Core Recognition Flow

```python
from PIL import Image
from kraken import blla, rpred
from kraken.lib import models, vgsl

# 1. Load recognition model
rec_model = models.load_any('/path/to/model.mlmodel')

# 2. Load segmentation model (optional - uses default if not specified)
seg_model = vgsl.TorchVGSLModel.load_model('/path/to/blla.mlmodel')

# 3. Load image
image = Image.fromarray(numpy_image)

# 4. Segment the page (get line regions)
segmentation = blla.segment(image, model=seg_model)

# 5. Recognize text
for record in rpred.rpred(rec_model, image, segmentation):
    text = record.prediction        # Full line text
    cuts = record.cuts              # Character positions [(x0, x1), ...]
    confs = record.confidences      # Per-character confidence [0-1]
    line_bbox = record.line         # Line bounding box
    baseline = record.baseline      # Baseline coordinates (if baseline seg)
```

### Record Object Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `prediction` | str | Recognized text for the line |
| `cuts` | list[tuple[int,int]] | Character x-coordinates (start, end) |
| `confidences` | list[float] | Per-character confidence (0-1) |
| `line` | tuple | Line bounding box (x0, y0, x1, y1) |
| `baseline` | list | Baseline polyline coordinates |
| `type` | str | 'baselines' or 'box' |

## Word-Level BBox Extraction Algorithm

Kraken provides character-level cuts, not word-level. We must derive word bboxes:

```python
def extract_words_from_record(record, line_bbox) -> list[tuple[str, tuple]]:
    """
    Extract word-level bboxes from Kraken's character cuts.

    Args:
        record: Kraken recognition record
        line_bbox: (x0, y0, x1, y1) of the line

    Returns:
        List of (word_text, (x0, y0, x1, y1)) tuples
    """
    text = record.prediction
    cuts = record.cuts  # [(char_x0, char_x1), ...]
    y0, y1 = line_bbox[1], line_bbox[3]  # Use line's y-coordinates

    words = []
    current_word = ""
    word_start_x = None
    word_end_x = None

    for i, char in enumerate(text):
        if i < len(cuts):
            char_x0, char_x1 = cuts[i]
        else:
            # Fallback if cuts don't match text length
            continue

        if char.isspace():
            # End of word
            if current_word and word_start_x is not None:
                words.append((
                    current_word,
                    (word_start_x, y0, word_end_x, y1)
                ))
            current_word = ""
            word_start_x = None
            word_end_x = None
        else:
            # Continue word
            if word_start_x is None:
                word_start_x = char_x0
            word_end_x = char_x1
            current_word += char

    # Don't forget the last word
    if current_word and word_start_x is not None:
        words.append((
            current_word,
            (word_start_x, y0, word_end_x, y1)
        ))

    return words
```

## Portadoc Integration

### New File: `src/portadoc/ocr/kraken_ocr.py`

```python
"""Kraken OCR wrapper for word-level extraction."""

from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image

from ..models import BBox, Word

# Lazy imports
_rec_model = None
_seg_model = None

def _get_models(model_path: Optional[str] = None):
    """Load or return cached Kraken models."""
    global _rec_model, _seg_model

    if _rec_model is None:
        from kraken.lib import models, vgsl

        # Use default model path or provided one
        if model_path is None:
            # Try to find a model in standard locations
            default_paths = [
                Path.home() / '.local/share/kraken',
                Path('models/kraken'),
            ]
            # Find first .mlmodel file
            for p in default_paths:
                if p.exists():
                    mlmodels = list(p.glob('*.mlmodel'))
                    if mlmodels:
                        model_path = str(mlmodels[0])
                        break

        if model_path is None:
            raise RuntimeError("No Kraken model found. Run: kraken get <model_doi>")

        _rec_model = models.load_any(model_path)
        # Use default segmentation model (blla)
        _seg_model = None  # Will use default

    return _rec_model, _seg_model


def extract_words_kraken(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    model_path: Optional[str] = None,
) -> list[Word]:
    """
    Extract words from an image using Kraken OCR.

    Args:
        image: RGB image as numpy array
        page_num: Page number for word IDs
        page_width: Page width in PDF points
        page_height: Page height in PDF points
        model_path: Path to Kraken model file (optional)

    Returns:
        List of Word objects with bounding boxes in PDF coordinates
    """
    from kraken import blla, rpred

    rec_model, seg_model = _get_models(model_path)

    img_height, img_width = image.shape[:2]
    pil_image = Image.fromarray(image)

    # Segment the page
    if seg_model:
        segmentation = blla.segment(pil_image, model=seg_model)
    else:
        segmentation = blla.segment(pil_image)

    # Scale factors for coordinate conversion
    scale_x = page_width / img_width
    scale_y = page_height / img_height

    words = []

    # Recognize each line
    for record in rpred.rpred(rec_model, pil_image, segmentation):
        text = record.prediction
        cuts = record.cuts
        line_bbox = record.line  # (x0, y0, x1, y1) in image coords

        # Calculate average confidence for the line
        line_conf = sum(record.confidences) / len(record.confidences) * 100 if record.confidences else 0

        # Extract words from character cuts
        y0, y1 = line_bbox[1], line_bbox[3]

        current_word = ""
        word_start_x = None
        word_end_x = None
        word_confs = []

        for i, char in enumerate(text):
            if i < len(cuts):
                char_x0, char_x1 = cuts[i]
                char_conf = record.confidences[i] if i < len(record.confidences) else 0
            else:
                continue

            if char.isspace():
                if current_word and word_start_x is not None:
                    # Create Word object
                    avg_conf = sum(word_confs) / len(word_confs) * 100 if word_confs else line_conf

                    word = Word(
                        word_id=-1,
                        text=current_word,
                        bbox=BBox(
                            x0=word_start_x * scale_x,
                            y0=y0 * scale_y,
                            x1=word_end_x * scale_x,
                            y1=y1 * scale_y,
                        ),
                        page=page_num,
                        engine="kraken",
                        confidence=avg_conf,
                    )
                    words.append(word)

                current_word = ""
                word_start_x = None
                word_end_x = None
                word_confs = []
            else:
                if word_start_x is None:
                    word_start_x = char_x0
                word_end_x = char_x1
                current_word += char
                word_confs.append(char_conf)

        # Last word in line
        if current_word and word_start_x is not None:
            avg_conf = sum(word_confs) / len(word_confs) * 100 if word_confs else line_conf

            word = Word(
                word_id=-1,
                text=current_word,
                bbox=BBox(
                    x0=word_start_x * scale_x,
                    y0=y0 * scale_y,
                    x1=word_end_x * scale_x,
                    y1=y1 * scale_y,
                ),
                page=page_num,
                engine="kraken",
                confidence=avg_conf,
            )
            words.append(word)

    return words


def is_kraken_available() -> bool:
    """Check if Kraken is installed."""
    try:
        import kraken
        return True
    except ImportError:
        return False


def get_kraken_version() -> Optional[str]:
    """Get Kraken version string, or None if not available."""
    try:
        from importlib.metadata import version
        return version("kraken")
    except Exception:
        return None
```

### Config Schema Addition

Add to `config/harmonize.yaml`:

```yaml
harmonize:
  secondary:
    engines:
      kraken:
        enabled: false          # Disabled by default
        weight: 1.0             # Adjust after testing
        garbage_penalty: 0.1
        bbox_type: word

ocr:
  kraken:
    model_path: null            # Path to recognition model
    seg_model_path: null        # Path to segmentation model (optional)
```

### Model Updates

Add to `HarmonizedWord` in `src/portadoc/models.py`:

```python
kraken_text: str = ""
dist_kraken: int = -1
```

### Harmonize Updates

In `src/portadoc/harmonize.py`, add to `ENGINE_CODES`:

```python
ENGINE_CODES = {
    "tesseract": "T",
    "easyocr": "E",
    "doctr": "D",
    "paddleocr": "P",
    "surya": "S",
    "kraken": "K",  # Add this
}
```

### Extractor Updates

In `src/portadoc/extractor.py`:

1. Add import:
```python
from .ocr.kraken_ocr import extract_words_kraken, is_kraken_available
```

2. Add parameter:
```python
use_kraken: bool = False,
```

3. Add availability check:
```python
kraken_ok = is_kraken_available() if use_kraken else False
```

4. Add to extraction:
```python
if kraken_ok:
    all_engine_results["kraken"] = extract_words_kraken(
        ocr_image, page_num, page_width, page_height
    )
```

### CLI Updates

In `src/portadoc/cli.py`, add flags:

```python
@click.option("--use-kraken/--no-kraken", default=False, help="Use Kraken OCR")
```

## Testing Strategy

### Unit Tests

Create `tests/test_kraken_ocr.py`:

```python
import pytest
from portadoc.ocr.kraken_ocr import is_kraken_available, extract_words_kraken

@pytest.mark.skipif(not is_kraken_available(), reason="Kraken not installed")
def test_kraken_available():
    assert is_kraken_available()

@pytest.mark.skipif(not is_kraken_available(), reason="Kraken not installed")
def test_extract_words_basic(sample_image):
    words = extract_words_kraken(sample_image, 1, 612, 792)
    assert isinstance(words, list)
    # Add more assertions based on expected output
```

### Integration Tests

Run full pipeline eval on test documents:

```bash
# With Kraken enabled
./portadoc eval data/input/peter_lou.pdf data/input/peter_lou_words_slim.csv --use-kraken

# Verify F1 >= baseline
```

## Validation Criteria

| Metric | Requirement |
|--------|-------------|
| peter_lou.pdf F1 | >= 97.06% (current baseline) |
| peter_lou_50dpi.pdf F1 | >= baseline (TBD) |
| Word extraction | No crashes on test files |
| Bbox precision | Visual inspection passes |
| Reading order | Geometric clustering works |

## Future Improvements

1. **Custom model training** - Fine-tune on specific document types
2. **Segmentation model** - Use custom trained BLLA model
3. **GPU acceleration** - Kraken supports CUDA
4. **Model caching** - Pre-load models for batch processing
