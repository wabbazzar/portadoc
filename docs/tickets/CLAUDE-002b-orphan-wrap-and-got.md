# [CLAUDE] Orphan Word Wrapping + GOT Integration

**Type:** CLAUDE (interactive session)
**Status:** TODO
**Created:** 2026-01-20
**Parent:** CLAUDE-002-vlm-bbox-fusion.md
**Depends on:** CLAUDE-002a-vlm-fusion-tdd.md (COMPLETE)

## Objective

Two improvements to the VLM fusion algorithm:

1. **Better orphan placement fallback** - When pixel detection returns no viable regions, intelligently place orphan words with proper line wrapping
2. **GOT integration** - Add General OCR Theory (GOT) as the first real VLM provider

---

## Part 1: Orphan Word Wrapping

### Problem

Currently when pixel detection fails, orphan words are interpolated but stack at the same x-position:

```
5: "did" [vlm_interpolated] bbox=(360,51,394,81)
6: "a" [vlm_interpolated] bbox=(360,51,372,81)      # Same x0!
7: "flip" [vlm_interpolated] bbox=(360,51,406,81)   # Same x0!
...
```

### Solution

When pixel detection returns no regions, use intelligent text wrapping:

1. **Start position**: x0 = left_anchor.x1 + space_width, y0 = left_anchor.y0
2. **Sequential placement**: Each word starts after previous word ends
3. **Line wrapping**: When approaching cluster right boundary, wrap to next line
4. **Cluster boundaries**: Use geometric clustering to determine wrap points

### Algorithm

```python
def place_orphan_segment_with_wrap(
    orphan_words: List[str],
    left_anchor: HarmonizedWord,
    right_anchor: Optional[HarmonizedWord],
    char_sizes: CharSizes,
    cluster: Cluster,  # For boundary detection
) -> List[BBox]:
    """
    Place orphan words sequentially with line wrapping.

    Args:
        orphan_words: List of orphan word texts in reading order
        left_anchor: Last matched word before orphan segment
        right_anchor: First matched word after orphan segment (may be on different line)
        char_sizes: Character size estimates for this cluster
        cluster: Geometric cluster for boundary detection

    Returns:
        List of BBox for each orphan word
    """
    bboxes = []

    # Get cluster boundaries
    cluster_x0 = cluster.bounding_box.x0 if cluster.bounding_box else 50
    cluster_x1 = cluster.bounding_box.x1 if cluster.bounding_box else 550

    # Estimate space width (typically 0.3 * char_width)
    space_width = char_sizes.avg_char_width * 0.3

    # Start position
    current_x = left_anchor.bbox.x1 + space_width
    current_y0 = left_anchor.bbox.y0
    current_y1 = left_anchor.bbox.y1
    line_height = char_sizes.avg_char_height

    for word in orphan_words:
        word_width = len(word) * char_sizes.avg_char_width

        # Check if word fits on current line
        if current_x + word_width > cluster_x1 - 10:  # 10pt margin
            # Wrap to next line
            current_x = cluster_x0 + 10  # Indent from left edge
            current_y0 += line_height + 5  # Line spacing
            current_y1 += line_height + 5

        # Place word
        bbox = BBox(
            x0=current_x,
            y0=current_y0,
            x1=current_x + word_width,
            y1=current_y1,
        )
        bboxes.append(bbox)

        # Move to next position
        current_x = bbox.x1 + space_width

    return bboxes
```

### Changes Required

| File | Change |
|------|--------|
| `src/portadoc/ocr/vlm_fusion.py` | Add `place_orphan_segment_with_wrap()` function |
| `src/portadoc/ocr/vlm_fusion.py` | Update `fuse_vlm_with_ocr()` to use new placement |
| `tests/test_vlm_fusion.py` | Add tests for line wrapping behavior |

### Test Cases

1. **Single line fit**: Orphan words fit on same line as left anchor
2. **Wrap to next line**: Orphan words exceed line width, wrap correctly
3. **Multiple wraps**: Long orphan segment wraps multiple times
4. **Cross-cluster**: Orphan segment spans between two clusters on different lines

---

## Part 2: GOT Integration

### Overview

GOT (General OCR Theory) is a document-focused VLM that produces high-quality text extraction. We'll integrate it as the first real VLM provider.

**Repository**: https://github.com/Ucas-HaoranWei/GOT-OCR2.0

### Architecture

```
src/portadoc/ocr/
├── vlm_fusion.py      # Core fusion algorithm (existing)
├── vlm_providers/     # NEW: VLM provider implementations
│   ├── __init__.py
│   ├── base.py        # Abstract VLM provider interface
│   └── got.py         # GOT implementation
```

### Provider Interface

```python
# src/portadoc/ocr/vlm_providers/base.py

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image

class VLMProvider(ABC):
    """Abstract base class for VLM providers."""

    @abstractmethod
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from a page image.

        Args:
            image: PIL Image of the document page

        Returns:
            Extracted text (no positional information)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/status."""
        pass

    @property
    def requires_gpu(self) -> bool:
        """Whether this provider requires GPU."""
        return False
```

### GOT Implementation

```python
# src/portadoc/ocr/vlm_providers/got.py

from PIL import Image
from .base import VLMProvider

class GOTProvider(VLMProvider):
    """GOT-OCR2.0 VLM provider."""

    def __init__(self, model_path: str = "models/GOT"):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the GOT model."""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map='cuda' if torch.cuda.is_available() else 'cpu',
            )
            self._model.eval()

    def extract_text(self, image: Image.Image) -> str:
        """Extract text using GOT."""
        self._load_model()

        # GOT expects specific input format
        result = self._model.chat(
            self._tokenizer,
            image,
            ocr_type='ocr',  # Plain OCR mode
        )

        return result

    @property
    def name(self) -> str:
        return "GOT"

    @property
    def requires_gpu(self) -> bool:
        return True  # GOT benefits significantly from GPU
```

### CLI Integration

```bash
# New CLI flag
./portadoc extract input.pdf --vlm got

# With specific model path
./portadoc extract input.pdf --vlm got --vlm-model models/GOT-OCR2.0
```

### Config Integration

```yaml
# config/harmonize.yaml
vlm:
  enabled: false
  provider: "got"  # got, olm-ocr, anthropic, etc.

  providers:
    got:
      model_path: "models/GOT"
      # Or HuggingFace model ID
      # model_path: "stepfun-ai/GOT-OCR2_0"
```

### Changes Required

| File | Change |
|------|--------|
| `src/portadoc/ocr/vlm_providers/__init__.py` | NEW: Provider registry |
| `src/portadoc/ocr/vlm_providers/base.py` | NEW: Abstract interface |
| `src/portadoc/ocr/vlm_providers/got.py` | NEW: GOT implementation |
| `src/portadoc/cli.py` | Add `--vlm` and `--vlm-model` flags |
| `src/portadoc/config.py` | Add VLM config schema |
| `src/portadoc/extractor.py` | Wire VLM into extraction pipeline |
| `config/harmonize.yaml` | Add VLM section |

### Testing

1. **Unit tests**: Mock GOT output, test fusion
2. **Integration test**: If GOT model available locally, run end-to-end
3. **Test with cat_in_hat.pdf**: Compare GOT output vs Tesseract

---

## Success Criteria

### Part 1: Orphan Wrapping
- [ ] Orphan words placed sequentially (not stacked)
- [ ] Line wrapping at cluster boundary
- [ ] Proper line spacing on wrap
- [ ] Tests for single-line, wrap, and multi-wrap scenarios

### Part 2: GOT Integration
- [ ] VLM provider interface defined
- [ ] GOT provider implemented
- [ ] CLI `--vlm got` flag working
- [ ] Config file support
- [ ] Graceful fallback when GOT not available
- [ ] Integration test with real GOT (if model available)

---

## Notes

- GOT model is ~4GB, requires download or local path
- Check if GOT is already in `models/` directory
- CPU inference is slow but possible
- Consider adding `--vlm-device cpu|cuda` flag
