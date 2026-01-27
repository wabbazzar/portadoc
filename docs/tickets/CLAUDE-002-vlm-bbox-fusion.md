# [CLAUDE] VLM + BBox OCR Fusion

**Type:** CLAUDE (interactive session)
**Status:** DRAFT
**Created:** 2026-01-20
**Complexity:** High

## Problem Statement

Vision Language Models (VLMs) produce highly accurate text extraction but do not provide word-level bounding boxes. Traditional OCR engines (Tesseract, Surya, etc.) provide bounding boxes but may misspell or miss text, especially on degraded documents.

**Goal:** Combine VLM text accuracy with traditional OCR bbox precision by matching VLM output to OCR-detected word positions.

## Target VLMs

Specialized document OCR VLMs (no bbox output):
- **OLM-OCR** - Open-source document OCR
- **GOT** - General OCR Theory model
- **DOTS** - Document Text Spotter
- **DEEPSEEK-OCR** - DeepSeek's document model
- **Anthropic Claude** (fallback) - General VLM with vision

## Architecture Decision

This is a **new OCR engine** (`ocr/vlm_ocr.py`) with **custom harmonization logic** because:
- VLM provides **zero positional information** (unlike other engines which have at least line-level)
- Requires a bbox-providing OCR to have already run
- Acts as a "text refinement" layer on top of existing OCR results

## Algorithm Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: PDF Page Image                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │   Traditional OCR    │         │        VLM          │
        │  (Tesseract, etc.)   │         │   (OLM-OCR, etc.)   │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │  Words with BBoxes   │         │   Text only (flat)  │
        │  [{text, bbox}, ...]│         │   "The cat in..."   │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    ▼                               │
        ┌─────────────────────┐                     │
        │ Geometric Clustering │                     │
        │  (existing module)   │                     │
        └─────────────────────┘                     │
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │      Context-Based Matching      │
                    │  (within geometric clusters)     │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │   Matched Words      │         │   Orphan VLM Words  │
        │  (VLM text + bbox)   │         │   (no OCR match)    │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    │                               ▼
                    │               ┌─────────────────────────────────┐
                    │               │  Character Size Estimation       │
                    │               │  (from matched words per cluster)│
                    │               └─────────────────────────────────┘
                    │                               │
                    │                               ▼
                    │               ┌─────────────────────────────────┐
                    │               │  Pixel Detection + Placement     │
                    │               │  (detection.py + interpolation)  │
                    │               └─────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │         Final Output             │
                    │  All words with bbox + status    │
                    └─────────────────────────────────┘
```

## Detailed Algorithm

### Phase 1: Preparation

```python
def vlm_bbox_fusion(page_image, ocr_words: List[Word], vlm_text: str) -> List[Word]:
    """
    Fuse VLM text accuracy with OCR bounding boxes.

    Args:
        page_image: PIL Image of the page
        ocr_words: Words from traditional OCR with bboxes
        vlm_text: Raw text output from VLM (no positions)

    Returns:
        List of Words with refined text and bboxes
    """

    # 1. Tokenize VLM output into words
    vlm_words = tokenize_preserving_punctuation(vlm_text)
    # ["The", "cat", "in", "the", "hat", "did", "a", "flip", ...]

    # 2. Cluster OCR words geometrically (existing module)
    clusters = geometric_clustering(ocr_words)
    # Each cluster = contiguous text block (paragraph, column, etc.)

    # 3. Build reading-order sequence per cluster
    for cluster in clusters:
        cluster.words = sort_reading_order(cluster.words)  # top-to-bottom, left-to-right
```

### Phase 2: Context-Based Matching

The key insight: match words **within clusters** to avoid "gap traversal" problems where distant words accidentally match.

```python
def build_context_signature(words: List[str], index: int, window: int = 1) -> str:
    """
    Build minimal unique context for a word.

    Example: words = ["The", "cat", "in", "the", "hat"]
    - "The" at index 0: context "The cat" (expands right)
    - "the" at index 3: context "in the hat" (expands both)
    """
    start = max(0, index - window)
    end = min(len(words), index + window + 1)
    return " ".join(words[start:end])


def match_within_cluster(ocr_words: List[Word], vlm_words: List[str]) -> MatchResult:
    """
    Match VLM words to OCR words within a single geometric cluster.

    Strategy:
    1. Try exact match with minimal context
    2. Expand context window until unique
    3. Try fuzzy match (Levenshtein) if exact fails
    4. Track unmatched words on both sides
    """

    ocr_texts = [w.text for w in ocr_words]
    matched = {}          # vlm_idx -> ocr_idx
    ocr_used = set()      # track used OCR words

    # Pass 1: Exact matches with expanding context
    for vlm_idx, vlm_word in enumerate(vlm_words):
        for window in range(1, 5):  # expand context up to 4 words
            vlm_context = build_context_signature(vlm_words, vlm_idx, window)

            candidates = []
            for ocr_idx, ocr_word in enumerate(ocr_words):
                if ocr_idx in ocr_used:
                    continue
                ocr_context = build_context_signature(ocr_texts, ocr_idx, window)

                if fuzzy_context_match(vlm_context, ocr_context, threshold=0):
                    candidates.append(ocr_idx)

            if len(candidates) == 1:
                matched[vlm_idx] = candidates[0]
                ocr_used.add(candidates[0])
                break

    # Pass 2: Fuzzy matches for remaining (Levenshtein 1, 2, 3)
    for max_distance in [1, 2, 3]:
        for vlm_idx, vlm_word in enumerate(vlm_words):
            if vlm_idx in matched:
                continue

            for window in range(1, 5):
                vlm_context = build_context_signature(vlm_words, vlm_idx, window)

                best_match = None
                best_distance = max_distance + 1

                for ocr_idx, ocr_word in enumerate(ocr_words):
                    if ocr_idx in ocr_used:
                        continue
                    ocr_context = build_context_signature(ocr_texts, ocr_idx, window)

                    distance = levenshtein_distance(vlm_context, ocr_context)
                    if distance <= max_distance and distance < best_distance:
                        best_match = ocr_idx
                        best_distance = distance

                if best_match is not None:
                    matched[vlm_idx] = best_match
                    ocr_used.add(best_match)
                    break

    # Identify orphans
    vlm_orphans = [i for i in range(len(vlm_words)) if i not in matched]
    ocr_orphans = [i for i in range(len(ocr_words)) if i not in ocr_used]

    return MatchResult(matched, vlm_orphans, ocr_orphans)
```

### Phase 3: Character Size Estimation

Use matched words to estimate character dimensions per cluster.

```python
def estimate_char_sizes(matched_pairs: List[Tuple[str, BBox]]) -> CharSizeModel:
    """
    From matched (VLM_text, OCR_bbox) pairs, estimate character dimensions.

    Algorithm:
    1. Calculate char_width = bbox.width / len(text) for each match
    2. Calculate char_height = bbox.height for each match
    3. Cluster these measurements (fonts may vary within document)
    4. Return size model with clusters
    """

    measurements = []
    for vlm_text, bbox in matched_pairs:
        if len(vlm_text) == 0:
            continue
        char_width = bbox.width / len(vlm_text)
        char_height = bbox.height
        measurements.append(CharMeasurement(char_width, char_height, vlm_text))

    # Sort by char_height (proxy for font size)
    measurements.sort(key=lambda m: m.char_height)

    # Greedy clustering: group measurements within 2σ
    size_clusters = []
    current_cluster = [measurements[0]]

    for m in measurements[1:]:
        cluster_mean = mean([c.char_height for c in current_cluster])
        cluster_std = std([c.char_height for c in current_cluster]) or 1.0

        if abs(m.char_height - cluster_mean) <= 2 * cluster_std:
            current_cluster.append(m)
        else:
            size_clusters.append(CharSizeCluster(current_cluster))
            current_cluster = [m]

    if current_cluster:
        size_clusters.append(CharSizeCluster(current_cluster))

    return CharSizeModel(size_clusters)
```

### Phase 4: Orphan Placement

For VLM words with no OCR match, use pixel detection + neighbor interpolation.

```python
def place_orphan_segment(
    orphan_words: List[str],
    left_anchor: Optional[Word],    # last matched word before orphan
    right_anchor: Optional[Word],   # first matched word after orphan
    page_image: Image,
    char_size_model: CharSizeModel,
    pixel_detector: PixelDetector   # existing detection.py
) -> List[Word]:
    """
    Place orphan VLM words that had no OCR match.

    Strategy:
    1. Define search region between anchors (or edge of page/cluster)
    2. Run pixel detection to find text regions
    3. Estimate required width from char_size_model
    4. Assign words to detected regions
    5. If regions don't fit, try multi-line assignment
    """

    # 1. Define search region
    if left_anchor and right_anchor:
        # Region between two anchors
        search_region = BBox(
            x0=left_anchor.bbox.x1,
            y0=min(left_anchor.bbox.y0, right_anchor.bbox.y0),
            x1=right_anchor.bbox.x0,
            y1=max(left_anchor.bbox.y1, right_anchor.bbox.y1)
        )
    elif left_anchor:
        # Region after left anchor (to end of line/cluster)
        search_region = extend_to_cluster_boundary(left_anchor, direction="right")
    elif right_anchor:
        # Region before right anchor (from start of line/cluster)
        search_region = extend_to_cluster_boundary(right_anchor, direction="left")
    else:
        # No anchors - full cluster region
        search_region = cluster.bounds

    # 2. Pixel detection in search region
    detected_regions = pixel_detector.detect_text_regions(
        page_image.crop(search_region.as_tuple())
    )

    # 3. Estimate character sizes for this region
    # Use the size cluster closest to anchors
    if left_anchor:
        char_sizes = char_size_model.get_sizes_for_height(left_anchor.bbox.height)
    elif right_anchor:
        char_sizes = char_size_model.get_sizes_for_height(right_anchor.bbox.height)
    else:
        char_sizes = char_size_model.get_dominant_sizes()

    # 4. Try to fit orphan words into detected regions
    placed_words = []
    orphan_text = " ".join(orphan_words)
    estimated_width = len(orphan_text) * char_sizes.avg_char_width

    # Strategy A: Single region fits all
    for region in detected_regions:
        if region.width >= estimated_width * 0.8:  # 80% tolerance
            # Assign all orphan words to this region, distribute evenly
            words = distribute_words_in_region(orphan_words, region, char_sizes)
            placed_words.extend(words)
            break

    # Strategy B: Multi-region assignment (words span multiple lines)
    if not placed_words:
        # Sort regions in reading order
        regions_sorted = sort_reading_order(detected_regions)

        remaining_words = orphan_words[:]
        for region in regions_sorted:
            if not remaining_words:
                break

            # How many words fit in this region?
            fit_count = estimate_word_fit(remaining_words, region, char_sizes)
            if fit_count > 0:
                words_to_place = remaining_words[:fit_count]
                words = distribute_words_in_region(words_to_place, region, char_sizes)
                placed_words.extend(words)
                remaining_words = remaining_words[fit_count:]

        # Any remaining words get interpolated between anchors
        if remaining_words:
            words = interpolate_between_anchors(
                remaining_words, left_anchor, right_anchor, char_sizes
            )
            for w in words:
                w.status = "vlm_interpolated"
            placed_words.extend(words)

    # Mark all placed words with appropriate status
    for word in placed_words:
        if word.status != "vlm_interpolated":
            word.status = "vlm_pixel_placed"
        word.confidence = calculate_placement_confidence(word, char_sizes)

    return placed_words


def distribute_words_in_region(
    words: List[str],
    region: BBox,
    char_sizes: CharSizes
) -> List[Word]:
    """
    Distribute words evenly within a detected region.
    """
    total_chars = sum(len(w) for w in words)
    total_spaces = len(words) - 1

    # Estimate space width as 0.3 * char_width
    space_width = char_sizes.avg_char_width * 0.3
    text_width = total_chars * char_sizes.avg_char_width
    total_width = text_width + (total_spaces * space_width)

    # Scale factor if region is smaller/larger than estimate
    scale = region.width / total_width if total_width > 0 else 1.0

    placed = []
    current_x = region.x0

    for word_text in words:
        word_width = len(word_text) * char_sizes.avg_char_width * scale

        word = Word(
            text=word_text,
            bbox=BBox(
                x0=current_x,
                y0=region.y0,
                x1=current_x + word_width,
                y1=region.y1
            ),
            source="vlm",
            status="vlm_pixel_placed"
        )
        placed.append(word)

        current_x += word_width + (space_width * scale)

    return placed
```

### Phase 5: Final Assembly

```python
def assemble_results(
    clusters: List[Cluster],
    all_matched: Dict[int, Tuple[str, Word]],  # vlm_idx -> (vlm_text, ocr_word)
    all_placed_orphans: List[Word]
) -> List[Word]:
    """
    Combine matched and placed words into final output.
    """

    final_words = []

    # Add matched words (VLM text + OCR bbox)
    for vlm_idx, (vlm_text, ocr_word) in all_matched.items():
        word = Word(
            text=vlm_text,                    # Use VLM text (more accurate)
            bbox=ocr_word.bbox,               # Use OCR bbox (positioned)
            source=f"vlm+{ocr_word.source}",  # Track provenance
            status="vlm_matched",
            confidence=1.0 if vlm_text == ocr_word.text else 0.9
        )
        final_words.append(word)

    # Add placed orphans
    final_words.extend(all_placed_orphans)

    # Sort by reading order
    final_words = sort_reading_order(final_words)

    # Assign word_ids
    for i, word in enumerate(final_words):
        word.word_id = i

    return final_words
```

## Output Status Values

| Status | Meaning |
|--------|---------|
| `vlm_matched` | VLM word matched to OCR bbox (high confidence) |
| `vlm_pixel_placed` | VLM word placed via pixel detection (medium confidence) |
| `vlm_interpolated` | VLM word placed via neighbor interpolation (lower confidence) |
| `ocr_only` | OCR word with no VLM match (keep original) |

## Configuration

Add to `config/harmonize.yaml`:

```yaml
vlm:
  enabled: false
  provider: "olm-ocr"  # olm-ocr, got, dots, deepseek-ocr, anthropic

  # API configuration (provider-specific)
  api_key_env: "VLM_API_KEY"
  model: "default"

  # Matching parameters
  matching:
    max_context_window: 4          # Max words of context for matching
    max_levenshtein_distance: 3    # Max edit distance for fuzzy match
    context_match_threshold: 0.8   # Similarity threshold for context

  # Character size estimation
  char_sizing:
    clustering_std_multiple: 2.0   # σ multiple for size clustering
    min_samples_per_cluster: 3     # Minimum matches to form size cluster

  # Orphan placement
  placement:
    width_tolerance: 0.8           # Accept region if 80% of estimated width
    space_width_ratio: 0.3         # Space width as ratio of char width
    use_pixel_detection: true      # Use detection.py for orphan regions
    use_interpolation: true        # Fall back to neighbor interpolation
```

## File Changes

| File | Change |
|------|--------|
| `src/portadoc/ocr/vlm_ocr.py` | **NEW** - VLM integration and fusion logic |
| `src/portadoc/ocr/__init__.py` | Export VLM module |
| `src/portadoc/config.py` | Add VLM config schema |
| `src/portadoc/models.py` | Add new status values, CharSizeModel |
| `src/portadoc/extractor.py` | Integrate VLM as optional engine |
| `src/portadoc/cli.py` | Add `--vlm` flag |
| `config/harmonize.yaml` | Add VLM configuration section |

## Test Cases

### Test 1: Basic Matching
```
OCR:  "The crt in thr hat"
VLM:  "The cat in the hat"
Expected: All 5 words matched, VLM text adopted
```

### Test 2: Orphan Placement
```
OCR:  "The hat" (middle words missing)
VLM:  "The cat in the hat"
Expected: "cat in the" placed via pixel detection between "The" and "hat"
```

### Test 3: Multi-line Orphan
```
OCR:  "Hello" ... "world"
VLM:  "Hello beautiful wonderful world"
Expected: "beautiful wonderful" split across detected text regions
```

### Test 4: Font Size Variation
```
Document with 40pt header, 12pt body
Expected: Separate char size clusters, correct width estimation for each
```

### Test 5: No OCR Match at All
```
OCR:  (completely misses a paragraph)
VLM:  "Hidden paragraph text here"
Expected: Entire paragraph placed via pixel detection with vlm_interpolated status
```

## Open Questions

1. **VLM prompt engineering**: What prompt produces cleanest text-only output from each VLM?
2. **Multi-column handling**: Should geometric clustering handle columns automatically, or do we need explicit column detection?
3. **Table cells**: How to match VLM text to table cell bboxes?
4. **Performance**: VLM API latency - batch pages or stream?
5. **Cost**: API costs per page for each provider?

## Success Criteria

- [ ] VLM text accuracy adopted for matched words
- [ ] Orphan words placed with reasonable bboxes
- [ ] Per-cluster character size estimation working
- [ ] All layout types handled (single column, multi-column, mixed)
- [ ] Status field correctly indicates confidence level
- [ ] Configurable via harmonize.yaml
- [ ] Works with any bbox-providing OCR as base

## Sub-tickets

| Ticket | Status | Description |
|--------|--------|-------------|
| CLAUDE-002a-vlm-fusion-tdd.md | COMPLETE | TDD implementation with cat_in_hat test |
| CLAUDE-002b-orphan-wrap-and-got.md | TODO | Orphan line wrapping + GOT integration |

## References

- Existing geometric clustering: `src/portadoc/geometric_clustering.py`
- Existing pixel detection: `src/portadoc/detection.py`
- VLM fusion implementation: `src/portadoc/ocr/vlm_fusion.py`
- VLM providers:
  - OLM-OCR: https://github.com/allenai/OLM-OCR
  - GOT: https://github.com/Ucas-HaoranWei/GOT-OCR2.0
  - DOTS: TBD
  - DeepSeek-OCR: TBD
