# Browser Client Specification

## Overview

A fully client-side version of Portadoc that runs OCR entirely in the browser using Tesseract.js and docTR-TFJS.

## Goals

1. **Offline-capable**: No server required after initial model download
2. **Privacy-first**: Documents never leave the user's device
3. **Feature parity**: Same OCR quality as Python version (≥80% word match)
4. **Reading order**: Correct document reading sequence via geometric clustering

## Non-Goals

- Mobile optimization (works but not optimized)
- PWA/Service Worker (future enhancement)
- Framework usage (vanilla TypeScript only)
- Surya/EasyOCR/PaddleOCR support (no browser ports exist)

## Architecture

### Pipeline

```
User uploads PDF
       ↓
   PDF.js renders to canvas
       ↓
   ┌───┴───┐
   ↓       ↓
Tesseract  docTR
   ↓       ↓
   └───┬───┘
       ↓
Geometric Clustering (reading order)
       ↓
Harmonization (multi-engine fusion)
       ↓
Display + Export (CSV/JSON)
```

### File Structure

```
src/portadoc/browser/
├── index.html              # Single page app
├── app.ts                  # Main coordinator
├── pdf-loader.ts           # PDF.js integration
├── ocr/
│   ├── tesseract.ts        # Tesseract.js wrapper
│   └── doctr.ts            # docTR-TFJS wrapper
├── geometric-clustering.ts # Reading order (port from Python)
├── harmonize.ts            # Multi-engine fusion
├── models.ts               # TypeScript interfaces
└── styles.css              # UI styling
```

## OCR Engines

### Tesseract.js
- **Source**: https://github.com/naptha/tesseract.js
- **Output**: Word-level bounding boxes
- **Model size**: ~10MB (eng language data)
- **Performance**: ~2-5s per page

### docTR-TFJS
- **Source**: https://github.com/mindee/doctr-tfjs-demo
- **Detection model**: db_mobilenet_v2 (~20MB)
- **Recognition model**: crnn_vgg16_bn (~30MB)
- **Output**: Word-level bounding boxes
- **Performance**: ~2-5s per page

## Reading Order Algorithm

The reading order is determined by a geometric clustering algorithm ported from `src/portadoc/geometric_clustering.py`.

### Algorithm Steps

1. **Calculate distance thresholds**: Use Q1 * 1.5 of inter-word distances
2. **Detect column boundaries**: Row-based gap analysis
3. **Build clusters**: Union-Find based on spatial proximity
4. **Group into row bands**: Clusters with overlapping y-ranges
5. **Order clusters**: Left-to-right within bands, top-to-bottom across bands
6. **Sort within clusters**: Y-fuzz row grouping, then left-to-right

### Key Functions to Port

| Python Function | Purpose |
|-----------------|---------|
| `calculate_distance_thresholds()` | Q1-based threshold calculation |
| `detect_column_boundaries()` | Row-based gap analysis |
| `build_clusters()` | Spatial proximity clustering with Union-Find |
| `sort_words_within_cluster()` | Y-fuzz row grouping |
| `group_clusters_into_row_bands()` | Cluster ordering |
| `order_words_by_reading()` | Main entry point |

## Data Models

### Word Interface
```typescript
interface Word {
  word_id: number;
  page: number;
  text: string;
  bbox: BBox;
  engine: 'tesseract' | 'doctr';
  confidence: number;
}

interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}
```

### Cluster Interface
```typescript
interface Cluster {
  words: Word[];
  centroid: { x: number; y: number };
  boundingBox: BBox;
}
```

## UI Requirements

### Layout
```
┌─────────────────────────────────────────────────────┐
│  [Upload PDF]  [Engine: ▼]  [Extract]  [Export ▼]   │
├─────────────────────────────────┬───────────────────┤
│                                 │                   │
│     PDF Preview                 │   Word List       │
│     with bbox overlays          │   (scrollable)    │
│                                 │                   │
│  [◀ Page 1/3 ▶]                 │                   │
│                                 │                   │
├─────────────────────────────────┴───────────────────┤
│  Status: Loading Tesseract... 45%                   │
└─────────────────────────────────────────────────────┘
```

### Features
- PDF upload (drag-drop or file picker)
- Page navigation for multi-page PDFs
- Engine selector (Tesseract / docTR / Both)
- Bounding box overlay on PDF preview
- Word list with text and coordinates
- CSV export (page, word_id, text, x0, y0, x1, y1, engine)
- JSON export
- Progress indicators for model loading and OCR

## Validation

### Benchmark: peter_lou.pdf (clean)
| Metric | Target |
|--------|--------|
| Tesseract.js word match | ≥80% (≥320 of 401) |
| docTR-TFJS word match | ≥80% (≥320 of 401) |
| Reading order | Matches ground truth sequence |

### Stress test: peter_lou_50dpi.pdf (degraded)
| Metric | Target |
|--------|--------|
| Processing | Completes without crash |
| Words extracted | Any count (accuracy not required) |

### Ground Truth Reference
File: `data/input/peter_lou_words_slim.csv`

First 12 words in correct reading order:
```
0: "7/24/25,"
1: "10:28"
2: "AM"
3: "Patient"
4: "Intake"
5: "Summary"
6: "-"
7: "Peter"
8: "Lou"
9: (pixel_detector - skip)
10: "NORTHWEST"
11: "VETERINARY"
12: "ASSOCIATES"
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Initial model load (cold) | <60s |
| OCR time per page (clean) | <10s |
| OCR time per page (degraded) | <15s |
| Memory usage | <500MB |

## Browser Support

- Chrome (primary target)
- Firefox
- Safari
- Edge

WebGL/WebGPU acceleration preferred for TensorFlow.js.
